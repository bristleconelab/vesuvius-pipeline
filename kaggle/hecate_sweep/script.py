# Kaggle kernel: Hecate depth-convention sweep on held-out PHerc0841 segments (vesuvius-pipeline).
# The first full run gave AUC ~0.52 on 0841 w00 (reverse, z-centre 13.5): our renders are flipped (--flip-normals,
# 28 layers) while Hecate's reference renders are not flipped (66 layers, z-centre 32.5 + d). This sweeps the
# z-centre and both directions on the densest 1024 x 1024 labelled window of each segment, then scores every variant
# with the repo's metric (ROC AUC inside the supervision mask, labels aligned from CT texture).
import glob, hashlib, json, os, shutil, subprocess, sys, time, urllib.request


def sh(cmd):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    r.check_returncode()
    return r.stdout


sh('pip install -q "zarr>=3" tifffile imagecodecs awscli')
if not os.path.isdir('/kaggle/working/vp'):
    sh('git clone -q --depth 1 https://github.com/bristleconelab/vesuvius-pipeline /kaggle/working/vp')
sys.path.insert(0, '/kaggle/working/vp/src')
import numpy as np, tifffile, torch, zarr
from PIL import Image
from vpipe import compare, crop, evaluate
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)

W = '/kaggle/working/hec'
os.makedirs(W, exist_ok=True)
FILES = {
    'hecate.py': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py',
                  'c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59'),
    'hecate_9.6um.pth': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth',
                         '809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d'),
    'hecate_prep.py': ('https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1/ink/scripts/hecate_prep.py',
                       '8249994e6da8d29a1c4005c518450456f570642da03531f10a874ee97f2e4a5a'),
}
for f, (u, want) in FILES.items():
    p = f'{W}/{f}'
    if not os.path.exists(p):
        urllib.request.urlretrieve(u, p)
    assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == want, f
print('sha256 OK', list(FILES), flush=True)

B = 's3://vesuvius-challenge-open-data/PHerc0841/segments'
SV9 = '9.366um-1.2m-113keV-volume-20250821151531.zarr'
SV24 = '2.403um-0.22m-77keV-volume-20260319124803.zarr'
SEGS = {'0841-w00': '20260220213127-w00', '0841-ag1': '20260220214732-auto_grown_20260220144552896'}
ZCENTRES = [11.5, 12.5, 13.5, 14.5, 15.5, 16.5]
WIN = 1024


def s3(src, dst):
    sh(f'aws s3 sync --no-sign-request --only-show-errors {src} {dst}')


def s3cp(src, dst):
    subprocess.run(f'aws s3 cp --no-sign-request --only-show-errors {src} {dst}', shell=True)


def densest_window(lab, sup, shape, win):
    """(y0, x0) of the win x win window with the most supervised pixels, on the render grid."""
    m = compare.resize_to((np.asarray(sup) > 0).astype(np.float32), shape) > 0.5
    c = np.cumsum(np.cumsum(np.pad(m, ((1, 0), (1, 0))).astype(np.int64), 0), 1)
    H, W_ = shape
    best, at = -1, (0, 0)
    for y in range(0, max(H - win, 0) + 1, 64):
        for x in range(0, max(W_ - win, 0) + 1, 64):
            s = c[y + win, x + win] - c[y, x + win] - c[y + win, x] + c[y, x]
            if s > best:
                best, at = s, (y, x)
    return at


RES = {}
for name, seg in SEGS.items():
    D = f'/kaggle/working/data/{name}'
    s3(f'{B}/{seg}/surface-volumes/{SV9}/0', f'{D}/sv9.zarr/0')
    for m in ('.zattrs', '.zgroup'):
        s3cp(f'{B}/{seg}/surface-volumes/{SV9}/{m}', f'{D}/sv9.zarr/{m}')
    s3(f'{B}/{seg}/surface-volumes/{SV24}/2', f'{D}/sv24.zarr/2')
    for m in ('.zattrs', '.zgroup'):
        s3cp(f'{B}/{seg}/surface-volumes/{SV24}/{m}', f'{D}/sv24.zarr/{m}')
    s3(f'{B}/{seg}/ink-labels/2.403um-volume-20260319124803', f'{D}/labels')
    lab = sorted(glob.glob(f'{D}/labels/*/inklabels.zarr'))[0].rsplit('/', 1)[0]
    shape = tuple(zarr.open(f'{D}/sv9.zarr', mode='r')['0'].shape[1:])
    sup = np.asarray(zarr.open(f'{lab}/supervision.zarr', mode='r')['0'][...])
    y0, x0 = densest_window(None, sup, shape, WIN)
    print(name, 'render', shape, 'window', y0, x0, WIN, flush=True)
    for zc in ZCENTRES:
        T = f'{D}/zc{zc}'
        os.makedirs(T, exist_ok=True)
        sh(f'cd {T} && python {W}/hecate_prep.py --src {D}/sv9.zarr --out {T}/prep.zarr --native-um 9.366 '
           f'--native-um-z 9.366 --target-um 9.6 --z-center {zc} --nz 16 --crop {y0} {x0} {WIN} {WIN}')
        for direction in ('reverse', 'forward'):
            t = time.time()
            out = f'{T}/{direction}.png'
            sh(f'cd {T} && python {W}/hecate.py --checkpoint {W}/hecate_9.6um.pth --input {T}/prep.zarr '
               f'--spacing-um 9.6 --output {out} --device cuda --batch-size 32' + (' --reverse' if direction == 'reverse' else ''))
            tif = f'{T}/{direction}.tif'
            tifffile.imwrite(tif, crop.paste(np.asarray(Image.open(out)), shape, y0, x0, WIN, WIN))
            # score only inside the window: build a window-limited supervision mask on the label grid
            r = evaluate.eval_ink(tif, f'{lab}/inklabels.zarr', None, None, mask=f'{lab}/supervision.zarr',
                                  align_ref=f'{D}/sv24.zarr', align_ref_level=2, align_ours=f'{D}/sv9.zarr',
                                  window=(y0, x0, WIN, WIN))
            mm = r['labels']['metrics_in_mask']
            key = f'{name} zc{zc} {direction}'
            RES[key] = {'roc_auc': mm['roc_auc'], 'average_precision': mm['average_precision'],
                        'n_pixels': mm['n_pixels'], 'seconds': round(time.time() - t)}
            print('RESULT', key, 'AUC %.4f AP %.4f n %d %ds' % (mm['roc_auc'], mm['average_precision'],
                                                                 mm['n_pixels'], time.time() - t), flush=True)
            json.dump(RES, open('/kaggle/working/hecate_sweep.json', 'w'), indent=1)
    shutil.rmtree(D)
print(json.dumps(RES, indent=1))
