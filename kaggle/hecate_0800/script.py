# Kaggle kernel (private): Hecate (3D, 9.6 um) on the six official PHerc0800 segments, both depth directions.
# New relative to prior work: PHerc0800's segments were screened with a three-model villa ensemble (ScrollPrize/villa
# PR 2453), not with Hecate. Input = the published 8.64 um surface volumes (no tracing). Outputs stay private:
# revealed text must not be published before the organisers allow it (Data Agreement).
import hashlib, json, os, subprocess, time, urllib.request


def sh(cmd, check=True):
    print('$', cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-1200:], r.stderr[-1200:], flush=True)
    if check:
        r.check_returncode()
    return r


sh('pip install -q "zarr>=3" awscli')
import numpy as np, zarr
from PIL import Image

H = '/kaggle/working/hec'
OUT = '/kaggle/working/out'
os.makedirs(H, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
FILES = {
    'hecate.py': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py',
                  'c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59'),
    'hecate_9.6um.pth': ('https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth',
                         '809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d'),
    'hecate_prep.py': ('https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1/ink/scripts/hecate_prep.py',
                       '8249994e6da8d29a1c4005c518450456f570642da03531f10a874ee97f2e4a5a'),
}
for f, (u, want) in FILES.items():
    p = f'{H}/{f}'
    if not os.path.exists(p):
        urllib.request.urlretrieve(u, p)
    assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == want, f
print('sha256 OK', list(FILES), flush=True)

B = 's3://vesuvius-challenge-open-data/PHerc0800/segments'
SV = '8.64um-1.2m-116keV-volume-20250521135224.zarr'
SEGS = ['20251028213516-auto_grown_20251028213516907', '20251028220042-auto_grown_20251028220042762',
        '20251028220955-auto_grown_20251028220955262', '20251028222030-auto_grown_20251028222030940',
        '20251028225813-auto_grown_20251028225813045', '20251029010146-auto_grown_20251029010146642']
ZC = 14.5          # 31-layer published renders: surface at layer 15; same "surface - 0.5" rule that worked on 0841


def stretch(a):
    v = a[a > 0]
    lo, hi = (np.percentile(v, [1, 99.5]) if v.size else (0, 1))
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


info = {}
for seg in SEGS:
    t0 = time.time()
    D = f'/kaggle/working/d/{seg}'
    sh(f'aws s3 sync --no-sign-request --only-show-errors {B}/{seg}/surface-volumes/{SV} {D}/sv.zarr --exclude "*" --include "0/*" --include ".z*"')
    shape = zarr.open(f'{D}/sv.zarr', mode='r')['0'].shape
    sh(f'cd {D} && python {H}/hecate_prep.py --src {D}/sv.zarr --out {D}/prep.zarr --native-um 8.64 --native-um-z 8.64 '
       f'--target-um 9.6 --z-center {ZC} --nz 16')
    res = {'shape': list(shape)}
    for d in ('forward', 'reverse'):
        t = time.time()
        o = f'{D}/{d}.png'
        sh(f'cd {D} && python {H}/hecate.py --checkpoint {H}/hecate_9.6um.pth --input {D}/prep.zarr --spacing-um 9.6 '
           f'--output {o} --device cuda --batch-size 32' + (' --reverse' if d == 'reverse' else ''))
        a = np.asarray(Image.open(o)).astype(np.float32)
        Image.open(o).save(f'{OUT}/{seg[:14]}__{d}_raw.png')
        Image.fromarray((stretch(a) * 255).astype(np.uint8)).save(f'{OUT}/{seg[:14]}__{d}.png')
        v = a[a > 0]
        res[d] = {'seconds': round(time.time() - t), 'mean': float(v.mean()) if v.size else 0.0,
                  'frac_gt_128': float((v > 128).mean()) if v.size else 0.0,
                  'frac_gt_192': float((v > 192).mean()) if v.size else 0.0}
    f, r = res['forward']['frac_gt_128'], res['reverse']['frac_gt_128']
    res['one_sidedness'] = round(max(f, r) / max(min(f, r), 1e-6), 2)
    res['brighter_side'] = 'forward' if f >= r else 'reverse'
    ct = np.asarray(zarr.open(f'{D}/sv.zarr', mode='r')['0'][shape[0] // 2])
    Image.fromarray((stretch(ct) * 255).astype(np.uint8)).save(f'{OUT}/{seg[:14]}__ct_mid.png')
    res['total_s'] = round(time.time() - t0)
    info[seg] = res
    json.dump(info, open(f'{OUT}/summary.json', 'w'), indent=1)
    print('DONE', seg, json.dumps(res), flush=True)
print(json.dumps(info, indent=1))
