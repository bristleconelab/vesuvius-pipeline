# Kaggle kernel (private): Studio (the PHerc0211 First Letters ink model) on the 40 snapped windings w050-w089 of
# PHerc0800 z 5700-7200. The 31-layer renders are the public outputs of NovaCorpAI's band0800-render-0..3 kernels
# (MIT code, CC BY-NC data; their screen with Reader V3 was negative). Studio read the PHerc0306B claim line at r 0.79
# against the 306B claim map, where Reader V3 missed it; fp16 + stride 96 keeps that (r 0.79) at 4.6 Mpx/min on a T4.
# Both depth orders. Outputs (2x-downsampled uint8 maps + stats) stay private.
import glob, hashlib, json, os, subprocess, sys, time, urllib.request

def sh(cmd):
    print('$', cmd[:300], flush=True)
    subprocess.run(cmd, shell=True, check=True)

W = '/kaggle/working'
STUDIO_SHA = '235e3be8a06749344862c89a1e52c81d90b15e0e'
WEIGHTS = 'https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors'
WEIGHTS_SHA = '96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90'
sh(f'git clone -q https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt.git {W}/studio && cd {W}/studio && git checkout -q {STUDIO_SHA}')
sh('pip install -q "zarr>=3" numcodecs safetensors')
ck = f'{W}/studio-ema.safetensors'
urllib.request.urlretrieve(WEIGHTS, ck)
h = hashlib.sha256()
with open(ck, 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''):
        h.update(b)
assert h.hexdigest() == WEIGHTS_SHA
sys.path.insert(0, f'{W}/studio/studio-multiscroll-cpt-20261023/hf')
import numpy as np, torch, zarr
from inference import load_model, crop
from PIL import Image

model = load_model(ck, 'cuda')
torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def run(volume, bs=32):
    h, w = volume.shape[-2:]
    res = np.zeros((h, w), np.float32)
    coords = [(y, x) for y in range(0, h, 96) for x in range(0, w, 96)]
    empty = (volume == 0).all(0)
    coords = [(y, x) for y, x in coords if not empty[y:y + 96, x:x + 96].all()]
    for i in range(0, len(coords), bs):
        batch = coords[i:i + bs]
        raw = np.stack([crop(volume, y - 16, x - 16) for y, x in batch])
        xx = torch.from_numpy(raw[:, None]).to('cuda', torch.float32).clamp_(0, 200).div_(255)
        with torch.autocast('cuda', dtype=torch.float16):
            lg = model(xx)
        p = lg[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
        for (y, x), q in zip(batch, p):
            res[y:y + min(96, h - y), x:x + min(96, w - x)] = q[:min(96, h - y), :min(96, w - x)]
    res[empty] = 0
    return res


def open_render(p):
    z = zarr.open(p, mode='r')
    return z if hasattr(z, 'shape') else z['0']


paths = sorted(glob.glob('/kaggle/input/**/RENDERS/w*.zarr', recursive=True))
print(len(paths), 'renders', paths[:3], flush=True)
OUT = f'{W}/out'; os.makedirs(OUT, exist_ok=True)
info = {}
for p in paths:
    wname = os.path.basename(p)[:-5]
    t0 = time.time()
    try:
        vol = open_render(p)
        L = vol.shape[0]; S = L // 2                      # 31 layers -> surface 15
        full = np.asarray(vol[:])
    except Exception as e:
        info[wname] = {'error': repr(e)[:300]}; print(wname, info[wname], flush=True); continue
    res = {'shape': list(full.shape)}
    for name, idx in (('asc', list(range(S - 8, S + 8))), ('desc', list(range(S + 8, S - 8, -1)))):
        m = run(np.ascontiguousarray(full[idx]))
        valid = m > 0
        v = m[valid]
        res[name] = {'mean': float(v.mean()) if v.size else 0.0, 'frac_gt_0.5': float((v > 0.5).mean()) if v.size else 0.0,
                     'frac_gt_0.8': float((v > 0.8).mean()) if v.size else 0.0}
        hh, ww = m.shape[0] // 2 * 2, m.shape[1] // 2 * 2
        sm = m[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).mean((1, 3))
        Image.fromarray((np.clip(sm, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/{wname}_{name}.png')
    ct = full[S]; hh, ww = ct.shape[0] // 2 * 2, ct.shape[1] // 2 * 2
    Image.fromarray(ct[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).mean((1, 3)).astype(np.uint8)).save(f'{OUT}/{wname}_ct.png')
    res['seconds'] = round(time.time() - t0)
    info[wname] = res
    json.dump(info, open(f'{OUT}/summary.json', 'w'), indent=1)
    print(wname, json.dumps(res), flush=True)
sh(f'rm -rf {W}/studio {ck}')
print('DONE', len(info))
