# Kaggle kernel (private): can Studio (PHerc0211 First Letters ink model) run fast enough on a T4 for band screens?
# fp32 + stride 64 + Hann (the published tiling) took ~0.4 Mpx/min. Compare fp16 autocast and the stride-96 tiling on a
# 3 Mpx crop of the published PHerc0306B claim render around the claimed line: speed, agreement with the fp32 map, and
# agreement with the 306B team's claim map. Outputs stay private.
import hashlib, json, os, subprocess, sys, time, urllib.request
from contextlib import nullcontext

def sh(cmd):
    print('$', cmd[:300], flush=True)
    subprocess.run(cmd, shell=True, check=True)

W = '/kaggle/working'
STUDIO_SHA = '235e3be8a06749344862c89a1e52c81d90b15e0e'
DATA_REV = 'f435998b55269762942a979052044309b29650ac'
WEIGHTS = 'https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors'
WEIGHTS_SHA = '96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90'
sh(f'git clone -q https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt.git {W}/studio && cd {W}/studio && git checkout -q {STUDIO_SHA}')
sh('pip install -q "zarr<3" numcodecs safetensors huggingface_hub')
ck = f'{W}/studio-ema.safetensors'
urllib.request.urlretrieve(WEIGHTS, ck)
h = hashlib.sha256()
with open(ck, 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''):
        h.update(b)
assert h.hexdigest() == WEIGHTS_SHA
sys.path.insert(0, f'{W}/studio/studio-multiscroll-cpt-20261023/hf')
import numpy as np, torch, zarr
from inference import load_model, crop, hann96
from huggingface_hub import snapshot_download
from PIL import Image

Y0, Y1, X0, X1 = 1900, 2900, 2400, 5400          # 1000 x 3000 px around the claimed line
S = 20
D = snapshot_download('Prodigle/pherc0306b-first-letters', repo_type='dataset', revision=DATA_REV, local_dir=f'{W}/d306',
                      allow_patterns=['results/claim_surface_render_41layers.zarr/.z*',
                                      'results/claim_surface_render_41layers.zarr/0/.z*',
                                      *[f'results/claim_surface_render_41layers.zarr/0/0/{cy}/{cx}'
                                        for cy in range(Y0 // 128 - 1, Y1 // 128 + 2) for cx in range(X0 // 128 - 1, X1 // 128 + 2)],
                                      'results/claim_map_fwd_fin.npy'], max_workers=16)
vol = zarr.open(f'{D}/results/claim_surface_render_41layers.zarr/0', mode='r')
planes = np.ascontiguousarray(vol[S - 8:S + 8, Y0:Y1, X0:X1])     # the order that reads 306B (layers 12..27 ascending)
claim = np.load(f'{D}/results/claim_map_fwd_fin.npy', mmap_mode='r')[Y0:Y1, X0:X1].astype(np.float32)
print('planes', planes.shape, 'claim nan', float(np.isnan(claim).mean()), flush=True)
model = load_model(ck, 'cuda')
torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def run(volume, dtype, stride, blend, bs):
    h, w = volume.shape[-2:]
    res = np.zeros((h, w), np.float32); wt = np.zeros((h, w), np.float32); k = hann96()
    ys = sorted(set(list(range(0, h, stride)) + ([max(h - 96, 0)] if blend else [])))
    xs = sorted(set(list(range(0, w, stride)) + ([max(w - 96, 0)] if blend else [])))
    coords = [(y, x) for y in ys for x in xs]
    torch.cuda.synchronize(); t = time.time()
    for i in range(0, len(coords), bs):
        batch = coords[i:i + bs]
        raw = np.stack([crop(volume, y - 16, x - 16) for y, x in batch])
        xx = torch.from_numpy(raw[:, None]).to('cuda', torch.float32).clamp_(0, 200).div_(255)
        ctx = torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext()
        with ctx:
            lg = model(xx)
        p = lg[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
        for (y, x), q in zip(batch, p):
            hh, ww = min(96, h - y), min(96, w - x)
            if blend:
                res[y:y + hh, x:x + ww] += q[:hh, :ww] * k[:hh, :ww]; wt[y:y + hh, x:x + ww] += k[:hh, :ww]
            else:
                res[y:y + hh, x:x + ww] = q[:hh, :ww]
    torch.cuda.synchronize(); sec = time.time() - t
    if blend:
        res = np.where(wt > 0, res / np.maximum(wt, 1e-12), 0).astype(np.float32)
    return res, sec, len(coords)


def small(a, k=8):
    H, Wd = a.shape[0] // k * k, a.shape[1] // k * k
    return np.nanmean(a[:H, :Wd].reshape(H // k, k, Wd // k, k), axis=(1, 3))


def r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


OUT = f'{W}/out'; os.makedirs(OUT, exist_ok=True)
mpx = planes.shape[1] * planes.shape[2] / 1e6
variants = [('fp32_s64_hann', None, 64, True, 16), ('fp16_s64_hann', torch.float16, 64, True, 32),
            ('fp16_s96', torch.float16, 96, False, 32), ('bf16_s96', torch.bfloat16, 96, False, 32)]
maps, out = {}, {}
for name, dt, st, bl, bs in variants:
    try:
        m, sec, n = run(planes, dt, st, bl, bs)
    except Exception as e:
        out[name] = {'error': repr(e)[:300]}; print(name, out[name], flush=True); continue
    maps[name] = m
    out[name] = {'seconds': round(sec, 1), 'windows': n, 'mpx_per_min': round(mpx / sec * 60, 2), 'nan': int(np.isnan(m).sum()),
                 'pearson_vs_claim_small': r(small(m), small(claim))}
    if 'fp32_s64_hann' in maps:
        out[name]['pearson_vs_fp32_full'] = r(m, maps['fp32_s64_hann'])
    Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/{name}.png')
    print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(f'{OUT}/speed.json', 'w'), indent=1)
sh(f'rm -rf {W}/d306 {W}/studio {ck}')
print(json.dumps(out, indent=1))
