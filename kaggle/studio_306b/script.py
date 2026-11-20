# Kaggle kernel (private): positive control for the First Letters chain.
# Runs the published PHerc0211 First Letters ink model (Studio multi-scroll CPT, soft-loss EMA weights) on the published
# 41-layer render of the PHerc0306B First Letters claim surface, in both depth directions, and compares the result with
# the 306B team's own claim map. If Studio cannot see the 306B text here, it is the wrong reader for the 116 keV A/B
# half-volumes. Outputs stay private.
import hashlib, json, os, subprocess, sys, time, urllib.request

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
assert h.hexdigest() == WEIGHTS_SHA, 'weights sha256 mismatch'
print('weights sha256 OK', flush=True)

from huggingface_hub import snapshot_download
import numpy as np, zarr
from PIL import Image
t0 = time.time()
D = snapshot_download('Prodigle/pherc0306b-first-letters', repo_type='dataset', revision=DATA_REV, local_dir=f'{W}/d306',
                      allow_patterns=['results/claim_surface_render_41layers.zarr/*', 'results/claim_map_fwd_fin.npy',
                                      'results/depth_shuffle_map_fwd_fin.npy'], max_workers=16)
print('download s', round(time.time() - t0), flush=True)
vol = zarr.open(f'{D}/results/claim_surface_render_41layers.zarr/0', mode='r')
print('render', vol.shape, flush=True)
S = 20  # surface layer of the 41-layer render (dataset README)
orders = {'fwd': list(range(S + 8, S - 8, -1)),   # Studio depth order: surface+8 down to surface-7 ("planes 23 down to 8" at surface 15)
          'rev': list(range(S - 8, S + 8))}       # the mirrored stack
OUT = f'{W}/out'
os.makedirs(OUT, exist_ok=True)
claim = np.load(f'{D}/results/claim_map_fwd_fin.npy', mmap_mode='r')
shuf = np.load(f'{D}/results/depth_shuffle_map_fwd_fin.npy', mmap_mode='r')
print('claim map', claim.shape, claim.dtype, flush=True)

def small(a, k=8):
    a = np.asarray(a, np.float32)
    H, Wd = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:H, :Wd].reshape(H // k, k, Wd // k, k).mean((1, 3))

def pearson(a, b, m):
    # the claim map is about 28 % NaN, and NaN poisons np.corrcoef: score only pixels finite in both arrays
    f = m & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[f], b[f])[0, 1])

def png(a, path):
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, [1, 99.5])
    Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)).save(path)

cs, ss = small(claim), small(shuf)
png(cs, f'{OUT}/claim_map_small.png')
png(small(vol[S]), f'{OUT}/ct_surface_small.png')
res = {}
for name, idx in orders.items():
    t = time.time()
    np.save(f'{W}/planes_{name}.npy', np.stack([np.asarray(vol[i]) for i in idx]))
    o = f'{W}/studio_{name}.npy'
    sh(f'cd {W}/studio/studio-multiscroll-cpt-20261023/hf && {sys.executable} inference.py --input {W}/planes_{name}.npy '
       f'--output {o} --checkpoint {ck} --stride 64 --blend hann --batch-size 16 --device cuda --amp fp32')
    p = np.load(o, mmap_mode='r')
    ps = small(p)
    np.save(f'{OUT}/studio_{name}_small.npy', ps)
    png(ps, f'{OUT}/studio_{name}_small.png')
    m = (cs > 0) | (ps > 0)
    res[name] = {'seconds': round(time.time() - t), 'mean': float(ps[m].mean()),
                 'frac_gt_0.5': float((np.asarray(p) > 0.5).mean()),
                 'pearson_vs_claim': pearson(ps, cs, m),
                 'pearson_vs_depth_shuffle_claim': pearson(ps, ss, m)}
    print(name, json.dumps(res[name]), flush=True)
    os.remove(f'{W}/planes_{name}.npy')
    json.dump(res, open(f'{OUT}/summary.json', 'w'), indent=1)
sh(f'rm -rf {W}/d306 {W}/studio {ck}')
print(json.dumps(res, indent=1))
