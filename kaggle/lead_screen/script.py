# Kaggle kernel (private): trace -> render -> Hecate on First Letters survey leads, plus a PHerc 343 positive control.
# New relative to prior work: the public survey (ketchupbyheinz/vesuvius-scroll-survey) screened these leads with the
# 2.5D models (ink_9um seed42, dense_native) and read no letters; the 3D Hecate model has not been run on them.
# Everything streams from the anonymous open-data bucket; outputs (small PNG previews + JSON) stay in this private run.
# Revealed text must not be published before the organisers allow it (Data Agreement).
import glob, hashlib, json, os, subprocess, sys, time, urllib.request

T0 = time.time()
OUT = '/kaggle/working/out'
os.makedirs(OUT, exist_ok=True)


def sh(cmd, check=True):
    print('$', cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if check:
        r.check_returncode()
    return r


# --- tools ----------------------------------------------------------------------------------------------------
sh('apt-get -qq update && apt-get -qq install -y libegl1 libglx0 libopengl0 libice6 libsm6 libx11-xcb1 '
   'libfontconfig1 libharfbuzz0b libgmp10 >/dev/null', check=False)
sh('pip install -q "zarr>=3" fsspec aiohttp requests tifffile imagecodecs')
V = '/kaggle/working/vc3d'
ASSET = 'VC3D-f7756c7-2026-11-03-linux-x86_64.AppImage'
VC3D_SHA = '17d9f2649c5f42b76c39e4e09ec87412002f505577e48400693c477b581e9c05'
if not os.path.exists(f'{V}/squashfs-root/AppRun'):
    os.makedirs(V, exist_ok=True)
    # per-build archive release first (holds this exact asset); the rolling `stable` tag only as a fallback
    for tag in ('vc3d-archive-latest-433320258', 'stable'):
        try:
            urllib.request.urlretrieve(f'https://github.com/ScrollPrize/villa/releases/download/{tag}/{ASSET}', f'{V}/vc3d.AppImage')
            break
        except OSError as e:
            print('VC3D not under release', tag, e, flush=True)
    assert hashlib.sha256(open(f'{V}/vc3d.AppImage', 'rb').read()).hexdigest() == VC3D_SHA, 'VC3D sha mismatch'
    sh(f'cd {V} && chmod +x vc3d.AppImage && ./vc3d.AppImage --appimage-extract >/dev/null && rm vc3d.AppImage')
VC = f'{V}/squashfs-root/AppRun'
sh(f'{VC} vc_render_tifxyz --help | head -3')

H = '/kaggle/working/hec'
os.makedirs(H, exist_ok=True)
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

import numpy as np, zarr, fsspec, tifffile
from PIL import Image

HTTP = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
S3 = 's3://vesuvius-challenge-open-data'
M7 = 'surface-20260413222639-surface-m7-L0-th0.2'
# name, sample, volume id, voxel um, survey lead (x, y, z) in that volume's voxels
TARGETS = [
    ('control-343', 'PHerc0343', '20250521140437', '8.640um-1.2m-116keV', 8.640, (5911, 4059, 10523)),
    ('1545-a', 'PHerc1545', '20250821151648', '9.362um-1.2m-113keV', 9.362, (3499, 4074, 9431)),
    ('846B-a', 'PHerc0846B', '20250804142305', '9.362um-1.2m-113keV', 9.362, (3962, 4983, 7051)),
    ('846B-b', 'PHerc0846B', '20250804142305', '9.362um-1.2m-113keV', 9.362, (3561, 4951, 6961)),
    ('813-a', 'PHerc0813', '20250821151723', '9.362um-1.2m-113keV', 9.362, (4779, 5850, 8497)),
]
GEN, HALF, NSL = 60, 700, 41            # tracer generations, box half-size (voxels), render slices (surface at index 20)
ZCS = [15.5, 17.5, 19.5, 21.5, 23.5]    # Hecate z-centres on the 41-layer flipped render (reading side = forward)


def snap_seed(sample, vid, p, r=12):
    """Nearest m7-positive voxel to the lead point within r voxels (the tracer must start on the predicted sheet)."""
    a = zarr.open_group(fsspec.get_mapper(f'{HTTP}/{sample}/representations/predictions/surfaces/{vid}-{M7}.zarr'), mode='r')['0']
    x, y, z = p
    box = np.asarray(a[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1]) > 0
    if not box.any():
        return None
    zz, yy, xx = np.nonzero(box)
    k = np.argmin((zz - r) ** 2 + (yy - r) ** 2 + (xx - r) ** 2)
    return int(x - r + xx[k]), int(y - r + yy[k]), int(z - r + zz[k])


# --- 1. seeds + tracers in parallel (one CPU thread each) ---------------------------------------------------------
procs, info = {}, {}
for name, sample, vid, vol, um, lead in TARGETS:
    D = f'/kaggle/working/t/{name}'
    os.makedirs(D, exist_ok=True)
    seed = snap_seed(sample, vid, lead)
    info[name] = {'sample': sample, 'lead': lead, 'seed': seed}
    print(name, 'lead', lead, 'seed', seed, flush=True)
    if seed is None:
        continue
    grids = f'{D}/grids'
    os.makedirs(grids, exist_ok=True)
    json.dump({'url': f'{HTTP}/{sample}/representations/predictions/surfaces/{vid}-{M7}.normal-grids'},
              open(f'{grids}/normal-grids-remote.json', 'w'))
    x, y, z = seed
    params = {'mode': 'seed', 'generations': GEN, 'step_size': 20, 'thread_limit': 1, 'voxelsize': um,
              'min_area_cm': 0.0, 'normal_grid_path': grids,
              'x_min': x - HALF, 'x_max': x + HALF, 'y_min': y - HALF, 'y_max': y + HALF,
              'z_min': z - HALF, 'z_max': z + HALF}
    json.dump(params, open(f'{D}/params.json', 'w'))
    cmd = (f'VC_GROWPATCH_RNG_SEED=1 {VC} vc_grow_seg_from_seed -v {HTTP}/{sample}/representations/predictions/surfaces/'
           f'{vid}-{M7}.zarr -t {D}/surface -p {D}/params.json --segment-name surface --rng-seed 1 -s {x} {y} {z}')
    procs[name] = subprocess.Popen(cmd, shell=True, stdout=open(f'{D}/grow.log', 'w'), stderr=subprocess.STDOUT)
for name, p in procs.items():
    p.wait()
    m = f'/kaggle/working/t/{name}/surface/meta.json'
    info[name]['grow_ok'] = os.path.exists(m)
    if info[name]['grow_ok']:
        meta = json.load(open(m))
        info[name]['area_cm2'] = meta.get('area_cm2')
    print(name, 'grown', info[name], flush=True)
json.dump(info, open(f'{OUT}/summary.json', 'w'), indent=1)

# --- 2. render (41 layers, flipped normals) and 3. Hecate sweep on the GPU ------------------------------------------
def stretch(a):
    lo, hi = np.percentile(a[a > 0], [1, 99.5]) if (a > 0).any() else (0, 1)
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


for name, sample, vid, vol, um, lead in TARGETS:
    if not info[name].get('grow_ok'):
        continue
    D = f'/kaggle/working/t/{name}'
    t = time.time()
    sh(f'{VC} vc_render_tifxyz --volume {S3}/{sample}/volumes/{vid}-{vol}-masked.zarr/ --segmentation {D}/surface '
       f'--zarr-output {D}/render.zarr --scale 1 --group-idx 0 --num-slices {NSL} --cache-gb 8 '
       f'--voxel-size {um} --voxel-unit micrometer --flip-normals')
    info[name]['render_s'] = round(time.time() - t)
    ct = np.asarray(zarr.open(f'{D}/render.zarr', mode='r')['0'][NSL // 2])
    Image.fromarray((stretch(ct) * 255).astype(np.uint8)).save(f'{OUT}/{name}__ct_mid.png')
    res = {}
    for zc in ZCS:
        P = f'{D}/zc{zc}'
        os.makedirs(P, exist_ok=True)
        sh(f'cd {P} && python {H}/hecate_prep.py --src {D}/render.zarr --out {P}/prep.zarr --native-um {um} '
           f'--native-um-z {um} --target-um 9.6 --z-center {zc} --nz 16')
        dirs = ['forward'] + (['reverse'] if zc == 19.5 else [])
        for d in dirs:
            o = f'{P}/{d}.png'
            sh(f'cd {P} && python {H}/hecate.py --checkpoint {H}/hecate_9.6um.pth --input {P}/prep.zarr --spacing-um 9.6 '
               f'--output {o} --device cuda --batch-size 32' + (' --reverse' if d == 'reverse' else ''))
            a = np.asarray(Image.open(o)).astype(np.float32)
            Image.fromarray((stretch(a) * 255).astype(np.uint8)).save(f'{OUT}/{name}__zc{zc}_{d}.png')
            v = a[a > 0]
            res[f'zc{zc}_{d}'] = {'mean': float(v.mean()) if v.size else 0.0,
                                  'p99': float(np.percentile(v, 99)) if v.size else 0.0,
                                  'frac_gt_128': float((v > 128).mean()) if v.size else 0.0}
    info[name]['hecate'] = res
    info[name]['total_s'] = round(time.time() - t)
    json.dump(info, open(f'{OUT}/summary.json', 'w'), indent=1)
    print('DONE', name, json.dumps(info[name])[:400], flush=True)
print('ALL DONE in %d s' % (time.time() - T0))
print(json.dumps(info, indent=1))
