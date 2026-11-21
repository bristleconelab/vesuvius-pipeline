# Kaggle kernel (private): the validated ink stage, run with ink-sdk (github.com/LimeGS/ink-sdk, MIT, LimeGS) on 2x T4.
# ink-sdk wraps each model's own inference code; nothing here re-implements a model or its tiling:
#   ink_9um@reader-v3 (domenicor046, MIT), ink_9um@argus-v2 (Louis De Puyfontaine, MIT) and ink_9um@trace-v3b-step5000
#   (TRACE v3B, Jashanjot Singh Gill & Nhat Nam Tran, CC BY-NC-SA 4.0) through villa's koine_machines infer (ScrollPrize, MIT);
#   studio@softloss-20261023 (Studio, Priyank Aranke; code MIT, weights CC BY-NC 4.0), stride 64 + Hann;
#   v8in@pherc0306b-ahn-s0..s2 (A-hn, Liam Warner & Nicola Bodill, CC BY-NC 4.0) and v8in@ink-8um-v8in (v8-in, Youssef Nader, MIT)
#   through v8-in's own predict.py.
# Every profile runs in both depth directions, one process per GPU (CUDA_VISIBLE_DEVICES). Ensembles: the mean of per-model
# z-scores (after Erwin Nieuwlaar's PHerc0343 recipe, MIT), the PHerc0306B product rule TRACE v3B x A-hn (Warner & Bodill) and
# Studio x Reader V3, the last two with `ink-sdk combine`. Outputs per segment: 2x-downsampled PNGs, a CT PNG of the surface layer,
# summary.json and a window triage with contact sheets. Maps are screening data, not evidence of ink or letters.
# TARGET '306b' is the positive control on the published PHerc0306B claim render (HF Prodigle/pherc0306b-first-letters,
# CC BY-NC 4.0); TARGET '0358' runs on our private PHerc0358 renders. make_script.py writes one script.py per target.
# 306b run (2026-11-20, 2x T4, 7.9 Mpx claim box + margin, 3.4 h): Pearson vs the claim map on the box (2x): TRACE v3B x A-hn
# forward 0.971, z-mean 0.927, A-hn 0.902, TRACE 0.884, Studio x Reader V3 0.876, Reader V3 0.846, Argus V2 0.819, Studio 0.791;
# every reverse map <= 0 and every map <= 0.09 against the depth-shuffle control. T4 GPU seconds per Mpx, both directions:
# Reader V3 / Argus V2 11, TRACE 20, Studio (fp32) 342, each A-hn seed or v8-in ~565.
import glob, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TARGET = '306b'

INKSDK_REPO, INKSDK_SHA = 'https://github.com/LimeGS/ink-sdk.git', '63c6c5a3b293ae953ceff88273319d74b45367ba'
# villa revision the 9 um profiles pin (source.upstream_revision; the merge-ink-pipelines tree with ink-detection/koine_machines)
VILLA_REPO, VILLA_SHA = 'https://github.com/ScrollPrize/villa.git', '3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e'
# koine_machines pins zarr 2; ink-sdk needs zarr 3. The villa side gets its own site dir, used only through the --python wrapper.
VILLA_SITE_PKGS = ['zarr==2.18.7', 'numcodecs==0.15.1', 'asciitree', 'fasteners', 'deprecated', 'wrapt']
DATA_REPO, DATA_REV = 'Prodigle/pherc0306b-first-letters', 'f435998b55269762942a979052044309b29650ac'

# key: (ink-sdk profile id, runs through villa, --amp override, tier). Lower tiers run first.
# Studio runs fp32: on a T4 bf16 is emulated and slower than fp32 (our studio_speed kernel). The 9 um lanes keep the profiles'
# validated bf16; A-hn / v8-in use predict.py's own fp16 autocast.
PROFILES = {
    'studio': ('studio@softloss-20261023', False, 'fp32', 1),
    'reader-v3': ('ink_9um@reader-v3', True, None, 1),
    'argus-v2': ('ink_9um@argus-v2', True, None, 1),
    'trace': ('ink_9um@trace-v3b-step5000', True, None, 1),
    'ahn-s0': ('v8in@pherc0306b-ahn-s0', False, None, 2),
    'ahn-s1': ('v8in@pherc0306b-ahn-s1', False, None, 2),
    'ahn-s2': ('v8in@pherc0306b-ahn-s2', False, None, 2),
    'trace-native': ('ink_9um@trace-v3b-step5000-native864', True, None, 3),   # 8.64 um sources only: TRACE without the resample
    'v8in': ('v8in@ink-8um-v8in', False, None, 3),
}
Z_MEMBERS = ['reader-v3', 'argus-v2', 'trace', 'studio', 'ahn']       # one vote per model; A-hn = mean of its three seeds
ADAPTER = {k: ('ink9um' if v[1] else 'other') for k, v in PROFILES.items()}
ADAPTER['ahn'] = 'other'
SIDES = {'fwd': 'probability.npy', 'rev': 'probability_reverse.npy'}   # fwd = layers in stored order (every adapter)

CONFIGS = {
    # Positive control: the 306B claim surface render (41 layers, surface = layer 20, 8.64 um, 4900 x 7780). Inference runs on
    # the claim box (u -15..+23, v -6.84..+2.72 mm; render px y 1639..2745, x 2431..6829, the box the claim's depth-shuffle
    # control covers) plus a 256 px margin; Pearson is scored on the box only.
    '306b': dict(voxel_um=8.64, roi=(1383, 3001, 2175, 7085), box=(1639, 2745, 2431, 6829),
                 profiles=['studio', 'reader-v3', 'argus-v2', 'trace', 'ahn-s0', 'ahn-s1', 'ahn-s2', 'trace-native', 'v8in'],
                 no_start_after_h=9.5, kill_at_h=10.8, shard=(0, 1), delete_runs=False),
    # PHerc0358 windings from vesuvius-render-0358-0..3 (31 layers, surface = layer 15, 9.362 um). SHARD (i, n) takes every
    # n-th winding starting at i, for splitting the windings over several sessions.
    # HerculeX 17-layer written-face stacks of PHerc0358 mosaics m001/m002 (vesuvius-herculex-ink-0358); see hx0358top.py.
    'hx0358top': dict(voxel_um=9.362, roi=None, box=None, profiles=['reader-v3', 'argus-v2', 'trace', 'studio'],
                      no_start_after_h=9.5, kill_at_h=10.8, shard=(0, 1), delete_runs=False),
    '0358': dict(voxel_um=9.362, roi=None, box=None, glob='/kaggle/input/**/RENDERS/w*.zarr',
                 profiles=['studio', 'reader-v3', 'argus-v2', 'trace', 'ahn-s0', 'ahn-s1', 'ahn-s2'],
                 no_start_after_h=9.5, kill_at_h=10.8, shard=(0, 1), delete_runs=True),
}
CFG = CONFIGS[TARGET]
T0 = time.time()
OUT_LIMIT_GB = 17.0


# ---------------------------------------------------------------------------------------------------------------- analysis
# Pure numpy, unit-tested in tests/test_inksdk_lane.py.

def pool(a, k):
    """k x k block mean that ignores NaN; a block with no finite pixel is NaN. Trailing rows/cols are dropped."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[0] // k * k, a.shape[1] // k * k
    b = a[:H, :W].reshape(H // k, k, W // k, k)
    f = np.isfinite(b)
    n = f.sum((1, 3))
    s = np.where(f, b, 0).sum((1, 3), dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def pool_mask(m, k):
    """A pooled pixel is valid when at least half of its block is valid."""
    m = np.asarray(m, bool)
    H, W = m.shape[0] // k * k, m.shape[1] // k * k
    return m[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3)) >= 0.5


def pearson(a, b, mask=None):
    """NaN-safe Pearson r over the pixels finite in both maps (and inside mask)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    f = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        f &= mask
    if f.sum() < 3 or a[f].std() == 0 or b[f].std() == 0:
        return float('nan')
    return float(np.corrcoef(a[f], b[f])[0, 1])


def stretch(a, valid, lo=1.0, hi=99.5):
    """Valid-normalised map in [0, 1]: p1..p99.5 of the valid pixels; NaN outside valid."""
    v = a[valid & np.isfinite(a)]
    if v.size == 0:
        return np.full(a.shape, np.nan, np.float32)
    p0, p1 = np.percentile(v, [lo, hi])
    out = np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1).astype(np.float32)
    out[~(valid & np.isfinite(a))] = np.nan
    return out


def zscore(a, valid):
    v = a[valid & np.isfinite(a)]
    if v.size < 2 or v.std() == 0:
        return np.full(a.shape, np.nan, np.float32)
    z = ((a - v.mean()) / v.std()).astype(np.float32)
    z[~valid] = np.nan
    return z


def map_stats(a, valid):
    v = a[valid & np.isfinite(a)]
    if v.size == 0:
        return dict(mean=None, frac_gt_0_5=None, frac_gt_0_8=None, n=0)
    return dict(mean=float(v.mean()), frac_gt_0_5=float((v > 0.5).mean()), frac_gt_0_8=float((v > 0.8).mean()), n=int(v.size))


def window_score(ink, valid, min_line_valid=0.3):
    """max(row std, col std) of the ink fraction per row / column, over rows / columns with >= 30 % valid pixels."""
    rv, cv = valid.sum(1), valid.sum(0)
    rows, cols = rv >= min_line_valid * valid.shape[1], cv >= min_line_valid * valid.shape[0]
    if rows.sum() < 2 or cols.sum() < 2:
        return None
    k = ink & valid
    rp = k.sum(1)[rows] / rv[rows]
    cp = k.sum(0)[cols] / cv[cols]
    return float(max(rp.std(), cp.std()))


def triage(norm, valid, win=384, step=192, min_valid=0.5, thr=0.6, top=40, max_overlap=0.5):
    """Windows on a valid-normalised 2x map: keep windows with >= min_valid valid pixels, score = window_score of
    (norm > thr), then greedy selection by score that drops a window overlapping an already chosen one by more than
    max_overlap of its area. Coordinates are in the pixels of `norm`."""
    H, W = norm.shape
    ink = np.nan_to_num(norm, nan=0.0) > thr
    cand = []
    for y in range(0, max(H - win, 0) + 1, step):
        for x in range(0, max(W - win, 0) + 1, step):
            v = valid[y:y + win, x:x + win]
            vf = float(v.mean())
            if vf < min_valid:
                continue
            s = window_score(ink[y:y + win, x:x + win], v)
            if s is None:
                continue
            cand.append(dict(y=y, x=x, h=v.shape[0], w=v.shape[1], score=s, valid_frac=vf,
                             ink_frac=float((ink[y:y + win, x:x + win] & v).sum() / max(v.sum(), 1))))
    cand.sort(key=lambda c: -c['score'])
    chosen = []
    for c in cand:
        ok = True
        for d in chosen:
            oy = max(0, min(c['y'] + c['h'], d['y'] + d['h']) - max(c['y'], d['y']))
            ox = max(0, min(c['x'] + c['w'], d['x'] + d['w']) - max(c['x'], d['x']))
            if oy * ox > max_overlap * c['h'] * c['w']:
                ok = False
                break
        if ok:
            chosen.append(c)
        if len(chosen) == top:
            break
    return chosen


# ---------------------------------------------------------------------------------------------------------------- plumbing

def hours():
    return (time.time() - T0) / 3600


def sh(cmd, check=True, **kw):
    print('$', cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, **kw)


def note(**rec):
    rec['h'] = round(hours(), 3)
    print('NOTE', json.dumps(rec, default=str), flush=True)
    with open(OUT / 'jobs.jsonl', 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')


def free_gb(p):
    return shutil.disk_usage(p).free / 2 ** 30


T = Path(max([p for p in ('/kaggle/tmp', '/tmp', '/kaggle/temp') if os.path.isdir(p)], key=free_gb)) / 'inksdk'
OUT = Path('/kaggle/working/inksdk')
SDK, VILLA, VSITE = T / 'ink-sdk', T / 'villa', T / 'villa_site'
MODELS, CACHE, RUNS, SV = T / 'models', T / 'cache', T / 'runs', T / 'sv'
PROFILE_DIR, VPY = T / 'profiles', T / 'villa_python'


def ink_sdk(*args):
    return [sys.executable, '-m', 'ink_sdk.cli', *map(str, args)]


def setup():
    for d in (T, OUT, MODELS, CACHE, RUNS, SV, PROFILE_DIR, OUT / 'env', OUT / 'logs'):
        d.mkdir(parents=True, exist_ok=True)
    sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2; df -h /kaggle/working ' + str(T.parent))
    if not SDK.exists():
        sh(f'git clone -q {INKSDK_REPO} {SDK} && git -C {SDK} checkout -q {INKSDK_SHA}')
    # ink-sdk core (zarr 3) + the extras' runtime modules; torch is Kaggle's own build and is left alone.
    sh(f'{sys.executable} -m pip install -q -e {SDK} imagecodecs safetensors huggingface_hub scipy pynrrd einops timm')
    sh(f'{sys.executable} -m pip install -q -U "tifffile>=2026.3.3" "imagecodecs>=2026.3.6"', check=False)   # koine_machines' pins
    if not VILLA.exists():
        sh(f'git clone -q --filter=blob:none --no-checkout {VILLA_REPO} {VILLA}')
        sh(f'git -C {VILLA} sparse-checkout set ink-detection vesuvius/src')
        sh(f'git -C {VILLA} checkout -q {VILLA_SHA}')
    sh(f'{sys.executable} -m pip install -q --no-deps --target {VSITE} ' + ' '.join(VILLA_SITE_PKGS))
    VPY.write_text('#!/bin/sh\n'
                   f'export PYTHONPATH="{VSITE}:{VILLA}/vesuvius/src${{PYTHONPATH:+:$PYTHONPATH}}"\n'
                   f'exec {sys.executable} "$@"\n')
    VPY.chmod(0o755)
    chk = subprocess.run([str(VPY), '-c', 'import koine_machines.inference.infer as m, koine_machines.models.make_model, vesuvius, '
                          'vesuvius.models.build.build_network_from_config, zarr, numcodecs, torch; '
                          'print("villa ok zarr", zarr.__version__, "numcodecs", numcodecs.__version__, "torch", torch.__version__)'],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(VILLA / 'ink-detection')))
    print(chk.stdout, chk.stderr[-3000:], flush=True)
    if chk.returncode:
        raise SystemExit('villa import check failed')
    # TRACE v3B pins 9.362 +- 0.1 um. On an 8.64 um source the profile's own recipe resamples XY to 9.362 (--resample-from-um);
    # the extra 'native864' profile (same checkpoint and digest) runs the same lane on the native 8.64 grid for comparison.
    src = (SDK / 'src/ink_sdk/profiles/ink_9um-trace-v3b-step5000.toml').read_text()
    assert 'id = "ink_9um@trace-v3b-step5000"' in src and 'scale_tolerance_um = 0.1' in src
    (PROFILE_DIR / 'trace-native864.toml').write_text(
        src.replace('id = "ink_9um@trace-v3b-step5000"', 'id = "ink_9um@trace-v3b-step5000-native864"')
           .replace('scale_tolerance_um = 0.1', 'scale_tolerance_um = 0.75'))
    os.environ.update(INK_SDK_PROFILES=str(PROFILE_DIR), INK_SDK_MODELS=str(MODELS), INK_SDK_CACHE=str(CACHE),
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path.insert(0, str(SDK / 'src'))
    env = {'python': sys.version, 'ink_sdk': INKSDK_SHA, 'villa': VILLA_SHA,
           'villa_head': subprocess.run(['git', '-C', str(VILLA), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'ink_sdk_head': subprocess.run(['git', '-C', str(SDK), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'villa_check': chk.stdout.strip()}
    try:                             # no CUDA context in this process: the GPUs belong to the job processes
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda, n_gpus=torch.cuda.device_count())
    except Exception as exc:
        env['torch_error'] = repr(exc)
    (OUT / 'env' / 'env.json').write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1), flush=True)


def fetch(keys):
    """ink-sdk fetch (size + sha256 pinned). v8-in's base folder carries predict.py + ink8um, which A-hn needs."""
    ids = []
    if any(k.startswith('ahn') or k == 'v8in' for k in keys):
        ids.append(PROFILES['v8in'][0])
    ids += [PROFILES[k][0] for k in keys if PROFILES[k][0] not in ids]
    ok = set()
    for pid in ids:
        r = subprocess.run(ink_sdk('fetch', '--models-dir', MODELS, '-p', pid), capture_output=True, text=True)
        print(pid, 'fetch rc', r.returncode, (r.stdout + r.stderr)[-600:], flush=True)
        if r.returncode == 0:
            ok.add(pid)
    sh(ink_sdk('doctor', '--models-dir', MODELS, '--villa', VILLA), check=False)
    doc = subprocess.run(ink_sdk('doctor', '--json', '--models-dir', MODELS, '--villa', VILLA), capture_output=True, text=True)
    (OUT / 'env' / 'doctor.json').write_text(doc.stdout)
    verified = {p['id'] for p in json.loads(doc.stdout)['profiles'] if p.get('digest_matches')}
    usable = [k for k in keys if PROFILES[k][0] in verified and PROFILES[k][0] in ok]
    if PROFILES['v8in'][0] not in verified:          # no predict.py / ink8um without the v8-in base folder
        usable = [k for k in usable if not k.startswith('ahn')]
    note(stage='fetch', wanted=keys, verified=sorted(verified), usable=usable)
    return usable


def write_v2(dst, src_arr, attrs, y0=0, y1=None, x0=0, x1=None):
    """Copy (a crop of) a ZYX uint8 array into an OME-style zarr v2 group with array '0', by row bands. villa's 9 um infer
    reads zarr 2 only, so every surface volume is staged as v2 whatever its source format."""
    import zarr
    y1 = src_arr.shape[1] if y1 is None else y1
    x1 = src_arr.shape[2] if x1 is None else x1
    D = src_arr.shape[0]
    if dst.exists():
        shutil.rmtree(dst)
    g = zarr.open_group(str(dst), mode='w', zarr_format=2)
    a = g.create_array('0', shape=(D, y1 - y0, x1 - x0), dtype='uint8', chunks=(D, 128, 128))
    for b in range(y0, y1, 1024):
        e = min(y1, b + 1024)
        a[:, b - y0:e - y0] = np.asarray(src_arr[:, b:e, x0:x1])
    for k, v in attrs.items():
        g.attrs[k] = v
        a.attrs[k] = v
    return dst


def open_render(p):
    import zarr
    z = zarr.open(str(p), mode='r')
    return z if hasattr(z, 'shape') else z['0']


def stage_segments():
    """Returns {seg: dict(sv, valid_npy, shape, surface)}; for 306b also downloads the claim maps."""
    segs = {}
    um = CFG['voxel_um']
    if TARGET == '306b':
        from huggingface_hub import snapshot_download
        t = time.time()
        d = snapshot_download(DATA_REPO, repo_type='dataset', revision=DATA_REV, local_dir=str(T / 'd306'), max_workers=16,
                              allow_patterns=['results/claim_surface_render_41layers.zarr/*', 'results/claim_map_fwd_fin.npy',
                                              'results/depth_shuffle_map_fwd_fin.npy', 'results/fused/fwd_prod.npy',
                                              'results/fused/fwd_ensA.npy',
                                              'results/member_maps/fwd_trace_v3B.npy', 'results/member_maps/fwd_hn4_s*.npy',
                                              'results/member_maps/fwd_reader_v3.npy', 'results/member_maps/fwd_v8in_3sigma.npy'])
        note(stage='download', seconds=round(time.time() - t))
        arr = open_render(f'{d}/results/claim_surface_render_41layers.zarr')
        y0, y1, x0, x1 = CFG['roi']
        write_v2(SV / '306b_claimbox.zarr', arr, {'pixel_um': [um] * 3, 'source': f'{DATA_REPO}@{DATA_REV}', 'roi_yxyx': CFG['roi']},
                 y0, y1, x0, x1)
        segs['306b_claimbox'] = dict(sv=SV / '306b_claimbox.zarr', valid_npy=None, shape=(arr.shape[0], y1 - y0, x1 - x0),
                                     surface=20, ref_dir=Path(d) / 'results')
    else:
        paths = sorted(glob.glob(CFG['glob'], recursive=True))
        i, n = CFG['shard']
        paths = paths[i::n]
        note(stage='inputs', n=len(paths), first=[str(p) for p in paths[:4]])
        for p in paths:
            name = Path(p).stem
            if name in segs:
                name = f'{name}_{len(segs)}'
            arr = open_render(p)
            if arr.dtype != np.uint8 or arr.ndim != 3:
                note(stage='skip-input', seg=name, why=f'{arr.dtype} {arr.shape}')
                continue
            t = time.time()
            write_v2(SV / f'{name}.zarr', arr, {'pixel_um': [um] * 3, 'source': str(p)})
            vn = Path(str(p)[:-len('.zarr')] + '.valid.npy')
            segs[name] = dict(sv=SV / f'{name}.zarr', valid_npy=vn if vn.exists() else None, shape=tuple(arr.shape),
                              surface=arr.shape[0] // 2, ref_dir=None)
            note(stage='stage', seg=name, shape=list(arr.shape), seconds=round(time.time() - t))
    return segs


# ---------------------------------------------------------------------------------------------------------------- scheduler

def job_cmd(seg, key, sd):
    pid, villa, amp, _ = PROFILES[key]
    out = RUNS / seg / key
    cmd = ink_sdk('infer', '-p', pid, '--surface-volume', sd['sv'], '--pixel-um', CFG['voxel_um'], '--out', out,
                  '--overwrite', '--on-degenerate', 'warn', '--device', 'cuda', '--models-dir', MODELS, '--cache-dir', CACHE)
    if villa:
        cmd += ['--villa', str(VILLA), '--python', str(VPY)]
    if amp:
        cmd += ['--amp', amp]
    if key == 'trace' and abs(CFG['voxel_um'] - 8.64) < 0.2:
        cmd += ['--resample-from-um', '8.64']      # ink-sdk's documented TRACE recipe for an 8.64 um source
    return cmd, out


def run_all(segs, keys, post_fn=None):
    post_fn = post_fn or post_segment_safe
    jobs = []
    for tier in sorted({PROFILES[k][3] for k in keys}):
        for seg in segs:
            for k in keys:
                if PROFILES[k][3] != tier:
                    continue
                if k == 'trace-native' and abs(CFG['voxel_um'] - 8.64) >= 0.2:
                    continue
                jobs.append((seg, k))
    note(stage='plan', jobs=len(jobs), order=[f'{s}/{k}' for s, k in jobs])
    pending = list(jobs)
    running = {}                     # gpu -> (seg, key, Popen, t0, logfile)
    status = {}                      # (seg, key) -> dict
    post = ThreadPoolExecutor(1)
    post_lock = threading.Lock()
    post_futures = []
    try:
        import torch
        ngpu = max(1, torch.cuda.device_count())
    except Exception:
        ngpu = 1

    posted = {}                      # seg -> status snapshot of its last post-processing

    def snapshot(seg):
        return sorted((k, r.get('state')) for (s, k), r in status.items() if s == seg)

    def maybe_post(seg, tier):
        # post-process a segment when no job of it runs and none of the same tier waits; again after later tiers.
        busy = any(s == seg for s, *_ in running.values())
        later = any(s == seg and PROFILES[k][3] == tier for s, k in pending)
        if not busy and not later:
            final = not any(s == seg for s, _ in pending)
            posted[seg] = snapshot(seg)
            with post_lock:
                post_futures.append(post.submit(post_fn, seg, segs[seg], dict(status), final))

    while pending or running:
        for g in range(ngpu):
            if g in running or not pending:
                continue
            if hours() > CFG['no_start_after_h']:
                for s, k in pending:
                    status[(s, k)] = dict(state='skipped', why='time budget')
                    note(stage='skip', seg=s, key=k, why='time budget')
                pending = []
                break
            seg, key = pending.pop(0)
            cmd, out = job_cmd(seg, key, segs[seg])
            (T / f'tmp{g}').mkdir(exist_ok=True)
            log = OUT / 'logs' / f'{seg}__{key}.log'
            fh = open(log, 'w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TMPDIR=str(T / f'tmp{g}'), PYTHONUNBUFFERED='1')
            print('+', ' '.join(map(str, cmd)), flush=True)
            running[g] = (seg, key, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), time.time(), fh)
            note(stage='start', seg=seg, key=key, gpu=g)
        for g, (seg, key, p, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is None and hours() > CFG['kill_at_h']:
                p.kill()
                rc = p.wait()
                note(stage='killed', seg=seg, key=key, why='kill_at_h')
            if rc is None:
                continue
            fh.close()
            secs = time.time() - t0
            rec = dict(state='done' if rc == 0 else 'failed', rc=rc, seconds=round(secs, 1), gpu=g)
            rp = RUNS / seg / key / 'INK_SDK_RECEIPT.json'
            if rp.exists():
                r = json.loads(rp.read_text())
                rec['liveness'] = {k: v.get('verdict') for k, v in r.get('liveness', {}).items()}
                rec['primary'] = r.get('primary')
                rec['scale_plan'] = r.get('scale_plan') or r.get('scale')
            if rc != 0:
                rec['tail'] = open(OUT / 'logs' / f'{seg}__{key}.log').read()[-2500:]
                print(rec['tail'], flush=True)
            status[(seg, key)] = rec
            note(stage='end', seg=seg, key=key, **{k: v for k, v in rec.items() if k != 'tail'})
            del running[g]
            maybe_post(seg, PROFILES[key][3])
        time.sleep(5)
    for f in post_futures:
        f.result()
    post.shutdown()
    # final pass: segments whose job states changed after their last post-processing (e.g. jobs skipped by the time budget)
    for seg in segs:
        if posted.get(seg) != snapshot(seg) and hours() < 11.5:
            posted[seg] = snapshot(seg)
            post_fn(seg, segs[seg], dict(status), True)
    return status


# ---------------------------------------------------------------------------------------------------------------- outputs

def png(a8, path):
    from PIL import Image
    Image.fromarray(a8).save(path, optimize=True)


def to8(a):
    return (np.nan_to_num(np.clip(a, 0, 1), nan=0.0) * 255).round().astype(np.uint8)


def out_gb():
    return sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) / 2 ** 30


def post_segment_safe(seg, sd, status, final):
    try:
        post_segment(seg, sd, status)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        note(stage='post-failed', seg=seg, error=repr(exc))
        return
    if final and CFG['delete_runs']:          # full-resolution maps are not kept; the outputs are the 2x products
        shutil.rmtree(RUNS / seg, ignore_errors=True)


def combine(ins, mode, side, out):
    r = subprocess.run(ink_sdk('combine', *ins, '--mode', mode, '--side', side, '--out', out), capture_output=True, text=True)
    if r.returncode:
        note(stage='combine-failed', out=str(out), side=side, tail=(r.stdout + r.stderr)[-800:])
    return r.returncode == 0


def post_segment(seg, sd, status):
    import zarr
    from ink_sdk.ensemble import map_values, resize_bilinear
    t = time.time()
    od = OUT / seg
    (od / 'maps').mkdir(parents=True, exist_ok=True)
    vol = open_render(sd['sv'])
    D, H, W = vol.shape
    c = sd['surface']
    ct = np.asarray(vol[c])
    valid = ct > 0
    if sd.get('valid_npy') is not None:
        vm = np.load(sd['valid_npy'])
        if vm.shape == valid.shape:
            valid &= vm.astype(bool)
    valid2 = pool_mask(valid, 2)
    ct2 = pool(np.where(valid, ct, np.nan), 2)
    png(to8(stretch(ct2, valid2)), od / 'ct_surface.png')
    done = {k for (s, k), r in status.items() if s == seg and r.get('state') == 'done'}
    runs = {k: RUNS / seg / k for k in done}

    # A-hn = mean of its seeds; the 306B product rule and Studio x Reader V3, all with ink-sdk combine at full resolution
    ens_dir = RUNS / seg / '_combine'
    seeds = [runs[k] for k in ('ahn-s0', 'ahn-s1', 'ahn-s2') if k in runs]
    combos = {}
    if len(seeds) >= 2:
        combos['ahn'] = ('mean', seeds)
    if len(seeds) == 1:
        runs['ahn'] = seeds[0]
    pairs = {'prod306': ('ahn', 'trace'), 'prod306n': ('ahn', 'trace-native'), 'prodSR': ('studio', 'reader-v3')}
    shutil.rmtree(ens_dir, ignore_errors=True)
    made = set()
    for side in ('forward', 'reverse'):
        for name, (mode, ins) in combos.items():
            if combine(ins, mode, side, ens_dir / name):
                runs[name] = ens_dir / name
        for name, (a, b) in pairs.items():
            if a in runs and b in runs:        # the first input sets the grid: the native-grid map goes first
                if combine([runs[a], runs[b]], 'product', side, ens_dir / name):
                    made.add(name)
    for name in made:
        runs[name] = ens_dir / name

    summary = dict(segment=seg, shape_zyx=[D, H, W], voxel_um=CFG['voxel_um'], surface_layer=c, valid_frac=float(valid.mean()),
                   source=str(sd['sv']), runs={}, maps={}, ensembles={}, triage={}, pearson={})
    for (s, k), r in status.items():
        if s == seg:
            rr = {kk: vv for kk, vv in r.items() if kk != 'tail'}
            if r.get('state') == 'done':
                rr['mpx'] = round(H * W / 1e6, 3)
                rr['s_per_mpx_both_dirs'] = round(r['seconds'] / (H * W / 1e6), 2)
            summary['runs'][k] = rr

    small = {}                       # name_side -> 2x float32 map (NaN outside valid)
    zsum = {s: np.zeros(valid2.shape, np.float64) for s in SIDES}
    zn = {s: 0 for s in SIDES}
    zmembers = {s: [] for s in SIDES}
    for key in sorted(runs):
        for side, fname in SIDES.items():
            f = runs[key] / fname
            if not f.exists():
                continue
            m = np.load(f).astype(np.float32)
            if m.shape != (H, W):
                m = resize_bilinear(m, (H, W))
            m[~valid] = np.nan
            name = f'{key}_{side}'
            summary['maps'][name] = map_stats(m, valid) | dict(scale='probability' if not key.startswith('prod') else 'product')
            m2 = pool(m, 2)
            small[name] = m2
            if key.startswith('prod'):
                png(to8(stretch(m2, valid2)), od / f'{name}.png')
            else:
                png(to8(m2), od / f'{name}.png')
            if key in Z_MEMBERS:             # 9 um maps as Nieuwlaar's clip((v - 64) / 128) first (ink_sdk.ensemble.map_values)
                mv = map_values(np.nan_to_num(m, nan=0.0), ADAPTER[key])
                mv[~valid] = np.nan
                z = zscore(pool(mv, 2), valid2)
                del mv
                if np.isfinite(z).any():
                    zsum[side] += np.nan_to_num(z)
                    zn[side] += 1
                    zmembers[side].append(key)
            del m
    for side in SIDES:
        if zn[side] >= 2:
            zm = (zsum[side] / zn[side]).astype(np.float32)
            zm[~valid2] = np.nan
            small[f'zmean_{side}'] = zm
            summary['ensembles'][f'zmean_{side}'] = dict(members=zmembers[side])
            nz = stretch(zm, valid2)
            summary['maps'][f'zmean_{side}'] = map_stats(nz, valid2) | dict(scale='p1-p99.5 stretch of the z-mean (2x)')
            png(to8(nz), od / f'zmean_{side}.png')
    for name in [n for n in small if n.startswith('prod')]:
        summary['ensembles'][name] = dict(rule='ink-sdk combine --mode product', inputs=list(pairs[name.rsplit('_', 1)[0]]))

    # triage on every ensemble: 384 px windows on the 2x maps, step 192
    ens_names = [n for n in small if n.startswith(('zmean', 'prod'))]
    for name in ens_names:
        norm = stretch(small[name], valid2)
        win = triage(norm, valid2)
        for w in win:
            w['y_full'], w['x_full'] = 2 * w['y'], 2 * w['x']
        (od / f'triage_{name}.json').write_text(json.dumps(dict(map=name, grid='2x', window=384, step=192, threshold=0.6,
                                                                windows=win), indent=1))
        contact_sheet(ct2, norm, valid2, win, od / f'triage_{name}.jpg', f'{seg} {name}')
        summary['triage'][name] = dict(n=len(win), top_score=win[0]['score'] if win else None)
    for name in ens_names:
        if out_gb() < OUT_LIMIT_GB - 1:
            np.save(od / 'maps' / f'{name}_2x.npy', small[name].astype(np.float16))

    if sd.get('ref_dir') is not None:
        summary['pearson'] = score_306b(small, sd['ref_dir'], valid2)
    summary['post_seconds'] = round(time.time() - t, 1)
    (od / 'summary.json').write_text(json.dumps(summary, indent=1))
    note(stage='post', seg=seg, seconds=summary['post_seconds'], maps=len(small), out_gb=round(out_gb(), 2))


def contact_sheet(ct2, norm, valid2, wins, path, title, tile=256, cols=4):
    from PIL import Image, ImageDraw
    ctn = stretch(ct2, valid2)
    rows = max(1, (len(wins) + cols - 1) // cols)
    pw, ph = 2 * tile + 12, tile + 22
    sheet = Image.new('RGB', (cols * pw, rows * ph + 24), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), title + '  (left: CT surface layer, right: valid-normalised ensemble; 384 px at 2x)', fill=(230, 230, 230))
    for i, w in enumerate(wins):
        y, x, h, wd = w['y'], w['x'], w['h'], w['w']
        a = Image.fromarray(to8(ctn[y:y + h, x:x + wd])).resize((tile, tile))
        b = Image.fromarray(to8(norm[y:y + h, x:x + wd])).resize((tile, tile))
        ox, oy = (i % cols) * pw, 24 + (i // cols) * ph
        sheet.paste(a, (ox, oy + 18))
        sheet.paste(b, (ox + tile + 4, oy + 18))
        d.text((ox + 2, oy + 2), f"#{i + 1} s={w['score']:.3f} y={2 * y} x={2 * x}", fill=(255, 220, 120))
    sheet.save(path, quality=90)


def score_306b(small, ref, valid2):
    """Pearson on the claim box only (the region the claim's depth-shuffle control covers), at 2x and 8x block means.
    claim: the claim map; shuffle: its depth-shuffle control; member: the claim team's own map of the same model."""
    y0, y1, x0, x1 = CFG['roi']
    b0, b1, c0, c1 = CFG['box']
    load = lambda n: np.load(ref / n, mmap_mode='r')[y0:y1, x0:x1].astype(np.float32)  # noqa: E731
    refs = {'claim': load('claim_map_fwd_fin.npy'), 'shuffle': load('depth_shuffle_map_fwd_fin.npy')}
    members = {'trace_fwd': 'member_maps/fwd_trace_v3B.npy', 'trace-native_fwd': 'member_maps/fwd_trace_v3B.npy',
               'ahn_fwd': 'fused/fwd_ensA.npy', 'ahn-s0_fwd': 'member_maps/fwd_hn4_s0.npy',
               'ahn-s1_fwd': 'member_maps/fwd_hn4_s1.npy', 'ahn-s2_fwd': 'member_maps/fwd_hn4_s2.npy',
               'reader-v3_fwd': 'member_maps/fwd_reader_v3.npy', 'v8in_fwd': 'member_maps/fwd_v8in_3sigma.npy',
               'prod306_fwd': 'fused/fwd_prod.npy', 'prod306n_fwd': 'fused/fwd_prod.npy'}
    box = np.zeros((y1 - y0, x1 - x0), bool)
    box[b0 - y0:b1 - y0, c0 - x0:c1 - x0] = True
    box2 = pool_mask(box, 2)
    box8 = pool_mask(box, 8)
    r2 = {k: pool(v, 2) for k, v in refs.items()}
    r8 = {k: pool(v, 8) for k, v in refs.items()}
    out = {}
    for name, m2 in small.items():
        m8 = pool(m2, 4)
        h2 = min(m2.shape[0], box2.shape[0]), min(m2.shape[1], box2.shape[1])
        h8 = min(m8.shape[0], box8.shape[0]), min(m8.shape[1], box8.shape[1])
        rec = {}
        for k in refs:
            rec[f'r_{k}_2x'] = pearson(m2[:h2[0], :h2[1]], r2[k][:h2[0], :h2[1]], box2[:h2[0], :h2[1]])
            rec[f'r_{k}_8x'] = pearson(m8[:h8[0], :h8[1]], r8[k][:h8[0], :h8[1]], box8[:h8[0], :h8[1]])
        if name in members and (ref / members[name]).exists():
            mm = pool(load(members[name]), 2)
            rec['member'] = members[name]
            rec['r_member_2x'] = pearson(m2[:h2[0], :h2[1]], mm[:h2[0], :h2[1]], box2[:h2[0], :h2[1]])
        out[name] = rec
    return out


def main():
    setup()
    keys = fetch(CFG['profiles'])
    segs = stage_segments()
    note(stage='segments', segs={k: list(v['shape']) for k, v in segs.items()})
    status = run_all(segs, keys)
    allsum = {}
    for seg in segs:
        f = OUT / seg / 'summary.json'
        if f.exists():
            allsum[seg] = json.loads(f.read_text())
    (OUT / 'summary_all.json').write_text(json.dumps(dict(target=TARGET, ink_sdk=INKSDK_SHA, villa=VILLA_SHA, hours=round(hours(), 2),
                                                          jobs={f'{s}/{k}': {kk: vv for kk, vv in r.items() if kk != 'tail'}
                                                                for (s, k), r in status.items()},
                                                          segments=allsum), indent=1, default=str))
    sh(f'du -sh {OUT}; ls -la {OUT}', check=False)
    print(json.dumps({s: v.get('pearson') for s, v in allsum.items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
