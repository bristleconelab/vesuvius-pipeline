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

TARGET = 'hx0358top'

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


# ===================================================================================================== hx0358top
# The HerculeX layer stacks of the two largest PHerc0358 mosaics (m001, m002) from vesuvius-herculex-ink-0358 v1
# (stacks_top.tar, HerculeX 9f59e966, MIT, adambetka2), re-read with ink-sdk in BOTH depth directions.
# Stack layout (HerculeX ink/reading/piece_reading.py keep()/kept_face()): one file per piece,
#   <scroll>/<scan>/<sampling>/<unit>/piece-<id>.zst = struct 'III' (17, H, W) + zstd(uint8 C-order), in the piece image
#   frame (2 margin cells of `stride` px around the tifxyz grid). The 17 layers are window(cube) of a 31-layer stack
#   sampled along normal * turn (towards the scroll centre), reversed first when HerculeX chose the 'far' written face;
#   the window is centred on the intensity fall (layer 8 = the face). Layer order = the order HerculeX fed Reader v3.
# Here: ink-sdk 'forward' = that stored order (HerculeX's read), 'reverse' = the flipped stack. Pieces are packed into
# a few canvases (zarr v2, 17 layers, gaps of zeros) so each model loads once per canvas; triage windows must lie on one
# piece. Phase 1: Reader V3, Argus V2, TRACE v3B on every canvas. Phase 2: Studio + Reader V3 on crops around the top
# windows of both directions (Studio on whole canvases would take ~ 340 GPU-s/Mpx), then Studio x Reader V3 (ink-sdk
# combine). A-hn / v8-in need 24 layers and cannot run on 17-layer stacks.
import math, struct, tarfile

HX = dict(pack_mpx=110.0, gap=64, ct_layers=(5, 9), studio_top=10, crop_margin=128, win=384, step=192)
AN = T / 'an'


def find_input(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    return Path(hits[0]) if hits else None


def tar_members(path, want):
    """Yield (member name, bytes) of an uncompressed tar for names accepted by want(name), streaming once."""
    with tarfile.open(path) as tar:
        for m in tar:
            if m.isfile() and want(m.name):
                yield m.name, tar.extractfile(m).read()


def zst_stack(raw):
    import zstandard
    shape = struct.unpack_from('III', raw)
    return np.frombuffer(zstandard.ZstdDecompressor().decompress(memoryview(raw)[12:], max_output_size=int(np.prod(shape))),
                         dtype=np.uint8).reshape(shape)


def piece_key(name):
    p = Path(name)
    return p.parent.name, int(p.stem.split('-')[1])


def shelf_pack(sizes, width, gap):
    """Shelf packing of (h, w) boxes, tallest first. Returns {i: (y, x)} and the canvas (H, W)."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0])
    pos, y, x, shelf, W = {}, 0, 0, 0, 0
    for i in order:
        h, w = sizes[i]
        if x and x + w > width:
            y, x, shelf = y + shelf + gap, 0, 0
        pos[i] = (y, x)
        x += w + w % 2 + gap                      # even offsets keep the 2x grids aligned
        shelf = max(shelf, h + h % 2)
        W = max(W, x - gap)
    return pos, (y + shelf, W)


def stage_hx():
    src = find_input('stacks_top.tar')
    if src is None:
        sh('find /kaggle/input -maxdepth 4 | head -60', check=False)
        raise SystemExit('stacks_top.tar not found: vesuvius-herculex-ink-0358 must be its v1 output (v2 keeps no stacks)')
    sh(f'{sys.executable} -m pip install -q zstandard', check=False)
    level, faces_tar, mos_json = find_input('level.tar'), find_input('faces.tar'), find_input('mosaics.json')
    mosaic_of = {}
    if mos_json:
        for m in json.loads(mos_json.read_text()):
            for u, k in m['pieces']:
                mosaic_of[(u, int(k))] = m['rank']
    AN.mkdir(parents=True, exist_ok=True)
    # pass 1: shapes only (headers), to plan the canvases
    pieces = []
    for name, raw in tar_members(src, lambda n: n.endswith('.zst')):
        D, H, W = struct.unpack_from('III', raw)
        u, k = piece_key(name)
        pieces.append(dict(name=name, unit=u, piece=k, shape=(D, H, W), mosaic=mosaic_of.get((u, k))))
    if not pieces:
        raise SystemExit(f'no .zst stacks in {src}')
    note(stage='stacks', n=len(pieces), mpx=round(sum(p['shape'][1] * p['shape'][2] for p in pieces) / 1e6, 1),
         layers=sorted({p['shape'][0] for p in pieces}))
    # side records, sheet masks and HerculeX's own Reader v3 maps (faces.tar), tifxyz + layout (level.tar)
    keys = {(p['unit'], p['piece']) for p in pieces}
    side, sheet, taught, xyz, layout = {}, {}, {}, {}, {}
    from PIL import Image
    import io
    if faces_tar:
        for name, raw in tar_members(faces_tar, lambda n: '/faces/level-0/' in n):
            u = Path(name).parent.name
            stem = Path(name).name
            try:
                k = int(stem.split('-')[1].split('.')[0])
            except (IndexError, ValueError):
                continue
            if (u, k) not in keys:
                continue
            if stem.endswith('-taught-side.json'):
                side[(u, k)] = json.loads(raw)
            elif stem.endswith('-taught.png'):
                taught[(u, k)] = np.asarray(Image.open(io.BytesIO(raw)))
            elif stem == f'piece-{k:04d}.png':
                sheet[(u, k)] = np.asarray(Image.open(io.BytesIO(raw))) > 0
    if level:
        import tifffile
        def wanted(n):
            if n.endswith('sheet/layout.json'):
                return True
            q = Path(n).parts
            if len(q) < 4 or q[-4] != 'pieces' or q[-1] not in ('x.tif', 'y.tif', 'z.tif'):
                return False
            try:
                return (q[-3], int(q[-2].split('-')[1])) in keys
            except (IndexError, ValueError):
                return False
        for name, raw in tar_members(level, wanted):
            if name.endswith('sheet/layout.json'):
                for p in json.loads(raw)['pieces']:
                    layout[(p['unit'], int(p['piece']))] = p
                continue
            parts = Path(name).parts
            if len(parts) < 3 or parts[-1] not in ('x.tif', 'y.tif', 'z.tif'):
                continue
            try:
                key = (parts[-3], int(parts[-2].split('-')[1]))
            except (IndexError, ValueError):
                continue
            if key in keys:
                xyz.setdefault(key, {})[parts[-1][0]] = tifffile.imread(io.BytesIO(raw))
    note(stage='side-data', sides=len(side), sheets=len(sheet), taught=len(taught), xyz=len(xyz), layout=len(layout),
         written={w: sum(1 for s in side.values() if s.get('written') == w) for w in ('near', 'far')})
    # canvases
    total = sum(p['shape'][1] * p['shape'][2] for p in pieces) / 1e6
    n = max(2, int(math.ceil(total / HX['pack_mpx'])))
    bins = [[] for _ in range(n)]
    load = [0.0] * n
    for i in sorted(range(len(pieces)), key=lambda i: -pieces[i]['shape'][1] * pieces[i]['shape'][2]):
        b = int(np.argmin(load))
        bins[b].append(i)
        load[b] += pieces[i]['shape'][1] * pieces[i]['shape'][2]
    segs = {}
    for b, members in enumerate(bins):
        sizes = [pieces[i]['shape'][1:] for i in members]
        width = max(max(w for _, w in sizes), int(math.sqrt(sum(h * w for h, w in sizes) * 1.3)))
        pos, (Hc, Wc) = shelf_pack(sizes, width, HX['gap'])
        Hc, Wc = Hc + Hc % 2, Wc + Wc % 2
        for j, i in enumerate(members):
            pieces[i]['pack'], pieces[i]['yx'] = b, pos[j]
        segs[f'pack{b}'] = dict(sv=SV / f'pack{b}.zarr', valid_npy=None, shape=(17, Hc, Wc), surface=8, ref_dir=None,
                                members=members)
    import zarr
    arrays = {}
    for s, sd in segs.items():
        g = zarr.open_group(str(sd['sv']), mode='w', zarr_format=2)
        arrays[s] = g.create_array('0', shape=sd['shape'], dtype='uint8', chunks=(17, 128, 128))
        g.attrs['pixel_um'] = [CFG['voxel_um']] * 3
        H2, W2 = sd['shape'][1] // 2, sd['shape'][2] // 2
        sd.update(valid2=np.zeros((H2, W2), bool), owner2=np.full((H2, W2), -1, np.int32), ct2=np.zeros((H2, W2), np.float32),
                  taught2=np.full((H2, W2), np.nan, np.float32))
    by_name = {p['name']: i for i, p in enumerate(pieces)}
    for name, raw in tar_members(src, lambda n: n.endswith('.zst')):
        i = by_name[name]
        p = pieces[i]
        st = zst_stack(raw)
        if st.shape[0] != 17:
            raise SystemExit(f'{name}: {st.shape[0]} layers, expected 17')
        sd = segs[f"pack{p['pack']}"]
        y, x = p['yx']
        h, w = st.shape[1:]
        arrays[f"pack{p['pack']}"][:, y:y + h, x:x + w] = st
        key = (p['unit'], p['piece'])
        valid = sheet[key] if key in sheet and sheet[key].shape == (h, w) else st[8] > 0
        ctm = st[HX['ct_layers'][0]:HX['ct_layers'][1]].mean(0)
        y2, x2, h2, w2 = y // 2, x // 2, h // 2, w // 2
        v2 = pool_mask(valid, 2)[:h2, :w2]
        sd['valid2'][y2:y2 + h2, x2:x2 + w2] |= v2
        sd['owner2'][y2:y2 + h2, x2:x2 + w2][v2] = i
        sd['ct2'][y2:y2 + h2, x2:x2 + w2] = np.where(v2, pool(ctm, 2)[:h2, :w2], sd['ct2'][y2:y2 + h2, x2:x2 + w2])
        if key in taught and taught[key].shape == (h, w):
            t2 = pool(taught[key].astype(np.float32) / 255, 2)[:h2, :w2]
            sd['taught2'][y2:y2 + h2, x2:x2 + w2] = np.where(v2, t2, sd['taught2'][y2:y2 + h2, x2:x2 + w2])
        p['written'] = side.get(key, {}).get('written')
        lp = layout.get(key, {})
        p['stride'] = lp.get('stride')
        p['flip_rows'], p['flip_cols'] = bool(lp.get('flip_rows', False)), bool(lp.get('flip_cols', False))
        if key in xyz and len(xyz[key]) == 3:
            p['xyz'] = np.stack([xyz[key][a] for a in 'xyz'])
        del st
    for s, sd in segs.items():
        note(stage='pack', seg=s, shape=list(sd['shape']), pieces=len(sd['members']), valid_mpx=round(4 * sd['valid2'].sum() / 1e6, 1))
    return segs, pieces


def xyz_of(p, r, c):
    """Scan xyz (level-0 voxels) of full-resolution piece-frame pixel (r, c), after HerculeX's post xyz_at."""
    g, s = p.get('xyz'), p.get('stride')
    if g is None or not s:
        return None
    gi = int(np.clip(round((r - 2 * s) / s), 0, g.shape[1] - 1))
    gj = int(np.clip(round((c - 2 * s) / s), 0, g.shape[2] - 1))
    if g[0, gi, gj] < 0:
        ok = np.argwhere(g[0] >= 0)
        if not len(ok):
            return None
        gi, gj = ok[np.argmin(((ok - [gi, gj]) ** 2).sum(1))]
    return [round(float(v), 1) for v in g[:, gi, gj]]


def post_pack(seg, sd, status, final):
    """Phase 1: every map of the canvas pooled to 2x and kept (float16) for the global analysis."""
    try:
        d = AN / seg
        d.mkdir(parents=True, exist_ok=True)
        for (s, k), r in status.items():
            if s != seg or r.get('state') != 'done':
                continue
            for side, fname in SIDES.items():
                f = RUNS / seg / k / fname
                out = d / f'{k}_{side}.npy'
                if f.exists() and not out.exists():
                    m = np.load(f).astype(np.float32)
                    np.save(out, pool(m, 2)[:sd['valid2'].shape[0], :sd['valid2'].shape[1]].astype(np.float16))
        note(stage='post-pack', seg=seg)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        note(stage='post-failed', seg=seg, error=repr(exc))


def stroke_ratio(m, valid):
    """Stroke-scale over blob-scale structure of a 2x map (18.7 um px): std of a sigma 1.5 - 6 px band (strokes of
    ~0.1-0.4 mm) over std of the sigma 12 px low-pass (mm-scale blobs), inside valid."""
    from scipy import ndimage as ni
    if valid.sum() < 1000:
        return None
    f = np.where(valid, m, np.nanmedian(m[valid])).astype(np.float32)
    band = ni.gaussian_filter(f, 1.5) - ni.gaussian_filter(f, 6)
    low = ni.gaussian_filter(f, 12)
    inner = ni.binary_erosion(valid, iterations=12)
    if inner.sum() < 1000:
        return None
    return float(band[inner].std() / max(low[inner].std(), 1e-6))


def global_analysis(segs, pieces, keys):
    """z-mean per direction with statistics pooled over all canvases, stretches, triage, per-piece metrics, sheets."""
    from ink_sdk.ensemble import map_values
    maps = {s: {} for s in segs}
    for s in segs:
        for f in (AN / s).glob('*.npy'):
            maps[s][f.stem] = f
    members = [k for k in ('reader-v3', 'argus-v2', 'trace') if k in keys]
    stats = {}
    for k in members:
        for side in SIDES:
            vals = []
            for s, sd in segs.items():
                if f'{k}_{side}' in maps[s]:
                    m = map_values(np.load(maps[s][f'{k}_{side}']).astype(np.float32), 'ink9um')
                    vals.append(m[sd['valid2']][::7])
            if vals:
                v = np.concatenate(vals)
                stats[f'{k}_{side}'] = (float(v.mean()), float(v.std()))
    small = {s: {} for s in segs}
    for s, sd in segs.items():
        for side in SIDES:
            zs = []
            for k in members:
                name = f'{k}_{side}'
                if name not in maps[s]:
                    continue
                raw = np.load(maps[s][name]).astype(np.float32)
                small[s][name] = np.where(sd['valid2'], raw, np.nan)
                mu, sdv = stats[name]
                zs.append((map_values(raw, 'ink9um') - mu) / max(sdv, 1e-6))
            if len(zs) >= 2:
                small[s][f'zmean_{side}'] = np.where(sd['valid2'], np.mean(zs, 0), np.nan).astype(np.float32)
    names = sorted({n for s in small for n in small[s]})
    lohi = {}
    for n in names:
        v = np.concatenate([small[s][n][segs[s]['valid2']][::5] for s in segs if n in small[s]])
        lohi[n] = tuple(np.percentile(v, [1, 99.5]))
    normd = {s: {n: np.where(segs[s]['valid2'], np.clip((small[s][n] - lohi[n][0]) / max(lohi[n][1] - lohi[n][0], 1e-9), 0, 1),
                             np.nan).astype(np.float32) for n in small[s]} for s in segs}
    norm = lambda s, n: normd[s][n]  # noqa: E731
    summary = dict(target=TARGET, pieces=len(pieces), packs={s: list(sd['shape']) for s, sd in segs.items()}, members=members,
                   stretch={n: [float(a) for a in lohi[n]] for n in names}, maps={}, pieces_metrics=[], triage={})
    for s, sd in segs.items():
        for n in small[s]:
            summary['maps'].setdefault(n, []).append(dict(pack=s) | map_stats(norm(s, n), sd['valid2']))
    # per piece: stroke structure of each map and agreement with HerculeX's own Reader v3 map (forward should match it)
    for i, p in enumerate(pieces):
        sd = segs[f"pack{p['pack']}"]
        y2, x2 = p['yx'][0] // 2, p['yx'][1] // 2
        h2, w2 = p['shape'][1] // 2, p['shape'][2] // 2
        sl = (slice(y2, y2 + h2), slice(x2, x2 + w2))
        own = sd['owner2'][sl] == i
        rec = dict(unit=p['unit'], piece=p['piece'], mosaic=p['mosaic'], written=p.get('written'), pack=p['pack'],
                   shape=list(p['shape']), valid_mpx=round(4 * own.sum() / 1e6, 2), stroke_ratio={}, r_vs_herculex={})
        hxm = sd['taught2'][sl]
        for n in small[f"pack{p['pack']}"]:
            m = small[f"pack{p['pack']}"][n][sl]
            rec['stroke_ratio'][n] = stroke_ratio(m, own)
            rec['r_vs_herculex'][n] = pearson(m, hxm, own)
        summary['pieces_metrics'].append(rec)
    med = {}
    for n in names:
        vals = [r['stroke_ratio'].get(n) for r in summary['pieces_metrics'] if r['stroke_ratio'].get(n) is not None]
        rs = [r['r_vs_herculex'].get(n) for r in summary['pieces_metrics'] if np.isfinite(r['r_vs_herculex'].get(n, np.nan))]
        med[n] = dict(stroke_ratio_median=float(np.median(vals)) if vals else None,
                      r_vs_herculex_median=float(np.median(rs)) if rs else None, n=len(vals))
    summary['direction_medians'] = med
    # triage per canvas, windows on one piece only (>= 95 % of the window's valid pixels owned by one piece)
    W, S = HX['win'], HX['step']
    tops = {}
    for n in names:
        allw = []
        for s, sd in segs.items():
            if n not in small[s]:
                continue
            nm = norm(s, n)
            for w in triage(nm, sd['valid2'], win=W, step=S, top=400):
                o = sd['owner2'][w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']]
                o = o[o >= 0]
                if not o.size:
                    continue
                ids, cnt = np.unique(o, return_counts=True)
                if cnt.max() < 0.95 * o.size:
                    continue
                i = int(ids[cnt.argmax()])
                p = pieces[i]
                r_full = 2 * (w['y'] + w['h'] // 2) - p['yx'][0]
                c_full = 2 * (w['x'] + w['w'] // 2) - p['yx'][1]
                other = n[:-3] + ('rev' if n.endswith('fwd') else 'fwd')
                osc = None
                if other in small[s]:
                    ov = sd['valid2'][w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']]
                    osc = window_score((norm(s, other)[w['y']:w['y'] + w['h'], w['x']:w['x'] + w['w']] > 0.6), ov)
                allw.append(w | dict(pack=s, piece_index=i, unit=p['unit'], piece=p['piece'], mosaic=p['mosaic'],
                                     written=p.get('written'), centre_piece_px_full=[int(r_full), int(c_full)],
                                     scan_xyz=xyz_of(p, r_full, c_full), score_other_direction=osc))
        allw.sort(key=lambda w: -w['score'])
        tops[n] = allw[:40]
        for k, w in enumerate(tops[n], 1):
            w['rank'] = k
        (OUT / f'triage_{n}.json').write_text(json.dumps(dict(map=n, grid='2x of the canvas', window=W, step=S, threshold=0.6,
                                                              windows=tops[n]), indent=1, default=float))
        summary['triage'][n] = dict(n=len(tops[n]), top_score=tops[n][0]['score'] if tops[n] else None,
                                    median_top20=float(np.median([w['score'] for w in tops[n][:20]])) if tops[n] else None)
    # contact sheets: CT | this direction | the other direction, for the ensembles and each model
    for n in names:
        if not tops.get(n):
            continue
        other = n[:-3] + ('rev' if n.endswith('fwd') else 'fwd')
        panels = [('CT', None), (n, n), (other, other)]
        hx_sheet(tops[n], segs, small, norm, pieces, panels, OUT / f'triage_{n}.jpg', f'{n} top {len(tops[n])}')
    # overview PNGs at 4x
    for s, sd in segs.items():
        png(to8(stretch(pool(np.where(sd['valid2'], sd['ct2'], np.nan), 2), pool_mask(sd['valid2'], 2))), OUT / f'{s}_ct_4x.png')
        for n in small[s]:
            png(to8(pool(norm(s, n), 2)), OUT / f'{s}_{n}_4x.png')
    return summary, tops, small, norm


def oriented(a, p):
    a = a[::-1] if p.get('flip_rows') else a
    return a[:, ::-1] if p.get('flip_cols') ^ (p.get('written') == 'far') else a


def hx_sheet(wins, segs, small, norm, pieces, panels, path, title, tile=256, cols=2):
    from PIL import Image, ImageDraw
    k = len(panels)
    pw, ph = k * (tile + 4) + 12, tile + 22
    rows = (len(wins) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * pw, rows * ph + 24), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), title + '  panels: ' + ' | '.join(n for n, _ in panels) + '  (written face up, as HerculeX shows it; 384 px at 2x = 7.2 mm)',
           fill=(230, 230, 230))
    for j, w in enumerate(wins):
        s, p = w['pack'], pieces[w['piece_index']]
        sd = segs[s]
        sl = (slice(w['y'], w['y'] + w['h']), slice(w['x'], w['x'] + w['w']))
        v = sd['valid2'][sl]
        ox, oy = (j % cols) * pw, 24 + (j // cols) * ph
        for q, (label, n) in enumerate(panels):
            if n is None:
                a = stretch(sd['ct2'][sl], v)
            elif callable(n):
                a = n(w)
            elif n in small[s]:
                a = norm(s, n)[sl]
            else:
                continue
            img = Image.fromarray(to8(oriented(a, p))).resize((tile, tile))
            sheet.paste(img, (ox + q * (tile + 4), oy + 18))
        xyz = w.get('scan_xyz')
        where = f"xyz {xyz[0]:.0f},{xyz[1]:.0f},{xyz[2]:.0f}" if xyz else 'xyz ?'
        osc = w.get('score_other_direction')
        d.text((ox + 2, oy + 2), f"#{w.get('rank', j + 1)} m{w.get('mosaic')} {w.get('written') or '?'} s={w['score']:.3f}"
               f" other={osc if osc is None else round(osc, 3)} {where}", fill=(255, 220, 120))
    sheet.save(path, quality=90)


def studio_phase(segs, pieces, tops, small, norm, summary):
    """Studio + Reader V3 on crops around the top windows of both z-mean directions, then Studio x Reader V3."""
    import zarr
    chosen = []
    for n in ('zmean_fwd', 'zmean_rev'):
        for w in tops.get(n, [])[:HX['studio_top']]:
            if all(not (w['pack'] == c['pack'] and abs(w['y'] - c['y']) < w['h'] / 2 and abs(w['x'] - c['x']) < w['w'] / 2)
                   for c in chosen):
                chosen.append(dict(w, from_map=n))
    if not chosen:
        return
    m = HX['crop_margin']
    side_px = 2 * HX['win'] + 2 * m
    half = (len(chosen) + 1) // 2
    groups = [chosen[:half], chosen[half:]] if len(chosen) > 1 else [chosen]
    ssegs = {}
    for g, crops in enumerate(groups):
        if not crops:
            continue
        cols = min(4, len(crops))
        rows = (len(crops) + cols - 1) // cols
        Hc, Wc = rows * (side_px + HX['gap']), cols * (side_px + HX['gap'])
        path = SV / f'studio{g}.zarr'
        grp = zarr.open_group(str(path), mode='w', zarr_format=2)
        arr = grp.create_array('0', shape=(17, Hc, Wc), dtype='uint8', chunks=(17, 128, 128))
        grp.attrs['pixel_um'] = [CFG['voxel_um']] * 3
        for j, c in enumerate(crops):
            src = open_render(segs[c['pack']]['sv'])
            y0, x0 = 2 * c['y'] - m, 2 * c['x'] - m
            Hs, Ws = src.shape[1:]
            a0, b0, a1, b1 = max(y0, 0), max(x0, 0), min(y0 + side_px, Hs), min(x0 + side_px, Ws)
            cy, cx = (j // cols) * (side_px + HX['gap']), (j % cols) * (side_px + HX['gap'])
            block = np.asarray(src[:, a0:a1, b0:b1])
            # keep only the window's own piece in the crop
            own = segs[c['pack']]['owner2'][a0 // 2:(a1 + 1) // 2, b0 // 2:(b1 + 1) // 2] == c['piece_index']
            own = np.repeat(np.repeat(own, 2, 0), 2, 1)[:a1 - a0, :b1 - b0]
            block = np.where(own[None], block, 0).astype(np.uint8)
            arr[:, cy + a0 - y0:cy + a1 - y0, cx + b0 - x0:cx + b1 - x0] = block
            c['studio_yx'] = (cy, cx)
        ssegs[f'studio{g}'] = dict(sv=path, valid_npy=None, shape=(17, Hc, Wc), surface=8, ref_dir=None, crops=crops)
    status = run_all(ssegs, [k for k in ('studio', 'reader-v3') if k in summary['keys']], post_fn=lambda *a: None)
    lohi, res = {}, []
    loaded = {}
    for g, sd in ssegs.items():
        runs = {k: RUNS / g / k for k in ('studio', 'reader-v3') if status.get((g, k), {}).get('state') == 'done'}
        if len(runs) == 2:
            for side in ('forward', 'reverse'):
                combine([runs['studio'], runs['reader-v3']], 'product', side, RUNS / g / 'prodSR')
            runs['prodSR'] = RUNS / g / 'prodSR'
        for k, d in runs.items():
            for side, fname in SIDES.items():
                if (d / fname).exists():
                    loaded[(g, f'{k}_{side}')] = pool(np.load(d / fname).astype(np.float32), 2)
    names = sorted({n for _, n in loaded})
    for n in names:
        v = np.concatenate([a[np.isfinite(a)][::5] for (g, nn), a in loaded.items() if nn == n])
        lohi[n] = tuple(np.percentile(v, [1, 99.5]))
    W = HX['win']
    for g, sd in ssegs.items():
        for c in sd['crops']:
            cy, cx = c['studio_yx']
            y2, x2 = (cy + m) // 2, (cx + m) // 2
            v = segs[c['pack']]['valid2'][c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']]
            c['studio'] = {}
            for n in names:
                if (g, n) not in loaded:
                    continue
                a = loaded[(g, n)][y2:y2 + c['h'], x2:x2 + c['w']]
                nm = np.clip((a - lohi[n][0]) / max(lohi[n][1] - lohi[n][0], 1e-9), 0, 1)
                nm = np.where(v[:a.shape[0], :a.shape[1]], nm, np.nan)
                c['studio'][n] = window_score(np.nan_to_num(nm) > 0.6, v[:a.shape[0], :a.shape[1]])
                c.setdefault('_crops', {})[n] = nm
            res.append(c)
    res.sort(key=lambda c: -max([s for s in c['studio'].values() if s is not None] or [0]))
    panels = [('CT', None)] + [(n, (lambda n: lambda w: w['_crops'].get(n, np.full((w['h'], w['w']), np.nan)))(n))
                               for n in ('prodSR_fwd', 'prodSR_rev', 'studio_fwd', 'studio_rev')]
    for k, c in enumerate(res, 1):
        c['rank'] = k
    hx_sheet(res, segs, small, norm, pieces, panels, OUT / 'studio_windows.jpg', 'Studio x Reader V3 on the top z-mean windows', cols=1)
    for c in res:
        c.pop('_crops', None)
    (OUT / 'studio_windows.json').write_text(json.dumps(dict(stretch={n: [float(a) for a in lohi[n]] for n in names}, windows=res),
                                                        indent=1, default=float))
    summary['studio'] = {f'{g}/{k}': {kk: vv for kk, vv in r.items() if kk != 'tail'} for (g, k), r in status.items()}


def main_hx():
    setup()
    keys = fetch(CFG['profiles'])
    segs, pieces = stage_hx()
    phase1 = [k for k in ('reader-v3', 'argus-v2', 'trace') if k in keys]
    status = run_all(segs, phase1, post_fn=post_pack)
    summary, tops, small, norm = global_analysis(segs, pieces, phase1)
    summary['keys'] = keys
    summary['jobs'] = {f'{s}/{k}': {kk: vv for kk, vv in r.items() if kk != 'tail'} for (s, k), r in status.items()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    if 'studio' in keys and hours() < CFG['no_start_after_h'] - 1.5:
        try:
            studio_phase(segs, pieces, tops, small, norm, summary)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            note(stage='studio-failed', error=repr(exc))
    summary['hours'] = round(hours(), 2)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    sh(f'du -sh {OUT}; ls -la {OUT}', check=False)
    print(json.dumps(dict(direction_medians=summary.get('direction_medians'), triage=summary.get('triage')), indent=1, default=float))


if __name__ == '__main__':
    main_hx()
