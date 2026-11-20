# Kaggle kernel (private): HerculeX's own ink stage (github.com/adambetka2/HerculeX @ 9f59e966, MIT) with Reader v3
# (huggingface.co/domenicor046/reader-v3, MIT, Domenico Russo) on the large mosaics of the PHerc0358 unwrap made by
# vesuvius-herculex-0358 (attached through kernel_sources), then a window triage of the ink maps.
# - Only display mosaics of >= 4 cm2 (HerculeX's mosaics_of) are read, largest first, split over the two T4s by area.
#   HerculeX reads every piece of sheet/layout.json, so each GPU gets its own outputs root whose pieces/ and faces/
#   link to the shared ones and whose layout.json holds only that GPU's pieces; ORDER.json gives one CT store per
#   batch of mosaics (>= 30 cm2), in rank order. Unchanged HerculeX code does the fetch, the layer stacks, the written
#   face and the read; a store that fails twice is dropped and the lane goes on.
# - The reads stop at HX_INK_DEADLINE_H (10.2 h); what is read stays, and out/faces.tar lets a later run resume.
# - Post: every inked mosaic composed at 2x (ink, CT layer 0, valid mask; written face up, as HerculeX's viewer does),
#   then 384 px windows every 192 px with >= 50 % valid pixels scored by max(row std, col std) of the per-row/col
#   share of ink (map > 0.6) among valid pixels, rows/cols < 30 % valid left out; top 100 (no two windows of a mosaic
#   overlapping by IoU > 0.25) with scan xyz of the centre and contact sheets.
# Output (/kaggle/working/out, < 18 GB): mosaics.json, mosaics/m###_{ink,ct,valid}.png, triage/, faces.tar (every
# piece image and ink map HerculeX wrote), stacks_top.tar (17-layer stacks of the top candidate mosaics, as budget
# allows), logs/, timings.json.
import glob, json, math, os, shutil, signal, subprocess, sys, tarfile, threading, time
from pathlib import Path

HX_REPO = "https://github.com/adambetka2/HerculeX.git"
HX_COMMIT = "9f59e9669a2273e3bed6bbc39b519604f2a675b3"
SCROLL, SCAN = "PHerc0358", "20250821151737"
T0 = time.time()
INK_DEADLINE = T0 + float(os.environ.get("HX_INK_DEADLINE_H", "10.2")) * 3600
HARD_DEADLINE = T0 + float(os.environ.get("HX_HARD_DEADLINE_H", "11.6")) * 3600
MIN_MM2, STORE_MM2, BUDGET_GB = 400.0, 3000.0, 17.5
ROOT = Path("/tmp/hx"); CODE = ROOT / "HerculeX"; DATA = ROOT / "data"
OUTPUTS = DATA / "outputs"; WINDOWS = DATA / "stacks"
LEVEL_REL = Path("unwrapping") / "level" / SCROLL / SCAN
LEVEL = OUTPUTS / LEVEL_REL
FACES = LEVEL / "faces" / "level-0"
PY = CODE / "runtime" / "python" / "bin" / "python"
UV = f"{sys.executable} -m uv"
OUT = Path("/kaggle/working/out"); LOGS = OUT / "logs"
HXENV = dict(os.environ, HERCULEX_DATA=str(DATA), HERCULEX_WINDOWS=str(WINDOWS), HERCULEX_KEEP_STACKS="1",
             HERCULEX_SAMPLERS=os.environ.get("HX_SAMPLERS", "2"), UV_CACHE_DIR="/tmp/uvcache", PYTHONUNBUFFERED="1")
STATE = {"stage": "start", "planned": 0}
TIMINGS = {}

POST = r'''
import json, math, sys
from pathlib import Path
import numpy as np, tifffile
from PIL import Image, ImageDraw
code, level, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(code))
from sampling.sheet_images import MARGIN_CELLS
from website.sheet_tiles import seen_from_written_face
Image.MAX_IMAGE_PIXELS = None
faces = level / "faces" / "level-0"
mosaics = json.loads((out / "mosaics.json").read_text())
layout = {(p["unit"], p["piece"]): p for p in json.loads((level / "sheet" / "layout.json").read_text())["pieces"]}
WIN, STEP, INK_T, MIN_VALID, MIN_LINE, KEEP = 384, 192, round(0.6 * 255), 0.5, 0.3, 100
mdir, tdir = out / "mosaics", out / "triage"
mdir.mkdir(parents=True, exist_ok=True); tdir.mkdir(parents=True, exist_ok=True)


def image(p, suffix):
    path = faces / p["unit"] / f"piece-{p['piece']:04d}{suffix}.png"
    return np.asarray(Image.open(path)) if path.is_file() else None


def compose(m):
    ps = [dict(layout[(u, k)]) for u, k in m["pieces"]]
    seen_from_written_face(ps, faces, "taught-side")
    um, stride = ps[0]["um"], ps[0]["stride"]
    pix, margin_mm = 2 * um, MARGIN_CELLS * stride * um
    x0, y0 = min(p["x_mm"] for p in ps) - margin_mm, min(p["y_mm"] for p in ps) - margin_mm
    W = int(math.ceil((max(p["x_mm"] + p["width_mm"] for p in ps) + margin_mm - x0) / pix)) + 2
    H = int(math.ceil((max(p["y_mm"] + p["height_mm"] for p in ps) + margin_mm - y0) / pix)) + 2
    ink, ct = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    valid, owner = np.zeros((H, W), bool), np.full((H, W), -1, np.int32)
    held, transforms, inked_mm2 = [], {}, 0.0
    margin = round(MARGIN_CELLS * stride * um / pix)
    for i, p in enumerate(ps):
        t = image(p, "-taught")
        s, c = (image(p, ""), image(p, "-layer0")) if t is not None else (None, None)
        if t is None or s is None or c is None:
            continue
        inked_mm2 += p["area"]
        def up(a):
            a = a[::-1] if p["flip_rows"] else a
            return np.ascontiguousarray(a[:, ::-1] if p["flip_cols"] else a)
        t, s, c = up(t), up(s), up(c)
        full = t.shape
        size = (max(1, round(full[1] / 2)), max(1, round(full[0] / 2)))
        t2 = np.asarray(Image.fromarray(t).resize(size, Image.LANCZOS))
        c2 = np.asarray(Image.fromarray(c).resize(size, Image.LANCZOS))
        s2 = np.asarray(Image.fromarray(((s > 0) * 255).astype(np.uint8)).resize(size, Image.NEAREST)) > 0
        px, py = round((p["x_mm"] - x0) / pix) - margin, round((p["y_mm"] - y0) / pix) - margin
        a0, b0, a1, b1 = max(py, 0), max(px, 0), min(py + t2.shape[0], H), min(px + t2.shape[1], W)
        if a1 <= a0 or b1 <= b0:
            continue
        part = (slice(a0 - py, a1 - py), slice(b0 - px, b1 - px))
        top, bottom, left, right = p.get("overlap", (0, 0, 0, 0))
        top, bottom = (bottom, top) if p["flip_rows"] else (top, bottom)
        left, right = (right, left) if p["flip_cols"] else (left, right)
        cells = stride * um / pix
        own = np.zeros(t2.shape, bool)
        own[margin + round(top * cells): t2.shape[0] - margin - round(bottom * cells),
            margin + round(left * cells): t2.shape[1] - margin - round(right * cells)] = True
        held.append(((slice(a0, a1), slice(b0, b1)), t2[part], c2[part], s2[part], own[part], i))
        transforms[i] = {"py": py, "px": px, "h": full[0], "w": full[1], "fr": p["flip_rows"], "fc": p["flip_cols"],
                         "unit": p["unit"], "piece": p["piece"], "stride": stride}
    for margins in (False, True):
        for here, t2, c2, s2, own, i in held:
            free = s2 & ~valid[here] & (own | margins)
            ink[here][free], ct[here][free], owner[here][free] = t2[free], c2[free], i
            valid[here] |= free
    return {"ink": ink, "ct": ct, "valid": valid, "owner": owner, "transforms": transforms, "pix_mm": pix,
            "inked_cm2": inked_mm2 / 100, "inked_pieces": len(transforms), "pieces": len(ps)}


_tif = {}


def xyz_at(comp, r, c):
    owner = comp["owner"]
    i = int(owner[r, c])
    if i < 0:
        near = owner[max(0, r - 24): r + 25, max(0, c - 24): c + 25]
        if (near >= 0).any():
            vals, counts = np.unique(near[near >= 0], return_counts=True)
            i = int(vals[counts.argmax()])
        else:
            return None
    tr = comp["transforms"][i]
    rr, cc = (r - tr["py"]) * 2, (c - tr["px"]) * 2
    rr = tr["h"] - 1 - rr if tr["fr"] else rr
    cc = tr["w"] - 1 - cc if tr["fc"] else cc
    key = (tr["unit"], tr["piece"])
    if key not in _tif:
        folder = level / "pieces" / tr["unit"] / f"piece-{tr['piece']:04d}"
        _tif.clear()
        _tif[key] = np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"])
    g = _tif[key]
    gi = int(np.clip(round((rr - MARGIN_CELLS * tr["stride"]) / tr["stride"]), 0, g.shape[1] - 1))
    gj = int(np.clip(round((cc - MARGIN_CELLS * tr["stride"]) / tr["stride"]), 0, g.shape[2] - 1))
    if g[0, gi, gj] < 0:
        ok = np.argwhere(g[0] >= 0)
        if not len(ok):
            return None
        gi, gj = ok[np.argmin(((ok - [gi, gj]) ** 2).sum(1))]
    return [round(float(v), 1) for v in g[:, gi, gj]] + [tr["unit"], int(tr["piece"])]


def windows(comp):
    valid, hit = comp["valid"], (comp["ink"] >= INK_T) & comp["valid"]
    H, W = valid.shape
    found = []
    for r0 in range(0, H - WIN + 1, STEP):
        vb, mb = valid[r0: r0 + WIN], hit[r0: r0 + WIN]
        for c0 in range(0, W - WIN + 1, STEP):
            v = vb[:, c0: c0 + WIN]
            frac = float(v.mean())
            if frac < MIN_VALID:
                continue
            m = mb[:, c0: c0 + WIN]
            stds = []
            for axis in (1, 0):
                n, k = v.sum(axis), m.sum(axis)
                keep = n >= MIN_LINE * WIN
                stds.append(float((k[keep] / n[keep]).std()) if keep.sum() >= 2 else 0.0)
            found.append({"score": max(stds), "row_std": stds[0], "col_std": stds[1], "r0": r0, "c0": c0,
                          "valid_frac": round(frac, 3), "ink_frac": round(float(m.sum() / max(1, v.sum())), 3)})
    found.sort(key=lambda w: -w["score"])
    kept = []
    for w in found:
        if all(max(0, WIN - abs(w["r0"] - k["r0"])) * max(0, WIN - abs(w["c0"] - k["c0"])) / (2 * WIN * WIN - max(0, WIN - abs(w["r0"] - k["r0"])) * max(0, WIN - abs(w["c0"] - k["c0"]))) <= 0.25 for k in kept):
            kept.append(w)
        if len(kept) == KEEP:
            break
    return kept


summary, every = [], []
for m in mosaics:
    if not any((faces / u / f"piece-{k:04d}-taught.png").is_file() for u, k in m["pieces"]):
        continue
    comp = compose(m)
    stem = mdir / f"m{m['rank']:03d}"
    Image.fromarray(comp["ink"]).save(f"{stem}_ink.png", compress_level=6)
    Image.fromarray(comp["ct"]).save(f"{stem}_ct.png", compress_level=6)
    Image.fromarray(comp["valid"]).save(f"{stem}_valid.png")
    kept = windows(comp)
    for w in kept:
        w.update({"mosaic": m["rank"], "box_px": [w["r0"], w["c0"], w["r0"] + WIN, w["c0"] + WIN],
                  "centre_xyz_unit_piece": xyz_at(comp, w["r0"] + WIN // 2, w["c0"] + WIN // 2)})
        w["score"], w["row_std"], w["col_std"] = (round(w[k], 4) for k in ("score", "row_std", "col_std"))
    every += kept
    row = {k: m[k] for k in ("rank", "area_cm2", "gpu")} | {
        "inked_cm2": round(comp["inked_cm2"], 2), "inked_pieces": comp["inked_pieces"], "pieces": comp["pieces"],
        "png_shape": list(comp["ink"].shape), "mm_per_px": comp["pix_mm"], "windows_kept": len(kept),
        "best_score": kept[0]["score"] if kept else None}
    summary.append(row)
    print("mosaic", json.dumps(row), flush=True)
    del comp
(out / "mosaics_inked.json").write_text(json.dumps(summary, indent=1))
with open(tdir / "all_windows.jsonl", "w") as fh:
    for w in sorted(every, key=lambda w: -w["score"]):
        fh.write(json.dumps(w) + "\n")
top = sorted(every, key=lambda w: -w["score"])[:100]
for k, w in enumerate(top, 1):
    w["window"] = k
(tdir / "top100.json").write_text(json.dumps(top, indent=1))
crops = {}
for m in sorted({w["mosaic"] for w in top}):
    stem = mdir / f"m{m:03d}"
    arrays = [np.asarray(Image.open(f"{stem}_{n}.png")) for n in ("ct", "ink", "valid")]
    for w in top:
        if w["mosaic"] == m:
            r0, c0 = w["r0"], w["c0"]
            crops[w["window"]] = [a[r0: r0 + WIN, c0: c0 + WIN] for a in arrays]
TILE_W, TILE_H, CAP, PER = 2 * WIN + 8, WIN + 22, 22, 10
for s in range(0, len(top), PER):
    batch = top[s: s + PER]
    sheet = Image.new("L", (2 * TILE_W, 5 * TILE_H), 255)
    draw = ImageDraw.Draw(sheet)
    for j, w in enumerate(batch):
        ct, ink, valid = crops[w["window"]]
        valid = valid > 0
        lo, hi = (np.percentile(ct[valid], [1, 99]) if valid.any() else (0, 255))
        ct_s = np.where(valid, np.clip((ct.astype(np.float32) - lo) * 255 / max(1.0, hi - lo), 0, 255), 0).astype(np.uint8)
        x, y = (j % 2) * TILE_W, (j // 2) * TILE_H
        sheet.paste(Image.fromarray(ct_s), (x, y + CAP))
        sheet.paste(Image.fromarray(ink), (x + WIN + 4, y + CAP))
        xyz = w["centre_xyz_unit_piece"]
        where = f"xyz {xyz[0]:.0f},{xyz[1]:.0f},{xyz[2]:.0f}" if xyz else "xyz ?"
        draw.text((x + 2, y + 4), f"#{w['window']} m{w['mosaic']} s={w['score']:.3f} r{w['r0']} c{w['c0']} {where}", fill=0)
    sheet.save(tdir / f"contact_{s // PER + 1:02d}.png", optimize=True)
print("triage", len(every), "windows kept,", "top score", top[0]["score"] if top else None, flush=True)
'''


def note(**rec):
    rec["t_min"] = round((time.time() - T0) / 60, 1)
    print("NOTE", json.dumps(rec), flush=True)
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def sh(cmd, check=True, env=None):
    print("+", cmd[:300], flush=True)
    return subprocess.run(cmd, shell=True, check=check, env=env)


def inked():
    return len(glob.glob(f"{FACES}/*/piece-*-taught.png"))


def monitor():
    while True:
        time.sleep(300)
        print(f"--- {(time.time() - T0) / 60:.0f} min, stage {STATE['stage']}, ink maps {inked()} of "
              f"{STATE['planned']}", flush=True)
        subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; "
                       f"free -g | sed -n 2p; df -h /tmp /kaggle/working | tail -2; du -sh {DATA}/library/cache "
                       f"{WINDOWS} 2>/dev/null; for f in {LOGS}/ink-g*.log; do [ -f $f ] && echo $f && "
                       "tail -c 600 $f | tr '\\r' '\\n' | tail -2; done", shell=True)


def hx(name, args, env=None, until=None):
    """One ./run.sh command, logged to logs/NAME.log, killed at `until`; True when it exits 0."""
    log = LOGS / f"{name}.log"
    STATE["stage"] = name
    began, until = time.time(), until or HARD_DEADLINE
    with open(log, "a") as fh:
        p = subprocess.Popen([str(CODE / "run.sh"), *args], cwd=CODE, stdout=fh, stderr=subprocess.STDOUT,
                             env={**HXENV, **(env or {})}, start_new_session=True)
        while p.poll() is None:
            if time.time() > until:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                break
            time.sleep(5)
    rc = p.returncode
    TIMINGS[name] = {"seconds": round(time.time() - began), "rc": rc}
    note(stage=name, rc=rc, seconds=round(time.time() - began))
    if rc:
        subprocess.run(f"tail -c 3000 {log}", shell=True)
    return rc == 0


def py(code, timeout=1800):
    r = subprocess.run([str(PY), "-c", code], cwd=CODE, env=HXENV, capture_output=True, text=True, timeout=timeout)
    print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
    return r.returncode == 0


TORCH_PROBE = ("import torch; assert torch.cuda.is_available(); x = torch.randn(256, 256, device='cuda', "
               "dtype=torch.half); print('torch', torch.__version__, torch.version.cuda, torch.cuda.device_count(), "
               "float((x @ x).float().abs().mean()))")
READER_PROBE = ("from ink.model import models; from ink.model.reader import lane_reader; "
                "r = lane_reader(models.chosen().weights); print('reader ok', type(r).__name__)")


def setup():
    sh("nvidia-smi; nproc; free -g; df -h /tmp /kaggle/working", check=False)
    sh(f"{sys.executable} -m pip install -q uv")
    if not (CODE / ".git").exists():
        sh(f"git clone -q {HX_REPO} {CODE}")
    sh(f"git -C {CODE} checkout -q {HX_COMMIT}")
    head = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    assert head == HX_COMMIT, head
    t = time.time()
    sh(f"cd {CODE} && UV_PROJECT_ENVIRONMENT=runtime/python {UV} sync -q --locked --no-dev --extra cuda", env=HXENV)
    TIMINGS["setup_uv_sync"] = {"seconds": round(time.time() - t), "rc": 0}
    if not py(TORCH_PROBE, 900):
        sh(f"{UV} pip install -q --python {PY} --index-url https://download.pytorch.org/whl/cu126 'torch==2.9.1'",
           env=HXENV)
        if not py(TORCH_PROBE, 900):
            raise SystemExit("PyTorch sees no usable GPU")
        note(stage="setup", torch="cu126 fallback")
    if not hx("weights", ["ink.model.weights", "fetch"]):
        raise SystemExit("Reader v3 weights fetch failed")
    if not py(READER_PROBE, 1800):
        raise SystemExit("the CUDA ink reader does not build")
    note(stage="setup", herculex=head)


def restore():
    found = sorted(glob.glob("/kaggle/input/**/level.tar", recursive=True))
    if not found:
        sh("find /kaggle/input -maxdepth 4 | head -50", check=False)
        raise SystemExit("level.tar of vesuvius-herculex-0358 not found under /kaggle/input")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with tarfile.open(found[0]) as tar:
        tar.extractall(OUTPUTS)
    note(stage="restore", source=found[0])
    for prior in sorted(glob.glob("/kaggle/input/**/faces.tar", recursive=True)):
        with tarfile.open(prior) as tar:
            tar.extractall(OUTPUTS)
        note(stage="restore", source=prior, ink_maps=inked())


def plan():
    """Display mosaics (HerculeX's mosaics_of) of >= MIN_MM2, largest first, given to the GPU with less area so far,
    batched into CT stores of >= STORE_MM2 in rank order."""
    layout = json.loads((LEVEL / "sheet" / "layout.json").read_text())["pieces"]
    groups = {}
    for p in layout:
        groups.setdefault((p.get("group"), round(p.get("moved_from_joint_mm", 0.0))), []).append(p)
    big = sorted((g for g in groups.values() if sum(p["area"] for p in g) >= MIN_MM2),
                 key=lambda g: -sum(p["area"] for p in g))
    load, mosaics, stores = [0.0, 0.0], [], [[], []]
    for rank, g in enumerate(big, 1):
        area = sum(p["area"] for p in g)
        gpu = 0 if load[0] <= load[1] else 1
        load[gpu] += area
        m = {"rank": rank, "area_cm2": round(area / 100, 2), "gpu": gpu,
             "pieces": [[p["unit"], p["piece"]] for p in sorted(g, key=lambda p: -p["area"])]}
        mosaics.append(m)
        lane = stores[gpu]
        if not lane or lane[-1]["mm2"] >= STORE_MM2:
            lane.append({"store": f"ink-g{gpu}-{len(lane) + 1:03d}", "mm2": 0.0, "pieces": [], "fails": 0})
        lane[-1]["mm2"] += area
        lane[-1]["pieces"] += m["pieces"]
        m["store"] = lane[-1]["store"]
    (OUT / "mosaics.json").write_text(json.dumps(mosaics, indent=1))
    STATE["planned"] = sum(len(m["pieces"]) for m in mosaics)
    note(stage="plan", mosaics=len(mosaics), cm2=round(sum(m["area_cm2"] for m in mosaics), 1),
         pieces=STATE["planned"], gpu_cm2=[round(a / 100, 1) for a in load],
         stores=[len(s) for s in stores])
    return {(p["unit"], p["piece"]): p for p in layout}, stores


def done(unit, piece):
    return (FACES / unit / f"piece-{piece:04d}-taught.png").is_file()


def lane_root(gpu, placed, todo):
    """An outputs root for one GPU: pieces/ and faces/ shared, its own sheet/layout.json of the pieces it reads."""
    root = DATA / f"outputs_g{gpu}"
    level = root / LEVEL_REL
    (level / "sheet").mkdir(parents=True, exist_ok=True)
    FACES.mkdir(parents=True, exist_ok=True)
    for name in ("pieces", "faces"):
        if not (level / name).is_symlink():
            (level / name).symlink_to(LEVEL / name)
    if (LEVEL / "sheet" / "axis.json").exists():
        shutil.copy(LEVEL / "sheet" / "axis.json", level / "sheet" / "axis.json")
    (level / "sheet" / "layout.json").write_text(json.dumps({"record": {}, "pieces": [placed[k] for k in todo]}))
    return root


def ink_lane(gpu, placed, stores):
    dropped = []
    for attempt in range(1, 2 * len(stores) + 2):
        left = [(s, [tuple(k) for k in s["pieces"] if not done(*k) and tuple(k) not in dropped]) for s in stores]
        left = [(s, keys) for s, keys in left if keys]
        if not left or time.time() > INK_DEADLINE:
            break
        root = lane_root(gpu, placed, [k for _, keys in left for k in keys])
        order = root / "ORDER.json"
        order.write_text(json.dumps([[s["store"], [list(k) for k in keys]] for s, keys in left]))
        ok = hx(f"ink-g{gpu}-{attempt}", ["ink.reading.piece_reading", SCROLL, SCAN, str(order)],
                {"CUDA_VISIBLE_DEVICES": str(gpu), "HERCULEX_OUTPUTS": str(root)}, until=INK_DEADLINE)
        if ok or time.time() > INK_DEADLINE:
            continue
        stuck = next(((s, keys) for s, keys in left if any(not done(*k) for k in keys)), None)
        if stuck:
            s, keys = stuck
            s["fails"] += 1
            if s["fails"] >= 2:
                lost = [k for k in keys if not done(*k)]
                dropped += lost
                note(stage=f"ink-g{gpu}", dropped_store=s["store"], pieces=len(lost))
    note(stage=f"ink-g{gpu}-end", dropped=len(dropped))


def size_gb(path):
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 2**30
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 2**30


def package():
    STATE["stage"] = "package"
    if FACES.exists():
        t = time.time()
        with tarfile.open(OUT / "faces.tar", "w") as tar:
            tar.add(FACES, arcname=str(LEVEL_REL / "faces" / "level-0"),
                    filter=lambda i: None if i.name.endswith(".part") else i)
        note(stage="tar", file="faces.tar", gb=round(size_gb(OUT / "faces.tar"), 2), seconds=round(time.time() - t))
    top = OUT / "triage" / "top100.json"
    if top.exists() and WINDOWS.exists():
        stacks = {}
        for path in WINDOWS.rglob("piece-*.zst"):
            stacks[(path.parent.name, path.stem)] = path
        mosaics = {m["rank"]: m for m in json.loads((OUT / "mosaics.json").read_text())}
        order = list(dict.fromkeys(w["mosaic"] for w in json.loads(top.read_text())))
        room = (BUDGET_GB - size_gb(OUT)) * 2**30
        added, kept = 0, []
        with tarfile.open(OUT / "stacks_top.tar", "w") as tar:
            for rank in order:
                paths = [stacks[(u, f"piece-{k:04d}")] for u, k in mosaics[rank]["pieces"]
                         if (u, f"piece-{k:04d}") in stacks]
                need = sum(p.stat().st_size for p in paths)
                if not paths or added + need > room:
                    continue
                for p in paths:
                    tar.add(p, arcname=str(p.relative_to(WINDOWS)))
                added += need
                kept.append(rank)
        note(stage="tar", file="stacks_top.tar", mosaics=kept, gb=round(added / 2**30, 2),
             stacks_on_disk_gb=round(size_gb(WINDOWS), 1))
    (OUT / "timings.json").write_text(json.dumps({"herculex": HX_COMMIT, "total_minutes": round((time.time() - T0) / 60, 1),
                                                  "ink_maps": inked(), "planned": STATE["planned"],
                                                  "stages": TIMINGS}, indent=1))
    sh(f"ls -la {OUT} {OUT}/triage; du -sh {OUT}", check=False)


def main():
    for d in (OUT, LOGS, ROOT):
        d.mkdir(parents=True, exist_ok=True)
    (ROOT / "post.py").write_text(POST)
    threading.Thread(target=monitor, daemon=True).start()
    setup()
    restore()
    placed, stores = plan()
    lanes = [threading.Thread(target=ink_lane, args=(g, placed, stores[g])) for g in (0, 1)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    note(stage="ink", ink_maps=inked(), planned=STATE["planned"])
    STATE["stage"] = "post"
    began = time.time()
    with open(LOGS / "post.log", "w") as fh:
        p = subprocess.Popen([str(PY), str(ROOT / "post.py"), str(CODE), str(LEVEL), str(OUT)], cwd=ROOT,
                             env=HXENV, stdout=fh, stderr=subprocess.STDOUT)
        try:
            p.wait(max(60, HARD_DEADLINE - time.time() - 1200))
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
    TIMINGS["post"] = {"seconds": round(time.time() - began), "rc": p.returncode}
    note(stage="post", rc=p.returncode, seconds=round(time.time() - began))
    subprocess.run(f"tail -c 3000 {LOGS / 'post.log'}", shell=True)
    package()
    note(stage="done")


if __name__ == "__main__":
    main()
