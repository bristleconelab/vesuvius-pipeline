# Kaggle kernel (private): HerculeX's own ink stage (github.com/adambetka2/HerculeX @ 9f59e966, MIT) with Reader v3
# (huggingface.co/domenicor046/reader-v3, MIT, Domenico Russo) on the large mosaics of the PHerc0358 unwrap made by
# vesuvius-herculex-0358 (attached through kernel_sources), then a window triage of the ink maps.
# - Only display mosaics of >= 4 cm2 (HerculeX's mosaics_of) are read, largest first, split over the two T4s by area.
#   HerculeX reads every piece of sheet/layout.json, so each GPU gets its own outputs root whose pieces/ and faces/
#   link to the shared ones and whose layout.json holds only that GPU's pieces; ORDER.json cuts the pieces, in rank
#   order and by unit, into CT stores of at most STORE_CAP_MM2 (50 cm2, about 5 GiB of level-0 bricks).
# - Disk: v1 wrote two whole-mosaic stores (29 + 17 GiB), prefetched the next ones and kept 17-layer stacks; the
#   session's writable disk outside /kaggle/working then turned read-only (OSError 30 on every write, even the kernel
#   log). Now at most two capped stores per GPU are on disk, stacks are not kept, the uv cache is removed, and each
#   store's bricks are deleted right after its read.
# - read_stores.py (written below) drives HerculeX's own piece_reading functions store by store: a failed fetch or read
#   prints its traceback, the first failed piece is sampled again in the main process to show the full traceback, and
#   the lane goes on with the next store. The reads stop at HX_INK_DEADLINE_H (10.2 h); what is read stays.
# - Post: every inked mosaic composed at 2x (ink, CT layer 0, valid mask; written face up, as HerculeX's viewer does),
#   then 384 px windows every 192 px with >= 50 % valid pixels scored by max(row std, col std) of the per-row/col
#   share of ink (map > 0.6) among valid pixels, rows/cols < 30 % valid left out; top 100 (no two windows of a mosaic
#   overlapping by IoU > 0.25) with scan xyz of the centre and contact sheets.
# Output (/kaggle/working/out, < 18 GB): mosaics.json, mosaics_inked.json, mosaics/m###_{ink,ct,valid}.png, triage/,
# faces.tar (every piece image and ink map HerculeX wrote; a later run resumes from it), logs/ (kernel.log is this
# script's own output), timings.json.
import glob, json, math, os, shutil, signal, subprocess, sys, tarfile, threading, time
from pathlib import Path

HX_REPO = "https://github.com/adambetka2/HerculeX.git"
HX_COMMIT = "9f59e9669a2273e3bed6bbc39b519604f2a675b3"
SCROLL, SCAN = "PHerc0358", "20250821151737"
T0 = time.time()
INK_DEADLINE = T0 + float(os.environ.get("HX_INK_DEADLINE_H", "10.2")) * 3600
HARD_DEADLINE = T0 + float(os.environ.get("HX_HARD_DEADLINE_H", "11.6")) * 3600
MIN_MM2, STORE_CAP_MM2 = 400.0, 5000.0
ROOT = Path("/tmp/hx"); CODE = ROOT / "HerculeX"; DATA = ROOT / "data"
OUTPUTS = DATA / "outputs"; WINDOWS = DATA / "stacks"
LEVEL_REL = Path("unwrapping") / "level" / SCROLL / SCAN
LEVEL = OUTPUTS / LEVEL_REL
FACES = LEVEL / "faces" / "level-0"
PY = CODE / "runtime" / "python" / "bin" / "python"
UV = f"{sys.executable} -m uv"
OUT = Path("/kaggle/working/out"); LOGS = OUT / "logs"
HXENV = dict(os.environ, HERCULEX_DATA=str(DATA), HERCULEX_WINDOWS=str(WINDOWS), HERCULEX_KEEP_STACKS="0",
             HERCULEX_SAMPLERS=os.environ.get("HX_SAMPLERS", "2"), UV_CACHE_DIR="/tmp/uvcache", PYTHONUNBUFFERED="1",
             TMPDIR=str(CODE / "runtime" / "scratch"), PYTHONDONTWRITEBYTECODE="1")
BRICKS = DATA / "library" / "cache" / "bricks" / SCROLL
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


READ = r'''
import os, shutil, sys, traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.getcwd())
from ink.reading import piece_reading as pr
from sampling import column_sampling as sampler
from sampling.piece_images import store_lock

state = {"diagnosed": False, "failed_stores": 0}


def fetched(sample, volume, planned, folder):
    if folder in planned:
        pr.fetch_bricks_for_read(sample, volume, planned.pop(folder), folder)


def diagnose(jobs, store_folder):
    """The first piece of a failed store that has no ink map, sampled and written again here, to show its traceback."""
    state["diagnosed"] = True
    for job in jobs:
        if pr.done(job):
            continue
        folder, out, axis, turn, _ = job
        print("diagnosing", folder, flush=True)
        try:
            pr.start_sampler(str(store_folder))
            sheet, central, cube = pr.sampled(folder, axis, turn)
            pr.faces_of(cube, sheet > 0)
            pr.write_faces(out, sheet, central)
            print("diagnosis: the piece samples and writes in the main process", flush=True)
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
        return


def main(sample, volume, order):
    queue = pr.queue_of(sample, volume, order)
    print(f"{sample} {volume}: {sum(len(j) for _, j in queue)} pieces queued in {len(queue)} stores", flush=True)
    planned = pr.planned_stores(queue)
    with ThreadPoolExecutor(1) as ahead:
        coming = ahead.submit(fetched, sample, volume, planned, queue[0][0]) if queue else None
        for k, (folder, jobs) in enumerate(queue):
            fetch_ok = True
            sampler.reader_waiting.set()
            try:
                coming.result()
            except Exception:
                fetch_ok = False
                print("fetch failed", folder.name, flush=True)
                traceback.print_exc()
            finally:
                sampler.reader_waiting.clear()
            if k + 1 < len(queue):
                coming = ahead.submit(fetched, sample, volume, planned, queue[k + 1][0])
            if fetch_ok:
                try:
                    with store_lock(folder):
                        pr.read_all(jobs, folder, True)
                except Exception:
                    state["failed_stores"] += 1
                    print("store failed", folder.name, flush=True)
                    traceback.print_exc()
                    if not state["diagnosed"]:
                        diagnose(jobs, folder)
            else:
                state["failed_stores"] += 1
            shutil.rmtree(folder, ignore_errors=True)
            print("store end", folder.name, k + 1, "of", len(queue), "ink maps", sum(pr.done(j) for j in jobs), "of",
                  len(jobs), flush=True)
    return 3 if state["failed_stores"] else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
'''


class Tee:
    """Everything printed goes to the kernel log and to out/logs/kernel.log, which sits on /kaggle/working."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", buffering=1)

    def write(self, text):
        for target in (self.stream, self.file):
            try:
                target.write(text)
                target.flush()
            except OSError:
                pass

    def flush(self):
        for target in (self.stream, self.file):
            try:
                target.flush()
            except OSError:
                pass


def note(**rec):
    rec["t_min"] = round((time.time() - T0) / 60, 1)
    print("NOTE", json.dumps(rec), flush=True)
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(cmd, check=True, env=None):
    """A shell command with its output in the kernel log and kernel.log."""
    print("+", cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-6000:], end="", flush=True)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[:200]}")
    return r


def inked():
    return len(glob.glob(f"{FACES}/*/piece-*-taught.png"))


def monitor():
    while True:
        time.sleep(300)
        print(f"--- {(time.time() - T0) / 60:.0f} min, stage {STATE['stage']}, ink maps {inked()} of "
              f"{STATE['planned']}", flush=True)
        run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p; "
            f"df -h /kaggle/working | tail -1; du -sh /tmp/hx {BRICKS} 2>/dev/null; for f in {LOGS}/ink-g*.log; do "
            "[ -f $f ] && echo $f && grep -E 'store end|store failed|fetch failed|fetched' $f | tail -2; done",
            check=False)


def hx(name, args, env=None, until=None, cmd=None):
    """One ./run.sh command (or cmd), logged to logs/NAME.log, killed at `until`; its exit code."""
    log = LOGS / f"{name}.log"
    STATE["stage"] = name
    began, until = time.time(), until or HARD_DEADLINE
    with open(log, "a") as fh:
        p = subprocess.Popen(cmd or [str(CODE / "run.sh"), *args], cwd=CODE, stdout=fh, stderr=subprocess.STDOUT,
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
        run(f"grep -v 'Read-only file system' {log} | tail -c 4000", check=False)
    return rc


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
    run("nvidia-smi; nproc; free -g; df -h /tmp /kaggle/working", check=False)
    run(f"{sys.executable} -m pip install -q uv")
    if not (CODE / ".git").exists():
        run(f"git clone -q {HX_REPO} {CODE}")
    run(f"git -C {CODE} checkout -q {HX_COMMIT}")
    head = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    assert head == HX_COMMIT, head
    t = time.time()
    run(f"cd {CODE} && UV_PROJECT_ENVIRONMENT=runtime/python {UV} sync -q --locked --no-dev --extra cuda", env=HXENV)
    TIMINGS["setup_uv_sync"] = {"seconds": round(time.time() - t), "rc": 0}
    if not py(TORCH_PROBE, 900):
        run(f"{UV} pip install -q --python {PY} --index-url https://download.pytorch.org/whl/cu126 'torch==2.9.1'",
            env=HXENV)
        if not py(TORCH_PROBE, 900):
            raise SystemExit("PyTorch sees no usable GPU")
        note(stage="setup", torch="cu126 fallback")
    shutil.rmtree("/tmp/uvcache", ignore_errors=True)
    (CODE / "runtime" / "scratch").mkdir(parents=True, exist_ok=True)
    if hx("weights", ["ink.model.weights", "fetch"]):
        raise SystemExit("Reader v3 weights fetch failed")
    if not py(READER_PROBE, 1800):
        raise SystemExit("the CUDA ink reader does not build")
    run("du -sh /tmp/hx /tmp/hx/HerculeX/runtime", check=False)
    note(stage="setup", herculex=head)


def restore():
    found = sorted(glob.glob("/kaggle/input/**/level.tar", recursive=True))
    if not found:
        run("find /kaggle/input -maxdepth 4 | head -50", check=False)
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
    """Display mosaics (HerculeX's mosaics_of) of >= MIN_MM2, largest first, given to the GPU with less area so far;
    each GPU's pieces cut, in rank order and by unit within a mosaic, into stores of at most STORE_CAP_MM2."""
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
        ordered = sorted(g, key=lambda p: (p["unit"], -p["area"]))
        m = {"rank": rank, "area_cm2": round(area / 100, 2), "gpu": gpu,
             "pieces": [[p["unit"], p["piece"]] for p in ordered], "stores": []}
        mosaics.append(m)
        lane = stores[gpu]
        for p in ordered:
            if not lane or lane[-1]["mm2"] + p["area"] > STORE_CAP_MM2 and lane[-1]["pieces"]:
                lane.append({"store": f"ink-g{gpu}-{len(lane) + 1:03d}", "mm2": 0.0, "pieces": []})
            lane[-1]["mm2"] += p["area"]
            lane[-1]["pieces"].append([p["unit"], p["piece"]])
            if lane[-1]["store"] not in m["stores"]:
                m["stores"].append(lane[-1]["store"])
    (OUT / "mosaics.json").write_text(json.dumps(mosaics, indent=1))
    STATE["planned"] = sum(len(m["pieces"]) for m in mosaics)
    note(stage="plan", mosaics=len(mosaics), cm2=round(sum(m["area_cm2"] for m in mosaics), 1),
         pieces=STATE["planned"], gpu_cm2=[round(a / 100, 1) for a in load], stores=[len(s) for s in stores])
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
    """Up to 3 runs of read_stores.py on one GPU, each over the pieces still without an ink map."""
    try:
        for attempt in range(1, 4):
            for stale in BRICKS.glob(f"ink-g{gpu}-*"):
                shutil.rmtree(stale, ignore_errors=True)
            left = [(s, [tuple(k) for k in s["pieces"] if not done(*k)]) for s in stores]
            left = [(s, keys) for s, keys in left if keys]
            if not left or time.time() > INK_DEADLINE - 600:
                break
            root = lane_root(gpu, placed, [k for _, keys in left for k in keys])
            order = root / "ORDER.json"
            order.write_text(json.dumps([[s["store"], [list(k) for k in keys]] for s, keys in left]))
            rc = hx(f"ink-g{gpu}-{attempt}", [], {"CUDA_VISIBLE_DEVICES": str(gpu), "HERCULEX_OUTPUTS": str(root)},
                    until=INK_DEADLINE, cmd=[str(PY), str(ROOT / "read_stores.py"), SCROLL, SCAN, str(order)])
            if rc == 0:
                break
    except Exception:
        import traceback
        print(f"ink lane {gpu} stopped:", traceback.format_exc(), flush=True)
    finally:
        for stale in BRICKS.glob(f"ink-g{gpu}-*"):
            shutil.rmtree(stale, ignore_errors=True)
    note(stage=f"ink-g{gpu}-end", left=sum(not done(*k) for s in stores for k in s["pieces"]))


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
    (OUT / "timings.json").write_text(json.dumps({"herculex": HX_COMMIT, "total_minutes": round((time.time() - T0) / 60, 1),
                                                  "ink_maps": inked(), "planned": STATE["planned"],
                                                  "stages": TIMINGS}, indent=1))
    run(f"ls -la {OUT} {OUT}/triage; du -sh {OUT}", check=False)


def main():
    for d in (OUT, LOGS, ROOT):
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, LOGS / "kernel.log")
    sys.stderr = Tee(sys.__stderr__, LOGS / "kernel.log")
    (ROOT / "post.py").write_text(POST)
    (ROOT / "read_stores.py").write_text(READ)
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
    run(f"tail -c 3000 {LOGS / 'post.log'}", check=False)
    package()
    note(stage="done")


if __name__ == "__main__":
    main()
