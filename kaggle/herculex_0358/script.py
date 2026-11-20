# Kaggle kernel (private): HerculeX (github.com/adambetka2/HerculeX @ 9f59e966, MIT, Adam Betka) cuda lane on
# PHerc0358, scan 20250821151737 (9.362 um, 113 keV): CT -> sheet nodes (GPU) -> exact winding levels -> single-sheet
# tifxyz pieces -> joins + scan frame -> one laid-out sheet. The stages are run one by one (not `./run.sh unwrap`), so
# the level fits get all 4 vCPUs and the run stops cleanly before Kaggle's 12 h limit.
# Ink (Reader v3) is off by default (HX_INK=1 turns it on): on 4 vCPUs the layer-stack sampling and the level-0 CT
# download (several hundred GB) cost hours, and a separate ink kernel reads these pieces later.
# Resume: a run that stops early writes out/nodes.tar and out/level.tar; attach this kernel's output through
# kernel_sources and the next run extracts them and skips every finished unit.
# Output (/kaggle/working/out): level.tar (pieces/rows-*/piece-*/{x,y,z}.tif + meta.json + quality.npy, unit records,
# joints, frame, sheet/layout.json), piece_index.csv, units.json, survey_status.json, summary.json, previews/*.png,
# sheet_tifxyz.tar.gz (the whole laid-out sheet as one tifxyz), nodes.tar (when it fits), logs/, timings.json.
import glob, json, os, shutil, signal, subprocess, sys, tarfile, threading, time
from pathlib import Path

HX_REPO = "https://github.com/adambetka2/HerculeX.git"
HX_COMMIT = "9f59e9669a2273e3bed6bbc39b519604f2a675b3"
SCROLL, SCAN = "PHerc0358", "20250821151737"
T0 = time.time()
DEADLINE = T0 + float(os.environ.get("HX_DEADLINE_H", "10.8")) * 3600
INK = os.environ.get("HX_INK", "0") == "1"
FIT_WORKERS = int(os.environ.get("HX_FIT_WORKERS", "2"))
BUDGET_GB = 17.5
ROOT = Path("/tmp/hx"); CODE = ROOT / "HerculeX"; DATA = ROOT / "data"
OUTPUTS = DATA / "outputs"
PY = CODE / "runtime" / "python" / "bin" / "python"
UV = f"{sys.executable} -m uv"
LEVEL_DIR = OUTPUTS / "unwrapping" / "level" / SCROLL / SCAN
NODE_DIR = OUTPUTS / "unwrapping" / "nodes" / SCROLL / SCAN / "nodes"
OUT = Path("/kaggle/working/out"); LOGS = OUT / "logs"; PREVIEWS = OUT / "previews"
HXENV = dict(os.environ, HERCULEX_DATA=str(DATA), HERCULEX_LIST_CT_BRICKS="1", HERCULEX_FETCH_CONNECTIONS="32",
             HERCULEX_KEEP_STACKS="0", UV_CACHE_DIR="/tmp/uvcache", PYTHONUNBUFFERED="1")
STATE = {"stage": "start", "log": None, "timed_out": False}
TIMINGS = {}

POST = r'''
import csv, json, math, sys
from pathlib import Path
import numpy as np, tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
scan, out = Path(sys.argv[1]), Path(sys.argv[2])
prev = out / "previews"; prev.mkdir(parents=True, exist_ok=True)
rows, units = [], {}
for rec_path in sorted((scan / "pieces").glob("rows-*/record.json")):
    unit = rec_path.parent.name
    rec = json.loads(rec_path.read_text())
    units[unit] = {k: v for k, v in rec.items() if k != "pieces"}
    for r in rec["pieces"]:
        if not r.get("written"):
            continue
        iso = r.get("isometry") or {}
        lk, al = iso.get("locked") or {}, iso.get("all") or {}
        rows.append({"unit": unit, "piece": r["piece"], "source": r.get("source", r["piece"]),
                     "area_mm2": round(r["area_mm2"], 2), "cells": r["cells"], "nodes": r["nodes"],
                     "winding": r["winding"], "turns_spanned": round(r["turns_spanned"], 3),
                     "width_mm": round(r["width_mm"], 2), "height_mm": round(r["height_mm"], 2),
                     "median_radius_mm": round(r["median_radius_mm"], 2), "crowding": round(r["crowding"], 4),
                     "isometric": iso.get("isometric"), "locked_within_10pct": lk.get("within_10pct"),
                     "all_within_10pct": al.get("within_10pct"), "locked_jitter": lk.get("jitter")})
(out / "units.json").write_text(json.dumps(units, indent=1))
summary = {"units": len(units), "written_pieces": len(rows),
           "unwrapped_cm2": round(sum(u.get("area_cm2", 0) for u in units.values()), 1)}
layout_path = scan / "sheet" / "layout.json"
if layout_path.exists():
    layout = json.loads(layout_path.read_text())
    summary["layout_record"] = layout["record"]
    placed = {(p["unit"], p["piece"]): p for p in layout["pieces"]}
    for r in rows:
        p = placed.get((r["unit"], r["piece"]))
        if p:
            r.update({k: p[k] for k in ("x_mm", "y_mm", "group", "group_area", "turn", "level", "flip_rows",
                                         "flip_cols", "moved_from_joint_mm")})
    pieces = layout["pieces"]
    summary["laid_pieces"] = len(pieces)
    summary["laid_cm2"] = round(sum(p["area"] for p in pieces) / 100, 1)
    side = pieces[0]["stride"] * pieces[0]["um"]
    ext_w = max(p["x_mm"] + p["width_mm"] for p in pieces); ext_h = max(p["y_mm"] + p["height_mm"] for p in pieces)
    mm_px = max(side, max(ext_w, ext_h) / 16000)
    W, H = int(math.ceil(ext_w / mm_px)) + 2, int(math.ceil(ext_h / mm_px)) + 2
    group = np.full((H, W), -1, np.int32); zmap = np.full((H, W), np.nan, np.float32)
    for p in sorted(pieces, key=lambda p: p["area"]):
        folder = scan / "pieces" / p["unit"] / f"piece-{p['piece']:04d}"
        x, z = tifffile.imread(folder / "x.tif"), tifffile.imread(folder / "z.tif")
        if p["flip_rows"]:
            x, z = x[::-1], z[::-1]
        if p["flip_cols"]:
            x, z = x[:, ::-1], z[:, ::-1]
        f = side / mm_px
        tw, th = max(1, round(x.shape[1] * f)), max(1, round(x.shape[0] * f))
        m = np.asarray(Image.fromarray(((x >= 0) * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)) > 0
        zz = np.asarray(Image.fromarray(np.ascontiguousarray(z, dtype=np.float32)).resize((tw, th), Image.NEAREST))
        r0, c0 = int(round(p["y_mm"] / mm_px)), int(round(p["x_mm"] / mm_px))
        h, w = min(th, H - r0), min(tw, W - c0)
        if h <= 0 or w <= 0:
            continue
        sub = m[:h, :w]
        group[r0:r0 + h, c0:c0 + w][sub] = p["group"]
        zmap[r0:r0 + h, c0:c0 + w][sub] = zz[:h, :w][sub]
    area_of = {p["group"]: p["group_area"] for p in pieces}
    rng = np.random.default_rng(358)
    ng = max(max(area_of), int(group.max())) + 2
    pal = (rng.uniform(70, 255, (ng, 3))).astype(np.uint8)
    for g, a in area_of.items():
        if a < 400:
            pal[g] = (110, 110, 110)
    pal[-1] = (0, 0, 0)
    rgb_g = pal[group]
    zn = np.nan_to_num((zmap - np.nanmin(zmap)) / max(1.0, float(np.nanmax(zmap) - np.nanmin(zmap))), nan=-1)
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    t = np.clip(zn, 0, 1) * 4
    lo = np.floor(t).astype(int).clip(0, 3); fr = (t - lo)[..., None]
    rgb_z = (anchors[lo] * (1 - fr) + anchors[lo + 1] * fr).astype(np.uint8)
    rgb_z[zn < 0] = 0
    n = max(1, round(math.sqrt(W / (2 * H))))
    def shelves(img):
        if n == 1:
            return img
        sw = int(math.ceil(W / n)); gap = 8
        canvas = np.zeros((n * H + (n - 1) * gap, sw, 3), np.uint8)
        for k in range(n):
            part = img[:, k * sw:(k + 1) * sw]
            canvas[k * (H + gap):k * (H + gap) + H, :part.shape[1]] = part
        return canvas
    for name, img in (("layout_mosaics", rgb_g), ("layout_height_z", rgb_z)):
        im = Image.fromarray(shelves(img))
        im.save(prev / f"{name}.png", optimize=True)
        small = im.copy(); small.thumbnail((3000, 3000), Image.NEAREST); small.save(prev / f"{name}_3000px.png")
    summary["preview"] = {"mm_per_px": mm_px, "sheet_mm": [ext_w, ext_h], "shelves": n,
                          "colours": "mosaics >= 4 cm2 in colour, smaller grey; height_z = scan z (viridis)"}
rows.sort(key=lambda r: -r["area_mm2"])
if rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "piece_index.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "layout_record"}), flush=True)
'''


def note(**rec):
    rec["t_min"] = round((time.time() - T0) / 60, 1)
    print("NOTE", json.dumps(rec), flush=True)
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def sh(cmd, check=True, env=None):
    print("+", cmd[:300], flush=True)
    return subprocess.run(cmd, shell=True, check=check, env=env)


def monitor():
    while True:
        time.sleep(300)
        print(f"--- {(time.time() - T0) / 60:.0f} min, stage {STATE['stage']}", flush=True)
        subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; "
                       f"free -g | sed -n 2p; df -h /tmp /kaggle/working | tail -2; du -sh {OUTPUTS} 2>/dev/null",
                       shell=True)
        log = STATE["log"]
        if log and Path(log).exists():
            subprocess.run(f"tail -c 3000 {log} | tr '\\r' '\\n' | tail -4", shell=True)


def hx(name, args, env=None, timeout_s=None):
    """One ./run.sh command, logged to logs/NAME.log, killed at the deadline; True when it exits 0."""
    log = LOGS / f"{name}.log"
    STATE["stage"], STATE["log"] = name, str(log)
    began = time.time()
    with open(log, "a") as fh:
        p = subprocess.Popen([str(CODE / "run.sh"), *args], cwd=CODE, stdout=fh, stderr=subprocess.STDOUT,
                             env={**HXENV, **(env or {})}, start_new_session=True)
        limit = min(DEADLINE, began + timeout_s) if timeout_s else DEADLINE
        while p.poll() is None:
            if time.time() > limit:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                STATE["timed_out"] = time.time() > DEADLINE
                break
            time.sleep(5)
    rc = p.returncode
    TIMINGS[name] = {"seconds": round(time.time() - began), "rc": rc}
    note(stage=name, rc=rc, seconds=round(time.time() - began))
    if rc:
        subprocess.run(f"tail -c 4000 {log}", shell=True)
    return rc == 0


PROBE = "from unwrapping.detection.kernel_bindings import gpu_library; gpu_library(); print('sheet-node kernel ok')"


def probe():
    r = subprocess.run([str(PY), "-c", PROBE], cwd=CODE, env=HXENV, capture_output=True, text=True, timeout=900)
    print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
    return r.returncode == 0


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
    if not probe():
        # the locked wheels carry NVRTC 13; a driver older than CUDA 13 cannot load its PTX, NVRTC 12 can
        sh(f"{UV} pip install -q --python {PY} 'nvidia-cuda-nvrtc-cu12>=12.4,<13'", env=HXENV)
        sh(f"{UV} pip uninstall -q --python {PY} nvidia-cuda-nvrtc", check=False, env=HXENV)
        if not probe():
            raise SystemExit("the CUDA sheet-node kernel does not build or load on this machine")
        note(stage="setup", nvrtc="cu12 fallback")
    note(stage="setup", herculex=head)


def restore():
    for name in ("nodes.tar", "level.tar"):
        found = sorted(glob.glob(f"/kaggle/input/**/out/{name}", recursive=True))
        if found:
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            with tarfile.open(found[0]) as tar:
                tar.extractall(OUTPUTS)
            note(stage="restore", source=found[0])


def omp(n):
    return {"OMP_NUM_THREADS": str(n)}


def fit_units():
    """Every unit's level fit, largest first, FIT_WORKERS at a time; failures once more alone on all cores."""
    units = sorted((d for d in NODE_DIR.glob("rows-*") if (d / "record.json").exists()),
                   key=lambda d: -(d / "nodes.zst").stat().st_size if (d / "nodes.zst").exists() else 0)
    names = [u.name for u in units]
    note(stage="fit-plan", units=names)
    pending, failed, lock = list(names), [], threading.Lock()
    threads = max(1, (os.cpu_count() or 4) // FIT_WORKERS)

    def worker():
        while True:
            with lock:
                if not pending or STATE["timed_out"]:
                    return
                name = pending.pop(0)
            if not hx(f"fit-{name}", ["unwrapping.level.run_levels", "unit", SCROLL, name], omp(threads)):
                failed.append(name)

    pool = [threading.Thread(target=worker) for _ in range(FIT_WORKERS)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    for name in list(failed):
        if STATE["timed_out"]:
            break
        if hx(f"fit-{name}-retry", ["unwrapping.level.run_levels", "unit", SCROLL, name], omp(os.cpu_count() or 4)):
            failed.remove(name)
    done = [n for n in names if (LEVEL_DIR / "pieces" / n / "record.json").exists()]
    note(stage="fits", done=len(done), of=len(names), failed=failed)
    return len(done) == len(names) and bool(names)


def size_gb(path):
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 2**30
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 2**30


def tar_dir(src, dest, arcname, compress=False):
    def keep(info):
        base = os.path.basename(info.name)
        return None if base.endswith(".building") or base.endswith(".claim") or "/faces" in info.name else info
    t = time.time()
    with tarfile.open(dest, "w:gz", compresslevel=3) if compress else tarfile.open(dest, "w") as tar:
        tar.add(src, arcname=arcname, filter=keep)
    note(stage="tar", file=dest.name, gb=round(dest.stat().st_size / 2**30, 2), seconds=round(time.time() - t))


def package(complete):
    STATE["stage"] = "package"
    if LEVEL_DIR.exists():
        r = subprocess.run([str(PY), str(ROOT / "post.py"), str(LEVEL_DIR), str(OUT)], cwd=ROOT,
                           capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        tar_dir(LEVEL_DIR, OUT / "level.tar", str(LEVEL_DIR.relative_to(OUTPUTS)))
    used = size_gb(OUT)
    nodes_gb = size_gb(NODE_DIR)
    sheet = OUTPUTS / "website" / "sheets" / "herculex-0358" / "tifxyz"
    if not complete and NODE_DIR.exists() and used + nodes_gb < BUDGET_GB:
        tar_dir(NODE_DIR, OUT / "nodes.tar", str(NODE_DIR.relative_to(OUTPUTS)))
        used = size_gb(OUT)
    if sheet.exists() and used + size_gb(sheet) * 0.5 < BUDGET_GB:
        tar_dir(sheet, OUT / "sheet_tifxyz.tar.gz", "sheet_tifxyz", compress=True)
        used = size_gb(OUT)
    if complete and NODE_DIR.exists() and used + nodes_gb < BUDGET_GB:
        tar_dir(NODE_DIR, OUT / "nodes.tar", str(NODE_DIR.relative_to(OUTPUTS)))
    elif NODE_DIR.exists() and not (OUT / "nodes.tar").exists():
        note(stage="package", skipped="nodes.tar", nodes_gb=round(nodes_gb, 2), out_gb=round(size_gb(OUT), 2))
    (OUT / "timings.json").write_text(json.dumps({"complete": complete, "timed_out": STATE["timed_out"],
                                                  "herculex": HX_COMMIT, "total_minutes": round((time.time() - T0) / 60, 1),
                                                  "stages": TIMINGS}, indent=1))
    sh(f"ls -la {OUT} {PREVIEWS} 2>/dev/null; du -sh {OUT}", check=False)


def main():
    for d in (OUT, LOGS, PREVIEWS, ROOT):
        d.mkdir(parents=True, exist_ok=True)
    (ROOT / "post.py").write_text(POST)
    threading.Thread(target=monitor, daemon=True).start()
    setup()
    restore()
    ok = hx("sheets", ["runner.pipeline", "run", "sheets", SCROLL, "--scan", SCAN], {"CUDA_VISIBLE_DEVICES": "0"})
    ok = ok and hx("curl", ["unwrapping.level.run_levels", "curl", SCROLL], omp(os.cpu_count() or 4))
    ok = ok and fit_units()
    ok = ok and hx("joins-frame", ["runner.pipeline", "run", "pieces", SCROLL, "--scan", SCAN], omp(os.cpu_count() or 4))
    ok = ok and hx("layout", ["runner.pipeline", "run", "layout", SCROLL, "--scan", SCAN], omp(os.cpu_count() or 4))
    if ok:
        if hx("survey_status", ["website.survey_status", SCROLL, SCAN]):
            shutil.copy(LOGS / "survey_status.log", OUT / "survey_status.json")
        hx("pipeline_status", ["runner.pipeline", "status", SCROLL])
        hx("tifxyz_sheet", ["unwrapping.layout.tifxyz_export", SCROLL, SCAN, "herculex-0358"])
    package(ok)
    if ok and INK and DEADLINE - time.time() > 3 * 3600:
        if hx("ink-weights", ["ink.model.weights", "fetch"]):
            hx("ink", ["runner.pipeline", "run", "ink", SCROLL, "--scan", SCAN], {"CUDA_VISIBLE_DEVICES": "0"})
            if hx("survey_status_ink", ["website.survey_status", SCROLL, SCAN]):
                shutil.copy(LOGS / "survey_status_ink.log", OUT / "survey_status_ink.json")
        (OUT / "timings.json").write_text(json.dumps({"complete": ok, "herculex": HX_COMMIT, "stages": TIMINGS}, indent=1))
    note(stage="done", complete=ok, timed_out=STATE["timed_out"])


if __name__ == "__main__":
    main()
