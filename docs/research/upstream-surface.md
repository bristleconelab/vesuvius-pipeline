# Upstream stage: CT volume -> papyrus surface -> tifxyz, on CPU (research for M7)

State as of 2026-11-16. Read-only research. Nothing was posted, registered or executed from a
downloaded repository. Code was read at the pinned villa commit `2d3fb38d` (local clone); where the
VC3D AppImage `f7756c7` (2026-11-03) differs, it is said. S3 sizes are MEASURED from anonymous
`ListObjectsV2` listings on 2026-11-16 (sum of object sizes). Anything marked **estimate** was not
measured; anything marked **untested** was not run.

## TL;DR

- **The GPU step can be skipped for PHerc0139 at 9.362 µm.** The organisers publish, on the exact grid
  of the volume our render stage already uses (`20250728140407-9.362um`), a thresholded m7 nnU-Net
  surface prediction (13.5 GB level 0) **and the matching normal grids** (11.8 GB) that GrowPatch needs.
  An older recto 3D-U-Net prediction (12.7 GB) is there too. A 12-vertex probe found w035 lying on the
  m7 prediction (11/12 vertices within 2 vx). The Scroll 3 "phantom" defect is absent there (median
  0.015 % of positives on CT == 0).
- **The tracer is `vc_grow_seg_from_seed` (GrowPatch, Ceres least squares, CPU only).** `-v` takes the
  surface-prediction zarr (not the CT), `normal_grid_path` the grids, `--seed x y z`. villa's own
  regression tests grow on exactly this PHerc0139 prediction, at seed `3770 3698 4608`, which lies
  inside the w035 bounding box.
- **Scoring:** villa has no "score one traced surface against a reference tifxyz" command. The closest
  code is `score_surface()` inside `vc_grow_seg_from_segments --sweep` (two-way sampled nearest-vertex
  distances, coverage within a tolerance, area ratio, hole fraction). It always grows first, so for M7 we
  re-implement those definitions (about 150 lines of NumPy/SciPy) and add a sheet-switch / wrong-wrap
  measure and Henderson's ChD and AD.
- **Recommended M7 recipe:** seed GrowPatch on the m7 prediction near the w035 centre, bound it to the
  w035 bounding box, run 4 seeds × 1 thread for about 150 generations, score against w035, then run our
  existing render + ink stages on the grown surface. Estimated cost: about 1.5-4 GB of data and
  0.5-2 h wall on 16 vCPU (**estimate**, untested).

## 1. Surface-prediction inputs already on S3 (PHerc0139)

`s3://vesuvius-challenge-open-data/PHerc0139/representations/` holds `predictions/{surfaces,fibers,lasagna}/`
and `umbilicus/`.

| Product (prefix under `.../representations/predictions/`) | Grid | Format | Size (MEASURED) |
|---|---|---|---|
| `surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr` | 9.362 µm volume, level 0 `[20974, 6621, 6621]` | zarr v2, OME 6 levels, uint8 binary (thresholded at 0.2), blosc-zstd, **192³ chunks** | level 0: **13.48 GB**, 48,430 chunks (mean 278 kB) |
| `surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.normal-grids/` | same | `xy/`, `xz/`, `yz/` one `.grid` file per slice (index = slice coordinate), `*_img/` JPEG previews, `metadata.json` (`grid-step 64`, `spiral-step 20.0`, `input-level 0`) | **11.77 GB** total: xy 3.55 GB (20,974 files), xz 2.87 GB (6,621), yz 3.19 GB (6,621), xy_img 0.80 GB (110) |
| `surfaces/20250728140407-surface-20250701154204-surface-recto-090.zarr` | same `[20974, 6621, 6621]` | zarr v2, 6 levels, uint8, 128³ chunks; `meta.json` name `4.681um_113keV_1.2m_binmean_2_PHerc_0139_surf` | level 0: **12.72 GB**, 124,944 chunks |
| `surfaces/20260102150214-surface-20260203205514-surface-recto-2um-ps192-L0-th0.5.zarr` | 2.399 µm volume | zarr | not measured |
| `lasagna/20260102150214-lasagna-20260419180421-L2/` | 2.399 µm volume, level 2 | `_cos`, `_grad_mag`, `_nx`, `_ny` OME-Zarr + `.lasagna.json` | **43.90 GB**, 1,497,809 objects |
| `fibers/20260102150214-fibers-20260915212757-L1/` (also `...-20260801084232-L1/`) | 2.399 µm volume, level 1 | `_nx`, `_ny`, `_presence` OME-Zarr + `inference.json` | > 6.75 GB (listing stopped at 30,000 objects) |
| `umbilicus/20260102150214-umbilicus-20260826112529.json` | 2.399 µm volume | JSON | 27,960 B |

Provenance of the m7 zarr, from its own `metadata.json`: model `s3://scrollprize-reconstruction/models/m7_nnunet/`,
`disable_tta: true`, threshold 0.2, `topo_postprocess: false`, input = the **masked** 9.362 µm volume,
workflow `surface-inference-9um-pherc0139-20250728140407` created 2026-04-24, generated 2026-05-06.

What this means for us:

- For PHerc0139 at 9.362 µm the full upstream input set (prediction + normal grids) is published. **No
  model inference is needed for M7.** VC3D's Open Data Catalog pairs these same grids with this
  prediction automatically (`apps/VC3D/OpenDataNormalGrids.cpp`).
- The recto-090 prediction has no published normal grids. Using it means running
  `vc_gen_normalgrids generate -i pred.zarr -o grids [--threshold N]` first. That tool exists in villa
  (`apps/CMakeLists.txt`). Whether the AppImage ships it is **not confirmed**: our tool list names
  `vc_ngrids`, which is a different (Ceres) tool. Check with `ls squashfs-root/usr/bin | grep normalgrids`.
- The lasagna and fiber products are on the 2.4 µm grid, which is about 64× more data for the same
  region. They feed Lasagna or spiral fitting, not GrowPatch. Out of scope for M7.

**Known defect to guard against: phantom positives.** On Scroll 3 (PHerc0332), vesuvius-automesh measured
that about 70 % of the m7 positives sit where the masked CT is exactly 0 (a halo ring and end caps).
Interior blocks were clean (≤ 6.1e-05). The tracer does not read the CT, so it follows those shells.
Fix used there: `supported = preds & (CT > 5)`, then regenerate the grids
([automesh TECHNICAL_NOTES §4](https://github.com/spencerdavis-tx/vesuvius-automesh/blob/main/TECHNICAL_NOTES.md);
upstream issue villa#1114). Our prior-work notes quote a 43.3 % phantom share for m7 in general
(Herculaneum Scroll Tools). Our own probe for the w035 region is in §1a.

### 1a. Probe: does the published prediction support w035? (MEASURED, small sample)

Method: a throwaway script (scratchpad, not in the repo) drew 12 random valid vertices of the
w035 9.362 µm tifxyz (`numpy` rng seed 0). For each one it fetched the m7 192³ chunk, the recto
128³ chunk and the CT 128³ chunks under the m7 chunk. It recorded: the Euclidean distance from the
vertex to the nearest prediction voxel ≥ 128; the share of m7 positives in that chunk where the CT
is 0 ("phantom"); and the CT value at the vertex. Wall time was 4 min 33 s, almost all of it
download. Raw lines are kept in the session task output.

| Measure (12 vertices) | m7 (th 0.2) | recto-090 |
|---|---|---|
| Vertex on a positive voxel (distance 0) | 10 / 12 | 11 / 12 |
| Within 2 vx (≈ 19 µm) | 11 / 12 (91.7 %) | 11 / 12 (91.7 %) |
| Median / p90 distance | 0 / 1.37 vx | 0 / 0.90 vx |
| Worst vertex | 5.0 vx (x, y, z = 4139.6, 2800.9, 3023.3; near the bbox z edge) | 5.0 vx (same vertex) |
| Positive share of the whole chunk (chance level of "distance 0") | 12-23 % | 18-47 % |
| Phantom share of m7 positives (CT == 0) | median 0.015 %, max 0.22 % | - |

Reading: w035 lies on both published predictions far above chance. The phantom problem measured on
Scroll 3 is **not** present in these w035-region chunks of PHerc0139 at 9.362 µm. The m7 run used the
masked volume as input, which fits this. The recto prediction is about 2× denser (thicker or more
sheets marked), so m7 is the cleaner input for GrowPatch, and it is also the one that has normal
grids. The sample is small (12 points, 12 chunks of about 5,460 in the bbox), so a full-bbox phantom
count is still worth doing once (step 3 below), but it is no longer a blocker.

## 2. Tracing on CPU: GrowPatch (`vc_grow_seg_from_seed`) and vesuvius-automesh

### 2.1 What the tool does (read from source at 2d3fb38d)

- `apps/src/vc_grow_seg_from_seed.cpp` -> `tracer()` in `core/src/GrowPatch.cpp` (3,457 lines). It grows a
  quad grid (`step_size` voxels per cell, default 20) outward from the seed, one ring per "generation".
  Each new vertex is placed by Ceres nonlinear least squares. The loss terms printed at start are
  `DIST, STRAIGHT, SNAP, NORMAL, REFERENCE_RAY, SDIR`: spacing, smoothness, snapping to the predicted
  surface, and agreement with the normal grid.
- **`-v` must be the surface-prediction volume** ("the volume to pass to the `tracer()` call _should be a
  surface prediction volume!_", `scrollprize.org/docs/35_segmentation.md`). The tool refuses non-uint8
  input and reads level 0 only. Seed search treats values ≥ 128 as surface.
- **Normal grids are effectively required.** `normal_grid_path` is loaded by `NormalGridVolume`. If
  `step_size` is given explicitly it must equal the grids' `spiral-step` (20 here) or the run throws.
  automesh A/B: without grids, "the tracer's surface energy relaxes across wraps"; 235 cm² of no-grid
  traces were rejected wholesale.
- Remote input: `-v` takes `https://…` or `s3://…`. The tool fetches only the chunks it touches and
  keeps them in `~/.VC3D/remote_cache` (or the `cache_root` param). Normal grids can stream too: put a
  `normal-grids-remote.json` `{"url": "<grids url>"}` in a directory laid out as
  `<root>/s3/vesuvius-challenge-open-data/PHerc0139/.../<name>.normal-grids` (`NormalGridVolume.cpp`,
  `vcr::zarr::decodedCachePath`). villa's own test documents both options ("must contain the published
  grids or a normal-grids-remote.json"). Streaming grids is **untested** by us; downloading a slice
  subset is the safe path (§5).
- Output in tifxyz mode: `<target-dir>/auto_grown_<timestamp>/` (or `--segment-name`) holding `x.tif`,
  `y.tif`, `z.tif`, `generations.tif` and `meta.json` (`area_cm2`, the params, the mode). This is the
  same format our render stage reads.
- Threads: tracing "does not scale past a few threads". Measured upstream: wall time per cm² is lowest
  at 4 threads, and unbounded threads cost 2.2-3.5×. VC3D runs it with `thread_limit: 1`. The tool also
  caps the OpenBLAS pool, because spinning BLAS workers cost 1,705 vs 226 CPU-s for 40 generations
  (source comment). **For 16 vCPU: run several seeds in parallel at `thread_limit` 1-4, not one fat run.**
- Determinism: `--rng-seed N` (or `VC_GROWPATCH_RNG_SEED`) is repeatable **only single-threaded**.
  villa's tests use `VC_GROWPATCH_RNG_SEED=1` with `thread_limit: 1`.
- Bounding: `x_min/x_max/y_min/y_max/z_min/z_max` confine growth, `generations` is the total stop
  generation (`0` = no limit), `min_area_cm` discards small results (needs `voxelsize`), and
  `--resume <tifxyz>` continues an earlier run (raise `generations`; it is a total, not an increment).
- Other modes in the same binary: `mode: expansion` seeds new patches from the edges of existing
  `auto_grown_*` patches (an automatic coverage sweep). `mode: gen_neighbor` casts a surface along its
  normals to the next wrap ("copy out/in"). `growth_method: projection|streamline` grows on the **raw
  CT** from its own sheet normals, with no prediction and no grids
  (`docs/grow_patch_sheet_seeds.md`). The docs say plainly that streamline "does not snap back onto a
  particular papyrus surface", so it is a baseline, not the main path. Present at f7756c7.
- AppImage f7756c7 vs pin 2d3fb38d: two later commits touch these files (#2440 vc_reader refactor,
  #2462 object-store rework). The CLI options used below are identical at f7756c7 (checked with
  `git show f7756c7:…`).

### 2.2 Exact command lines found in docs and tests

villa README (`volume-cartographer/README.md`, "Working on remote volumes"):

```bash
echo '{"mode": "seed", "cache_size": 2000000000, "generations": 8, "thread_limit": 4}' > params.json
vc_grow_seg_from_seed -v "$URL" -t out/ -p params.json --seed 1408 1408 1024
# 8 generations: 10-20 s on a 14-core workstation, ~2 MiB fetched (45.5 um PHercParis4)
# resume with generations 16: 1.41 cm^2 -> 7.02 cm^2
```

VC3D GUI "Create Segment (GrowPatch)" writes this params JSON (`SegmentationCommandHandler.cpp`). The VC3D
tutorial says to use 35 iterations:

```json
{"mode": "seed", "step_size": 20, "min_area_cm": <dialog>, "generations": <N>, "thread_limit": 1,
 "normal_grid_path": "<grids paired with the volume>", "voxelsize": <um>}
```

villa regression tests on **our volume** (`core/test/check_growpatch_grid_edge.py`,
`check_growpatch_resume_bounds.py`, issue #2104 / PR #2134):

```python
VOLUME = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
SEED = ["3770", "3698", "4608"]                      # x y z, inside the w035 bbox
params = {"mode": "seed", "generations": 200, "thread_limit": 1, "step_size": 20, "voxelsize": 9.362,
          "x_min": 3072, "x_max": 4095, "y_min": 2944, "y_max": 3967, "z_min": 4352, "z_max": 4863,
          "normal_grid_path": "<published grids or dir with normal-grids-remote.json>"}
# env VC_GROWPATCH_RNG_SEED=1; timeout 1800 s
```

The resume test also checks that the grown surface "does not fold back onto itself (a layer jump)".
It does this with `closest_far_pair`, the smallest 3D distance between vertices at least 6 grid cells
apart. That is a cheap self-intersection / wrap-jump guard we can reuse.

vesuvius-automesh (`tracer/params_seed_v3.json`, villa @ 4d0ce2881):

```json
{"mode": "seed", "step_size": 20, "min_area_cm": 0.2, "generations": 200, "thread_limit": 0,
 "cache_size": 8000000000, "cache_root": "...", "use_cuda": false,
 "normal_grid_path": ".../normal_grids_supported", "normal_grid_level": 0}
```

### 2.3 vesuvius-automesh: what it proved, and what it did not

([repo](https://github.com/spencerdavis-tx/vesuvius-automesh), MIT, pushed 2026-07-14; README + TECHNICAL_NOTES read via the GitHub API)

- Chain: m7 preds -> `preds & (CT > 5)` -> `vc_gen_normalgrids` (about 25 min, 3 directions) -> seed sweep
  over `vc_grow_seg_from_seed` (12 concurrent single-core traces, "minutes per patch", seeds only on
  CT-supported prediction voxels, away from known segments) -> 66-layer render per 25 mm window -> gate.
- Result on Scroll 3: 17 patches, 157 windows, 106 pass render QC, 61 pass topology verification,
  **279.4 cm² verified**, 0 annotation, 0 GPU, Apple-silicon CPU. Per-patch yield is bimodal
  (45-52 cm² each from 7 patches in clean outer wraps; 2 patches 0 cm²). Of the rendered area, 30 % was
  rejected. Where its windows overlap human segments, it tracks them at a **median 28-41 µm** (4
  windows); 5 more windows sit at 140-200 µm, which the author does not claim either way (registration
  uncertainty).
- Gates per 25 mm window: coherence_ratio within 2× of a human-render reference, band_contrast ≥ 1.05,
  merge_fraction < 45 %, mask_fraction ≥ 60 %, **recenter found_fraction ≥ 0.9** (true sheets
  0.97-1.00, wandering traces 0.36-0.54). Topology tier: angular march must not backtrack > 10°,
  prediction-support density, **coverage@50 µm ≥ 0.80 and mean surface distance ≤ 30 µm** against the
  supported predictions.
- Limits the author states: renders at 4.8 µm (not ink-ready); found_fraction saturates on fused
  terrain; calibrated on Scroll 3 only, never on a scroll with confirmed ink. The tracer "free-grows
  past data support" (40-70 % on-sheet per patch), which is why it accepts windows rather than whole
  patches.
- Runtime numbers are qualitative ("minutes per patch", one core). No per-generation timing is
  published. **CPU time for our case is therefore an estimate** (§5).

## 3. Running the surface models themselves on CPU

- Code path: `vesuvius.predict --model_path hf://scrollprize/surface_m7_nnunet (or a local dir)
  --input_dir s3://…9.362um…masked.zarr --input_anon --bbox "z0:z1,y0:y1,x0:x1" --device cpu
  --disable_tta --blend_in_memory out.zarr`, then threshold (`--threshold --threshold_value 0.2`, as
  the published run did). `--device cpu` works since villa #1239 (2026-07-28: autocast was hard-coded to
  CUDA). Since #1988, mirroring TTA off CUDA runs the 8 flips one at a time, so **always disable TTA on
  CPU** (8× cost otherwise). `torch.compile` is off by default on CPU.
- Memory (from `vesuvius/docs/inference.md`): `--blend_in_memory` needs `(classes + 1) × 4` bytes per
  voxel of the bbox grown by one patch. For the 2-class m7 model and a 512³ bbox that is 7.2 GiB, which
  fits in 62 GB.
- Model facts: m7 = nnU-Net v2 `nnUNetResEncUNetLPlans` 3d_fullres, patch 192³, one fold, about 783 MB,
  Apache-2.0. `surface_recto_3dunet` = ResEnc U-Net with scSE, features up to 320 over 7 stages,
  **patch 256³**, MIT; the card gives no timing or hardware.
- **Nobody has published a CPU timing.** None was found in villa's docs, commit messages, the HF cards
  or the community repos read (the legacy tutorial assumes a 10 GB GPU; HercUNet quotes about
  5 min per 192³ cube on GPU). Patch count is exact arithmetic: at overlap 0.5 a 256³ bbox needs 8-27
  m7 patches, and a 512³ bbox (grown to 862³) about 512. Per-patch CPU time is **unknown**. My
  **estimate** for a ResEnc-L 192³ forward pass on 16 vCPU fp32 is about 1-10 min. That puts a 256³
  crop at roughly 10 min-4 h and a 512³ crop at 8-85 h. How to measure: one `--bbox` of exactly
  192³ with `--device cpu --disable_tta`, timed. That is the only honest way to pin this number.
- Conclusion: for M7 on PHerc0139, use the published prediction. CPU surface inference is only worth
  benchmarking as a side measurement (one 192³ patch), or for scrolls that have no published
  prediction (M8).

## 4. Scoring an automatic surface against a published tifxyz

| Metric family | Where defined in villa / literature | Usable for one patch vs w035? |
|---|---|---|
| Two-way sampled nearest-vertex distance (mean, RMS, p95), coverage within a tolerance (default 10 vx), `area_ratio`, `enclosed_hole_fraction`, `valid_bbox_fill_fraction`, `mesh_completeness_score = target_coverage × min(area_ratio, 1) × fill × (1 − holes)`, `objective = |area err| + mean/tol + (1 − target_cov) + 0.5 (1 − result_cov)` | `score_surface()` in `apps/src/vc_grow_seg_from_segments.cpp` (`--sweep <target tifxyz>`, `--sweep-metric-tolerance 10`, `--sweep-max-distance-samples 20000`) | Yes, but the CLI always grows first: it cuts a 5 cm seed patch out of the target and grows from source patches. **Re-implement the definitions** for score-only use. |
| Spiral "satisfaction": a patch quad counts as satisfied within **6 vx** absolute (and 0.45 winding in spiral space); a patch is satisfied if ≥ 95 % of its quads are | `spiral-fitting/satisfaction_metrics.py` `metrics_config` | Use the 6 vx / 95 % thresholds as a "patch agrees" rule. The spiral-space part needs a fit, so not applicable. |
| Henderson (WACV 2026): **ChD** = one-directional chamfer distance GT -> prediction; **WJF** = fraction of GT slice-line segments whose ends map to different spiral windings; **AD** = mean angular defect (should be about 0 for a developable sheet); **Str** = scan-space vs UV edge length | arXiv:2512.04927 | ChD and AD: yes. WJF needs a spiral; replace it with the wrong-wrap fraction below. Str needs a UV map (the tifxyz grid is one, so it is cheap). Implementation in pmh47/spiral-fitting: not confirmed. |
| Sheet switch / fold guards | `scripts/check_sheet_grid.py` (row/column order reversals, crossed quads, flipped normals, spacing percentiles); `closest_far_pair` in `check_growpatch_resume_bounds.py` | Yes, run on every grown surface. Neither is a general self-intersection detector (stated in the docs). Community: `windcheck` (MIT). |
| Kaggle TopoScore / SurfaceDice / VOI | competition metric | For voxel predictions, not meshes. Not for M7. |
| automesh render gates (found_fraction ≥ 0.9, coverage@50 µm ≥ 0.80, mean distance ≤ 30 µm) | vesuvius-automesh | Optional second tier after render. |

Proposed `src/vpipe/surface_compare.py` (new, tested like `compare.py`). Inputs: auto tifxyz, reference
tifxyz, voxel size. Outputs a JSON with:

1. `pred_to_ref` and `ref_to_pred` nearest-vertex distances (KD-tree on the reference upsampled to about 2 vx
   spacing, so the 20 vx grid does not dominate), reported in µm: mean, median, p95, and coverage at
   2 vx (≈ 19 µm, about half a sheet thickness), 6 vx (satisfaction rule) and 10 vx (villa sweep default).
   Note: 10 vx ≈ 94 µm, which on PHerc0139 may already be a neighbouring wrap.
2. **Area overlap**: the area of w035 covered within 6 vx, in cm² and as a fraction of w035's
   21.98 cm² (25,082,013 vx² × 9.362² µm²).
3. **Wrong-wrap / sheet-switch fraction**: for pred vertices inside w035's footprint, the signed offset
   along w035's normal. Vertices with |offset| > half the local wrap spacing are on another wrap.
   Connected regions of them are sheet switches; report count and area. The local wrap spacing comes
   from `gen_neighbor` casts or from prediction peaks along the normal (to measure; unknown now).
4. Henderson ChD (ref -> pred) and AD on the pred grid. Fold checks from `check_sheet_grid.py`.
5. Pass rule for M7 (proposed, to tune on data): `ref_coverage@6vx ≥ 0.8`, `pred_precision@6vx ≥ 0.9`,
   0 crossed quads, wrong-wrap area < 1 %.

Ink-level check (second tier): render the auto surface, run `ink_9um`, then resample its ink map onto
w035's grid through the nearest-vertex correspondence of step 1. Compute AUC/AP against w035's labels
with our existing `evaluate.py`. This is the number that matters for the prize, and it reuses M1-M3 code.

## 5. M7 step list: PHerc0139 w035, CPU only

Reference: w035 9.362 µm tifxyz = 291 × 262 grid, step about 20 vx (scale 0.05), 63,503 valid vertices
(83.3 %), area 21.98 cm². bbox (x, y, z) = (1792-4420, 2289-4572, 2843-8061). Centre-most valid vertex
(x, y, z) = (4194.3, 3836.6, 5442.6) (MEASURED from the downloaded tifxyz).

| # | Step | Command / detail | Data (MEASURED or **estimate**) | CPU time (**estimate**) |
|---|---|---|---|---|
| 0 | Confirm tools in the AppImage | `vc3d vc_grow_seg_from_seed --help`; `ls squashfs-root/usr/bin \| grep -E 'grow_seg|normalgrids'` | - | seconds |
| 1 | Fetch reference | already in `fetch_w035.sh` (tifxyz 0.9 MB) | MEASURED 0.9 MB | seconds |
| 2 | Normal grids for the bbox (download, not stream) | `xy/00{2843..8061}.grid`, `xz/00{2289..4572}.grid`, `yz/00{1792..4420}.grid` + `metadata.json`, a margin of about 64 slices each side, into `/data/vesuvius/grids/m7-9um/` | MEASURED per-plane means -> **about 3.1 GB** (xy 0.88 + xz 0.99 + yz 1.27); whole set without `*_img` 9.6 GB | 2-5 min at 10-25 MB/s |
| 3 | (Guard) phantom / support check | as §1a: share of m7 positives with CT == 0 in the bbox; if it is material, build `preds & (CT > 5)` for the bbox only and run `vc_gen_normalgrids` on it (if the tool is present) | bbox m7 ≤ 1.5 GB (5,460 chunk slots × 278 kB upper bound); CT for a bbox mask is large, so mask only touched chunks | 10-40 min if needed |
| 4 | Seeds | A: villa test seed `3770 3698 4608` (inside the bbox; on w035's wrap or not is **unknown**). B: w035 centre vertex snapped to the nearest m7 voxel ≥ 128. C, D: two more w035 vertices about 1 cm from the centre. Seeds B-D use the reference, so the M7 report must say "one seed point per run taken from the reference". Fully automatic seeding (prediction-only, as automesh does) is M7b. | - | - |
| 5 | Grow | `params.json`: `{"mode":"seed","step_size":20,"generations":160,"thread_limit":1,"voxelsize":9.362,"min_area_cm":0,"cache_size":8000000000,"normal_grid_path":"/data/vesuvius/grids/m7-9um","x_min":1728,"x_max":4484,"y_min":2225,"y_max":4636,"z_min":2779,"z_max":8125}`; `VC_GROWPATCH_RNG_SEED=1 vc3d vc_grow_seg_from_seed -v s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr -t runs/m7/ --segment-name seedB -p params.json -s X Y Z`. Run 4 seeds in parallel (4 × 1 thread). | m7 chunks touched: about 1,000 × 0.3-0.6 MB = **0.3-0.6 GB** (surface area / 192² × 1.5) | about 15-90 min per seed, 1 thread (from "minutes per patch" at 200 gens and 226 CPU-s per 40 gens on a remote volume; w035 needs about 145 generations from its centre to reach its edge) |
| 6 | Score | `python -m vpipe.surface_compare runs/m7/seedB <w035 tifxyz> --voxel-size 9.362` (§4) + `check_sheet_grid.py` | - | < 1 min |
| 7 | Render | existing `render_w035.sh` with `-s runs/m7/<best>` (28 slices, `--flip-normals`; check the normal orientation of the grown grid, because the flip may differ from w035) | about 10.9 GB of CT chunks for w035-sized output, mostly cached from M2 if the surface overlaps | about 11 min (M3 measured 11 min 28 s for w035) |
| 8 | Ink | existing `ink_9um` seed42 step-20000 inference | - | about 8 min (M3: 7 min 50 s) |
| 9 | Ink score | map onto w035 via §4 correspondence, AUC/AP/F0.5 vs w035 labels | - | about 1 min |

Total: about 1.5-4 GB new data, about 0.5-2 h wall on 16 vCPU (**estimate**, dominated by step 5).

Untested as of 2026-11-16 (not run on our VM; the AppImage was not executed here):

- that the f7756c7 tracer runs headless from the extracted AppImage the way `vc_render_tifxyz` does;
- streaming grids through `normal-grids-remote.json` vs a partial slice download (behaviour when a
  needed slice file is missing is unknown, so the bounds plus a margin must cover every access);
- per-generation CPU time on our VM;
- whether the grown surface stays on w035's wrap (the main risk; this is exactly what step 6 measures);
- normal orientation of the grown grid relative to w035 for `--flip-normals`;
- `vesuvius.predict --device cpu` timing on any crop.

## 6. Sources

- villa @ 2d3fb38d (local clone): `volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp`,
  `core/src/GrowPatch.cpp`, `core/src/NormalGridVolume.cpp`, `apps/src/vc_grow_seg_from_segments.cpp`,
  `apps/VC3D/SegmentationCommandHandler.cpp`, `core/test/check_growpatch_grid_edge.py`,
  `core/test/check_growpatch_resume_bounds.py`, `scripts/check_sheet_grid.py`,
  `docs/grow_patch_sheet_seeds.md`, `docs/sheet_growth_fronts.md`, `docs/normal_grid_threshold.md`,
  `README.md`, `vesuvius/docs/inference.md`, `spiral-fitting/satisfaction_metrics.py`,
  `scrollprize.org/docs/{06_tutorial_VC3D,35_segmentation,38_tutorial_spiral}.md`; commits c73374b4f
  (#1239), 27a2fe3d8 (#1988), ded54c6ca (#2134).
- S3 listings and `metadata.json` / `.zarray` / `meta.json` of the products in §1 (2026-11-16).
- vesuvius-automesh README and TECHNICAL_NOTES (GitHub API, 2026-11-16).
- Henderson, arXiv:2512.04927 (HTML, metric definitions).
- HF card `scrollprize/surface_recto_3dunet`.
