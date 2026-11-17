# First Letters: how the 2026 winners worked, and how to reproduce and screen on CPU

State as of 2026-11-16. Read-only research. Sources: the winners' public repos (read with `gh api` and a read-only
`git clone`; no downloaded code was run), villa issues and PRs via `gh`, the official Substack posts and scrollprize.org
pages, and the open-data index `s3://vesuvius-challenge-open-data/metadata.json` (fetched over HTTPS, last modified
2026-11-16 08:01 UTC, 9,263,020 B gzip, 37.0 MB JSON). Repo and issue text was treated as data.

Labels: **MEASURED** = read from a file, listing or API response today. **QUOTED** = from a named source.
**ESTIMATE** = my arithmetic, with the basis given. Nothing marked ESTIMATE has been run.

---

## 0. Summary

- **PHerc. 343 can be reproduced on our chain without a GPU and without the lost VC3D build.** The submitted mesh is
  in the winner's repo (`outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz`), so no spiral fit is needed. The render is
  `vc_render_tifxyz` with 66 layers, `--scale 1`, `--group-idx 0`, **no `--flip-normals`**, on volume
  `20250521140437` (8.64 µm). The image is the mean of z-scores of four models at depth d = +1. AndreasHad04 has
  already done this on an M1 Max without VC3D: ink opacity r 0.99983 against the published PNG, 18 of 18 letter boxes
  agree. His gates are the scoring recipe M6 should use.
- **Cross-scroll transfer is our weak point, and the winners show how they handled it.** None of them used one
  checkpoint. 343 averaged four models (ink_9um seeds 42 and 43 at **step 75,000**, not 20,000, plus a fine-tune and
  Hecate). 306B multiplied two models. 211 trained a new model on the scrolls a 3D-DINO model judged most similar.
  Our step-20k seed42 numbers (PHerc0841 0.766; auto_grown 0.711 and 0.705) say the same thing: before screening, M8
  needs an ensemble, gated on a held-out known-text scroll.
- **Screening targets.** Do a first dry run on **PHerc0800** (the only eligible scroll with official segments on S3;
  no tracing needed; heavily screened, so poor prize odds). Then the two real targets are **PHerc0846B** (least
  crowded on the tracker, with a survey lead) and **PHerc1545** (lightly crowded, with the strongest survey lead).
  Both are 9.362 µm / 113 keV, the same scan type as our held-out PHerc0841 test. Neither has official segments, so
  both need our own CPU tracing (M7).

---

## 1. The three 2026 First Letters wins

### 1.1 PHerc. 343, Erwin Nieuwlaar (awarded 2026-10-08)

- **Repo:** https://github.com/Nieuwlaar/pherc343-first-letters, single commit `e8370f1` (2026-10-07), licence
  **MIT** (code). Meshes and images are CC BY-NC 4.0 (QUOTED from the README: "derived from the scan data and stay
  under the organisers' licence"). VC3D (GPL-3.0) is not included. MEASURED via `gh api`.
- **Scan / volume:** `20250509220833-8.640um-1.2m-116keV` / `20250521140437-8.640um-1.2m-116keV-masked.zarr`,
  `s3://vesuvius-challenge-open-data/PHerc0343/volumes/`. Level 0 shape [17998, 8595, 8595], 128³ chunks, uint8,
  **uncompressed** (`"compressor": null`), so 2 MiB per chunk (MEASURED `.zarray`).
- **Mesh source** (QUOTED `mesh/README.md`): the villa spiral chain at commit `0fb38c45`, band z 9500–11000, with "the
  public m7 surface prediction, lasagna normals and spiral tracks of PHerc. 343 and my own scroll axis".
  1. Distance field: `make_surf_sdt.py` on m7 at level 1, threshold 128, z 9500–11000 ± 64, masked by CT occupancy at
     level 4.
  2. `fit_spiral`: clockwise, no patches, overrides `{"z_begin": 9500, "z_end": 11000, "input_disable_patches": true,
     "loss_weight_shell_outer": 0, "loss_weight_shell_patch_radius": 0, "dense_spacing_mode": "phase",
     "optimizer_num_training_steps": 30000}`. One-line patch so it runs without `outer_shell/`.
  3. `snap_meshes.py`: moves every node along its normal onto the nearest predicted sheet. This is his "new way to snap
     the segments" from the Substack post.
  4. Lasagna flatten (`configs/flatten_fast_nofilter.json`, `--device cuda`).
  5. `build_concat.py` joins windings w047 and w048 across the θ 360|0 seam (transition strip, max shift 64 px).
  - Final mesh: 128 × 336 nodes, 20 voxels per node, 27,598 valid nodes, `area_vx2` 10,835,600 (= 8.09 cm² at
    8.64 µm). Bbox x 3954–5985, y 3331–5263, z 9503–11004 (MEASURED `meta.json`).
- **Render** (QUOTED `mesh/render_66_layers.sh`): VC3D `41701aa` (2026-09-24). Builds `d285029` and `33b0b91` are
  stated to give bit-identical layers:
  `vc_render_tifxyz --volume CT --group-idx 0 --scale 1 --segmentation MESH --num-slices 66 --slice-step 1 --cache-gb 4 --zarr-output OUT`,
  with `QT_QPA_PLATFORM=offscreen`. "Layer k of 66 samples the CT at (k − 32.5) voxels along the mesh normal ... depth
  d = k − 33. **No --flip-normals**: + is away from the scroll axis." Output 66 × 2560 × 6720 uint8, 945,626,701 B on
  disk, 1009 files (MEASURED sidecar).
- **Ink models** (MEASURED `ink/models.tsv`, all sha256-pinned):
  | key | file | source |
  |---|---|---|
  | dnative | `dense_native-016000.pth` (sha `aacf6cb8…`, 414,967,866 B, not public) / rebuilt from `huggingface.co/Nieuwlaar/ink9um-dense-native` safetensors + config | his fine-tune (Substack: "PHerc. 0139 used for Erwin's dense_native fine-tune") |
  | hybrid_3d2d-seed42 | `scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000.pth` (sha `e635558a…`) | official |
  | hybrid_3d2d-seed43 | `scrollprize/ink_9um` `hybrid_3d2d-seed43/step-075000.pth` (sha `2aeaa85a…`) | official |
  | hecate | `scrollprize/hecate` `hecate_9.6um.pth` (sha `809f4f10…`) + `hecate.py` (sha `c232c18a…`) | official, 9.6 µm |
  "None of them was trained on this PHerc. 343 volume."
- **Inference** (QUOTED `ink/scripts/run_ink.sh`): villa models use a 17-layer window centred on layer 33 + d,
  `python -m vesuvius.ink_detection.inference.infer <zarr> <ckpt> out.tif --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --layer-start S --layer-end S+17`
  with S = 25 + d. The `_reverse` output is the **reading side**. Hecate: `hecate_prep.py` resamples to 9.6 µm with 16
  planes centred on 32.5 + d, then `hecate.py --spacing-um 9.6 --reverse`. The sweep covers every d in −15..+15, for
  all 4 models, on two tilings (the render, and a copy zero-padded by 40 rows and 200 columns). `merge_tile_phase.py`
  then takes the w047 pixels from the padded run.
- **Post-processing** (QUOTED README and sidecar): per model, z = (map − mean)/sd, with mean and sd pooled over the
  reading-side maps at all 31 depths. Ensemble = mean of the four z-scores, stretched p1..p99.5
  (p1 = −0.71824, p99.5 = 3.86293). Shown at **d = +1**, over dimmed CT layer 34 (stretch 26–164, dim 0.45). The ink
  colour is (255, 235, 120) with opacity a = 0.9·clip((u − 0.35)/0.40, 0, 1). The shown area is 48.7 × 21.2 mm
  (10.34 cm² flat, 6.63 cm² of papyrus). 18 letter boxes in 2 rows. Scale bar 1157.4 px/cm.
- **How the letters were confirmed:** papyrologists. "According to most of our papyrologists, the sequence should
  belong to the expression πᾶ̣ν τ̣ὸ γιν̣όμ̣ε̣[νον" (QUOTED,
  [Substack 2026-10-08](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for)). The letter boxes and
  rows are hand-placed inputs (`inputs/*_letter_boxes.json`, `*_rows.json`). They are drawn as row marks only, not as
  annotations.
- **Compute:** "bought a second hand RTX 3090 and rented GPUs for the bigger runs (about 300 dollars spent on rent
  uptil now)" (Substack). The ink sweep is "2 x 248 maps on 2560 x 6720 px (roughly 6 hours on one RTX 4090)"
  (`reproduce.sh`). Peak GPU memory ~7.5 GB on a 4090.
- **Manual effort:** not stated in hours. The hand inputs were his own umbilicus (`inputs/umbilicus_PHerc0343.json`),
  the choice of windings and seam, letter boxes and rows, and the choice of depth.
- **Reproducibility notes from the author:** the mesh rebuild is bit-exact. A map recomputed on an RTX 3090 against
  the rented 4090/5090 gave r 0.99999. The Dockerfile is "a draft I haven't built".

### 1.2 PHerc. 306B, Liam Warner and Nicola Bodill (announced 2026-11-02)

- **Repo:** https://github.com/SirProdigle/pherc0306b-first-letters, HEAD `feabff6` (2026-11-09). Code **MIT**;
  weights, data and images **CC BY-NC 4.0** (`LICENSE-DATA`). Weights: https://huggingface.co/Prodigle/pherc0306b-ink-ahn
  (6 GB). Data and Docker image: https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters (17 GB). MEASURED.
- **Volume:** `20250521133212` (8.64 µm, 116 keV).
- **How the line was found** (QUOTED `docs/METHOD.md`): "We ran the villa spiral-fit chain over one band of each
  eligible scroll on rented GPUs. The 40 windings were snapped to the organisers' m7 surface prediction, flattened
  with Lasagna, rendered, and inked with Hecate and Reader v2." Each window was scored by row periodicity, a small
  glyph CNN and a line finder. On 306B the glyph CNN "ranked a window on winding 74 first of 72 candidates".
- **Mesh:** VC3D GrowPatch from seed (3559, 3224, 9348) on the m7 normal grids, step 20 voxels, 140 generations.
  Where the patch drifted at the right end of the line, a hand-adopted touch-up moved nodes to where two spiral fits
  and a sheet tracker agree. Then re-flattened.
- **Render:** `vc_render_tifxyz`, **41 layers**, 1 voxel apart, −20..+20, giving 4900 × 7780 px.
- **Ink:** **product** of TRACE v3B (public, Jashann; 8 TTA flips, mean of layers 19–21) and A-hn (v8-in fine-tuned
  3 times for 600 steps; half PHerc. 1447 team labels, half machine labels from 4 public models on 5 other 306B
  surfaces; hard-negative loss; 3-D exclusion zone of 2 mm). One rented GPU, about 12 min per fine-tune, peak 19 GB.
- **Image:** linear from the median to the 99.9th percentile of the fused map; a 3.60 cm² box.
- **Validation** (QUOTED): depth shuffle 8.6 % → 0.00 % dark pixels in the box; a blank strip fixed in advance 0.42 %;
  a decoy model trained on labels shifted 3.5 × 3.0 mm; known text **PHerc. 0841 (held out): AUC 0.874, AP 0.731**
  for the product rule, against 0.870 for TRACE v3B alone and 0.840 for v8-in.
- **Letters confirmed by** the team papyrologist (Bodill) and the VC papyrology team ("τὸ δι᾽ αὑτὸ αἱρ̣ε̣τ̣όν̣").
- **Compute:** "a spare RTX 4090" plus rented GPUs. No total stated.
- **Manual effort:** three hand choices are listed in METHOD §7 (the surface touch-up, the box, the model choice).
  Bodill reports "hundreds of hours manually creating ink labels" since 2023 (Substack).
- **Reproduction:** a Docker image redraws the claim image byte for byte on **4 CPU cores** (126 s). A `smoke` mode
  re-runs A-hn on a 169 × 169 crop in 297 s. The full pipeline mode is "not run end to end".
- **Crowding signal:** "a massive 25 submissions on this scroll"; "multiple submissions identifying the exact same
  area"; submissions on "13 other scrolls" (Substack 2026-11-02).

### 1.3 PHerc. 211, Priyank Aranke (announced 2026-11-15)

- **Repo:** https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt, HEAD `235e3be` (2026-11-13), code
  **MIT**. Model release `studio-softloss-20261023` (`studio-softloss-20261023-ema.safetensors`, 893,443,724 B,
  MEASURED), CC BY-NC 4.0. VC3D fork: https://github.com/priyankaranke/vc3d-manual (GPL-3.0). **The mesh and outputs
  are on Google Drive** (a mesh ZIP), not on GitHub or S3. Not checked.
- **Volume:** `20250821151803` (9.362 µm, 113 keV; the eligible volume).
- **Method** (QUOTED/paraphrased from [Substack 2026-11-15](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211)):
  - He chose training scrolls by **3D-DINO similarity** to the target; PHerc0009B, PHerc0841 and PHerc0814 ranked
    highest.
  - Labels: he ran the team's 2 µm ink model on high-resolution scans and transferred the predictions onto ~9 µm
    segments, then "checked and corrected by hand".
  - Mesh: a full-height spiral fit of 0211, neighbouring windings stitched, refined by hand in VC3D Manual, and
    extended along the rows by DinoVol-guided growth.
  - Render: stock `vc_render_tifxyz`, planes 00–30, model input = planes 23 down to 8 (16 planes) (QUOTED README).
  - Model: residual 3D U-Net, 223,355,921 parameters, 16 native planes, 128 px window. Trained from scratch on
    814/841/9B, then continued on 12 other scroll groups with replay, then a soft-loss stage. **Not trained on any
    competition scroll.** 0211 held-out AP 0.8081 → 0.8222.
- **Compute:** one **NVIDIA GH200 (96 GB)**, ~1 s per update. The continued pretraining took 8 h 46 min. Training
  needs ≥ 80–96 GB of VRAM and ≥ 128 GB of RAM. Inference on CPU is documented (`--device cpu --batch-size 1 --amp fp32`,
  "slower").
- **Manual effort:** hand surface refinement and hand label correction; no hours given.
- **Crowding signal:** "37 submissions on PHerc. 211 from 15 individual contestants and teams". "There are still
  seven First Letters Prizes waiting". The organisers "recently received several particularly promising
  submissions". MEASURED: `prizeEligibility.json` in villa still lists PHerc0211; the site has not been updated for
  this award yet.

### 1.4 What the three have in common (inference)

1. Geometry comes from the public m7 surface prediction, via the spiral fitter or GrowPatch, and is **hand-corrected**
   near the letters.
2. **Several models combined** (mean of z-scores, or a product), with at least one model or fine-tune trained off the
   target scroll. No winner relied on one off-the-shelf checkpoint.
3. Depth matters: 343 swept 31 depths and chose d = +1; 306B used a 41-layer render; 211 used 16 of 31 planes.
4. Each submission carries explicit controls against hallucination: other depths, depth shuffle, a blank strip fixed
   in advance, a decoy model, or a held-out known-text scroll (PHerc0841 for 306B; nine labelled 0211 segments held
   out for 211).
5. All were found with GPUs. Only the final redraw (306B) or the render (343, via AndreasHad04) has been shown to
   reproduce on CPU.

---

## 2. villa #2049 and the CPU renderer: what 343 reproduction needs

### 2.1 The issue (MEASURED via `gh issue view 2049`, open)

- Opened by rodriguescarson on 2026-10-12. VC3D promotion deletes the previous `stable`/`latest` release, so
  **`41701aa` cannot be downloaded**. His clean-box reproduction rebuilt the segment bit for bit, then failed at the
  render step.
- Release listing today (MEASURED): `vc3d-archive-latest-*` tags exist from 2026-11-03 onward, plus one
  `vc3d-archive-stable-363877681` (2026-08-02). None of them is 41701aa (2026-09-24). The archive gap still covers the
  343 build.
- **AndreasHad04 (comment 2026-10-12):** wrote a CPU Python renderer, `firstletters/render66.py` +
  `render_tifxyz.py`, in https://github.com/AndreasHad04/villa-apple-silicon (MIT; commits `4c7f4c1`, `3506341`). It
  was checked against the published PNG, whose grey pixels are CT layer 34 under stretch 26–164 and dim 0.45:
  | his layer | share of 7,523,228 grey pixels within 1 grey level |
  |---|---|
  | 34 | 0.9999991 |
  | 33 / 35 | 0.1410 / 0.1405 |
  | 31 (flipped normal) | 0.0603 |
  7 pixels differ by more than 1. The layer sha256 does **not** match the sidecar. With the four models at d = +1 on
  both tilings (M1 Max, MPS, fp32) and his `make_submission_image.py` fed the **sidecar statistics**:
  | measured against the published image | value |
  |---|---|
  | Pearson r of ink opacity, 8,840,757 px | **0.99983** |
  | the same inside the 18 letter boxes | 0.99984 |
  | Pearson r of luminance | 0.99990 |
  | letter boxes with ink fraction within 0.8–1.25 of the original | 18 / 18 |
  | depth control: d = 0 / d = +2 instead | 0.9009 / 0.8841 |
  Source: `results/nieuwlaar/nl343_summary.json` (MEASURED). The pre-registered full arm (all 248 runs, statistics
  recomputed from his own maps) "has not finished yet" (README, last push 2026-10-31).
- **rodriguescarson (2026-10-13, 2026-10-18):** compared the renderer with VC3D `4b3c728` on PHerc0800 `z7472_w060`
  and PHerc0813 `z4704_w080`. On the same grid (`--num-slices 31 --flip-normals`) for 0813: 99.603 % of voxels
  identical, 100.000 % within 1, max |Δ| 1, r 0.9999985. Laptop streaming cost: 974–1320 chunks, 2.0–2.8 GB, 331–534 s
  fetch. Gotcha: VC3D's default 66-layer grid sits half a voxel off its 31-layer grid, so the two grids are only
  comparable after averaging adjacent layers.
- **saschaivanov-lgtm** proposed create-only archived releases (2026-10-19, 10-27). No maintainer reply yet.

### 2.2 Exact inputs for 343 on our chain

| input | where | size |
|---|---|---|
| mesh `concat_w047-w048_R5B2_z9500-11000.tifxyz` (x/y/z.tif + meta.json) | winner repo `outputs/` at `e8370f1`; sha in `outputs/SHA256SUMS` (x `8f31c8d7…`, y `7d416197…`, z `04feacf2…`) | 0.52 MB (MEASURED) |
| CT, level 0 | `s3://vesuvius-challenge-open-data/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0` (https: `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr`) | 1,053 chunks / **2.21 GB** needed (QUOTED, AndreasHad04). Upper bound if a renderer reads the whole bbox: 17 × 16 × 12 = 3,264 chunks × 2 MiB = **6.8 GB** (ESTIMATE). Our VC3D streamed 10.9 GB for w035, so VC3D's cache may over-read; measure it. |
| reference image + sidecar | `outputs/concat_…png` (sha `59307c84…`, 8.8 MB), `outputs/concat_….json` (all stats, area, letter boxes, layer-34 sha `df1a3e8d…`, valid-mask sha `f6054921…`) | 8.8 MB |
| letter boxes, rows, umbilicus | `inputs/` | < 15 kB |
| checkpoints | ink_9um seed42 and seed43 step-075000, `scrollprize/hecate` (weights + `hecate.py`), `Nieuwlaar/ink9um-dense-native` safetensors + config (rebuilt into a `.pth` by his `rebuild_dense_native_pth.py`) | sizes not measured; exact dnative `.pth` is 0.41 GB, so ~0.4 GB each is the ESTIMATE, ~1.5–2 GB total |
| code | `ink/scripts/{pad_render,merge_tile_phase,ensemble_maps,make_submission_image}.py` from the winner repo (MIT) | small; **review before running**, it is third-party code |
| villa for ink | `0fb38c45` (2026-08-18; Python 3.14). Our pin `2d3fb38d` (2026-11-14) is 513 commits ahead, with 13 commits to `ink_detection/inference` in between (e.g. #2003, #2169, #2207, #2400; MEASURED). Some claim identical output and some change behaviour, so run both and report the drift. | — |
| VC3D | 41701aa is not available. Use our pinned `f7756c7` with his flags, and AndreasHad04's `render66.py` as an independent second renderer. | — |

**Volume IDs to know:** the 343 scroll volume `20250521140437`. The fragment `PHerc0343P` (`20250521134555`,
`20260304131111`) is **a different object**: it has 8 segments with ink and is in Reader v2/v3 training data. Do not
mix them up.

---

## 3. The official reference workflow (Substack 2026-08-18, Paul Henderson) and the tutorials

Source: [From CT Scan to Ancient Text: A First Letters Prize Workflow](https://scrollprize.substack.com/p/from-ct-scan-to-ancient-text-a-first),
[tutorial_spiral](https://scrollprize.org/tutorial_spiral), [tutorial5](https://scrollprize.org/tutorial5),
[tutorial_VC3D](https://scrollprize.org/tutorial_VC3D). Summary of the recommended steps:

1. **Pick an eligible volume.** Open it in VC3D via *File → Open Data Catalog*. Data is streamed.
2. **Surface via the spiral fitter.** Inputs: umbilicus, tracks (`dl.ash2txt.org/datasets/spiral_datasets/<scroll>/<vol>/tracks/`,
   which exist for all FL scrolls; MEASURED), lasagna normals (`normal_zarr_group` "2", `lasagna_scale` 4 for 8/9 µm).
   Run a ~1,000-slice band first with `input_disable_patches`, `input_use_tracks`, no outer shell,
   `dense_spacing_mode: grad_mag`. **Needs an NVIDIA GPU, Python ≥ 3.14** ("a ~1,000-slice range is a good first run
   on a smaller GPU"). Track extraction has a slow CPU fallback. Check the meshes in VC3D with *Attach Segments*.
   Fallback: **GrowPatch** (Ctrl + right-click → Create Segment).
3. **Flatten** with Lasagna (`fit.py configs/flatten_fast_nofilter.json … --device cuda`): optional, GPU.
4. **Render:** `vc_render_tifxyz --group-idx 0 --scale 1 --num-slices 28 --slice-step 1 --cache-gb 16 --zarr-output …`
   (28 layers "suit the released 9 µm ink models"). Look at the layers and maximum-intensity composites first: "If
   that's already enough legible letters, that by itself qualifies".
5. **Ink:** `scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth`, `--overlap 0.5 --blend-mode hann --batch-size 4 --direction both`.
6. **Label → retrain → held-out test → repeat** (tutorial5 "iterative labeling").
7. **False-positive checks:** small receptive field; never train on the claim region; rows must be coherent and
   aligned with the fibres. If nothing shows, consider: no ink here / model does not transfer / mesh jumps sheets.
   Then check the mesh in cross-section, search above and below, try another winding or scroll.

**Not in the official workflow, but used by every winner** (section 1.4): a depth sweep, model ensembles, and explicit
controls (depth shuffle, a blank strip fixed in advance, a held-out known-text scroll).

---

## 4. Eligible scrolls: public geometry and crowding

### 4.1 What is on S3 (MEASURED from `metadata.json` + S3 `ListObjectsV2`, 2026-11-16)

The First Letters list in villa `prizeEligibility.json` has 20 volumes (0211 still listed). Every eligible volume has an
m7 surface prediction and normal grids on S3. Spiral **tracks** exist for all of them on `dl.ash2txt.org`.

| scroll | eligible volume (µm/keV) | level-0 shape (z,y,x) | official segments | lasagna | umbilicus | villa issues / PRs mentioning it* |
|---|---|---|---|---|---|---|
| PHerc0125 | 20250821151825 (9.362/113) | 20840, 8387, 8387 | 0 | yes | yes | 8 / 19 |
| PHerc0175A | 20250521115057 (8.64/116) | 12748, 9363, 9363 | 0 | no | no | 2 / 3 |
| PHerc0175B | 20250521125822 (8.64/116) | 15897, 9613, 9613 | 0 | no | no | 1 / 1 |
| PHerc0191 | 20250821151635 (9.362/113) | 18977, 8387, 8387 | 0 | yes | no | 2 / 10 |
| PHerc0211 | 20250821151803 (9.362/113) | 19416, 7948, 7948 | 0 | yes | yes | won 2026-11-15 |
| PHerc0257 | 20250821151750 (9.362/113) | 18872, 8388, 8388 | 0 | yes | no | 2 / 2 |
| PHerc0268 | 20251110183117 (8.64/116) | 14833, 12145, 12145 | 0 | yes | no | 2 / 4 |
| PHerc0358 | 20250821151737 (9.362/113) | 14744, 7783, 7783 | 0 | yes | yes | 3 / 8 |
| PHerc0483A | 20250521140913 (8.64/116) | 15898, 8849, 8849 | 0 | no | no | 2 / 0 |
| PHerc0483B | 20251124083638 (8.64/116) | 11880, 8343, 8343 | 0 | no | no | 1 / 1 |
| PHerc0490A | 20250521151210 (8.64/116) | 11698, 9101, 9101 | 0 | no | no | 2 / 1 |
| PHerc0490B | 20250521151215 (8.64/116) | 10648, 8343, 8343 | 0 | no | no | 3 / 5 |
| **PHerc0800** | 20250521135224 (8.64/116) | 24298, 9867, 9867 | **6** (auto_grown, 2025-10-28/29; ~9.4 cm² valid per #2452; 22.4 cm² by bbox) | yes | yes | 9 / 40 |
| PHerc0813 | 20250821151723 (9.362/113) | 16993, 7947, 7947 | 0 | yes | yes | 8 / 13 |
| PHerc0826 | 20250821151701 (9.362/113) | 16920, 8169, 8169 | 0 | yes | yes | 6 / 26 |
| PHerc0846A | 20250728152254 (9.362/113); also 2.403 µm 20260319102732 | 14019, 7726, 7726 | 0 | no | no | 8 / 5 |
| **PHerc0846B** | 20250804142305 (9.362/113) | 13926, 7859, 7859 | 0 | no | no | **1 / 0** |
| PHerc1203 | 20250820131727 (9.362/113); also 2.403 µm | 18977, 6844, 6844 | 0 in metadata; **22 raw** under `segments/raw/` (~130 cm² by `area_cm2`, quality disputed) | yes | yes | 17 / 26 |
| PHerc1218 | 20250521120456 (8.64/116) | 23247, 7593, 7593 | 0 | yes | no | 9 / 20 |
| **PHerc1545** | 20250821151648 (9.362/113) | 20961, 7506, 7506 | 0 | yes | no | **3 / 1** |

\* `gh search issues|prs --repo ScrollPrize/villa "PHercNNNN"` hit counts, 2026-11-16. Rough: the search tokenises
loosely, and Discord activity (not read) is invisible here.

**Segments per cm²:** only PHerc0800 has official segments (6 segments, ~9.4 cm² of papyrus, i.e. ~0.6 segment per
cm²). On every other eligible scroll the public geometry is community-made:

- **rodriguescarson/eligible-scroll-atlas** (MIT code, CC BY-NC data): 340 meshes on the eight 9 µm eligible scrolls
  (names include 0125, 0813, 0800, 0191, 0826), 83 GB of renders and maps. 5 meshes pass his pre-registered screen,
  and **their maps are held back and sent privately to the organisers**.
- **AndreasHad04/villa-apple-silicon** (MIT): 147 CPU-traced patches, 127.8 cm², on 0483B (39.4), 0175A (39.0),
  0483A (31.1), 0175B (10.6), 0490A (7.7). Made with a CPU tracer that needs no umbilicus (`uncontested/sheet_patches.py`).
- **ketchupbyheinz/vesuvius-scroll-survey** (licence NOASSERTION): all 21 FL scrolls, GrowPatch patches with a sheet
  check and a "face asymmetry" ink diagnostic. **No letters.** Its leads include 846A (2601, 2784, 3851),
  1545 (3499, 4074, 9431), 846B (3561, 4951, 6961), 813 (4779, 5850, 8497) and 257 (3549, 4692, 13210).
- **Bullo27/first-letters-survey:** 65 patches, 971 cm² on 21 scrolls, no letters. "Most automatic patches do not
  follow a single sheet for long near the compressed core" (from our `prior-work.md`).
- **Warner/Bodill** screened "one band of each eligible scroll" with spiral fits (306B METHOD).
- **PHerc1203:** #2534 (2026-11-13) and its replies. The raw meshes fall in two batches; 9 grown without a normal grid
  are likely phantom surfaces. m7 there is "one fused mass". Two CPU attempts saw no strokes.
- **PHerc0800:** #2452/#2453 screened all six segments with a 3-model ensemble on CPU (3.47 VM-hours): no claim
  reported. pscamillo's mesh set also covers 0800.

### 4.2 Choosing what to screen, given our transfer numbers

Our CPU results with ink_9um seed42 **step-20000** (coordinator, 2026-11-16): PHerc0139 w030 AUC 0.927, w045 0.867
(training scroll); **PHerc0841 w00 0.766** (unseen scroll, 9.366 µm/113 keV); two auto_grown segments 0.711 and
0.705. The winners' reference point on the same held-out scroll: the 306B product rule reaches **AUC 0.874 / AP 0.731
on PHerc0841**, TRACE v3B alone 0.870, v8-in 0.840 (306B METHOD §6). So the gap is mostly the model and checkpoint, not
the scan, and closing it comes before screening any new scroll. Consequences:

1. **Model first, then scroll.** Gate M8 on PHerc0841: an ensemble must reach AUC ≥ 0.85 on w00 before any unread
   scroll is screened. Candidates, all public and all CPU-inferable: ink_9um seeds 42 + 43 at step 75,000 (the 343
   recipe), Hecate 9.6 µm, Reader V3 (DomRusso2/reader-v3, "used in the majority of First Letters submissions"),
   TRACE v3B, and dnative. Caveat: TRACE v3B's authors used 0841 for checkpoint selection (about +0.01 AUC
   optimistic), and Reader v2/v3 trained on 0814/0009B. **Record which scrolls each model saw**, and do not score a
   model on a scroll it trained on.
2. **Match the scan type to the evidence we have.** Our only measured unseen-scroll number is on 9.36 µm / 113 keV
   (0841). Eligible scrolls on the same protocol: 0125, 0191, 0257, 0358, 0813, 0826, 0846A, 0846B, 1203, 1545. The
   8.64 µm / 116 keV scrolls (0800, 0175A/B, 0268, 0483A/B, 0490A/B, 1218) would need their own held-out check; the
   closest public one is PHerc0009B (8.64 µm, labelled, but in the Reader/Studio training data).
3. **Similarity selection is cheap on CPU.** The 211 winner ranked scrolls by 3D-DINO similarity to labelled ones. A
   CPU-feasible stand-in: compare CT intensity histograms and the m7 sheet spacing of each eligible volume with 0841
   and 0139 at a coarse pyramid level. ESTIMATE: well under 1 GB per scroll at level 3–4.
4. **Crowding.** Every eligible scroll has been surveyed by someone. The real differentiators are a scroll with few
   tracker mentions plus an open survey lead, and a careful depth sweep with controls, which most surveys did not run.

**Recommendation**

- **Dry run: PHerc0800.** It has 6 official segments with published 31-layer surface volumes (`layers-zarr`), so
  there is no tracing and the render can be scored against the published surface volumes, exactly as for w035. There
  is a published CPU anchor (3.47 VM-hours, #2452), so our negatives are comparable. It is the most screened scroll
  (40 PRs), so treat it as calibration, not a prize bet.
- **Target 1: PHerc0846B.** It is the least mentioned on the tracker (1 issue, 0 PRs). It uses the same scan protocol
  as our held-out 0841 test, so the measured transfer applies directly. There is a survey lead at (3561, 4951, 6961),
  plus two more patches with face-asymmetry 3.5–4.1. Cost: there is no umbilicus or lasagna, so the spiral fitter is
  out. Geometry must come from a CPU tracer (AndreasHad04's `sheet_patches.py`, or `vc_grow_seg_from_seed` with the
  m7 normal grids, CPU), which is M7 work.
- **Target 2: PHerc1545.** Lightly mentioned (3 issues, 1 PR), same scan protocol. It has the strongest single survey
  lead, (3499, 4074, 9431), with face-asymmetry 5.12, the highest of the leads table, above the 343 positive control's
  3.23 on that metric (a diagnostic, not evidence of ink). Lasagna exists (no umbilicus), so a later GPU spiral fit
  stays possible.
- **Avoid for now:** 0800 as a prize bet, 1203 (fused m7, disputed raw meshes, 17/26 mentions), 0826 and 1218 (20+
  PRs each), and 0211 (won).

---

## 5. Concrete step lists

All times below are ESTIMATES for our 16 vCPU / 62 GB / 1 TB VM unless marked MEASURED. The bases are: our w035 run
(MEASURED: render 688 s, 10.9 GB streamed; ink 470 s for one model, one direction, over a 5820 × 5240 px render, about
26.7 cm² by bbox, i.e. ~18 s per cm² of bbox), and #2452 (seed42 on a 1.32 cm² 343 window, both directions, 114 s on 2
Zen 3 cores).

### M6: reproduce PHerc. 343 on our chain, scored numerically against the published image

| # | step | data | CPU time (ESTIMATE) |
|---|---|---|---|
| 1 | Pin inputs. Fetch the winner repo at `e8370f1` (git, no execution); `sha256sum -c outputs/SHA256SUMS`; store a copy under `refs/pherc0343/` with its commit and sha. | 20 MB | 1 min |
| 2 | **G1 mesh gate (optional):** after reviewing it, run `mesh/build_concat.py` per `reproduce.sh` step 1. Pass iff x/y/z sha equal `SHA256SUMS`. The repo already holds the result, so render from `outputs/` if we skip G1. | — | seconds |
| 3 | **Render** with our VC3D `f7756c7`, exactly his flags: `vc_render_tifxyz --volume s3://…/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr --group-idx 0 --scale 1 --segmentation concat_….tifxyz --num-slices 66 --slice-step 1 --cache-gb 16 --zarr-output render66.zarr`. **No `--flip-normals`, no `--voxel-size`** (unlike our w035 runner). Record bytes streamed. | 2.2 GB needed; up to 6.8 GB bbox; VC3D may read more (w035: 10.9 GB) | 10–20 min |
| 4 | **G2 render gate** (his sidecar + the AndreasHad04 recipe): (a) sha256 of decoded layer 34 over raw area rows 105..2560, cols 380..6021 vs `df1a3e8d…`; valid-mask packbits sha vs `f6054921…` (bit-identical = best case). (b) Predict grey pixels P = rint(f32(ct_u8(L34; 26, 164)) · f32(0.45)), map to reading orientation (`raw[::-1, ::-1]`), compare with the PNG data panel (x 173..5814, y 182..2637) where R = G = B, excluding row lines. **Pass iff ≥ 0.99 within 1 grey level**, while layers 33 and 35 stay < 0.99 (discrimination control). (c) Cross-check: render with `render66.py` too, report the voxel agreement. | — | 5 min |
| 5 | Padded copy: `pad_render.py --pad-rows 40 --pad-cols 200`. | +1 GB | 1 min |
| 6 | Checkpoints: download the 4 models, verify sha against `models.tsv`, rebuild dnative from the public safetensors. | ~1.5–2 GB | 5 min |
| 7 | **Arm A (fast, sidecar statistics):** reading side only, d = +1, 4 models × 2 tilings = **8 runs**. villa models on `0fb38c45` (Python 3.14 env), `--device cpu --batch-size 8`; layer window 26..42 (`--layer-start 26 --layer-end 43`). Hecate with `HEC_DEVICE=cpu`. Then `merge_tile_phase.py`, then `make_submission_image.py` with a stats file built from the sidecar (per-model mean/sd; ensemble p1 −0.71824, p99.5 3.86293). | maps ~0.1 GB | villa: ~8 cm² valid × ~18 s ≈ 2.5 min per run → 6 runs ≈ 15 min. Hecate on CPU: unmeasured, guess 5–15 min per run. **Total ≈ 0.5–1 h** |
| 8 | **Score Arm A** (new `vpipe.compare` function + unit tests): opacity a = (R − B)/135 from both PNGs → Pearson r over the valid panel (target ≥ 0.999; AndreasHad04 got 0.99983); r inside the 18 letter boxes; per-box ink fraction within 0.8–1.25 (target 18/18); depth controls d = 0 and d = +2 must score lower (theirs 0.901 / 0.884); image sha equal? (not expected). | — | 5 min |
| 9 | **Drift check:** repeat step 7 with our villa `2d3fb38d` and report r against the `0fb38c45` maps. This answers whether 13 inference commits changed outputs, and closes part of M4's "different inference configuration" question. | — | ~0.5 h |
| 10 | **Arm B (full, statistics recomputed):** 4 models × 31 depths × 2 tilings = **248 runs**, reading side only; `ensemble_maps.py --stat-depths -15..15`; compare our per-model mean/sd and ensemble p1/p99.5 with the sidecar, then the image as in step 8. Resumable, d = 0 outward. Also finishes the pre-registered primary arm that AndreasHad04 left open. | maps ~3–4 GB | 31 × (6 villa × 2.5 min + 2 Hecate × 5–15 min) ≈ **13–23 h** (range 8–30 h) |
| 11 | Publish `results/pherc0343-m6/`: render stats, gate JSONs, timings, figure (his image / ours / |Δa|), pins (VC3D, villa, checkpoint sha). Credit Nieuwlaar (MIT) and AndreasHad04. Images stay CC BY-NC 4.0, and **the reading is already public** (Substack), so publishing the reproduction raises no disclosure issue. | — | — |

**Optional transfer check inside M6:** none of the four models trained on PHerc0343, and his 18 letter boxes are weak
labels. Ink fraction inside the boxes against blank inter-row strips gives a per-model box-level separation on a
scroll unseen by all four. It is cheap, uses the Arm B maps, and puts single models next to the ensemble.

### M8: screen one eligible scroll (dry run on PHerc0800, then PHerc0846B or PHerc1545)

**8a. Model gate (before any screen):** on PHerc0841 w00 (and the two auto_grown segments), score candidate
checkpoints and ensembles with our existing evaluator, all on CPU: seed42/43 at step 75,000, Hecate, Reader V3,
TRACE v3B; combine by mean z-score (343 rule) and by product (306B rule). **Pass = AUC ≥ 0.85** (306B reference 0.874).
Pre-register the gate, the combination rule and the threshold in a dated file before running. ESTIMATE: ~6 models ×
1 segment × 2 directions × a few depths ≈ 2–4 h CPU.

**8b. Dry run, PHerc0800** (calibration):
1. Fetch the 6 official tifxyz meshes and their published `layers-zarr` (31-layer surface volumes). ESTIMATE ~3–5 GB
   of surface volumes; geometry < 50 MB.
2. Render each with our VC3D (31 layers, `--flip-normals` to match the published grid). Score against the published
   surface volumes (render gate as for w035). Stream ~0.5–2 GB per cm² (#2534: ~0.5 GB/cm²; ours: 10.9 GB / 26.7 cm²
   ≈ 0.4 GB/cm²), so ~4–20 GB for 9.4 cm². ~5–15 min per segment.
3. Run the gated ensemble at depths −6..+6, both directions. ESTIMATE: 22.4 cm² bbox × 18 s × ~5 models × 13 depths ×
   2 directions ≈ **29 h**. That is too much, so sweep 3 depths first (≈ 7 h), then the full sweep only where the row
   score fires. Compare our cost with #2452's 3.47 VM-hours.
4. Controls on every candidate window: depth shuffle, a blank strip fixed in advance, the blind side (forward), and a
   neighbouring winding. Publish the negatives (render scores, max row score, per-window maps); maps of any hit stay
   private.

**8c. Target, PHerc0846B (or PHerc1545):**
1. Geometry (M7 dependency): stream the m7 prediction (`representations/predictions/surfaces/20250804142305-surface-…-m7-L0-th0.2.zarr`)
   only around the band of the survey leads (z ≈ 6700–7300 for 846B; z ≈ 9100–9800 for 1545). Trace on CPU
   (`vc_grow_seg_from_seed` with the published normal grids, or AndreasHad04's tracer). Keep a patch only if it passes
   on-sheet ≥ 0.9, no row jump > 8 voxels, and normals outward ≥ 0.9 (his gates). ESTIMATE: ~3 min CPU and ~0.5 GB of
   reads per cm² (#2534 numbers); 20 cm² ≈ 1 h and ~10 GB.
2. Render: 31 layers + 66 layers (to allow the 343-style d sweep). ~0.5 GB/cm² streamed, so ~10 GB.
3. Ink: the gated ensemble, 3 depths then the full sweep on the top windows. 20 cm² × 18 s × 5 models × 3 depths × 2
   directions ≈ **3 h**; the full sweep on the top 2 cm² is another ~3 h.
4. Ranking: row periodicity (fibre-aligned) + agreement across the ensemble + the controls in 8b.4. Hand review only
   after the automatic ranking, and log human minutes (the Grand Prize 8-hour rule is good practice here too).
5. If something passes: do not publish the image or text. Follow the First Letters submission rules (tifxyz mesh,
   1 cm scale bar, held-out validation, Docker image). Note: submitting is the user's decision, through the official
   form.

**Disk budget (ESTIMATE):** M6 ≈ 15 GB including caches; M8 dry run ≈ 30–40 GB; one target scroll ≈ 30 GB. Well
inside 1 TB.

---

## 6. Gaps and things not verified

- No code was run. All CPU times are estimates. The Hecate CPU speed is a guess; measure it first in M6 step 7.
- The PHerc0211 mesh is on Google Drive and was not inspected. Its render parameters are only "planes 00–30" (README).
- The checkpoint sizes on Hugging Face (ink_9um step-075000, Hecate, Reader V3, TRACE v3B) were not measured.
- Not verified that our VC3D `f7756c7` renders 66 layers bit-identical to `41701aa`. The 343 author says three
  builds agree; step 4 tests it.
- Crowding is measured only on GitHub. Discord (where most screening talk happens) was not read, by design. The 306B
  post's "13 other scrolls" with submissions are not named.
- The ketchupbyheinz survey has no clear licence (NOASSERTION). Use its lead coordinates as pointers only, and cite
  it.
- Whether the Grand Prize ban on "data derived from higher resolution scans of the submitted scroll volume" also
  applies to First Letters was not confirmed. Keep the 2.4 µm volumes of 846A and 1203 for diagnostics only.

## Sources

- Winner repos: [Nieuwlaar/pherc343-first-letters @ e8370f1](https://github.com/Nieuwlaar/pherc343-first-letters),
  [SirProdigle/pherc0306b-first-letters @ feabff6](https://github.com/SirProdigle/pherc0306b-first-letters),
  [priyankaranke/vesuvius-studio-multiscroll-cpt @ 235e3be](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt)
- [villa #2049](https://github.com/ScrollPrize/villa/issues/2049), [#2534](https://github.com/ScrollPrize/villa/issues/2534),
  [PR #2452](https://github.com/ScrollPrize/villa/pull/2452), [PR #2453](https://github.com/ScrollPrize/villa/pull/2453),
  [PR #2037](https://github.com/ScrollPrize/villa/pull/2037)
- [AndreasHad04/villa-apple-silicon](https://github.com/AndreasHad04/villa-apple-silicon) (`results/nieuwlaar/`, `firstletters/`)
- [rodriguescarson/eligible-scroll-atlas](https://github.com/rodriguescarson/eligible-scroll-atlas),
  [ketchupbyheinz/vesuvius-scroll-survey](https://github.com/ketchupbyheinz/vesuvius-scroll-survey),
  [Bullo27/first-letters-survey](https://github.com/Bullo27/first-letters-survey)
- Substack: [343](https://scrollprize.substack.com/p/50k-first-letters-prize-awarded-for),
  [306B](https://scrollprize.substack.com/p/writing-seen-in-pherc-306b-for-the),
  [211](https://scrollprize.substack.com/p/first-letters-emerge-from-pherc-211),
  [workflow](https://scrollprize.substack.com/p/from-ct-scan-to-ancient-text-a-first)
- [scrollprize.org/tutorial_spiral](https://scrollprize.org/tutorial_spiral), [prizes](https://scrollprize.org/prizes),
  villa `scrollprize.org/src/data/prizeEligibility.json`, `scrollprize.org/docs/15_winners.md`
- [metadata.json](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json); spiral tracks
  `https://dl.ash2txt.org/datasets/spiral_datasets/`
