# Vesuvius Challenge: existing open-source software and past winning entries (state as of 2026-11-16)

Method note: repository stars / last-push dates / licences were read from the GitHub REST API on 2026-11-16 (the repo URL is given as the source). "Last push" = `pushed_at`. Site text was read from https://scrollprize.org/llms-full.txt (574 kB dump of the whole site) and https://scrollprize.org/winners. The PHerc. 1667 paper (arXiv:2606.29085, 46 pp.) was read in full-text form. No downloaded code was executed.

## 1. Tool-by-tool inventory: what each does, inputs/outputs, licence, hardware, status

### Takeaway
Everything "official" now lives in one MIT monorepo, `ScrollPrize/villa` (383 stars, pushed 2026-11-15, nightly VC3D builds); the 2023-era tools (ThaumatoAnakalyptor, the Grand Prize TimeSformer repo, Crackle-Viewer, the standalone `vesuvius` lib) are archived since Nov 2024 and survive only under `villa/deprecated` or `villa/vesuvius`. The current production chain is: nnU-Net/3D-U-Net surface prediction -> VC3D GrowPatch / Copy-Out-In / Lasagna tracing into `tifxyz` quad meshes -> (optional) diffeomorphic spiral fit -> flatboi (SLIM) flattening -> `vc_render_tifxyz` surface volumes -> ResNet3D-50+2D-UNet / hybrid 3D->2D ink models; it is explicitly "semi-automated" and human-checked at almost every stage.

### Cited Findings

**villa monorepo (umbrella)**
- `ScrollPrize/villa`: MIT, 383 stars, last push 2026-11-15, not archived; description "The Vesuvius Challenge monorepo" — [GitHub](https://github.com/ScrollPrize/villa)
- Top-level directories: `volume-cartographer` (VC3D + `vc_*` CLI), `vesuvius` (Python lib, models, neural tracing, ink training), `lasagna`, `spiral-fitting`, `ink-detection/optimized_inference`, `segmentation/vc_proofreader`, `dinovol` (3D DINOv2), `foundation` (scanning, scrollcase, photogrammetry, volume-registration, `scroll-unwrap-pipeline`, kaggle-visualizer, datasets), `gpu_manager`, `vc_object_store`, `vc_reader`, `scripts` (fiber_5class, fiber test runner), `scrollprize.org`, `deprecated` (crackle-viewer, thaumato-anakalyptor, ink-detection, vesuvius-c, discord_chatbot), plus `AGENTS.md`, `CLAUDE.md`, `SCROLL_ANATOMY.md` — [GitHub contents](https://github.com/ScrollPrize/villa)
- CI workflows present: `vc3d-ci.yml`, `vc3d-linux.yml`, `vc3d-macos.yml`, `vc3d-windows.yml`, `vc3d-codeql.yml`, `vc3d-publish-ghcr.yml`, `vc3d-publish-latest.yml`, `vc3d-promote-stable.yml`, `test-vesuvius.yml`, `test-gpu-manager.yml`, `deploy-vesuvius.yml`, `deploy-scroll-renderer.yml`, `build-crackle-viewer.yml`, plus progress-prize page automation — [villa/.github/workflows](https://github.com/ScrollPrize/villa/tree/main/.github/workflows)
- Releases are rolling: tags `latest` (published 2026-11-13) and `vc3d-archive-latest-*` (2026-11-11, 2026-11-13) — [villa releases](https://github.com/ScrollPrize/villa/releases)
- Other org repos: `vc-delta3d` (MIT, pushed 2026-11-13, CT compression), `dinovol` (archived 2026-08-21, moved into villa), `vc3d-deps` (prebuilt dependency containers, 2026-07-12), `vesuvius-gui` (MPL-2.0, 2026-07-02), `open-data` (2026-03-26), `scroll2zarr` (GPL-3.0, 2025-09-12); archived Nov 2024: `vesuvius`, `scrollrenderer`, `vesuvius-c`, `awesome-scroll-tools`, `scrollprize-website`; `ScrollPrize/volume-cartographer` archived 2025-09-05 — [ScrollPrize org](https://github.com/ScrollPrize)

**Volume Cartographer (original, EduceLab)**
- `educelab/volume-cartographer`: GPL-3.0, 75 stars, last push 2026-07-03; "Volumetric processing toolkit and C++ libraries for the recovery and restoration of damaged cultural materials" — [GitHub](https://github.com/educelab/volume-cartographer)
- Described by the Challenge as "the OG virtual unwrapping toolkit ... First built by EduceLab"; the community fork by Philip Allgaier (`spacegaier/volume-cartographer`, GPL-3.0, 9 stars, last push 2025-06-04) was the segmentation-team tool before VC3D; CLI tools named in the legacy tutorial: `vc_render`, `vc_layers_from_ppm`, `vc_area`, `vc_convert_pointset`, `vc_packager`; Docker image `ghcr.io/spacegaier/volume-cartographer:edge` — [scrollprize.org llms-full.txt](https://scrollprize.org/llms-full.txt)
- `hendrikschilling/volume-cartographer` (GPL-3.0, last push 2025-07-21) added OME-Zarr access, `vc_grow_seg_from_seed`, `vc_render_tifxyz`, `vc_tifxyz2obj`, `vc_grow_seg_from_segments` (patch consensus), `vc_tifxyz_winding` (winding-number diffusion), `vc_fill_quadmesh` (fusion/inpainting), low-memory tiled rendering — the origin of VC3D — [community projects page](https://scrollprize.org/community_projects)

**VC3D (villa/volume-cartographer)**
- VC3D is "the team's interactive tool for viewing, segmenting, and virtually unwrapping scroll volumes"; prebuilt installers for macOS, Windows, Linux AppImage (AppImage bundles VC3D + `vc_*` CLI); Docker `ghcr.io/scrollprize/villa/volume-cartographer:edge`; source build "presently requires a *nix like environment"; needs `ulimit -Sn 750000` on Ubuntu — [VC3D tutorial / llms-full.txt](https://scrollprize.org/tutorial_VC3D)
- Mesh format `tifxyz`: 2D grid of 3D vertices saved as three float32 TIFFs (x/y/z), invalid cells = -1 or a mask; enforces grid topology (single sheet); converters `vc_tifxyz2obj` / `vc_obj2tifxyz` — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Tracing: `vc_gen_normalgrids` builds normal grids from the CT; GrowPatch grows a quad mesh from a seed on a binary recto-surface prediction using Ceres nonlinear least squares, optimizing surface adherence, smoothness, spacing, orientation; "fails when the prediction topology does not match the real papyrus topology ... In practice, automatic growth still needs human inspection and correction. The current system is therefore best described as semi-automated." — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Copy Out/In (`vc_grow_seg_from_seed.cpp`, `GrowPatch.cpp`) transfers a traced wrap to the neighbouring wrap; a neural version (`infer_rowcol_triplet_wraps.py`, checkpoint `scrollprize/copy_displacement_latest`, trained on 4.8 um data) is exposed in a Neural Tracer panel; errors in the source mesh propagate — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Flattening: `flatboi` uses SLIM (Scalable Locally Injective Mappings), minimizing symmetric Dirichlet energy — [2026 open problems](https://scrollprize.org/2026_open_problems)
- VC3D streams CT directly from `s3://vesuvius-challenge-open-data/` (cloud-optimized OME-Zarr, AWS Open Data) — [2026 open problems](https://scrollprize.org/2026_open_problems)

**Lasagna (villa/lasagna)**
- PyTorch global optimizer for one or several stacked quad-mesh sheets; needs a U-Net-predicted dense normal volume and a "gradient magnitude" (winding density) volume; "more computationally intensive" than GrowPatch; also drives a semi-automatic fiber tracer (`atlas.py`); model card `scrollprize/lasagna-fiber` — [2026 open problems](https://scrollprize.org/2026_open_problems)

**Spiral fitting / global optimisation**
- Method: Paul Henderson, "Virtually Unrolling the Herculaneum Papyri by Diffeomorphic Spiral Fitting", WACV 2026, arXiv:2512.04927; fits a stationary-velocity-field diffeomorphism from an ideal Archimedean spiral to the scan; output is guaranteed a single continuous sheet — [arXiv](https://arxiv.org/abs/2512.04927)
- Original research repo `pmh47/spiral-fitting`: 12 stars, last push 2025-12-25, no licence file detected by GitHub; README requires "any modern CPU; 96GB RAM; 24GB VRAM (e.g. RTX 3090 / A5000)" — [GitHub](https://github.com/pmh47/spiral-fitting)
- Paper cost: "On a workstation with one Nvidia RTX 3090 GPU, our approach takes 19 hours for PHerc. Paris 4. This includes 16 hrs to postprocess the existing nnUNet outputs. Spiral-fitting takes around 3 hours for 20,000 iterations"; metrics defined: WJF (winding jump fraction), ChD, AD, Str; compared against ThaumatoAnakalyptor, "the only existing fully-automated approach that can tackle large portions of Herculaneum Papyri" — [arXiv HTML](https://arxiv.org/html/2512.04927v1)
- Production version `villa/spiral-fitting` (`fit_spiral.py`): needs Python >= 3.14 and an NVIDIA GPU; torch must be < 2.13 or VRAM use rises significantly; inputs = verified/unverified tifxyz patches, fiber/line annotations, winding constraints, predicted normals, gradient magnitude, skeletonised surface-prediction tracks (`extract_surface_tracks.py`); 30,000 optimisation steps by default; multi-GPU via `torchrun`; outputs meshes, checkpoint, "satisfaction metrics" — [spiral tutorial](https://scrollprize.org/tutorial_spiral)
- "To run the whole pipeline in one command — fit, then render ink, then score it — use `runners/run_single.py`"; `runners/run_sweep.py` runs a folder of configs across GPUs — [spiral tutorial](https://scrollprize.org/tutorial_spiral)
- Track extraction timings in the README: slices 10900-11300 of 8174x8174 predictions in ~4.1-4.2 s on one RTX 4090, ~4.4-4.6 s on 4x H100; slices 10752-14848 in 40.0 s on 1x H100; otherwise "falls back to a much slower CPU skeletonizer" — [spiral-fitting README](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/README.md)
- Limitation stated by the team: "where annotations are sparse or the volume is highly ambiguous, spiral fitting is still under-constrained, and won't necessarily follow the true sheet surfaces"; relative winding-number annotations have a large impact and are not automated — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Curated `spiral-input` dataset (2026-07): Paris 4 annotations with ~27,000 verified and ~204,000 unverified surface patches, ~49.6 GB — [curated datasets](https://scrollprize.org/data_datasets)
- Community: `spiral-fit-consumer-gpu` (Shuhan Yang, MIT, pushed 2026-08-02) runs the official fitter on a 12 GB consumer GPU; replaces point-to-patch linking where "a measured 41-hour ETA drops to minutes" — [GitHub](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu)
- Community evaluators: `spiralcheck` (Nicolas Dolegieviez, MIT, pushed 2026-08-09; CPU-only held-out scoring of spiral fits; found 54.8% leakage in a naive name-level split on Paris 4), `windaudit`, `windcon`, `winding-sync`, `spiral-winding-audit` (CPU only; 15-17% of consecutive snapped windings stacked on PHerc. 0175A/0800 fits) — [community projects page](https://scrollprize.org/community_projects)

**khartes**
- `KhartesViewer/khartes`: GPL-3.0, 20 stars, last push 2025-09-13 (about 14 months stale); manual segment-mesh creation with live preview of the rendered segment — [GitHub](https://github.com/KhartesViewer/khartes)
- README: no pre-conversion of TIFFs needed; data volumes are loaded into RAM, or streamed as needed in zarr mode; conversion of a TIFF stack "will take 8 to 24 hours" and ~1.2x the input size — [khartes README](https://github.com/KhartesViewer/khartes)
- Companion `KhartesViewer/scroll2zarr` (GPL-3.0, 5 stars, last push 2025-09-17) converts TIFF stacks to (OME-)Zarr — [GitHub](https://github.com/KhartesViewer/scroll2zarr)

**ThaumatoAnakalyptor (2023 Grand Prize autosegmentation)**
- `schillij95/ThaumatoAnakalyptor`: MIT, 86 stars, archived, last push 2024-11-21; copy under `villa/deprecated/thaumato-anakalyptor` — [GitHub](https://github.com/schillij95/ThaumatoAnakalyptor)
- Pipeline: gradient-based edge detection -> surface point cloud (`grid_to_pointcloud`) -> Mask3D instance segmentation (`pointcloud_to_instances`) -> graph solver for winding assignment (`graph_problem`, CPU or GPU) -> mesh -> GPU rendering (`large_mesh_to_surface`); operates on 8 um grid cells; Docker (`DockerfileThaumato`, `--shm-size=150g`) — [README](https://github.com/schillij95/ThaumatoAnakalyptor)
- Hardware: "RTX4090 or equivalent CUDA-enabled GPU with at least 24GB VRAM, 196GB RAM + 250GB swap and a multithreaded CPU with >= 32 threads is required ... Approximately twice the storage space of the initial scroll scan is required" — [README](https://github.com/schillij95/ThaumatoAnakalyptor)
- Sub-tools reused elsewhere: `sheet_to_mesh.py` fast GPU rendering (with a CPU-rendering commit), `mesh_merger.py`, `slim_uv.py` flattening — [community projects page](https://scrollprize.org/community_projects)
- In Henderson's Table 1 ThaumatoAnakalyptor scores ChD 5.59 / AD 0.1567 / Str 1.027 versus spiral fit (+SLIM) variants with lower AD — [arXiv HTML](https://arxiv.org/html/2512.04927v1)

**`vesuvius` Python library**
- Standalone `ScrollPrize/vesuvius`: MIT, 18 stars, archived 2024-11-21, "Now maintained at villa/tree/main/vesuvius" — [GitHub](https://github.com/ScrollPrize/vesuvius)
- Purpose: "Python library for accessing Vesuvius Challenge data ... without managing download scripts or storing terabytes of CT scans locally" — [community projects page](https://scrollprize.org/community_projects)
- Now also hosts model code: `vesuvius.tifxyz` reader, `CrossFrameZarrDataset`, `neural_tracing` (copy/extrapolation tracer, `autoreg_mesh`, `winding_models`), ink-detection training configs (`ink_detection/configs/aligned21_hybrid_3d2d.json`), `vesuvius.predict` — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Two neural tracing modes are "on hold": extrapolation-mode neural tracer and neural mesh autoregression ("a first MVP"); both suffer slow inference at 2.4 um and rollout drift — [2026 open problems, Appendix A](https://scrollprize.org/2026_open_problems)

**Surface-prediction models (autosegmentation front end)**
- Released checkpoints: `scrollprize/surface_recto_3dunet` (used for the paper) and `scrollprize/surface_m7_nnunet` (nnU-Net "m7" component of the Kaggle 1st-place solution) — [2026 open problems](https://scrollprize.org/2026_open_problems)
- `surface_m7_nnunet`: nnU-Net v2, `nnUNetResEncUNetLPlans`, 3d_fullres patch 192^3, 786 training volumes, single fold checkpoint ~783 MB, Apache-2.0 — [Hugging Face](https://huggingface.co/scrollprize/surface_m7_nnunet)
- Paper recto-surface 3D U-Net: nnU-Net-style residual-encoder U-Net with scSE attention, patch 256^3, 116,531 train patches from PHerc. 0139, 1667, 0343P, 0500P2, MAN Bp; Medial Surface Recall loss; final validation surface-class Dice 0.308 / IoU 0.189 (mean Dice 0.596) at epoch 3,864 — [arXiv:2606.29085 Suppl. Table 1](https://arxiv.org/abs/2606.29085)
- Legacy tutorial: nnUNetv2 inference needs an "Nvidia GPU with 10GB VRAM"; "Inference on this entire volume [Scroll 4 grids] takes around 13 hours on two RTX 3090s" — [tutorial / llms-full.txt](https://scrollprize.org/llms-full.txt)
- Community audit: 43.3% of positives in the published m7 predictions sit outside the masked CT ("phantoms"); 16.9% of stored prediction chunks are one-chunk halo — [Herculaneum Scroll Tools](https://github.com/axiosdevs/herculaneum-scroll-tools)
- Failure analysis: "compressed sheets are fused rather than missed across three architectures, the CT carries no intensity dip between contacting sheets" — [vesuvius-surface-geometry-diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic)
- HercUNet / HercuViT (Jaime Lomeli-Rodriguez, $20,000 Oct 2026; MIT; pushed 2026-11-02): iterative full-volume refiner with affinity head, clDice/MALIS losses, DAgger; "about 5 minutes per 192vx per-side cubes"; needs NVIDIA GPU — [GitHub](https://github.com/jimmylomro/hercUNet/tree/hercuvit)
- Yupeng Tang's Slurm scripts and quantitative comparison of three 9 um surface models (m7, Socratic, HercUNet) — [GitHub](https://github.com/yupengtang/vesuvius-scroll-inference/blob/main/eval/report/comparison.md)

**First Automated Segmentation Prize entries (Dec 2024, $60,000 split)**
- $30,000 to Hendrik Schilling & Sean Johnson (surface tracer) and $30,000 to Paul Henderson (spiral fitting); the $100k automated-unwrapping prize itself "was left unclaimed" — [winners](https://scrollprize.org/winners); [archived unwrapping page](https://scrollprize.org/unwrapping)
- `hendrikschilling/FASP` (7 stars, last push 2025-01-19, no licence detected): "created in approximately 28 hours of computing and 4 hours of human supervision" on "an AMD 5950x, Nvidia RTX3090 with 64GB of RAM" (+256 GB swap); half-resolution rendering was used to limit RAM — [GitHub](https://github.com/hendrikschilling/FASP)

**ScrollFiesta (Ben Kyles; $10k May, $20k Jul, $2.5k Aug, $5k Sep, $1k Oct 2026)**
- `Hob3rMallow/scrollfiesta_public`: 7 stars, last push 2026-10-31, licence not SPDX-recognised (NOASSERTION); "automatic mesher of surface predictions that attempts to fix topological mistakes"; takes a surface-probability volume as input "and does not produce or repair one"; default path runs on CPU with optional GPU MLS projection; tools `grid_pipeline`, `scroll_whole`, `scroll_unroll`, `ribbon_verdict`, `vcr_gpu` (renders villa's `vc_render_tifxyz` surface volume on an NVIDIA GPU "voxel for voxel"); has ctest suite — [GitHub](https://github.com/Hob3rMallow/scrollfiesta_public); [winners](https://scrollprize.org/winners)
- GPU ports: CUDA FP32 (~6x throughput, byte-identical), HIP/ROCm (~16.8x kernel), Rust/CubeCL portable kernel — [community projects page](https://scrollprize.org/community_projects)

**Patch-based unwrapping (William Stevens; $20,000 Aug 2026)**
- "Detects when the 2D alignments of overlapping patches imply incompatible 3D locations, drops the offending patches, and joins the rest into flattened components — around 365 cm² of coverage on PHerc. 1667" — [scrollprize.org](https://scrollprize.org/llms-full.txt)
- `WillStevens/scrollreading`: GPL-3.0, 3 stars, last push 2026-11-04; C + Python; flood-fill-derived autosegmentation (FAFF, DAFF) on 512^3 cubes, patch stitching (`pipeline9`), monthly PDF reports; README lists problems: tight layers not identified, no guarantee a surface is one layer, stitching can re-introduce intersections — [GitHub](https://github.com/WillStevens/scrollreading)

**Ink-detection validation harness (Hyojun Kwon; $1,000 Jul 2026 + $500 Sep 2026)**
- `khj1222/vesuvius-challenge`: MIT, 1 star, last push 2026-11-15; generates a `_validation_mask` holding out whole annotated regions, scores with threshold sweep, DRD and pseudo-F-measure, per-region breakdown, checkpoint sweep, k-fold; motivation: "The ink-detection tutorial trains with no held-out data, so improvements cannot be told apart from noise" — [community projects page](https://scrollprize.org/community_projects); [GitHub](https://github.com/khj1222/vesuvius-challenge)
- Its scorecard of all 14 `ink_9um` checkpoints: "the honest ceiling is F1 ~0.74-0.77", memorization gaps 0.22-0.45, noise floor ~0.03 (as quoted by inkbench) — [inkbench README](https://github.com/Alex55288/inkbench)

**2023 Grand Prize ink model (Nader, Farritor, Schilliger)**
- `younader/Vesuvius-Grandprize-Winner`: MIT, 625 stars, archived, last push 2024-11-25; final model TimeSformer-small with divided space-time attention, plus ResNet3D-101 and I3D variants; ~15 rounds of label expansion; Dockerfile with `--shm-size=150g`; weights on Google Drive; segmentation done with ThaumatoAnakalyptor — [GitHub](https://github.com/younader/Vesuvius-Grandprize-Winner)
- Prize: $700,000 of an $850,000 round on 2023-12-31; runners-up $50k each: SQ Mah; Lou Schlessinger & Arefeh Sherafati; Elian Dal Pra team — [winners](https://scrollprize.org/winners)
- Runner-up repos: [SQMah](https://github.com/SQMah/Vesuvius-Grand-Prize-Submission/), [lschlessinger1](https://github.com/lschlessinger1/vesuvius-grand-prize-submission), [erdpx](https://github.com/erdpx/vesuvius-grand-prize) — [community projects page](https://scrollprize.org/community_projects)
- Later optimisation: "1/5 the memory consumption and 20x the speed compared to the baseline GP ink detection for large segments" (Discord thread, Hendrik Schilling) — [community projects page](https://scrollprize.org/community_projects)

**First Letters models (Oct 2023)**
- Luke Farritor $40,000, Youssef Nader $10,000, Casey Handmer $10,000 (crackle discovery) on 2023-10-12 — [winners](https://scrollprize.org/winners)
- `younader/Vesuvius-First-Letters`: 138 stars, last push 2023-11-03, no licence detected; Farritor's `lukeboi/scroll-first-letters` is gone from GitHub (404) and only on web.archive.org — [GitHub](https://github.com/younader/Vesuvius-First-Letters); [community projects page](https://scrollprize.org/community_projects)

**Crackle-Viewer**
- `schillij95/Crackle-Viewer`: Apache-2.0, 11 stars, archived, last push 2024-11-25; "tool to browse and annotate surface volumes of rendered segments"; copy in `villa/deprecated/crackle-viewer` with a `build-crackle-viewer.yml` workflow — [GitHub](https://github.com/schillij95/Crackle-Viewer); [villa workflows](https://github.com/ScrollPrize/villa/tree/main/.github/workflows)

**Current ink models and tooling**
- PHerc. 1667 ink model: ResNet3D-50 backbone (Kinetics-700 init) feeding a 2D U-Net decoder; six iteration checkpoints `scrollprize/PHerc.1667-iteration-0..5` on Hugging Face — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Current 2.4 um model `scrollprize/ink_canonical_2um`; inference code in `villa/ink-detection/optimized_inference` — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Cross-scroll `scrollprize/ink_9um`: small 3D stem + 2D U-Net, two seeds x seven checkpoints, trained on aligned labels from PHerc. 0139, 1667, Paris 4, 0814 at ~9 um — [ink tutorial](https://scrollprize.org/tutorial5)
- Tutorial timings: 20,000 iterations at batch 8 "takes about ten hours on an H100"; tutorial-segment inference (~78,000 patches) "about two and a half hours on an H100"; full 78,125-iteration cross-scroll run "about five hours on one H100"; tutorial assumes Linux + NVIDIA GPU with CUDA — [ink tutorial](https://scrollprize.org/tutorial5)
- Hecate (`scrollprize/hecate`, MIT, "experimental"): distilled from `ink_canonical_2um`, predicts volumetric ink and a 2D map with learned depth attention; 2.4 um and 9.6 um checkpoints; no new manual labels — [Hugging Face](https://huggingface.co/scrollprize/hecate)
- Direct 3D ink segmentation works only where ink is visible: `scrollprize/ink_3d_dino_guided` "is trained specifically for PHerc. Paris. 4" — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Community ink models: reader-v2 ($5,000 Sep 2026) and reader-v3 ($5,000 Oct 2026; MIT; pushed 2026-10-20; claims ">10x less false ink than Hecate") by Domenico Russo; Rectina 9 um ($250) — [winners](https://scrollprize.org/winners); [reader-v3](https://github.com/DomRusso2/reader-v3)
- `LimeGS/ink-sdk` (MIT, pushed 2026-11-11): standalone inference for many ink models from a rendered surface volume or tifxyz + OME-Zarr; writes probability map, preview, receipt, "liveness verdict" — [GitHub](https://github.com/LimeGS/ink-sdk)
- First Title Prize ($60,000, May 2025): Marcel Roth & Micha Nowak, custom MiniUNETR 3D; `mvrcii/vesuvius_first_title_prize` 15 stars, last push 2025-11-23, no licence detected — [winners](https://scrollprize.org/winners); [GitHub](https://github.com/mvrcii/vesuvius_first_title_prize)
- Older 3D ink: `ryanchesler/3d-ink-detection` (Apache-2.0, last push 2024-11-25); `educelab/ink-id` (GPL-3.0, 67 stars, last push 2025-05-19) — [3d-ink-detection](https://github.com/ryanchesler/3d-ink-detection); [ink-id](https://github.com/educelab/ink-id)

**Automated fiber extraction and other 2026 prize tools**
- Samir Lerognon: $20,000 (Sep 2026) + $2,500 (Oct 2026) for automated fiber extraction (fine-tuned models + spline fitting), merged as villa PRs #1930 / #2309 — [winners](https://scrollprize.org/winners)
- QA tools: `windcheck` (self-intersections in tifxyz; MIT; pushed 2026-09-18), `tifxyz-doctor` (MIT; pushed 2026-08-24; benchmark on 709 human-reviewed Paris 4 `same_wrap` patches), `tifxyz-repair` (found 106 of 4,922 verified Paris 4 patches with stale bboxes), `scroll-data-audit`, `vesuvius-repro` (40 TTA-off m7 spot-checks match at Dice 0.9983-1.0000) — [community projects page](https://scrollprize.org/community_projects)
- UFSM "ultra fast scroll model" (Forrest McDonald, $1,000 Oct 2026): C23 + CUDA, no PyTorch, one binary covering ingest, 3D U-Net training/inference (winding fields, ink, papyrus), winding fitter, wrap extraction, flattening, rendering; requires CUDA 13.1+ and LLVM 22; no licence detected; 0 stars; pushed 2026-11-15; "coarse" stage "~3 h for Paris 4 on the desk CPU" — [GitHub](https://github.com/SuperOptimizer/ufsm)

### Inferences
- Archived-in-2024 tools (ThaumatoAnakalyptor, GP TimeSformer repo, Crackle-Viewer, First Letters repos) are historical references, not bases to build on; the team's own superseding pieces are VC3D tracing + spiral fit (geometry) and ResNet3D-50/hybrid 3D->2D models (ink).
- khartes (14 months without a push) and the EduceLab upstream (4 months) look dormant relative to villa's daily cadence.
- Many 2026 community prize repos have 0-7 stars and a single author; several key ones (ScrollFiesta, UFSM, FASP, pmh47/spiral-fitting, First Title repo) have no clear SPDX licence, which matters because the 2027 Grand Prize requires a permissive open-source licence.

### Gaps
- TimeSformer input specifics (number of layers, tile size), training hardware and runtime for the 2023 Grand Prize model: not confirmed (README summary did not state them).
- khartes GPU requirements and whether it reads tifxyz: not confirmed.
- Exact licence text of ScrollFiesta (GitHub reports NOASSERTION): not confirmed.
- "Socratic" surface model provenance: not confirmed.
- GitHub API rate limit prevented metadata for `DarthCeltic/argus`; README states Apache-2.0 per the community page only.

## 2. PHerc. 1667 end-to-end pipeline (first complete reading)

### Takeaway
PHerc. 1667 (Scroll 4) was read with a semi-automated pipeline: ESRF BM18 phase-contrast CT at 2.4 um -> 3D U-Net recto-surface prediction -> seed/consensus mesh growing and a wrap-by-wrap copy tool in VC3D -> heavy manual correction (~25 h per wrap x 31 wraps, i.e. roughly 775 annotator-hours by the paper's own figures) -> SLIM flattening and surface-conditioned rendering -> ResNet3D-50 + 2D U-Net ink model bootstrapped by five pseudo-labelling rounds -> transcription by eight papyrologists. Code, weights and data are public; the unwrapping was not automatic.

### Cited Findings
- Paper: "Complete virtual unwrapping and reading of a rolled Herculaneum papyrus", Angelotti et al. (27 authors), arXiv:2606.29085, submitted 27 June 2026 — [arXiv](https://arxiv.org/abs/2606.29085)
- "The reconstructed surface contains 22 columns or column-equivalents over approximately 860 cm² of preserved writing surface"; untranscribed material is restricted to the surviving parts of the first three columns (33 cm²) — [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- The website likewise says "the lower parts of some twenty-two columns, transcribed and reviewed by papyrologists"; the scroll's surviving core is about 8 cm of an original 19-24 cm height; text is a Stoic ethical treatise naming Aristocreon — [scrollprize.org/firstscroll](https://scrollprize.org/firstscroll)
- "The complete virtual unwrapping of PHerc. 1667 amounted to 31 wraps and 1231 cm² of papyrus surface. The unwrapping was completed using the most efficient semi-automated segmentation tooling available at the time, a wrap by wrap copy tool combined with ~25 hours per wrap of manual annotation." — [arXiv:2606.29085, Statistics and reproducibility](https://arxiv.org/abs/2606.29085)
- Automated steps: binary recto-surface prediction (3D U-Net); centreline traces per slice family giving orientation evidence; seed-based mesh growing; a consensus surface tracer over many overlapping base meshes ("in some cases approaching the scale of an entire scroll" but "some holes and deviations remained"); mesh-offset tool tracing rays along normals to the next predicted surface — [arXiv:2606.29085, Methods](https://arxiv.org/abs/2606.29085)
- Manual steps: "For meshes from all sources, manual correction was performed"; annotators add control points, move vertices, push/pull along normals, inspect overlays on orthogonal slices, and mark regions with an approval mask; small holes repaired by local geometric inpainting only where unambiguous — [arXiv:2606.29085, Methods](https://arxiv.org/abs/2606.29085)
- Ink: iterative pseudo-labelling; "To fully read PHerc. 1667, we applied the same process at scale, using three columns for iterative labeling and performing five iterations"; ablation saturates around iteration 5; Dice + BCE with strong label smoothing — [arXiv:2606.29085, Methods](https://arxiv.org/abs/2606.29085)
- Ink ablation compute: 1x NVIDIA H100 80 GB, "≈ 12 GB GPU, ≈ 160 GB host RAM", "≈ 2 h end-to-end (load + train + l_5 inference)"; patch 62x256x256; 33,061 train tiles; seed 130697; deterministic cudnn — [arXiv:2606.29085, Suppl. Table 2](https://arxiv.org/abs/2606.29085)
- Papyrology: transcription "produced and interpreted at column level by eight qualified papyrologists"; acceptance needed geometric tightness, letterform consistency across repeated renderings/inference passes, and independent endorsement or consensus — [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- Code availability: "Code sufficient to reproduce the main analyses, including mesh tracing, flattening, surface-conditioned rendering and ink-detection inference, is available at https://github.com/ScrollPrize/villa/commit/e583fb67468f483fadd73d52e06f0ab0fe5ba813"; weights on the `scrollprize` Hugging Face org; ablation checkpoints collection `pherc1667-ink-detection-ablation` — [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- Data: volumes, surfaces, renderings, transcriptions at scrollprize.org/data_browser under CC BY-NC 4.0; JP2 stacks at ESRF cultural-heritage database — [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- Scan regime: ~2.4 um isotropic voxels, 22 cm propagation distance, ~78 keV on ESRF BM18; a 1.1 um region of interest gave more legible text than 2.4 um — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Same paper: Paris 4 ink directly visible at 2.4 um and segmented in 3D (DINOv2-derived labels); PHerc. 139 title read as Philodemus, On Gods, Book 8 — [arXiv abstract](https://arxiv.org/abs/2606.29085)
- Stated remaining bottlenecks: geometric (mergers, holes, sheet switches in compressed regions) and radiometric (weak, recipe-dependent ink signal) — [arXiv:2606.29085, Discussion](https://arxiv.org/abs/2606.29085)

### Inferences
- 31 wraps x ~25 h = ~775 hours of manual annotation for one small scroll core; the 2027 Grand Prize allows at most 8 documented human hours — roughly a 100x reduction is what the prize is asking for.
- The released "code sufficient to reproduce" covers tracing, flattening, rendering and ink inference; the human correction itself is not reproducible by command, so the end-to-end result cannot be regenerated automatically from the raw volume.
- The assignment's premise of "20 columns" does not match the primary sources, which both say 22 columns/column-equivalents.

### Gaps
- Exact public announcement date (2026-06-25 per the assignment): not confirmed; the paper is dated 27 June 2026 and the 2027 Grand Prize deadline is 25 June 2027.
- Total wall-clock or GPU-hours for the full 1667 unwrapping + inference: not stated in the paper sections read.
- Number of annotators: not confirmed.

## 3. Kaggle competitions: Surface Detection (2025-26, $200k) and Ink Detection (2023, $100k)

### Takeaway
Surface Detection was won by long-trained nnU-Net ensembles plus topology-oriented post-processing; private-LB scores of the top 10 sit in a narrow 0.614-0.627 band on a combined TopoScore/SurfaceDice/VOI metric, and even the winners reported no solution for touching sheets. The 2023 ink competition (fragments only) was won by Ryan Chesler's team with 3D-encoder -> 2D SegFormer ensembles.

### Cited Findings
**Surface Detection**
- "$200,000 prize pool in November 2025"; metric "deliberately topology-aware rather than voxel-accuracy-based: it rewarded connected, gap-free predictions and penalized ... holes, mergers, and sheet switches"; winning recipe "take an off-the-shelf nnU-Net model configured for a couple of patch sizes, train it for a very long time (well over the default 1,000 epochs), and average their predictions" — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Prizes awarded March 2026: 1st "Vesuvius Team" $60,000; 2nd "risk of overfitting" $40,000; 3rd "W & A" $30,000; 4th "Starry" $20,000; 5th "Dieter" $15,000; 6th "#hui" $10,000; 7th "DECEM" $10,000; 8th "lingyundev" $5,000; 9th $5,000; 10th "Vibes & Scrolls Trade-off" $5,000; each with a Kaggle write-up link — [winners](https://scrollprize.org/winners)
- Secondary summary (Japanese SpeakerDeck by a participant, 2026-02-28; not the official leaderboard): metric = 0.30 x TopoScore (Betti matching) + 0.35 x SurfaceDice (2.0-voxel tolerance) + 0.35 x VOI score; 1,427 teams; private LB: 1st 0.627 (nnU-Net, 128->256 fine-tune, ~4000 epochs, 4-model weighted ensemble, height-map patch fill + 1-voxel hole plug), 2nd 0.622, 3rd 0.620 (nnU-Net cascade, 8000 epochs, Hessian ridge + EDT dilation), 4th 0.619 (7-stage ResEnc, 2000 epochs, PCA hole fill), 7th 0.618 (3d_lowres, Frangi + CED diffusion), 8th 0.616, 9th 0.616 (medial-surface recall), 10th 0.614 (ResEnc-L + Primus-B) — [SpeakerDeck](https://speakerdeck.com/sugupoko/vesuvius2-combined)
- Same summary: all top teams used nnU-Net v2; post-processing added roughly 0.02-0.035; the 1st-place team found no solution for touching sheets — [SpeakerDeck](https://speakerdeck.com/sugupoko/vesuvius2-combined)
- Released artefact: `scrollprize/surface_m7_nnunet` (Apache-2.0), "partial component" of the 1st-place solution — [Hugging Face](https://huggingface.co/scrollprize/surface_m7_nnunet)
- 5th place used a distance-transform representation instead of binary segmentation, flagged by the team as worth investigating — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Follow-ups: Benjamin Hamm's nnInteractive label refinement ($1,000 Aug 2026); ScrollSlabViewer 3D viewer for the competition ($1,000 Mar 2026); `villa/foundation/kaggle-visualizer` — [winners](https://scrollprize.org/winners)
- Oracle finding: "the official topometrics blend does not respond to repairing these merges" (fused sheets in compressed regions) — [vesuvius-surface-geometry-diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic)

**Ink Detection 2023**
- Awarded 2023-07-14, $100,000 total: 1st "ryches" $25k, 2nd "RTX23090" $20k, 3rd "wuyu" $15k; top-10 write-up links listed — [winners](https://scrollprize.org/winners); [community projects page](https://scrollprize.org/community_projects)
- 1st place repo `ainatersol/Vesuvius-InkDetection` (MIT, 15 stars, last push 2023-08-28): two-stage — 3D CNN / 3D U-Net / UNETR encoders flattened along depth, then 2D SegFormer (b1/b3/b5); ensemble of 9 models; AdamW, Dice+BCE; trained on three A6000 GPUs, ~10 h per model — [GitHub](https://github.com/ainatersol/Vesuvius-InkDetection)
- 7th place (MIC-DKFZ "OverthINKingSegmenter") published a resolution analysis follow-up — [community projects page](https://scrollprize.org/community_projects)
- Fourth-place Kaggle model was fine-tuned on scroll data by Luke Farritor (`lukeboi/scroll-fourth-second`) — [community projects page](https://scrollprize.org/community_projects)

### Inferences
- The Kaggle surface models are a solved commodity (drop-in nnU-Net weights exist); the unsolved part is topology in compressed regions, which neither the metric nor the winners addressed.
- Fragment-trained Kaggle 2023 models did not directly read scrolls; the scroll breakthroughs came from crackle labels + iterative pseudo-labelling.

### Gaps
- Official Kaggle leaderboard numbers and the text of the write-ups could not be read (Kaggle pages render client-side); the scores above come from a single secondary summary. Team-level code repositories for places 1-10: not confirmed.
- Competition start/end dates beyond "November 2025" launch and "March 2026" payout: not confirmed.
- 2023 ink competition metric and scores (commonly cited as F0.5) were not verified against a primary source in this session: not confirmed.

## 4. Key papers

### Takeaway
Four papers anchor the field in 2026: the complete-reading paper (arXiv:2606.29085), the Scientific Reports surface-topography ink paper (Dice ~0.88-0.90 from height maps alone at 0.34 um), Henderson's diffeomorphic spiral fitting (WACV 2026), and EduceLab-Scrolls (2023).

### Cited Findings
- Angelotti, Nicolardi, Henderson, Seales, "Ink Detection from Surface Topography of the Herculaneum Papyri", Scientific Reports 2026 (doi 10.1038/s41598-026-58467-1), preprint arXiv:2603.27698 — [arXiv](https://arxiv.org/abs/2603.27698); [Nature](https://www.nature.com/articles/s41598-026-58467-1)
- Method/result (search-snippet level): optical profilometry (Sensofar S lynx 2, confocal mode) of opened fragments PHerc. 248, 250, 500P2; nnU-Net segmentation on height maps; held-out Dice ~0.88-0.90 at native 0.34 um lateral sampling, degrading as lateral resolution is coarsened — [arXiv:2603.27698](https://arxiv.org/abs/2603.27698)
- Dataset `scrollprize/profilometer`: 14 co-registered profilometry regions with 16 letters; ships nnU-Net dataset descriptor, nine resolution-specific plans, five cross-validation and three leave-one-papyrus-out splits — [Hugging Face dataset](https://huggingface.co/datasets/scrollprize/profilometer)
- Team's reading: "surface shape alone carries usable information for telling inked regions from uninked ones", but "ink doesn't always show up as simple raised relief"; the 1667 paper says its results operate "in a mixed-signal regime, rather than as reading topography alone" — [2026 open problems, Appendix B](https://scrollprize.org/2026_open_problems); [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- Henderson, WACV 2026, arXiv:2512.04927; code https://github.com/pmh47/spiral-fitting; ground truth for Paris 4 covers ~30 windings created by "hundreds of hours of human input" — [arXiv HTML](https://arxiv.org/html/2512.04927v1)
- Parsons et al., "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT", arXiv:2304.02084 (2023) — [arXiv](https://arxiv.org/abs/2304.02084)
- Stephen Parsons' PhD dissertation "Hard-Hearted Scrolls" — [UKnowledge](https://uknowledge.uky.edu/cs_etds/138/)
- The 1667 paper cites a fully automatic unfolding approach by Kulagin et al. and Dambrogio et al. (automated virtual unfolding of sealed letters) as related work — [arXiv:2606.29085 references](https://arxiv.org/abs/2606.29085)

### Inferences
- The topography paper explains why resolution matters for ink (signal lives at sub-micron lateral scales), consistent with 1.1 um scans reading better than 2.4 um, and with weak results on ~9 um scans.

### Gaps
- Per-resolution Dice table and code-availability statement of the Scientific Reports paper: not confirmed (nature.com redirected to a login/cookie wall; only search snippets and the HF dataset card were read).
- Google Scholar was not queried; other 2025-26 papers (e.g. Kulagin et al.) were seen only as citations.

## 5. Is there a fully automated end-to-end pipeline? What runs without a GPU?

### Takeaway
No robust one-command "scroll volume -> readable ink image" pipeline exists; the team itself calls the workflow semi-automated and the 2027 $1M Grand Prize is precisely for making it automated (<= 8 human hours) and reproducible. Partial one-command runners exist (spiral `run_single.py`, `fls.py`, UFSM, vesuvius-automesh, FASP), and VC3D has real CI plus Docker/AppImage builds, but ML stages assume NVIDIA GPUs; CPU-only is practical for tracing, flattening, rendering, QA and scoring, not for surface/ink inference at scale.

### Cited Findings
- Team statement: "The pipeline works, but not without a person checking its output at almost every stage"; "No method yet traces a complete, correct surface through a scroll automatically" — [2026 open problems](https://scrollprize.org/2026_open_problems)
- 2027 Grand Prize (deadline 2027-06-25; $800k/$100k/$50k/$50k): requires a fully reproducible open-source pipeline integrated in VC3D, 100% of recto surface unrolled as tifxyz with isometric flattening, >= 70% of preserved characters legible per column, fixed seeds, public experiment tracking, and "The unrolling pipeline should be fully automated; up to 8 documented hours of human annotation / input are tolerated"; submissions are asked for a Docker image with system requirements — [prizes](https://scrollprize.org/prizes)
- Spiral fitting one-command runner: `runners/run_single.py` does "fit, then render ink, then score it"; still depends on pre-existing verified patches/annotations — [spiral tutorial](https://scrollprize.org/tutorial_spiral)
- `first-letters-survey` `fls.py`: one command from an eligible scroll to `ink_9um` maps on a freshly grown patch (seed on m7 prediction -> GrowPatch tracer -> render -> ink), needs a CUDA GPU; a 0.42 cm² test patch took 3 min 46 s; result over 65 patches / 971 cm² on 21 scrolls: no letters, and "Most automatic patches do not follow a single sheet for long near the compressed core" — [GitHub](https://github.com/Bullo27/first-letters-survey)
- `vesuvius-automesh` (MIT, pushed 2026-07-14): fully automated QC-gated harvest; "279.4 cm² of independently verified rendered papyrus surface" on Scroll 3 "with zero manual annotation and zero GPU"; hardware "Apple-silicon Mac (CPU only), ~30 GB chunk cache, anonymous S3"; 12 concurrent single-core traces, minutes per patch; 235 cm² of traces from one configuration were rejected wholesale by render-QC — [GitHub](https://github.com/spencerdavis-tx/vesuvius-automesh)
- FASP 2024: full pipeline in ~28 h compute + 4 h human on Ryzen 5950X + RTX 3090 + 64 GB RAM; ships `jammy.Dockerfile` — [GitHub](https://github.com/hendrikschilling/FASP)
- UFSM claims a single binary for the whole chain but requires CUDA 13.1+; has a tiered test suite (smoke tier ~60 tests under a minute) — [GitHub](https://github.com/SuperOptimizer/ufsm)
- Containers/CI: VC3D Docker image `ghcr.io/scrollprize/villa/volume-cartographer:edge`; `vc3d-deps` prebuilt containers; CI for Linux/macOS/Windows and `test-vesuvius.yml` — [VC3D docs](https://scrollprize.org/tutorial_VC3D); [villa workflows](https://github.com/ScrollPrize/villa/tree/main/.github/workflows)
- Reproducibility harnesses are community-built and partial: `vesuvius-repro` (regional m7 spot checks), ARGUS (evidence-receipt workbench; `argus run pherc0139-w016-ink9um-control`), `scroll-data-audit`, `vesuvius-autoresearch` — [community projects page](https://scrollprize.org/community_projects)
- CPU-only-capable pieces: GrowPatch uses the Ceres solver; ScrollFiesta "default path runs on CPU"; `vesuvius-render` is "fast self-contained CPU-based rendering of segments ... downloading data on-the-fly"; ThaumatoAnakalyptor has a CPU rendering commit; `extract_surface_tracks.py` has a "much slower CPU skeletonizer" fallback; `spiralcheck` and `spiral-winding-audit` are CPU only — [2026 open problems](https://scrollprize.org/2026_open_problems); [ScrollFiesta](https://github.com/Hob3rMallow/scrollfiesta_public); [community projects page](https://scrollprize.org/community_projects); [spiral tutorial](https://scrollprize.org/tutorial_spiral)
- GPU-bound pieces: spiral fit ("You'll need Python >= 3.14 and an NVIDIA GPU"; 24 GB VRAM / 96 GB RAM in the research repo, 12 GB with the consumer-GPU patches), ink training/inference tutorial (assumes NVIDIA GPU; H100 timings), nnU-Net surface inference (10 GB VRAM; 13 h on two RTX 3090s for a full old-resolution scroll) — [spiral tutorial](https://scrollprize.org/tutorial_spiral); [pmh47/spiral-fitting](https://github.com/pmh47/spiral-fitting); [ink tutorial](https://scrollprize.org/tutorial5)
- Low-end GPU evidence: inkbench reproduces all its numbers "on one 8 GB GPU (RTX 4060 Ti)"; Marcel Gwerder's lighter ink tutorial needs 2.4 GB of data instead of 92 GB — [inkbench](https://github.com/Alex55288/inkbench); [winners](https://scrollprize.org/winners)
- Data scale is itself a blocker: full-resolution volumes reach "dozens of terabytes"; halving voxel size is 8x data — [2026 open problems](https://scrollprize.org/2026_open_problems)

### Inferences
- On a CPU-only 16-core / 62 GB machine, realistic work is: streaming OME-Zarr crops, GrowPatch-style tracing on already-published surface predictions (vesuvius-automesh proves this at "minutes per patch"), flattening, CPU rendering, mesh QA, evaluation/benchmark harnesses and data audits. Surface-model and ink-model inference over anything larger than small patches, and spiral fitting, are not practical there; no CPU timing for ink inference was found.
- 62 GB RAM is below the stated needs of ThaumatoAnakalyptor (196 GB), the paper's ink ablation (160 GB host RAM) and the research spiral fitter (96 GB), but in line with FASP (64 GB + swap).
- The clearest tooling gaps (stages lacking robust, reproducible, automated tools): (a) automatic detection/repair of sheet switches and mergers in compressed regions; (b) automatic winding-number constraints for the spiral fit; (c) held-out, leakage-free evaluation suites for both spiral fits and ink models; (d) an integrated, containerised one-command runner with pinned seeds and provenance.

### Gaps
- No published CPU-only timing for nnU-Net surface inference or ink inference: "noma'lum" / not confirmed.
- Whether `vc3d-ci.yml` runs any end-to-end regression on real scroll data: not confirmed (workflow contents not read).
- Whether any Docker image bundles the whole chain (surface model + tracer + ink): none found.

## 6. Existing benchmarks and metrics

### Takeaway
Metrics exist per stage but there is no accepted end-to-end benchmark: topology-aware surface scores from Kaggle, geometric metrics from the spiral-fitting paper and villa's satisfaction/ink-coverage metrics, and several independent community ink benchmarks (F1 / AP / DRD / pseudo-F on held-out human labels). The Grand Prize itself is judged by legible characters per column, not a metric leaderboard.

### Cited Findings
- Surface prediction: Kaggle combined score = 0.30 TopoScore (Betti matching) + 0.35 SurfaceDice (tolerance 2.0 voxels) + 0.35 VOI score (secondary source) — [SpeakerDeck](https://speakerdeck.com/sugupoko/vesuvius2-combined)
- Surface model in the paper reports Dice/IoU on approximate labels (surface Dice 0.308) — labels are "approximate surface annotations rather than voxel-exact ground truth" — [arXiv:2606.29085](https://arxiv.org/abs/2606.29085)
- Surface-model comparison report for three 9 um models (m7, Socratic, HercUNet) — [yupengtang eval report](https://github.com/yupengtang/vesuvius-scroll-inference/blob/main/eval/report/comparison.md)
- `vesuvius-surface-geometry-diagnostic` ships a patch-mode eval entry point and a 200-patch ground-truth pool "so any surface model or splitter can be scored the same way" — [GitHub](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic)
- Unwrapping geometry: Henderson's WJF (winding jump fraction), ChD, AD, Str — [arXiv HTML](https://arxiv.org/html/2512.04927v1)
- Spiral fit: "satisfaction metrics — per-input-type statistics of how much of the evidence the final surface actually honors"; `get_ink_metrics.py` uses `scrollprize/ink-coverage-32um` (2D nnU-Net) to measure detected ink area, column coherence/width and line pitch — [spiral tutorial](https://scrollprize.org/tutorial_spiral)
- Team's call for help: "Devise better evaluation suites and loss functions to fit the spiral" — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Mesh QA: windcheck (self-intersection), tifxyz-doctor (overlap-isolated benchmark on 709 human-reviewed `same_wrap` patches), ScrollFiesta `ribbon_verdict` acceptance gates, vesuvius-automesh two-part gate (surface-lock score >= 0.9 for real pages vs 0.36-0.54 for an impostor; cov@50 um, mean surface distance ~22-24 um) — [community projects page](https://scrollprize.org/community_projects); [vesuvius-automesh](https://github.com/spencerdavis-tx/vesuvius-automesh)
- Ink: validation harness metrics = threshold-swept F1, DRD, pseudo-F-measure, per-region breakdown, k-fold — [khj1222/vesuvius-challenge](https://github.com/khj1222/vesuvius-challenge)
- Ink: inkbench (MIT, pushed 2026-09-07) — held-out human-label average precision: recommended `ink_9um` checkpoint 0.79, weight "soup" 0.85, average of all 14 checkpoints 0.87; late checkpoints "memorize" (training-pixel AP 0.86 vs held-out 0.76 for step-75k; step-20k 0.84 vs 0.83) — [inkbench](https://github.com/Alex55288/inkbench)
- Ink: Domenico Russo's benchmark of 14 9 um ink checkpoints ($1,000 Aug 2026); reader-v3 results use "same pixels, same masks, same metric for every model" on PHerc0841 and PHerc1447 — [winners](https://scrollprize.org/winners); [reader-v3](https://github.com/DomRusso2/reader-v3)
- Ink: team reports validation Dice computed on pseudo-labels (an agent swarm "nearly doubled the validation Dice score ... on PHerc. 1667 while training only on PHerc. 139 data") — [2026 open problems](https://scrollprize.org/2026_open_problems)
- Data-leak issues found by the community: 3D ink training leakage fix (Miika Kestila, Sep 2026); 54.8% leakage in naive spiral held-out split — [winners](https://scrollprize.org/winners); [community projects page](https://scrollprize.org/community_projects)
- Grand Prize judging: >= 70% of each counted column's preserved characters legible letter-by-letter; ties ranked by number of legible lines; no overlap between training and prediction regions — [prizes](https://scrollprize.org/prizes)
- Validation-harness author's note: "This is not a metric leaderboard" — [khj1222/vesuvius-challenge](https://github.com/khj1222/vesuvius-challenge)

### Inferences
- Ink "ground truth" in sealed scrolls is pseudo-labels or human labels derived from model output, so all ink metrics are near-training or circular to some degree; independent community benchmarks disagree with the recommended checkpoints, showing evaluation is immature.
- There is no automatic metric for "sheet-switch-free surface over a whole scroll" other than human approval masks; this is the largest evaluation gap and matches the team's own request.

### Gaps
- Numerical contents of the three-surface-model comparison and of Russo's 14-checkpoint benchmark were not read: not confirmed.
- Whether villa ships an official ink validation split after the harness prize: not confirmed.
