# Grand Prize plan (2027)

Target: the 2027 Grand Prize ($800k / $100k / $50k / $50k, deadline 2027-06-25 23:59 PT): one eligible scroll,
100 % of the recto surface unrolled, >= 70 % of characters legible per counted column, a fully automated unrolling
pipeline with <= 8 documented human hours, Docker image, fixed seeds, public tracking, VC3D integration
(`docs/prize-rules.md`). PHerc. 1667 needed ~775 manual hours (`docs/prior-work.md`), so the whole game is removing
human checking from segmentation while keeping ink detection good on an unseen scroll.

## What we have measured (2026-11-16, CPU only)

| Stage | Status | Evidence |
|---|---|---|
| Stream CT, render a surface volume | reproduces published renders: 99.6-99.75 % voxels identical, 100 % within 1 grey level, 6 segments | `results/*/eval_render.json` |
| Ink inference on CPU | full segment in 4-11 min; deterministic (byte-identical rerun) | `results/w035-e2e-1/` |
| Ink quality, same scroll, held out | AUC 0.87-0.94 (seed42 step 20k/75k) | `results/heldout-w0*/`, bake-off |
| Ink quality, unseen scroll | AUC 0.71-0.77 with one checkpoint; winners' ensembles report ~0.87 on PHerc0841 | `results/heldout-0841-*`, `docs/research/first-letters.md` |
| Surface tracing from CT prediction | 17.5 cm2 grown in 30 min on 1 CPU thread from one seed; follows w035's footprint (94 % of vertices) but sits a median 11 voxels deeper; ink test of the grown surface pending | `results/m7-w035-seedB/` |
| Label alignment, scoring | model-free texture alignment validated (w035: (-8,-8) vs (-8,-7) from labels) | `src/vpipe/align.py` |

## Where the work is

1. **Segmentation without a human (the gap).** Tracing is cheap on CPU; trust is not. We need automatic error
   detection: sheet switches, wrong-wrap jumps, folds, and the depth offset seen in M7. Every grown patch must carry
   numbers that say whether it can be kept, so humans are not needed to look at each stage.
2. **Ink on an unseen scroll.** A single public checkpoint loses ~0.15-0.2 AUC across scrolls. Ensembles, depth
   sweeps and model choice per scroll are the levers; every choice is pre-registered and scored on held-out scrolls.
3. **Scale and cost.** A full scroll is 10-100x the area of one segment. Render is network-bound (uncompressed
   128^3 chunks, ~0.5 GB per cm2); inference is CPU-bound. GPU will be needed for surface prediction on scrolls that
   have no published prediction, and for any ink fine-tuning (Kaggle/Colab first; a costed cloud option if it blocks).

## Milestones

| When | Milestone | Done when |
|---|---|---|
| 2026-11 | Reproducible CPU chain public + November Progress Prize submission | repo public, README numbers generated from `results/`, CI green |
| 2026-11/12 | Model bake-off on held-out scrolls; ensemble gate AUC >= 0.85 on PHerc0841 | `results/bakeoff/` table |
| 2026-12 | M7 closed: grown surface scored by geometry **and** by ink; automatic offset/sheet-switch detector | ink image of the grown surface shows letters; detector flags known bad patches |
| 2026-12 | M6: PHerc 343 First Letters image reproduced on our chain | r vs published image and letter-box agreement reported |
| 2027-01 | Screening run on an eligible scroll (PHerc0846B, then PHerc1545), negatives published | per-patch ink statistics + images |
| 2027-01..04 | Multi-seed automatic tracing over a full scroll region with automatic QA gates; patch merging | coverage map, QA metrics, hours of human input logged |
| 2027-04..06 | Full eligible scroll, Docker + seeds + tracking; submission | prize criteria checklist in `docs/prize-rules.md` all met |

## Rules we keep

- Reproduce before claiming; every number from a file in `results/`; "not tested" where true.
- Human hours are logged from day one (`docs/human-hours.md`), because the prize caps them at 8.
- Open source (MIT) code; data products CC BY-NC 4.0; no revealed text published before the organisers allow it.
