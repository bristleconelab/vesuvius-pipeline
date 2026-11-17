# Wrap drift of CPU tracing on the published 9 um surface predictions

Question: when VC3D's GrowPatch tracer (`vc_grow_seg_from_seed`) runs on the organisers' published m7 surface prediction
and normal grids, does the traced surface stay on the papyrus sheet it was seeded on? Measured against two reference
meshes that are known to sit on written sheets. All numbers come from [`results/wrap-drift/`](../results/wrap-drift/)
and [`results/m7-w035-seed*/`](../results/).

## Cases

| Case | Scroll / volume | Seed (x, y, z) | Reference mesh |
|---|---|---|---|
| control343 | PHerc. 343, 8.640 um | (5911, 4059, 10523), a lead from a public survey next to the First Letters text | Nieuwlaar's First Letters mesh (`concat_w047-w048`) |
| w035-C | PHerc0139, 9.362 um | (4121, 3935, 5477), on the m7 band at w035's grid centre | published w035 |
| w035-D | PHerc0139, 9.362 um | (4132, 2857, 4459), same rule at 30 % of w035's grid | published w035 |
| w035-E | PHerc0139, 9.362 um | (3294, 4481, 6673), same rule at 70 % of w035's grid | published w035 |

Tracer settings: `step_size 20`, one thread, fixed RNG seed (runs are reproducible: the control patch re-grown on a
second machine has the same area, 1.04 cm2).

## Result: traced surfaces drift to neighbouring wraps

Distances along the reference normal, in voxels; "within 2 / 6" = share of traced vertices that close to the reference.

| Case | Area (cm2) | Median signed offset | Within 2 vx | Within 6 vx |
|---|---|---|---|---|
| control343 | 1.04 | -9.50 | 12.2 % | 38.8 % |
| w035-C | 14.44 | +0.07 | 13.4 % | 65.1 % |
| w035-D | 7.38 | -12.58 | 3.8 % | 20.0 % |
| w035-E | 7.19 | +9.35 | 4.9 % | 30.6 % |

Sheets are ~11-13 voxels apart in these regions (m7 profile along w035's normals: bands at -10, 0..+2 and +13 voxels),
so offsets of 9-13 voxels mean the neighbouring wrap. Even the best case (C, median offset 0.07) keeps only 13 % of its
vertices within 2 voxels of the written sheet, and none of the traced patches shows letters when inked: the w035 seeds B
and C with ink_9um seed42 step 75k in both depth directions, and the PHerc. 343 control with Hecate in five depth windows
plus the reverse direction.

## Two local fixes that do not work, and why

1. **CT-peak refinement** (`vpipe.refine`: move each vertex to the CT maximum within +-5 voxels along its normal).
   It leaves w035 itself in place (median shift 0.00 voxels) but cannot fix seed C: where C is off the sheet the mean
   CT profile has no peak (the surface sits in the gap), and the shifts found do not correlate with the shifts needed
   (r = -0.08).
2. **Wrap-consistent band snapping** (`vpipe.snap`: grow over the grid from the seed, giving each vertex the m7 band
   centre nearest to its neighbours' offset). The traced surfaces already sit on the m7 band (median |offset| 0.5 voxels
   for the w035 cases), so the drift happens *inside* the prediction: neighbouring sheets are connected in the
   thresholded 9 um m7 prediction, and following one band continuously walks across wraps.

| Case | Within 2 vx before -> after snapping | Within 6 vx before -> after |
|---|---|---|
| control343 | 12.2 % -> 0.0 % | 38.8 % -> 0.1 % |
| w035-C | 13.4 % -> 14.3 % | 65.1 % -> 66.9 % |
| w035-D | 3.8 % -> 0.8 % | 20.0 % -> 8.5 % |
| w035-E | 4.9 % -> 5.4 % | 30.6 % -> 31.5 % |

## What this implies

Keeping a traced surface on its wrap needs information the local 9 um prediction does not carry: a global winding
constraint (spiral fitting) or a sharper surface prediction. Local CT or band refinements are not enough. The four cases
above, with their reference meshes, can serve as a small regression benchmark for any tracer change
(`python -m vpipe.surface_compare GROWN REF --voxel-um ...`).

Not tested: the 2.4 um volumes, other thresholds of the m7 prediction, and the spiral-fitting route.
