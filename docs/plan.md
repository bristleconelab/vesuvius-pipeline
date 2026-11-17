# Plan

Goal: a pinned, headless, CPU-only chain that a judge can rerun with one command, scored against published
results at every stage; then use it to screen unread scrolls cheaply. Progress Prize deadline: 2026-11-30 23:59 PT.

## Milestones

| # | Milestone | Status (2026-11-16) |
|---|---|---|
| M1 | Ink inference on CPU, scored vs native 9 µm labels | done: AUC 0.959 / AP 0.927 / F0.5 0.881 (training segment) |
| M2 | Headless streamed render, scored vs published surface volume | done: 99.73 % identical, 100 % within 1 |
| M3 | One-command end-to-end runner in a pinned container | done: 20 min 29 s cold, byte-identical rerun |
| M4 | Explain the r 0.81 gap to the official tutorial image | open |
| M5 | Held-out check: segments NOT in the ink_9um training set (PHerc0139 w030, w045; PHerc0841 x3) | done: same scroll AUC 0.87-0.93, unseen scroll 0.71-0.77 |
| M6 | Reproduce a First Letters result (PHerc 343) end to end | done: render bit-identical; ink image r 0.99999, 18/18 letter boxes (CPU, Arm A) |
| M7 | Upstream: tifxyz from CT (surface model / tracer) on CPU, scored vs published mesh | partial: 17.5 cm2 grown in 30 min, on w035 footprint, median 11 vx deeper; ink test queued |
| M8 | Screening run on eligible unread scrolls, negatives published | first run: Hecate on survey leads (1545/846B/813) shows no letters, but the positive control also failed; cause measured = wrap drift of 9 um m7 tracing ([wrap-drift.md](wrap-drift.md)) |
| M9 | CPU-slim image (no CUDA wheels), CI that reruns M1-M3 nightly | Dockerfile.cpu + unit-test CI written; image build running |

## Not in scope (for now)

Spiral fitting (needs GPU + 96 GB RAM), training new ink models (GPU via Kaggle/Colab later if ever).

Long-range plan: [grand-prize-plan.md](grand-prize-plan.md). Manual decisions: [human-hours.md](human-hours.md).
