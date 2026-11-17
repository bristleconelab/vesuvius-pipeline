# Human hours log

The 2027 Grand Prize allows at most 8 documented hours of human annotation / input in the unrolling pipeline.
Every manual decision that changes pipeline output is logged here, with date, minutes and what was decided.
Engineering work (writing code, reading docs) is not pipeline input and is not counted.

| Date | Minutes | Stage | Decision | Affects which outputs |
|---|---|---|---|---|
| 2026-11-16 | 0 | trace (M7) | seed point taken from the published w035 mesh centre (4194, 3837, 5443) | `results/m7-w035-seedB/` (a validation run, not a submission run) |
| 2026-11-16 | 0 | trace (M7) | seed C placed automatically on the m7 band at the w035 grid centre (4121, 3935, 5477): first m7-positive voxel along w035's normal | `results/m7-w035-seedC/` (validation run) |
| 2026-11-16 | 0 | trace (M7) | seeds D (4132, 2857, 4459) and E (3294, 4481, 6673) placed by the same automatic rule at 30 % / 70 % of w035's grid | `results/m7-w035-seedD/`, `seedE/` (validation runs) |
