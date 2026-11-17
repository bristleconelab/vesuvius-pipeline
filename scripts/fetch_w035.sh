#!/usr/bin/env bash
# Fetch the PHerc0139 w035 reference set (~1.2 GB) from the anonymous open-data bucket.
# Inputs: DATA_ROOT (default /data/vesuvius/raw). Idempotent: aws s3 sync skips existing files.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/data/vesuvius/raw}"
SEG=20260317000000-w035_2026031718
SRC="s3://vesuvius-challenge-open-data/PHerc0139/segments/${SEG}"
DST="${DATA_ROOT}/PHerc0139/${SEG}"
mkdir -p "$DST"
sync() { aws s3 sync --no-sign-request --only-show-errors "$SRC/$1" "$DST/$1"; }

sync mesh/20260317000000-on-20250728140407-9.362um.tifxyz   # geometry on the 9.362 um volume (~1 MB)
sync mesh/20260317000000-on-20260102150214-2.399um.tifxyz   # geometry on the 2.4 um volume (for label/pred alignment)
sync surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr  # published 9 um render (~1 GB)
sync ink-detection                                          # published 2.4 / 1.1 um predictions
sync ink-labels                                             # published ink labels (2.4 um grid)
du -sh "$DST"
