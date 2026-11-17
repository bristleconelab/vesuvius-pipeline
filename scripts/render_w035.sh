#!/usr/bin/env bash
# Stage "render": stream the PHerc0139 9.362 um volume from S3 and render the w035 surface volume
# (28 slices, flipped normals) exactly as the official ink tutorial does. Writes timing to the log.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/data/vesuvius/raw}"
WORK="${WORK:-/data/vesuvius/work/w035}"
SEG="$DATA_ROOT/PHerc0139/20260317000000-w035_2026031718"
VOL="s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/"
mkdir -p "$WORK"
export HOME="${VC3D_HOME:-/data/vesuvius/vc3d-home}"   # keeps ~/.VC3D/remote_cache on /data
mkdir -p "$HOME"
/usr/bin/time -v vc3d vc_render_tifxyz \
  --volume "$VOL" \
  --segmentation "$SEG/mesh/20260317000000-on-20250728140407-9.362um.tifxyz" \
  --zarr-output "$WORK/w035_9um.zarr" \
  --scale 1 --group-idx 0 --num-slices 28 \
  --cache-gb 16 \
  --voxel-size 9.362 --voxel-unit micrometer \
  --flip-normals
du -sh "$WORK/w035_9um.zarr" "$HOME/.VC3D/remote_cache" 2>/dev/null || true
