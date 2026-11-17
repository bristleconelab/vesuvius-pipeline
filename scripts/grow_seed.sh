#!/usr/bin/env bash
# Stage "trace": grow a papyrus surface (tifxyz) on CPU from one seed point, on the organisers' published
# m7 surface prediction for PHerc0139 (9.362 um volume), streaming prediction chunks and normal grids from S3.
# Usage: scripts/grow_seed.sh OUT_DIR X Y Z [GENERATIONS] [BOX: x_min x_max y_min y_max z_min z_max]
set -euo pipefail
OUT=$1 X=$2 Y=$3 Z=$4 GEN=${5:-160}
shift $(( $# >= 5 ? 5 : $# ))
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:dev}" ROOT="${ROOT:-/data/vesuvius}"
BASE="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/representations/predictions/surfaces"
PRED="$BASE/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
GRIDS_URL="$BASE/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.normal-grids"
GRIDS="${GRIDS:-$ROOT/cache/pherc0139-m7-normal-grids}"   # loader caches fetched slices here
mkdir -p "$OUT" "$GRIDS"
[ -f "$GRIDS/normal-grids-remote.json" ] || printf '{"url": "%s"}\n' "$GRIDS_URL" > "$GRIDS/normal-grids-remote.json"

BOX=""
if [ $# -eq 6 ]; then
  BOX=$(printf ', "x_min": %d, "x_max": %d, "y_min": %d, "y_max": %d, "z_min": %d, "z_max": %d' "$@")
fi
cat > "$OUT/params.json" <<EOF
{"mode": "seed", "generations": $GEN, "step_size": 20, "thread_limit": 1, "voxelsize": 9.362,
 "min_area_cm": 0.0, "normal_grid_path": "$GRIDS"$BOX}
EOF
"$ENGINE" run --rm -e HOME="$ROOT/vc3d-home-shared" -e VC_GROWPATCH_RNG_SEED=1 -v "$ROOT:$ROOT:z" "$IMAGE" \
  /usr/bin/time -v vc3d vc_grow_seg_from_seed -v "$PRED" -t "$OUT/surface" -p "$OUT/params.json" \
  --segment-name surface --rng-seed 1 -s "$X" "$Y" "$Z" > "$OUT/grow.log" 2>&1
grep -E "Elapsed|Maximum resident|Exit status" "$OUT/grow.log" || true
cat "$OUT/surface/meta.json" 2>/dev/null | head -30
