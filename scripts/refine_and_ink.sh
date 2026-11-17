#!/usr/bin/env bash
# Refine a traced surface onto the CT sheet maximum, score it against a reference mesh, then render + ink it.
# Usage: scripts/refine_and_ink.sh IN.tifxyz OUT_DIR REFERENCE.tifxyz
set -euo pipefail
IN=$1 OUT=$2 REF=$3
ROOT="${ROOT:-/data/vesuvius}"
CT_HTTP="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
CT_S3="s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/"
mkdir -p "$OUT" "$ROOT/cache/fsspec"
cd "$(dirname "$0")/.."
export PATH=/root/.local/bin:$PATH
t=$(date -u +%s)
uv run -q --python 3.12 python -I -m vpipe.refine "$IN" "$OUT/surface" \
  --volume "$CT_HTTP" --radius 5 --iterations 3 --grid-sigma 1.5 | tee "$OUT/refine.json"
echo "refine_s $(( $(date -u +%s) - t ))"
uv run -q --python 3.12 python -I -m vpipe.surface_compare "$OUT/surface" "$REF" -o "$OUT/vs_ref.json" > /dev/null
python3 -c "import json;r=json.load(open('$OUT/vs_ref.json'));print('vs ref: median_off %.2f within2 %.3f within6 %.3f'%(r['signed_normal_offset']['median'],r['precision_within']['2'],r['precision_within']['6']))"
[ "${SKIP_INK:-0}" = 1 ] || IMAGE="${IMAGE:-vpipe:cpu}" scripts/render_ink_surface.sh "$OUT/surface" "$CT_S3" 9.362 "$OUT/ink"
