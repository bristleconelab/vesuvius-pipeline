#!/usr/bin/env bash
# Score the Hecate 9.6 um ink model (scrollprize/hecate, as used in the PHerc 343 First Letters result) on held-out
# renders. Hecate only sees the labelled region (supervision-mask bounding box + margin) to keep CPU time down;
# its output is resized back onto the render grid and scored exactly like scripts/bakeoff.sh.
# Usage: scripts/bakeoff_hecate.sh DIRECTION RUN [RUN ...]     DIRECTION = forward | reverse
set -euo pipefail
DIR=$1; shift
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:cpu}" ROOT="${ROOT:-/data/vesuvius}"
CK="${CK:-$ROOT/checkpoints/pherc343}"                         # hecate.py + hecate_9.6um.pth, sha-checked
S="${S:-$ROOT/src/pherc343-first-letters/ink/scripts}"         # hecate_prep.py (MIT)
THREADS="${VPIPE_THREADS:-$(nproc)}"
REV=(); [ "$DIR" = reverse ] && REV=(--reverse)
NAME="hecate-$DIR"
cd "$(dirname "$0")/.."
export PATH=/root/.local/bin:$PATH
run() { "$ENGINE" run --rm --shm-size 8g -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" \
          -v "$ROOT:$ROOT:z" "$IMAGE" "$@"; }
py() { uv run -q --python 3.12 python -I "$@"; }

for r in "$@"; do
  RUN="$ROOT/runs/$r"; OUT="$RUN/bakeoff/$NAME"; mkdir -p "$OUT"
  SAMPLE=$(awk '/^== PHerc/{print $2; exit}' "$RUN/run.log"); SEG=$(awk '/^== PHerc/{print $3; exit}' "$RUN/run.log")
  V24ID=$(awk '/^== PHerc/{print $NF; exit}' "$RUN/run.log" | cut -d- -f1)
  UM=$(awk '/^== PHerc/{print $6; exit}' "$RUN/run.log" | cut -d- -f2); UM=${UM%um}
  RAW="$ROOT/raw/$SAMPLE/$SEG"
  LAB=$(find "$RAW/ink-labels" -maxdepth 3 -type d -name inklabels.zarr -path "*volume-${V24ID}*" | head -1); LAB=${LAB%/inklabels.zarr}
  SV24=$(find "$RAW/surface-volumes" -maxdepth 1 -type d -name "*volume-${V24ID}*.zarr" | head -1)
  if [ ! -f "$OUT/ink.tif" ]; then
    read -r Y0 X0 H W < <(py -m vpipe.crop bbox "$RUN/render.zarr" "$LAB/supervision.zarr" --margin 128)
    echo "$r: native ${UM} um, crop y0=$Y0 x0=$X0 h=$H w=$W" | tee "$OUT/crop.txt"
    T="$OUT/tmp"; rm -rf "$T"; mkdir -p "$T"
    t=$(date -u +%s)
    run bash -c "cd '$T' && python '$S/hecate_prep.py' --src '$RUN/render.zarr' --out '$T/prep.zarr' --native-um $UM \
        --native-um-z $UM --target-um 9.6 --z-center 13.5 --nz 16 --crop $Y0 $X0 $H $W \
      && TMPDIR='$T' python '$CK/hecate.py' --checkpoint '$CK/hecate_9.6um.pth' --input '$T/prep.zarr' --spacing-um 9.6 \
        --output '$T/out.png' --device cpu --batch-size 4 ${REV[*]}" > "$OUT/infer.log" 2>&1 \
      || { echo "$NAME $r INFER FAILED"; continue; }
    echo "infer_s $(( $(date -u +%s) - t ))" >> "$OUT/crop.txt"
    py -m vpipe.crop paste "$T/out.png" "$RUN/render.zarr" "$Y0" "$X0" "$H" "$W" "$OUT/ink.tif"
    rm -rf "$T"
  fi
  py -m vpipe.evaluate ink "$OUT/ink.tif" --labels "$LAB/inklabels.zarr" --mask "$LAB/supervision.zarr" \
    --align-ref "$SV24" --align-ref-level 2 --align-ours "$RUN/render.zarr" -o "$OUT/eval_ink.json" > /dev/null
  python3 -c "import json;m=json.load(open('$OUT/eval_ink.json'))['labels']['metrics_in_mask'];print('$NAME','$r','AUC %.4f AP %.4f F05 %.4f'%(m['roc_auc'],m['average_precision'],m['best_f05']))"
done
