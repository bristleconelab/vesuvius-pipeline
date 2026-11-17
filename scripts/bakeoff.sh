#!/usr/bin/env bash
# Score one ink checkpoint (or a mean ensemble) on already-rendered held-out runs, without re-rendering.
# Usage: [INFER_EXTRA='--layer-start 3 --layer-end 20'] scripts/bakeoff.sh NAME CKPT [ENSEMBLE_CKPT ...]
#   CKPT paths are relative to $ROOT/checkpoints. Output: $RUN/bakeoff/NAME/{ink.tif,eval_ink.json} per run.
set -euo pipefail
NAME=$1; CKPT=$2; shift 2
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:dev}" ROOT="${ROOT:-/data/vesuvius}"
RUNS="${RUNS:-heldout-w030 heldout-w045 heldout-0841-w00 heldout-0841-ag1 heldout-0841-ag2}"
ENS=()
for e in "$@"; do ENS+=(--ensemble-checkpoint "$ROOT/checkpoints/$e"); done
# torch/OpenMP threads per container: set VPIPE_THREADS=<cores/n> when n jobs share the machine,
# otherwise every container opens one thread per core and they thrash each other
THREADS="${VPIPE_THREADS:-$(nproc)}"
run() { "$ENGINE" run --rm --shm-size 8g -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -v "$ROOT:$ROOT:z" "$IMAGE" "$@"; }

for r in $RUNS; do
  RUN="$ROOT/runs/$r"; OUT="$RUN/bakeoff/$NAME"; mkdir -p "$OUT"
  # recover the segment's label and alignment inputs from the original run log
  SAMPLE=$(awk '/^== PHerc/{print $2; exit}' "$RUN/run.log"); SEG=$(awk '/^== PHerc/{print $3; exit}' "$RUN/run.log")
  V24ID=$(awk '/^== PHerc/{print $NF; exit}' "$RUN/run.log" | cut -d- -f1)
  RAW="$ROOT/raw/$SAMPLE/$SEG"
  LAB=$(find "$RAW/ink-labels" -maxdepth 3 -type d -name inklabels.zarr -path "*volume-${V24ID}*" | head -1); LAB=${LAB%/inklabels.zarr}
  SV24=$(find "$RAW/surface-volumes" -maxdepth 1 -type d -name "*volume-${V24ID}*.zarr" | head -1)
  t=$(date -u +%s)
  if [ ! -f "$OUT/ink.tif" ]; then
    rc=0
    run python -m vesuvius.ink_detection.inference.infer "$RUN/render.zarr" "$ROOT/checkpoints/$CKPT" \
      "$OUT/ink.tif.part.tif" "${ENS[@]}" --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu ${INFER_EXTRA:-} \
      > "$OUT/infer.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] || [ ! -f "$OUT/ink.tif.part.tif" ]; then echo "$NAME $r INFER FAILED rc=$rc"; continue; fi
    mv "$OUT/ink.tif.part.tif" "$OUT/ink.tif"
  fi
  t1=$(date -u +%s)
  run python -m vpipe.evaluate ink "$OUT/ink.tif" --labels "$LAB/inklabels.zarr" --mask "$LAB/supervision.zarr" \
    --align-ref "$SV24" --align-ref-level 2 --align-ours "$RUN/render.zarr" -o "$OUT/eval_ink.json" > /dev/null
  python3 -c "import json;m=json.load(open('$OUT/eval_ink.json'))['labels']['metrics_in_mask'];print('$NAME','$r','AUC %.4f AP %.4f F05 %.4f infer_s %d'%(m['roc_auc'],m['average_precision'],m['best_f05'],$t1-$t))"
done
