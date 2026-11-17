#!/usr/bin/env bash
# Render any tifxyz against a streamed volume and run the 9 um ink model on it (no labels needed).
# Usage: scripts/render_ink_surface.sh TIFXYZ VOLUME_URL VOXEL_UM OUT_DIR [WAIT_FOR_TMUX_SESSION]
set -euo pipefail
SURF=$1 VOL=$2 UM=$3 OUT=$4 WAIT=${5:-}
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:dev}" ROOT="${ROOT:-/data/vesuvius}"
CKPT="${CKPT:-ink_9um/hybrid_3d2d-seed42/step-075000.pth}"
if [ -n "$WAIT" ]; then while tmux has-session -t "$WAIT" 2>/dev/null; do sleep 60; done; fi
mkdir -p "$OUT"
# torch/OpenMP threads per container: set VPIPE_THREADS=<cores/n> when n jobs share the machine,
# otherwise every container opens one thread per core and they thrash each other
THREADS="${VPIPE_THREADS:-$(nproc)}"
run() { "$ENGINE" run --rm --shm-size 8g -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e HOME="$ROOT/vc3d-home-shared" -v "$ROOT:$ROOT:z" "$IMAGE" "$@"; }
t=$(date -u +%s)
run /usr/bin/time -v vc3d vc_render_tifxyz --volume "$VOL" --segmentation "$SURF" --zarr-output "$OUT/render.zarr" \
  --scale 1 --group-idx 0 --num-slices 28 --cache-gb 16 --voxel-size "$UM" --voxel-unit micrometer --flip-normals
t1=$(date -u +%s)
run /usr/bin/time -v python -m vesuvius.ink_detection.inference.infer "$OUT/render.zarr" "$ROOT/checkpoints/$CKPT" \
  "$OUT/ink.tif" --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu --direction both
t2=$(date -u +%s)
printf '{"render_s":%d,"infer_s":%d,"checkpoint":"%s"}\n' $((t1-t)) $((t2-t1)) "$CKPT" | tee "$OUT/timings.json"
