#!/usr/bin/env bash
# One command, end to end, CPU only: PHerc0139 w035 tifxyz -> streamed 9.362 um render -> ink -> metrics.
# Host needs: podman (or docker via ENGINE=docker), aws CLI (anonymous S3), the vpipe image.
# Usage: scripts/run_w035.sh [RUN_DIR]     (RUN_DIR default /data/vesuvius/runs/w035-<timestamp>)
set -euo pipefail
ENGINE="${ENGINE:-podman}"
IMAGE="${IMAGE:-vpipe:cpu}"   # README: podman build -t vpipe:cpu -f docker/Dockerfile.cpu .
ROOT="${ROOT:-/data/vesuvius}"
RUN="${1:-$ROOT/runs/w035-$(date -u +%Y%m%dT%H%M%SZ)}"
# step-020000 is the checkpoint of the w035 image shown in Tutorial 5 ("both at checkpoint step-020000") and of the
# README results; the tutorial's command line uses step-075000 (CKPT=hybrid_3d2d-seed42/step-075000.pth)
CKPT="${CKPT:-hybrid_3d2d-seed42/step-020000.pth}"
SEG=20260317000000-w035_2026031718
RAW="$ROOT/raw/PHerc0139/$SEG"
LAB="$RAW/ink-labels/9.362um-volume-20250728140407/20260918"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
stamp() { date -u +%s; }
declare -A T

# torch/OpenMP threads per container: set VPIPE_THREADS=<cores/n> when n jobs share the machine,
# otherwise every container opens one thread per core and they thrash each other
THREADS="${VPIPE_THREADS:-$(nproc)}"
run() { "$ENGINE" run --rm --shm-size 8g -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e HOME="$RUN/home" -v "$ROOT:$ROOT:z" "$IMAGE" "$@"; }

echo "== vpipe w035 run: $RUN"
"$ENGINE" image inspect "$IMAGE" --format 'image {{.Id}}' | tee "$RUN/image.txt"
run cat /opt/villa.REF | sed 's/^/villa /' | tee -a "$RUN/image.txt"
run cat /opt/vc3d/vc3d.AppImage.sha256 | sed 's/^/vc3d /' | tee -a "$RUN/image.txt"

t=$(stamp); DATA_ROOT="$ROOT/raw" "$(dirname "$0")/fetch_w035.sh"; T[fetch]=$(( $(stamp) - t ))

if [ ! -f "$ROOT/checkpoints/ink_9um/$CKPT" ]; then
  uvx --from huggingface_hub hf download scrollprize/ink_9um "$CKPT" --local-dir "$ROOT/checkpoints/ink_9um"
fi
sha256sum "$ROOT/checkpoints/ink_9um/$CKPT" | tee "$RUN/checkpoint.sha256"

t=$(stamp)
mkdir -p "$RUN/home"
run /usr/bin/time -v vc3d vc_render_tifxyz \
  --volume s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/ \
  --segmentation "$RAW/mesh/20260317000000-on-20250728140407-9.362um.tifxyz" \
  --zarr-output "$RUN/w035_9um.zarr" \
  --scale 1 --group-idx 0 --num-slices 28 --cache-gb 16 \
  --voxel-size 9.362 --voxel-unit micrometer --flip-normals
T[render]=$(( $(stamp) - t ))

t=$(stamp)
run /usr/bin/time -v python -m vesuvius.ink_detection.inference.infer \
  "$RUN/w035_9um.zarr" "$ROOT/checkpoints/ink_9um/$CKPT" "$RUN/w035_ink.tif" \
  --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu
T[infer]=$(( $(stamp) - t ))

t=$(stamp)
run python -m vpipe.evaluate render "$RUN/w035_9um.zarr" \
  "$RAW/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr" -o "$RUN/eval_render.json" >/dev/null
run python -m vpipe.evaluate ink "$RUN/w035_ink.tif" --labels "$LAB/inklabels.zarr" --mask "$LAB/supervision.zarr" \
  -o "$RUN/eval_ink.json" >/dev/null
T[evaluate]=$(( $(stamp) - t ))

du -sb "$RUN/home/.VC3D/remote_cache" 2>/dev/null | awk '{print "remote_cache_bytes", $1}' | tee "$RUN/streamed.txt" || true
printf '{"fetch_s":%d,"render_s":%d,"infer_s":%d,"evaluate_s":%d}\n' \
  "${T[fetch]}" "${T[render]}" "${T[infer]}" "${T[evaluate]}" | tee "$RUN/timings.json"
echo "== done: $RUN"
