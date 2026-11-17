#!/usr/bin/env bash
# One segment end to end on CPU: tifxyz -> streamed ~9.4 um render -> ink -> metrics vs published products.
# Labels on the 2.4 um grid are resized to our grid and aligned from CT texture (vpipe.align), never from ink.
#
# Usage: scripts/run_segment.sh SAMPLE SEGMENT VOL9 VOL24 [RUN_DIR]
#   SAMPLE   e.g. PHerc0139           SEGMENT  e.g. 20260126000000-w045_2026012619
#   VOL9     9.4 um volume id, e.g. 20250728140407-9.362um-1.2m-113keV
#   VOL24    2.4 um volume id the labels live on, e.g. 20260102150214-2.399um-0.22m-78keV
set -euo pipefail
SAMPLE=$1 SEG=$2 VOL9=$3 VOL24=$4
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:dev}" ROOT="${ROOT:-/data/vesuvius}"
CKPT="${CKPT:-hybrid_3d2d-seed42/step-020000.pth}"
RUN="${5:-$ROOT/runs/${SEG}-$(date -u +%Y%m%dT%H%M%SZ)}"
B="s3://vesuvius-challenge-open-data/$SAMPLE"
RAW="$ROOT/raw/$SAMPLE/$SEG"
V9ID=${VOL9%%-*} V24ID=${VOL24%%-*}
UM9=$(echo "$VOL9" | cut -d- -f2)
VOL9_URL=$(aws s3 ls --no-sign-request "$B/volumes/" | awk '{print $2}' | grep "^${V9ID}-" | head -1)
# VC3D keeps streamed chunks under $HOME/.VC3D/remote_cache. A shared home lets segments on the same
# volume reuse chunks; set VC3D_HOME=$RUN/home for an isolated cold-cache timing run.
VC3D_HOME="${VC3D_HOME:-$ROOT/vc3d-home-shared}"
mkdir -p "$RUN" "$RAW" "$VC3D_HOME"
exec > >(tee -a "$RUN/run.log") 2>&1
s3() { aws s3 sync --no-sign-request --only-show-errors "$B/segments/$SEG/$1" "$RAW/$1"; }
# torch/OpenMP threads per container: set VPIPE_THREADS=<cores/n> when n jobs share the machine,
# otherwise every container opens one thread per core and they thrash each other
THREADS="${VPIPE_THREADS:-$(nproc)}"
run() { "$ENGINE" run --rm --shm-size 8g -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e HOME="$VC3D_HOME" -v "$ROOT:$ROOT:z" "$IMAGE" "$@"; }
now() { date -u +%s; }

echo "== $SAMPLE $SEG  render on $VOL9_URL  labels on $VOL24"
t=$(now)
MESHES=$(aws s3 ls --no-sign-request "$B/segments/$SEG/mesh/" | awk '{print $2}')
# prefer the plain "<seg>-on-<vol>-<um>.tifxyz"; "-t2" and other variants only as a fallback
MESH9=$(echo "$MESHES" | grep -E "on-${V9ID}-${UM9}\.tifxyz/$" | head -1)
[ -z "$MESH9" ] && MESH9=$(echo "$MESHES" | grep "on-${V9ID}-" | head -1)
MESH9=${MESH9%/}; echo "mesh: $MESH9"
s3 "mesh/$MESH9"
SV9=$(aws s3 ls --no-sign-request "$B/segments/$SEG/surface-volumes/" | awk '{print $2}' | grep "volume-${V9ID}" | head -1); SV9=${SV9%/}
SV24=$(aws s3 ls --no-sign-request "$B/segments/$SEG/surface-volumes/" | awk '{print $2}' | grep "volume-${V24ID}" | head -1); SV24=${SV24%/}
[ -n "$SV9" ] && s3 "surface-volumes/$SV9"
for f in .zattrs .zgroup zarr.json; do aws s3 cp --no-sign-request --only-show-errors "$B/segments/$SEG/surface-volumes/$SV24/$f" "$RAW/surface-volumes/$SV24/$f" 2>/dev/null || true; done
s3 "surface-volumes/$SV24/2"
LABDIR=$(aws s3 ls --no-sign-request --recursive "$B/segments/$SEG/ink-labels/" | awk '{print $4}' | grep "volume-${V24ID}/.*/inklabels.zarr/" | head -1 | sed 's|/inklabels.zarr/.*||; s|.*/segments/[^/]*/||')
s3 "$LABDIR"
T_FETCH=$(( $(now) - t ))

if [ ! -f "$ROOT/checkpoints/ink_9um/$CKPT" ]; then
  uvx --from huggingface_hub hf download scrollprize/ink_9um "$CKPT" --local-dir "$ROOT/checkpoints/ink_9um"
fi

t=$(now)
run /usr/bin/time -v vc3d vc_render_tifxyz --volume "$B/volumes/$VOL9_URL" \
  --segmentation "$RAW/mesh/$MESH9" --zarr-output "$RUN/render.zarr" \
  --scale 1 --group-idx 0 --num-slices 28 --cache-gb 16 \
  --voxel-size "${UM9%um}" --voxel-unit micrometer --flip-normals
T_RENDER=$(( $(now) - t ))

t=$(now)
run /usr/bin/time -v python -m vesuvius.ink_detection.inference.infer "$RUN/render.zarr" \
  "$ROOT/checkpoints/ink_9um/$CKPT" "$RUN/ink.tif" --overlap 0.5 --blend-mode hann --batch-size 8 --device cpu
T_INFER=$(( $(now) - t ))

t=$(now)
[ -n "$SV9" ] && run python -m vpipe.evaluate render "$RUN/render.zarr" "$RAW/surface-volumes/$SV9" -o "$RUN/eval_render.json" >/dev/null
run python -m vpipe.evaluate ink "$RUN/ink.tif" --labels "$RAW/$LABDIR/inklabels.zarr" --mask "$RAW/$LABDIR/supervision.zarr" \
  --align-ref "$RAW/surface-volumes/$SV24" --align-ref-level 2 --align-ours "$RUN/render.zarr" -o "$RUN/eval_ink.json" >/dev/null
T_EVAL=$(( $(now) - t ))

du -sb "$VC3D_HOME/.VC3D/remote_cache" 2>/dev/null | awk -v h="$VC3D_HOME" '{print "remote_cache_bytes", $1, h}' > "$RUN/streamed.txt" || true
printf '{"sample":"%s","segment":"%s","checkpoint":"%s","fetch_s":%d,"render_s":%d,"infer_s":%d,"evaluate_s":%d}\n' \
  "$SAMPLE" "$SEG" "$CKPT" "$T_FETCH" "$T_RENDER" "$T_INFER" "$T_EVAL" | tee "$RUN/timings.json"
echo "== done: $RUN"
