#!/usr/bin/env bash
# M6: render the published PHerc 343 First Letters mesh exactly as its author did (66 layers, step 1, no
# --flip-normals) from the streamed 8.640 um volume, so the image can be compared with the published one.
# Mesh: github.com/Nieuwlaar/pherc343-first-letters @ e8370f1 (MIT), outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz
set -euo pipefail
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:cpu}" ROOT="${ROOT:-/data/vesuvius}"
REPO="${REPO:-$ROOT/src/pherc343-first-letters}"
MESH="$REPO/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz"
VOL="s3://vesuvius-challenge-open-data/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/"
OUT="${1:-$ROOT/runs/m6-pherc343}"
mkdir -p "$OUT"
(cd "$REPO/outputs" && sha256sum -c SHA256SUMS) > "$OUT/sha256-check.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/source-commit.txt"
t=$(date -u +%s)
"$ENGINE" run --rm -e HOME="$ROOT/vc3d-home-shared" -v "$ROOT:$ROOT:z" "$IMAGE" \
  /usr/bin/time -v vc3d vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$MESH" \
  --num-slices 66 --slice-step 1 --cache-gb 16 --voxel-size 8.64 --voxel-unit micrometer --zarr-output "$OUT/render.zarr"
printf '{"render_s":%d}\n' $(( $(date -u +%s) - t )) | tee "$OUT/timings.json"
