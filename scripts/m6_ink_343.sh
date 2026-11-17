#!/usr/bin/env bash
# M6 Arm A: the ink step of the PHerc 343 First Letters image, on CPU, at the published depth d only, with the
# author's statistics taken from his sidecar (no 31-depth sweep). Runs the author's own scripts unchanged
# (github.com/Nieuwlaar/pherc343-first-letters @ e8370f1, MIT) inside our villa image:
#   1. mesh/build_concat.py   -> node source map for merge_tile_phase.py (and the G1 mesh gate: sha256 vs SHA256SUMS)
#   2. pad_render.py          -> zero-padded copy of our render (40 rows, 200 columns) for the second tile phase
#   3. run_ink.sh's commands  -> 4 models x 2 tilings at depth D, reading side only (villa: layers 25+D..41+D,
#                                --direction reverse; Hecate: hecate_prep.py 16 planes at 9.6 um centred on layer
#                                32.5+D, then hecate.py --reverse), CPU
#   4. merge_tile_phase.py    -> w047 pixels from the padded run, the rest from the plain run
#   5. make_submission_image.py with an ink-stats file built from the sidecar (per-model mean/sd/p1/p99.5,
#      ensemble p1/p99.5, CT stretch), the author's command line otherwise
# Usage: scripts/m6_ink_343.sh [RUN_DIR]   (RUN_DIR holds render.zarr from scripts/m6_render_343.sh)
# Resumable: maps that already exist are kept.
set -euo pipefail
ENGINE="${ENGINE:-podman}" IMAGE="${IMAGE:-vpipe:cpu}" ROOT="${ROOT:-/data/vesuvius}"
REPO="${REPO:-$ROOT/src/pherc343-first-letters}"
CK="${CKPT_DIR:-$ROOT/checkpoints/pherc343}"
RUN="${1:-$ROOT/runs/m6-pherc343}"
D="${D:-1}"
MODELS="${MODELS:-hybrid_3d2d-seed42 hybrid_3d2d-seed43 dnative hecate}"
N=concat_w047-w048_R5B2_z9500-11000
S="$REPO/ink/scripts"
W="$RUN/ink"
mkdir -p "$W/tmp" "$W/logs"
TT="$W/logs/timings.tsv"
run() {
  "$ENGINE" run --rm -i --shm-size 8g -e CKPT_DIR="$CK" -e PY=python -e HEC_DEVICE=cpu -e TMPDIR="$W/tmp" \
    -v "$ROOT:$ROOT:z" "$IMAGE" "$@"
}
stamp() { date -u +%s; }
log() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$TT"; }

# 1. node source map (+ mesh gate)
if [[ ! -s $W/mesh/${N}_node_source.npz ]]; then
  t=$(stamp)
  run python -I "$REPO/mesh/build_concat.py" --w047 "$REPO/outputs/w047_R5B2_z9500-11000.tifxyz" \
    --w048 "$REPO/outputs/w048_R5B2_z9500-11000.tifxyz" --letter-boxes "$REPO/inputs/${N}_letter_boxes.json" \
    --out-dir "$W/mesh"
  log build_concat - $(( $(stamp) - t ))
fi
(cd "$W/mesh" && grep -E "  $N\.tifxyz/[xyz]\.tif\$" "$REPO/outputs/SHA256SUMS" | sha256sum -c -) > "$W/mesh_gate.txt" 2>&1 || echo "mesh gate FAILED, see $W/mesh_gate.txt"

# 2. level-0-only view of the render (as the author's maps were computed) and the padded copy
P="$W/${N}_66.zarr"
mkdir -p "$P"; echo '{"zarr_format": 2}' > "$P/.zgroup"; ln -sfn "$RUN/render.zarr/0" "$P/0"
Q="$W/${N}_66_padded.zarr"
if [[ ! -f $Q/0/.zarray ]]; then
  t=$(stamp)
  run python -I "$S/pad_render.py" "$P" "$Q" --pad-rows 40 --pad-cols 200
  log pad_render - $(( $(stamp) - t ))
fi

# 3. ink maps, READING side only, one container per (tiling, model), sequential (16 threads each).
# These are run_ink.sh's own commands, minus the blind side, with its output names:
#  - villa: run_ink.sh passes --direction both, which villa's infer.py main() runs as two independent passes of
#    infer_single_zarr() with the same model ("forward", then "reverse"); --direction reverse is that second pass.
#  - Hecate: run_ink.sh calls hecate.py twice (without and with --reverse); only the --reverse call is made here.
ZC=$(awk -v d="$D" 'BEGIN{printf "%.2f", 32.5 + d}')
SL=$(( 25 + D )); EL=$(( SL + 17 ))
for Z in "$P" "$Q"; do
  tag=plain; [[ $Z == "$Q" ]] && tag=padded
  name=$(basename "$Z" .zarr)
  O="$W/maps_$tag"; mkdir -p "$O"
  for m in $MODELS; do
    case $m in
      dnative) ck="$CK/dense_native-016000.rebuilt.pth" ;;
      hecate) ck="$CK/hecate_9.6um.pth" ;;
      *) ck="$CK/${m}_step-075000.pth" ;;
    esac
    if [[ $m == hecate ]]; then f="$O/${name}__hecate__L66zc${ZC}_reverse.png"; else f="$O/${name}__${m}__L66s${SL}_reverse.tif"; fi
    [[ -s $f ]] && continue
    T=$(mktemp -d "$W/tmp/ink.XXXXXX")
    t=$(stamp)
    if [[ $m == hecate ]]; then
      run bash -c "cd '$T' && python '$S/hecate_prep.py' --src '$Z' --out '$T/prep.zarr' --native-um 8.64 \
          --native-um-z 8.64 --target-um 9.6 --z-center $ZC --nz 16 \
        && TMPDIR='$T' python '$CK/hecate.py' --checkpoint '$ck' --input '$T/prep.zarr' --spacing-um 9.6 \
          --output '$T/out_reverse.png' --device cpu --batch-size 4 --reverse" > "$W/logs/${tag}_${m}_d${D}.log" 2>&1
      mv -f "$T/out_reverse.png" "$f"; mv -f "$T/out_reverse.json" "${f%.png}.json"
    else
      run bash -c "cd '$T' && TMPDIR='$T' python -m vesuvius.ink_detection.inference.infer '$Z' '$ck' '$T/out.tif' \
          --overlap 0.5 --blend-mode hann --batch-size 4 --direction reverse --device cpu \
          --layer-start $SL --layer-end $EL" > "$W/logs/${tag}_${m}_d${D}.log" 2>&1
      mv -f "$T/out.tif" "$f"
    fi
    log "$tag" "$m" $(( $(stamp) - t ))
    rm -rf "$T"
  done
done

# 4. merge the two tile phases
t=$(stamp)
run python -I "$S/merge_tile_phase.py" --plain "$W/maps_plain" --plain-name "${N}_66" \
  --padded "$W/maps_padded" --padded-name "${N}_66_padded" \
  --node-source "$W/mesh/${N}_node_source.npz" --out "$W/maps" --out-name "$N"
log merge - $(( $(stamp) - t ))

# 5. ink-stats from the author's sidecar, then the image with his command line
run python -I - "$REPO/outputs/$N.json" "$W/${N}__stats.json" <<'EOF'
import json, sys
side = json.load(open(sys.argv[1]))
nm = side['ink']['normalisation']
models = {k: dict(model=k, depths=list(range(-15, 16)), **{f: v[f] for f in ('p1', 'p99_5', 'mean', 'sd')})
          for k, v in nm['per_model'].items()}
st = dict(surface=side['ct']['file'], H=side['ct']['shape'][1], W=side['ct']['shape'][2],
          ct=dict(p1=side['ct']['stretch']['lo'], p99_5=side['ct']['stretch']['hi'], layers=[18, 48]),
          models=models,
          ensemble=dict(models=nm['models'], depths=list(range(-15, 16)),
                        p1=nm['ensemble_stretch']['p1'], p99_5=nm['ensemble_stretch']['p99_5']),
          source='copied from the published sidecar (Arm A); not recomputed')
json.dump(st, open(sys.argv[2], 'w'), indent=1)
print('stats from sidecar ->', sys.argv[2])
EOF
M="$W/maps"
mkdir -p "$W/image"
t=$(stamp)
run python -I "$S/make_submission_image.py" --mesh "$REPO/outputs/$N.tifxyz" \
  --umbilicus "$REPO/inputs/umbilicus_PHerc0343.json" --ct "$P" --overlay-style dimmed-ct --caption compact \
  --geometry-sums "$REPO/outputs/SHA256SUMS" --area 380,105,6021,2560 --max-cm2 12 --ct-layer d+1 \
  --ink "$M/${N}__dnative__L66s26_reverse.tif" "$M/${N}__hecate__L66zc33.50_reverse.png" \
        "$M/${N}__hybrid_3d2d-seed42__L66s26_reverse.tif" "$M/${N}__hybrid_3d2d-seed43__L66s26_reverse.tif" \
  --ensemble --ink-stats "$W/${N}__stats.json" \
  --rows "$REPO/inputs/${N}_rows.json" --letters "$REPO/inputs/${N}_letter_boxes.json" --letter-markers off \
  --sizes-csv "$W/image/${N}_letter_sizes.csv" --out-dir "$W/image" --force
log image - $(( $(stamp) - t ))
(cd "$M" && sha256sum ./*_reverse.tif ./*_reverse.png) > "$W/map_sha256.txt"
sha256sum "$W/image/$N.png" "$REPO/outputs/$N.png" > "$W/image_sha256.txt"
echo "done: $W/image/$N.png"
