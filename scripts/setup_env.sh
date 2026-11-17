#!/usr/bin/env bash
# Pin villa to a known commit, build the vesuvius env (CPU) and fetch the 9 um cross-scroll checkpoints.
# Inputs: VILLA_DIR, VILLA_REF, CKPT_DIR. Re-runnable.
set -euo pipefail
VILLA_DIR="${VILLA_DIR:-/data/vesuvius/src/villa}"
VILLA_REF="${VILLA_REF:-2d3fb38d33a18b257230c0bd23986436c3d4722e}"   # main @ 2026-11-14
CKPT_DIR="${CKPT_DIR:-/data/vesuvius/checkpoints/ink_9um}"

if [ ! -d "$VILLA_DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/ScrollPrize/villa.git "$VILLA_DIR"
fi
git -C "$VILLA_DIR" fetch --depth 1 origin "$VILLA_REF" 2>/dev/null || true
git -C "$VILLA_DIR" checkout -q "$VILLA_REF"
echo "villa @ $(git -C "$VILLA_DIR" rev-parse HEAD)"

cd "$VILLA_DIR/vesuvius"
uv sync --extra models
uv run --extra models python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"

mkdir -p "$CKPT_DIR"
for seed in 42 43; do
  for step in 020000 075000; do
    uvx --from huggingface_hub hf download scrollprize/ink_9um \
      "hybrid_3d2d-seed${seed}/step-${step}.pth" --local-dir "$CKPT_DIR"
  done
done
find "$CKPT_DIR" -name '*.pth' -exec ls -la {} \;
