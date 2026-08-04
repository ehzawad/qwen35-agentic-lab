#!/usr/bin/env bash
# Build the lab venv. Ordered installs, not resolver roulette:
# vllm pins torch exactly (2.11.0), so it goes first and wins that argument.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_NO_PROGRESS=1
VENV="$ROOT/.venv"

echo "==> creating venv (python 3.12)"
uv venv --python 3.12 "$VENV"

echo "==> vllm 0.25.1 (pins torch==2.11.0, transformers>=5.5.3)"
uv pip install --python "$VENV" "vllm==0.25.1"

echo "==> trl 1.9.2 + post-training deps"
uv pip install --python "$VENV" \
  "trl==1.9.2" \
  "peft>=0.20.0" \
  "accelerate>=1.14.0" \
  "datasets>=4.7.0" \
  "math-verify" \
  "bitsandbytes" \
  "Pillow" \
  "num2words" \
  "pytest"

# NOTE: deliberately NOT installing flash-linear-attention / causal-conv1d.
# transformers advertises them as the Gated DeltaNet "fast path", but 0.5.2 +
# 1.6.2.post1 segfault the forward pass on torch 2.11.0+cu130 (exit 139, no
# traceback). The torch fallback is slower and correct.

echo "==> versions"
"$VENV/bin/python" - <<'PY'
import torch, transformers, trl, peft, datasets
print(f"torch        {torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
print(f"datasets     {datasets.__version__}")
try:
    import vllm; print(f"vllm         {vllm.__version__}")
except Exception as e:
    print(f"vllm         FAILED: {e}")
PY

echo "==> done"
