#!/usr/bin/env bash
# Stage 4c -- serve the model with vLLM, tool calling enabled.
#
#   bash scripts/serve.sh                      # base model
#   bash scripts/serve.sh out/qwen35-4b-merged # a merged checkpoint
#
# Then: curl http://localhost:8000/v1/models
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${QWEN_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${PORT:-8000}"

# PCI_BUS_ID so index 1 means the A6000, matching nvidia-smi. Without it CUDA
# orders fastest-first and index 1 is the A5000.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 262144 native context would reserve a KV cache far larger than this task needs;
# 32768 leaves room for a healthy batch on a 48 GB card. Raise if you need it.
MAXLEN="${MAXLEN:-32768}"

echo "==> serving $MODEL on port $PORT (A6000, max_model_len=$MAXLEN)"
exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${@:2}"
