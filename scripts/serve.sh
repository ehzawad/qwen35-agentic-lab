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

# PCI_BUS_ID makes CUDA ordinals agree with the order nvidia-smi prints.
# CUDA_VISIBLE_DEVICES is passed through untouched if you set it.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# The native 262144 context would reserve a KV cache far larger than this task
# needs. Lower this if your card has less memory; raise it if you need the range.
MAXLEN="${MAXLEN:-32768}"

echo "==> serving $MODEL on port $PORT (max_model_len=$MAXLEN)"
exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${@:2}"
