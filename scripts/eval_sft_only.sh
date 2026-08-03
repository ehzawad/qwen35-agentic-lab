#!/usr/bin/env bash
# Quantify the SFT regression. n is small on purpose: the SFT policy loops on
# tool calls instead of terminating, so each episode costs ~9 min instead of
# ~24 s, and the effect is large enough that 20 episodes settle it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export EXPECT_GPU=A6000
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

N="${N:-20}"
LOG=out/comparison
mkdir -p "$LOG"

echo "[$(date -u +%H:%M:%S)] === eval: sft (n=$N, thinking off) ===" >> "$LOG/progress.log"
AGENTLAB_TRACE="$LOG/trace-sft.jsonl" .venv/bin/python -u -m agentlab.eval \
  --n "$N" --no-thinking --max-new-tokens 768 --tag sft \
  --adapter out/qwen35-4b-sft-lora --quiet > "$LOG/eval-sft.log" 2>&1
rc=$?
echo "[$(date -u +%H:%M:%S)] eval sft exit=$rc" >> "$LOG/progress.log"
echo "[$(date -u +%H:%M:%S)] SFT_EVAL_DONE" >> "$LOG/progress.log"
