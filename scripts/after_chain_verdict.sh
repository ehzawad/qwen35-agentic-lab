#!/usr/bin/env bash
# Waits for the RS-SFT chain to finish, then completes the experiment:
#   1. re-run base at n=200 so every checkpoint is compared at equal n
#      (the seeded slice makes base@50 a strict prefix of base@200);
#   2. run the analyzer -- harness sanity first, then the pre-registered gates --
#      and save the verdict next to the traces.
# No pkill anywhere in this file, deliberately.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export EXPECT_GPU=A6000
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

PY=.venv/bin/python
LOG=out/chain
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/progress.log"; }

until grep -q "CHAIN_DONE" "$LOG/progress.log" 2>/dev/null; do sleep 120; done

say "=== post: base at n=200 for an equal-n comparison ==="
AGENTLAB_TRACE="$LOG/trace-base.jsonl" $PY -u -m agentlab.eval --n 200 --no-thinking \
    --max-new-tokens 768 --tag base --quiet > "$LOG/eval-base200.log" 2>&1
say "base@200 exit=$?"

say "=== post: verdict ==="
$PY -m agentlab.analyze --tags base sft rssft rsgrpo \
    --trace-dirs out/chain out/comparison \
    --save "$LOG/verdict.md" >> "$LOG/progress.log" 2>&1
say "VERDICT_DONE -> $LOG/verdict.md"
