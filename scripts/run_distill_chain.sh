#!/usr/bin/env bash
# RS-SFT -> GRPO -> eval. The first run of this pipeline that could plausibly
# show it teaching something, rather than proving it merely executes.
#
# Problem slices are deliberately disjoint:
#   distill  train[0     : 1500]
#   GRPO     train[2000  : 2300]     <- never SFT'd on, so reward has variance
#   eval     test  (untouched split)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# Device choice belongs to the caller; nothing here requires a specific card.
# On a multi-GPU box: CUDA_VISIBLE_DEVICES=<idx> [EXPECT_GPU=A6000] make chain
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
RS=out/qwen35-4b-rssft-lora
GR=out/qwen35-4b-rsgrpo-lora
LOG=out/chain
mkdir -p "$LOG"
# Fresh log per run: the verdict waiter keys on CHAIN_DONE, and an old
# sentinel from a previous run would fire it against stale artifacts.
: > "$LOG/progress.log"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/progress.log"; }
# Fail FAST: a failed stage must not feed stale artifacts into later stages
# for hours before the run finally reports failure.
check(){ rc=$1; stage=$2; say "$stage exit=$rc"; if [ "$rc" -ne 0 ]; then say "CHAIN_FAILED at $stage"; exit 1; fi; }

say "=== 1/4 distill: rejection-sample trajectories from base ==="
$PY -u -m agentlab.distill --n-problems 1500 --k 4 --offset 0 \
    --out data/distill.jsonl --checkpoint data/distill-raw.jsonl \
    > "$LOG/distill.log" 2>&1
check $? distill; say "rows=$(wc -l < data/distill.jsonl 2>/dev/null || echo 0)"

say "=== 2/4 SFT on the distilled set (2 epochs) ==="
$PY -u -m agentlab.sft --data distill --distill-path data/distill.jsonl \
    --epochs 2 --out "$RS" > "$LOG/sft.log" 2>&1
check $? sft

say "=== 3/4 eval: rs-sft (n=200, thinking off) ==="
AGENTLAB_TRACE="$LOG/trace-rssft.jsonl" $PY -u -m agentlab.eval --n 200 --no-thinking \
    --max-new-tokens 768 --tag rssft --adapter "$RS" --quiet > "$LOG/eval-rssft.log" 2>&1
check $? eval-rssft

say "=== 4/4 GRPO on a DISJOINT slice, continuing the RS-SFT adapter ==="
$PY -u -m agentlab.grpo --mode tools --n 300 --grpo-offset 2000 \
    --num-generations 8 --bsz 4 --accum 2 \
    --max-completion-length 1024 --vllm-max-len 4096 --vllm-mem 0.24 \
    --adapter "$RS" --out "$GR" > "$LOG/grpo.log" 2>&1
check $? grpo

say "=== eval: rs-grpo ==="
AGENTLAB_TRACE="$LOG/trace-rsgrpo.jsonl" $PY -u -m agentlab.eval --n 200 --no-thinking \
    --max-new-tokens 768 --tag rsgrpo --adapter "$GR" --quiet > "$LOG/eval-rsgrpo.log" 2>&1
check $? eval-rsgrpo

say "=== results ==="
$PY - <<'PYEOF' | tee -a "$LOG/progress.log"
import json, pathlib
h=f"{'ckpt':<8}{'accuracy':>10}{'tool_use':>10}{'tool_err':>10}{'turns':>8}{'n':>6}"
print(h); print("-"*len(h))
for t in ("base","sft","rssft","rsgrpo"):
    p=pathlib.Path(f"out/eval-{t}.json")
    if p.exists():
        r=json.loads(p.read_text())
        print(f"{r['tag']:<8}{r['accuracy']:>10.3f}{r['tool_use_rate']:>10.3f}"
              f"{r['tool_error_rate']:>10.3f}{r['mean_turns']:>8.2f}{r['n']:>6}")
PYEOF
say "CHAIN_DONE"
