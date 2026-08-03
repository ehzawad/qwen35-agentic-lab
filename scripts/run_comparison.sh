#!/usr/bin/env bash
# Retrain both stages with the current (fixed) code, then evaluate
# base vs SFT vs GRPO on the same held-out split.
#
# Everything before this point proved the pipeline RUNS. This is the run that
# asks whether it TEACHES anything -- the comparison the repo has never made.
#
# Deliberate choices:
#   * all three checkpoints are evaluated with thinking OFF, because that is the
#     grammar both stages are now trained in. Same condition for all three, so
#     the comparison is controlled.
#   * the same held-out slice for every checkpoint (build_eval is seeded).
#   * artifacts go to the canonical Makefile paths, not the _smoke-* dirs, so
#     nothing here is contaminated by the pre-fix runs.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export EXPECT_GPU=A6000
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
# The OOM this hit once was 4.42 GiB reserved-but-unallocated, i.e. fragmentation
# between vLLM's pool and the training allocator. torch suggests exactly this.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
SFT=out/qwen35-4b-sft-lora
GRPO=out/qwen35-4b-grpo-lora
LOG=out/comparison

mkdir -p "$LOG"
stamp() { date -u +%H:%M:%S; }
say()   { echo "[$(stamp)] $*" | tee -a "$LOG/progress.log"; }

if [ "${SKIP_SFT:-0}" = "1" ] && [ -f "$SFT/adapter_model.safetensors" ]; then
  say "=== stage 1: SFT SKIPPED (reusing existing $SFT) ==="
else
  say "=== stage 1: SFT (fixed code: prompt/completion, no vision LoRA, thinking off) ==="
  $PY -u -m agentlab.sft --n 4000 --out "$SFT" > "$LOG/sft.log" 2>&1
  say "sft exit=$?"
fi

say "=== stage 3: GRPO continuing the SFT adapter ==="
$PY -u -m agentlab.grpo --mode tools --n 300 --num-generations 8 --bsz 4 --accum 2 \
    --max-completion-length 1024 --vllm-mem 0.24 --vllm-max-len 4096 \
    --adapter "$SFT" --out "$GRPO" > "$LOG/grpo.log" 2>&1
say "grpo exit=$?"

# ---- the comparison -------------------------------------------------------
N=${N:-100}
for cfg in "base::" "sft::$SFT" "grpo::$GRPO"; do
  tag="${cfg%%::*}"; adapter="${cfg##*::}"
  say "=== eval: $tag (n=$N, thinking off) ==="
  args=(--n "$N" --no-thinking --max-new-tokens 768 --tag "$tag" --quiet)
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  AGENTLAB_TRACE="$LOG/trace-$tag.jsonl" \
    $PY -u -m agentlab.eval "${args[@]}" > "$LOG/eval-$tag.log" 2>&1
  say "eval $tag exit=$?"
done

say "=== results ==="
$PY - <<'PY' | tee -a "$LOG/progress.log"
import json, pathlib
rows = []
for tag in ("base", "sft", "grpo"):
    p = pathlib.Path(f"out/eval-{tag}.json")
    if p.exists():
        rows.append(json.loads(p.read_text()))
if rows:
    hdr = f"{'checkpoint':<8}{'acc':>8}{'tool_use':>10}{'tool_err':>10}{'turns':>8}{'n':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<8}{r['accuracy']:>8.3f}{r['tool_use_rate']:>10.3f}"
              f"{r['tool_error_rate']:>10.3f}{r['mean_turns']:>8.2f}{r['n']:>6}")
else:
    print("no eval results found")
PY
say "ALL_DONE"
