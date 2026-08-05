#!/usr/bin/env bash
# Serve the model with vLLM under THE REGISTERED ENGINE CONTRACT.
#
#   bash scripts/serve.sh                       # base model
#   bash scripts/serve.sh out/qwen35-4b-merged  # a merged checkpoint
#
# Then: curl http://localhost:8000/v1/models
#
# Every engine setting below is READ from configs/multifaceted.yaml `engine:`
# through agentlab.suite.configio.engine_contract() -- the one copy every stage
# reads. Nothing here carries its own number, because the previous 32,768-token
# default was a second source of truth: it sized the KV cache for a context this
# task never uses, on a card that measures 23.546 GiB.
#
# Two flags are load-bearing rather than cosmetic:
#
#   --default-chat-template-kwargs '{"enable_thinking":false}'
#       This checkpoint defaults thinking ON. Offline rejection sampling already
#       renders with thinking disabled, so without this the served arms would run
#       a DIFFERENT policy from the trained one, spend the completion budget on
#       reasoning, and read as "the model never committed an answer".
#
#   --limit-mm-per-prompt '{"image":0,"video":0}'
#       The contract says multimodal inputs are REJECTED, not merely unused.
#       Qwen3.5 is a natively multimodal checkpoint, so an image part would
#       otherwise be accepted and contribute an episode no registered claim
#       describes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${QWEN_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${PORT:-8000}"

# PCI_BUS_ID makes CUDA ordinals agree with the order nvidia-smi prints, and this
# run is registered on index 0 of one RTX A5000.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-$ROOT/src}"

# The contract, as shell variables. One python call; imports yaml, not torch.
eval "$("$ROOT/.venv/bin/python" - <<'PYEOF'
from agentlab.suite.configio import engine_contract
c = engine_contract()
print(f"C_DTYPE={c['dtype']}")
print(f"C_UTIL={c['gpu_memory_utilization']}")
print(f"C_MAXLEN={c['max_model_len']}")
print(f"C_MAXSEQS={c['max_num_seqs']}")
print(f"C_MAXBATCHED={c['max_num_batched_tokens']}")
print(f"C_EAGER={str(c['enforce_eager']).lower()}")
print(f"C_THINKING={str(c['enable_thinking']).lower()}")
print(f"C_TP={c['tensor_parallel_size']}")
PYEOF
)"

MAXLEN="${MAXLEN:-$C_MAXLEN}"
if [[ "$MAXLEN" != "$C_MAXLEN" ]]; then
  if [[ "${ALLOW_OFF_CONTRACT:-0}" != "1" ]]; then
    echo "REFUSED: MAXLEN=$MAXLEN disagrees with the registered engine contract" >&2
    echo "         max_model_len=$C_MAXLEN. A study stage may not serve an" >&2
    echo "         off-contract engine: S19 reads an engine_fingerprint that" >&2
    echo "         disagrees with the contract as a BUG. Set ALLOW_OFF_CONTRACT=1" >&2
    echo "         only for debugging that produces no claim-bearing trace." >&2
    exit 2
  fi
  echo "==> WARNING: OFF-CONTRACT serve (max_model_len=$MAXLEN, contract $C_MAXLEN)."
  echo "    Nothing this engine produces may enter a registered trace set."
fi

EAGER_FLAG=()
[[ "$C_EAGER" == "true" ]] && EAGER_FLAG=(--enforce-eager)

echo "==> serving $MODEL on port $PORT"
echo "    engine contract: dtype=$C_DTYPE util=$C_UTIL max_model_len=$MAXLEN"
echo "                     max_num_seqs=$C_MAXSEQS max_num_batched_tokens=$C_MAXBATCHED"
echo "                     enforce_eager=$C_EAGER enable_thinking=$C_THINKING tp=$C_TP"
echo "    multimodal inputs: REJECTED (image=0, video=0)"
exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --dtype "$C_DTYPE" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$C_UTIL" \
  --max-num-seqs "$C_MAXSEQS" \
  --max-num-batched-tokens "$C_MAXBATCHED" \
  --tensor-parallel-size "$C_TP" \
  "${EAGER_FLAG[@]}" \
  --default-chat-template-kwargs "{\"enable_thinking\":$C_THINKING}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${@:2}"
