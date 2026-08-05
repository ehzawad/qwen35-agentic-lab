#!/usr/bin/env bash
# Serve the model with vLLM under THE REGISTERED ENGINE CONTRACT, as the
# PRODUCER AUTHORITY for everything the served arms claim.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
#       out/qwen35-4b-merged --enable-lora --lora-modules trained=PATH \
#       --max-lora-rank 32
#
# Then: curl http://localhost:8000/v1/models
#
# Every engine setting below is READ from configs/multifaceted.yaml `engine:`
# through agentlab.suite.configio.engine_contract() -- the one copy every stage
# reads. Nothing here carries its own number, because the previous 32,768-token
# default was a second source of truth: it sized the KV cache for a context this
# task never uses, on a card that measures 23.546 GiB.
#
# THIS SCRIPT IS THE PRODUCER-SIDE ATTESTATION (D1). The evaluator is a pure HTTP
# client: it opens no CUDA context, so it cannot know which card answered it.
# This script previously defaulted CUDA_VISIBLE_DEVICES to 0, started vLLM and
# wrote nothing -- so the client's fingerprint legitimately found no hardware and
# a claim-bearing trace was written with `gpu_uuid: null`. Now, in study mode, it:
#
#   * refuses an unspecified device instead of defaulting to GPU 0, and requires
#     explicit PCI-ordered visibility (under FASTEST_FIRST "index 0" need not be
#     the card nvidia-smi shows, and on a shared box the other card belongs to
#     someone else);
#   * runs the registered GPU check and CAPTURES a unique session manifest
#     (agentlab.env capture) before vLLM starts;
#   * `exec`s vLLM, so the pid recorded in that manifest IS the server's pid and
#     a consumer can tell a live session from a stale manifest;
#   * accepts ONLY the registered LoRA-serving options after the model argument.
#     The old unrestricted trailing-argument passthrough let an operator override
#     any engine flag while the recorded engine fingerprint still described the
#     registered config.
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
#
# Environment:
#   CUDA_VISIBLE_DEVICES  REQUIRED. The registered index (0), PCI-ordered.
#   CUDA_DEVICE_ORDER     REQUIRED to be PCI_BUS_ID.
#   PORT                  listen port (default 8000).
#   AGENTIC_RUN_ID        the run this server produces for.
#   AGENTIC_SESSION_ID    session id to use (default: generated, unique).
#   RUNTIME_MANIFEST      where to write this session's manifest (default: the
#                         registered manifests/ directory, unique per session).
#   ALLOW_OFF_CONTRACT=1  DEBUG ONLY: serve off-contract. No manifest is captured
#                         and nothing this engine produces may enter a trace set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${QWEN_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${PORT:-8000}"
PY="$ROOT/.venv/bin/python"

export PYTHONPATH="${PYTHONPATH:-$ROOT/src}"

# The contract, as shell variables. One python call; imports yaml, not torch.
eval "$("$PY" - <<'PYEOF'
from agentlab.suite.configio import engine_contract, load_config
c = engine_contract()
print(f"C_DTYPE={c['dtype']}")
print(f"C_UTIL={c['gpu_memory_utilization']}")
print(f"C_MAXLEN={c['max_model_len']}")
print(f"C_MAXSEQS={c['max_num_seqs']}")
print(f"C_MAXBATCHED={c['max_num_batched_tokens']}")
print(f"C_EAGER={str(c['enforce_eager']).lower()}")
print(f"C_THINKING={str(c['enable_thinking']).lower()}")
print(f"C_TP={c['tensor_parallel_size']}")
print(f"C_LORARANK={load_config()['sft']['lora_rank']}")
PYEOF
)"

MAXLEN="${MAXLEN:-$C_MAXLEN}"
STUDY=1
if [[ "$MAXLEN" != "$C_MAXLEN" ]]; then
  if [[ "${ALLOW_OFF_CONTRACT:-0}" != "1" ]]; then
    echo "REFUSED: MAXLEN=$MAXLEN disagrees with the registered engine contract" >&2
    echo "         max_model_len=$C_MAXLEN. A study stage may not serve an" >&2
    echo "         off-contract engine: S19 reads an engine_fingerprint that" >&2
    echo "         disagrees with the contract as a BUG. Set ALLOW_OFF_CONTRACT=1" >&2
    echo "         only for debugging that produces no claim-bearing trace." >&2
    exit 2
  fi
  STUDY=0
  echo "==> WARNING: OFF-CONTRACT serve (max_model_len=$MAXLEN, contract $C_MAXLEN)."
  echo "    Nothing this engine produces may enter a registered trace set, so NO"
  echo "    producer manifest is captured and no claim-bearing stage can consume it."
fi

# ---------------------------------------------------------------------------
# the passthrough is a WHITELIST, not a blanket forward of every later argument
# ---------------------------------------------------------------------------
# The recorded engine fingerprint describes the registered contract. If an
# arbitrary flag can reach vLLM, the fingerprint describes a config the server is
# not running -- so only the registered LoRA-serving options are accepted.
EXTRA=()
LORA_ENABLED=0
LORA_MODULE=""
LORA_RANK=""
(( $# )) && shift || true
while (( $# )); do
  case "$1" in
    --enable-lora)
      LORA_ENABLED=1; EXTRA+=(--enable-lora); shift ;;
    --lora-modules)
      [[ $# -ge 2 ]] || { echo "REFUSED: --lora-modules needs alias=path" >&2; exit 2; }
      [[ -z "$LORA_MODULE" ]] || { echo "REFUSED: this study serves exactly ONE \
adapter alias; two --lora-modules would let an arm request a checkpoint no lock \
describes" >&2; exit 2; }
      LORA_MODULE="$2"; EXTRA+=(--lora-modules "$2"); shift 2 ;;
    --max-lora-rank)
      [[ $# -ge 2 ]] || { echo "REFUSED: --max-lora-rank needs a value" >&2; exit 2; }
      LORA_RANK="$2"; EXTRA+=(--max-lora-rank "$2"); shift 2 ;;
    *)
      echo "REFUSED: '$1' is not a registered serving option." >&2
      echo "         scripts/serve.sh accepts ONLY --enable-lora," >&2
      echo "         --lora-modules ALIAS=PATH and --max-lora-rank after the" >&2
      echo "         model argument. Every engine setting comes from" >&2
      echo "         configs/multifaceted.yaml 'engine:', because an operator" >&2
      echo "         override would leave the recorded engine_fingerprint" >&2
      echo "         describing a config this server is not running (S19)." >&2
      exit 2 ;;
  esac
done

LORA_ALIAS=""
LORA_PATH=""
if [[ -n "$LORA_MODULE" ]]; then
  (( LORA_ENABLED )) || { echo "REFUSED: --lora-modules without --enable-lora; \
the server would ignore the adapter and serve the BASE weights under a trained \
label" >&2; exit 2; }
  LORA_ALIAS="${LORA_MODULE%%=*}"
  LORA_PATH="${LORA_MODULE#*=}"
  [[ -n "$LORA_ALIAS" && -n "$LORA_PATH" && "$LORA_MODULE" == *=* ]] || {
    echo "REFUSED: --lora-modules must be ALIAS=PATH, got '$LORA_MODULE'" >&2; exit 2; }
  [[ -e "$LORA_PATH" ]] || { echo "REFUSED: adapter path '$LORA_PATH' does not \
exist; a trained arm would silently evaluate the base weights" >&2; exit 2; }
  [[ "$LORA_RANK" == "$C_LORARANK" ]] || { echo "REFUSED: --max-lora-rank \
'$LORA_RANK' is not the registered LoRA rank $C_LORARANK (configs/\
multifaceted.yaml sft.lora_rank)" >&2; exit 2; }
elif (( LORA_ENABLED )); then
  echo "REFUSED: --enable-lora with no --lora-modules registers no adapter, so \
every trained arm would evaluate the base weights" >&2; exit 2
fi

# ---------------------------------------------------------------------------
# explicit, PCI-ordered visibility -- never a default of GPU 0
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSED: CUDA_VISIBLE_DEVICES is unset/empty. This script no longer" >&2
  echo "         defaults it to 0: on a shared box index 0 may not be the card" >&2
  echo "         you mean, and a server nobody pinned produces episodes no" >&2
  echo "         hardware declaration covers (S19)." >&2
  echo "  Fix: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\" >&2
  echo "       EXPECT_GPU=A5000 bash scripts/serve.sh $MODEL" >&2
  exit 2
fi
if [[ "${CUDA_DEVICE_ORDER:-}" != "PCI_BUS_ID" ]]; then
  echo "REFUSED: CUDA_DEVICE_ORDER is '${CUDA_DEVICE_ORDER:-unset}' and the" >&2
  echo "         registered hardware contract is PCI_BUS_ID. Under" >&2
  echo "         FASTEST_FIRST the index you pin need not be the card" >&2
  echo "         nvidia-smi shows." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER

EAGER_FLAG=()
[[ "$C_EAGER" == "true" ]] && EAGER_FLAG=(--enforce-eager)

# ---------------------------------------------------------------------------
# capture the producer manifest, THEN exec vLLM with this same pid
# ---------------------------------------------------------------------------
if (( STUDY )); then
  SESSION_ID="${AGENTIC_SESSION_ID:-}"
  CAP=("$PY" -m agentlab.env capture --stage serve --model "$MODEL"
       --port "$PORT" --pid "$$")
  [[ -n "${AGENTIC_RUN_ID:-}" ]] && CAP+=(--run-id "$AGENTIC_RUN_ID")
  [[ -n "$SESSION_ID" ]] && CAP+=(--session-id "$SESSION_ID")
  [[ -n "${RUNTIME_MANIFEST:-}" ]] && CAP+=(--out "$RUNTIME_MANIFEST")
  if [[ -n "$LORA_PATH" ]]; then
    CAP+=(--adapter "$LORA_PATH" --served-adapter-name "$LORA_ALIAS")
  fi
  # The capture process runs the full hardware veto (PCI order, registered index,
  # registered card, exclusive free memory, the run's UUID binding) and refuses
  # rather than serving an unattested engine.
  MANIFEST_PATH="$("${CAP[@]}")" || {
    echo "REFUSED: the registered GPU check/capture failed; not starting vLLM." >&2
    exit 3; }
  echo "==> producer manifest: $MANIFEST_PATH"
  echo "    (the driver countersigns it with agentlab.env ready once /v1/models"
  echo "     answers; a claim-bearing evaluator requires that signature)"
fi

echo "==> serving $MODEL on port $PORT"
echo "    engine contract: dtype=$C_DTYPE util=$C_UTIL max_model_len=$MAXLEN"
echo "                     max_num_seqs=$C_MAXSEQS max_num_batched_tokens=$C_MAXBATCHED"
echo "                     enforce_eager=$C_EAGER enable_thinking=$C_THINKING tp=$C_TP"
echo "    multimodal inputs: REJECTED (image=0, video=0)"
echo "    card: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($CUDA_DEVICE_ORDER)"
# exec, so the pid recorded in the manifest is the pid that owns the card.
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
  "${EXTRA[@]}"
