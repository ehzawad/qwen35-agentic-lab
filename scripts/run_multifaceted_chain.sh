#!/usr/bin/env bash
# The one supported end-to-end entry point: `make agentic`.
#
#   suite -> prompt -> baselock -> distill -> views -> sft -> probe
#         -> grpo? -> lock -> eval -> verdict -> ship
#
# Design rules this driver obeys, because the whole run's credibility rests on
# them and a driver is exactly where they get quietly broken:
#
#   RESUMABLE      Every stage is idempotent and decides for itself whether it
#                  is already done, from artifacts on disk. Re-running the whole
#                  chain after a kill re-does no completed work. GPU stages are
#                  driven as LOOPS over short sub-invocations (each module has a
#                  --budget-minutes ceiling well under 8 minutes) rather than one
#                  long blocking call.
#   NO GPU ON A LOOK
#                  --help, --dry-run and --list touch nothing: no CUDA import, no
#                  server, no nvidia-smi allocation. --dry-run prints the exact
#                  commands each pending stage would run.
#   ONE CARD, PINNED
#                  Every GPU stage runs behind require_gpu, which verifies the
#                  pinned PCI index really is the expected card with enough free
#                  memory. One GPU process at a time; the served-model stages
#                  stop their server before returning.
#   LEDGER CEILING The GPU ledger is read before any GPU stage and the projected
#                  cost is refused against the hard ceiling. The stage modules
#                  append their own measured minutes; the stages driven here
#                  through a server append theirs from this script.
#   BLIND UNTIL LOCKED
#                  The held-out eval split is not touched until locks.json
#                  carries BOTH the prompt winner and the trained checkpoint and
#                  seed_reveal.json exists. That ordering is enforced by
#                  scripts/agentic_locks.py, not by convention.
#
# Usage:
#   scripts/run_multifaceted_chain.sh [--dry-run] [--list] [--from STAGE]
#                                     [--only STAGE[,STAGE...]] [--to STAGE]
#                                     [--force STAGE[,STAGE...]] [--yes]
#
# Environment:
#   CUDA_VISIBLE_DEVICES  which card (PCI order). Required for GPU stages.
#   EXPECT_GPU            substring the pinned card's name must contain.
#   MIN_FREE_MIB          free VRAM a GPU stage requires (default 45056).
#   MODEL                 base model id (default Qwen/Qwen3.5-4B).
#   PORT                  vLLM serve port (default 8000).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export PYTHONPATH="src"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45056}"

DATA_DIR="data/suite/v1"
CERTSPECS="$DATA_DIR/certspecs"
MULTIFACE_OUT="out/multiface"
RSSFT="$MULTIFACE_OUT/rssft-lora"
RSGRPO="$MULTIFACE_OUT/rsgrpo-lora"
TRACES="results/agentic/traces"
SECRET_FILE="out/agentic/run_secret.hex"
FROZEN_PROMPT="configs/frozen_prompt.json"
NEUTRAL_PROMPT="prompts/agentic/p1_minimal.txt"
VERDICT_TXT="results/agentic/verdict.txt"
VERDICT_JSON="results/agentic/verdict.json"

STAGES=(suite prompt baselock distill views sft probe grpo lock eval verdict ship)

DRY_RUN=0
LIST=0
FROM=""
TO=""
ONLY=""
FORCE=""

# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

say()  { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# In --dry-run every command is printed and nothing runs. Outside it, a failing
# command stops the chain: a half-finished stage must never look complete.
run() {
  if (( DRY_RUN )); then
    printf '    $ %s\n' "$*"
    return 0
  fi
  printf '    $ %s\n' "$*"
  "$@" || die "command failed: $*"
}

# Same, but the caller inspects the status (used by the resumable loops).
try() {
  if (( DRY_RUN )); then
    printf '    $ %s\n' "$*"
    return 0
  fi
  printf '    $ %s\n' "$*"
  "$@"
}

contains() { [[ ",$1," == *",$2,"* ]]; }

# A prerequisite that a real run must have. In --dry-run it is REPORTED, not
# fatal: the point of a dry run is to show the whole plan, and a chain started
# from scratch legitimately has none of the later stages' inputs yet.
need() {  # path, explanation
  [[ -e "$1" ]] && return 0
  if (( DRY_RUN )); then
    info "[dry-run] prerequisite not present yet: $1"
    info "          ($2)"
    return 1
  fi
  die "$2 (missing: $1)"
}

forced()  { [[ -n "$FORCE" ]] && contains "$FORCE" "$1"; }

# A stage is skipped when its completion marker exists and it was not forced.
done_already() {
  local stage="$1"; shift
  forced "$stage" && return 1
  for artifact in "$@"; do
    [[ -e "$artifact" ]] || return 1
  done
  return 0
}

require_gpu() {
  (( DRY_RUN )) && { info "[dry-run] would verify the pinned GPU"; return 0; }
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die \
    "GPU stage needs an explicit pin, e.g. CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 EXPECT_GPU=A6000 make agentic"
  local idx="${CUDA_VISIBLE_DEVICES%%,*}"
  local line name free
  line="$(nvidia-smi --query-gpu=index,name,memory.free \
          --format=csv,noheader,nounits -i "$idx" 2>/dev/null)" \
    || die "nvidia-smi cannot read PCI index $idx"
  name="$(echo "$line" | cut -d, -f2 | xargs)"
  free="$(echo "$line" | cut -d, -f3 | xargs)"
  if [[ -n "${EXPECT_GPU:-}" && "$name" != *"$EXPECT_GPU"* ]]; then
    die "PCI index $idx is '$name', EXPECT_GPU=$EXPECT_GPU -- wrong card pinned"
  fi
  (( free >= MIN_FREE_MIB )) || die \
    "PCI index $idx ('$name') has ${free} MiB free, need ${MIN_FREE_MIB}; \
another process owns the card"
  info "GPU ok: index $idx '$name', ${free} MiB free"
}

ledger_status() {
  (( DRY_RUN )) && { info "[dry-run] would read the GPU ledger"; return 0; }
  "$PY" - <<'PYEOF'
from agentlab.suite.configio import ledger_hours, load_config
cfg = load_config()
print(f"    GPU ledger: {ledger_hours(cfg):.2f} h used of "
      f"{float(cfg['budget']['gpu_hours_ceiling']):.0f} h ceiling")
PYEOF
}

ledger_note() {  # stage, minutes -- for stages whose module does not self-record
  (( DRY_RUN )) && return 0
  "$PY" - "$1" "$2" <<'PYEOF'
import sys
from agentlab.suite.configio import ledger_append
stage, minutes = sys.argv[1], float(sys.argv[2])
print(f"    ledger: {stage} +{minutes:.1f} min -> "
      f"{ledger_append(stage, minutes):.2f} h cumulative")
PYEOF
}

# ---- vLLM server lifecycle -------------------------------------------------
SERVER_PID=""

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  info "stopping vLLM (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    sleep 3
    info "VRAM after stop: $(nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null)"
  fi
}
trap stop_server EXIT INT TERM

start_server() {  # extra serve flags in "$@"
  local log="out/multiface/serve.$$.log"
  if (( DRY_RUN )); then
    printf '    $ bash scripts/serve.sh %s %s   (background, log -> %s)\n' \
      "$MODEL" "$*" "$log"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  info "starting vLLM: $MODEL $* (log $log)"
  PORT="$PORT" bash scripts/serve.sh "$MODEL" "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  # vLLM can sit for minutes compiling CUDA graphs before the KV cache is
  # allocated. That is not a hang, so the wait is generous and reports progress.
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      info "server up after ~$((i * 10))s"
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$log"; die "vLLM died; see $log"; }
    sleep 10
  done
  tail -30 "$log"
  die "vLLM did not answer /v1/models within 15 min; see $log"
}

# ---- resumable shard loop --------------------------------------------------
# One arm/condition/control over one spec manifest, looped until the shard
# reports complete. Each invocation self-limits with --time-budget-s, so no
# single call approaches the 8-minute ceiling.
eval_arm() {  # arm condition control specs adapter
  local arm="$1" condition="$2" control="$3" specs="$4" adapter="${5:-}"
  local prompt="$NEUTRAL_PROMPT"
  case "$arm" in
    BP|TP|RP) prompt="$(prompt_winner_file)" ;;
  esac
  local -a cmd=("$PY" -m agentlab.suite.evaluate
    --model "$MODEL" --base-id "$MODEL" --arm "$arm" --condition "$condition"
    --control "$control" --specs "$specs" --prompt "$prompt"
    --out "$TRACES" --secret-file "$SECRET_FILE"
    --server "http://127.0.0.1:$PORT" --time-budget-s 360)
  [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter")
  local pass=0 t0 elapsed
  t0=$(date +%s)
  while : ; do
    pass=$((pass + 1))
    (( pass > 200 )) && die "$arm/$condition/$control did not converge in 200 passes"
    local out
    if (( DRY_RUN )); then
      printf '    $ %s   (looped until complete)\n' "${cmd[*]}"
      return 0
    fi
    out="$("${cmd[@]}")" || die "eval shard failed: $arm/$condition/$control"
    printf '    %s\n' "$out"
    echo "$out" | grep -q '"complete": *true' && break
  done
  elapsed=$(( $(date +%s) - t0 ))
  ledger_note "eval:$arm.$condition.$control" "$(awk "BEGIN{print $elapsed/60}")"
}

# The tournament records the winner as a candidate FILE NAME; the directory is
# the preregistered one, so the path is derived, never taken on trust. Before the
# tournament has run there is no winner and the neutral default stands in, which
# is only ever reached in --dry-run: every arm that needs the winner refuses
# outside it.
prompt_winner_file() {
  if [[ ! -f "$FROZEN_PROMPT" ]] && ! (( DRY_RUN )); then
    die "REFUSED: a BP/TP/RP arm needs the frozen tournament winner, and \
$FROZEN_PROMPT does not exist. Silently substituting the neutral prompt would \
produce an arm no preregistered gate describes (S16)."
  fi
  if [[ -f "$FROZEN_PROMPT" ]]; then
    "$PY" - <<PYEOF
import json, pathlib
d = json.load(open("$FROZEN_PROMPT", encoding="utf-8"))
w = d.get("winner", d)
name = w.get("candidate") or w.get("file") or w.get("path")
prereg = json.load(open("configs/agentic_preregister.json", encoding="utf-8"))
print(f"{prereg['prompt_candidates']['directory']}/{pathlib.PurePosixPath(name).name}")
PYEOF
  else
    echo "$NEUTRAL_PROMPT"
  fi
}

candidate_ids() {
  "$PY" -c "
from agentlab.prompt_control import candidates
print(' '.join(c['id'] for c in candidates()))"
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

stage_suite() {  # CPU
  say "suite: generate the committed task suite, validate it, export eval specs"
  if done_already suite "$DATA_DIR/SHA256SUMS" "$CERTSPECS/SHA256SUMS"; then
    info "already generated ($DATA_DIR); validating anyway -- it is 4 s of CPU"
  else
    run "$PY" scripts/generate_suite.py
    run "$PY" scripts/export_eval_specs.py
  fi
  # Validation is NEVER skipped. It is the only thing that proves the bytes
  # follow from the committed seeds, and it costs no GPU.
  run "$PY" scripts/validate_suite.py
}

stage_prompt() {  # GPU
  say "prompt: the eight-candidate elicitation tournament (frozen by hash)"
  if done_already prompt "$FROZEN_PROMPT"; then
    info "already frozen: $FROZEN_PROMPT"; return 0
  fi
  run "$PY" -m agentlab.prompt_control verify
  run "$PY" -m agentlab.prompt_control axes
  require_gpu
  ledger_status
  # Resumable by file presence: each (candidate, round) writes its own result
  # file, so re-invoking replays only what is missing.
  local round cand
  for round in 1 2; do
    for cand in $(candidate_ids); do
      # Round 2 is preregistered for the top two only; the module refuses the
      # rest, so a non-zero status there is expected rather than a failure.
      if (( round == 2 )); then
        try "$PY" -m agentlab.prompt_control run --candidate "$cand" --round 2 \
          --model "$MODEL" --budget-minutes 7 || \
          info "round 2 skipped for $cand (not among the preregistered top two)"
      else
        try "$PY" -m agentlab.prompt_control run --candidate "$cand" --round 1 \
          --model "$MODEL" --budget-minutes 7 \
          || die "tournament candidate $cand r1 failed"
      fi
    done
  done
  run "$PY" -m agentlab.prompt_control finalize
}

stage_baselock() {  # CPU + GPU (dev only; the held-out split stays untouched)
  say "baselock: lock the prompt winner, then measure the base arms on DEV"
  need "$FROZEN_PROMPT" "the prompt winner must be frozen before it can be locked" \
    && run "$PY" scripts/agentic_locks.py lock-prompt --file "$FROZEN_PROMPT"
  if done_already baselock "$TRACES/BP.clean.none.jsonl" "$TRACES/BP.faulted.none.jsonl"; then
    info "dev base arms already traced"; return 0
  fi
  require_gpu
  ledger_status
  start_server
  local specs="$CERTSPECS/dev.jsonl"
  eval_arm B0 clean   none "$specs"
  eval_arm BP clean   none "$specs"
  eval_arm BP faulted none "$specs"
  stop_server
}

stage_distill() {  # GPU
  say "distill: rejection-sample verified trajectories from the base model"
  if done_already distill "data/multiface/accepted.jsonl"; then
    info "accepted corpus already built"; return 0
  fi
  # This refusal is deliberate: sampling before the elicitation control is
  # frozen is the mistake the previous headline made.
  need "$FROZEN_PROMPT" \
    "production rejection sampling requires the frozen prompt winner; run the \
prompt stage first" || return 0
  require_gpu
  ledger_status
  run "$PY" -m agentlab.multidistill plan
  # --auto takes the first pending shard, so the loop is the resume mechanism.
  local pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 400 )) && die "distill did not converge in 400 shards"
    if (( DRY_RUN )); then
      printf '    $ %s   (looped over pending shards)\n' \
        "$PY -m agentlab.multidistill run --auto --model $MODEL --budget-minutes 7"
      break
    fi
    local status
    status="$("$PY" -m agentlab.multidistill status)" || die "distill status failed"
    printf '    %s\n' "$status"
    echo "$status" | grep -q '"pending": *0' && break
    try "$PY" -m agentlab.multidistill run --auto --model "$MODEL" \
      --budget-minutes 7 || die "distill shard failed"
  done
  run "$PY" -m agentlab.multidistill finalize
}

stage_views() {  # CPU
  say "views: build the completion-only SFT views from accepted trajectories"
  if done_already views "data/multiface/sft_views.jsonl"; then
    info "views already built"; return 0
  fi
  need "data/multiface/accepted.jsonl" \
    "view building consumes the accepted corpus; run the distill stage first" \
    || return 0
  run "$PY" -m agentlab.suite.datasets \
    --accepted data/multiface/accepted.jsonl \
    --out data/multiface/sft_views.jsonl
}

stage_sft() {  # GPU
  say "sft: LoRA RS-SFT on the verified views (loss on assistant turns only)"
  if done_already sft "$RSSFT/adapter_model.safetensors"; then
    info "adapter already trained: $RSSFT"; return 0
  fi
  need "data/multiface/sft_views.jsonl" \
    "RS-SFT trains on the verified views; run the views stage first" || return 0
  require_gpu
  ledger_status
  local t0; t0=$(date +%s)
  # Hyperparameters come from configs/multifaceted.yaml `sft:`; passing them
  # explicitly keeps the config the single source rather than the CLI defaults.
  run "$PY" -m agentlab.sft --model "$MODEL" \
    --distill-path data/multiface/sft_views.jsonl \
    --out "$RSSFT" \
    --rank "$(cfg_get sft.lora_rank)" \
    --lr "$(cfg_get sft.lr)" \
    --epochs "$(cfg_get sft.epochs)" \
    --bsz "$(cfg_get sft.bsz)" \
    --accum "$(cfg_get sft.accum)" \
    --max-length "$(cfg_get sft.max_length)"
  ledger_note rs_sft "$(awk "BEGIN{print ($(date +%s) - $t0)/60}")"
}

stage_probe() {  # GPU
  say "probe: does any within-group reward variance exist for GRPO to optimise?"
  if done_already probe "results/agentic/variance_report.json"; then
    info "variance report already written"; return 0
  fi
  need "$RSSFT/adapter_model.safetensors" \
    "the probe measures the RS-SFT policy's group variance; run the sft stage first" \
    || return 0
  require_gpu
  ledger_status
  run "$PY" -m agentlab.variance plan
  local pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 40 )) && die "variance probe did not converge in 40 cells"
    if (( DRY_RUN )); then
      printf '    $ %s   (looped over pending cells)\n' \
        "$PY -m agentlab.variance run --auto --model $MODEL --budget-minutes 7"
      break
    fi
    try "$PY" -m agentlab.variance run --auto --model "$MODEL" \
      --budget-minutes 7 || break
  done
  run "$PY" -m agentlab.variance report
}

grpo_gate_open() {
  (( DRY_RUN )) && return 1  # dry-run reports the conditional as pending, never runs it
  "$PY" - <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path("results/agentic/variance_report.json")
if not p.exists():
    sys.exit(2)
rep = json.loads(p.read_text(encoding="utf-8"))
gate = rep.get("gate") or rep
ok = gate.get("pass")
if ok is None:
    ok = gate.get("gate_pass")
sys.exit(0 if bool(ok) else 1)
PYEOF
}

stage_grpo() {  # GPU, CONDITIONAL
  say "grpo: conditional -- only when the variance probe found outcome variance"
  if done_already grpo "$RSGRPO/adapter_model.safetensors"; then
    info "GRPO adapter already trained: $RSGRPO"; return 0
  fi
  if (( DRY_RUN )); then
    info "[dry-run] would consult results/agentic/variance_report.json; GRPO runs"
    info "          only if its preregistered gate passed, and is SKIPPED (a valid"
    info "          preregistered outcome) if it did not"
    return 0
  fi
  grpo_gate_open
  local gate=$?
  if (( gate == 2 )); then
    die "no variance report; run the probe stage first"
  elif (( gate == 1 )); then
    info "SKIP: the variance probe's preregistered gate did not pass, so GRPO has"
    info "      nothing to optimise. This is a recorded outcome, not a failure:"
    info "      the RS-SFT checkpoint is the trained candidate."
    return 0
  fi
  # The gate opened, so GRPO is required. There is no multifaceted GRPO module
  # yet: the legacy agentlab.grpo trains against the retired GSM8K prompts and
  # tools and would silently produce a checkpoint measured on the wrong task.
  # Refusing loudly is the only honest option.
  if ! "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('agentlab.multigrpo') else 1)"; then
    die "the variance gate OPENED, so GRPO is required, but agentlab.multigrpo \
does not exist yet. Do NOT substitute agentlab.grpo: it trains on the retired \
GSM8K prompts/tools and its checkpoint would be evaluated on the multifaceted \
suite. Implement the module (settings are in configs/multifaceted.yaml 'grpo:') \
or re-run with --only lock,eval,verdict to ship the RS-SFT candidate."
  fi
  require_gpu
  ledger_status
  local t0; t0=$(date +%s)
  run "$PY" -m agentlab.multigrpo --model "$MODEL" --adapter "$RSSFT" --out "$RSGRPO"
  ledger_note grpo "$(awk "BEGIN{print ($(date +%s) - $t0)/60}")"
}

stage_lock() {  # CPU
  say "lock: pick ONE trained candidate on dev, then unblind the held-out seed"
  local candidate stage_name
  if [[ -e "$RSGRPO/adapter_model.safetensors" ]]; then
    candidate="$RSGRPO"; stage_name="grpo"
  elif [[ -e "$RSSFT/adapter_model.safetensors" ]]; then
    candidate="$RSSFT"; stage_name="rs_sft"
  elif (( DRY_RUN )); then
    candidate="$RSSFT"; stage_name="rs_sft"
    info "[dry-run] no adapter on disk yet; would lock whichever exists, "
    info "          preferring the GRPO checkpoint when the probe gate opened"
  else
    die "no trained adapter to lock; run the sft stage first"
  fi
  info "trained candidate: $candidate (stage $stage_name)"
  run "$PY" scripts/agentic_locks.py lock-checkpoint --path "$candidate" \
    --stage "$stage_name"
  run "$PY" scripts/agentic_locks.py reveal
  run "$PY" scripts/agentic_locks.py status
}

stage_eval() {  # GPU -- the first and only stage that touches the held-out split
  say "eval: paired held-out evaluation, all arms, identical specs and decoding"
  if ! (( DRY_RUN )); then
    [[ -f results/agentic/locks.json && -f results/agentic/seed_reveal.json ]] \
      || die "REFUSED: the held-out split stays blind until locks.json and \
seed_reveal.json exist (S18). Run the lock stage."
  fi
  local specs="$CERTSPECS/eval.jsonl"
  local adapter; adapter="$("$PY" -c "
import json,pathlib
p=pathlib.Path('results/agentic/locks.json')
print(json.loads(p.read_text())['checkpoint']['path'] if p.exists() else '$RSSFT')" 2>/dev/null || echo "$RSSFT")"
  require_gpu
  ledger_status
  # One server, LoRA enabled, so base and trained arms face the identical
  # server, parser, schemas and decoding -- the S8 pairing requirement.
  start_server --enable-lora --lora-modules "trained=$adapter" --max-lora-rank 32
  local arm
  for arm in B0 BP; do
    eval_arm "$arm" clean   none "$specs"
    eval_arm "$arm" faulted none "$specs"
    eval_arm "$arm" clean   redacted "$specs"
    eval_arm "$arm" clean   permuted "$specs"
  done
  for arm in T0 TP; do
    eval_arm "$arm" clean   none "$specs" "$adapter"
    eval_arm "$arm" faulted none "$specs" "$adapter"
    eval_arm "$arm" clean   redacted "$specs" "$adapter"
    eval_arm "$arm" clean   permuted "$specs" "$adapter"
  done
  eval_arm TP stress none "$CERTSPECS/eval_stress.jsonl" "$adapter"
  eval_arm BP stress none "$CERTSPECS/eval_stress.jsonl"
  stop_server
}

stage_verdict() {  # CPU
  say "verdict: S8-S18 vetoes, then the preregistered gates, floors and winner"
  # S10 is handed the merged train/dev/eval GROUP manifests, not one file per
  # split. The three training splits deliberately share template ids 0-7 and eval
  # shares 10-11 with eval_stress, so per-split manifests make the leakage veto
  # report a template overlap as a harness BUG and NO verdict could ever issue.
  # scripts/validate_suite.py is the exhaustive check (it also hashes whole-task
  # content); this is the independent confirmation from the exact manifests the
  # arms were evaluated against.
  run "$PY" -m agentlab.analyze --agentic \
    --traces "$TRACES" \
    --preregister configs/agentic_preregister.json \
    --secret "$SECRET_FILE" \
    --specs "$CERTSPECS/eval.jsonl" \
    --split-manifest "train=$CERTSPECS/groups/train.jsonl" \
    --split-manifest "dev=$CERTSPECS/groups/dev.jsonl" \
    --split-manifest "eval=$CERTSPECS/groups/eval.jsonl" \
    --results-dir results/agentic \
    --save "$VERDICT_TXT" --save-json "$VERDICT_JSON"
  (( DRY_RUN )) || info "verdict written to $VERDICT_TXT"
}

stage_ship() {  # GPU, small
  say "ship: serve the configuration the verdict selected and smoke it"
  need "$VERDICT_JSON" "the shipped configuration is whatever the verdict chose; \
run the verdict stage first" || return 0
  local winner adapter=""
  winner="$("$PY" -c "
import json; print(json.load(open('$VERDICT_JSON'))['winner'])" 2>/dev/null || echo "?")"
  info "verdict winner: $winner"
  if (( DRY_RUN )); then
    info "[dry-run] would serve the winning arm and run an 8-episode dev smoke"
    return 0
  fi
  case "$winner" in
    NO\ VERDICT*|none*)
      info "SKIP: the verdict shipped nothing (\"$winner\"), so there is no"
      info "      configuration to smoke. Fix what it flagged and re-run."
      return 0 ;;
    TP*|RP*)
      adapter="$("$PY" -c "
import json; print(json.load(open('results/agentic/locks.json'))['checkpoint']['path'])")"
      info "shipping the trained arm with adapter $adapter" ;;
    *)
      info "shipping the prompt-only base arm (no adapter)" ;;
  esac
  require_gpu
  ledger_status
  if [[ -n "$adapter" ]]; then
    start_server --enable-lora --lora-modules "trained=$adapter" --max-lora-rank 32
  else
    start_server
  fi
  local t0; t0=$(date +%s)
  run "$PY" -m agentlab.suite.evaluate \
    --model "$MODEL" --base-id "$MODEL" \
    --arm "$( [[ -n "$adapter" ]] && echo TP || echo BP )" \
    --condition clean --control none \
    --specs "$CERTSPECS/dev.jsonl" --prompt "$(prompt_winner_file)" \
    --out out/multiface/ship_smoke --secret-file "$SECRET_FILE" \
    --server "http://127.0.0.1:$PORT" --limit 8 --time-budget-s 300 \
    ${adapter:+--adapter "$adapter"}
  ledger_note ship_smoke "$(awk "BEGIN{print ($(date +%s) - $t0)/60}")"
  stop_server
  info "shipped configuration answered an 8-episode dev smoke"
}

cfg_get() {  # dotted path into configs/multifaceted.yaml
  "$PY" - "$1" <<'PYEOF'
import sys
from agentlab.suite.configio import load_config
node = load_config()
for part in sys.argv[1].split("."):
    node = node[part]
print(node)
PYEOF
}

# ---------------------------------------------------------------------------
# argument parsing -- deliberately before anything that could touch a GPU
# ---------------------------------------------------------------------------

usage() { sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while (( $# )); do
  case "$1" in
    -h|--help)  usage; exit 0 ;;
    --dry-run)  DRY_RUN=1 ;;
    --list)     LIST=1 ;;
    # "can a GPU stage start right now?" -- the same guard every GPU stage uses,
    # runnable on its own. Useful on a shared box: it answers before a stage has
    # spent anything, and it never allocates.
    --check-gpu) require_gpu; ledger_status; exit 0 ;;
    --from)     FROM="$2"; shift ;;
    --to)       TO="$2"; shift ;;
    --only)     ONLY="$2"; shift ;;
    --force)    FORCE="$2"; shift ;;
    --yes|-y)   : ;;  # accepted for symmetry with the Makefile; no prompts here
    *)          echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if (( LIST )); then
  printf 'stages, in order:\n'
  for s in "${STAGES[@]}"; do printf '  %s\n' "$s"; done
  exit 0
fi

selected=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a selected <<<"$ONLY"
  for s in "${selected[@]}"; do
    contains "$(IFS=,; echo "${STAGES[*]}")" "$s" || die "unknown stage: $s"
  done
else
  started=0
  [[ -z "$FROM" ]] && started=1
  for s in "${STAGES[@]}"; do
    [[ "$s" == "$FROM" ]] && started=1
    (( started )) && selected+=("$s")
    [[ -n "$TO" && "$s" == "$TO" ]] && break
  done
  (( ${#selected[@]} )) || die "no stages selected (--from '$FROM' --to '$TO')"
fi

printf '# multifaceted chain  model=%s  stages=%s%s\n' \
  "$MODEL" "$(IFS=' '; echo "${selected[*]}")" \
  "$( (( DRY_RUN )) && echo '  [DRY RUN: nothing runs, no GPU is touched]')"

for s in "${selected[@]}"; do
  "stage_$s"
done

say "chain complete: $(IFS=' '; echo "${selected[*]}")"
