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
#                  chain after a kill re-does no completed work.
#   ONE ENGINE PER STAGE
#                  Engine startup measured 289.7 s on the registered A5000, so a
#                  per-shard engine is a budget line, not an implementation
#                  detail: 50 rejection-sampling shards cost 4.02 GPU-hours of
#                  pure model loading and 10 tournament processes another 0.80 h.
#                  Each stage now builds ONE long-lived engine (or server) and
#                  feeds it every pending work unit; the units still checkpoint
#                  and resume, they just no longer pay for an engine. 85 cold
#                  starts become 6: 6.840 h -> 0.483 h, saving 6.357 GPU-hours.
#   NO GPU ON A LOOK
#                  --help, --dry-run and --list touch nothing: no CUDA import, no
#                  server, no nvidia-smi allocation. --dry-run prints the exact
#                  commands each pending stage would run.
#   ONE CARD, PINNED
#                  This is a SINGLE-CARD RTX A5000 study. Every GPU stage runs
#                  behind require_gpu (PCI ordering, the registered index, the
#                  registered card name, exclusive free memory) and then pin_gpu,
#                  which binds the physical GPU UUID for the whole run. The A6000
#                  is released completely: not waited for, not reserved, not
#                  probed. One GPU process at a time; served stages stop their
#                  server before returning.
#   THE PRODUCER ATTESTS
#                  The process that produced the tokens is the authority on the
#                  hardware that produced them. scripts/serve.sh captures a unique
#                  session manifest and then execs vLLM (so the recorded pid owns
#                  the card); this driver countersigns it once /v1/models answers
#                  and hands it to every evaluator invocation, which refuses to
#                  write an episode without it. The run binding, the client's
#                  environment and this driver are CONSTRAINTS, not competing
#                  sources of truth -- if they disagree with the producer, the
#                  stage aborts as a hardware-integrity failure rather than
#                  preferring one of them.
#   LEDGER CEILING The GPU ledger is read before any GPU stage and the projected
#                  cost is refused against the hard ceiling. Every row is a
#                  receipt: timestamps, GPU identity, driver, engine fingerprint,
#                  git SHA and work counts. A served stage is charged ONCE for the
#                  whole server-resident interval INCLUDING startup and idle gaps
#                  -- the startup used to happen before the timer began -- and no
#                  per-arm row overlaps that charge.
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
#   CUDA_VISIBLE_DEVICES  which card (PCI order). Required for GPU stages; the
#                         registered value is 0 (the one RTX A5000).
#   EXPECT_GPU            substring the pinned card's name must contain
#                         (default A5000, the registered card).
#   MIN_FREE_MIB          free VRAM a GPU stage requires (default: the registered
#                         hardware.min_free_mib, 23500 -- an exclusivity guard on
#                         a 24,111 MiB CUDA-visible card, not a 48 GB one).
#   MODEL                 base model id (default Qwen/Qwen3.5-4B).
#   PORT                  vLLM serve port (default 8000).
#   AGENTIC_RUN_ID        S19 run_id stamped on every trace and ledger row.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export PYTHONPATH="src"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export AGENTIC_RUN_ID="${AGENTIC_RUN_ID:-agentic-v1}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
# The card, the index and the exclusivity floor all come from the registered
# hardware block, so there is one place to amend them. CUDA_VISIBLE_DEVICES is
# deliberately NOT defaulted: a GPU stage must be pinned explicitly.
EXPECT_GPU="${EXPECT_GPU:-A5000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-}"        # resolved from the config on first use
REG_INDEX=""
REG_NAME=""

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

# The registered hardware block, read once. CPU only: it parses YAML, never a card.
load_hardware() {
  [[ -n "$REG_INDEX" ]] && return 0
  local line
  line="$("$PY" - <<'PYEOF'
from agentlab.suite.configio import hardware_contract
hw = hardware_contract()
print(hw["cuda_visible_devices"], hw["min_free_mib"], hw["expected_name"])
PYEOF
)" || die "cannot read the registered hardware block from configs/multifaceted.yaml"
  REG_INDEX="$(echo "$line" | awk '{print $1}')"
  [[ -n "$MIN_FREE_MIB" ]] || MIN_FREE_MIB="$(echo "$line" | awk '{print $2}')"
  REG_NAME="$(echo "$line" | cut -d' ' -f3-)"
}

require_gpu() {
  (( DRY_RUN )) && { info "[dry-run] would verify the pinned GPU"; return 0; }
  load_hardware
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die \
    "GPU stage needs an explicit pin: CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=$REG_INDEX EXPECT_GPU=$EXPECT_GPU make agentic"
  [[ "${CUDA_DEVICE_ORDER:-}" == "PCI_BUS_ID" ]] || die \
    "CUDA_DEVICE_ORDER is '${CUDA_DEVICE_ORDER:-unset}'; the registered contract \
is PCI_BUS_ID. Under FASTEST_FIRST the index you pin need not be the card \
nvidia-smi shows."
  local idx="${CUDA_VISIBLE_DEVICES%%,*}"
  [[ "$idx" == "$REG_INDEX" ]] || die \
    "CUDA_VISIBLE_DEVICES=$idx, and this run is registered on index $REG_INDEX \
($REG_NAME). Another card is a SEPARATE registered run with its own run_id, \
locks, seeds and ledger -- not an opportunistic branch of this one."
  local line name free uuid
  line="$(nvidia-smi --query-gpu=index,name,memory.free,uuid \
          --format=csv,noheader,nounits -i "$idx" 2>/dev/null)" \
    || die "nvidia-smi cannot read PCI index $idx"
  name="$(echo "$line" | cut -d, -f2 | xargs)"
  free="$(echo "$line" | cut -d, -f3 | xargs)"
  uuid="$(echo "$line" | cut -d, -f4 | xargs)"
  # EXPECT_GPU is effectively mandatory: it defaults to the registered card and
  # there is no value that switches the check off.
  [[ -n "${EXPECT_GPU:-}" ]] || die "EXPECT_GPU may not be empty; this study is \
registered on one $REG_NAME"
  if [[ "$name" != *"$EXPECT_GPU"* ]]; then
    die "PCI index $idx is '$name', EXPECT_GPU=$EXPECT_GPU -- wrong card pinned"
  fi
  (( free >= MIN_FREE_MIB )) || die \
    "PCI index $idx ('$name') has ${free} MiB free, need ${MIN_FREE_MIB}; \
another process owns the card. This run waits for its own card; it never \
selects a different one."
  # If the run already bound a physical card, it must be THIS one.
  local locked
  locked="$("$PY" -c "
import json,pathlib
p=pathlib.Path('results/agentic/hardware.json')
print(json.loads(p.read_text()).get('gpu_uuid') or '' if p.exists() else '')" \
    2>/dev/null || echo "")"
  if [[ -n "$locked" && -n "$uuid" && "$locked" != "$uuid" ]]; then
    die "this run is locked to GPU $locked and index $idx is $uuid. One run may \
not span two physical cards (S19)."
  fi
  info "GPU ok: index $idx '$name', ${free} MiB free, $uuid"
}

# Bind the physical card for the whole run. Opens a small CUDA context, so it runs
# only inside a GPU stage -- never in --check-gpu or --dry-run -- and it is what
# lets the pure-HTTP evaluator carry the same fingerprint as the engine.
pin_gpu() {
  (( DRY_RUN )) && { info "[dry-run] would bind the run's GPU UUID"; return 0; }
  run "$PY" -m agentlab.env pin --run-id "$AGENTIC_RUN_ID"
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

ledger_note() {  # stage, minutes, [kind], [work-json], [started], [manifest]
  (( DRY_RUN )) && return 0
  "$PY" - "$1" "$2" "${3:-stage}" "${4:-{\}}" "${5:-}" "${6:-}" <<'PYEOF'
import json
import sys
from agentlab.suite.configio import ledger_append
stage, minutes, kind = sys.argv[1], float(sys.argv[2]), sys.argv[3]
work = json.loads(sys.argv[4] or "{}")
started = sys.argv[5] or None
# The producer session that actually spent the minutes. Given one, the receipt
# COPIES that session's snapshot (UUID, driver, engine, manifest digest, session
# id) instead of recomputing a fingerprint in this bookkeeping process, which is
# how the ledger row ends up carrying the same identity as the traces it charges.
manifest = sys.argv[6] or None
cumulative = ledger_append(stage, minutes, kind=kind, work=work, started_at=started,
                           manifest=manifest)
print(f"    ledger: {stage} ({kind}) +{minutes:.1f} min -> "
      f"{cumulative:.2f} h cumulative  work={json.dumps(work)}")
PYEOF
}

# ---- server-session accounting ---------------------------------------------
# A served stage is charged ONCE, for the whole interval the server owned the
# card: startup, every client shard, and the idle gaps between them. The previous
# driver started the server BEFORE the per-arm timer and then charged each arm
# separately, so ~0.241 h of startup was unaccounted while the arms' wall clocks
# overlapped the same seconds. These two calls replace both mistakes.
SESSION_T0=""
SESSION_STARTED=""
SESSION_WORK=""

session_begin() {
  SESSION_T0=$(date +%s)
  SESSION_STARTED="$("$PY" -c "from agentlab.suite.configio import now_utc; print(now_utc())" 2>/dev/null || echo "")"
  SESSION_WORK=""
}

session_add() { SESSION_WORK="${SESSION_WORK:+$SESSION_WORK,}\"$1\""; }

session_end() {  # stage-name
  (( DRY_RUN )) && return 0
  [[ -n "$SESSION_T0" ]] || return 0
  local minutes
  minutes="$(awk "BEGIN{print ($(date +%s) - $SESSION_T0)/60}")"
  # The manifest is passed even though the server has already been stopped: a
  # session is charged AFTER it ends, and the row must carry the identity of the
  # process that spent the minutes, not of this shell. A session that died before
  # writing one is still charged -- the card really was busy, and an unrecorded
  # interval understates the run -- from the run's own fingerprint.
  local manifest=""
  [[ -n "$RUNTIME_MANIFEST" && -s "$RUNTIME_MANIFEST" ]] && manifest="$RUNTIME_MANIFEST"
  ledger_note "$1" "$minutes" "server_session" \
    "{\"unit\":\"work_units\",\"units\":[${SESSION_WORK}]}" "$SESSION_STARTED" \
    "$manifest"
  SESSION_T0=""
}

# ---- vLLM server lifecycle -------------------------------------------------
# The server is the PRODUCER AUTHORITY for every served arm: it captures a unique
# session manifest (physical UUID, driver, PCI bus, CUDA-visible bytes, engine
# fingerprint, thinking mode, model, adapter hash, pid/process-start, port) and
# then execs vLLM, so the pid in that manifest is the pid that owns the card. This
# driver countersigns the manifest once /v1/models answers, and every evaluator
# invocation is handed it. Nothing else may attest this hardware: the evaluator is
# a pure HTTP client and would otherwise write `gpu_uuid: null` (S19).
SERVER_PID=""
SESSION_ID=""
RUNTIME_MANIFEST=""

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
# On a kill the server session must still be charged: the card really was busy,
# and an unrecorded interval is a ledger that understates the run. session_end
# clears its own timer, so a normal stage exit cannot be charged twice.
on_exit() { stop_server; session_end "interrupted_session"; }
trap on_exit EXIT INT TERM

start_server() {  # extra serve flags in "$@"
  local log="out/multiface/serve.$$.log"
  if (( DRY_RUN )); then
    printf '    $ bash scripts/serve.sh %s %s   (background, log -> %s)\n' \
      "$MODEL" "$*" "$log"
    printf '    $ %s -m agentlab.env ready --manifest <session manifest>   (after /v1/models)\n' "$PY"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  # One session id and one manifest path per server, derived from the registered
  # hardware.lock location so there is no second copy of that path in this driver.
  SESSION_ID="$("$PY" -c "
from agentlab.suite.configio import new_session_id
print(new_session_id('serve'))")" || die "cannot mint a session id"
  RUNTIME_MANIFEST="$("$PY" - "$AGENTIC_RUN_ID" "$SESSION_ID" <<'PYEOF'
import sys
from agentlab.suite.configio import manifest_path
print(manifest_path("serve", sys.argv[1], sys.argv[2]))
PYEOF
)" || die "cannot resolve the runtime manifest path"
  info "starting vLLM: $MODEL $* (log $log)"
  info "session $SESSION_ID -> $RUNTIME_MANIFEST"
  PORT="$PORT" AGENTIC_SESSION_ID="$SESSION_ID" \
    RUNTIME_MANIFEST="$RUNTIME_MANIFEST" \
    bash scripts/serve.sh "$MODEL" "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  # vLLM can sit for minutes compiling CUDA graphs before the KV cache is
  # allocated. That is not a hang, so the wait is generous and reports progress.
  local up=0
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      info "server up after ~$((i * 10))s"
      up=1
      break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$log"; die "vLLM died; see $log"; }
    sleep 10
  done
  (( up )) || { tail -30 "$log"; die "vLLM did not answer /v1/models within 15 min; see $log"; }
  # Re-validate the producer attestation now that the engine is actually serving,
  # and countersign it. A manifest for a server that never came up must certify
  # nothing, and the evaluator requires this signature.
  [[ -s "$RUNTIME_MANIFEST" ]] || { tail -30 "$log"; die \
    "the server answered but wrote no producer manifest at $RUNTIME_MANIFEST. An \
unattested server may not produce claim-bearing episodes (S19/D1)."; }
  run "$PY" -m agentlab.env ready --manifest "$RUNTIME_MANIFEST" \
    --run-id "$AGENTIC_RUN_ID" --stage serve --port "$PORT"
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
  if [[ -z "$RUNTIME_MANIFEST" ]] && ! (( DRY_RUN )); then
    die "REFUSED: no producer session manifest for $arm/$condition/$control. The \
evaluator is a pure HTTP client and cannot attest the card that answers it; the \
serving process writes the manifest (see start_server)."
  fi
  local -a cmd=("$PY" -m agentlab.suite.evaluate
    --model "$MODEL" --base-id "$MODEL" --arm "$arm" --condition "$condition"
    --control "$control" --specs "$specs" --prompt "$prompt"
    --out "$TRACES" --secret-file "$SECRET_FILE" --run-id "$AGENTIC_RUN_ID"
    --server "http://127.0.0.1:$PORT" --time-budget-s 360
    --runtime-manifest "${RUNTIME_MANIFEST:-<session manifest>}")
  # A trained arm REQUESTS the LoRA alias the server registered. Passing the
  # adapter for provenance while sending the base model id would evaluate the
  # base weights and label them trained.
  [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter" --served-adapter-name trained)
  local pass=0
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
  # No per-arm ledger row: the whole server session is charged once, so an arm's
  # wall clock must not also charge the same seconds (see session_end).
  session_add "$arm.$condition.$control"
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
  pin_gpu
  ledger_status
  # ONE engine per round, not one per candidate. Each (round, candidate) unit
  # still writes its own file, so re-invoking replays only what is missing --
  # but ten cold starts (0.80 GPU-hours of pure startup) become two.
  local pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 20 )) && die "tournament round 1 did not converge in 20 passes"
    if (( DRY_RUN )); then
      printf '    $ %s   (one engine, every pending candidate)\n' \
        "$PY -m agentlab.prompt_control run --round 1 --model $MODEL"
      break
    fi
    local out
    out="$("$PY" -m agentlab.prompt_control run --round 1 --model "$MODEL" \
           --budget-minutes 55)" || die "tournament round 1 failed"
    printf '    %s\n' "$out"
    echo "$out" | grep -q '"complete": *true' && break
    echo "$out" | grep -q "already done" && break
  done
  # Round 2 is preregistered for the top two only; the module derives them from
  # the round-1 files rather than being told.
  pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 20 )) && die "tournament round 2 did not converge in 20 passes"
    if (( DRY_RUN )); then
      printf '    $ %s   (one engine, the preregistered top two)\n' \
        "$PY -m agentlab.prompt_control run --round 2 --model $MODEL"
      break
    fi
    local out2
    out2="$("$PY" -m agentlab.prompt_control run --round 2 --model "$MODEL" \
            --budget-minutes 55)" || die "tournament round 2 failed"
    printf '    %s\n' "$out2"
    echo "$out2" | grep -q '"complete": *true' && break
    echo "$out2" | grep -q "already done" && break
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
  pin_gpu
  ledger_status
  session_begin
  start_server
  local specs="$CERTSPECS/dev.jsonl"
  eval_arm B0 clean   none "$specs"
  eval_arm BP clean   none "$specs"
  eval_arm BP faulted none "$specs"
  stop_server
  session_end baselock
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
  pin_gpu
  ledger_status
  run "$PY" -m agentlab.multidistill plan
  # ONE engine serves every pending shard; each shard is still an atomic file, so
  # a kill costs one shard and re-invoking resumes. 50 shards each paying the
  # measured 289.7 s startup was 4.02 GPU-hours of pure model loading.
  local pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 40 )) && die "distill did not converge in 40 engine passes"
    if (( DRY_RUN )); then
      printf '    $ %s   (one engine, every pending shard)\n' \
        "$PY -m agentlab.multidistill run --model $MODEL --budget-minutes 55"
      break
    fi
    local out
    out="$("$PY" -m agentlab.multidistill run --model "$MODEL" \
           --budget-minutes 55)" || die "distill pass failed"
    printf '    %s\n' "$out"
    echo "$out" | grep -q '"complete": *true' && break
    echo "$out" | grep -q "all shards done" && break
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
  pin_gpu
  ledger_status
  local t0; t0=$(date +%s)
  # Every hyperparameter -- including lora_alpha, lora_dropout and the evaluation
  # batch settings -- is now a real flag whose DEFAULT is the config value, so a
  # config edit reaches the trainer whether or not this driver passes it. They are
  # still passed explicitly, so the command line records what ran.
  run "$PY" -m agentlab.sft --model "$MODEL" \
    --distill-path data/multiface/sft_views.jsonl \
    --out "$RSSFT" \
    --rank "$(cfg_get sft.lora_rank)" \
    --lora-alpha "$(cfg_get sft.lora_alpha)" \
    --lora-dropout "$(cfg_get sft.lora_dropout)" \
    --lr "$(cfg_get sft.lr)" \
    --epochs "$(cfg_get sft.epochs)" \
    --bsz "$(cfg_get sft.bsz)" \
    --accum "$(cfg_get sft.accum)" \
    --eval-bsz "$(cfg_get sft.eval_bsz)" \
    --eval-accumulation-steps "$(cfg_get sft.eval_accumulation_steps)" \
    --prediction-loss-only "$(cfg_get sft.prediction_loss_only)" \
    --gradient-checkpointing "$(cfg_get sft.gradient_checkpointing)" \
    --packing "$(cfg_get sft.packing)" \
    --max-length "$(cfg_get sft.max_length)"
  ledger_note rs_sft "$(awk "BEGIN{print ($(date +%s) - $t0)/60}")" stage \
    "{\"unit\":\"epochs\",\"count\":$(cfg_get sft.epochs)}"
}

stage_probe() {  # CPU in v1: the probe is NOT EVALUATED on this card
  say "probe: the GRPO variance probe -- NOT EVALUATED on the registered card"
  if done_already probe "results/agentic/variance_report.json"; then
    info "variance-probe artifact already written"; return 0
  fi
  local disposition; disposition="$(cfg_get variance_probe.stage_disposition)"
  if [[ "$disposition" != "RUN" ]]; then
    info "registered disposition: $disposition"
    info "GRPO cannot instantiate on this card, and the probe exists only to"
    info "decide whether GRPO hours are worth spending, so it is short-circuited"
    info "BEFORE it runs. Recording it as \"closed\" would claim the complete"
    info "144-group probe ran and a binding gate failed -- a different claim, and"
    info "one this run has no evidence for."
    run "$PY" -m agentlab.variance report
    return 0
  fi
  # The RUN path (a future registered study): one engine, every pending cell, and
  # the RS-SFT ADAPTER -- probing the base model would measure the wrong policy.
  need "$RSSFT/adapter_model.safetensors" \
    "the probe measures the RS-SFT policy's group variance; run the sft stage first" \
    || return 0
  require_gpu
  pin_gpu
  ledger_status
  run "$PY" -m agentlab.variance plan
  local pass=0
  while : ; do
    pass=$((pass + 1))
    (( pass > 20 )) && die "variance probe did not converge in 20 engine passes"
    if (( DRY_RUN )); then
      printf '    $ %s   (one engine, every pending cell)\n' \
        "$PY -m agentlab.variance run --model $MODEL --adapter $RSSFT"
      break
    fi
    local out
    out="$("$PY" -m agentlab.variance run --model "$MODEL" --adapter "$RSSFT" \
           --budget-minutes 55)" || die "variance probe pass failed"
    printf '    %s\n' "$out"
    echo "$out" | grep -q '"complete": *true' && break
    echo "$out" | grep -q "all cells done" && break
  done
  run "$PY" -m agentlab.variance report
}

stage_grpo() {  # CPU: record the stage DISPOSITION, which is not a gate state
  say "grpo: record the registered stage disposition (no GRPO on this card)"
  local artifact; artifact="$(cfg_get grpo.disposition_artifact)"
  if done_already grpo "$artifact"; then
    info "disposition already recorded: $artifact"; return 0
  fi
  if [[ -e "$RSGRPO/adapter_model.safetensors" ]]; then
    die "a GRPO adapter exists at $RSGRPO, but this run's registered disposition \
is $(cfg_get grpo.stage_disposition). A checkpoint from an unregistered branch \
may not enter this run: it is a separate registered study."
  fi
  info "disposition: $(cfg_get grpo.stage_disposition)"
  info "vLLM carve 0.24 x 23.546 GiB = 5.651 GiB < its own 8.455 GiB policy copy;"
  info "trainer static 9.420 + copy 8.455 = 17.875 GiB before any KV, graphs or"
  info "the 1.53 GiB of logits per 3072-token completion. It cannot instantiate."
  # The receipt is written by the receipts script, so its payload has ONE
  # definition and can be tested on CPU rather than living in this driver.
  run "$PY" scripts/agentic_locks.py grpo-disposition
}

stage_lock() {  # CPU
  say "lock: select the trained candidate, finalize the prereg, unblind the seed"
  local artifact; artifact="$(cfg_get grpo.disposition_artifact)"
  # The GRPO disposition is REQUIRED, not inferred. "GRPO was skipped for a reason
  # nobody wrote down" is indistinguishable from "GRPO ran and lost", and only one
  # of those is reportable -- so a missing GRPO checkpoint with no disposition
  # artifact is an error here rather than a silent fallback to RS-SFT.
  if [[ ! -e "$RSGRPO/adapter_model.safetensors" ]]; then
    need "$artifact" \
      "the GRPO disposition artifact must exist before a trained candidate can be \
selected without a GRPO checkpoint; run the grpo stage" || return 0
    run "$PY" scripts/agentic_locks.py grpo-disposition --verify
  fi
  # RS-SFT is selected EXPLICITLY, never by preferring whichever artifact happens
  # to exist. There is no RS-SFT-versus-GRPO dev comparison to run, because the
  # GRPO branch produced no second candidate -- and saying so is the point.
  local candidate="$RSSFT" stage_name="rs_sft"
  if [[ ! -e "$RSSFT/adapter_model.safetensors" ]] && ! (( DRY_RUN )); then
    die "no RS-SFT adapter to lock; run the sft stage first"
  fi
  info "trained candidate: $candidate (stage $stage_name, the SOLE candidate:"
  info "no RS-SFT-vs-GRPO 0.005 tie rule is applied because no GRPO checkpoint"
  info "exists to compare against)"
  run "$PY" scripts/agentic_locks.py lock-checkpoint --path "$candidate" \
    --stage "$stage_name"
  # Hash-pin the COMPLETED preregistration before the seed is derived from it, so
  # the anchor is the finalized commitment rather than the oldest commit that ever
  # added a preregistration file.
  run "$PY" scripts/agentic_locks.py finalize-prereg
  if ! (( DRY_RUN )); then
    "$PY" scripts/agentic_locks.py verify-prereg --require >/dev/null 2>&1 || die \
      "the finalization marker is not committed yet, so it anchors nothing. \
Commit configs/preregistration_final.json, then re-run --only lock."
  fi
  run "$PY" scripts/agentic_locks.py reveal --require-finalization
  run "$PY" scripts/agentic_locks.py status
}

stage_eval() {  # GPU -- the first and only stage that touches the held-out split
  say "eval: paired held-out evaluation, all arms, identical specs and decoding"
  if ! (( DRY_RUN )); then
    [[ -f results/agentic/locks.json && -f results/agentic/seed_reveal.json ]] \
      || die "REFUSED: the held-out split stays blind until locks.json and \
seed_reveal.json exist (S18). Run the lock stage."
  fi
  local adapter; adapter="$("$PY" -c "
import json,pathlib
p=pathlib.Path('results/agentic/locks.json')
print(json.loads(p.read_text())['checkpoint']['path'] if p.exists() else '$RSSFT')" 2>/dev/null || echo "$RSSFT")"
  require_gpu
  pin_gpu
  ledger_status
  session_begin
  # One server for the WHOLE stage, LoRA enabled, so base and trained arms face
  # the identical server, parser, schemas and decoding -- the S8 pairing
  # requirement -- and the 289.7 s startup is paid once.
  start_server --enable-lora --lora-modules "trained=$adapter" --max-lora-rank 32
  # The work table comes from configs/multifaceted.yaml `eval.manifests`, so a
  # registered manifest cannot be preregistered and then never invoked. MT and H8
  # used to be exactly that; the redacted/permuted controls used to run against
  # the core manifest instead of their own registered ones.
  #
  # MANDATORY first (BP/TP over core clean+faulted, MT, H8, absent, permutation =
  # 7,800 episodes), then the OPTIONAL descriptive arms and the stress set, so a
  # budget stop can never consume mandatory episodes. R0/RP are never scheduled:
  # they are ABSENT BY DESIGN, not missing.
  local arm name specs condition control mandatory
  while read -r arm name specs condition control mandatory; do
    [[ -z "$arm" ]] && continue
    local ad=""
    case "$arm" in T0|TP|RP) ad="$adapter" ;; esac
    info "unit: $arm $name $condition/$control ($([[ "$mandatory" == "True" ]] \
      && echo MANDATORY || echo optional))"
    eval_arm "$arm" "$condition" "$control" "$CERTSPECS/$specs" "$ad"
  done < <(eval_units)
  stop_server
  session_end eval
}

# The registered evaluation work table, mandatory units first.
eval_units() {
  "$PY" - <<'PYEOF'
from agentlab.suite.configio import load_config

cfg = load_config()
ev = cfg["eval"]
rows = []


def emit(arms, manifests, mandatory):
    for man in manifests:
        for arm in arms:
            for condition in man["conditions"]:
                rows.append((arm, man["name"], man["specs"], condition,
                             man["control"], mandatory))


core = [m for m in ev["manifests"] if m["mandatory"]]
extra = [m for m in ev["manifests"] if not m["mandatory"]]
# 1. the MANDATORY census: BP/TP over core clean+faulted, MT, H8, absent, perm.
emit(ev["mandatory_arms"], core, True)
# 2. cut rank 4: the two-fault stress set on the mandatory arms.
emit(ev["mandatory_arms"], extra, False)
# 3. cut rank 3: the descriptive B0/T0 arms, over everything. Cut FIRST, so
#    scheduled LAST -- a budget stop then lands on the cheapest thing to lose.
emit(ev["descriptive_arms"], core + extra, False)
for arm, name, specs, condition, control, mandatory in rows:
    print(arm, name, specs, condition, control, mandatory)
PYEOF
}

stage_verdict() {  # CPU
  # S19 HARDWARE-INTEGRITY is now mandatory too: this driver and the evaluator own
  # the fingerprint every trace and ledger row carries, the analyzer reads it.
  say "verdict: S8-S19 vetoes, then the preregistered gates, floors and winner"
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
  pin_gpu
  ledger_status
  session_begin
  if [[ -n "$adapter" ]]; then
    start_server --enable-lora --lora-modules "trained=$adapter" --max-lora-rank 32
  else
    start_server
  fi
  run "$PY" -m agentlab.suite.evaluate \
    --model "$MODEL" --base-id "$MODEL" --run-id "$AGENTIC_RUN_ID" \
    --arm "$( [[ -n "$adapter" ]] && echo TP || echo BP )" \
    --condition clean --control none \
    --specs "$CERTSPECS/dev.jsonl" --prompt "$(prompt_winner_file)" \
    --out out/multiface/ship_smoke --secret-file "$SECRET_FILE" \
    --server "http://127.0.0.1:$PORT" --limit 8 --time-budget-s 300 \
    --runtime-manifest "${RUNTIME_MANIFEST:-<session manifest>}" \
    ${adapter:+--adapter "$adapter" --served-adapter-name trained}
  session_add "ship_smoke"
  stop_server
  session_end ship_smoke
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

# The header IS the usage text, so it cannot drift: print every comment line
# above `set -uo pipefail` rather than a hardcoded line range that silently
# truncates the moment a design rule is added.
usage() {
  sed -n '2,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" \
    | sed '$d' | sed 's/^# \{0,1\}//'
}

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
