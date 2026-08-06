#!/usr/bin/env bash
# Drive ONE chain stage to completion, then commit and push its artifacts.
#
# WHY THIS IS IN THE REPO. The first version of this driver lived under `out/`,
# which is gitignored, so the thing actually executing the study was not in the
# study. A reviewer cloning this repository would have found twelve resumable
# stages and no record of how they were driven. That is a reproducibility hole,
# not a convenience gap, so the driver ships.
#
# WHY IT LOOPS. Each chain invocation returns when its registered time budget
# expires -- that is the designed resume path, since every stage decides from
# artifacts on disk whether it is already done. Looping until the stage's own
# completion artifact appears is therefore reading the contract, not polling.
#
# WHY IT COMMITS. A long unattended run on a shared box will be interrupted.
# Committing at each stage boundary means a crash costs the current stage's
# partial work and nothing earlier. Mid-stage artifacts are deliberately NOT
# committed: the ledger and the session files are appended to by a live process,
# and a commit taken mid-write snapshots a torn record.
#
# WHAT IT WILL NOT DO. It never commits the two lock artifacts. `S18` proves the
# held-out ordering by git ancestry (P < L < R <= E), so the locks commit and the
# reveal commit must each be unique and dedicated; a driver that swept them into
# a stage commit would corrupt the one receipt they exist to provide. Those two
# are made by hand, deliberately, one at a time.
#
#   usage: scripts/run_stage.sh <stage> <completion-artifact> [max_iters] [--no-push]
# `pipefail` is load-bearing: `git push | tail` without it reports a FAILED push as
# success, which is the one lie a crash-resilience driver must not tell.
set -euo pipefail
stage="${1:?stage name}"
done_artifact="${2:?completion artifact path}"
max_iters="${3:-40}"
push="${4:-push}"

# THE LOCK ARTIFACTS ARE NEVER OURS. `results/agentic/locks.json` and
# `seed_reveal.json` are the S18 receipts, and their ordering is proved by git
# ancestry (P < L < R <= E), so each must be added by its own dedicated commit.
# An earlier revision of this driver promised exactly that and then ran
# `git add -A`, which would have swept the currently untracked prompt-only
# locks.json into whatever stage happened to finish next. Refusing here is the
# difference between the promise and the behaviour.
LOCK_PATHS=("results/agentic/locks.json" "results/agentic/seed_reveal.json")
for lock in "${LOCK_PATHS[@]}"; do
  if [ -e "$lock" ] && ! git ls-files --error-unmatch "$lock" >/dev/null 2>&1; then
    echo "[driver] REFUSING: $lock exists and is untracked. It is an S18 receipt and"
    echo "         must be committed alone, by hand, as L or R -- never swept into a"
    echo "         stage commit. Commit it dedicatedly first, then re-run this stage."
    exit 3
  fi
done

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXPECT_GPU="${EXPECT_GPU:-A5000}"
mkdir -p out/driver
log="out/driver/${stage}.log"

for i in $(seq 1 "$max_iters"); do
  if [ -e "$done_artifact" ]; then
    echo "[driver] $stage COMPLETE after $((i - 1)) invocation(s): $done_artifact"
    break
  fi
  echo "[driver] === $stage invocation $i/$max_iters $(date -u +%H:%M:%SZ) ==="
  make agentic ARGS="--only $stage" >>"$log" 2>&1
  rc=$?
  echo "[driver] invocation $i exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[driver] STAGE $stage FAILED (exit $rc). Last 30 log lines:"
    tail -30 "$log"
    exit "$rc"
  fi
done

if [ ! -e "$done_artifact" ]; then
  echo "[driver] $stage did not complete within $max_iters invocations;" \
       "artifact still absent: $done_artifact"
  exit 2
fi

# The stage is done, so its artifacts are a coherent unit. Anything a live
# process is still appending to is excluded by .gitignore or simply not yet
# written; a stage boundary is the only safe commit point.
# EXPLICIT PATHS, never `git add -A`. A blanket add is how an unrelated artifact --
# an S18 receipt, a stray adapter, an operator's scratch file -- ends up inside a
# stage commit, and a run log a reviewer cannot trust is worse than no run log.
STAGE_PATHS=(
  "results/agentic/gpu_ledger.jsonl"
  "results/agentic/gpu_sessions.jsonl"
  "results/agentic/manifests"
  "results/agentic/hardware.json"
  "results/agentic/token_census.json"
  "results/agentic/preflight"
  "configs/frozen_prompt.json"
  "data/suite/v1/manifest.train-dev.json"
  "data/suite/v1/SHA256SUMS.train-dev"
)
to_add=()
for p in "${STAGE_PATHS[@]}"; do
  [ -e "$p" ] && to_add+=("$p")
done
if [ "${#to_add[@]}" -eq 0 ]; then
  echo "[driver] no stage artifacts present to commit for $stage"
elif git diff --quiet -- "${to_add[@]}" && \
     git diff --cached --quiet -- "${to_add[@]}" && \
     [ -z "$(git status --porcelain --untracked-files=normal -- "${to_add[@]}")" ]; then
  echo "[driver] nothing to commit for $stage"
else
  git add -- "${to_add[@]}"
  git commit -q -m "Run the $stage stage

Driven by scripts/run_stage.sh to the completion artifact $done_artifact.
Committed at the stage boundary so an interruption costs this stage and nothing
earlier. GPU minutes are on results/agentic/gpu_ledger.jsonl."
  echo "[driver] committed $stage: $(git log --oneline -1)"
fi

if [ "$push" = "push" ]; then
  git push origin main
  # A push is only pushed if the remote agrees. Reporting otherwise is how work
  # gets lost on a box that is expected to crash.
  local_head="$(git rev-parse HEAD)"
  remote_head="$(git ls-remote origin -h refs/heads/main | cut -f1)"
  if [ "$local_head" != "$remote_head" ]; then
    echo "[driver] PUSH DID NOT LAND: local $local_head vs origin/main $remote_head"
    exit 4
  fi
  echo "[driver] push receipt: origin/main == $local_head"
fi
echo "[driver] $stage done"
