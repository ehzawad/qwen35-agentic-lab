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
set -u
stage="${1:?stage name}"
done_artifact="${2:?completion artifact path}"
max_iters="${3:-40}"
push="${4:-push}"

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
if git diff --quiet && git diff --cached --quiet && \
   [ -z "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "[driver] nothing to commit for $stage"
else
  git add -A
  git commit -q -m "Run the $stage stage

Driven by scripts/run_stage.sh to the completion artifact $done_artifact.
Committed at the stage boundary so an interruption costs this stage and nothing
earlier. GPU minutes are on results/agentic/gpu_ledger.jsonl."
  echo "[driver] committed $stage: $(git log --oneline -1)"
fi

if [ "$push" = "push" ]; then
  git push origin main 2>&1 | tail -2
fi
echo "[driver] $stage done"
