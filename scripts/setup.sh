#!/usr/bin/env bash
# Build the lab venv. Two modes, and they answer different questions.
#
#   --frozen     (use this to reproduce the study) install the hash-locked graph
#                in env/requirements.lock.txt on CPython 3.12.13. 213 pinned
#                distributions, every one hash-checked; the resolver makes no
#                choices, so a fresh clone gets the environment the study ran on
#                rather than an environment that satisfies the same ranges.
#
#   --resolve    (the historical path, kept) ordered installs against loose
#                ranges: vllm pins torch exactly (2.11.0) so it goes first and
#                wins that argument. This is how the recorded environment was
#                FIRST built. Re-running it today resolves whatever is current
#                and will NOT reproduce the study.
#
# Default is --frozen. Neither mode overwrites an existing .venv without
# --recreate, and both refuse outright while a process is using that venv --
# swapping wheels under a running stage would corrupt a corpus assembled by two
# different programs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_NO_PROGRESS=1
VENV="$ROOT/.venv"
LOCK="$ROOT/env/requirements.lock.txt"
PYVER="$(cat "$ROOT/.python-version" 2>/dev/null || echo 3.12.13)"

MODE=frozen
RECREATE=0
PROBE_CUDA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --frozen)      MODE=frozen ;;
    --resolve)     MODE=resolve ;;
    --recreate)    RECREATE=1 ;;
    --probe-cuda)  PROBE_CUDA=1 ;;
    -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (want --frozen | --resolve | --recreate | --probe-cuda)" >&2; exit 2 ;;
  esac
  shift
done

# --- refuse to disturb a live venv -----------------------------------------
# The box is shared and stages are long. `pgrep -f` on the venv path catches the
# driver loop, the vLLM engine and any pytest run that is using this
# interpreter.
if pgrep -af "$VENV/bin/" >/dev/null 2>&1; then
  echo "REFUSED: something is running out of $VENV:" >&2
  pgrep -af "$VENV/bin/" >&2 || true
  echo "Installing into it now would change the program mid-run. Wait for the" >&2
  echo "stage to finish, or build a separate venv elsewhere." >&2
  exit 1
fi
if [ -d "$VENV" ] && [ "$RECREATE" -eq 0 ]; then
  echo "REFUSED: $VENV already exists. Pass --recreate to replace it, or point" >&2
  echo "UV_PROJECT_ENVIRONMENT elsewhere. This script will not silently mutate" >&2
  echo "the environment a recorded run was produced with." >&2
  exit 1
fi

if [ "$MODE" = frozen ]; then
  [ -f "$LOCK" ] || { echo "REFUSED: $LOCK is missing." >&2; exit 1; }

  echo "==> creating venv (python $PYVER, exact)"
  uv venv --python "$PYVER" "$VENV"

  echo "==> uv pip sync --require-hashes $(basename "$LOCK")"
  # `sync`, not `install`: the lock is the whole environment, so anything not in
  # it is removed rather than left behind. `--require-hashes` turns a
  # substituted or re-uploaded artifact into a failed install.
  uv pip sync --python "$VENV" --require-hashes "$LOCK"
else
  echo "==> creating venv (python $PYVER)"
  uv venv --python "$PYVER" "$VENV"

  echo "==> vllm 0.25.1 (pins torch==2.11.0, transformers>=5.5.3)"
  uv pip install --python "$VENV" "vllm==0.25.1"

  echo "==> trl 1.9.2 + post-training deps"
  uv pip install --python "$VENV" \
    "trl==1.9.2" \
    "peft>=0.20.0" \
    "accelerate>=1.14.0" \
    "datasets>=4.7.0" \
    "math-verify" \
    "bitsandbytes" \
    "Pillow" \
    "num2words" \
    "pytest"

  echo "==> NOTE: this is a fresh resolution, not the recorded environment."
  echo "    Compare it with: uv pip freeze --python .venv | diff - <(grep -v '^#' requirements-lock.txt)"
fi

# NOTE: deliberately NOT installing flash-linear-attention / causal-conv1d in
# either mode. transformers advertises them as the Gated DeltaNet "fast path",
# but 0.5.2 + 1.6.2.post1 segfault the forward pass on torch 2.11.0+cu130 (exit
# 139, no traceback). The torch fallback is slower and correct. The lock does
# not contain them and scripts/record_host_apparatus.py check FAILS if either
# becomes importable.

echo "==> versions"
# No CUDA probe unless asked. Creating a context costs GPU memory on a card that
# may be running someone else's stage, and this script does not need one to
# report what it installed.
AGENTLAB_PROBE_CUDA="$PROBE_CUDA" "$VENV/bin/python" - <<'PY'
import os
import torch, transformers, trl, peft, datasets
print(f"torch        {torch.__version__}  cuda_build={torch.version.cuda}")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
print(f"datasets     {datasets.__version__}")
try:
    import vllm; print(f"vllm         {vllm.__version__}")
except Exception as e:
    print(f"vllm         FAILED: {e}")
for mod in ("flash_linear_attention", "causal_conv1d"):
    import importlib.util
    if importlib.util.find_spec(mod) is not None:
        raise SystemExit(f"REFUSED: {mod} is importable and segfaults the "
                         f"forward pass on this stack. Uninstall it.")
if os.environ.get("AGENTLAB_PROBE_CUDA") == "1":
    print(f"cuda avail   {torch.cuda.is_available()}")
else:
    print("cuda avail   not probed (pass --probe-cuda; it allocates on the card)")
PY

echo "==> apparatus"
"$VENV/bin/python" "$ROOT/scripts/record_host_apparatus.py" check || {
  echo "    ^ this host differs from env/host_apparatus.json. See docs/REPRODUCE.md:" >&2
  echo "      an original replay needs the recorded apparatus; an independent" >&2
  echo "      replication is a NEW run_id, not an append to the old one." >&2
  exit 1
}

echo "==> done"
