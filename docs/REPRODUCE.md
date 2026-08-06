# Reproducing this study from a clean clone

Written 2026-08-06, after the preregistration was finalized (P = `5844a97`) and
while rejection sampling was still running. It describes the apparatus as it now
is, including the parts that do not work yet. Nothing here amends the
preregistration; `configs/agentic_preregister.json`,
`configs/preregistration_final.json`, `configs/multifaceted.yaml`,
`configs/suite_v1.toml` and `docs/AGENTIC_PROTOCOL.md` are hash-pinned by the
finalization marker and are untouched by everything in this document.

## Two different goals, and they need different things

**Original replay** — reproduce *this run's* numbers. Requires the recorded
apparatus: CPython 3.12.13, the 213-package hash-locked graph, torch
2.11.0+cu130, CUDA runtime 13.0, NVIDIA driver 610.43.02, one RTX A5000, and
base-model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It also
requires the artifacts git does not carry (`out/`, `data/*`): see
`ARTIFACTS.json` and `scripts/hf_artifacts.py`.

**Independent replication** — run the same protocol on your own card. A
different A5000, a different driver, a different kernel are all fine. It is a
**new run**: new `run_id`, new locks, new ledger, new verdict. It must never
append to `agentic-v1`'s traces or ledger — mixed GPU UUIDs inside one trace set
are a harness `BUG`, not experimental noise (S19 HARDWARE-INTEGRITY).

The two words are not interchangeable anywhere in this repository.

## What you need

| | Requirement | Evidence |
|---|---|---|
| GPU | one NVIDIA RTX A5000, 23.546 GiB CUDA-visible, exclusive | measured; `configs/multifaceted.yaml` `hardware.cuda_visible_bytes = 25282805760` |
| Driver | 610.43.02 for replay; anything supporting torch 2.11.0+cu130 on Ampere for replication | `env/host_apparatus.json` |
| CPU / RAM | **16 GiB RAM** is the suggested floor (estimate) | measured: the live distill stage held **4.35 GiB RSS** across its whole process tree at 2 h 24 m in. Tested host: 40 threads, 125.5 GiB. No lower bound has been established by experiment |
| OS | Linux x86_64, glibc ≥ 2.35 (tested: Ubuntu 22.04.1, kernel 5.15.0-58) | `env/host_apparatus.json` |
| Disk | **~60 GiB free** | see the breakdown below |
| Network | Hub access for the model (8.7 GiB) and the wheels; no token needed for the model | `Qwen/Qwen3.5-4B` is public, apache-2.0, ungated as of 2026-08-06 |

### Disk, measured on the tested host

| Path | Size | Note |
|---|---|---|
| HF hub cache, `Qwen/Qwen3.5-4B` | 8.7 GiB | 9,342,816,694 B across 11 files; `env/model_revision.json` |
| `.venv` | 9.5 GiB | torch + the nvidia CUDA 13 wheels dominate |
| `data/` | 279 MiB | of which `data/suite/` is 217 MiB, regenerated not committed |
| `out/` | 3.1 GiB | two LoRA adapters at 1.1 GiB each, preflight evidence 1.1 GiB |
| `.git` | 7.2 MiB | |

That is ~22 GiB of working set. Budget ~60 GiB: the uv download cache holds
another copy of the wheels unless you set `UV_NO_CACHE=1`, and GRPO, evaluation
traces and the held-out phase all still have to land. *(The 60 GiB figure is an
estimate with headroom, not a measurement.)*

## Step 1 — clone

```bash
git clone https://github.com/ehzawad/qwen35-agentic-lab.git
cd qwen35-agentic-lab
```

## Step 2 — the interpreter and the environment, hash-locked

The environment is `env/requirements.lock.txt`: 213 distributions, each pinned
with `==` and carrying SHA-256 hashes, resolved for CPython 3.12.13 on
x86_64-unknown-linux-gnu. `.python-version` pins the interpreter to the patch
level, because `python 3.12` is not a version.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv 0.11.21 is the tested version
bash scripts/setup.sh --frozen                    # this is now the default mode
```

`setup.sh --frozen` runs `uv venv --python 3.12.13` then
`uv pip sync --require-hashes env/requirements.lock.txt`, refuses to overwrite
an existing `.venv` without `--recreate`, and refuses **outright** while any
process is running out of that venv — installing wheels under a live stage would
produce a corpus assembled by two different programs. It ends by running
`scripts/record_host_apparatus.py check`.

Equivalently, by hand:

```bash
uv venv --python 3.12.13 .venv
uv pip sync --python .venv --require-hashes env/requirements.lock.txt
```

Verified: installing this lock into an empty 3.12.13 venv resolves to exactly
the 213 distributions the study has installed, with `--require-hashes`
satisfied.

`scripts/setup.sh --resolve` is the historical path that first built this
environment, kept for the record. It installs against loose ranges and will
**not** reproduce the study today.

`requirements-lock.txt` at the repository root is the older unhashed snapshot.
Keep it for continuity; it is not authoritative, and its header names a
`results/verdict.md` this study has not produced.

### The landmine

Do not install **flash-linear-attention** or **causal-conv1d**. `transformers`
advertises them as the Gated DeltaNet fast path for this architecture and will
use them if importable; on this stack (0.5.2 / 1.6.2.post1 with torch
2.11.0+cu130) they **segfault the forward pass** — exit 139, no Python
traceback, no partial output. The torch fallback is slower and correct. The lock
omits them, `setup.sh` says why, and
`scripts/record_host_apparatus.py check` FAILS if either becomes importable.

## Step 3 — the model, pinned to bytes

`configs/multifaceted.yaml` names the subject as `Qwen/Qwen3.5-4B` and nothing
more. A Hub name is mutable; an upstream re-upload would silently change what
was studied, and the tokenizer digests in `results/agentic/token_census.json`
would not notice, because they identify the text pipeline rather than the
weights.

`env/model_revision.json` closes that. It is an **additive, dated
apparatus-identification record made after P** — the revision was recovered
from the local cache after the study began and disclosed; it was *not*
registered earlier, and it does not amend the frozen preregistration. It pins:

- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- SHA-256 and size of all 11 snapshot files: both weight shards,
  `model.safetensors.index.json`, `config.json`, the four tokenizer files, the
  chat template, and both processor configs
- weights manifest `030638c1b010c6e7…`, full manifest `ede202f395713748…`

Fetch and check it:

```bash
export HF_HUB_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"   # or your own path
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

python scripts/record_model_revision.py verify        # re-hashes; non-zero if a byte moved
python scripts/record_model_revision.py resolve       # the pinned snapshot directory
eval "$(python scripts/record_model_revision.py env)" # HF_HUB_OFFLINE=1 + the pin
```

`verify` also reports whether the cache's `refs/main` has moved away from the
pinned revision. That is **not** a failure — it is exactly the silent
substitution the record exists to make visible.

Verified: with `HF_HUB_OFFLINE=1`, both `from_pretrained(repo_id,
revision=…)` and `from_pretrained(<snapshot dir>)` load the pinned bytes with
no network.

**Known gap, deferred.** The loaders in `src/agentlab/env.py`,
`src/agentlab/multidistill.py`, `src/agentlab/sft.py`, `scripts/serve.sh`,
`scripts/token_census.py` and `scripts/preflight_dev.py` still call
`from_pretrained` without `revision=`. Those files are on the import path of
the rejection-sampling stage that is running, so editing them would change the
program between shards. Until that propagation lands, offline pinning is an
**operator obligation**: export the variables above, and treat
`record_model_revision.py verify` as the gate.

### HF cache and auth

- **Cache location**: `HF_HUB_CACHE`, else `HF_HOME/hub`, else
  `~/.cache/huggingface/hub`. `record_model_revision.py` resolves it in that
  order, so pointing `HF_HUB_CACHE` at a big disk is enough; nothing in this
  repository hardcodes a cache path.
- **Auth for the model**: none. `Qwen/Qwen3.5-4B` is public and ungated.
- **Auth for the run artifacts**: required. `out/` and `data/*` are gitignored,
  so the GPU-hours live in the private dataset repository
  `ehzawad/qwen35-agentic-lab-artifacts-agentic-v1`, indexed by
  `ARTIFACTS.json`. Reading it needs a token with access to that repo:
  `hf auth login`, or `export HF_TOKEN=hf_…`. Without it you can still run the
  protocol from scratch; you cannot verify *this* run's checkpoint.
- **Offline**: `export HF_HUB_OFFLINE=1` once the snapshot is materialized. Every
  hub read then comes from the cache, and a missing file fails loudly instead of
  silently re-downloading a moved `main`.

## Step 4 — CPU-only, before any GPU work

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
python scripts/record_host_apparatus.py check
python scripts/record_model_revision.py verify
```

Known pre-existing failure as of 2026-08-06:
`tests/test_chain.py::test_lock_prompt_refuses_before_P_exists` fails in *this*
checkout because it asserts a refusal message that only appears while P does not
exist — and P now does. It is a test-isolation defect (council Priority 1), not
an apparatus defect, and it is untouched here. Everything else passes:
**912 passed, 1 failed, 6 skipped** on the tree at this commit — the 890 that
passed before the apparatus records landed, plus the 22 in
`tests/test_apparatus_records.py`, with the same single failure before and after.

## Step 5 — the suite payload (deterministic, CPU, ~3 s)

13,200 train/dev specs — 4,800 `oracle_sft`, 3,600 `dev`, 2,400 `distill`,
2,400 `grpo_train` — are regenerated from the seeds in `configs/suite_v1.toml`
rather than committed. `data/suite/v1/manifest.train-dev.json` and
`data/suite/v1/SHA256SUMS.train-dev` are the commitment, and validation proves
the bytes follow from the seeds. (The `Makefile` comment saying "11,320 specs"
is stale; the sealed manifest is the authority.)

Use the **phased** sequence:

```bash
python scripts/generate_suite.py --phase train-dev
python scripts/export_eval_specs.py --splits oracle_sft distill grpo_train dev
python scripts/generate_suite.py --phase train-dev --seal
python scripts/validate_suite.py --require-phase train-dev
```

**`make suite` and the suite stage of `make agentic` are broken right now.** They
call `generate_suite.py` without `--phase`, which the script deliberately
refuses (`REFUSED: --phase is required.`), and they check the obsolete
`SHA256SUMS` filename. This is council blocker 1; the fix belongs to
`scripts/run_multifaceted_chain.sh` and the `Makefile`, both of which the
running stage reads, so it is deferred rather than done. The four commands above
are the sequence the council verified. They were not re-executed while writing
this document — regenerating the payload under a live reader would be
reckless — so treat them as council-verified, not verified here.

Held-out splits do not exist yet and must not: they are a function of a seed
that is revealed only after the checkpoint lock is committed.

## Step 6 — the GPU chain, and the ordering that makes it a study

```bash
make agentic-plan          # prints every pending stage's exact command, no GPU
make agentic               # runs them; resumable, one stage per commit
make agentic ARGS="--only distill"
```

The commit ordering is the science, not bookkeeping. `P < L < R <= E`, proved by
git ancestry:

- **P** — preregistration finalized (`5844a97`).
- **L** — `results/agentic/locks.json`, the prompt winner and the trained
  checkpoint, in a **dedicated commit that changes nothing else**. The reveal
  refuses to run until L is committed, because the held-out seed is a function
  of L's sha.
- **R** — `results/agentic/seed_reveal.json`, plus the sealed held-out
  commitments, in its own later dedicated commit.
- **E** — evaluation, never before R.

`python scripts/agentic_locks.py status` shows where you are.
`docs/DURABILITY.md` covers crash recovery and what each stage writes;
`docs/AGENTIC_PROTOCOL.md` is the frozen protocol itself.

## What is still not reproducible

Honest list, so nobody discovers these the hard way:

1. **The canonical entry point fails at the suite stage** (Step 5). Deferred:
   the fix is in files the running stage reads.
2. **Model revision is not propagated into the loaders** (Step 3). Deferred for
   the same reason. Pinning currently depends on the operator plus `verify`.
3. **Run-scoped paths.** `results/agentic/hardware.json` is bound to this box's
   GPU UUID and says `run_id: dev-preflight-v1`, while the study ledger says
   `agentic-v1`. Hardware, ledgers, manifests, traces and locks share
   un-namespaced paths, so a second card cannot cleanly become a second run
   until they move under `runs/<run_id>/`. Council Priority 4.
4. **Rollout seeds.** Rejection-sampling sampling is not seeded strongly enough
   for byte-exact regeneration; the accepted corpus is published as an artifact
   instead. Council Priority 5.
5. **The locked run's own artifacts** are only verifiable through
   `ARTIFACTS.json` plus the private dataset repo. A stranger without that token
   can replicate the protocol but cannot re-hash this checkpoint.

## The tracked apparatus records

| File | What it pins | Written by |
|---|---|---|
| `.python-version` | CPython 3.12.13 | by hand |
| `env/requirements.lock.txt` | 213 distributions, `==` + SHA-256 | `uv pip compile --generate-hashes` (command in its header) |
| `env/model_revision.json` | Hub revision + SHA-256 of all 11 model files | `scripts/record_model_revision.py record` |
| `env/host_apparatus.json` | OS, kernel, glibc, driver, CUDA runtime, cards, landmine | `scripts/record_host_apparatus.py record` |
| `ARTIFACTS.json` | off-host copies of the gitignored GPU output | `scripts/hf_artifacts.py upload` |

All four are checkable: `record_model_revision.py verify`,
`record_host_apparatus.py check`, `hf_artifacts.py verify`, and
`tests/test_apparatus_records.py`.
