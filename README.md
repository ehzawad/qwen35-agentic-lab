# qwen35-agentic-lab

A verifiable multifaceted agent pipeline for **[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)**
on **one RTX A5000 (24 GB)**. The capability target is deliberately bounded and
machine-checkable: multi-tool composition, dependency depth up to 8 required
calls, stateful constraints with irreversible commitments, and recovery from
injected tool failures — all scored by exact verifiers, never by an LLM judge.

## Status — 2026-08-06

**The preregistration is FINALIZED and pushed. The prompt winner is locked. The
checkpoint is not, so `L` does not exist. Rejection sampling is running on the
card right now. NO CAPABILITY RESULT EXISTS YET, and none can until the
held-out set is derived, generated and evaluated — which cannot happen before
`L`.**

| fact | state | receipt |
|---|---|---|
| `P` — preregistration finalized | **commit `5844a97d2096fb55186e8559f4a4481dc3b75e9d`**, pushed; anchor `finalization-marker`, no drift | `configs/preregistration_final.json` |
| prompt winner | **locked under `P`**: `prompts/agentic/p2_plan_state_act.txt`, `sha256 5facfd02997d…`, frozen at commit `4c840e3` | `results/agentic/locks.json` (on disk, deliberately **not** committed yet — `L` must be the dedicated commit adding the *complete* lock) |
| `L` — checkpoint lock | **does not exist, and was correctly refused**: `status` reports `REFUSED: the lock is INCOMPLETE, so there is nothing for the held-out seed to be a function of` | `python scripts/agentic_locks.py status` |
| `R` — seed reveal | does not exist; the six held-out generation seeds are a function of `L`, so the held-out set **cannot be generated yet** | `heldout-master-v2` derivation, `configs/agentic_preregister.json` |
| current stage | **`distill` (rejection sampling) IN PROGRESS** on the pinned A5000. Snapshot at `2026-08-06T17:42:45Z`: 15 of 60 planned shards sealed and receipt-validated, 4,400 verified rollouts. Both counts move while the stage runs | `data/multiface/raw/shard-*.receipt.json` |
| GPU hours | **non-zero**: **9.249 h** charged at that same instant, and still rising. The ledger is the authority, not this table | `results/agentic/gpu_ledger.jsonl`, `gpu_sessions.jsonl` |
| CPU test suite | **1,052 passed, 6 skipped, 0 failed** on commit `1b07a64` | `PYTHONPATH=src .venv/bin/python -m pytest -q` |
| dev preflight | **five probes green, 87 checks, all pass** | `results/agentic/preflight/probe{1..5}.json` |
| capability claim | **NONE. No trained checkpoint exists, no held-out bytes exist, no gate has been evaluated.** | — |
| registered evaluation | the full **7,800**-episode mandatory census **is being completed**, not descoped — projected **~6.4 GPU-h** from a measured 177.6 min per 3,600 episodes | [docs/AMENDED_REPLICATION_NOT_RUN.md](docs/AMENDED_REPLICATION_NOT_RUN.md) |
| dropped work | the **optional cluster-balanced amended replication is not being run**. It was never registered, so this is **not a deviation**, and no deviation notice is due | same file, with the tripwire that would make one mandatory |
| read before using it | **[docs/USAGE.md](docs/USAGE.md)** — the deep-horizon collapse, what the recovery number is and is not, the exact serving contract, the licence and the pinned base revision | `env/model_revision.json` |
| results document | **[docs/RESULTS.md](docs/RESULTS.md)** — the verbatim required wording and every gate template, with **all held-out slots UNFILLED** on purpose | — |

The brittle `tests/test_chain.py::test_lock_prompt_refuses_before_P_exists` test
was removed in commit `1b07a64`. Its placeholder prompt hash always triggered
the earlier S16 candidate-hash refusal, so it never reached the no-`P` gate its
name claimed to exercise. The no-`P` refusal remains covered on the reveal path
by `test_reveal_always_requires_a_committed_P`, and the non-candidate prompt
refusal has its own focused test. The recorded run at that commit is green:
**1,052 passed, 6 skipped, 0 failed**. D5 is closed; the remaining repairs are
tracked, without rewriting the historical diagnosis, in
[docs/DEFERRED_REPAIRS.md](docs/DEFERRED_REPAIRS.md).

One of those deferrals is load-bearing: **the prompt-tournament receipt
`configs/frozen_prompt.json` is malformed and must be corrected before `L`.**
Its final ranking and `round2_candidates` name `p8` — a candidate with 300
observations that never ran round two — while the actual round-two finalist was
`p6` at 900. The **winner is unaffected** (`p2` leads at 0.9367 over 900 against
`p6` at 0.9178), no GPU work needs rerunning, and the file cannot be touched
while distill re-reads it between shards.

The earlier S18 defect — "test-set commitment, not test-blindness", because the
eval seeds were public in `configs/suite_v1.toml` and the held-out answers were
regenerable from `P` — **is closed**, by the dated amendment of 2026-08-05 that
moved the derivation onto `L`, split generation into two phases with two
commitments, renamed the veto **S18 POST-LOCK HELDOUT**, and quarantined the
stale old-seed bundles with a receipt. That amendment landed before any GPU
hour, any held-out result and any optimizer step; `P` was finalized after it.

What is verified today is still the harness, not a capability: deterministic
generation, the strict verifier, the analyzer and its vetoes, the paired engine
contract, the preflight probes above, and the test suite. The training legs
(distilled SFT; GRPO is **not run** on this card) are candidates, not
predetermined winners: if training does not beat the locked elicitation control
by the registered margin without harming clean performance, the shipped
pipeline is the prompted base model.

**How every endpoint will be reported is fixed in advance**, before the numbers
exist: [docs/RESULTS.md](docs/RESULTS.md) carries the verbatim required wording,
the per-gate template, the underpowered-primary template and the descriptive-number
prefix with every slot unfilled, and
[docs/INTERPRETATION.md](docs/INTERPRETATION.md) carries the analyzer semantics
behind them — the analyzer produces `FAIL` (not "INCONCLUSIVE by saturation")
when a computable lower bound does not clear `+0.05`, and a saturated
*development* endpoint does not establish that training has no effect.

## What is measured today — all of it training-side

**Descriptive only; not a preregistered claim.** Every row is a **dev** or
**distill** observation from the base model under the frozen prompt, with
repeated samples per task. None of it is held-out, none of it is the adapter,
and no gate is computed on any of it. The sealed shards at that snapshot hold the
`fulfillment` family only, so no row is a whole-suite rate; the corpus is still
growing, and a later snapshot replaces these rather than mixing with them. Every
caveat, the per-cell decomposition and the reason no intervals are printed:
[docs/RESULTS.md](docs/RESULTS.md) §6 and §10.

| observation | value | split | snapshot |
|---|---|---|---|
| certified success, `fulfillment-h4` | 1,174/1,200 (97.8%) | distill | 13 shards, `2026-08-06T16:46:19Z` |
| certified success, `fulfillment-h8` | 1,509/1,600 (94.3%) | distill | 13 shards, `2026-08-06T16:46:19Z` |
| certified success, `fulfillment-h14` | **59/1,152 (5.1%)** | distill | 13 shards, `2026-08-06T16:46:19Z` |
| recovery predicate met, fault-assigned | 1,302/1,752 (74.3%) — a **mixture** over cells: 575/600 H4, 714/800 H8, 13/352 H14 | distill | 13 shards, `2026-08-06T16:46:19Z` |
| certified success, all sealed cells pooled | 2,742/3,952 (69.4%) | distill | 13 shards, `2026-08-06T16:46:19Z` |
| prompt-only control, H4 all-tools orchestration | 295/300 (0.9833) | dev | prompt tournament, `2026-08-06` |
| prompt-only control, H8 execution | 296/300 (0.9867) | dev | prompt tournament, `2026-08-06` |
| anything held-out | **none exists** | — | — |

> **Deep horizons collapse: fulfillment certified success falls from 94.3–97.8%
> at H4/H8 to 59/1,152 (5.1%) at H14, so H12/H14/H20 are outside the supported
> reliability envelope and are measured-only cells.**

## The one supported entry point

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 make agentic
make agentic-plan        # the same chain as a dry run: prints every command,
                         # touches no GPU
```

`make agentic` runs twelve resumable stages:

| stage | what it does | GPU |
|---|---|---|
| `suite` | generate the committed task suite, validate all eleven binding conditions, export the evaluation spec manifests | no |
| `prompt` | the eight-candidate elicitation tournament, frozen by hash | yes |
| `baselock` | lock the prompt winner; measure the base arms on **dev** | yes |
| `distill` | rejection-sample verified trajectories from the base model | yes |
| `views` | build the completion-only SFT views (loss on assistant turns only) | no |
| `sft` | LoRA RS-SFT on the verified views | yes |
| `probe` | the GRPO variance probe — **not evaluated** on this card (see below) | no |
| `grpo` | records the registered GRPO stage **disposition**; no GRPO runs | no |
| `lock` | lock the RS-SFT checkpoint, finalize the prereg, unblind the seed | no |
| `eval` | paired held-out evaluation, the registered manifests, one server | yes |
| `verdict` | S8–S19 vetoes, then the preregistered gates, floors and winner | no |
| `ship` | serve the configuration the verdict selected and smoke it | yes |

Every stage decides from artifacts on disk whether it is already done, so a
killed run resumes by re-invoking. Each GPU stage builds **one long-lived engine
or server** and feeds it every pending work unit; the units still checkpoint and
resume, they just no longer pay for an engine. At the measured 289.7 s startup
that is the difference between 85 cold starts (6.840 GPU-hours of pure model
loading) and 6 (0.483 h) — **6.357 GPU-hours saved**, and the startup is now on
the ledger instead of happening before the timer began.

The held-out split is not touched until the prompt winner and the checkpoint are
locked and the seed is revealed — `scripts/agentic_locks.py` refuses the wrong
order rather than trusting it, and the seed now anchors to a **finalization
marker** that hash-pins the completed preregistration (the old anchor was the
oldest commit that ever added a preregistration file, which no later edit could
change). Every gate threshold was committed before held-out results existed, and
a harness BUG vetoes every model-level gate, floor and claim.

That ordering is only worth something if it is checkable, so
[`scripts/verify_heldout_release.py`](scripts/verify_heldout_release.py) checks it
before a card is pinned: `P < L < R <= E` by **git ancestry**, never by
timestamps, plus the reveal rederiving from the lock commit it names, the held-out
manifest and checksums committed in that one reveal commit, and every released
byte hashing to its committed value. It reads no mtime, imports no GPU stack, and
refuses instead of warning. What it proves, what it refuses, and the exact
one-line call the `eval` stage still needs are in
[docs/HELDOUT_RELEASE_GATE.md](docs/HELDOUT_RELEASE_GATE.md).

The `eval` stage runs the registered manifest census, not a hand-written list:
7,800 mandatory episodes — BP/TP over core clean+faulted (4,800), MT (1,200), H8
augmentation (400), the absent-information control (1,200) and the counterfactual
permutation control (200) — scheduled before the optional stress set and the
descriptive `B0`/`T0` arms, so a budget stop can never eat a mandatory sample.
`R0`/`RP` are **absent by design**, never merely missing.

Select stages with `make agentic ARGS="--from sft"`, `ARGS="--only verdict"`, or
`ARGS="--to probe"`; `make agentic-stages` lists them.

Helper targets (`make help`) exist for tests and debugging — `make verify`
(CPU test suite), `make suite`, `make validate-suite`, `make locks`,
`make smoke`, `make serve`, `make gpu` — but there is exactly one supported
end-to-end workflow, and alternate pipelines are out of scope.

## Serving the shippable configuration, and the demo

Two client model ids, one script, one demo — the whole shipping surface is
[docs/SERVING.md](docs/SERVING.md), and what a user must know before running it
is [docs/USAGE.md](docs/USAGE.md). `Qwen/Qwen3.5-4B` (the base weights under the
frozen winning prompt, Hub revision `851bf6e8…`) is always shipped and is the
default; `trained` (the same base plus a validated `out/multiface/rssft-lora`,
LoRA rank 32) is shipped only if training completes and validates, is labelled
experimental, and **is not claimed to beat the prompt-only base** unless
separately reported evidence says so.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 \
AGENTIC_RUN_ID=agentic-v1 PORT=8000 bash scripts/serve.sh Qwen/Qwen3.5-4B

# once out/multiface/rssft-lora exists and validates, the same one command
# serves both ids:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 \
AGENTIC_RUN_ID=agentic-v1 PORT=8000 bash scripts/serve.sh Qwen/Qwen3.5-4B \
  --enable-lora --lora-modules trained=out/multiface/rssft-lora \
  --max-lora-rank 32
```

The prompt is **not** injected by the server. Every request must begin with a
system message carrying the exact bytes of
`prompts/agentic/p2_plan_state_act.txt` (`sha256 5facfd02997d…`); a bare request
to `Qwen/Qwen3.5-4B` without it is not the shipped configuration and must not be
presented as one. Thinking is disabled and multimodal input is rejected.

Against a server you started:

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_agentic.py
```

[`scripts/demo_agentic.py`](scripts/demo_agentic.py) runs a five-episode panel
whose task ids are fixed in the source before the server is contacted — one of
them under a real injected `rate_limit` fault, and one at the deep-horizon
boundary where the model is expected to fail — and prints the exact tool calls,
the fault envelope the model read, its recovery token, the remedial call and the
verifier's verdict for each. It selects nothing by outcome and rerolls nothing.
**These are five synthetic dev demonstrations, not a benchmark and not held-out
evidence** — the deep-horizon warning above is the one to carry away, and
[docs/USAGE.md](docs/USAGE.md) states the rest, including that certified
recovery was measured under an injected token-bearing fault contract rather
than arbitrary production failures.

## Task families and the tool cap

Three task families, no more. The names in parentheses are the committed family
identifiers used by the code, the suite manifests, the preregistration and the
analyzer — there is one taxonomy, not a prose one and a machine one:

1. **Compositional quantitative tasks** (`typed_relay`) — require `kb_lookup →
   unit_convert → calculator` in a causally forced order, with facts unavailable
   in the prompt and exact independently computed answers.
2. **Synthetic multi-hop knowledge-graph tasks** (`lookup_chain`) — each lookup
   reveals the key for the next; dependency depth generated at 2, 4, 8 and 12;
   the verifier confirms the minimal dependency path.
3. **Constrained stateful procurement** (`fulfillment`) — inventory, units,
   budget, and irreversible commitments, adding exactly two environment tools:
   `warehouse_query` and `warehouse_update`.

That is a hard cap of **five tools**: `kb_lookup`, `unit_convert`, `calculator`,
`warehouse_query`, `warehouse_update`. Only `fulfillment` is offered the two
environment operations; the other two families see three tools.

> **Correction (forward, no history rewrite).** Earlier revisions of this README
> named the two environment tools `inspect_inventory` and `commit_order`. Those
> names never existed in the code. The tools are, and always were,
> `warehouse_query` and `warehouse_update` — the names the committed schemas, the
> eight prompt candidates and every trace use. Nothing but this README was
> wrong, and the mistake is corrected here rather than by rewriting the pushed
> commit that contained it.

Error recovery is an evaluation and training axis applied across the families
(deterministic, seeded fault injection with held-out wording), not a fourth
family.

## The mandatory elicitation control

Round 1 of this lab measured that a one-sentence system prompt recovered 81.8%
of an observed SFT gain. Consequently, **every claim in this repository must
beat a best-of-eight frozen-prompt control**: eight system-prompt candidates
are committed by hash before prompt development, the winner is chosen on a
disjoint development split under a fixed search budget, and the primary
comparison is always the trained checkpoint versus the untouched base model
under that same winning prompt. A training leg that cannot beat the prompt-only
winner by the registered margin is dropped from the shipped path — that result
is reported, not hidden.

## Results

**No held-out result yet, and the results document says so on purpose.**
[docs/RESULTS.md](docs/RESULTS.md) is the study ledger — this README is not. It
already carries, verbatim and *before* any number exists, the study-status
paragraph, the saturated-secondaries paragraph, the per-gate template, the
underpowered-primary template and the descriptive-number prefix; every held-out
slot in it is **UNFILLED**, and a slot is filled only from the analyzer's emitted
output. It also holds the training-side observations in full, the
failure-category skeleton with complete denominators, and the corrections that
travel with the results. The analyzer semantics behind the templates are in
[docs/INTERPRETATION.md](docs/INTERPRETATION.md).

## Hardware and stack

This is a **single-card RTX A5000 (24 GB)** study. The card is pinned, not
assumed:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 make smoke
```

`EXPECT_GPU` has no off switch: it defaults to the registered card, and a study
stage additionally refuses a non-PCI device order, a non-registered index, a
non-exclusive card (< 23,500 MiB free) and a second physical GPU inside one run.
Every claim-bearing trace row and every GPU-ledger row carries `gpu_name`,
`gpu_uuid`, `cuda_visible_bytes`, `driver_version`, `engine_fingerprint`,
`enable_thinking_effective`, `run_id`, `git_sha`, `config_hash` and a UTC
timestamp; the **S19 HARDWARE-INTEGRITY** veto reads them, and the evaluator
refuses to append to a trace file another card produced.

Measured on the card at `gpu_memory_utilization 0.85` / `max_model_len 8192`
under vLLM 0.25.1: 25,282,805,760 CUDA-visible bytes (23.546 GiB), 8.68 GiB
checkpoint, 0.54 GiB CUDA-graph pool, 9.08 GiB (242,741-token) KV cache,
19.857 GiB used, 3.69 GiB free, 289.7 s engine startup, and **thinking mode ON by
default**. The registered engine contract deliberately runs slightly leaner —
every inference stage reads these from `configs/multifaceted.yaml` `engine:`,
which is the only copy:

| setting | value |
|---|---|
| `dtype` | `bfloat16` |
| `gpu_memory_utilization` | `0.80` (not 0.85, and not compensated up to 0.8725) |
| `max_model_len` | `8192` |
| `max_num_seqs` / `max_num_batched_tokens` | `8` / `8192` |
| `enforce_eager` | `false` |
| `enable_thinking` | `false`, explicitly, in the server default **and** every request |
| multimodal inputs | **rejected** (`--limit-mm-per-prompt image=0,video=0`) |

Thinking is the one that bites: this checkpoint thinks by default, offline
rejection sampling renders with thinking disabled, and an HTTP evaluator that
does not say so runs a different policy, spends the completion budget on
reasoning, and reads as "the model never committed an answer".

The **GRPO branch is not run**: the registered colocate configuration cannot
instantiate on 23.546 GiB. The `vllm_gpu_memory_utilization 0.24` carve is
5.651 GiB, which is smaller than vLLM's own 8.455 GiB colocated policy copy, and
the trainer's 9.420 GiB static footprint plus that copy is 17.875 GiB before any
KV cache, CUDA graphs or the 1.53 GiB of logits per 3,072-token completion. The
stage therefore records the **disposition** `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`
(a stage disposition, never a gate state), and the variance probe records
`NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT` — not "closed", which would claim the full
144-group probe ran and a binding gate failed. Microbatch 1, 2,048-token
completions, no-vLLM generation, quantization, offload and another card each
define a different treatment and are not substituted. RS-SFT is the sole trained
candidate, selected explicitly rather than by preferring whichever adapter
happens to exist.

The measured GPU-hour ceiling is **120 h**, an accounting envelope rather than a
threshold: the registered pre-calibration projection for the mandatory
7,800-episode evaluation is 75–99 h from the older *serial* evaluator rates, so
the earlier 36 h would have silently forced mandatory-sample shrinkage. The
throughput this tree actually measured is far higher — 177.6 charged GPU-minutes
per 3,600 tournament episodes, i.e. **~6.4 h projected for the 7,800**, with the
one-arm and paired-arm caveats spelled out in
[docs/AMENDED_REPLICATION_NOT_RUN.md](docs/AMENDED_REPLICATION_NOT_RUN.md) — a
projection from a measured rate, never a measured cost, never a reason to lower
the ceiling, and never enterable into the ledger. Mandatory samples may never
shrink; if the post-calibration projection does not fit, optional arms are cut in
the frozen order, and if the mandatory work still does not fit the run STOPS and
reports INCOMPLETE / INCONCLUSIVE.

The venv is hash-locked: `env/requirements.lock.txt` pins all 213 distributions
with `==` and SHA-256 hashes on CPython 3.12.13 (`.python-version`), and
`bash scripts/setup.sh --frozen` installs exactly that with `--require-hashes`.
`requirements-lock.txt` at the root is the older unhashed snapshot, kept for
continuity. These four pins are where the constraints genuinely bind:

| Package | Version | Why |
|---|---|---|
| `vllm` | 0.25.1 | TRL supports `<=0.25.1`; pins `torch==2.11.0+cu130` |
| `trl` | 1.9.2 | GRPO tool rollouts + `environment_factory` |
| `transformers` | 5.14.1 | vLLM needs `>=5.5.3` |
| `peft` | 0.20.0 | LoRA |

Three gotchas that will bite you, all found by running this stack:

1. **`CUDA_VISIBLE_DEVICES` does not index in `nvidia-smi` order.** CUDA
   defaults to `FASTEST_FIRST`; `nvidia-smi` lists by PCI bus ID. Export
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` (the Makefile does) and set `EXPECT_GPU` so a
   wrong pin fails loudly instead of OOMing obscurely.
2. **Qwen3.5 emits XML tool calls, not JSON.** vLLM parses them with
   `--tool-call-parser qwen3_coder`; a JSON-only parser silently finds zero
   calls, which reads as "the model never uses tools" rather than "the harness
   cannot see them". `chat.parse_tool_calls` handles the XML form, tolerates a
   missing closing tag on truncation, and casts string-typed arguments.
3. **Do not install `flash-linear-attention` / `causal-conv1d`.** The advertised
   Gated DeltaNet "fast path" (0.5.2 / 1.6.2.post1) segfaults the forward pass
   on torch 2.11.0+cu130 — exit 139, no traceback. The torch fallback is slower
   and correct; `scripts/setup.sh` deliberately leaves you with it.

vLLM startup can sit for minutes at a few hundred MB of VRAM compiling CUDA
graphs before allocating the KV cache. That is not a hang.

### The apparatus, identified by bytes

The study subject was registered by NAME — `Qwen/Qwen3.5-4B`, no revision — so an
upstream re-upload would have silently changed what was studied. Three tracked
records close that, all of them **additive and dated 2026-08-06, written after
the preregistration was finalized**; none of them amends it:

| record | pins | check |
|---|---|---|
| `env/model_revision.json` | Hub revision `851bf6e8…` and the SHA-256 of all 11 snapshot files, both weight shards included | `python scripts/record_model_revision.py verify` |
| `env/requirements.lock.txt` | 213 distributions, `==` + SHA-256, CPython 3.12.13 | `bash scripts/setup.sh --frozen` |
| `env/host_apparatus.json` | Ubuntu 22.04.1 / kernel 5.15.0-58 / glibc 2.35, driver 610.43.02, CUDA runtime 13.0, the registered A5000, and the segfault landmine above | `python scripts/record_host_apparatus.py check` |

Clean-clone requirements, disk and RAM figures, the HF cache and auth procedure,
and the exact command sequence — plus an honest list of what is still *not*
reproducible — are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Durability: what git does not protect

`.gitignore` excludes `out/` and `data/*`, which is where every GPU-hour of work
lands — the preflight evidence tree, the prompt-tournament rollouts, the
rejection-sampling shards, the accepted corpus, the SFT views, the adapter and
the run secret. Atomic writes and sealed shard receipts already make a *process*
crash survivable; they do nothing about losing this disk.

[`scripts/hf_artifacts.py`](scripts/hf_artifacts.py) copies those bytes to a
private, run-scoped Hugging Face dataset repository and commits `ARTIFACTS.json`
as the in-tree index — per file, the source path, a commit-pinned `hf://` URI,
the size, the SHA-256, and the producer receipt fields the artifact itself
carries. It digests, uploads, then re-digests, and refuses to record a file whose
digest moved, which is what makes it safe to run while a producer is appending.

The 32-byte run secret gets a **hash commitment committed before `L`** and an
encrypted off-host backup now; the plaintext is published only after the verdict
and checked against that commitment. Ordering, refusals (the held-out release is
never publishable) and what remains open are in
[docs/DURABILITY.md](docs/DURABILITY.md).

## Provenance

This repository previously hosted a single-family GSM8K tool-loop study,
including deliberately retained negative results. That study is summarized in
[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md), and every retired
artifact — code, corpus, evidence, verdict — remains recoverable at the
pre-cleanup commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f`
(`git show 9ebe44f:<path>`). History was not rewritten; the preregistration
receipts in `git log` remain valid.

## Background: read these before the report

The method here is an assembly of published techniques, not a new one. These are the
papers this repository assumes you know, in the order they matter to it. Each line says
why it matters *here* — the point is not a bibliography, it is the minimum reading that
makes the design choices legible.

**The training method itself**

- **STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., 2022
  ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)). The direct ancestor: build
  training data from the model's own correct outputs. Everything `make agentic` does in
  the `distill` stage is this idea with an exact verifier in place of an answer check.
- **Scaling Relationship on Learning Mathematical Reasoning with LLMs (RFT)** — Yuan et
  al., 2023 ([arXiv:2308.01825](https://arxiv.org/abs/2308.01825)). Rejection-sampling
  fine-tuning, named and measured. This is the recipe.
- **Llama 2** — Touvron et al., 2023
  ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)). §3 is where rejection sampling
  becomes a production post-training stage rather than a paper trick.
- **Qwen3 Technical Report** — Qwen team, 2025. Rejection-sampled SFT is their own
  Stage 3 ("thinking mode fusion"); this study re-applies the model family's method one
  level down, which is why the base model is already strong at it.

**The agent loop being trained**

- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al., 2022
  ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)). The think → act → observe loop
  every episode in `agentlab.suite.runtime` follows.
- **Toolformer** — Schick et al., 2023
  ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)). Self-supervised tool use;
  useful for why tool *selection* is learnable at all.

**The mechanics on a 24 GB card**

- **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021
  ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)). Why a 260 MB adapter can
  fine-tune a 4.7B model here when full-parameter training cannot.

**The wider post-training landscape this study deliberately does only part of**

- **Training language models to follow instructions with human feedback (InstructGPT)** —
  Ouyang et al., 2022 ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)). The
  canonical SFT → reward model → PPO pipeline. Read it to see what is *missing* here and
  why (`docs/EXPERIMENT_HISTORY.md` records the reward-model leg as excluded, at chance
  accuracy).
- **Direct Preference Optimization** — Rafailov et al., 2023
  ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)). The preference-tuning stage
  planned as a sibling study.
- **DeepSeekMath (GRPO)** — Shao et al., 2024
  ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)). The RL method registered here as
  `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`: colocated rollouts need a second full copy of the
  policy weights, which does not fit alongside the trainer on 23.5 GiB.

**Evaluation, and why this suite scores itself**

- **τ-bench** — Yao et al., 2024
  ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)). Agentic tool-use evaluation with
  a simulated user. Read it for the confounds this project refuses: an LLM judge and a
  model-simulated user put the measurement inside the thing being measured.
- **Berkeley Function Calling Leaderboard (BFCL), multi-turn** — Berkeley, 2024. State- and
  trace-based scoring of multi-turn tool use; the closest published relative of the
  verifier in `agentlab.suite.verify`.

**Why the thresholds were published before the results**

- **The garden of forking paths** — Gelman & Loken, 2013. Analysis decisions made after
  seeing data invalidate the inference even with no intent to deceive. This is the whole
  reason for `configs/preregistration_final.json` and the hash-pinned protocol.
- **False-Positive Psychology** — Simmons, Nelson & Simonsohn, 2011. Researcher degrees
  of freedom, and why the sample sizes, margins and stopping rules here are fixed in a
  committed file rather than chosen during the run.

If you read only two: **STaR** for what the training does, and **Gelman & Loken** for why
the study is shaped the way it is.
