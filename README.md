# qwen35-agentic-lab

A verifiable multifaceted agent pipeline for **[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)**
on **one RTX A5000 (24 GB)**. The capability target is deliberately bounded and
machine-checkable: multi-tool composition, dependency depth up to 8 required
calls, stateful constraints with irreversible commitments, and recovery from
injected tool failures — all scored by exact verifiers, never by an LLM judge.

**Status: pipeline under evaluation — machine verdict pending.** No capability
claim is made here until the preregistered gates have run. The training legs
(distilled SFT, conditional GRPO) are candidates, not predetermined winners:
if training does not beat the locked elicitation control without harming clean
performance, the shipped pipeline is the prompted base model.

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

## Task families and the tool cap

Three task families, no more:

1. **Compositional quantitative tasks** — require `kb_lookup → unit_convert →
   calculator`, with facts unavailable in the prompt and exact independently
   computed answers.
2. **Synthetic multi-hop knowledge-graph tasks** — each lookup reveals the key
   for the next; dependency depth generated at 2, 4, and 8+; the verifier
   confirms the minimal dependency path.
3. **Constrained stateful procurement** — inventory, units, budget, and
   irreversible commitments, adding exactly two environment tools:
   `inspect_inventory` and `commit_order`.

That is a hard cap of **five tools**: the three existing ones plus the two
environment operations. Error recovery is an evaluation and training axis
applied across the families (deterministic, seeded fault injection with held-out
wording), not a fourth family.

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

Pipeline under evaluation — machine verdict pending. This section will carry
only machine-checked numbers produced by the preregistered analyzer, with the
paired base-model arm and the elicitation control on every row.

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
threshold: the mandatory 7,800-episode evaluation projects 75–99 h from directly
measured evaluator rates, so the earlier 36 h would have silently forced
mandatory-sample shrinkage. Mandatory samples may never shrink; if the
post-calibration projection does not fit, optional arms are cut in the frozen
order, and if the mandatory work still does not fit the run STOPS and reports
INCOMPLETE / INCONCLUSIVE.

The venv is pinned where the constraints genuinely bind (see
`requirements-lock.txt` for the frozen snapshot):

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

## Provenance

This repository previously hosted a single-family GSM8K tool-loop study,
including deliberately retained negative results. That study is summarized in
[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md), and every retired
artifact — code, corpus, evidence, verdict — remains recoverable at the
pre-cleanup commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f`
(`git show 9ebe44f:<path>`). History was not rewritten; the preregistration
receipts in `git log` remain valid.
