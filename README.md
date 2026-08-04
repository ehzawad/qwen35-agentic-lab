# qwen35-agentic-lab

A verifiable multifaceted agent pipeline for **[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)**
on a single 48 GB GPU. The capability target is deliberately bounded and
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
make agentic     # the end-to-end pipeline (lands with the build phase)
```

`make agentic` will run: generate locked scenarios → establish the best-prompt
base control → sample verified trajectories → RS-SFT → conditional GRPO (only
when pre-SFT rollout groups retain outcome variance) → paired multifaceted
evaluation → select one winning configuration → serve smoke. Every stage gate
is preregistered and committed before held-out results exist; a harness BUG
vetoes any model-level verdict.

Helper targets (`make help`) exist for tests and debugging — `make verify`
(CPU test suite), `make smoke`, `make serve`, `make gpu` — but there is exactly
one supported end-to-end workflow, and alternate pipelines are out of scope.

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

Developed and evaluated on **one RTX A6000 (48 GB)**. On a multi-GPU machine
pin explicitly:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 EXPECT_GPU=A6000 make smoke
```

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
