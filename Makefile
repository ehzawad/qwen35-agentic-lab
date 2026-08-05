SHELL := /bin/bash

# CUDA_VISIBLE_DEVICES indexes in CUDA order, which defaults to FASTEST_FIRST and
# need NOT match the order nvidia-smi prints. PCI_BUS_ID makes them agree, so an
# index read off nvidia-smi selects the card you actually meant.
export CUDA_DEVICE_ORDER ?= PCI_BUS_ID
# Deliberately not forced: a single-GPU machine should just work. On a multi-GPU
# box pick explicitly, e.g. `CUDA_VISIBLE_DEVICES=1 make smoke`, and optionally
# set EXPECT_GPU=A6000 to make a wrong pin fail loudly instead of silently.
export PYTHONPATH := src
export TOKENIZERS_PARALLELISM := false

PY      := .venv/bin/python
MODEL   ?= Qwen/Qwen3.5-4B

# Derive the artifact slug from MODEL rather than hardcoding it, so
# `make rs-sft MODEL=Qwen/Qwen3.5-2B` cannot overwrite the 4B adapter.
# Matches env.SLUG: basename, dots stripped, lowercased.
SLUG    := $(shell echo '$(notdir $(MODEL))' | tr -d '.' | tr 'A-Z' 'a-z')
RSSFT   := out/$(SLUG)-rssft-lora
GRPO    := out/$(SLUG)-rsgrpo-lora
MERGED  := out/$(SLUG)-merged

.PHONY: help setup gpu smoke data inspect rs-sft grpo grpo-from-base \
        agentic agentic-plan agentic-stages locks \
        suite validate-suite export-specs variance-report \
        distill \
        verify verify-v trace-eval trace-grpo trace-view \
        eval-base eval-rssft eval-grpo merge serve clean

help:
	@echo "Qwen3.5-4B agentic pipeline lab  (model=$(MODEL), GPU=A6000)"
	@echo
	@echo "  agentic       THE supported end-to-end pipeline (see README)"
	@echo "  agentic-plan  the same chain as a dry run: prints every command,"
	@echo "                touches no GPU"
	@echo "  agentic-stages  list the stage names (for --from/--only/--to)"
	@echo
	@echo "  setup       build the venv (vllm 0.25.1 -> torch 2.11, trl 1.9.2)"
	@echo "  gpu         show what is on the card right now"
	@echo "  smoke       stage 0: generation, tool schemas, real agent loop"
	@echo "  inspect     dump the module tree (LoRA target selection)"
	@echo "  data        build + preview all three datasets"
	@echo
	@echo "  distill     rejection-sample verified trajectories from the base model"
	@echo "  rs-sft      LoRA SFT on the verified distilled trajectories"
	@echo "  grpo        GRPO with real tool execution, continuing the RS-SFT adapter"
	@echo "  grpo-from-base  the same GRPO stage from base (comparison arm)"
	@echo
	@echo "  verify      CPU-only test suite (parsing, rewards, data, every stage config)"
	@echo "  trace-eval  run eval with tracing, then render the per-problem trace"
	@echo "  trace-view  render an existing trace file (FILE=..., LIMIT=...)"
	@echo "  verify-v    verbose test suite; trace-grpo: traced GRPO run; clean: rm artifacts"
	@echo ""
	@echo "  eval-base   agentic eval, base model"
	@echo "  eval-rssft  agentic eval, after RS-SFT"
	@echo "  eval-grpo   agentic eval, after GRPO"
	@echo "  merge       fold an adapter into the base weights"
	@echo "  serve       vLLM OpenAI server with tool calling"

setup:
	bash scripts/setup.sh

gpu:
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader

smoke:
	$(PY) -m agentlab.smoke --model $(MODEL)

inspect:
	$(PY) -m agentlab.inspect_model --model $(MODEL)

data:
	$(PY) -m agentlab.data

# ---- SFT on verified distilled trajectories ---------------------------------
rs-sft:
	$(PY) -m agentlab.sft --model $(MODEL) --distill-path data/distill.jsonl --out $(RSSFT)

# ---- GRPO --------------------------------------------------------------------
# GRPO continues the RS-SFT adapter: RL can only reinforce behaviour the policy
# already emits sometimes, and the SFT stage is what buys that non-zero baseline.
grpo: $(RSSFT)
	$(PY) -m agentlab.grpo --model $(MODEL) --adapter $(RSSFT) --out $(GRPO)

# The same stage without the chain, to see what RL alone does from base.
grpo-from-base:
	$(PY) -m agentlab.grpo --model $(MODEL) --out $(GRPO)-frombase

# Guard: `make grpo` depends on this, so the chain cannot silently fall back to
# base just because the SFT stage was never run.
$(RSSFT):
	@echo "missing $(RSSFT) -- run 'make distill' then 'make rs-sft' first"; exit 1

# ---- eval --------------------------------------------------------------------
eval-base:
	$(PY) -m agentlab.eval --model $(MODEL) --tag base

eval-rssft:
	$(PY) -m agentlab.eval --model $(MODEL) --adapter $(RSSFT) --tag rssft

eval-grpo:
	$(PY) -m agentlab.eval --model $(MODEL) --adapter $(GRPO) --tag grpo

merge:
	$(PY) -m agentlab.merge --model $(MODEL) --adapter $(GRPO) --out $(MERGED)

serve:
	bash scripts/serve.sh $(MODEL)

# Deliberately narrow: eval summaries and traces only. Checkpoints -- including
# the validated RS-SFT/RS-GRPO baselines under out/ -- are never glob-deleted.
clean:
	rm -f out/eval-*.json out/trace*.jsonl

# ---- verify + trace ---------------------------------------------------------
# `verify` is CPU-only and fast: it constructs every stage config, so a TRL
# upgrade that moves an argument fails here in seconds instead of after the
# model has loaded onto the card.
verify:
	$(PY) -m pytest tests/ -q

# ---- THE supported end-to-end pipeline ---------------------------------------
# One entry point, eleven resumable stages, every gate preregistered:
#   suite -> prompt -> baselock -> distill -> views -> sft -> probe
#         -> grpo? -> lock -> eval -> verdict
#
# Pin the card for a real run; the plan target needs no GPU at all:
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 EXPECT_GPU=A6000 make agentic
#
# Pass stage selection through ARGS, e.g. `make agentic ARGS="--from sft"` or
# `make agentic ARGS="--only verdict"`. Re-running after a kill resumes: each
# stage decides from artifacts on disk whether it is already done, and every GPU
# stage is a loop of short sub-invocations rather than one long blocking call.
ARGS ?=

agentic:
	bash scripts/run_multifaceted_chain.sh $(ARGS)

# Never touches a GPU: prints the exact command each pending stage would run.
agentic-plan:
	bash scripts/run_multifaceted_chain.sh --dry-run $(ARGS)

agentic-stages:
	@bash scripts/run_multifaceted_chain.sh --list

# S16/S18 receipts: the prompt winner and the trained checkpoint are locked
# BEFORE the held-out seed is revealed, and the reveal refuses to run first.
locks:
	@$(PY) scripts/agentic_locks.py status

# ---- multifaceted suite v1 (CPU only, deterministic) -------------------------
# The suite data is regenerated rather than committed: 11,320 specs are 44 MB and
# generation is byte-identical in ~3 s, so the seeds in configs/suite_v1.toml are
# the commitment and validate-suite proves the bytes follow from them. Only
# manifest.json and SHA256SUMS are committed.
suite:
	$(PY) scripts/generate_suite.py

validate-suite:
	$(PY) scripts/validate_suite.py

# Adapts the committed bundles into the certification-layer spec contract the
# evaluation arms consume, plus the merged train/dev/eval group manifests the
# split-leakage veto needs.
export-specs:
	$(PY) scripts/export_eval_specs.py

variance-report:
	$(PY) -m agentlab.variance report

verify-v:
	$(PY) -m pytest tests/ -v

TRACE ?= out/trace.jsonl
FILE  ?= $(TRACE)

# Run the eval with tracing on, then show what actually happened per problem.
trace-eval:
	AGENTLAB_TRACE=$(TRACE) $(PY) -m agentlab.eval --model $(MODEL) --n $(or $(N),8) --tag trace
	@$(MAKE) --no-print-directory trace-view FILE=$(TRACE)

trace-grpo:
	AGENTLAB_TRACE=$(TRACE) $(PY) -m agentlab.grpo --model $(MODEL) --adapter $(RSSFT) --out $(GRPO)

trace-view:
	@$(PY) -m agentlab.trace $(FILE) $(or $(LIMIT),5)

# ---- rejection sampling ------------------------------------------------------
distill:
	$(PY) -m agentlab.distill --model $(MODEL)
