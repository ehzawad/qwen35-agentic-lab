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
        distill chain verdict \
        verify verify-v trace-eval trace-grpo trace-view \
        eval-base eval-rssft eval-grpo merge serve clean

help:
	@echo "Qwen3.5-4B agentic post-training lab  (model=$(MODEL), GPU=A6000)"
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
	@echo "  chain       run the flagship RS-SFT -> GRPO -> paired eval chain"
	@echo "  verdict     after chain: run base@200 + the machine-checked verdict"
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

clean:
	rm -rf out/qwen35-*-lora out/qwen35-*-merged out/eval-*.json

# ---- verify + trace ---------------------------------------------------------
# `verify` is CPU-only and fast: it constructs every stage config, so a TRL
# upgrade that moves an argument fails here in seconds instead of after the
# model has loaded onto the card.
verify:
	$(PY) -m pytest tests/ -q

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

# ---- flagship: rejection-sampled SFT -> GRPO -> machine-checked verdict -----
distill:
	$(PY) -m agentlab.distill --model $(MODEL)

chain:
	bash scripts/run_distill_chain.sh

verdict:
	bash scripts/after_chain_verdict.sh
