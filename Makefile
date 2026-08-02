SHELL := /bin/bash

# CUDA_VISIBLE_DEVICES indexes in CUDA order, which defaults to FASTEST_FIRST and
# does NOT match nvidia-smi. On this box that inverts the cards, so PCI_BUS_ID is
# what makes "1" mean the A6000. env.require_single_gpu() asserts it stuck.
export CUDA_DEVICE_ORDER := PCI_BUS_ID
export CUDA_VISIBLE_DEVICES ?= 1
export PYTHONPATH := src
export TOKENIZERS_PARALLELISM := false

PY      := .venv/bin/python
MODEL   ?= Qwen/Qwen3.5-4B

# Derive the artifact slug from MODEL rather than hardcoding it, so
# `make sft MODEL=Qwen/Qwen3.5-2B` cannot overwrite the 4B adapter.
# Matches env.SLUG: basename, dots stripped, lowercased.
SLUG    := $(shell echo '$(notdir $(MODEL))' | tr -d '.' | tr 'A-Z' 'a-z')
SFT     := out/$(SLUG)-sft-lora
DPO     := out/$(SLUG)-dpo-lora
GRPO    := out/$(SLUG)-grpo-lora
MERGED  := out/$(SLUG)-merged

.PHONY: help setup gpu smoke data inspect sft dpo reward grpo grpo-env \
        eval-base eval-sft eval-grpo merge serve clean

help:
	@echo "Qwen3.5-4B agentic post-training lab  (model=$(MODEL), GPU=A6000)"
	@echo
	@echo "  setup       build the venv (vllm 0.25.1 -> torch 2.11, trl 1.9.2)"
	@echo "  gpu         show what is on the card right now"
	@echo "  smoke       stage 0: generation, tool schemas, real agent loop"
	@echo "  inspect     dump the module tree (LoRA target selection)"
	@echo "  data        build + preview all three datasets"
	@echo
	@echo "  sft         stage 1: LoRA SFT on tool calling (xlam-60k)"
	@echo "  reward      stage 2a: explicit reward model (classical RLHF)"
	@echo "  dpo         stage 2b: preference alignment (ultrafeedback)"
	@echo "  grpo        stage 3: GRPO with real tool execution (gsm8k)"
	@echo "  grpo-env    stage 3': GRPO with an environment-owned reward"
	@echo
	@echo "  eval-base   stage 4: agentic eval, base model"
	@echo "  eval-sft    stage 4: agentic eval, after SFT"
	@echo "  eval-grpo   stage 4: agentic eval, after GRPO"
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

# ---- stage 1 ----------------------------------------------------------------
sft:
	$(PY) -m agentlab.sft --model $(MODEL) --out $(SFT)

# ---- stage 2 ----------------------------------------------------------------
reward:
	$(PY) -m agentlab.reward

dpo: $(SFT)
	$(PY) -m agentlab.dpo --model $(MODEL) --adapter $(SFT) --out $(DPO)

# ---- stage 3 ----------------------------------------------------------------
# GRPO continues the SFT adapter: RL can only reinforce behaviour the policy
# already emits sometimes, and stage 1 is what buys that non-zero baseline.
grpo: $(SFT)
	$(PY) -m agentlab.grpo --model $(MODEL) --mode tools --adapter $(SFT) --out $(GRPO)

# The same stage without the chain, to see what RL alone does from base.
grpo-from-base:
	$(PY) -m agentlab.grpo --model $(MODEL) --mode tools --out $(GRPO)-frombase

grpo-env:
	$(PY) -m agentlab.grpo --model $(MODEL) --mode env --out out/$(SLUG)-grpo-env

# Guard: `make dpo` / `make grpo` depend on this, so the chain cannot silently
# fall back to base just because stage 1 was never run.
$(SFT):
	@echo "missing $(SFT) -- run 'make sft' first (stages chain through the adapter)"; exit 1

# ---- stage 4 ----------------------------------------------------------------
eval-base:
	$(PY) -m agentlab.eval --model $(MODEL) --tag base

eval-sft:
	$(PY) -m agentlab.eval --model $(MODEL) --adapter $(SFT) --tag sft

eval-grpo:
	$(PY) -m agentlab.eval --model $(MODEL) --adapter $(GRPO) --tag grpo

merge:
	$(PY) -m agentlab.merge --model $(MODEL) --adapter $(GRPO) --out $(MERGED)

serve:
	bash scripts/serve.sh $(MODEL)

clean:
	rm -rf out/qwen35-*-lora out/qwen35-*-merged out/eval-*.json
