# qwen35-agentic-lab

The full post-training path for a small agentic model, on one GPU, end to end:
**SFT → reward modelling → DPO → GRPO with real tool execution → eval → serve.**

Pretraining is deliberately out of scope. Everything else that a frontier lab
does after the base model exists is here, at a size that iterates in minutes
rather than weeks.

---

## The model

**[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)** — 4.7B params,
Apache-2.0, released 2026-03-02.

Chosen because it is the smallest checkpoint that still offers the *whole*
frontier feature set, which is what makes the path worth walking:

| Capability | Qwen3.5-4B |
|---|---|
| Thinking mode | yes, on by default (`<think>…</think>`) |
| Tool / function calling | yes, native in the chat template |
| Context | 262,144 native (→1.01M with YaRN) |
| Multimodal | text + image + video in one stack |
| License | Apache-2.0 |
| Architecture | hybrid: 24 Gated DeltaNet layers + 8 full-attention layers |

It will not match a frontier model on correctness. It is not supposed to — it
shows you the whole path at a size where each stage is a coffee break.

The pipeline is size-agnostic. Swap in a sibling for faster iteration:

```bash
make smoke MODEL=Qwen/Qwen3.5-0.8B     # 0.87B
make sft   MODEL=Qwen/Qwen3.5-2B       # 2.3B
make grpo  MODEL=Qwen/Qwen3.5-9B       # 9.7B, still fits the A6000 with LoRA
```

## The stack

Version-locked because the constraints genuinely bind:

| Package | Version | Why this one |
|---|---|---|
| `vllm` | 0.25.1 | TRL supports `>=0.17.0,<=0.25.1`. Pins `torch==2.11.0`. |
| `torch` | 2.11.0+cu130 | dictated by vLLM |
| `transformers` | 5.14.1 | vLLM needs `>=5.5.3` |
| `trl` | 1.9.2 | GRPO tool rollouts + `environment_factory` |
| `peft` | 0.20.0 | LoRA |
| `datasets` | 5.0.1 | needs the `Json()` feature type for the `tools` column |

> **Do not install `flash-linear-attention` / `causal-conv1d` on this stack.**
> See gotcha 3 below — they segfault the forward pass.

```bash
make setup     # builds .venv, ~10 min
make smoke     # stage 0 — prove the base model works before spending GPU-hours
```

## The path

```
stage 0   smoke        generation, tool schemas, a real agent loop
stage 1   sft          LoRA SFT on tool calling            xlam-function-calling-60k
stage 2a  reward       explicit reward model               ultrafeedback_binarized
stage 2b  dpo          preference alignment                ultrafeedback_binarized
stage 3   grpo         GRPO, tools executed for real       gsm8k (verifiable reward)
stage 3'  grpo-env     GRPO, environment-owned reward      BudgetEnv
stage 4   eval/merge/serve
```

All three datasets are already in the local HF cache, so every stage runs offline.

```bash
make sft        # teach schema adherence — GRPO assumes this already works
make dpo        # RLHF leg: preferences straight into the policy loss
make grpo       # the interesting one
make eval-base  # then eval-sft / eval-grpo to see whether it moved
```

### Why SFT before GRPO

GRPO can only reinforce behaviour the policy already emits sometimes. If the
model never produces a parseable tool call, every rollout in the group scores the
same zero, the advantage is zero, and the gradient is noise. Stage 1 buys the
non-zero baseline that stage 3 amplifies.

### What GRPO actually does here

`--mode tools` hands the real Python functions to `GRPOTrainer`. TRL runs the
multi-turn loop — sample → parse the call → **execute it** → feed the result back
→ continue — and the whole transcript receives the advantage. Reward is
verifiable, no judge model:

| Reward | Weight | Purpose |
|---|---|---|
| `correctness_reward` | 1.0 | boxed answer matches GSM8K ground truth |
| `tool_use_reward` | 0.3 | shaping: +calling, +calculator, **−erroring** |
| `format_reward` | 0.2 | the answer is extractable at all |

Shaping weights sit well below correctness on purpose: they should nudge toward
tool use, never pay the model to spam calls it does not need.

`--mode env` instead gives `GRPOTrainer` an `environment_factory`. The env holds
state across the rollout, exposes its public methods as tools, and scores itself
via `get_reward()` — the shape you need for a sandbox, a game, or a booking flow.

## Three things that will bite you

All three were found by running this, not by reading docs.

**1. `CUDA_VISIBLE_DEVICES` does not index in `nvidia-smi` order.**
CUDA defaults to `FASTEST_FIRST`; `nvidia-smi` lists by PCI bus ID. On this box
that *inverts* the two cards, so `CUDA_VISIBLE_DEVICES=1` lands on the 24 GB
A5000 while you think you are on the 48 GB A6000 — and a 48 GB-shaped run OOMs
with no hint why. The Makefile exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and
`env.require_single_gpu()` asserts the card is actually an A6000 and dies with
the fix if not.

**2. Qwen3.5 does not emit JSON tool calls.** It uses an XML form — the one vLLM
parses with `--tool-call-parser qwen3_coder`:

```
<tool_call>
<function=calculator>
<parameter=expression>
4871 * 209 - 1337
</parameter>
</function>
</tool_call>
```

A JSON parser silently finds zero calls, which during GRPO reads as "the model
never uses tools" rather than "the harness cannot see them". `chat.parse_tool_calls`
handles this form (and the JSON form as a fallback), tolerates a **missing
closing tag** so a completion truncated at `max_completion_length` still yields
its call, and casts arguments to the declared types — the wire format carries
every parameter as a string, so `value` arrives as `"26.2"` and an uncast float
would fail every numeric call the model got *right*.

**3. The Gated DeltaNet "fast path" segfaults this stack.** `transformers` prints

```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

which reads like an invitation. Installing `flash-linear-attention==0.5.2` +
`causal-conv1d==1.6.2.post1` makes the model **segfault in the forward pass**
(exit 139, core dumped) on torch 2.11.0+cu130 — no Python traceback, so from the
outside it looks like an OOM or a killed job. The torch fallback is correct and
is what `scripts/setup.sh` deliberately leaves you with. Minimal repro if you
want to retest on a newer torch: LoRA-wrap the model, one forward with
`labels=`, watch it die before it prints the loss.

## What to expect on one A6000

Measured on this box, 4B policy, LoRA r=32 (78.0M trainable, 1.69% of 4.6B),
`max_length=2048`, gradient checkpointing on, torch fallback (no fast path):

| | |
|---|---|
| SFT | ~8.0 s/step at 8 samples/step, ~23 GB resident |
| SFT, default `--n 4000` | ~250 steps, so budget ~35 min |
| base model generation | ~14-16 tok/s in thinking mode |

Thinking mode is on by default and spends a large share of a small token budget
before the tool call appears — which is why the agent loop and eval default to
`max_new_tokens=1024` rather than 512. At 512 the model reliably gets truncated
*mid-tool-call*, which looks exactly like "the model refused to use the tool".

## Hardware

Pinned to the **A6000** (48 GB, PCI `AF:00.0`). The box is shared — the A5000 is
left alone. Everything runs LoRA, so the 4B policy at bf16 plus vLLM colocate
rollouts sits comfortably inside the card.

```bash
make gpu       # what is on the cards right now
```

For GRPO, vLLM runs *inside* the trainer process (`vllm_mode="colocate"`) and
takes `--vllm-mem` of the card (0.25 by default). Drop it if you hit contention,
or pass `--no-vllm` to generate with transformers instead — slower, but it removes
vLLM from the picture when you are debugging a reward function.

## Layout

```
src/agentlab/
  env.py            model resolution, GPU pinning, auto-class probing
  tools.py          the tool suite — one definition serves SFT, GRPO and eval
  chat.py           chat template, tool-call parsing, the agent loop
  data.py           one builder per stage, each emitting TRL's exact expected type
  peft_cfg.py       shared LoRA config
  inspect_model.py  dump the module tree before choosing LoRA targets
  smoke.py          stage 0
  sft.py dpo.py reward.py grpo.py     stages 1-3
  eval.py merge.py                    stage 4
scripts/setup.sh scripts/serve.sh
```

`make help` lists every target.
