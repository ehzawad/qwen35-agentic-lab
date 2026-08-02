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

It will not match a frontier model on correctness. It is not supposed to — the
point is to walk the whole path at a size where each stage finishes in minutes
to hours rather than weeks.

The pipeline is size-agnostic. Swap in a sibling for faster iteration:

```bash
make smoke MODEL=Qwen/Qwen3.5-0.8B     # 0.87B
make sft   MODEL=Qwen/Qwen3.5-2B       # 2.3B
make grpo  MODEL=Qwen/Qwen3.5-9B       # 9.7B (untested here; expect to lower --vllm-mem)
```

## The stack

Pinned where the constraints genuinely bind. This is an observed working
environment, not a lockfile — only vLLM and TRL are pinned exactly:

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

All three datasets come from the Hub and are cached after the first run;
subsequent runs are offline. The base model is a ~9 GB download.

Every stage except `grpo-env` has been run end to end at smoke scale. Those runs
prove the **plumbing** — data shape, tool parsing, trainer arguments, checkpoint
round-trip. They prove nothing about model quality: a handful of GRPO steps on a
handful of prompts is not evidence that a policy improved, and the reward-model
smoke finished at chance accuracy. Run `make verify` for the invariants, and
measure quality yourself before believing any of it.

```bash
make sft        # teach schema adherence — GRPO assumes this already works
make dpo        # RLHF leg: preferences straight into the policy loss
make grpo       # the interesting one
make eval-base  # then eval-sft / eval-grpo to see whether it moved
```

### Why SFT before GRPO

GRPO's advantage is the *spread* of reward within a sampled group. If every
rollout in a group scores the same, the advantage is zero and the step teaches
nothing — so the policy has to already produce the target behaviour *sometimes*
for RL to have anything to reinforce. Stage 1 buys that variance, and
`make grpo` continues the SFT adapter rather than restarting from base.

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

### Watch `completions/clipped_ratio`

The single most useful number in the GRPO logs. Thinking mode is on by default
and spends a large share of the completion budget before the tool call appears,
so a budget that looks generous truncates most rollouts. Truncated rollouts are
dropped by `mask_truncated_completions=True`, and a step where *every* completion
clipped teaches nothing at all:

```
clipped_ratio: 1     -> loss: 0, grad_norm: 0, frac_reward_zero_std: 1
clipped_ratio: 0     -> loss: -0.0048, grad_norm: 0.029, accuracy: 0.75
```

Both lines are from the same smoke run. If `clipped_ratio` sits high, raise
`--max-completion-length` before you touch the learning rate — you are not
looking at a policy that will not learn, you are looking at a policy whose
rollouts never finished.

## Three things that will bite you

All three were found by running this, not by reading docs.

**1. `CUDA_VISIBLE_DEVICES` does not index in `nvidia-smi` order.**
CUDA defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST`; `nvidia-smi` lists by PCI bus
ID. On a heterogeneous multi-GPU machine those orderings can disagree, so the
index you read off `nvidia-smi` selects a *different* card — and a run sized for
the bigger one then OOMs on the smaller one with no hint why. This cost real
debugging time here. The Makefile exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`; set
`EXPECT_GPU=<substring>` and `env.require_single_gpu()` will refuse to start on
the wrong card instead of failing later and obscurely.

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

## What to expect

Rough shape on a single 48 GB card, `Qwen3.5-4B`, LoRA r=32, `max_length=2048`,
gradient checkpointing, torch fallback (no fast-path kernels):

| | |
|---|---|
| SFT | seconds per step, tens of GB resident |
| GRPO with tools, vLLM colocate | tens of seconds per step; the card saturates |
| generation, thinking mode | low tens of tokens/s |
| eval through the agent loop | ~a minute-plus per problem |

Deliberately no precise figures here. The numbers this repo *did* measure were
taken before the fixes in `git log` — vision-tower LoRA, SFT loss landing on the
prompt, DPO slicing at the wrong offset — so quoting them now would be quoting
a different program. Run `make verify`, then measure your own.

Thinking mode is on by default and spends a large share of a small token budget
before the tool call appears, which is why the agent loop and eval default to
`max_new_tokens=1024`. At 512 the model gets truncated *mid-tool-call*, which
looks exactly like "the model refused to use the tool".

> **Eval is slow and not vLLM-backed.** `eval.py` drives the agent loop through
> `transformers.generate`, so the default `--n 100` is a long job rather than a
> coffee break. For anything beyond a sanity check, serve the checkpoint and
> evaluate against the endpoint. The in-process path is kept because it needs no
> server and shares exactly the parsing code the training stages use.

## Serving

```bash
make serve                              # base model
bash scripts/serve.sh out/qwen35-4b-merged   # a merged checkpoint
PORT=8077 MAXLEN=8192 bash scripts/serve.sh  # pick your own port
```

`--tool-call-parser qwen3_coder` turns Qwen's XML back into OpenAI-shaped
`tool_calls`, and `--reasoning-parser qwen3` splits the chain of thought into its
own `reasoning` field instead of leaving it in `content`. A verified response:

```json
"content": null,
"tool_calls": [{"type": "function", "function": {
    "name": "calculator", "arguments": "{\"expression\": \"4871 * 209 - 1337\"}"}}],
"reasoning": "The user wants me to calculate 4871 * 209 minus 1337...",
"finish_reason": "tool_calls"
```

So an OpenAI-compatible client gets structured tool calls with no Qwen-specific
parsing on your side — the XML handling in `chat.py` is only needed for the
in-process training and eval paths, which never go through the server.

> Startup is slower than it looks: the engine can sit at a few hundred MB of
> VRAM for minutes — compiling and capturing CUDA graphs — before it allocates
> the KV cache. That is not a hang.

## Hardware

Developed on a single **48 GB** card. Everything is LoRA, so the 4B policy at
bf16 plus vLLM colocate rollouts fits comfortably; smaller cards will need a
lower `--vllm-mem` and `--max-completion-length`, and the 0.8B/2B siblings.

Nothing forces a device. On a multi-GPU machine pick one explicitly, and set
`EXPECT_GPU` if you want a wrong pin to fail loudly instead of silently:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 EXPECT_GPU=A6000 make smoke
make gpu     # what is on the cards right now
```

For GRPO, vLLM runs *inside* the trainer process (`vllm_mode="colocate"`) and
takes `--vllm-mem` of the card. Drop it on contention, or pass `--no-vllm` to
generate with transformers instead — slower, but it removes vLLM from the
picture while you are debugging a reward function.

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
