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

Every stage has been run end to end, `grpo-env` at smoke scale (8 steps: the
environment's `reset()` supplies prompts, its methods become tools, its
`get_reward()` scores the rollouts -- all verified working; BudgetEnv itself is
trivially solved by a 4B model, reward 1.0 with zero variance, so it validates
the *mechanism*, not learning). Those runs
prove the **plumbing** — data shape, tool parsing, trainer arguments, checkpoint
round-trip. They prove nothing about model quality: a handful of GRPO steps on a
handful of prompts is not evidence that a policy improved, and the reward-model
smoke finished at chance accuracy. Run `make verify` for the invariants, and
measure quality yourself before believing any of it.

The **flagship path** — the one that produced the headline result:

```bash
make distill    # rejection-sample terminating trajectories from the base model
make chain      # RS-SFT -> GRPO -> paired eval at n=200 (unattended, ~9 h)
make verdict    # base@200 rerun + the machine-checked verdict
```

The original stages remain runnable, but know what you are running:

```bash
make sft        # the xlam stage -- reproduces the 16x DEGRADATION on purpose
make dpo        # RLHF leg: preferences straight into the policy loss
make grpo       # GRPO continuing whatever adapter make sft produced
```

## Results at a glance

Measured on one 48 GB GPU on held-out GSM8K with the calculator tools, thinking
off, the same generation budget, and the same evaluation harness. Base, RS-SFT
and RS-GRPO are evaluated on the **same 200 seeded problems** (paired); the
smaller xlam run is retained because it is the run that exposed the termination
failure.

| checkpoint | accuracy (95% CI) | calls/ep | runaway >10 | no box | tool-ok acc | tool err | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base `Qwen3.5-4B` | **0.810** [0.750, 0.858] | 2.8 | 4/200 | 28/200 | 0.810 | 0.055 | 200 |
| xlam single-turn SFT | **0.050** [0.009, 0.236] | 50.0 | 19/20 | 19/20 | 0.050 | 0.600 | 20 |
| rejection-sampled multi-turn SFT | **0.920** [0.874, 0.950] | 1.4 | 0/200 | 4/200 | 0.840 | 0.010 | 200 |
| RS-SFT + GRPO | **0.930** [0.886, 0.958] | 1.2 | 0/200 | 4/200 | **0.930** | 0.015 | 200 |

Paired McNemar on the shared 200 problems: **RS-SFT vs base +0.110, p < 0.001**
(b=5, c=27). **RS-GRPO vs RS-SFT +0.010, p = 0.804** -- a null, expected and
informative (see below).

> These results were machine-checked, not read off a table. The generated
> [comparison verdict](results/verdict.md) runs harness-sanity checks S0-S7
> *before* gates G1-G5; a harness BUG vetoes any model-level verdict, and all
> scoring uses a notation-tolerant normalizer applied uniformly to every
> checkpoint after a review found a right answer scored wrong on notation.
> Gate thresholds were committed (`d81644e`) during SFT training, before any
> evaluation of the new checkpoints existed — the git history is the receipt.
> The full evidence (eval summaries + traces) is tracked under
> [results/evidence/](results/evidence/); regenerate the verdict with
> `python -m agentlab.analyze --out-dir results/evidence --trace-dirs results/evidence`.
> The exact 1,275-trajectory corpus is tracked at `data/distill.jsonl`
> (sha256 `9aff1eae…d658f81`), and `requirements-lock.txt` freezes the
> verdict-producing environment.

Read together, the three results describe one behaviour. Single-turn xlam SFT
taught the model to call tools without teaching it to *stop* (16x degradation).
Rejection-sampled multi-turn SFT trained the missing conditional -- observe a
tool result, commit an answer, terminate -- and reached 0.920. GRPO then found
little to improve because most sampled groups no longer disagreed in reward;
its one attributable effect is restoring full tool use (0.905 -> 1.000), which
lifts tool-compliant accuracy to 0.930.

## Result: the pipeline runs, and this recipe makes the model worse

Measured on one A6000, held-out GSM8K, thinking off for both, identical harness:

| checkpoint | accuracy | tool calls / episode | runaway (>10 calls) | sec / episode | n |
|---|---|---|---|---|---|
| base `Qwen3.5-4B` | **0.800** | 3.3 | 3 / 50 | 24.8 | 50 |
| after SFT | **0.050** | **50.0** | **19 / 20** | 242.1 | 20 |

A **16x degradation**, and the mechanism is visible in the traces:

```
CALL calculator('20 * 40 + 80 * 30') -> 3200     <- correct, first try
CALL calculator('3200')              -> 3200     <- then 68 more times
CALL calculator('3200')              -> 3200
...
```

The model solves the problem on call one and then cannot stop. 19 of 20 episodes
never produce a final answer at all (`tool_use_rate` 1.000, `tool_error_rate`
0.600 as the loop degenerates into empty expressions).

**Why: `xlam-function-calling-60k` is single-turn function-calling data, not
agentic trajectories.** Every target is a bare tool call. Not one example shows a
model receiving a tool result and *concluding*. Train 4,000 of those with the
whole loss on the tool call and the model learns to emit tool calls — thoroughly,
including the part where a response never ends in prose.

Three things worth taking from this:

1. **A correctness fix made the outcome worse.** An earlier version put only ~12%
   of the loss on the tool call (a bug). Fixing that to 100% and scaling 250 -> 4000
   examples made training much better at learning the wrong lesson. The earlier
   GRPO run scored 0.25-1.00 *because the SFT was too weak to do damage*. The bug
   was masking the data problem.
2. **GRPO could not rescue it and could not even measure it.** With no rollout
   ever producing a boxed answer, `correctness_reward` was identically 0 across
   every step, so the advantage had no variance and RL had nothing to reinforce.
3. **Loss going down is not the model getting better.** SFT ended at
   train_loss 0.0221, eval_loss 0.0125, token accuracy 0.9971 — a textbook clean
   fit, on an objective that was not the one that mattered.

The pipeline is not what failed here. Every stage did exactly what it was asked.

### What would actually be needed

Multi-turn trajectories where an observation changes the next action *and* the
episode terminates. When this section was first written that was out of scope;
the follow-up below then implemented exactly that (rejection-sampled from the
base model) and it worked. Process-level reward remains genuinely out of scope —
a terminal-only signal on a long-horizon task is almost no signal at all, and
nothing here addresses it.

### Follow-up: terminating trajectories recover the headroom

The fix the diagnosis implies: generate multi-turn trajectories **with the base
model itself** against the real tools (6,000 rollouts over 1,500 train problems,
disjoint from evaluation), keep only those that end in a *correct committed
answer* (plus: >=1 tool call, <=6 calls, <=128-token final, no stray reasoning
tags), and fine-tune on the **terminal turn only** -- everything before it is
masked context. 1,275 accepted trajectories, ~80 minutes of SFT.

Result: 0.920 vs base 0.810 on the paired 200 (p < 0.001), with the failure mode
the corpus was filtered against essentially gone -- no-box failures 14% -> 2%,
tool errors 0.055 -> 0.010, zero runaway episodes.

**What this does not show:** better arithmetic. Base's errors were dominated by
running out of budget before `\boxed{}`; the gain is termination and formatting
headroom, which is exactly what an outcome filter can capture and no more. The
corpus is the model teaching itself its own successes -- calls/episode dropped
to 1.4 because accepted trajectories skew short, a selection effect, not a
demonstration that fewer calls are better.

### The zero-training control arm: most of the gain was elicitation

Motivated by the Spurious-Rewards line of work (Qwen models' post-training gains
are often elicitation of latent behaviour) and by the fact that Qwen small
models are logit-distilled from RLVR-trained flagships, we ran the control the
headline deserved: **base model, no training, one added system sentence** --
"Keep your final response under 60 words: state the result and the boxed
answer, nothing more" -- same 200 paired problems, same harness. The
interpretation function was committed before the number existed
(regime boundaries derived from exact McNemar counts; see git history).

> **What this does and does not show:** The no-training concise-prompt control
> scored **180/200 (0.900)**, recovering **81.8%** of the observed
> base-to-RS-SFT gain. It significantly exceeded base (b=5, c=23, p=0.0007) and
> could not be distinguished from RS-SFT (b=11, c=7, p=0.481). The headline
> accuracy gain therefore cannot be attributed uniquely to SFT: on this single
> stochastic draw, an explicit brevity-and-boxing instruction reproduced the
> accuracy. The behavioural anatomy also landed on the trained side of every
> pre-registered guardrail (calls/ep 1.56, no-box 4/200, tool-error episodes 3,
> correct-with-tool 172). The fixed 27-problem multi-call tail stayed
> unresolved (control 23/27 vs RS-SFT 26/27, p=0.375). This does not show
> better arithmetic or knowledge absent from the base.

So the honest final reading of the whole experiment: **single-turn xlam SFT
destroys termination; outcome-filtered self-distillation restores it; and most
of what it restores is behaviour a one-sentence prompt can elicit from this
model family anyway.** What training buys over the prompt is internalisation
(no prompt engineering required downstream), a still-unresolved edge on the
multi-call tail, and -- per the Qwen3 technical report -- exactly what Qwen's
own Stage-3 "thinking mode fusion" buys: rejection-sampled SFT is their method,
re-applied one level down. This is a stronger teaching result than the
uncontrolled version, and a humbler one.

### GRPO after outcome-filtered SFT: an informative null

GRPO on a disjoint problem slice, continuing the RS-SFT adapter, 300 steps under
valid conditions (thinking off in the rollout renderer, committed-answer-only
rewards, parity-wrapped tools): held-out accuracy 0.930 vs 0.920, p = 0.804.
During training, **56% of steps (168/300) had zero within-group reward variance** -- all
eight samples scored identically, so the advantage was zero and those steps
taught nothing. RS-SFT had already captured the available headroom; RL had
almost nothing left to grip on this task. Its one measurable effect is real and
small: the tool_use_reward pushed tool use from 0.905 back to 1.000.

Scope: one task, one model, one seed, one budget. None of this licenses claims
beyond that.

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

### The completion budget is shared by the whole rollout

`max_completion_length` bounds the **entire** multi-turn completion — every
assistant turn plus every tool result, not one turn
(`grpo_trainer.py:2043`). With a reasoning trace per turn and four tool
round-trips, a budget that looks generous per turn is exhausted before the final
answer, and the rollout is scored as if the model simply never answered.

Three defaults address it, and they are a set rather than a menu:

- **Thinking is off for GRPO rollouts.** Stage 1 trains tool calls with no
  reasoning block, so sampling stage 3 with one is a format mismatch — and the
  generated reasoning competes for the budget with the answer that earns reward.
  GSM8K plus a calculator does not need hidden reasoning.
- **`mask_truncated_completions=False`.** DAPO recommends masking for a long
  *answer* clipped by an infrastructure limit, where correctness is unknown.
  A clip here means the tool loop ran out of an intentional task budget without
  producing the required answer. That is a failure to train against, not data to
  delete.
- **`format_reward` is asymmetric** (`+0.1` / `−0.5`). Once truncated rollouts
  reach the loss this carries weight: with a symmetric ±0.1, an unfinished
  rollout that had called the calculator scored **+0.07** while one that
  finished with the *wrong* answer scored **+0.02** — running out of budget paid
  better than committing. Not finishing must be the worst outcome.

A `ClipGuard` callback then **aborts** the run when the last 8 logged steps
average ≥50% clipped, rather than letting it burn GPU-hours producing weak
gradients behind an unremarkable loss curve. Tune with `--clip-window` /
`--clip-threshold` (`1.0` disables).

> One limit worth knowing: TRL calls a completion clipped when its last token is
> not EOS/PAD, so a rollout that stops on a tool call at the iteration limit can
> be counted as finished. Read the guard alongside `format_reward` and a trace,
> not on its own.

GRPO hard-disables thinking in the rollout renderer (`GRPOConfig
chat_template_kwargs`; row-level kwargs are ignored by TRL). If you want
thinking during GRPO, edit that config line *and* raise
`--max-completion-length` to 2048 — the measured GSM8K prompt is ~790 tokens
(max 887), so 2048 still fits the default `--vllm-max-len 4096`.

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
scripts/
  setup.sh serve.sh                    environment + OpenAI-compatible serving
  run_distill_chain.sh                 the flagship chain (make chain)
  after_chain_verdict.sh               base@200 + machine-checked verdict (make verdict)
  archive_xlam_comparison.sh           the failed xlam experiment, kept as record
```

`make help` lists the targets.
