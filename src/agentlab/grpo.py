"""Stage 3 -- agentic RL with GRPO, where the model really calls the tools.

Two modes, both exercising TRL v1's agentic rollout:

  --mode tools   Pass our Python functions straight to GRPOTrainer. TRL runs the
                 multi-turn loop: sample -> parse tool call -> execute -> feed the
                 result back -> continue, and the whole transcript is what gets
                 the advantage. Reward is verifiable (GSM8K ground truth).

  --mode env     Environment-owned rewards. The environment holds state across a
                 rollout, exposes its public methods as tools, and scores itself
                 via get_reward(). This is the shape you need for anything
                 stateful -- a sandbox, a game, a multi-step booking flow.

GRPO needs no value network: it samples a group of `num_generations` completions
per prompt and uses the spread of rewards within that group as the advantage.
"""

from __future__ import annotations

import argparse
import random
import re

from . import env, trace
from .data import build_grpo
from .peft_cfg import describe, policy_and_peft
from .tools import TOOLS, trl_tools

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _answer_text(completion) -> str:
    """Only what the model *committed to the user*, for scoring correctness.

    Deliberately narrower than `_as_text`. That one renders tool calls and tool
    results into the text for tracing, which makes it a reward-hacking surface:
    `calculator(expression="\\boxed{42}")` would put the ground truth into the
    scored string without the model ever answering, and a box appearing in a tool
    RESULT would score too. Correctness and format therefore read assistant
    `content` only -- never arguments, never tool output, never
    `reasoning_content` (which is text the model never committed past </think>).
    """
    if isinstance(completion, str):
        return completion
    parts = []
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") in (None, "assistant") and msg.get("content"):
            parts.append(str(msg["content"]))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# completion helpers
# --------------------------------------------------------------------------

def _as_text(completion) -> str:
    """Flatten a completion to text.

    TRL hands a reward function either a plain string or a list of messages,
    decided by `is_conversational(prompt)` and NOT by whether tools were passed
    (grpo_trainer.py:2176-2189), so every reward function normalises here first.

    Deliberately does not read `reasoning_content`. When a rollout never closes
    its `</think>`, TRL's `parse_response` puts the whole completion under that
    key; folding it in would let a `\boxed{}` written inside the chain of thought
    collect `correctness_reward` without the model ever committing to an answer.
    Excluding it is the reward-hacking guard, not an oversight.
    """
    if isinstance(completion, str):
        return completion
    parts = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("content"):
            parts.append(str(msg["content"]))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", tc)
            parts.append(f"<tool_call>{fn.get('name')} {fn.get('arguments')}</tool_call>")
    return "\n".join(parts)


def _tool_names(completion) -> list[str]:
    """Names of the tools actually invoked in this rollout.

    Both shapes TRL 1.9.2 can hand a reward function are handled, and the text
    forms delegate to `chat.parse_tool_calls` rather than re-implementing the
    wire format. The previous regex here matched only `_as_text`'s own synthetic
    render -- neither real Qwen XML nor the JSON form -- so a genuine tool call
    arriving as text counted as no call at all and scored 0.0.

    Which shape you get is decided purely by `is_conversational(prompt)`
    (grpo_trainer.py:2176-2189), NOT by whether `tools=` was passed:
      * non-conversational prompt      -> list[str], the raw decoded completion
      * conversational, tools set      -> list[dict] with structured `tool_calls`
      * conversational, no tools       -> list[dict] whose `content` is raw text,
                                          which may still contain tool-call XML
    """
    from .chat import parse_tool_calls

    if isinstance(completion, str):
        return [c["name"] for c in parse_tool_calls(completion)]

    names = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        structured = msg.get("tool_calls") or []
        for tc in structured:
            fn = tc.get("function", tc)
            if fn.get("name"):
                names.append(fn["name"])
        # No structured calls: the model may still have emitted XML into content
        # (the conversational-without-tools branch). Deliberately does NOT read
        # `reasoning_content` -- see _as_text.
        if not structured and msg.get("content"):
            names.extend(c["name"] for c in parse_tool_calls(str(msg["content"])))
    return names


def _numeric(s: str):
    try:
        return float(str(s).strip().replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


# A failed tool reaches us in two different shapes and both must be penalised:
#   1. our own tools return the string "error: ..."  (tools.call_tool)
#   2. a tool that RAISES is wrapped by TRL itself as {"error": str(e)}, which
#      stringifies to "{'error': '...'}" -- no "error:" substring anywhere.
# TRL calls the raw functions, not call_tool, so a bad-argument TypeError takes
# path 2. Matching only "error:" paid +0.2 for a call that had crashed.
# (trl/trainer/grpo_trainer.py: `result = {"error": str(e)}`)
_TOOL_ERROR_RE = re.compile(r"^\s*error:|['\"]error['\"]\s*:", re.IGNORECASE)

# Same pattern, MULTILINE, for a whole transcript arriving as one string -- there
# the failure sits on some interior line, so an ^ anchored to the start of the
# string never matches. Kept separate on purpose: adding MULTILINE to the shared
# regex would widen the per-message path, where "42\nerror: none" would newly
# count as a failure.
_TOOL_ERROR_TEXT_RE = re.compile(
    r"^\s*error:|['\"]error['\"]\s*:", re.IGNORECASE | re.MULTILINE
)


def _tool_errored(completion) -> bool:
    """True when any tool result in this rollout represents a failure."""
    if isinstance(completion, str):
        return bool(_TOOL_ERROR_TEXT_RE.search(completion))
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if _TOOL_ERROR_RE.search(str(msg.get("content", ""))):
                return True
    return False


# --------------------------------------------------------------------------
# reward tracing
# --------------------------------------------------------------------------

# TRL calls each reward function separately on the same batch, so no single one
# of them can see the full breakdown. Each deposits its scores here and the last
# to arrive writes one record per rollout carrying every component. Keyed by
# batch identity so concurrent batches cannot blend into each other.
REWARD_NAMES = ("correctness", "tool_use", "format")
_REWARD_BUF: dict[int, dict] = {}


# Cap the buffer. An id() key is only valid while the caller keeps the list
# alive; a caller that never completes a full triple would otherwise leak an
# entry per batch forever.
_REWARD_BUF_MAX = 64


def _record(name: str, completions, scores, ground_truth=None) -> None:
    """Accumulate one reward component; the last of the triple writes the record.

    Keyed on id(completions), which requires the CALLER to pass the SAME list
    object to all three reward functions. TRL does. A caller that builds a fresh
    list per function gets no record at all -- and worse, sporadic ones when
    CPython recycles a freed id, mixing scores across rollouts. eval.py scores
    through a single shared list for exactly this reason.
    """
    if not trace.enabled():
        return
    if len(_REWARD_BUF) > _REWARD_BUF_MAX:
        _REWARD_BUF.clear()  # stale partials from a caller that never completed
    key = id(completions)
    slot = _REWARD_BUF.setdefault(key, {"scores": {}, "gt": None})
    slot["scores"][name] = list(scores)
    if ground_truth is not None:
        slot["gt"] = list(ground_truth)
    if set(slot["scores"]) != set(REWARD_NAMES):
        return

    gt = slot["gt"] or [None] * len(completions)
    for i, comp in enumerate(completions):
        text = _as_text(comp)
        rewards = {n: slot["scores"][n][i] for n in REWARD_NAMES}
        trace.emit(
            "rollout",
            ground_truth=gt[i],
            correct=bool(slot["scores"]["correctness"][i]),
            boxed=(_BOXED_RE.findall(text) or [None])[-1],
            tools_called=_tool_names(comp),
            tool_error=_tool_errored(comp),
            rewards=rewards,
            total=round(sum(w * rewards[n] for n, w in zip(REWARD_NAMES, REWARD_WEIGHTS)), 4),
            completion=text[-1500:],
        )
    _REWARD_BUF.pop(key, None)


# Single source of truth: grpo.py passes these to GRPOConfig and the trace uses
# them for the weighted total, so the two cannot drift apart.
REWARD_WEIGHTS = (1.0, 0.3, 0.2)


# --------------------------------------------------------------------------
# reward functions
# --------------------------------------------------------------------------

def correctness_reward(completions, ground_truth, **kwargs) -> list[float]:
    """1.0 when the boxed answer matches the reference, else 0.0.

    This is the only reward that carries task signal; the others are shaping.
    """
    log_metric = kwargs.get("log_metric")
    out = []
    for comp, gt in zip(completions, ground_truth):
        text = _answer_text(comp)
        hits = _BOXED_RE.findall(text)
        if not hits:
            out.append(0.0)
            continue
        got, want = _numeric(hits[-1]), _numeric(gt)
        if got is not None and want is not None:
            out.append(1.0 if abs(got - want) < 1e-4 else 0.0)
        else:
            out.append(1.0 if hits[-1].strip() == str(gt).strip() else 0.0)
    if log_metric:
        log_metric("accuracy", sum(out) / max(len(out), 1))

    _record("correctness", completions, out, ground_truth=ground_truth)
    return out


def tool_use_reward(completions, **kwargs) -> list[float]:
    """Shaping: reward calling the calculator, penalise a call that errored.

    Deliberately small relative to correctness -- it should nudge the policy toward
    using tools, never pay it to spam calls it does not need.
    """
    log_metric = kwargs.get("log_metric")
    out, used, errored = [], 0, 0
    for comp in completions:
        names = _tool_names(comp)
        score = 0.0
        if names:
            used += 1
            score += 0.2
        if "calculator" in names:
            score += 0.1
        # A failed tool means the model built a bad call. The penalty must
        # outweigh the +0.2/+0.1 it just earned, or crashing still pays.
        if _tool_errored(comp):
            errored += 1
            score -= 0.4
        out.append(score)
    if log_metric:
        log_metric("tool_use_rate", used / max(len(completions), 1))
        log_metric("tool_error_rate", errored / max(len(completions), 1))
    _record("tool_use", completions, out)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    """Shaping: the answer must be extractable at all.

    Asymmetric on purpose, and the asymmetry is load-bearing now that truncated
    rollouts are no longer masked out. With a symmetric +/-0.1 an unfinished
    rollout that had called the calculator scored
        0*1.0 + 0.3*0.3 + (-0.1)*0.2 = +0.07
    while a rollout that finished with the WRONG answer and no tool scored
        0*1.0 + 0*0.3   + (+0.1)*0.2 = +0.02
    i.e. running out of budget paid better than committing to an answer. Failing
    to produce the terminal answer has to be the worst outcome, so the miss is
    penalised hard enough to outweigh any shaping the rollout collected.
    """
    out = [0.1 if _BOXED_RE.search(_answer_text(c)) else -0.5 for c in completions]
    _record("format", completions, out)
    return out


# --------------------------------------------------------------------------
# truncation guard
# --------------------------------------------------------------------------

class ClipGuard:
    """Abort when most rollouts are running out of completion budget.

    `max_completion_length` bounds the WHOLE multi-turn completion -- every
    assistant turn plus every tool result (grpo_trainer.py:2043) -- so a budget
    that looks generous per turn can still be exhausted before the final answer
    appears. Half the group failing to terminate is not a run worth continuing,
    and without this it fails silently: the wasted rollouts simply produce
    weaker gradients while the loss curve looks unremarkable.

    Caveat worth knowing: TRL defines clipped as "last token is not EOS/PAD", so
    a rollout that stops on a tool call at the iteration limit can be counted as
    finished. Read this alongside the format reward and a trace, not alone.
    """

    KEY = "completions/clipped_ratio"

    def __init__(self, window: int = 8, threshold: float = 0.5):
        from collections import deque

        self.window = window
        self.threshold = threshold
        self.seen = deque(maxlen=window)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.KEY not in logs:
            return  # e.g. the final runtime summary
        self.seen.append(float(logs[self.KEY]))
        if len(self.seen) < self.window:
            return
        mean = sum(self.seen) / len(self.seen)
        if mean >= self.threshold:
            raise RuntimeError(
                f"{mean:.0%} of the last {self.window} logged steps hit "
                f"max_completion_length. The budget is exhausted before the final "
                f"answer, so those rollouts teach little or nothing.\n"
                f"  Raise --max-completion-length (currently bounded by "
                f"--vllm-max-len minus the prompt), lower --max-tool-iters, or "
                f"keep thinking disabled for GRPO."
            )


# --------------------------------------------------------------------------
# stateful environment (--mode env)
# --------------------------------------------------------------------------

class BudgetEnv:
    """A small stateful task: hit a target total by spending through a tool.

    Every public method becomes a tool the model can call, `reset` starts a
    rollout, and `get_reward` scores it from internal state -- so correctness is
    defined by the environment rather than by parsing the text.
    """

    def reset(self, **kwargs) -> str:
        self.target = random.randint(20, 200)
        self.spent = 0
        self.calls = 0
        return (
            f"You have a budget tracker starting at 0. Spend so the total lands "
            f"exactly on {self.target}. Use the `spend` tool, then `check` to "
            f"confirm. You may make at most 5 spends."
        )

    def spend(self, amount: int) -> str:
        """
        Adds an amount to the running total.

        Args:
            amount: The amount to add to the running total.

        Returns:
            The new running total after spending.
        """
        if self.calls >= 5:
            return "error: no spends remaining"
        self.calls += 1
        self.spent += amount
        return f"total is now {self.spent}"

    def check(self) -> str:
        """
        Reports the running total and how far it is from the target.

        Returns:
            The current total and the remaining distance to the target.
        """
        return f"total {self.spent}, target {self.target}, difference {self.target - self.spent}"

    def get_reward(self) -> float:
        """Exact hit scores 1.0, near misses decay, overshoot scores 0."""
        gap = abs(self.target - self.spent)
        if gap == 0:
            return 1.0
        return max(0.0, 1.0 - gap / max(self.target, 1))


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--adapter", default=None, help="SFT adapter to start the policy from")
    ap.add_argument("--mode", choices=["tools", "env"], default="tools")
    ap.add_argument("--n", type=int, default=1000, help="prompts, --mode tools only")
    ap.add_argument("--grpo-offset", type=int, default=0,
                    help="start index into the shuffled train split; must NOT overlap the "
                         "slice used for distillation, or RL trains on problems already SFT'd "
                         "and within-group reward variance collapses")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="--mode env only: required, since a procedural env has no dataset length")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 disables the ref model")
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--max-tool-iters", type=int, default=4,
                    help="tool round-trips allowed per rollout")
    ap.add_argument("--clip-window", type=int, default=8,
                    help="steps averaged by the truncation guard")
    ap.add_argument("--clip-threshold", type=float, default=0.5,
                    help="mean clipped_ratio over the window that aborts the run; 1.0 disables")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--loss-type", default="dapo", choices=["grpo", "dr_grpo", "dapo", "sapo"])
    ap.add_argument("--no-vllm", action="store_true", help="generate with transformers instead of vLLM")
    ap.add_argument("--vllm-mem", type=float, default=0.25,
                    help="fraction of the card vLLM may take in colocate mode")
    ap.add_argument("--vllm-max-len", type=int, default=4096,
                    help="rollout context cap; the model's native 262144 will not fit a colocated KV cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env.require_single_gpu()
    out_dir = args.out or str(env.GRPO_DIR)

    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        beta=args.beta,
        loss_type=args.loss_type,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        # Truncated rollouts have no terminal reward; letting them into the loss
        # teaches the policy that running out of budget is fine.
        # False on purpose. DAPO recommends masking for a long *answer* clipped
        # by an infrastructure limit, where correctness is unknown. Here a clip
        # means the tool loop ran out of an intentional task budget without
        # producing the required final answer -- that is a failure the policy
        # should be trained against, not data to delete.
        mask_truncated_completions=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        # Without this vLLM sizes the KV cache for the model's *native* 262144
        # context and refuses to start: 8.06 GiB needed vs 3.79 GiB available at
        # 0.30 utilization. A rollout here is a prompt plus a bounded completion,
        # nowhere near 262K, so cap it rather than surrendering the card to a KV
        # cache we will never fill.
        vllm_max_model_length=args.vllm_max_len,
        # reward_weights lives on the config, not on GRPOTrainer -- `tools` and
        # `environment_factory` are the trainer kwargs, these are not.
        reward_weights=list(REWARD_WEIGHTS) if args.mode == "tools" else None,
        # How many tool round-trips one rollout may take before it is cut off.
        max_tool_calling_iterations=args.max_tool_iters,
        # MUST be set here, not on the dataset rows. TRL stores only
        # `self.chat_template_kwargs = args.chat_template_kwargs or {}` and uses
        # that global dict when rendering rollout prompts (grpo_trainer.py:742,
        # :1758). Row-level chat_template_kwargs reach the reward functions but
        # never the renderer, so the previous run sampled with thinking ON while
        # the SFT adapter had been trained with it OFF -- a grammar mismatch, and
        # reasoning then ate the shared completion budget every tool round.
        chat_template_kwargs={"enable_thinking": False},
    )

    # --adapter is what chains stage 1 into stage 3. Without it the policy starts
    # from raw base and the "SFT -> GRPO" path is decorative.
    policy, peft = policy_and_peft(args.model, args.adapter, rank=args.rank)
    common = dict(
        model=policy,
        args=cfg,
        peft_config=peft,
    )

    if args.mode == "tools":
        ds = build_grpo(n=args.n, offset=args.grpo_offset)
        print(f"[data] {len(ds)} prompts, cols={ds.column_names}")
        print(f"[tools] {', '.join(t.__name__ for t in TOOLS)}")
        trainer = GRPOTrainer(
            train_dataset=ds,
            reward_funcs=[correctness_reward, tool_use_reward, format_reward],
            tools=trl_tools(),
            **common,
        )
    else:
        # No train_dataset at all: TRL allows it to be omitted when a single
        # environment_factory owns the data, because BudgetEnv.reset() generates
        # the prompt procedurally. In exchange max_steps must be set, since there
        # is no dataset length to infer the schedule from.
        cfg.max_steps = args.max_steps
        print(f"[data] procedural: {args.max_steps} steps against BudgetEnv "
              f"(environment-owned reward, no dataset)")
        trainer = GRPOTrainer(
            environment_factory=BudgetEnv,
            **common,
        )

    from transformers import TrainerCallback

    class _Guard(TrainerCallback, ClipGuard):
        pass

    trainer.add_callback(_Guard(window=args.clip_window, threshold=args.clip_threshold))

    describe(trainer.model)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[done] adapter -> {out_dir}")


if __name__ == "__main__":
    main()
