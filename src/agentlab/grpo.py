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

from . import env
from .data import build_grpo
from .peft_cfg import describe, lora_config
from .tools import TOOLS, calculator, kb_lookup, unit_convert

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


# --------------------------------------------------------------------------
# completion helpers
# --------------------------------------------------------------------------

def _as_text(completion) -> str:
    """Flatten a completion to text.

    With tools in play a completion is a list of messages (assistant turns, tool
    calls, tool results), not a string -- so every reward function normalises here
    first rather than assuming one shape.
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
    """Names of the tools actually invoked in this rollout."""
    if isinstance(completion, str):
        return re.findall(r"<tool_call>\s*(\w+)", completion)
    names = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if fn.get("name"):
                names.append(fn["name"])
    return names


def _numeric(s: str):
    try:
        return float(str(s).strip().replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


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
        text = _as_text(comp)
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
    return out


def tool_use_reward(completions, **kwargs) -> list[float]:
    """Shaping: reward calling the calculator, penalise a call that errored.

    Deliberately small relative to correctness -- it should nudge the policy toward
    using tools, never pay it to spam calls it does not need.
    """
    log_metric = kwargs.get("log_metric")
    out, used = [], 0
    for comp in completions:
        names = _tool_names(comp)
        text = _as_text(comp)
        score = 0.0
        if names:
            used += 1
            score += 0.2
        if "calculator" in names:
            score += 0.1
        # A tool that returned "error: ..." means the model built a bad call.
        if "error:" in text:
            score -= 0.2
        out.append(score)
    if log_metric:
        log_metric("tool_use_rate", used / max(len(completions), 1))
    return out


def format_reward(completions, **kwargs) -> list[float]:
    """Shaping: the answer must be extractable at all."""
    return [0.1 if _BOXED_RE.search(_as_text(c)) else -0.1 for c in completions]


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
        mask_truncated_completions=True,
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
        reward_weights=[1.0, 0.3, 0.2] if args.mode == "tools" else None,
        # How many tool round-trips one rollout may take before it is cut off.
        max_tool_calling_iterations=args.max_tool_iters,
    )

    common = dict(
        model=args.model,
        args=cfg,
        peft_config=lora_config(r=args.rank),
    )

    if args.mode == "tools":
        ds = build_grpo(n=args.n)
        print(f"[data] {len(ds)} prompts, cols={ds.column_names}")
        print(f"[tools] {', '.join(t.__name__ for t in TOOLS)}")
        trainer = GRPOTrainer(
            train_dataset=ds,
            reward_funcs=[correctness_reward, tool_use_reward, format_reward],
            tools=[calculator, unit_convert, kb_lookup],
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

    describe(trainer.model)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[done] adapter -> {out_dir}")


if __name__ == "__main__":
    main()
