"""Dataset builders, one per stage, each emitting the exact type TRL expects.

  distilled SFT   data/distill.jsonl (verified trajectories) -> prompt/completion
  GRPO / eval     gsm8k -> prompt-only + ground truth

Sources are read from the local HF cache, so every stage runs without touching
the network.
"""

from __future__ import annotations

import json

from datasets import Dataset, load_dataset

from .tools import tool_schemas


GRPO_SYSTEM = (
    "You are a careful quantitative assistant. Use the calculator tool for every "
    "arithmetic step rather than doing mental arithmetic. When you have the final "
    "number, state it inside \\boxed{}."
)


def _gsm8k_answer(text: str) -> str:
    """Pull the reference answer out of GSM8K's '#### 42' suffix."""
    tail = text.split("####")[-1].strip()
    return tail.replace(",", "").replace("$", "")


def build_grpo(n: int = 1000, split: str = "train", seed: int = 0, offset: int = 0,
               system_suffix: str = "") -> Dataset:
    """Prompt-only set with verifiable ground truth, for tool-using GRPO.

    GSM8K is the right shape here: the reward is checkable without a judge, and
    every problem has an arithmetic step the calculator tool can serve.
    """
    raw = load_dataset("openai/gsm8k", "main", split=split)
    # `offset` exists so distillation and GRPO can take DISJOINT slices of the
    # same shuffled train split. Without it both take range(0, n) and GRPO ends
    # up doing RL on the exact problems the policy was just SFT'd on, which
    # collapses within-group reward variance and leaves nothing to learn from.
    raw = raw.shuffle(seed=seed).select(range(offset, min(offset + n, len(raw))))

    def to_prompt(ex):
        sys_msg = GRPO_SYSTEM + (" " + system_suffix if system_suffix else "")
        return {
            "prompt": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ex["question"]},
            ],
            "ground_truth": _gsm8k_answer(ex["answer"]),
            # Same rollout grammar the SFT adapter was trained with. Leaving
            # thinking on here re-introduced a format mismatch -- stage 1 teaches
            # tool calls with no reasoning block, then stage 3 would sample with
            # one -- and the generated reasoning competes for the single
            # completion budget shared by every turn and tool result.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    return raw.map(to_prompt, remove_columns=raw.column_names)


def build_eval(n: int = 200, system_suffix: str = "") -> Dataset:
    """Held-out GSM8K slice for the tool-use evaluation harness."""
    return build_grpo(n=n, split="test", seed=1, system_suffix=system_suffix)


def build_distill_sft(path: str = "data/distill.jsonl") -> Dataset:
    """Load the rejection-sampled trajectories written by `agentlab.distill`.

    Emitted as **prompt/completion**, not as a single `messages` column, and the
    distinction decides what SFT actually learns. TRL sets

        completion_only_loss = "prompt" in sample and "completion" in sample

    so the `messages` form silently trains on the whole sequence -- dominated by
    the rendered tool-schema preamble -- while this shape puts the loss on the
    committed assistant turn. The completions come from the model's own
    *successful* multi-turn episodes, so every row demonstrates observing a tool
    result and terminating with a committed answer.
    """
    import json
    import pathlib

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run `make distill` first to generate trajectories"
        )
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{p} is empty; the acceptance filter kept nothing")
    return Dataset.from_list(rows, on_mixed_types="use_json")


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "grpo"):
        d = build_grpo(n=64)
        print(f"[grpo] {len(d)} rows, columns={d.column_names}")
        print(json.dumps(d[0], indent=2)[:900])
    print(f"[tools] {len(tool_schemas())} schemas built from the local tool suite")

