"""Dataset builders, one per stage, each emitting the exact type TRL expects.

  stage 1 SFT     xlam-function-calling-60k -> language modeling + `tools` column
  stage 2 reward  ultrafeedback_binarized   -> preference, implicit prompt
  stage 2 DPO     ultrafeedback_binarized   -> preference, explicit prompt
  stage 3 GRPO    gsm8k                     -> prompt-only + ground truth

All three source datasets are already in the local HF cache, so every stage runs
without touching the network.
"""

from __future__ import annotations

import json
import re

from datasets import Dataset, load_dataset

from .tools import tool_schemas

# xlam annotates parameters with Python-ish type names; the chat template wants
# JSON Schema types.
_TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array", "set": "array",
    "dict": "object", "object": "object", "any": "string",
}


def _json_type(raw: str) -> str:
    """Map an xlam type annotation onto a JSON Schema type, defaulting to string."""
    base = str(raw).split(",")[0].strip().lower()
    base = re.sub(r"\[.*", "", base).strip()  # "List[int]" -> "list"
    return _TYPE_MAP.get(base, "string")


def _is_optional(pspec: dict) -> bool:
    """Whether an xlam parameter is optional.

    xlam marks this two ways and the `, optional` suffix inside the type string
    is the primary one. Keying only on `default` mislabels a large share of
    optional parameters as required, which teaches the model that it must invent
    a value for every argument rather than omitting the ones it does not need.
    """
    if "optional" in str(pspec.get("type", "")).lower():
        return True
    default = pspec.get("default", None)
    # A supplied default marks optionality -- including an empty string or 0,
    # which are legitimate defaults rather than "no default given".
    return "default" in pspec and default is not None


def _xlam_tool_to_schema(tool: dict) -> dict:
    """Convert one xlam tool description into an OpenAI-style function schema."""
    props, required = {}, []
    for pname, pspec in (tool.get("parameters") or {}).items():
        if not isinstance(pspec, dict):
            continue
        props[pname] = {
            "type": _json_type(pspec.get("type", "string")),
            "description": pspec.get("description", "") or "",
        }
        if not _is_optional(pspec):
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", "") or "",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def build_sft(n: int = 4000, seed: int = 0) -> Dataset:
    """Tool-calling SFT set: a user turn, and an assistant turn that is a tool call.

    This is the stage that teaches schema adherence -- emitting a syntactically
    valid call against the *provided* tool list rather than inventing a function.

    Emitted as **prompt/completion**, not as a single `messages` column, and that
    distinction decides what stage 1 actually learns. TRL sets

        completion_only_loss = "prompt" in sample and "completion" in sample

    so the `messages` form silently trains on the whole sequence -- and here the
    sequence is dominated by the rendered tool-schema preamble and the user turn,
    leaving only a small fraction of the loss on the tool call itself. The
    obvious alternative, `assistant_only_loss=True`, is not available: TRL raises
    "Assistant-only loss is not yet supported for vision-language models" and
    Qwen3.5 is detected as one.

    `chat_template_kwargs={"enable_thinking": False}` is needed for the same
    reason it is on the preference rows. SFTTrainer builds the loss mask as

        completion_mask = [0]*len(prompt_ids) + [1]*(len(prompt_completion_ids)-len(prompt_ids))

    so a prompt render that is not a token prefix of the joint render puts the
    mask boundary in the wrong place: 8/8 rows misaligned with thinking on, 0/8
    with it off. It also stops every target rendering a hollow `<think></think>`
    pair, which would otherwise teach the model to open and immediately close
    its reasoning before each tool call. xlam carries no reasoning traces, so
    nothing is lost by turning it off for this stage.
    """
    raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    raw = raw.shuffle(seed=seed).select(range(min(n, len(raw))))

    rows = []
    for ex in raw:
        try:
            tools = json.loads(ex["tools"])
            answers = json.loads(ex["answers"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not tools or not answers:
            continue
        tool_calls = [
            {
                "type": "function",
                "function": {"name": a["name"], "arguments": a.get("arguments", {})},
            }
            for a in answers
            if isinstance(a, dict) and "name" in a
        ]
        if not tool_calls:
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": ex["query"]}],
                "completion": [{"role": "assistant", "tool_calls": tool_calls}],
                "tools": [_xlam_tool_to_schema(t) for t in tools if isinstance(t, dict)],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    # on_mixed_types="use_json" gives `tools` and `arguments` the Json() feature
    # type; without it datasets tries to unify wildly different argument structs
    # into one schema and throws.
    return Dataset.from_list(rows, on_mixed_types="use_json")


def build_preference(n: int = 3000, split: str = "train_prefs", explicit_prompt: bool = True) -> Dataset:
    """Preference pairs for reward modelling (implicit) or DPO (explicit).

    The explicit form carries `chat_template_kwargs={"enable_thinking": False}`,
    and that is not cosmetic. DPOTrainer tokenises the prompt and prompt+chosen
    separately and then slices:

        output["chosen_ids"] = prompt_chosen_ids[len(prompt_ids):]

    which is only correct if the prompt render is a token prefix of the joint
    render. With thinking on, Qwen3.5 ends the generation prompt at `<think>\\n`
    while the completed conversation renders `<think>\\n\\n</think>\\n\\n`, so the
    prefix property fails on **100%** of rows and the slice cuts in the wrong
    place. TRL notices and merely logs a warning, then trains on the misaligned
    completion anyway. Disabling thinking for these rows takes it to 0/8
    misaligned -- and matches the data, since ultrafeedback responses carry no
    reasoning traces to learn from in the first place.
    """
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    raw = raw.select(range(min(n, len(raw))))

    def to_explicit(ex):
        # chosen/rejected arrive as full conversations; strip the shared user turn
        # so the prompt is stated once rather than duplicated into both branches.
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [m for m in ex["chosen"] if m["role"] == "assistant"],
            "rejected": [m for m in ex["rejected"] if m["role"] == "assistant"],
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def to_implicit(ex):
        return {"chosen": ex["chosen"], "rejected": ex["rejected"]}

    fn = to_explicit if explicit_prompt else to_implicit
    return raw.map(fn, remove_columns=raw.column_names)


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


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "sft"):
        d = build_sft(n=64)
        print(f"[sft] {len(d)} rows, columns={d.column_names}")
        print(json.dumps(d[0], indent=2)[:1200])
    if which in ("all", "pref"):
        d = build_preference(n=64)
        print(f"[pref] {len(d)} rows, columns={d.column_names}")
    if which in ("all", "grpo"):
        d = build_grpo(n=64)
        print(f"[grpo] {len(d)} rows, columns={d.column_names}")
        print(json.dumps(d[0], indent=2)[:900])
    print(f"[tools] {len(tool_schemas())} schemas built from the local tool suite")

def build_distill_sft(path: str = "data/distill.jsonl") -> Dataset:
    """Load the rejection-sampled trajectories written by `agentlab.distill`.

    Same prompt/completion shape as build_sft, so everything downstream is
    unchanged -- but the completions come from the model's own *successful*
    multi-turn episodes rather than from single-turn xlam targets, so the corpus
    contains the terminating turns xlam never had.
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

