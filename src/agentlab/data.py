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
    "list": "array", "array": "array",
    "dict": "object", "object": "object",
}


def _json_type(raw: str) -> str:
    """Map an xlam type annotation onto a JSON Schema type, defaulting to string."""
    base = str(raw).split(",")[0].strip().lower()
    base = re.sub(r"\[.*", "", base).strip()  # "List[int]" -> "list"
    return _TYPE_MAP.get(base, "string")


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
        # xlam marks optionality by supplying a default; no default => required.
        if pspec.get("default", "") in ("", None):
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
                "messages": [
                    {"role": "user", "content": ex["query"]},
                    {"role": "assistant", "tool_calls": tool_calls},
                ],
                "tools": [_xlam_tool_to_schema(t) for t in tools if isinstance(t, dict)],
            }
        )

    # on_mixed_types="use_json" gives `tools` and `arguments` the Json() feature
    # type; without it datasets tries to unify wildly different argument structs
    # into one schema and throws.
    return Dataset.from_list(rows, on_mixed_types="use_json")


def build_preference(n: int = 3000, split: str = "train_prefs", explicit_prompt: bool = True) -> Dataset:
    """Preference pairs for reward modelling (implicit) or DPO (explicit)."""
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    raw = raw.select(range(min(n, len(raw))))

    def to_explicit(ex):
        # chosen/rejected arrive as full conversations; strip the shared user turn
        # so the prompt is stated once rather than duplicated into both branches.
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [m for m in ex["chosen"] if m["role"] == "assistant"],
            "rejected": [m for m in ex["rejected"] if m["role"] == "assistant"],
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


def build_grpo(n: int = 1000, split: str = "train", seed: int = 0) -> Dataset:
    """Prompt-only set with verifiable ground truth, for tool-using GRPO.

    GSM8K is the right shape here: the reward is checkable without a judge, and
    every problem has an arithmetic step the calculator tool can serve.
    """
    raw = load_dataset("openai/gsm8k", "main", split=split)
    raw = raw.shuffle(seed=seed).select(range(min(n, len(raw))))

    def to_prompt(ex):
        return {
            "prompt": [
                {"role": "system", "content": GRPO_SYSTEM},
                {"role": "user", "content": ex["question"]},
            ],
            "ground_truth": _gsm8k_answer(ex["answer"]),
        }

    return raw.map(to_prompt, remove_columns=raw.column_names)


def build_eval(n: int = 200) -> Dataset:
    """Held-out GSM8K slice for the tool-use evaluation harness."""
    return build_grpo(n=n, split="test", seed=1)


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
