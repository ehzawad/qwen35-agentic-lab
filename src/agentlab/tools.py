"""The tool suite the model is trained and evaluated against.

These are plain Python functions with type hints and Google-style docstrings,
which is exactly the contract both `transformers.utils.get_json_schema` and
TRL's `GRPOTrainer(tools=[...])` expect. One definition therefore serves:

  * SFT   -- schemas rendered into the `tools` column of the chat template
  * GRPO  -- executed for real inside the rollout loop
  * eval  -- executed to check the model's calls actually run

Keep them deterministic and side-effect free: GRPO will call them thousands of
times per epoch, and a nondeterministic tool makes the reward signal noise.
"""

from __future__ import annotations

import ast
import functools
import operator

# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        # Cap exponents: 9**9**9 would otherwise hang the rollout worker.
        if isinstance(node.op, ast.Pow):
            exp = _eval_node(node.right)
            if abs(exp) > 64:
                raise ValueError("exponent too large")
            return op(_eval_node(node.left), exp)
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    Evaluates an arithmetic expression and returns the numeric result.

    Args:
        expression: An arithmetic expression such as "3 * (17 + 5) / 2".
            Supports + - * / // % ** and parentheses. No variables or function
            calls are allowed.

    Returns:
        The result as a string, or a message beginning with "error:" if the
        expression could not be evaluated.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_node(tree.body)
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception as e:
        return f"error: {e}"
    # Render 12.0 as "12" so string comparison against integer answers works.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------

# Everything is stored as a multiplier into one canonical base unit per family,
# so conversion is (value * factor_from) / factor_to within a family.
_UNITS: dict[str, tuple[str, float]] = {
    # length -> metre
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "km": ("length", 1000.0), "inch": ("length", 0.0254), "ft": ("length", 0.3048),
    "mile": ("length", 1609.344),
    # mass -> kilogram
    "mg": ("mass", 1e-6), "g": ("mass", 0.001), "kg": ("mass", 1.0),
    "lb": ("mass", 0.45359237), "oz": ("mass", 0.028349523125),
    # time -> second
    "s": ("time", 1.0), "min": ("time", 60.0), "hour": ("time", 3600.0),
    "day": ("time", 86400.0),
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts a value between two units of the same kind.

    Args:
        value: The numeric quantity to convert.
        from_unit: The source unit, one of mm, cm, m, km, inch, ft, mile, mg, g,
            kg, lb, oz, s, min, hour, day.
        to_unit: The target unit, drawn from the same list and the same family
            as from_unit.

    Returns:
        The converted value as a string, or a message beginning with "error:" if
        the units are unknown or belong to different families.
    """
    src, dst = from_unit.strip().lower(), to_unit.strip().lower()
    if src not in _UNITS:
        return f"error: unknown unit {from_unit!r}"
    if dst not in _UNITS:
        return f"error: unknown unit {to_unit!r}"
    fam_a, fac_a = _UNITS[src]
    fam_b, fac_b = _UNITS[dst]
    if fam_a != fam_b:
        return f"error: cannot convert {fam_a} to {fam_b}"
    return f"{value * fac_a / fac_b:.6g}"


# --------------------------------------------------------------------------
# knowledge lookup
# --------------------------------------------------------------------------

_KB: dict[str, str] = {
    "a5000": "NVIDIA RTX A5000: 24 GB GDDR6, 230 W board power, GA102 die.",
    "a6000": "NVIDIA RTX A6000: 48 GB GDDR6, 300 W board power, GA102 die.",
    "qwen3.5-4b": "Qwen3.5-4B: 4.7B params, Apache-2.0, 262144-token context, thinking mode on by default.",
    "grpo": "Group Relative Policy Optimization: samples a group of completions per prompt and uses the within-group reward spread as the advantage, so no value network is needed.",
    "dpo": "Direct Preference Optimization: fits a policy to pairwise preferences directly through a classification loss, skipping explicit reward modelling.",
    "sft": "Supervised fine-tuning: ordinary cross-entropy next-token training on curated demonstrations.",
    "lora": "Low-Rank Adaptation: freezes base weights and trains low-rank update matrices injected into projection layers.",
}


def kb_lookup(key: str) -> str:
    """
    Looks up a short factual entry in the local knowledge base.

    Args:
        key: The entry to look up, for example "a6000", "grpo", or "lora".
            Matching is case-insensitive.

    Returns:
        The knowledge base entry, or a message listing the valid keys if the
        requested key is not present.
    """
    hit = _KB.get(key.strip().lower())
    if hit is not None:
        return hit
    return f"error: no entry for {key!r}. Known keys: {', '.join(sorted(_KB))}"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

TOOLS = [calculator, unit_convert, kb_lookup]
TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}


def tool_schemas() -> list[dict]:
    """JSON schemas for the full tool suite, for the `tools` chat-template arg."""
    from transformers.utils import get_json_schema

    return [get_json_schema(fn) for fn in TOOLS]


@functools.lru_cache(maxsize=None)
def _type_hints(fn) -> dict:
    """Resolved (non-string) type hints for a tool, cached -- GRPO calls this a lot."""
    import typing

    try:
        return typing.get_type_hints(fn)
    except Exception:
        return {}


def coerce_args(name: str, raw: dict) -> dict:
    """Cast string arguments to the types the tool actually declares.

    Qwen's XML call format carries every parameter as raw text, so `value` for
    unit_convert arrives as "26.2" rather than 26.2. Without this, every numeric
    tool call raises TypeError and the model gets punished for a call it made
    correctly. Uncastable values are left as-is so the tool reports the error.
    """
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return raw
    # get_type_hints, not __annotations__: this module uses postponed evaluation
    # (`from __future__ import annotations`), so __annotations__ holds the *string*
    # "float", which never equals the float type and silently skips every cast.
    hints = _type_hints(fn)
    out = {}
    for k, v in raw.items():
        want = hints.get(k)
        if want in (int, float) and isinstance(v, str):
            try:
                out[k] = want(v.strip())
                continue
            except ValueError:
                pass
        elif want is bool and isinstance(v, str):
            out[k] = v.strip().lower() in ("true", "1", "yes")
            continue
        out[k] = v.strip() if isinstance(v, str) else v
    return out


def call_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name, returning the result or an error string."""
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return f"error: no such tool {name!r}. Available: {', '.join(TOOLS_BY_NAME)}"
    try:
        return str(fn(**arguments))
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}"
