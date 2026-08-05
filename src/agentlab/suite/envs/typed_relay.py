"""Family B: typed_relay -- genuine orchestration across all three tools.

A deterministic compiler builds a typed linear DAG. A KB observation supplies
the operands and rule for the next conversion or calculation; the resulting
value determines the next KB key (records carry a `next_key` template like
"Q4ZX2P-{result}") or the final answer.

Grammar (H = required semantic nodes):

  H2   kb_lookup -> unit_convert                       (integer answer)
  H4   kb_lookup -> unit_convert -> calculator -> kb_lookup   (token answer)
  H8   (kb -> uc -> calc) x2 -> kb -> calc             (integer answer)
  H12  (kb -> uc -> calc) x4                           (integer answer)

Constraints (binding): integer-preserving conversions only (g<->kg, cm<->m,
min<->s families; no inch/mile floats); calculator expressions use bounded
integers and operators the shared calculator already supports; a calculator
call is accepted only if its safe AST contains the required prior value and
the generated coefficients and evaluates to the oracle result; numeric
terminal answers are exact integers, terminal KB answers are random tokens;
all three tools appear by horizon 4. This prevents a memorized final-value
shortcut: even when the arithmetic is mentally possible, strict success
requires completing the trace.
"""

from __future__ import annotations

from agentlab.tools import unit_convert as _unit_convert

from ..schema import OracleNode, TaskDraft

FAMILY = "typed_relay"
HORIZONS = (2, 4, 8, 12)


def has_unit_convert(horizon: int) -> bool:
    return True


# (from_unit, to_unit, factor, direction). "mul": source value c converts to
# c*factor; "div": source value c*factor converts to c. Both are integer-exact
# for the generated ranges.
_CONVERSIONS = (
    ("kg", "g", 1000, "mul"), ("m", "cm", 100, "mul"), ("km", "m", 1000, "mul"),
    ("min", "s", 60, "mul"), ("hour", "min", 60, "mul"), ("day", "hour", 24, "mul"),
    ("g", "kg", 1000, "div"), ("cm", "m", 100, "div"), ("s", "min", 60, "div"),
    ("mm", "cm", 10, "div"),
)

_ITEMS = ("coil", "spool", "beam", "ingot", "panel", "rod", "gasket", "flange",
          "bracket", "valve", "washer", "sleeve")


def _convert_str(value, from_unit: str, to_unit: str) -> str:
    out = _unit_convert(float(value), from_unit, to_unit)
    if out.startswith("error"):
        raise RuntimeError(f"generator produced invalid conversion: {out}")
    return out


def _draw_conversion(rng, used_numbers: set) -> tuple:
    """-> (V, from_unit, to_unit, C) with C not colliding with prior numbers."""
    while True:
        from_u, to_u, factor, direction = rng.choice(_CONVERSIONS)
        c = rng.randint(3, 499)
        if direction == "mul":
            v, conv = c, c * factor
        else:
            v, conv = c * factor, c
        if conv in used_numbers or v in used_numbers:
            continue
        # Sanity: the shared tool must render the exact integer.
        if _convert_str(v, from_u, to_u) != str(conv):
            continue
        used_numbers.add(conv)
        used_numbers.add(v)
        return v, from_u, to_u, conv


def _draw_calc(rng, prior_value: int, used_numbers: set) -> tuple:
    """-> (a, b, result) for result = prior_value*a + b, result fresh."""
    while True:
        a = rng.randint(2, 9)
        b = rng.randint(3, 97)
        r = prior_value * a + b
        if r in used_numbers:
            continue
        used_numbers.add(r)
        return a, b, r


def _structure(horizon: int) -> list[str]:
    if horizon == 2:
        return ["convert_only"]
    if horizon == 4:
        return ["full", "terminal_kb"]
    if horizon == 8:
        return ["full", "full", "value_calc"]
    if horizon == 12:
        return ["full", "full", "full", "full"]
    raise ValueError(f"typed_relay horizon must be one of {HORIZONS}")


def generate_task(rng, horizon: int) -> TaskDraft:
    blocks = _structure(horizon)
    kb: dict = {}
    nodes: list[OracleNode] = []
    used_numbers: set = set()
    used_keys: set = set()

    def fresh_start_key() -> str:
        while True:
            k = "K" + rng.base32(16)
            if k not in used_keys:
                used_keys.add(k)
                return k

    def fresh_prefix() -> str:
        while True:
            p = rng.base32(6) + "-"
            if p not in used_keys:
                used_keys.add(p)
                return p

    key = fresh_start_key()
    start_key = key
    n = 0
    answer: str | None = None
    answer_kind = "integer"

    for bi, kind in enumerate(blocks):
        last = bi == len(blocks) - 1

        if kind == "terminal_kb":
            code = rng.hex_token(128)
            rec = {"code": code}
            kb[key] = rec
            n += 1
            nodes.append(OracleNode(
                node_id=f"n{n}", tool="kb_lookup", args={"key": key},
                expect={"ok": True, "record": rec},
                match={"kind": "kb", "key": key}))
            answer, answer_kind = code, "token"
            continue

        if kind == "convert_only":
            v, from_u, to_u, conv = _draw_conversion(rng, used_numbers)
            rec = {"item": rng.choice(_ITEMS), "value": v, "unit": from_u,
                   "convert_to": to_u,
                   "rule": "convert value to convert_to; the converted number "
                           "is the final answer"}
            kb[key] = rec
            n += 1
            nodes.append(OracleNode(
                node_id=f"n{n}", tool="kb_lookup", args={"key": key},
                expect={"ok": True, "record": rec},
                match={"kind": "kb", "key": key}))
            n += 1
            nodes.append(OracleNode(
                node_id=f"n{n}", tool="unit_convert",
                args={"value": v, "from_unit": from_u, "to_unit": to_u},
                expect={"ok": True, "value": str(conv), "unit": to_u},
                match={"kind": "convert", "value": v, "from": from_u, "to": to_u}))
            answer, answer_kind = str(conv), "integer"
            continue

        if kind == "value_calc":
            while True:
                w = rng.randint(100, 9999)
                if w not in used_numbers:
                    used_numbers.add(w)
                    break
            a, b, r = _draw_calc(rng, w, used_numbers)
            rec = {"item": rng.choice(_ITEMS), "value": w, "a": a, "b": b,
                   "rule": "compute value*a+b; that number is the final answer"}
            kb[key] = rec
            n += 1
            nodes.append(OracleNode(
                node_id=f"n{n}", tool="kb_lookup", args={"key": key},
                expect={"ok": True, "record": rec},
                match={"kind": "kb", "key": key}))
            n += 1
            nodes.append(OracleNode(
                node_id=f"n{n}", tool="calculator",
                args={"expression": f"{w} * {a} + {b}"},
                expect={"ok": True, "value": str(r)},
                match={"kind": "calc", "required": [w, a, b], "result": r}))
            answer, answer_kind = str(r), "integer"
            continue

        # kind == "full": kb -> unit_convert -> calculator
        v, from_u, to_u, conv = _draw_conversion(rng, used_numbers)
        a, b, r = _draw_calc(rng, conv, used_numbers)
        rec = {"item": rng.choice(_ITEMS), "value": v, "unit": from_u,
               "convert_to": to_u, "a": a, "b": b}
        if last:
            rec["rule"] = ("convert value to convert_to, then compute "
                           "converted*a+b; that number is the final answer")
        else:
            prefix = fresh_prefix()
            rec["rule"] = ("convert value to convert_to, then compute "
                           "converted*a+b; look up next_key with {result} "
                           "replaced by that number")
            rec["next_key"] = prefix + "{result}"
        kb[key] = rec
        n += 1
        nodes.append(OracleNode(
            node_id=f"n{n}", tool="kb_lookup", args={"key": key},
            expect={"ok": True, "record": rec},
            match={"kind": "kb", "key": key}))
        n += 1
        nodes.append(OracleNode(
            node_id=f"n{n}", tool="unit_convert",
            args={"value": v, "from_unit": from_u, "to_unit": to_u},
            expect={"ok": True, "value": str(conv), "unit": to_u},
            match={"kind": "convert", "value": v, "from": from_u, "to": to_u}))
        n += 1
        nodes.append(OracleNode(
            node_id=f"n{n}", tool="calculator",
            args={"expression": f"{conv} * {a} + {b}"},
            expect={"ok": True, "value": str(r)},
            match={"kind": "calc", "required": [conv, a, b], "result": r}))
        if last:
            answer, answer_kind = str(r), "integer"
        else:
            key = prefix + str(r)
            used_keys.add(key)

    if len(nodes) != horizon:
        raise RuntimeError(f"typed_relay built {len(nodes)} nodes for H{horizon}")

    return TaskDraft(
        prompt_fields={"start_key": start_key,
                       "answer_format": ("the final code" if answer_kind == "token"
                                         else "the final number")},
        kb=kb, nodes=nodes, answer=answer, answer_kind=answer_kind,
        start={"start_key": start_key}, env=None, secret_tokens=[],
    )


_CORE = (
    "Records may contain: value with unit and convert_to (use unit_convert), "
    "coefficients a and b (use calculator to compute converted*a+b or "
    "value*a+b as the rule says), a next_key template whose {{result}} you "
    "replace with the computed number to get the next kb_lookup key, or a "
    "terminal code. Follow each record's rule exactly."
)

# 12 paraphrase templates; 0-7 train, 8-9 dev, 10-11 eval.
TEMPLATES = (
    # 0-7: training
    "Typed relay task. Start with kb_lookup on {start_key}. " + _CORE +
    " Commit {answer_format} as \\boxed{{answer}}.",
    "Work through a relay of typed steps beginning at key {start_key}. " + _CORE +
    " Reply with {answer_format} in \\boxed{{}}.",
    "Begin at KB key {start_key} and execute each step the records demand. " +
    _CORE + " Finish by committing {answer_format} inside \\boxed{{}}.",
    "Relay computation: your entry key is {start_key}. " + _CORE +
    " Your last message must contain {answer_format} as \\boxed{{answer}}.",
    "Chain the tools as instructed by each KB record, starting from "
    "{start_key}. " + _CORE + " End with {answer_format} in \\boxed{{}}.",
    "Start key: {start_key}. Each lookup tells you what to convert or compute "
    "next. " + _CORE + " Answer with {answer_format} as \\boxed{{answer}}.",
    "Follow the typed pipeline anchored at {start_key}. " + _CORE +
    " Conclude by writing {answer_format} in \\boxed{{}}.",
    "Execute the relay whose first key is {start_key}. " + _CORE +
    " Submit {answer_format} as \\boxed{{answer}}.",
    # 8-9: development
    "A typed chain of conversions and calculations starts at KB key "
    "{start_key}. " + _CORE + " Deliver {answer_format} as \\boxed{{answer}}.",
    "Resolve the relay rooted at {start_key}, one record at a time. " + _CORE +
    " Present {answer_format} inside \\boxed{{}}.",
    # 10-11: evaluation
    "Carry out the typed relay that opens with kb_lookup({start_key}). " +
    _CORE + " Report {answer_format} as \\boxed{{answer}}.",
    "Your task chain begins at key {start_key}. " + _CORE +
    " Close with {answer_format} committed in \\boxed{{}}.",
)


def template_text(template_id: int, fields: dict | None = None) -> str:
    """The raw paraphrase template behind a rendered prompt (structural id)."""
    return TEMPLATES[template_id]


def render_prompt(template_id: int, fields: dict) -> str:
    return template_text(template_id).format(**fields)


class TypedRelayEnv:
    """TRL-facing environment: reset(**row) rebuilds the episode exactly."""

    FAMILY = FAMILY

    def __init__(self) -> None:
        self.runtime = None

    def reset(self, *, spec: dict, kb: dict, nodes: list, **_ignored):
        from ..runtime import EpisodeRuntime
        from ..schema import OracleNode, TaskSpec

        self.runtime = EpisodeRuntime(
            TaskSpec.from_row(spec), kb, [OracleNode.from_row(n) for n in nodes])
        return self.runtime
