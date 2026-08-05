"""Family B: typed_relay -- genuine orchestration across all three tools.

A deterministic compiler builds a typed linear DAG. A KB observation supplies
the operands and the rule for the next conversion or calculation; the resulting
value determines the next KB key (records carry a `next_key` template like
"Q4ZX2P-{result}") or the final answer.

Core grammar (H = required semantic nodes):

  H2   kb_lookup -> unit_convert                       (integer answer)
  H4   kb_lookup -> unit_convert -> calculator -> kb_lookup   (token answer)
  H8   (kb -> uc -> calc) x2 -> kb -> calc             (integer answer)
  H12  (kb -> uc -> calc) x4                           (integer answer)

The six REGISTERED tool-order patterns (`pattern_id` 0..5, horizon 4, all three
canonical tools causally required in every one of them):

  0  kb -> uc -> calc -> kb     terminal code
  1  kb -> uc -> kb  -> calc    terminal number
  2  kb -> calc -> kb -> uc     terminal number
  3  kb -> kb -> uc  -> calc    terminal number
  4  uc -> kb -> calc -> kb     terminal code, conversion opens the episode
  5  calc -> kb -> uc -> kb     terminal code, calculation opens the episode

These are genuinely different ORDERS, not one order relabelled: each pattern's
node i+1 consumes a value that only node i could have produced (a converted
number becomes a calculator operand, a computed number becomes the next KB key,
a record's value becomes the conversion input), so the registered order is the
only order that earns credit and the two non-kb entry patterns defeat any
"always start with kb_lookup" heuristic.

Constraints (binding): integer-preserving conversions only (g<->kg, cm<->m,
min<->s families; no inch/mile floats); calculator expressions use bounded
integers and operators the shared calculator already supports; a calculator
call is accepted only if its safe AST contains the required prior value and the
generated coefficients and evaluates to the oracle result; numeric terminal
answers are exact integers, terminal KB answers are random tokens; all three
tools appear by horizon 4. This prevents a memorized final-value shortcut: even
when the arithmetic is mentally possible, strict success requires completing the
trace.

The arithmetic FORM is a structural role, not a drawn value: each calculator
node uses one of five committed forms (`value * a + b`, `value * a - b`,
`value + a * b`, `(value + b) * a`, `value * a * b`), written into the record as
a `formula` field the model must read. Ten conversion pairs x five forms give
the 50 structural templates per order pattern that
`schema.template_cluster_id` resolves into distinct bootstrap clusters.
"""

from __future__ import annotations

from agentlab.tools import unit_convert as _unit_convert

from ..schema import OracleNode, TaskDraft

FAMILY = "typed_relay"
HORIZONS = (2, 4, 8, 12)

# The six registered tool-order patterns, as tool-short sequences. The order of
# this tuple IS the pattern_id assignment and is frozen.
MT_PATTERNS = (
    "kb>uc>calc>kb",
    "kb>uc>kb>calc",
    "kb>calc>kb>uc",
    "kb>kb>uc>calc",
    "uc>kb>calc>kb",
    "calc>kb>uc>kb",
)
MT_HORIZON = 4


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

# The five committed arithmetic forms. `expr` renders the concrete calculator
# expression, `text` the symbolic formula the record exposes, `fn` the oracle
# result. Every form mentions x, a and b exactly once, so the matcher's
# `required` list is satisfied by the literal constants in the AST.
_FORMS = (
    ("mul_add", lambda x, a, b: x * a + b,
     lambda x, a, b: f"{x} * {a} + {b}", "{x} * a + b"),
    ("mul_sub", lambda x, a, b: x * a - b,
     lambda x, a, b: f"{x} * {a} - {b}", "{x} * a - b"),
    ("add_mul", lambda x, a, b: x + a * b,
     lambda x, a, b: f"{x} + {a} * {b}", "{x} + a * b"),
    ("sum_mul", lambda x, a, b: (x + b) * a,
     lambda x, a, b: f"({x} + {b}) * {a}", "({x} + b) * a"),
    ("mul_mul", lambda x, a, b: x * a * b,
     lambda x, a, b: f"{x} * {a} * {b}", "{x} * a * b"),
)
N_FORMS = len(_FORMS)
N_CONVERSIONS = len(_CONVERSIONS)

# The registered MT allocation: 50 (conversion, form) structural templates per
# order pattern, so 6 x 50 = 300 distinct clusters over 600 MT tasks and every
# cluster holds exactly 2 value instantiations.
MT_COMBOS = tuple((ci, fi) for ci in range(N_CONVERSIONS) for fi in range(N_FORMS))

# The absent-information control needs a terminal whose value cannot be guessed:
# `b` is drawn from a range of exactly 2**48 integers, so the terminal number
# carries 48 bits of hidden entropy even though it is an ordinary integer.
ABSENT_ENTROPY_BITS = 48

_ITEMS = ("coil", "spool", "beam", "ingot", "panel", "rod", "gasket", "flange",
          "bracket", "valve", "washer", "sleeve")
_ZONES = ("north dock", "south dock", "mezzanine", "cold row", "outbound")


def _convert_str(value, from_unit: str, to_unit: str) -> str:
    out = _unit_convert(float(value), from_unit, to_unit)
    if out.startswith("error"):
        raise RuntimeError(f"generator produced invalid conversion: {out}")
    return out


def _draw_conversion(rng, used_numbers: set, forced: int | None = None) -> tuple:
    """-> (V, from_unit, to_unit, C) with C not colliding with prior numbers.

    `forced` pins the conversion PAIR (a structural role) while the coefficient
    stays drawn: that is exactly the cluster/instantiation split the MT
    allocation needs.
    """
    while True:
        entry = (_CONVERSIONS[forced] if forced is not None
                 else rng.choice(_CONVERSIONS))
        from_u, to_u, factor, direction = entry
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


def _draw_calc(rng, prior_value: int, used_numbers: set, form: int = 0,
               xname: str = "converted", big_b: bool = False) -> dict:
    """One calculator node's operands in one committed arithmetic form.

    -> {"a", "b", "result", "expression", "formula", "form"}. `big_b` draws the
    additive coefficient from a 2**48-wide range, which is what makes the
    absent-information control's numeric terminal unguessable.
    """
    name, fn, expr, text = _FORMS[form % N_FORMS]
    while True:
        a = rng.randint(2, 9)
        b = (rng.randint(1 << 48, (1 << 49) - 1) if big_b
             else rng.randint(3, 97))
        r = fn(prior_value, a, b)
        if r < 2 or r in used_numbers:
            continue
        used_numbers.add(r)
        return {"a": a, "b": b, "result": r,
                "expression": expr(prior_value, a, b),
                "formula": text.format(x=xname), "form": name}


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


class _Builder:
    """Shared fresh-label/node plumbing for the core and MT grammars."""

    def __init__(self, rng) -> None:
        self.rng = rng
        self.kb: dict = {}
        self.nodes: list[OracleNode] = []
        self.used_numbers: set = set()
        self.used_keys: set = set()

    # -- fresh labels -------------------------------------------------------

    def start_key(self) -> str:
        while True:
            k = "K" + self.rng.base32(16)
            if k not in self.used_keys:
                self.used_keys.add(k)
                return k

    def prefix(self) -> str:
        while True:
            p = self.rng.base32(6) + "-"
            if p not in self.used_keys:
                self.used_keys.add(p)
                return p

    def item(self) -> str:
        return self.rng.choice(_ITEMS)

    # -- nodes --------------------------------------------------------------

    def kb_node(self, key: str, record: dict) -> None:
        self.kb[key] = record
        self.used_keys.add(key)
        n = len(self.nodes) + 1
        self.nodes.append(OracleNode(
            node_id=f"n{n}", tool="kb_lookup", args={"key": key},
            expect={"ok": True, "record": record},
            match={"kind": "kb", "key": key}))

    def uc_node(self, value: int, from_u: str, to_u: str, conv: int) -> None:
        n = len(self.nodes) + 1
        self.nodes.append(OracleNode(
            node_id=f"n{n}", tool="unit_convert",
            args={"value": value, "from_unit": from_u, "to_unit": to_u},
            expect={"ok": True, "value": str(conv), "unit": to_u},
            match={"kind": "convert", "value": value, "from": from_u,
                   "to": to_u}))

    def calc_node(self, calc: dict, prior_value: int) -> None:
        n = len(self.nodes) + 1
        self.nodes.append(OracleNode(
            node_id=f"n{n}", tool="calculator",
            args={"expression": calc["expression"]},
            expect={"ok": True, "value": str(calc["result"])},
            match={"kind": "calc",
                   "required": [prior_value, calc["a"], calc["b"]],
                   "result": calc["result"]}))

    def fresh_int(self, lo: int, hi: int) -> int:
        while True:
            w = self.rng.randint(lo, hi)
            if w not in self.used_numbers:
                self.used_numbers.add(w)
                return w


# ---------------------------------------------------------------------------
# rule sentences (one per record shape; the model must read them)
# ---------------------------------------------------------------------------

_R_CONVERT_FINAL = ("convert value from unit to convert_to; the converted "
                    "number is the final answer")
_R_CONVERT_THEN_FORMULA_FINAL = (
    "convert value from unit to convert_to, then evaluate formula with the "
    "converted number substituted for 'converted'; that number is the final "
    "answer")
_R_CONVERT_THEN_FORMULA_NEXT = (
    "convert value from unit to convert_to, evaluate formula with the "
    "converted number substituted for 'converted', then look up next_key with "
    "{result} replaced by that computed number")
_R_CONVERT_THEN_NEXT = (
    "convert value from unit to convert_to, then look up next_key with "
    "{result} replaced by the converted number")
_R_FORMULA_FINAL_FROM_PRIOR = (
    "evaluate formula with the number you converted in the previous step "
    "substituted for 'converted'; that number is the final answer")
_R_FORMULA_THEN_NEXT_FROM_PRIOR = (
    "evaluate formula with the number you converted in the previous step "
    "substituted for 'converted', then look up next_key with {result} replaced "
    "by that computed number")
_R_FORMULA_THEN_NEXT = (
    "evaluate formula using value, a and b, then look up next_key with "
    "{result} replaced by that computed number")
_R_FOLLOW_NEXT = "look up the key in next"


# ---------------------------------------------------------------------------
# the core grammar (H2 / H4 / H8 / H12)
# ---------------------------------------------------------------------------

def generate_task(rng, horizon: int, *, pattern_id: int | None = None,
                  combo: int | None = None, absent: bool = False) -> TaskDraft:
    """One typed_relay task.

    `pattern_id` selects one of the six registered MT order patterns (horizon 4
    only); `combo` pins the (conversion pair, arithmetic form) structural
    template; `absent` builds the absent-information control variant.
    """
    if absent:
        return _generate_absent(rng)
    if pattern_id is not None:
        return _generate_pattern(rng, pattern_id, combo)

    blocks = _structure(horizon)
    b = _Builder(rng)
    key = b.start_key()
    start_key = key
    answer: str | None = None
    answer_kind = "integer"

    for bi, kind in enumerate(blocks):
        last = bi == len(blocks) - 1

        if kind == "terminal_kb":
            code = rng.hex_token(128)
            b.kb_node(key, {"code": code})
            answer, answer_kind = code, "token"
            continue

        if kind == "convert_only":
            v, from_u, to_u, conv = _draw_conversion(rng, b.used_numbers)
            b.kb_node(key, {"item": b.item(), "value": v, "unit": from_u,
                            "convert_to": to_u, "rule": _R_CONVERT_FINAL})
            b.uc_node(v, from_u, to_u, conv)
            answer, answer_kind = str(conv), "integer"
            continue

        if kind == "value_calc":
            w = b.fresh_int(100, 9999)
            calc = _draw_calc(rng, w, b.used_numbers,
                              form=rng.randint(0, N_FORMS - 1), xname="value")
            b.kb_node(key, {"item": b.item(), "value": w, "a": calc["a"],
                            "b": calc["b"], "formula": calc["formula"],
                            "rule": "evaluate formula using value, a and b; "
                                    "that number is the final answer"})
            b.calc_node(calc, w)
            answer, answer_kind = str(calc["result"]), "integer"
            continue

        # kind == "full": kb -> unit_convert -> calculator
        forced_ci, forced_fi = (None, None)
        if combo is not None and bi == 0:
            # The registered structural-template allocation. It matters for the
            # H4 cell specifically: the MT1 gate is denominated over the
            # `["eval", "eval_mt"]` stratum, and every all-tools H4 task in it
            # shares one cluster space. Drawing the (conversion, form) template
            # at random here put 8 instantiations in one cluster where the
            # registered ceiling is 5, which would make MT1 INCONCLUSIVE on a
            # clustering technicality. Allocating it by index gives exactly two
            # per template per split instead.
            forced_ci, forced_fi = MT_COMBOS[combo % len(MT_COMBOS)]
        v, from_u, to_u, conv = _draw_conversion(rng, b.used_numbers,
                                                 forced=forced_ci)
        calc = _draw_calc(rng, conv, b.used_numbers,
                          form=(rng.randint(0, N_FORMS - 1) if forced_fi is None
                                else forced_fi), xname="converted")
        rec = {"item": b.item(), "value": v, "unit": from_u,
               "convert_to": to_u, "a": calc["a"], "b": calc["b"],
               "formula": calc["formula"]}
        if last:
            rec["rule"] = _R_CONVERT_THEN_FORMULA_FINAL
        else:
            prefix = b.prefix()
            rec["rule"] = _R_CONVERT_THEN_FORMULA_NEXT
            rec["next_key"] = prefix + "{result}"
        b.kb_node(key, rec)
        b.uc_node(v, from_u, to_u, conv)
        b.calc_node(calc, conv)
        if last:
            answer, answer_kind = str(calc["result"]), "integer"
        else:
            key = prefix + str(calc["result"])

    if len(b.nodes) != horizon:
        raise RuntimeError(f"typed_relay built {len(b.nodes)} nodes for H{horizon}")

    return TaskDraft(
        prompt_fields={"start_key": start_key,
                       "answer_format": ("the final code" if answer_kind == "token"
                                         else "the final number")},
        kb=b.kb, nodes=b.nodes, answer=answer, answer_kind=answer_kind,
        start={"start_key": start_key}, env=None, secret_tokens=[],
    )


# ---------------------------------------------------------------------------
# the six registered MT order patterns (horizon 4)
# ---------------------------------------------------------------------------

def _generate_pattern(rng, pattern_id: int, combo: int | None) -> TaskDraft:
    if not 0 <= pattern_id < len(MT_PATTERNS):
        raise ValueError(f"pattern_id must be 0..{len(MT_PATTERNS) - 1}")
    ci, fi = MT_COMBOS[(0 if combo is None else combo) % len(MT_COMBOS)]
    b = _Builder(rng)

    if pattern_id == 0:      # kb -> uc -> calc -> kb
        key = b.start_key()
        v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
        calc = _draw_calc(rng, conv, b.used_numbers, form=fi, xname="converted")
        prefix = b.prefix()
        b.kb_node(key, {"item": b.item(), "value": v, "unit": fu,
                        "convert_to": tu, "a": calc["a"], "b": calc["b"],
                        "formula": calc["formula"],
                        "next_key": prefix + "{result}",
                        "rule": _R_CONVERT_THEN_FORMULA_NEXT})
        b.uc_node(v, fu, tu, conv)
        b.calc_node(calc, conv)
        code = rng.hex_token(128)
        b.kb_node(prefix + str(calc["result"]), {"code": code})
        return _mt_draft(b, pattern_id, code, "token",
                         {"start_key": key})

    if pattern_id == 1:      # kb -> uc -> kb -> calc
        key = b.start_key()
        v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
        prefix = b.prefix()
        b.kb_node(key, {"item": b.item(), "value": v, "unit": fu,
                        "convert_to": tu, "next_key": prefix + "{result}",
                        "rule": _R_CONVERT_THEN_NEXT})
        b.uc_node(v, fu, tu, conv)
        calc = _draw_calc(rng, conv, b.used_numbers, form=fi, xname="converted")
        b.kb_node(prefix + str(conv),
                  {"item": b.item(), "a": calc["a"], "b": calc["b"],
                   "formula": calc["formula"],
                   "rule": _R_FORMULA_FINAL_FROM_PRIOR})
        b.calc_node(calc, conv)
        return _mt_draft(b, pattern_id, str(calc["result"]), "integer",
                         {"start_key": key})

    if pattern_id == 2:      # kb -> calc -> kb -> uc
        key = b.start_key()
        w = b.fresh_int(100, 9999)
        calc = _draw_calc(rng, w, b.used_numbers, form=fi, xname="value")
        prefix = b.prefix()
        b.kb_node(key, {"item": b.item(), "value": w, "a": calc["a"],
                        "b": calc["b"], "formula": calc["formula"],
                        "next_key": prefix + "{result}",
                        "rule": _R_FORMULA_THEN_NEXT})
        b.calc_node(calc, w)
        v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
        b.kb_node(prefix + str(calc["result"]),
                  {"item": b.item(), "value": v, "unit": fu,
                   "convert_to": tu, "rule": _R_CONVERT_FINAL})
        b.uc_node(v, fu, tu, conv)
        return _mt_draft(b, pattern_id, str(conv), "integer",
                         {"start_key": key})

    if pattern_id == 3:      # kb -> kb -> uc -> calc
        key, key2 = b.start_key(), b.start_key()
        b.kb_node(key, {"zone": rng.choice(_ZONES), "next": key2,
                        "rule": _R_FOLLOW_NEXT})
        v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
        calc = _draw_calc(rng, conv, b.used_numbers, form=fi, xname="converted")
        b.kb_node(key2, {"item": b.item(), "value": v, "unit": fu,
                         "convert_to": tu, "a": calc["a"], "b": calc["b"],
                         "formula": calc["formula"],
                         "rule": _R_CONVERT_THEN_FORMULA_FINAL})
        b.uc_node(v, fu, tu, conv)
        b.calc_node(calc, conv)
        return _mt_draft(b, pattern_id, str(calc["result"]), "integer",
                         {"start_key": key})

    if pattern_id == 4:      # uc -> kb -> calc -> kb
        v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
        prefix = b.prefix()
        b.uc_node(v, fu, tu, conv)
        calc = _draw_calc(rng, conv, b.used_numbers, form=fi, xname="converted")
        prefix2 = b.prefix()
        b.kb_node(prefix + str(conv),
                  {"item": b.item(), "a": calc["a"], "b": calc["b"],
                   "formula": calc["formula"],
                   "next_key": prefix2 + "{result}",
                   "rule": _R_FORMULA_THEN_NEXT_FROM_PRIOR})
        b.calc_node(calc, conv)
        code = rng.hex_token(128)
        b.kb_node(prefix2 + str(calc["result"]), {"code": code})
        return _mt_draft(b, pattern_id, code, "token",
                         {"start_value": v, "from_unit": fu, "to_unit": tu,
                          "key_prefix": prefix})

    # pattern_id == 5:       calc -> kb -> uc -> kb
    w = b.fresh_int(100, 9999)
    calc = _draw_calc(rng, w, b.used_numbers, form=fi, xname="value")
    prefix = b.prefix()
    b.calc_node(calc, w)
    v, fu, tu, conv = _draw_conversion(rng, b.used_numbers, forced=ci)
    prefix2 = b.prefix()
    b.kb_node(prefix + str(calc["result"]),
              {"item": b.item(), "value": v, "unit": fu, "convert_to": tu,
               "next_key": prefix2 + "{result}",
               "rule": _R_CONVERT_THEN_NEXT})
    b.uc_node(v, fu, tu, conv)
    code = rng.hex_token(128)
    b.kb_node(prefix2 + str(conv), {"code": code})
    return _mt_draft(b, pattern_id, code, "token",
                     {"start_expression": calc["expression"],
                      "key_prefix": prefix})


def _mt_draft(b: _Builder, pattern_id: int, answer: str, answer_kind: str,
              start: dict) -> TaskDraft:
    from ..schema import tool_pattern

    if len(b.nodes) != MT_HORIZON:
        raise RuntimeError(f"MT pattern {pattern_id} built {len(b.nodes)} nodes")
    got = tool_pattern(b.nodes)
    if got != MT_PATTERNS[pattern_id]:
        raise RuntimeError(f"MT pattern {pattern_id} produced order {got!r}, "
                           f"registered {MT_PATTERNS[pattern_id]!r}")
    tools = {n.tool for n in b.nodes}
    if not {"kb_lookup", "unit_convert", "calculator"} <= tools:
        raise RuntimeError(f"MT pattern {pattern_id} misses a required tool")
    fields = dict(start)
    fields["answer_format"] = ("the final code" if answer_kind == "token"
                               else "the final number")
    return TaskDraft(prompt_fields=fields, kb=b.kb, nodes=b.nodes,
                     answer=answer, answer_kind=answer_kind, start=start,
                     env=None, secret_tokens=[])


# ---------------------------------------------------------------------------
# the absent-information control variant
# ---------------------------------------------------------------------------

def _generate_absent(rng) -> TaskDraft:
    """kb -> uc -> kb -> calc with a 48-bit-entropy NUMERIC terminal.

    A KB-record deletion at the end of a numeric relay would do nothing: the
    terminal value is computed, not retrieved, and a 4-digit product is guessable
    anyway. So the redactable record here is the numeric terminal's SOURCE (the
    record carrying the terminal coefficients), and the additive coefficient is
    drawn from a 2**48-wide range, which leaves the committed answer 48 bits of
    hidden entropy once that record is gone.
    """
    b = _Builder(rng)
    key = b.start_key()
    v, fu, tu, conv = _draw_conversion(rng, b.used_numbers)
    prefix = b.prefix()
    b.kb_node(key, {"item": b.item(), "value": v, "unit": fu, "convert_to": tu,
                    "next_key": prefix + "{result}",
                    "rule": _R_CONVERT_THEN_NEXT})
    b.uc_node(v, fu, tu, conv)
    calc = _draw_calc(rng, conv, b.used_numbers, form=0, xname="converted",
                      big_b=True)
    terminal_key = prefix + str(conv)
    b.kb_node(terminal_key, {"item": b.item(), "a": calc["a"], "b": calc["b"],
                             "formula": calc["formula"],
                             "rule": _R_FORMULA_FINAL_FROM_PRIOR})
    b.calc_node(calc, conv)
    return TaskDraft(
        prompt_fields={"start_key": key, "answer_format": "the final number"},
        kb=b.kb, nodes=b.nodes, answer=str(calc["result"]),
        answer_kind="integer", start={"start_key": key}, env=None,
        secret_tokens=[],
    )


def redact_absent(draft: TaskDraft) -> dict:
    """Delete the numeric terminal's SOURCE record; return the descriptor.

    Mutates `draft.kb` in place -- the committed control split IS the redacted
    task, so no consumer has to reimplement the transform and none can silently
    skip it.
    """
    kb_nodes = [n for n in draft.nodes if n.tool == "kb_lookup"]
    if not kb_nodes:
        raise RuntimeError("typed_relay absent draft has no kb_lookup node")
    target = kb_nodes[-1].args["key"]
    if target not in draft.kb:
        raise RuntimeError(f"typed_relay redaction target {target} not in KB")
    del draft.kb[target]
    return {"kind": "kb_record", "target": target,
            "field": None, "hidden_entropy_bits": ABSENT_ENTROPY_BITS,
            "why": "the terminal coefficients are unavailable, and the additive "
                   "coefficient spans 2**48 integers"}


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

_CORE = (
    "Each record tells you what to do next. A record may contain: `next`, a key "
    "to look up directly; a `value` with `unit` and `convert_to` (call "
    "unit_convert on exactly that value); coefficients `a` and `b` with a "
    "`formula` written over `value`/`converted`, `a` and `b` (evaluate it with "
    "the calculator, substituting the numbers you already have); a `next_key` "
    "template whose {{result}} you replace with the number the rule names; or a "
    "terminal `code`. Follow each record's `rule` exactly."
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

# Pattern 4 opens with a conversion: the operands are in the prompt, and the
# first KB key is the prefix with the converted number appended.
_UC_FIRST_OPEN = (
    "Your first call is unit_convert on value {start_value} from {from_unit} to "
    "{to_unit}. Append the converted number to the key prefix {key_prefix} and "
    "kb_lookup that key. ")
TEMPLATES_UC_FIRST = tuple(
    open_ + _UC_FIRST_OPEN + _CORE + tail for open_, tail in (
        ("Typed relay task, conversion first. ",
         " Commit {answer_format} as \\boxed{{answer}}."),
        ("This relay starts from a measurement, not a key. ",
         " Reply with {answer_format} in \\boxed{{}}."),
        ("Begin with the conversion below, then follow the records. ",
         " Finish by committing {answer_format} inside \\boxed{{}}."),
        ("Relay computation opening on a unit conversion. ",
         " Your last message must contain {answer_format} as \\boxed{{answer}}."),
        ("Convert first, then chain the tools as the records instruct. ",
         " End with {answer_format} in \\boxed{{}}."),
        ("No entry key is given: derive it. ",
         " Answer with {answer_format} as \\boxed{{answer}}."),
        ("Follow the typed pipeline that opens with a conversion. ",
         " Conclude by writing {answer_format} in \\boxed{{}}."),
        ("Execute the relay whose first step is a conversion. ",
         " Submit {answer_format} as \\boxed{{answer}}."),
        ("A typed chain begins with the measurement below. ",
         " Deliver {answer_format} as \\boxed{{answer}}."),
        ("Resolve the conversion-rooted relay one record at a time. ",
         " Present {answer_format} inside \\boxed{{}}."),
        ("Carry out the typed relay that opens with a unit conversion. ",
         " Report {answer_format} as \\boxed{{answer}}."),
        ("Your task chain begins with a conversion, not a lookup. ",
         " Close with {answer_format} committed in \\boxed{{}}."),
    ))

# Pattern 5 opens with a calculation: the expression is in the prompt, and the
# first KB key is the prefix with the computed number appended.
_CALC_FIRST_OPEN = (
    "Your first call is calculator on the expression {start_expression}. Append "
    "the result to the key prefix {key_prefix} and kb_lookup that key. ")
TEMPLATES_CALC_FIRST = tuple(
    open_ + _CALC_FIRST_OPEN + _CORE + tail for open_, tail in (
        ("Typed relay task, calculation first. ",
         " Commit {answer_format} as \\boxed{{answer}}."),
        ("This relay starts from an expression, not a key. ",
         " Reply with {answer_format} in \\boxed{{}}."),
        ("Begin with the calculation below, then follow the records. ",
         " Finish by committing {answer_format} inside \\boxed{{}}."),
        ("Relay computation opening on an arithmetic step. ",
         " Your last message must contain {answer_format} as \\boxed{{answer}}."),
        ("Compute first, then chain the tools as the records instruct. ",
         " End with {answer_format} in \\boxed{{}}."),
        ("No entry key is given: compute it. ",
         " Answer with {answer_format} as \\boxed{{answer}}."),
        ("Follow the typed pipeline that opens with a calculation. ",
         " Conclude by writing {answer_format} in \\boxed{{}}."),
        ("Execute the relay whose first step is a calculation. ",
         " Submit {answer_format} as \\boxed{{answer}}."),
        ("A typed chain begins with the expression below. ",
         " Deliver {answer_format} as \\boxed{{answer}}."),
        ("Resolve the calculation-rooted relay one record at a time. ",
         " Present {answer_format} inside \\boxed{{}}."),
        ("Carry out the typed relay that opens with a calculation. ",
         " Report {answer_format} as \\boxed{{answer}}."),
        ("Your task chain begins with a calculation, not a lookup. ",
         " Close with {answer_format} committed in \\boxed{{}}."),
    ))


def template_text(template_id: int, fields: dict | None = None) -> str:
    """The raw paraphrase template behind a rendered prompt (structural id).

    The two non-kb entry patterns need their own wordings -- they hand the model
    operands instead of a start key -- so the template SET is selected by which
    bootstrap values the task exposes, which is recoverable from the committed
    spec (`start`) and therefore reproducible by the certification layer.
    """
    fields = fields or {}
    if "start_expression" in fields:
        return TEMPLATES_CALC_FIRST[template_id]
    if "start_value" in fields:
        return TEMPLATES_UC_FIRST[template_id]
    return TEMPLATES[template_id]


def render_prompt(template_id: int, fields: dict) -> str:
    return template_text(template_id, fields).format(**fields)


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
