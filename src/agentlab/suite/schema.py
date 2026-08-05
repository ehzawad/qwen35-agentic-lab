"""Suite v1 data contracts: TaskSpec, OracleNode, FaultSpec, TraceEvent.

Everything the generator writes and the runtime/verifier read is defined here,
plus the committed serialization: canonical JSON is sorted keys, compact
separators, UTF-8, and every file ends with exactly one trailing newline.

Terminology (binding, from the council env-architect spec):

  * horizon H     = number of successful semantic tool nodes in the hidden
                    oracle DAG. Retries, injected failures, malformed model
                    calls and redundant calls never change it. The terminal
                    assistant response is not part of the horizon.
  * oracle node   = one required semantic tool call, with canonical args, the
                    canonical exposed payload, and a matcher that decides
                    whether a model call semantically reaches this node.
  * fault spec    = one scheduled injected failure bound to a target node.
  * trace event   = the environment-side record of one dispatch; the model
                    never sees it.

Budgets: clean episodes get H+3 assistant decisions, single-fault H+5,
two-fault stress H+8; total tool calls are capped at 2H+4.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re

SUITE_NAME = "agentlab-suite-v1"
SUITE_VERSION = "1.0.0"

FAMILIES = ("lookup_chain", "typed_relay", "fulfillment")
FAULT_TYPES = ("transient", "malformed", "wrong_unit", "rate_limit")

# The 12 primary family/horizon cells.
CELLS: tuple[tuple[str, int], ...] = (
    ("lookup_chain", 2), ("lookup_chain", 4), ("lookup_chain", 8), ("lookup_chain", 12),
    ("typed_relay", 2), ("typed_relay", 4), ("typed_relay", 8), ("typed_relay", 12),
    ("fulfillment", 4), ("fulfillment", 8), ("fulfillment", 14), ("fulfillment", 20),
)

# Paraphrase-template split: 0-7 training, 8-9 development, 10-11 evaluation.
TEMPLATE_RANGES = {"train": tuple(range(0, 8)),
                   "dev": tuple(range(8, 10)),
                   "eval": tuple(range(10, 12))}


# ---------------------------------------------------------------------------
# canonical serialization
# ---------------------------------------------------------------------------

def canon(obj) -> str:
    """Sorted keys, compact separators, UTF-8: the committed serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(obj) -> str:
    return digest_text(canon(obj))


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_jsonl(path: str, rows: list) -> None:
    """Canonical JSONL with exactly one trailing newline."""
    write_text(path, "".join(canon(r) + "\n" for r in rows))


def write_json(path: str, obj) -> None:
    write_text(path, canon(obj) + "\n")


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the committed answer format (frozen in configs/agentic_preregister.json:
# "final assistant line matching /ANSWER\s*:\s*(\S+)/, last occurrence;
#  \boxed{} accepted as fallback")
#
# The two frozen forms NEST -- `ANSWER: \boxed{x}` -- because the neutral system
# prompt and the generated task prompt ask for one each. `extract_committed_answer`
# is the only reader of this grammar; every other module must call it rather than
# recompile these patterns, or "verified" and "certified" can drift apart.
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\s`*]+)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_committed_answer(final_text: str) -> str | None:
    """The one implementation of "what did the assistant commit to".

    The preregistered neutral prompt asks for `ANSWER: <value>`; the suite's
    generated task prompts ask for `\\boxed{}`, which the preregistration accepts
    as a fallback. Both the strict verifier and the certification layer must read
    a commitment the same way -- if they disagreed, a model following one prompt
    would be scored as never committing an answer at all.

    The two instructions are given TOGETHER (system prompt plus task prompt), so
    obeying both at once -- `ANSWER: \\boxed{x}` -- is reasonable compliance, and
    it must commit `x`, not the literal string `\\boxed{x}`. Reading only the
    `ANSWER:` token scored a correctly-answering model wrong on 7 of 12 dev
    episodes and mislabelled all 7 as hallucinations, because `\\boxed{x}` has no
    source in any tool observation.

    The unwrap is deliberately narrow: it is anchored at the first character of
    the selected `ANSWER:` value, so nothing is scraped out of surrounding prose.
    Precedence, unchanged from the preregistration:

      * the LAST `ANSWER:` bearing a value wins;
      * `ANSWER: \\boxed{x}` returns `x`;
      * with no `ANSWER:` at all, the last `\\boxed{x}` is accepted;
      * a later WRONG `ANSWER:` is never rescued by an earlier correct box --
        the boxed fallback is reachable only when no `ANSWER:` value exists.
    """
    text = final_text or ""
    hits = list(ANSWER_RE.finditer(text))
    if hits:
        value = hits[-1]
        # Anchored at value.start(1): a full `\boxed{...}` opening exactly where
        # the committed value begins. Matching against `text` rather than the
        # captured token also keeps a box whose contents contain spaces intact.
        nested = BOXED_RE.match(text, value.start(1))
        if nested and nested.group(1).strip():
            return nested.group(1).strip()
        return value.group(1).strip().rstrip(".,;")
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def normalize_number(v):
    """Integral floats become ints so canonical digests never depend on '2.0' vs '2'."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def normalize_args(args: dict) -> dict:
    return {k: normalize_number(v) for k, v in args.items()}


def call_args_digest(tool: str, args: dict) -> str:
    return digest({"tool": tool, "args": normalize_args(args)})


# ---------------------------------------------------------------------------
# STRUCTURAL template clusters (the bootstrap resampling unit)
#
# `template_id` is a PARAPHRASE id: which of the 12 wordings rendered the prompt.
# Held-out evaluation draws from ids 10-11 only, so clustering the bootstrap on
# it collapses every eval task into two clusters -- one per wording, pooled
# across families and horizons -- and the clustered confidence intervals then
# describe two units of information instead of the real structural variety.
#
# `template_cluster_id` is the STRUCTURAL identity instead: family, horizon, the
# tool-order pattern, and the operand ROLE of every oracle node (which fields a
# KB record exposes, which unit pair a conversion crosses, the arithmetic shape
# of an expression, which warehouse resource/action a call touches). It is a
# pure function of (family, horizon, oracle nodes) -- never of the drawn values
# -- so two tasks share a cluster exactly when they are two value
# instantiations of one structural template.
#
# It is deliberately NOT a split-isolation field. Parameterized instance
# identity is what must never cross train/dev/eval, and that is the whole-task
# content hash the validator already enforces; the unparameterized structure is
# shared by construction, because the same families and horizons are evaluated
# in every split.
# ---------------------------------------------------------------------------

_EXPR_OPS = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
             "FloorDiv": "//", "Mod": "%", "Pow": "**"}

TOOL_SHORT = {"kb_lookup": "kb", "unit_convert": "uc", "calculator": "calc",
              "warehouse_query": "wq", "warehouse_update": "wu"}


def expr_shape(expression: str) -> str:
    """The arithmetic SHAPE of an expression, constants erased.

    "1200 * 7 + 5" and "48 * 3 + 91" are the same structural template and both
    render as "((c*c)+c)"; "(48 + 91) * 3" renders as "((c+c)*c)" and is a
    different one. This is what makes the calculator form a structural role
    rather than a drawn value.
    """
    import ast

    def walk(node) -> str:
        if isinstance(node, ast.BinOp):
            op = _EXPR_OPS.get(type(node.op).__name__, "?")
            return f"({walk(node.left)}{op}{walk(node.right)})"
        if isinstance(node, ast.UnaryOp):
            return f"(-{walk(node.operand)})"
        if isinstance(node, ast.Constant):
            return "c"
        return "?"

    try:
        return walk(ast.parse(str(expression).strip(), mode="eval").body)
    except (SyntaxError, ValueError):
        return "?"


def node_role(node) -> dict:
    """One oracle node's structural role: no drawn value survives."""
    tool = node.tool
    if tool == "kb_lookup":
        rec = (node.expect or {}).get("record")
        return {"t": "kb", "f": sorted(rec) if isinstance(rec, dict) else []}
    if tool == "unit_convert":
        return {"t": "uc", "u": [str(node.args.get("from_unit", "")),
                                 str(node.args.get("to_unit", ""))]}
    if tool == "calculator":
        return {"t": "calc", "e": expr_shape(node.args.get("expression", ""))}
    if tool == "warehouse_query":
        return {"t": "wq", "r": str(node.args.get("resource", ""))}
    if tool == "warehouse_update":
        return {"t": "wu", "a": str(node.args.get("action", ""))}
    return {"t": tool}


def tool_pattern(nodes) -> str:
    """The tool-order pattern of an oracle DAG, e.g. "kb>uc>calc>kb"."""
    return ">".join(TOOL_SHORT.get(n.tool, n.tool) for n in nodes)


def structural_descriptor(family: str, horizon: int, nodes) -> dict:
    """The full canonical structure a `template_cluster_id` hashes."""
    return {"family": family, "horizon": horizon,
            "pattern": tool_pattern(nodes),
            "edges": [[i, i + 1] for i in range(len(nodes) - 1)],
            "roles": [node_role(n) for n in nodes]}


def template_cluster_id(family: str, horizon: int, nodes) -> str:
    """The bootstrap cluster label: "tc-" + 64 bits of the structural digest."""
    return "tc-" + digest(structural_descriptor(family, horizon, nodes))[:16]


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OracleNode:
    """One required semantic tool call in the hidden oracle DAG."""

    node_id: str           # "n1" .. "nH"
    tool: str              # calculator | unit_convert | kb_lookup | warehouse_*
    args: dict             # canonical oracle arguments
    expect: dict           # canonical semantic payload (no event_id)
    match: dict            # matcher spec, see runtime._match_node
    mutating: bool = False

    def to_row(self) -> dict:
        return {"node_id": self.node_id, "tool": self.tool, "args": self.args,
                "expect": self.expect, "match": self.match, "mutating": self.mutating}

    def args_digest(self) -> str:
        """Stable identity of this node's canonical call.

        This is the ONE node-identity digest in the suite: the digest of the
        node's canonical (tool, normalized args), the same function the runtime
        stamps on every dispatch event. It replaces the earlier per-tool
        "projection" digest (kb by key, unit_convert by unit pair, calculator by
        positional occurrence), which existed only because that stack matched
        calls textually; the canonical matcher in runtime._match_node decides
        node identity semantically, so a looser projection is not needed and a
        second digest scheme would be a second source of truth.
        """
        return call_args_digest(self.tool, self.args)

    def result_digest(self) -> str:
        """Digest of the node's canonical exposed payload (no event_id)."""
        return digest(self.expect)

    @classmethod
    def from_row(cls, row: dict) -> "OracleNode":
        return cls(node_id=row["node_id"], tool=row["tool"], args=row["args"],
                   expect=row["expect"], match=row["match"],
                   mutating=bool(row.get("mutating", False)))


@dataclasses.dataclass
class FaultSpec:
    """One scheduled injected failure bound to one oracle node."""

    fault_type: str        # transient | malformed | wrong_unit | rate_limit
    target_node: str       # node_id of the target oracle node
    params: dict           # wrong_unit: {"wrong_unit": u}; malformed: {"ambiguous_mutation": bool, "line": int}
                           # rate_limit: {"retry_after_turns": 1}

    def to_row(self) -> dict:
        return {"fault_type": self.fault_type, "target_node": self.target_node,
                "params": self.params}

    @classmethod
    def from_row(cls, row: dict) -> "FaultSpec":
        return cls(fault_type=row["fault_type"], target_node=row["target_node"],
                   params=row.get("params", {}))


@dataclasses.dataclass
class TaskSpec:
    """The full deterministic description of one episode."""

    task_id: str
    suite: str
    split: str             # oracle_sft | distill | grpo_train | dev | eval | eval_stress
    family: str
    horizon: int
    template_id: int
    prompt: str
    answer: str
    answer_kind: str       # "token" | "integer"
    start: dict            # model-visible bootstrap values (start key / order token)
    env: dict | None       # fulfillment initial state + oracle_final; None otherwise
    faults: list           # list[FaultSpec]
    max_decisions: int
    max_calls: int
    secret_tokens: list    # capability tokens for provenance checking (fulfillment)
    # Appended with defaults so any existing positional construction still works.
    template_cluster_id: str = ""   # STRUCTURAL cluster; the bootstrap unit
    pattern_id: int | None = None   # 0..5 for registered typed_relay MT orders
    control: str | None = None      # None | "redacted" | "permuted"
    control_meta: dict | None = None  # the committed control descriptor

    def to_row(self) -> dict:
        row = {
            "task_id": self.task_id, "suite": self.suite, "split": self.split,
            "family": self.family, "horizon": self.horizon,
            "template_id": self.template_id,
            "template_cluster_id": self.template_cluster_id,
            "pattern_id": self.pattern_id, "prompt": self.prompt,
            "answer": self.answer, "answer_kind": self.answer_kind,
            "start": self.start, "env": self.env,
            "faults": [f.to_row() for f in self.faults],
            "max_decisions": self.max_decisions, "max_calls": self.max_calls,
            "secret_tokens": self.secret_tokens,
        }
        # Control rows are the exception, not the norm: the 10,000+ ordinary
        # specs must not each carry two null fields.
        if self.control is not None:
            row["control"] = self.control
            row["control_meta"] = self.control_meta or {}
        return row

    @classmethod
    def from_row(cls, row: dict) -> "TaskSpec":
        return cls(
            task_id=row["task_id"], suite=row["suite"], split=row["split"],
            family=row["family"], horizon=int(row["horizon"]),
            template_id=int(row["template_id"]), prompt=row["prompt"],
            answer=str(row["answer"]), answer_kind=row["answer_kind"],
            start=row.get("start", {}), env=row.get("env"),
            faults=[FaultSpec.from_row(f) for f in row.get("faults", [])],
            max_decisions=int(row["max_decisions"]), max_calls=int(row["max_calls"]),
            secret_tokens=list(row.get("secret_tokens", [])),
            template_cluster_id=str(row.get("template_cluster_id", "")),
            pattern_id=(None if row.get("pattern_id") is None
                        else int(row["pattern_id"])),
            control=row.get("control"), control_meta=row.get("control_meta"),
        )

    def without_faults(self) -> "TaskSpec":
        """The paired clean arm of this spec (budgets recomputed for 0 faults)."""
        clean = dataclasses.replace(self, faults=[])
        clean.max_decisions = decision_budget(self.horizon, 0)
        return clean


@dataclasses.dataclass
class TraceEvent:
    """Environment-side record of one dispatch. The model never sees this."""

    decision_id: int
    call_id: int
    event_id: str
    tool: str
    oracle_node: str | None        # node whose matcher this call satisfies
    credited: bool                 # counted as oracle progress (dependency-valid)
    repeat: bool                   # semantically matched an already-credited node
    canonical_args_digest: str
    canonical_result_digest: str | None   # digest of the node's canonical payload
    exposed_result_digest: str
    exposed_canonical: bool        # exposed semantic payload == canonical payload
    ok: bool                       # exposed payload had ok == true
    fault_type: str | None
    fault_triggered: bool          # this event consumed the scheduled fault
    rate_limited: bool             # rate-limit response (first hit or same-decision repeat)
    mutating: bool
    state_mutated: bool
    replay: bool                   # idempotent replay of an earlier mutation
    unsafe: bool                   # state mutation outside the oracle plan
    token_known: bool              # capability token was revealed before use
    aux: dict                      # e.g. {"resource": ..., "line": ...} for warehouse calls
    state_before: str
    state_after: str

    def to_row(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------

def decision_budget(horizon: int, n_faults: int) -> int:
    if n_faults <= 0:
        return horizon + 3
    if n_faults == 1:
        return horizon + 5
    return horizon + 8


def call_budget(horizon: int) -> int:
    return 2 * horizon + 4


def oracle_plan_digest(nodes) -> str:
    """Digest of a whole oracle DAG: (node_id, tool, args, expect) in order.

    Any consumer that rebuilds an episode must reproduce this digest, or it is
    not running the task the verifier will score.
    """
    return digest([{"node_id": n.node_id, "tool": n.tool,
                    "args": normalize_args(n.args), "expect": n.expect}
                   for n in nodes])


def episode_digest(spec: "TaskSpec", kb: dict, nodes) -> str:
    """One digest binding a spec, its KB view and its oracle plan together."""
    return digest({"spec": spec.to_row(), "kb": kb,
                   "oracle": oracle_plan_digest(nodes)})


@dataclasses.dataclass
class TaskDraft:
    """What a family generator produces before split/fault/template binding."""

    prompt_fields: dict
    kb: dict               # {key: record}
    nodes: list            # list[OracleNode]
    answer: str
    answer_kind: str
    start: dict
    env: dict | None
    secret_tokens: list
