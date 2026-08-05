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
    """
    hits = ANSWER_RE.findall(final_text or "")
    if hits:
        return hits[-1].strip().rstrip(".,;")
    boxed = BOXED_RE.findall(final_text or "")
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

    def to_row(self) -> dict:
        return {
            "task_id": self.task_id, "suite": self.suite, "split": self.split,
            "family": self.family, "horizon": self.horizon,
            "template_id": self.template_id, "prompt": self.prompt,
            "answer": self.answer, "answer_kind": self.answer_kind,
            "start": self.start, "env": self.env,
            "faults": [f.to_row() for f in self.faults],
            "max_decisions": self.max_decisions, "max_calls": self.max_calls,
            "secret_tokens": self.secret_tokens,
        }

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
