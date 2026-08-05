"""Family A: lookup_chain -- pure multi-hop retrieval and state tracking.

The prompt exposes one random 80-bit base32 start key. Each nonterminal KB
record exposes exactly one next key and a small distractor fact; the terminal
record exposes a random 128-bit answer code. Keys, codes, record order and
distractors are generated separately for every task; train/dev/eval
namespaces never overlap (checked by the validator).

Strict success: every required key retrieved in order, every dependency edge
crossing a later assistant decision, and the final answer containing the
exact terminal code. Extra read-only calls are allowed within budget but earn
no credit. This family is deliberately simple cognitively; its value is a
clean survival-versus-depth curve.
"""

from __future__ import annotations

from ..schema import OracleNode, TaskDraft

FAMILY = "lookup_chain"
HORIZONS = (2, 4, 8, 12)


def has_unit_convert(horizon: int) -> bool:
    return False


_AISLES = ("A", "B", "C", "D", "E", "F", "G", "H")
_THINGS = ("crate", "pallet", "bin", "rack", "tote", "drum", "carton", "shelf")
_ZONES = ("north dock", "south dock", "mezzanine", "cold row", "outbound",
          "returns", "staging", "overflow")

# The eight committed distractor SCHEMAS. Every nonterminal record still exposes
# exactly one `next` key; what varies is which distractor fields surround it, so
# the model has to locate the pointer in differently shaped records instead of
# reading a fixed slot. This is the family's structural-role dimension: the field
# set enters `schema.node_role`, so two chains whose records are shaped alike are
# one bootstrap cluster and two shaped differently are not. Without it every
# lookup_chain cell would be a single cluster of 100 instantiations, and its
# contribution to a clustered interval would be one unit of information.
_SCHEMAS = (
    ("note",),
    ("zone",),
    ("note", "lot"),
    ("shelf",),
    ("bay", "note"),
    ("aisle", "slot"),
    ("tag",),
    ("note", "zone", "bin"),
)
N_SCHEMAS = len(_SCHEMAS)


def _distractor(rng, field: str):
    if field == "note":
        return (f"{rng.choice(_THINGS)} {rng.randint(2, 97)}, "
                f"aisle {rng.choice(_AISLES)}{rng.randint(1, 9)}")
    if field == "zone":
        return rng.choice(_ZONES)
    if field == "lot":
        return "L" + rng.base32(5)
    if field == "shelf":
        return f"{rng.choice(_AISLES)}{rng.randint(1, 9)}-{rng.randint(10, 89)}"
    if field == "bay":
        return rng.randint(1, 48)
    if field == "aisle":
        return f"{rng.choice(_AISLES)}{rng.randint(1, 9)}"
    if field == "slot":
        return rng.randint(1, 24)
    if field == "tag":
        return "T" + rng.base32(4)
    if field == "bin":
        return rng.randint(100, 999)
    raise ValueError(f"unknown distractor field {field!r}")


def generate_task(rng, horizon: int) -> TaskDraft:
    if horizon not in HORIZONS:
        raise ValueError(f"lookup_chain horizon must be one of {HORIZONS}")
    keys: list[str] = []
    seen: set[str] = set()
    while len(keys) < horizon:
        k = "K" + rng.base32(16)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    code = rng.hex_token(128)

    kb: dict = {}
    nodes: list[OracleNode] = []
    for i, key in enumerate(keys):
        if i < horizon - 1:
            schema = _SCHEMAS[rng.randint(0, N_SCHEMAS - 1)]
            rec = {"next": keys[i + 1]}
            for field in schema:
                rec[field] = _distractor(rng, field)
        else:
            rec = {"code": code}
        kb[key] = rec
        nodes.append(OracleNode(
            node_id=f"n{i + 1}", tool="kb_lookup", args={"key": key},
            expect={"ok": True, "record": rec},
            match={"kind": "kb", "key": key},
        ))

    return TaskDraft(
        prompt_fields={"start_key": keys[0]},
        kb=kb, nodes=nodes, answer=code, answer_kind="token",
        start={"start_key": keys[0]}, env=None, secret_tokens=[],
    )


def redact_absent(draft: TaskDraft) -> dict:
    """Delete the terminal record holding the 128-bit code; return the descriptor.

    Mutates `draft.kb` in place: the committed control split IS the redacted
    task, so nothing downstream has to reimplement the transform.
    """
    target = draft.nodes[-1].args["key"]
    if target not in draft.kb:
        raise RuntimeError(f"lookup_chain redaction target {target} not in KB")
    del draft.kb[target]
    return {"kind": "kb_record", "target": target, "field": "code",
            "hidden_entropy_bits": 128,
            "why": "the terminal 128-bit code is retrievable from exactly one "
                   "record, and that record is gone"}


# 12 paraphrase templates; 0-7 train, 8-9 dev, 10-11 eval.
TEMPLATES = (
    # 0-7: training
    "Follow a key chain in the knowledge base. Start key: {start_key}. Each "
    "record's 'next' field is the key for your next kb_lookup. The last record "
    "has a 'code' field instead. Report that code as \\boxed{{code}}.",
    "You are tracing a linked list of KB records. Begin with kb_lookup on "
    "{start_key}. Every record gives you 'next', the following key, until one "
    "gives 'code'. Answer with \\boxed{{code}}.",
    "Chain lookup task. First key: {start_key}. Look it up, read the 'next' "
    "key, look that up, and so on. The chain ends at a record carrying 'code'; "
    "commit it as \\boxed{{code}}.",
    "Resolve this pointer chain: start at key {start_key} with kb_lookup. Each "
    "hop's record names the next key under 'next'. Stop at the record with "
    "'code' and reply \\boxed{{code}}.",
    "Walk the KB chain beginning at {start_key}. A record either points onward "
    "via 'next' or terminates with 'code'. Your final reply must contain "
    "\\boxed{{code}} with the terminal code.",
    "Key-following exercise. Use kb_lookup starting from {start_key}; records "
    "link forward through their 'next' field. When you reach the record whose "
    "field is 'code', answer \\boxed{{code}}.",
    "Traverse the lookup chain: {start_key} is your entry key. Read 'next' from "
    "each record to get the following key. The terminal record holds 'code'; "
    "finish with \\boxed{{code}}.",
    "Start from key {start_key} and follow the chain of KB records; each "
    "'next' value is a key. At the end a record contains 'code'. Give it back "
    "as \\boxed{{code}}.",
    # 8-9: development
    "A chain of records is hidden in the knowledge base. Its head key is "
    "{start_key}. Hop from record to record via the 'next' field until a "
    "record exposes 'code', then commit \\boxed{{code}}.",
    "Retrieve the terminal code of the chain whose first key is {start_key}. "
    "kb_lookup each key; 'next' names the successor; the last record carries "
    "'code'. Reply with \\boxed{{code}}.",
    # 10-11: evaluation
    "Chase the linked keys: the opening key is {start_key}. Every kb_lookup "
    "result points at the next key via 'next' until one instead holds 'code'. "
    "Your answer is \\boxed{{code}}.",
    "The knowledge base stores a chain starting at {start_key}. Follow the "
    "'next' pointers with kb_lookup; the chain terminates in a record with a "
    "'code' field. Submit \\boxed{{code}}.",
)


def template_text(template_id: int, fields: dict | None = None) -> str:
    """The raw paraphrase template behind a rendered prompt (structural id)."""
    return TEMPLATES[template_id]


def render_prompt(template_id: int, fields: dict) -> str:
    return template_text(template_id).format(**fields)


class LookupChainEnv:
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
