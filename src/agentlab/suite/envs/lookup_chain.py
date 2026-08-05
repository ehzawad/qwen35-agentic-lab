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


def _note(rng) -> str:
    return (f"{rng.choice(_THINGS)} {rng.randint(2, 97)}, "
            f"aisle {rng.choice(_AISLES)}{rng.randint(1, 9)}")


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
            rec = {"next": keys[i + 1], "note": _note(rng)}
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
