"""The committed answer grammar: `ANSWER: <value>` with `\\boxed{}` as fallback.

The two frozen forms are asked for by two different prompts at the same time --
the neutral system prompt requires a final `ANSWER: <value>` line, the generated
task prompt says "Reply with \\boxed{code}" -- so a compliant model writes
`ANSWER: \\boxed{x}`. Reading only the `ANSWER:` token committed the literal
string `\\boxed{x}`, which matches no ground truth and has no source in any tool
observation, so seven of twelve dev episodes were scored wrong AND labelled
hallucinations while answering correctly.

`REGRESSION_ROWS` below is the verbatim final assistant text of the twelve
episodes of the `verify-a5000` dev run that exposed this
(`out/verify-a5000/traces/B0.clean.none.jsonl`, sha256
cdeff6bff94c40b2e3401fb31c4c7084e2a81273df5b476d6bf7632100b7ba27, produced at
git 0a1cda86318408134b268888f3e98db56b12d81d). `out/` is not committed, so the
rows are inlined here to keep the discriminating rescore permanently runnable
without a GPU; `test_full_trace_certification_rescore` re-runs the same check
through the real certification path whenever that run's artifacts are present.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from agentlab import provenance
from agentlab.suite import verify
from agentlab.suite.schema import extract_committed_answer

REPO = pathlib.Path(__file__).resolve().parents[2]
TRACE = REPO / "out" / "verify-a5000" / "traces" / "B0.clean.none.jsonl"
SECRET = REPO / "out" / "verify-a5000" / "secret.hex"


# --------------------------------------------------------------------------
# the four frozen precedence rules
# --------------------------------------------------------------------------

def test_last_answer_wins():
    assert extract_committed_answer("ANSWER: 41\nsorry\nANSWER: 42") == "42"


def test_nested_box_inside_answer_is_unwrapped():
    assert extract_committed_answer("ANSWER: \\boxed{abc123}") == "abc123"


def test_boxed_is_accepted_when_there_is_no_answer_line():
    assert extract_committed_answer(
        "first \\boxed{41} then \\boxed{42}") == "42"


def test_later_wrong_answer_is_not_rescued_by_an_earlier_box():
    """The box is a FALLBACK, never a second chance."""
    assert extract_committed_answer(
        "The code is \\boxed{correct_code}\n\nANSWER: wrong_code") == "wrong_code"


def test_later_wrong_boxed_answer_is_not_rescued_by_an_earlier_answer_line():
    assert extract_committed_answer(
        "ANSWER: correct_code\n\nANSWER: \\boxed{wrong_code}") == "wrong_code"


# --------------------------------------------------------------------------
# the unwrap stays narrow
# --------------------------------------------------------------------------

def test_prose_boxes_are_not_scraped_when_the_answer_value_is_plain():
    """A box elsewhere in the reasoning must not displace the committed value."""
    assert extract_committed_answer(
        "I first guessed \\boxed{41}, then checked it.\n\nANSWER: 42") == "42"


def test_unwrap_is_anchored_at_the_start_of_the_committed_value():
    assert extract_committed_answer("ANSWER: v42\\boxed{41}") == "v42\\boxed{41}"


def test_box_contents_may_contain_spaces():
    """Anchoring on the text, not on the whitespace-truncated token."""
    assert extract_committed_answer("ANSWER: \\boxed{1 234}") == "1 234"


def test_trailing_punctuation_after_a_box_is_dropped():
    assert extract_committed_answer("ANSWER: \\boxed{abc123}.") == "abc123"


def test_thousands_separator_inside_a_box_survives_for_numeric_answers():
    assert extract_committed_answer("ANSWER: \\boxed{1,234}") == "1,234"


def test_empty_box_is_not_treated_as_a_committed_value():
    assert extract_committed_answer("ANSWER: \\boxed{}") == "\\boxed{}"


def test_no_commitment_at_all_is_none():
    assert extract_committed_answer("the code is abc123") is None
    assert extract_committed_answer("") is None
    assert extract_committed_answer(None) is None


def test_plain_answer_line_behaviour_is_unchanged():
    assert extract_committed_answer("ANSWER: 42.") == "42"
    assert extract_committed_answer("answer : abc123") == "abc123"
    assert extract_committed_answer("ANSWER: `abc123`\n\\boxed{zz}") == "zz"


# --------------------------------------------------------------------------
# one grammar, three readers
# --------------------------------------------------------------------------

GRAMMAR_BATTERY = (
    "ANSWER: 42",
    "ANSWER: \\boxed{42}",
    "\\boxed{42}",
    "ANSWER: \\boxed{abc123}.",
    "no commitment here",
    "\\boxed{41}\n\nANSWER: 42",
)


@pytest.mark.parametrize("text", GRAMMAR_BATTERY)
def test_certification_layer_reads_the_same_commitment(text):
    """provenance must delegate, never recompile the grammar (drift guard)."""
    assert provenance.extract_final_answer(text) == extract_committed_answer(text)


@pytest.mark.parametrize("text", GRAMMAR_BATTERY)
def test_strict_verifier_reads_the_same_commitment(text):
    class _Spec:
        answer_kind = "token"
        answer = "42"

    got = extract_committed_answer(text)
    assert verify._answer_ok(_Spec(), text) is (got is not None
                                                and got.lower() == "42")


# --------------------------------------------------------------------------
# the discriminating rescore: the verify-a5000 12-episode dev run
# --------------------------------------------------------------------------

# (task_id, ground-truth answer, verbatim final assistant text)
REGRESSION_ROWS = (
    ("dev-lookup_chain-h2-0000", "305100346995a299f7e3eed136a68f20",
     'The terminal code for the chain starting with KK2OMXNND7R4Y4VJR is 12345.'),
    ("dev-lookup_chain-h2-0007", "55640a29d5971894a9d2349715e5eeb3",
     'ANSWER: 55640a29d5971894a9d2349715e5eeb3'),
    ("dev-lookup_chain-h2-0003", "99cab0e44c1ba86c7af796cfed1708fd",
     'The terminal code of the chain is 99cab0e44c1ba86c7af796cfed1708fd.\n\nANSWER: 99cab0e44c1ba86c7af796cfed1708fd'),
    ("dev-lookup_chain-h2-0004", "86bba1699f5a805b0fbd58312bea962a",
     'The terminal code of the chain is 86bba1699f5a805b0fbd58312bea962a.\n\nANSWER: \\boxed{86bba1699f5a805b0fbd58312bea962a}'),
    ("dev-lookup_chain-h2-0006", "00ed76148ea158f1a4ad1d44e70c98b1",
     'The terminal code of the chain is 00ed76148ea158f1a4ad1d44e70c98b1.\n\nANSWER: \\boxed{00ed76148ea158f1a4ad1d44e70c98b1}'),
    ("dev-lookup_chain-h2-0002", "4d3f0df9727d6d383d5b4e018377fbd0",
     'The terminal code of the chain is 4d3f0df9727d6d383d5b4e018377fbd0.\n\nANSWER: \\boxed{4d3f0df9727d6d383d5b4e018377fbd0}'),
    ("dev-lookup_chain-h2-0001", "2d921ec00d34d18a499ddb0d2bd016a8",
     "The chain of records has been traced. The head key was K3VVZJIOZR4B4DF5B, which pointed to the next key KBRNWAVOKGIILKTVD. That record contained a 'note' field and a 'next' field, leading to the final record which exposed the 'code'.\n\nThe code is: 2d921ec00d34d18a499ddb0d2bd016a8\n\nANSWER: \\boxed{2d921ec00d34d18a499ddb0d2bd016a8}"),
    ("dev-lookup_chain-h2-0005", "e9d3fb85bfeaa5bde4c59016e366336b",
     'The chain of records has been traced. Starting from the head key `KIPRIZHG6FULWIHP4`, the next record was `KN6UMDU3SN4KPDBZW`, which contained the note "pallet 57, aisle D7" and pointed to the next record. The final record exposed the `code` field.\n\nThe code is: e9d3fb85bfeaa5bde4c59016e366336b\n\nANSWER: \\boxed{e9d3fb85bfeaa5bde4c59016e366336b}'),
    ("dev-lookup_chain-h2-0008", "0323cbef63b2cb4e9dd3ce42c751ba0e",
     'The terminal code of the chain whose first key is K4TJSV3BXN2PSUL2H is 0323cbef63b2cb4e9dd3ce42c751ba0e.\n\nANSWER: \\boxed{0323cbef63b2cb4e9dd3ce42c751ba0e}'),
    ("dev-lookup_chain-h2-0011", "c36e05a75651253c37817931fb5a2b0e",
     'ANSWER: c36e05a75651253c37817931fb5a2b0e'),
    ("dev-lookup_chain-h2-0009", "eae5b7561ad41db89de1070c4f23f97b",
     'The terminal code of the chain is eae5b7561ad41db89de1070c4f23f97b.\n\nANSWER: \\boxed{eae5b7561ad41db89de1070c4f23f97b}'),
    ("dev-lookup_chain-h2-0010", "7dfe06550c36cfee7ce6d94ddee3a479",
     'The chain of records has been traced. The final record exposed the code.\n\nANSWER: 7dfe06550c36cfee7ce6d94ddee3a479'),
)

# The four episodes that were already scored correct, and the seven that the
# defect scored wrong while the model answered correctly. Named individually so a
# future grammar change cannot quietly move an episode between the two sets.
ALREADY_CORRECT = ("dev-lookup_chain-h2-0007", "dev-lookup_chain-h2-0003",
                   "dev-lookup_chain-h2-0011", "dev-lookup_chain-h2-0010")
RESCUED_BY_UNWRAP = ("dev-lookup_chain-h2-0004", "dev-lookup_chain-h2-0006",
                     "dev-lookup_chain-h2-0002", "dev-lookup_chain-h2-0001",
                     "dev-lookup_chain-h2-0005", "dev-lookup_chain-h2-0008",
                     "dev-lookup_chain-h2-0009")
# Never rescued: no commitment in any form, and the value is wrong anyway.
STILL_WRONG = ("dev-lookup_chain-h2-0000",)

_PRE_FIX_ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\s`*]+)", re.IGNORECASE)
_PRE_FIX_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _pre_fix_extract(final_text: str) -> str | None:
    """The defective grammar, verbatim, so "before" is measured and not recalled."""
    hits = _PRE_FIX_ANSWER_RE.findall(final_text or "")
    if hits:
        return hits[-1].strip().rstrip(".,;")
    boxed = _PRE_FIX_BOXED_RE.findall(final_text or "")
    if boxed:
        return boxed[-1].strip()
    return None


def test_regression_rows_cover_the_whole_run():
    assert len(REGRESSION_ROWS) == 12
    ids = {r[0] for r in REGRESSION_ROWS}
    assert ids == set(ALREADY_CORRECT) | set(RESCUED_BY_UNWRAP) | set(STILL_WRONG)
    assert len(ids) == 12


def test_before_the_fix_the_run_scored_4_of_12():
    correct = {tid for tid, ans, final in REGRESSION_ROWS
               if _pre_fix_extract(final) == ans}
    assert correct == set(ALREADY_CORRECT)
    assert len(correct) == 4


def test_after_the_fix_the_run_scores_11_of_12():
    correct = {tid for tid, ans, final in REGRESSION_ROWS
               if extract_committed_answer(final) == ans}
    assert correct == set(ALREADY_CORRECT) | set(RESCUED_BY_UNWRAP)
    assert len(correct) == 11


@pytest.mark.parametrize("task_id,answer,final", [
    (t, a, f) for t, a, f in REGRESSION_ROWS if t in RESCUED_BY_UNWRAP])
def test_each_rescued_episode_committed_a_nested_box(task_id, answer, final):
    assert final.endswith("ANSWER: \\boxed{%s}" % answer)
    assert _pre_fix_extract(final) == "\\boxed{%s}" % answer
    assert extract_committed_answer(final) == answer


def test_the_one_genuine_failure_is_still_a_failure():
    (task_id, answer, final), = [r for r in REGRESSION_ROWS
                                 if r[0] in STILL_WRONG]
    assert extract_committed_answer(final) is None
    assert answer not in final


# --------------------------------------------------------------------------
# same rescore through the real certification path, when the run is on disk
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (TRACE.exists() and SECRET.exists()),
                    reason="verify-a5000 run artifacts are not in the tree (out/ is not committed)")
def test_the_verify_a5000_dev_run_is_invalidated_by_the_unified_contract():
    """Those twelve episodes were produced under the RETIRED environment.

    They are excluded, clean-only apparatus evidence: they carried no
    fault-recovery outcomes and never entered the study trace set. What they DO
    still document is the drift this reconciliation closed, visibly, in real
    bytes:

      * no `environment_contract_sha256`, so the invalidation rule refuses them;
      * the retired event format (`exposed_digest` / `decision` /
        `fault_emitted`) instead of the canonical `TraceEvent` fields;
      * an EMPTY assistant message where the offline rollout carried a structured
        tool-call object, and a tool result with no `name` -- the transcript-shape
        drift the model can condition on;
      * no canonical verdict at all, so oracle-node completion, the final state,
        capability-token provenance and the call budget were never checked.

    The answer-grammar rescore this module exists for is preserved in
    `REGRESSION_ROWS` above, which is the discriminating evidence and needs no
    GPU. Certifying these rows through the current certifier is refused rather
    than emulated: a legacy-format compatibility path is exactly how a retired
    environment gets pooled back into a claim.
    """
    from agentlab.suite import contract

    secret = bytes.fromhex(SECRET.read_text().strip())
    rows = [json.loads(line) for line in
            TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12

    inlined = {tid: final for tid, _ans, final in REGRESSION_ROWS}
    assert {r["task_id"] for r in rows} == set(inlined)

    for row in rows:
        # the inlined fixture is the real thing, byte for byte
        assert provenance._final_assistant_text(row) == inlined[row["task_id"]]
        # (1) the invalidation rule refuses the row
        assert not contract.is_current(row)
        with pytest.raises(SystemExit):
            contract.require_current(row, "the verify-a5000 dev trace")
        # (2) the retired event format, and therefore no valid receipt chain
        #     under the canonical field names
        event = row["events"][0]
        assert "exposed_digest" in event and "exposed_result_digest" not in event
        assert "decision" in event and "decision_id" not in event
        rep = provenance.certify_episode(row, secret)
        assert not rep["receipts_ok"]
        assert not rep["certified_success"]
        # (3) no canonical verdict was ever recorded
        assert not rep["verdict_present"]
        # (4) the transcript-shape drift, in the real trace
        assistants = [m for m in row["messages"] if m.get("role") == "assistant"]
        assert any(m.get("content") == "" and "tool_calls" not in m
                   for m in assistants), "the empty-assistant-turn drift"
        assert all("name" not in m for m in row["messages"]
                   if m.get("role") == "tool"), "the nameless-tool-result drift"


def test_the_unified_contract_emits_the_shape_that_run_was_missing():
    """The same two drifts, asserted positively on a current episode."""
    from agentlab.chat import assistant_tool_message
    from agentlab.suite import runtime as rt_mod
    from agentlab.suite.generate import build_task

    bundle = build_task("agentlab-suite-v1", 0xA61E0005, "eval", "lookup_chain", 2,
                        0, None)
    msg = assistant_tool_message("thinking out loud",
                                 [{"name": "kb_lookup", "arguments": {"key": "K1"}}])
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["function"]["name"] == "kb_lookup"
    assert msg["content"] == "thinking out loud"

    rt = rt_mod.EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes,
                               secret=bytes.fromhex("5c" * 32))
    rt.begin_decision()
    text = rt.dispatch(bundle.nodes[0].tool, dict(bundle.nodes[0].args))
    tool_msg = {"role": "tool", "name": bundle.nodes[0].tool, "content": text}
    assert tool_msg["name"] == "kb_lookup"
    assert rt_mod.parse_observation(text)["receipt"]
    assert verify.RUNAWAY_TERMINATIONS  # the module is the one that owns runaway
