"""Receipts, certification, and the preregistered non-recovery cases.

`certify_episode` is no longer a second definition of success. It recomputes the
LEDGER-side conditions from the raw trace -- receipt chain, runaway,
hallucination, the answer's validated source -- and cross-checks them against the
canonical verdict `suite.verify.verify_episode` emitted. So these tests check two
things at once: the ledger recomputation has teeth, and the verifier's oracle,
final-state, capability-token and budget conditions are inside the same boolean.

The cross-check is ASYMMETRIC, and the last section of this module is why. The
ledger conditions are a strict SUBSET of the canonical conjuncts, so the two
booleans are not the same predicate:

  * the five conjuncts both sides compute must be EQUAL, field for field;
  * `certified_success` must IMPLY `ledger_ok`;
  * `ledger_ok and not certified_success` is a LEGITIMATE strict refusal for a
    reason the transcript cannot show, and must not read as a disagreement.

Requiring plain equality was the S17 cross-predicate seam: it turned every blind
retry, and every clean episode that batched a dependent hop into one decision,
into a harness BUG that vetoed every gate, claim and the winner.
"""

from agentic_helpers import (SECRET, Guesser, ScriptedOracle, chain_spec,
                             faulted_variant, relay_spec, run)

from agentlab import provenance
from agentlab.suite.faults import TOKEN_ARG
from agentlab.suite.runtime import parse_observation, recovery_token_in


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

def test_receipt_roundtrip_and_tamper_detection():
    text = '{"ok":true,"value":"42"}'
    digest = provenance.observation_digest(text)
    receipt = provenance.mint_receipt(SECRET, "task-1", 3, digest)
    event = {"call_id": 3, "exposed_text": text,
             "exposed_result_digest": digest, "receipt": receipt}
    assert provenance.receipt_valid(SECRET, "task-1", event)
    # tampering with the observation invalidates the chain
    bad = dict(event, exposed_text='{"ok":true,"value":"43"}')
    assert not provenance.receipt_valid(SECRET, "task-1", bad)
    # a receipt minted for another episode/call never validates
    assert not provenance.receipt_valid(SECRET, "task-2", event)
    assert not provenance.receipt_valid(b"\x00" * 32, "task-1", event)
    # the RETIRED field name is not silently accepted: an old-format event has no
    # `exposed_result_digest`, and a compatibility fallback here is exactly how a
    # pre-reconciliation trace would get certified under the new contract
    legacy = {"call_id": 3, "exposed_text": text, "exposed_digest": digest,
              "receipt": receipt}
    assert not provenance.receipt_valid(SECRET, "task-1", legacy)


def test_certified_success_requires_valid_receipts():
    spec = chain_spec(30, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    assert trace["score"]["certified_success"]
    assert trace["score"]["verdict_agrees"]
    trace["events"][0]["exposed_text"] = trace["events"][0]["exposed_text"][:-1] + " "
    rep = provenance.certify_episode(trace, SECRET)
    assert not rep["receipts_ok"]
    assert not rep["certified_success"]
    # and the tamper is VISIBLE as a disagreement with the recorded verdict, in
    # both of the ways it should be: the two sides read the SAME conjunct
    # differently, and the strict verdict no longer implies the ledger conditions
    assert rep["verdict_present"]
    assert rep["verdict_certified_success"]
    assert not rep["verdict_agrees"]
    assert any(d.startswith("receipts_ok") for d in rep["verdict_shared_mismatches"])
    assert rep["ledger_contradiction"] is not None
    assert "receipts_ok" in rep["ledger_contradiction"]
    assert rep["strict_refusal"] is False


def test_a_trace_without_a_canonical_verdict_can_never_be_certified():
    """Oracle completion, final state, token provenance and the call budget are
    not recoverable from the transcript, so their absence is not certifiable."""
    spec = chain_spec(31, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    stripped = dict(trace)
    stripped["verdict"] = None
    rep = provenance.certify_episode(stripped, SECRET)
    assert rep["raw_success"]
    assert rep["ledger_ok"]
    assert not rep["verdict_present"]
    assert not rep["certified_success"]


def test_every_observation_in_a_trace_carries_a_receipt():
    spec = chain_spec(32, horizon=4)
    trace = run(spec, ScriptedOracle(spec))
    assert trace["events"]
    for event in trace["events"]:
        assert provenance.receipt_valid(SECRET, spec["task_id"], event)
    for msg in trace["messages"]:
        if msg["role"] == "tool":
            assert parse_observation(msg["content"])["receipt"]


# ---------------------------------------------------------------------------
# the non-recovery cases
# ---------------------------------------------------------------------------

def _fspec(i, cls="transient", horizon=4):
    """A spec that COMMITS one fault of class `cls`."""
    return chain_spec(i, horizon=horizon, fault_class=cls)


def _fault_node_index(spec):
    return (spec.get("fault") or {})["node_index"]


class SequencePolicy:
    """Plays a fixed list of calls (dicts), then answers."""

    def __init__(self, steps, answer):
        self.steps = list(steps)
        self.answer = answer
        self.idx = 0

    def __call__(self, messages, tools):
        if self.idx < len(self.steps):
            call = self.steps[self.idx]
            self.idx += 1
            return {"content": "", "tool_calls": [call]}
        return {"content": f"ANSWER: {self.answer}", "tool_calls": []}


class _TokenAwareSequence(SequencePolicy):
    """SequencePolicy that adds the recovery token to the NEXT queued call
    whenever the previous observation was an error carrying one."""

    def __call__(self, messages, tools):
        for m in reversed(messages):
            if m.get("role") == "assistant":
                break
            if m.get("role") == "tool":
                token = recovery_token_in(str(m.get("content", "")))
                if token and self.idx < len(self.steps):
                    step = self.steps[self.idx]
                    self.steps[self.idx] = {
                        "name": step["name"],
                        "arguments": dict(step["arguments"], **{TOKEN_ARG: token})}
                break
        return super().__call__(messages, tools)


def test_case1_not_exposed_stays_in_itt_denominator():
    spec = _fspec(33)
    trace = run(spec, Guesser("nope"), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["assigned"] and not rec["exposed"]
    assert rec["reason"] == "not_exposed"
    assert not rec["certified_recovery"]


def test_case2_guessed_answer_after_exposure_is_not_recovery():
    """Reaches the fault, then GUESSES the terminal token: no validated source."""
    spec = _fspec(34, horizon=4)
    nodes = spec["oracle_nodes"]
    idx = _fault_node_index(spec)
    steps = [{"name": n["tool"], "arguments": dict(n["args"])}
             for n in nodes[:idx + 1]]
    trace = run(spec, SequencePolicy(steps, spec["answer"]), condition="faulted")
    rep = provenance.certify_episode(trace, SECRET)
    assert rep["raw_success"]
    assert rep["hallucination"]["hallucinated"]  # answer value unvalidated
    rec = trace["score"]["recovery"]
    assert not rec["certified_recovery"]
    assert rec["reason"] in ("hallucinated", "unvalidated_answer")


def test_case3_blind_retry_without_token_is_not_recovery():
    spec = _fspec(35)
    trace = run(spec, ScriptedOracle(spec, blind_retry=True), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "blind_retry"
    assert not rec["certified_recovery"]
    # operationally the task may still succeed -- that is exactly the point
    assert trace["score"]["raw_success"]
    assert not trace["score"]["certified_success"]
    assert trace["verdict"]["recovery_reason"] == "blind_retry"


def test_case4_pre_fault_answer_is_not_recovery():
    """The answer's validated source precedes the fault."""
    spec = _fspec(36, horizon=2)
    nodes = spec["oracle_nodes"]
    idx = _fault_node_index(spec)
    terminal = nodes[-1]
    faulted = nodes[idx]
    if terminal["node_id"] == faulted["node_id"]:
        return  # the fault sits on the terminal node: not this case
    steps = [{"name": terminal["tool"], "arguments": dict(terminal["args"])},
             {"name": faulted["tool"], "arguments": dict(faulted["args"])},
             {"name": faulted["tool"], "arguments": dict(faulted["args"])}]
    trace = run(spec, _TokenAwareSequence(steps, spec["answer"]),
                condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "pre_fault_answer"
    assert not rec["certified_recovery"]


def test_case5_fabricated_receipt_is_hallucination():
    spec = chain_spec(37, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    fake = "r-" + "0" * 32
    trace["messages"].append({"role": "assistant",
                              "content": f"verified by receipt {fake}\n"
                                         f"ANSWER: {spec['answer']}"})
    rep = provenance.certify_episode(trace, SECRET)
    assert rep["hallucination"]["hallucinated"]
    assert "cited_unminted_receipt" in rep["hallucination"]["reasons"]
    assert not rep["certified_success"]


def test_case6_answer_after_runaway_is_not_recovery():
    spec = _fspec(38, horizon=2)
    nodes = spec["oracle_nodes"]
    idx = _fault_node_index(spec)
    faulted, terminal = nodes[idx], nodes[-1]
    bad = {"name": "kb_lookup", "arguments": {"key": "eval-a-no-such-key"}}
    steps = ([{"name": faulted["tool"], "arguments": dict(faulted["args"])},
              {"name": faulted["tool"], "arguments": dict(faulted["args"])}]
             + [bad] * 4
             + [{"name": terminal["tool"], "arguments": dict(terminal["args"])}])
    trace = run(spec, _TokenAwareSequence(steps, spec["answer"]),
                condition="faulted")
    rep = provenance.certify_episode(trace, SECRET)
    assert rep["runaway"]["runaway"]
    assert "identical_error_loop" in rep["runaway"]["reasons"]
    rec = trace["score"]["recovery"]
    assert rec["reason"] == "runaway"
    assert not rec["certified_recovery"]


def test_no_remediation_when_the_agent_gives_up():
    spec = _fspec(39)
    trace = run(spec, ScriptedOracle(spec, abandon_on_error=True),
                condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "no_remediation"
    assert not trace["score"]["raw_success"]


def test_the_equality_cap_is_not_runaway():
    """A successful episode using exactly max_calls is within budget."""
    spec = chain_spec(40, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    trace["budgets"] = dict(trace["budgets"], max_calls=len(trace["events"]))
    assert not provenance.detect_runaway(trace)["runaway"]
    trace["budgets"] = dict(trace["budgets"], max_calls=len(trace["events"]) - 1)
    run_report = provenance.detect_runaway(trace)
    assert run_report["runaway"] and "over_call_cap" in run_report["reasons"]


# ---------------------------------------------------------------------------
# certified orchestration
# ---------------------------------------------------------------------------

def test_orchestration_certified_when_all_tools_feed_the_answer():
    spec = relay_spec(41)
    trace = run(spec, ScriptedOracle(spec))
    orch = trace["score"]["orchestration"]
    assert orch["certified_orchestration"], orch
    assert set(orch["contributing_tools"]) >= {"kb_lookup", "unit_convert",
                                               "calculator"}


def test_decorative_calls_do_not_count_as_orchestration():
    # a chain task: correct answer, but calculator/unit_convert are decorative
    spec = chain_spec(42, horizon=2)
    spec["all_tools_required"] = True
    steps = [{"name": "calculator", "arguments": {"expression": "1+1"}},
             {"name": "unit_convert", "arguments": {"value": 1, "from_unit": "g",
                                                    "to_unit": "kg"}}]
    steps += [{"name": n["tool"], "arguments": dict(n["args"])}
              for n in spec["oracle_nodes"]]
    trace = run(spec, SequencePolicy(steps, spec["answer"]))
    orch = trace["score"]["orchestration"]
    assert not orch["certified_orchestration"]
    assert "tools_not_on_causal_chain" in orch["reason"]


# ---------------------------------------------------------------------------
# absent-information control
# ---------------------------------------------------------------------------

def test_redacted_control_yields_zero_certified_success_by_construction():
    spec = chain_spec(43, horizon=4)
    red = provenance.redact_spec(spec)
    trace = run(red, ScriptedOracle(red), control="redacted")
    scores = provenance.certify_redacted(trace, SECRET)
    assert scores == {"raw_success": False, "certified_success": False}


def test_redacted_control_flags_a_leaking_harness():
    # if the hidden value somehow leaks (here: a guesser that knows it),
    # raw_success goes true and the analyzer must call BUG (S11)
    spec = chain_spec(44, horizon=2)
    red = provenance.redact_spec(spec)
    trace = run(red, Guesser(spec["answer"]), control="redacted")
    scores = provenance.certify_redacted(trace, SECRET)
    assert scores["raw_success"] is True       # the leak signal
    assert scores["certified_success"] is False  # never certifiable


def test_the_redacted_control_actually_runs_the_policy():
    """The regression that made S11 a vacuous pass.

    Redacting the hidden record breaks the oracle path BY CONSTRUCTION. The old
    `SpecRuntime` replayed the oracle at construction time and aborted the episode
    as a `spec_error` unless the control was `redacted` -- a special case that
    existed only because the abort did. The canonical runtime needs neither: it
    exposes `no_entry`, credits nothing, and the policy really runs.
    """
    spec = chain_spec(45, horizon=4)
    red = provenance.redact_spec(spec)
    trace = run(red, ScriptedOracle(red), control="redacted")
    runner = trace["runner"]
    assert runner["termination_reason"] != "spec_error", runner
    assert runner["n_decisions"] >= 1 and runner["n_calls"] >= 1, runner
    # the absent lookup really is exercised, and returns only no_entry
    assert any("no_entry" in e["exposed_text"] for e in trace["events"])
    assert not any(spec["answer"] in e["exposed_text"] for e in trace["events"])
    # the canonical verdict says exactly why: the oracle chain cannot complete
    assert trace["verdict"]["unique_valid_nodes"] < trace["verdict"]["nodes_total"]
    assert provenance.certify_redacted(trace, SECRET) == {
        "raw_success": False, "certified_success": False}


def test_the_permutation_control_credits_the_permuted_value():
    """The control must measure the policy, not the control's own bookkeeping.

    Permutation rewrites the terminal record, so the committed canonical payload
    of that node describes the pre-permutation world. If the runtime kept it, a
    permuted episode would expose the permuted value, fail to be credited, and
    score zero certified success by construction -- a control that measures
    nothing.
    """
    specs = [chain_spec(i, horizon=2, ns="perm") for i in range(60)]
    permuted = provenance.permute_hidden_values(specs, seed=2786983944)
    assert permuted
    p = permuted[0]
    assert p["answer"] != next(s for s in specs
                               if s["task_id"] == p["task_id"])["answer"]
    assert p["spec_row"]["answer"] == str(p["answer"])
    trace = run(p, ScriptedOracle(p), control="permuted")
    assert trace["score"]["raw_success"], trace["verdict"]["reasons"][:3]
    assert trace["score"]["certified_success"], trace["verdict"]["reasons"][:3]


def test_committing_an_unobserved_value_is_a_hallucinated_result():
    """The frozen ER7 numerator, pinned in both directions.

    docs/AGENTIC_PROTOCOL.md s3: a hallucinated result is "citing an unminted
    receipt, tool-role content without an environment event, or a committed answer
    value absent from every validated observation". So committing ANY value the
    ledger never produced counts -- including a wrong one -- while an episode that
    commits nothing at all is a plain failure, not a fabrication.
    """
    spec = chain_spec(46, horizon=2)
    # (a) commits a value that was never observed -> hallucinated
    made_up = run(spec, Guesser("not-a-real-token"))
    assert provenance.certify_episode(made_up, SECRET)["hallucination"] == {
        "hallucinated": True, "reasons": ["answer_value_unvalidated"]}

    # (b) commits nothing -> a failure, but no fabrication
    class Abstain:
        def __call__(self, messages, tools):
            return {"content": "I cannot determine this value.", "tool_calls": []}

    quiet = run(spec, Abstain())
    rep = provenance.certify_episode(quiet, SECRET)
    assert rep["raw_success"] is False
    assert rep["hallucination"]["hallucinated"] is False


def test_a_fulfillment_completion_token_is_a_validated_source():
    """Reading only `value` labelled every correct fulfillment answer a fabrication."""
    event = {"exposed_text": '{"ok":true,"status":"complete",'
                             '"completion_token":"FIN-ABC123"}'}
    assert "FIN-ABC123" in [str(v) for v in provenance._exposed_values(event)]
    # bookkeeping keys can never source an answer
    assert "complete" not in [str(v) for v in provenance._exposed_values(event)]


def test_non_recovery_precedence_is_frozen_and_the_boolean_is_a_conjunction():
    """Order decides the LABEL only; certified_recovery is the conjunction."""
    # this episode both fabricates (correct answer, never retrieved) and never
    # remediates: `hallucinated` outranks `no_remediation` because a fabricating
    # trace is not evidence about remediation at all.
    spec = _fspec(47, horizon=4)
    nodes = spec["oracle_nodes"]
    idx = _fault_node_index(spec)
    steps = [{"name": n["tool"], "arguments": dict(n["args"])}
             for n in nodes[:idx + 1]]
    trace = run(spec, SequencePolicy(steps, spec["answer"]), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["reason"] == "hallucinated"
    assert set(rec["violations"]) >= {"hallucinated", "no_remediation"}
    assert rec["certified_recovery"] is False
    # the reported reason is always the highest-precedence violation present
    order = list(provenance.NON_RECOVERY_PRECEDENCE)
    labels = {"blind_retry": "remediation", "no_remediation": "remediation",
              "no_post_fault_result": "no_post_fault_result"}
    ranks = [order.index(labels.get(v, v)) for v in rec["violations"]]
    assert order.index(labels.get(rec["reason"], rec["reason"])) == min(ranks)


# ---------------------------------------------------------------------------
# the cross-predicate relation between the ledger and the canonical verdict
# ---------------------------------------------------------------------------

class _BatchedOracle:
    """Emits the WHOLE oracle path in one decision, then commits the answer.

    The registered crediting rule requires a dependency edge to cross a LATER
    assistant decision, so the dependent node is never credited and the strict
    verifier correctly refuses certification -- while the transcript alone (right
    answer, valid receipts, a validated source, no runaway, no fabrication) has
    nothing to object to. It is the fault-free member of the same class as a blind
    retry, and the second mechanism the dev preflight reproduced.
    """

    def __init__(self, spec: dict):
        replay = provenance.execute_oracle(spec)
        self.calls = [{"name": n["tool"], "arguments": dict(n["args"])}
                      for n in replay["nodes"]]
        self.answer = replay["answer"]
        self.sent = False

    def __call__(self, messages, tools):
        if not self.sent:
            self.sent = True
            return {"content": "both hops in one decision",
                    "tool_calls": [dict(c) for c in self.calls]}
        return {"content": f"done\nANSWER: {self.answer}", "tool_calls": []}


def _strict_refusals():
    """The two reproduced mechanisms, as (label, trace) pairs."""
    fault = _fspec(60, horizon=4)
    clean = chain_spec(61, horizon=2)
    return [("blind_retry", run(fault, ScriptedOracle(fault, blind_retry=True),
                                condition="faulted")),
            ("batched_decision", run(clean, _BatchedOracle(clean)))]


def test_a_legitimate_strict_refusal_is_not_a_disagreement():
    """Both mechanisms: the weaker predicate holds, the strict one does not."""
    for label, trace in _strict_refusals():
        rep = provenance.certify_episode(trace, SECRET)
        assert rep["raw_success"] is True, label
        assert rep["receipts_ok"] is True, label
        assert rep["runaway"]["runaway"] is False, label
        assert rep["hallucination"]["hallucinated"] is False, label
        assert rep["ledger_ok"] is True, label
        assert rep["verdict_certified_success"] is False, label
        # so the two predicates DIFFER -- and that is the strict verifier working
        assert rep["strict_refusal"] is True, label
        assert rep["verdict_agrees"] is True, label
        assert rep["verdict_shared_mismatches"] == [], label
        assert rep["ledger_contradiction"] is None, label
        assert rep["unexplained_refusal"] is False, label
        assert rep["strict_refusal_reasons"], label
        # the claim-bearing boolean is still the strict one
        assert rep["certified_success"] is False, label
        assert trace["score"]["certified_success"] is False, label


def test_the_five_shared_conjuncts_are_compared_field_for_field():
    """A scorer that mis-reads a correct answer moves one side only, and is seen."""
    spec = chain_spec(62, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    assert trace["score"]["certified_success"]
    for name, wrong in (("raw_success", False), ("receipts_ok", False),
                        ("runaway", True), ("hallucinated", True),
                        ("answer_event_call_id", None)):
        tampered = dict(trace, verdict=dict(trace["verdict"], **{name: wrong}))
        rep = provenance.certify_episode(tampered, SECRET)
        assert any(d.startswith(name) for d in rep["verdict_shared_mismatches"]), \
            (name, rep["verdict_shared_mismatches"])
        assert rep["verdict_agrees"] is False, name
    # every registered shared field is actually checked, and no other one is
    assert provenance.SHARED_LEDGER_VERDICT_FIELDS == (
        "raw_success", "receipts_ok", "runaway", "hallucinated",
        "answer_event_call_id")
    for name in provenance.SHARED_LEDGER_VERDICT_FIELDS:
        stripped = dict(trace, verdict={k: v for k, v in trace["verdict"].items()
                                        if k != name})
        rep = provenance.certify_episode(stripped, SECRET)
        assert any(d.startswith(f"{name}: absent")
                   for d in rep["verdict_shared_mismatches"]), name


def test_certified_success_must_imply_the_ledger_conditions():
    """The one direction that is a defect, over each ledger conjunct in turn."""
    import copy

    spec = chain_spec(63, horizon=2)
    good = run(spec, ScriptedOracle(spec))
    # a verdict that claims certification while the transcript refuses it: build it
    # by making the ledger side fail and leaving the recorded verdict untouched
    tampered = copy.deepcopy(good)
    ev = tampered["events"][0]
    ev["exposed_text"] = ev["exposed_text"][:-1] + " "
    rep = provenance.certify_episode(tampered, SECRET)
    assert rep["verdict_certified_success"] and not rep["ledger_ok"]
    assert rep["ledger_contradiction"] is not None
    assert rep["verdict_agrees"] is False
    # and the implication holds on every honest episode this module builds
    for label, trace in _strict_refusals() + [("clean", good)]:
        r = provenance.certify_episode(trace, SECRET)
        assert (not r["verdict_certified_success"]) or r["ledger_ok"], label


def test_a_refusal_that_records_no_reason_is_a_disagreement():
    """The only place a genuine strict-side defect could hide behind a refusal."""
    for label, trace in _strict_refusals():
        assert trace["verdict"]["reasons"], label
        silent = dict(trace, verdict=dict(trace["verdict"], reasons=[]))
        rep = provenance.certify_episode(silent, SECRET)
        assert rep["unexplained_refusal"] is True, label
        assert rep["strict_refusal"] is True, label


def test_the_certifier_no_longer_compares_two_different_predicates():
    """The seam, stated as source: no equality between the strict and weak booleans."""
    import pathlib
    import re

    src = (pathlib.Path(provenance.__file__)).read_text(encoding="utf-8")
    body = src.split("def certify_episode", 1)[1].split("\ndef ", 1)[0]
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))
    assert not re.search(r"verdict_certified\s*==\s*ledger_ok", code)
    assert not re.search(r"ledger_ok\s*==\s*verdict_certified", code)
