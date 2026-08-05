"""Receipts, certification, and the six preregistered non-recovery cases."""

from agentic_helpers import (SECRET, Guesser, ScriptedOracle, chain_spec,
                             relay_spec, run)

from agentlab import provenance


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

def test_receipt_roundtrip_and_tamper_detection():
    text = '{"ok":true,"value":"42"}'
    digest = provenance.observation_digest(text)
    receipt = provenance.mint_receipt(SECRET, "task-1", 3, digest)
    event = {"call_id": 3, "exposed_text": text, "exposed_digest": digest,
             "receipt": receipt}
    assert provenance.receipt_valid(SECRET, "task-1", event)
    # tampering with the observation invalidates the chain
    bad = dict(event, exposed_text='{"ok":true,"value":"43"}')
    assert not provenance.receipt_valid(SECRET, "task-1", bad)
    # a receipt minted for another episode/call never validates
    assert not provenance.receipt_valid(SECRET, "task-2", event)
    assert not provenance.receipt_valid(b"\x00" * 32, "task-1", event)


def test_certified_success_requires_valid_receipts():
    spec = chain_spec(30, horizon=2)
    trace = run(spec, ScriptedOracle(spec))
    assert trace["score"]["certified_success"]
    trace["events"][0]["exposed_text"] = trace["events"][0]["exposed_text"][:-1] + " "
    rep = provenance.certify_episode(trace, SECRET)
    assert not rep["receipts_ok"]
    assert not rep["certified_success"]


# ---------------------------------------------------------------------------
# the six non-recovery cases
# ---------------------------------------------------------------------------

def _fspec(i, cls="transient", node=1, horizon=4):
    spec = chain_spec(i, horizon=horizon)
    return dict(spec, fault={"class": cls, "node_index": node})


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


def test_case1_not_exposed_stays_in_itt_denominator():
    spec = _fspec(31)
    trace = run(spec, Guesser("nope"), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["assigned"] and not rec["exposed"]
    assert rec["reason"] == "not_exposed"
    assert not rec["certified_recovery"]


def test_case2_guessed_answer_after_exposure_is_not_recovery():
    # reaches the fault, remediates, but then GUESSES the terminal token
    # instead of retrieving it: correct raw answer, no validated source.
    spec = _fspec(32, node=0, horizon=4)
    replay = provenance.execute_oracle(spec)
    n0 = replay["nodes"][0]
    fault_call = {"name": "kb_lookup", "arguments": dict(n0["args"])}
    trace = run(spec, SequencePolicy([fault_call], spec["answer"]),
                condition="faulted")
    # the policy answered correctly without ever retrieving the token
    rep = provenance.certify_episode(trace, SECRET)
    assert rep["raw_success"]
    assert rep["hallucination"]["hallucinated"]  # answer value unvalidated
    rec = trace["score"]["recovery"]
    assert not rec["certified_recovery"]
    assert rec["reason"] in ("hallucinated", "unvalidated_answer")


def test_case3_blind_retry_without_token_is_not_recovery():
    spec = _fspec(33, node=1)
    trace = run(spec, ScriptedOracle(spec, blind_retry=True), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "blind_retry"
    assert not rec["certified_recovery"]
    # operationally the task may still succeed -- that is exactly the point
    assert trace["score"]["raw_success"]


def test_case4_pre_fault_answer_is_not_recovery():
    # terminal token fetched BEFORE the fault fires at node 0; then the fault
    # is exposed and properly remediated, but the answer's validated source
    # precedes the fault.
    spec = _fspec(34, node=0, horizon=2)
    replay = provenance.execute_oracle(spec)
    n0, n1 = replay["nodes"]
    terminal_key = n1["args"]["key"]
    steps = [
        {"name": "kb_lookup", "arguments": {"key": terminal_key}},  # answer, pre-fault
        {"name": "kb_lookup", "arguments": dict(n0["args"])},       # fault fires
        {"name": "kb_lookup", "arguments": dict(n0["args"])},       # token retry below
    ]
    trace = run(spec, _TokenAwareSequence(steps, spec["answer"]), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "pre_fault_answer"
    assert not rec["certified_recovery"]


class _TokenAwareSequence(SequencePolicy):
    """SequencePolicy that adds the recovery token to the NEXT queued call
    whenever the previous observation was an error carrying one."""

    def __call__(self, messages, tools):
        import json as _json

        for m in reversed(messages):
            if m.get("role") == "assistant":
                break
            if m.get("role") == "tool":
                for line in m["content"].splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            obj = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        if obj.get("ok") is False and obj.get("recovery_token") \
                                and self.idx < len(self.steps):
                            self.steps[self.idx] = {
                                "name": self.steps[self.idx]["name"],
                                "arguments": dict(self.steps[self.idx]["arguments"],
                                                  recovery_token=obj["recovery_token"])}
                break
        return super().__call__(messages, tools)


def test_case5_fabricated_receipt_is_hallucination():
    spec = chain_spec(35, horizon=2)
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
    spec = _fspec(36, node=0, horizon=2)
    replay = provenance.execute_oracle(spec)
    n0, n1 = replay["nodes"]
    bad = {"name": "kb_lookup", "arguments": {"key": "eval-a-no-such-key"}}
    steps = ([{"name": "kb_lookup", "arguments": dict(n0["args"])},  # fault fires
              {"name": "kb_lookup", "arguments": dict(n0["args"])}]  # token retry
             + [bad] * 4                                             # identical errors
             + [{"name": "kb_lookup", "arguments": dict(n1["args"])}])
    trace = run(spec, _TokenAwareSequence(steps, spec["answer"]), condition="faulted")
    rep = provenance.certify_episode(trace, SECRET)
    assert rep["runaway"]["runaway"]
    assert "identical_error_loop" in rep["runaway"]["reasons"]
    rec = trace["score"]["recovery"]
    assert rec["reason"] == "runaway"
    assert not rec["certified_recovery"]


def test_no_remediation_when_the_agent_gives_up():
    spec = _fspec(37, node=1)
    trace = run(spec, ScriptedOracle(spec, abandon_on_error=True), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["exposed"]
    assert rec["reason"] == "no_remediation"
    assert not trace["score"]["raw_success"]


# ---------------------------------------------------------------------------
# certified orchestration
# ---------------------------------------------------------------------------

def test_orchestration_certified_when_all_tools_feed_the_answer():
    spec = relay_spec(38)
    trace = run(spec, ScriptedOracle(spec))
    orch = trace["score"]["orchestration"]
    assert orch["certified_orchestration"], orch
    assert set(orch["contributing_tools"]) >= {"kb_lookup", "unit_convert",
                                               "calculator"}


def test_decorative_calls_do_not_count_as_orchestration():
    # a chain task: correct answer, but calculator/unit_convert are decorative
    spec = chain_spec(39, horizon=2)
    spec["all_tools_required"] = True
    replay = provenance.execute_oracle(spec)
    steps = [{"name": "calculator", "arguments": {"expression": "1+1"}},
             {"name": "unit_convert", "arguments": {"value": 1, "from_unit": "g",
                                                    "to_unit": "kg"}}]
    steps += [{"name": "kb_lookup", "arguments": dict(n["args"])}
              for n in replay["nodes"]]
    trace = run(spec, SequencePolicy(steps, spec["answer"]))
    orch = trace["score"]["orchestration"]
    assert not orch["certified_orchestration"]
    assert "tools_not_on_causal_chain" in orch["reason"]


# ---------------------------------------------------------------------------
# absent-information control
# ---------------------------------------------------------------------------

def test_redacted_control_yields_zero_certified_success_by_construction():
    spec = chain_spec(40, horizon=4)
    red = provenance.redact_spec(spec)
    trace = run(red, ScriptedOracle(red), control="redacted")
    scores = provenance.certify_redacted(trace, SECRET)
    assert scores == {"raw_success": False, "certified_success": False}


def test_redacted_control_flags_a_leaking_harness():
    # if the hidden value somehow leaks (here: a guesser that knows it),
    # raw_success goes true and the analyzer must call BUG (S11)
    spec = chain_spec(41, horizon=2)
    red = provenance.redact_spec(spec)
    trace = run(red, Guesser(spec["answer"]), control="redacted")
    scores = provenance.certify_redacted(trace, SECRET)
    assert scores["raw_success"] is True       # the leak signal
    assert scores["certified_success"] is False  # never certifiable


def test_the_redacted_control_actually_runs_the_policy():
    """The regression that made S11 a vacuous pass.

    Redacting the hidden record breaks the oracle path BY CONSTRUCTION. When the
    episode runtime treated that as a spec defect, every redacted episode became a
    zero-decision `spec_error` stub: the policy never ran, so no leak could ever
    be observed and S11 certified "zero success" over episodes nobody attempted.
    """
    spec = chain_spec(42, horizon=4)
    red = provenance.redact_spec(spec)
    trace = run(red, ScriptedOracle(red), control="redacted")
    runner = trace["runner"]
    assert runner["termination_reason"] != "spec_error", runner
    assert runner["n_decisions"] >= 1 and runner["n_calls"] >= 1, runner
    # the absent lookup really is exercised, and returns only no_entry
    assert any("no_entry" in e["exposed_text"] for e in trace["events"])
    assert not any(spec["answer"] in e["exposed_text"] for e in trace["events"])
    assert provenance.certify_redacted(trace, SECRET) == {
        "raw_success": False, "certified_success": False}


def test_committing_an_unobserved_value_is_a_hallucinated_result():
    """The frozen ER7 numerator, pinned in both directions.

    docs/AGENTIC_PROTOCOL.md s3: a hallucinated result is "citing an unminted
    receipt, tool-role content without an environment event, or a committed answer
    value absent from every validated observation". So committing ANY value the
    ledger never produced counts -- including a wrong one -- while an episode that
    commits nothing at all is a plain failure, not a fabrication. Narrowing this
    to "correct answer only" would silently shrink a preregistered gate.
    """
    spec = chain_spec(43, horizon=2)
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


def test_non_recovery_precedence_is_frozen_and_the_boolean_is_a_conjunction():
    """Order decides the LABEL only; certified_recovery is the conjunction."""
    # this episode both fabricates (correct answer, never retrieved) and never
    # remediates: `hallucinated` outranks `no_remediation` because a fabricating
    # trace is not evidence about remediation at all.
    spec = _fspec(44, node=0, horizon=4)
    n0 = provenance.execute_oracle(spec)["nodes"][0]
    trace = run(spec, SequencePolicy([{"name": "kb_lookup",
                                       "arguments": dict(n0["args"])}],
                                     spec["answer"]), condition="faulted")
    rec = trace["score"]["recovery"]
    assert rec["reason"] == "hallucinated"
    assert set(rec["violations"]) >= {"hallucinated", "no_remediation"}
    assert rec["certified_recovery"] is False
    # the reported reason is always the highest-precedence violation present
    order = list(provenance.NON_RECOVERY_PRECEDENCE)
    labels = {"blind_retry": "remediation", "no_remediation": "remediation"}
    ranks = [order.index(labels.get(v, v)) for v in rec["violations"]]
    assert order.index(labels.get(rec["reason"], rec["reason"])) == min(ranks)
