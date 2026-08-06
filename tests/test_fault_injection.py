"""Fault scheduling and the four faults through the CLAIM-BEARING episode loop.

`evaluate.run_episode` no longer owns a runtime. It builds the one
`suite.runtime.EpisodeRuntime` from the certification spec's exported
`spec_row`/`oracle_nodes`, so these tests exercise the same environment the
training path rolls out in, through the evaluator's own driver: the tool surface
it shows the policy, the envelopes the policy reads, the receipts, the canonical
verdict written into the trace, and the registered remediation predicate.
"""

import json

from agentic_helpers import (SECRET, ScriptedOracle, chain_spec, relay_spec, run)

from agentlab.suite import evaluate, faults
from agentlab.suite.faults import TOKEN_ARG
from agentlab.suite.runtime import parse_observation, recovery_token_in


def _fault_events(trace):
    return [e for e in trace["events"] if e.get("fault_triggered")]


def _envelope(text):
    objs = parse_observation(text)["objects"]
    return objs[0] if objs else {}


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

def test_schedule_is_deterministic_per_task_and_seed():
    spec = chain_spec(1, horizon=4)
    a = faults.schedule_fault(spec["task_id"], spec["oracle"], 0xA61E0007)
    b = faults.schedule_fault(spec["task_id"], spec["oracle"], 0xA61E0007)
    assert a == b
    c = faults.schedule_fault(spec["task_id"], spec["oracle"], 0xA61E0007 + 1)
    d = faults.schedule_fault("other-task", spec["oracle"], 0xA61E0007)
    assert a != c or a != d  # at least one of seed/task changes the draw


def test_wrong_unit_only_schedulable_on_unit_convert_nodes():
    chain = chain_spec(2, horizon=4)
    assert faults.eligible_nodes(chain["oracle"], "wrong_unit") == []
    assert faults.schedule_fault(chain["task_id"], chain["oracle"], 1,
                                 fault_class="wrong_unit") is None
    relay = relay_spec(2)
    assert faults.eligible_nodes(relay["oracle"], "wrong_unit") == [2]


def test_the_wrong_unit_candidates_are_the_same_family_and_different():
    from agentlab.tools import _UNITS

    cands = faults.wrong_unit_candidates("kg")
    assert cands and "kg" not in cands
    assert all(_UNITS[u][0] == "mass" for u in cands)
    assert cands == sorted(cands)


def test_the_committed_trap_unit_is_what_the_model_sees():
    """The runtime must not re-derive a unit; `SpecRuntime` did, and drifted."""
    spec = relay_spec(3, fault_class="wrong_unit")
    committed = spec["spec_row"]["faults"][0]["params"]["wrong_unit"]
    trace = run(spec, ScriptedOracle(spec), condition="faulted")
    fe = _fault_events(trace)
    assert len(fe) == 1
    assert _envelope(fe[0]["exposed_text"])["unit"] == committed


# ---------------------------------------------------------------------------
# the one tool surface the evaluated policy is shown
# ---------------------------------------------------------------------------

def test_the_evaluator_shows_the_canonical_schemas_with_the_token():
    from agentlab.suite.runtime import tool_schemas_for_family

    seen = {}

    def chat_fn(messages, tools):
        seen["tools"] = tools
        return {"content": "ANSWER: nope", "tool_calls": []}

    spec = chain_spec(4, horizon=2)
    run(spec, chat_fn)
    assert seen["tools"] == tool_schemas_for_family("lookup_chain")
    for schema in seen["tools"]:
        assert TOKEN_ARG in schema["function"]["parameters"]["properties"]


def test_evaluate_has_no_second_runtime():
    assert not hasattr(evaluate, "SpecRuntime")
    assert not hasattr(evaluate, "suite_tool_schemas")
    assert not hasattr(faults, "FaultInjector")
    assert not hasattr(faults, "wrong_unit_target")


# ---------------------------------------------------------------------------
# transient / rate limit / malformed through the full episode loop
# ---------------------------------------------------------------------------

def _faulted_run(spec, fault_class, node_index, policy=None):
    spec = chain_spec_like(spec, fault_class, node_index)
    policy = policy or ScriptedOracle(spec)
    return run(spec, policy, condition="faulted")


def chain_spec_like(spec, fault_class, node_index):
    """Re-canonicalize `spec` with one pinned fault at `node_index`."""
    from agentic_helpers import _canonicalize

    spec = dict(spec)
    spec.pop("spec_row", None)
    spec.pop("oracle_nodes", None)
    spec["fault"] = {"class": fault_class, "node_index": node_index}
    spec["faults"] = None
    return _canonicalize(spec)


def test_transient_emits_once_with_token_then_recovers():
    spec = chain_spec(10, horizon=4)
    trace = _faulted_run(spec, "transient", 2)
    fe = _fault_events(trace)
    assert len(fe) == 1
    assert fe[0]["fault_type"] == "transient"
    token = fe[0]["recovery_token"]
    assert token and len(token) == 32
    body = _envelope(fe[0]["exposed_text"])
    assert body["ok"] is False and body[TOKEN_ARG] == token
    # the remedial retry carried the token and got the canonical result
    later = [e for e in trace["events"]
             if e["call_id"] > fe[0]["call_id"]
             and e["canonical_args_digest"] == fe[0]["canonical_args_digest"]]
    assert later and later[0]["token_provided"] == token
    assert later[0]["exposed_canonical"] is True
    assert trace["score"]["raw_success"] and trace["score"]["certified_success"]
    assert trace["score"]["recovery"]["certified_recovery"]
    assert trace["verdict"]["recovery_reason"] == "ok"


def test_rate_limit_same_decision_retry_stays_limited():
    spec = chain_spec_like(chain_spec(11, horizon=2), "rate_limit", 0)
    runtime = evaluate.episode_runtime(spec, SECRET, "faulted")
    first_args = dict(spec["oracle_nodes"][0]["args"])
    runtime.begin_decision()
    out1 = runtime.dispatch("kb_lookup", dict(first_args))
    token = recovery_token_in(out1)
    assert _envelope(out1)["error"] == "rate_limit"
    out2 = runtime.dispatch("kb_lookup", dict(first_args, **{TOKEN_ARG: token}))
    assert _envelope(out2)["error"] == "rate_limit_active"   # same decision
    runtime.begin_decision()
    out3 = runtime.dispatch("kb_lookup", dict(first_args, **{TOKEN_ARG: token}))
    assert _envelope(out3)["ok"] is True                    # next decision
    assert len([e for e in runtime.events if e.fault_triggered]) == 1


def test_rate_limit_full_episode_recovers_on_later_decision():
    spec = chain_spec(12, horizon=4)
    trace = _faulted_run(spec, "rate_limit", 1)
    fe = _fault_events(trace)
    assert len(fe) == 1
    rec = trace["score"]["recovery"]
    assert rec["certified_recovery"], rec
    remedial = [e for e in trace["events"] if e.get("token_provided")]
    assert remedial and remedial[0]["decision_id"] > fe[0]["decision_id"]


def test_malformed_never_leaks_the_canonical_value():
    spec = chain_spec(13, horizon=4)
    trace = _faulted_run(spec, "malformed", 3)  # terminal node: the answer token
    fe = _fault_events(trace)
    assert len(fe) == 1
    assert fe[0]["exposed_text"].startswith('{"ok":true,"value":')
    assert spec["answer"] not in fe[0]["exposed_text"]
    assert '"truncated_result"' in fe[0]["exposed_text"]
    assert trace["score"]["recovery"]["certified_recovery"]


def test_wrong_unit_trap_and_corrected_target_recovery():
    spec = relay_spec(14)
    trace = _faulted_run(spec, "wrong_unit", 2)
    fe = _fault_events(trace)
    assert len(fe) == 1
    body = _envelope(fe[0]["exposed_text"])
    assert body["ok"] is True and body["unit"] != "kg"  # explicit, valid-looking
    assert TOKEN_ARG not in body                       # not an error envelope
    rec = trace["score"]["recovery"]
    assert rec["certified_recovery"], rec
    assert trace["score"]["certified_success"]


def test_a_blind_retry_is_operationally_fine_and_never_certified():
    """The registered distinction the tokenless environment could not express."""
    spec = chain_spec(15, horizon=4)
    fspec = chain_spec_like(spec, "transient", 1)
    trace = run(fspec, ScriptedOracle(fspec, blind_retry=True), condition="faulted")
    assert len(_fault_events(trace)) == 1
    assert trace["score"]["raw_success"] is True         # the task was solved
    assert trace["score"]["certified_success"] is False  # but not earned
    assert trace["verdict"]["recovery_reason"] == "blind_retry"
    assert trace["score"]["recovery"]["reason"] == "blind_retry"
    assert not trace["score"]["recovery"]["certified_recovery"]


def test_clean_condition_never_emits_faults_and_still_carries_receipts():
    spec = chain_spec(16, horizon=4)
    trace = run(spec, ScriptedOracle(spec), condition="clean")
    assert _fault_events(trace) == []
    assert trace["score"]["certified_success"]
    assert trace["verdict"]["fault_assigned"] == 0
    for event in trace["events"]:
        assert event["receipt"].startswith("r-")
        assert "event_id" not in json.loads(event["exposed_text"])
    for msg in trace["messages"]:
        if msg["role"] == "tool":
            assert parse_observation(msg["content"])["receipt"]
            assert msg["name"]           # the tool name survives the transcript


def test_stress_condition_carries_two_distinct_faults():
    spec = chain_spec(17, horizon=8)
    spec = dict(spec)
    spec.pop("spec_row", None)
    spec.pop("oracle_nodes", None)
    spec["fault"] = None
    spec["faults"] = [{"class": "transient", "node_index": 2},
                      {"class": "malformed", "node_index": 5}]
    from agentic_helpers import _canonicalize

    spec = _canonicalize(spec)
    trace = run(spec, ScriptedOracle(spec), condition="stress")
    fe = _fault_events(trace)
    assert len(fe) == 2
    assert {e["fault_type"] for e in fe} == {"transient", "malformed"}
    assert trace["score"]["raw_success"]
    assert len(trace["verdict"]["fault_reports"]) == 2
    assert all(r["reason"] == "ok" for r in trace["verdict"]["fault_reports"])


def test_the_condition_decides_the_budget_and_the_fault_count():
    spec = chain_spec(18, horizon=4, fault_class="transient")
    clean = run(spec, ScriptedOracle(spec), condition="clean")
    faulted = run(spec, ScriptedOracle(spec), condition="faulted")
    assert clean["budgets"] == {"max_decisions": 7, "max_calls": 12}
    assert faulted["budgets"] == {"max_decisions": 9, "max_calls": 12}
    assert clean["verdict"]["fault_assigned"] == 0
    assert faulted["verdict"]["fault_assigned"] == 1
