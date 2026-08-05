"""Fault schedules and the four injectors: determinism, exactly-once
emission, remediation tokens, and the operational recovery paths."""

import json

from agentic_helpers import SECRET, ScriptedOracle, chain_spec, relay_spec, run

from agentlab.suite import faults
from agentlab.suite.evaluate import SpecRuntime


def _fault_events(trace):
    return [e for e in trace["events"] if e.get("fault_emitted")]


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


def test_wrong_unit_target_is_same_family_and_different():
    from agentlab.tools import _UNITS

    t = faults.wrong_unit_target("kg", "task-x", 1)
    assert t != "kg" and _UNITS[t][0] == "mass"


# ---------------------------------------------------------------------------
# transient / rate limit / malformed through the full episode loop
# ---------------------------------------------------------------------------

def _faulted_run(spec, fault_class, node_index, policy=None):
    spec = dict(spec, fault={"class": fault_class, "node_index": node_index})
    policy = policy or ScriptedOracle(spec)
    return run(spec, policy, condition="faulted")


def test_transient_emits_once_with_token_then_recovers():
    spec = chain_spec(10, horizon=4)
    trace = _faulted_run(spec, "transient", 2)
    fe = _fault_events(trace)
    assert len(fe) == 1
    assert fe[0]["fault_class"] == "transient"
    token = fe[0]["recovery_token"]
    assert token and len(token) == 32
    body = json.loads(fe[0]["exposed_text"])
    assert body["ok"] is False and body["recovery_token"] == token
    # the remedial retry carried the token and got the canonical result
    later = [e for e in trace["events"] if e["call_id"] > fe[0]["call_id"]
             and e["args_digest"] == fe[0]["args_digest"]]
    assert later and later[0]["token_provided"] == token
    assert later[0]["exposed_digest"] == later[0]["canonical_digest"]
    assert trace["score"]["raw_success"] and trace["score"]["certified_success"]
    assert trace["score"]["recovery"]["certified_recovery"]


def test_rate_limit_same_decision_retry_stays_limited():
    spec = chain_spec(11, horizon=2)
    runtime = SpecRuntime(dict(spec, fault={"class": "rate_limit", "node_index": 0}),
                          SECRET, "faulted", 1)
    first_args = runtime.oracle_nodes[0]["args"]
    out1 = runtime.dispatch("kb_lookup", dict(first_args), decision=1)
    assert '"rate_limit"' in out1
    out2 = runtime.dispatch("kb_lookup", dict(first_args), decision=1)  # same decision
    assert "rate_limit_active" in out2
    out3 = runtime.dispatch("kb_lookup", dict(first_args), decision=2)  # next decision
    assert '"ok":true' in out3.splitlines()[0].replace(" ", "")
    assert len([e for e in runtime.events if e.get("fault_emitted")]) == 1


def test_rate_limit_full_episode_recovers_on_later_decision():
    spec = chain_spec(12, horizon=4)
    trace = _faulted_run(spec, "rate_limit", 1)
    fe = _fault_events(trace)
    assert len(fe) == 1
    rec = trace["score"]["recovery"]
    assert rec["certified_recovery"], rec
    remedial = [e for e in trace["events"] if e.get("token_provided")]
    assert remedial and remedial[0]["decision"] > fe[0]["decision"]


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
    body = json.loads(fe[0]["exposed_text"])
    assert body["ok"] is True and body["unit"] != "kg"  # explicit, valid-looking
    rec = trace["score"]["recovery"]
    assert rec["certified_recovery"], rec
    assert trace["score"]["certified_success"]


def test_fault_never_fires_twice_even_with_blind_retries():
    spec = chain_spec(15, horizon=4)
    fspec = dict(spec, fault={"class": "transient", "node_index": 1})
    trace = run(fspec, ScriptedOracle(fspec, blind_retry=True), condition="faulted")
    assert len(_fault_events(trace)) == 1


def test_clean_condition_never_emits_faults():
    spec = chain_spec(16, horizon=4)
    trace = run(spec, ScriptedOracle(spec), condition="clean")
    assert _fault_events(trace) == []
    assert trace["score"]["certified_success"]


def test_stress_condition_carries_two_distinct_faults():
    spec = chain_spec(17, horizon=8)
    spec["faults"] = [{"class": "transient", "node_index": 2},
                      {"class": "malformed", "node_index": 5}]
    trace = run(spec, ScriptedOracle(spec), condition="stress")
    fe = _fault_events(trace)
    assert len(fe) == 2
    assert {e["fault_class"] for e in fe} == {"transient", "malformed"}
    assert trace["score"]["raw_success"]
