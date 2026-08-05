"""The four injectors: payloads, single firing, logical clock, idempotency."""

from __future__ import annotations

import dataclasses
import json

from agentlab.suite.faults import MALFORMED_LITERAL
from agentlab.suite.generate import TaskBundle
from agentlab.suite.runtime import EpisodeRuntime
from agentlab.suite.schema import FaultSpec, decision_budget
from agentlab.tools import unit_convert as global_unit_convert

from .conftest import mk_bundle


def bundle_with_fault(family, horizon, fault_type, node_index, params=None):
    b = mk_bundle(family, horizon)
    node = b.nodes[node_index]
    spec = dataclasses.replace(
        b.spec,
        faults=[FaultSpec(fault_type=fault_type, target_node=node.node_id,
                          params=params or {})],
        max_decisions=decision_budget(horizon, 1))
    return TaskBundle(spec=spec, kb=b.kb, nodes=b.nodes)


def rt_for(bundle) -> EpisodeRuntime:
    return EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes)


def test_transient_payload_and_single_fire():
    b = bundle_with_fault("lookup_chain", 2, "transient", 0)
    rt = rt_for(b)
    rt.begin_decision()
    obj = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert obj["ok"] is False
    assert obj["error"] == "transient_backend"
    assert obj["retryable"] is True
    assert obj["request_id"].startswith("req-")
    assert rt.events[-1].fault_triggered
    # the next correct attempt succeeds
    obj2 = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert obj2["ok"] is True
    assert rt.events[-1].credited
    assert sum(e.fault_triggered for e in rt.events) == 1


def test_wrong_call_does_not_consume_fault():
    b = bundle_with_fault("lookup_chain", 2, "transient", 1)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    miss = json.loads(rt.dispatch("kb_lookup", {"key": "KWRONGKEY"}))
    assert miss == {"ok": False, "error": "no_entry",
                    "event_id": rt.events[-1].event_id}
    assert not rt.events[-1].fault_triggered
    rt.begin_decision()
    obj = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[1].args)))
    assert obj["error"] == "transient_backend"
    assert rt.events[-1].fault_triggered


def test_batched_guess_does_not_consume_fault():
    """A same-decision (dependency-invalid) call at the target never fires it."""
    b = bundle_with_fault("lookup_chain", 4, "transient", 1)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.dispatch("kb_lookup", dict(b.nodes[1].args))  # batched guess
    guess = rt.events[-1]
    assert not guess.credited and not guess.fault_triggered
    rt.begin_decision()
    obj = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[1].args)))
    assert obj["error"] == "transient_backend"
    assert rt.events[-1].fault_triggered


def test_rate_limit_logical_clock():
    b = bundle_with_fault("lookup_chain", 2, "rate_limit", 0)
    rt = rt_for(b)
    rt.begin_decision()
    obj = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert obj["error"] == "rate_limit"
    assert obj["retry_after_turns"] == 1
    assert rt.events[-1].fault_triggered
    # repeating within the same assistant decision stays limited...
    again = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert again["error"] == "rate_limit"
    assert not rt.events[-1].fault_triggered  # same fault, not a second firing
    assert rt.events[-1].rate_limited
    # ...retrying on the next assistant decision succeeds
    rt.begin_decision()
    ok = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert ok["ok"] is True
    assert rt.events[-1].credited
    assert sum(e.fault_triggered for e in rt.events) == 1


def test_malformed_read_only_never_leaks():
    b = bundle_with_fault("lookup_chain", 2, "malformed", 1)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    exposed = rt.dispatch("kb_lookup", dict(b.nodes[1].args))
    assert exposed == MALFORMED_LITERAL
    assert b.spec.answer not in exposed  # neither the result nor the next key
    assert not rt.events[-1].credited
    rt.begin_decision()
    obj = json.loads(rt.dispatch("kb_lookup", dict(b.nodes[1].args)))
    assert obj["ok"] is True
    assert obj["record"]["code"] == b.spec.answer
    assert rt.events[-1].credited


def test_malformed_ambiguous_mutation_idempotent_replay():
    b = mk_bundle("fulfillment", 4, [("malformed", True)])
    fault = b.spec.faults[0]
    assert fault.params.get("ambiguous_mutation")
    rt = rt_for(b)
    reserve_idx = next(i for i, n in enumerate(b.nodes)
                       if n.node_id == fault.target_node)
    for node in b.nodes[:reserve_idx]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    reserve = b.nodes[reserve_idx]
    lot = rt.env.lines[0]["lot"]
    before = rt.env.inventory[lot]["quantity"]
    rt.begin_decision()
    exposed = rt.dispatch(reserve.tool, dict(reserve.args))
    assert exposed == MALFORMED_LITERAL
    assert rt.events[-1].state_mutated and rt.events[-1].credited
    assert rt.env.inventory[lot]["quantity"] == before - rt.env.lines[0]["quantity"]
    # replaying the same quote token returns the original reservation and
    # never double-decrements
    rt.begin_decision()
    obj = json.loads(rt.dispatch(reserve.tool, dict(reserve.args)))
    assert obj["ok"] is True
    assert obj["reservation_token"] == rt.env.lines[0]["reservation_token"]
    assert rt.events[-1].replay and not rt.events[-1].state_mutated
    assert rt.env.inventory[lot]["quantity"] == before - rt.env.lines[0]["quantity"]
    assert len([m for m in rt.env.mutations if m["kind"] == "reserve"]) == 1


def test_wrong_unit_is_valid_and_explicit():
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    fault = b.spec.faults[0]
    node = next(n for n in b.nodes if n.node_id == fault.target_node)
    assert node.tool == "unit_convert"
    wrong = fault.params["wrong_unit"]
    assert wrong != node.args["to_unit"]
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    obj = json.loads(rt.dispatch("unit_convert", dict(node.args)))
    assert obj["ok"] is True                 # a VALID result...
    assert obj["unit"] == wrong              # ...for an explicit different unit
    assert obj["value"] == global_unit_convert(
        float(node.args["value"]), node.args["from_unit"], wrong)
    assert not rt.events[-1].credited
    # a later correct-target conversion succeeds and feeds the next node
    rt.begin_decision()
    ok = json.loads(rt.dispatch("unit_convert", dict(node.args)))
    assert ok["unit"] == node.args["to_unit"]
    assert rt.events[-1].credited
