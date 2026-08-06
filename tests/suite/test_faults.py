"""The four injectors on the ONE runtime: envelopes, tokens, receipts, clock.

Every model-visible observation is the registered form -- a canonical envelope
carrying no event or request id, a newline, and a `receipt:` line -- and every
error envelope for transient / rate_limit / malformed carries the 128-bit
recovery token the remediation contract requires. These tests pin that wire
format, not the tokenless one the training path used to emit.
"""

from __future__ import annotations

import dataclasses
import json

from agentlab.suite.faults import MALFORMED_LITERAL, TOKEN_ARG
from agentlab.suite.generate import TaskBundle
from agentlab.suite.runtime import (EpisodeRuntime, parse_observation,
                                    recovery_token_in, tool_schemas_for_family)
from agentlab.suite.schema import FAMILIES, FaultSpec, decision_budget
from agentlab.tools import unit_convert as global_unit_convert

from .conftest import SECRET, mk_bundle


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
    return EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes, secret=SECRET)


def envelope(text: str) -> dict:
    """The first JSON envelope of a model-visible observation."""
    objs = parse_observation(text)["objects"]
    assert objs, f"no envelope in {text!r}"
    return objs[0]


# ---------------------------------------------------------------------------
# the one tool surface and the one observation form
# ---------------------------------------------------------------------------

def test_every_tool_of_every_family_declares_the_recovery_token():
    """The model does not know in advance which tool will fault."""
    for family in FAMILIES:
        schemas = tool_schemas_for_family(family)
        assert schemas
        for schema in schemas:
            params = schema["function"]["parameters"]
            assert TOKEN_ARG in params["properties"], (family,
                                                       schema["function"]["name"])
            assert params["properties"][TOKEN_ARG]["type"] == "string"
            # optional: a clean call must never be obliged to carry one
            assert TOKEN_ARG not in (params.get("required") or [])
    # and augmenting the schemas must not have mutated the shared tool dicts
    from agentlab import tools as global_tools

    for schema in global_tools.tool_schemas():
        assert TOKEN_ARG not in schema["function"]["parameters"]["properties"]


def test_every_observation_carries_a_receipt_and_no_event_id():
    """p5_provenance promises a receipt line on EVERY tool result."""
    from agentlab import provenance

    b = mk_bundle("lookup_chain", 2)
    rt = rt_for(b)
    rt.begin_decision()
    text = rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    parsed = parse_observation(text)
    assert parsed["receipt"] and parsed["receipt"].startswith("r-")
    assert len(parsed["receipt"]) == 2 + 32
    body = parsed["objects"][0]
    assert body["ok"] is True
    assert "event_id" not in body and "request_id" not in body
    # the receipt covers the envelope bytes and validates against the run secret
    assert provenance.receipt_valid(SECRET, b.spec.task_id, rt.events[-1].to_row())
    tampered = rt.events[-1].to_row()
    tampered["exposed_text"] = tampered["exposed_text"][:-1] + " "
    assert not provenance.receipt_valid(SECRET, b.spec.task_id, tampered)


def test_the_token_is_stripped_before_semantics_and_recorded_as_evidence():
    b = mk_bundle("lookup_chain", 2)
    rt = rt_for(b)
    rt.begin_decision()
    plain = rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    plain_event = rt.events[-1]
    rt.begin_decision()
    with_token = rt.dispatch(b.nodes[0].tool,
                             dict(b.nodes[0].args, **{TOKEN_ARG: "f" * 32}))
    tok_event = rt.events[-1]
    # identical call identity and identical canonical semantics ...
    assert tok_event.canonical_args_digest == plain_event.canonical_args_digest
    assert tok_event.exposed_result_digest == plain_event.exposed_result_digest
    assert TOKEN_ARG not in envelope(with_token)
    assert envelope(with_token) == envelope(plain)
    # ... but the supplied token IS recorded in the hidden ledger
    assert tok_event.token_provided == "f" * 32
    assert plain_event.token_provided is None


# ---------------------------------------------------------------------------
# transient
# ---------------------------------------------------------------------------

def test_transient_payload_and_single_fire():
    b = bundle_with_fault("lookup_chain", 2, "transient", 0)
    rt = rt_for(b)
    rt.begin_decision()
    text = rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    obj = envelope(text)
    assert obj["ok"] is False
    assert obj["error"] == "transient_backend"
    assert obj["remediation"]["retryable"] is True
    assert obj["remediation"]["action"] == "reissue_with_token"
    token = obj[TOKEN_ARG]
    assert len(token) == 32 and int(token, 16) >= 0
    assert "request_id" not in obj and "event_id" not in obj
    assert rt.events[-1].fault_triggered
    assert rt.events[-1].recovery_token == token
    # the next attempt with the token succeeds and is credited
    obj2 = envelope(rt.dispatch("kb_lookup",
                                dict(b.nodes[0].args, **{TOKEN_ARG: token})))
    assert obj2["ok"] is True
    assert rt.events[-1].credited
    assert rt.events[-1].token_provided == token
    assert sum(e.fault_triggered for e in rt.events) == 1


def test_the_token_is_unforgeable_without_the_run_secret():
    b = bundle_with_fault("lookup_chain", 2, "transient", 0)
    mine = rt_for(b)
    mine.begin_decision()
    ours = recovery_token_in(mine.dispatch("kb_lookup", dict(b.nodes[0].args)))
    other = EpisodeRuntime(b.spec, b.kb, b.nodes, secret=bytes.fromhex("11" * 32))
    other.begin_decision()
    theirs = recovery_token_in(other.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert ours and theirs and ours != theirs


def test_wrong_call_does_not_consume_fault():
    b = bundle_with_fault("lookup_chain", 2, "transient", 1)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    miss = envelope(rt.dispatch("kb_lookup", {"key": "KWRONGKEY"}))
    assert miss == {"ok": False, "error": "no_entry"}
    assert not rt.events[-1].fault_triggered
    rt.begin_decision()
    obj = envelope(rt.dispatch("kb_lookup", dict(b.nodes[1].args)))
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
    obj = envelope(rt.dispatch("kb_lookup", dict(b.nodes[1].args)))
    assert obj["error"] == "transient_backend"
    assert rt.events[-1].fault_triggered


# ---------------------------------------------------------------------------
# rate limit
# ---------------------------------------------------------------------------

def test_rate_limit_logical_clock_and_same_decision_retry():
    b = bundle_with_fault("lookup_chain", 2, "rate_limit", 0)
    rt = rt_for(b)
    rt.begin_decision()
    obj = envelope(rt.dispatch("kb_lookup", dict(b.nodes[0].args)))
    assert obj["error"] == "rate_limit"
    assert obj["remediation"]["retry_after_turns"] == 1
    assert obj["remediation"]["action"] == "reissue_with_token_after_next_decision"
    token = obj[TOKEN_ARG]
    assert rt.events[-1].fault_triggered
    # repeating within the same assistant decision stays limited, WITH the token...
    again = envelope(rt.dispatch("kb_lookup",
                                 dict(b.nodes[0].args, **{TOKEN_ARG: token})))
    assert again["error"] == "rate_limit_active"
    assert again[TOKEN_ARG] == token          # the same fault, still in force
    assert not rt.events[-1].fault_triggered  # never a second firing
    assert rt.events[-1].rate_limited
    # ...retrying with the token on the next decision succeeds
    rt.begin_decision()
    ok = envelope(rt.dispatch("kb_lookup",
                              dict(b.nodes[0].args, **{TOKEN_ARG: token})))
    assert ok["ok"] is True
    assert rt.events[-1].credited
    assert sum(e.fault_triggered for e in rt.events) == 1


# ---------------------------------------------------------------------------
# malformed
# ---------------------------------------------------------------------------

def test_malformed_read_only_never_leaks_and_carries_a_token():
    b = bundle_with_fault("lookup_chain", 2, "malformed", 1)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    exposed = rt.dispatch("kb_lookup", dict(b.nodes[1].args))
    lines = exposed.splitlines()
    assert lines[0] == MALFORMED_LITERAL          # the truncated prefix, verbatim
    assert b.spec.answer not in exposed           # neither the result nor the key
    marked = json.loads(lines[1])
    assert marked["ok"] is False
    assert marked["error"] == "truncated_result"
    assert marked["remediation"]["action"] == "reissue_with_token"
    assert set(marked) == {"ok", "error", TOKEN_ARG, "remediation"}
    assert parse_observation(exposed)["truncated_prefix"] is True
    assert not rt.events[-1].credited
    rt.begin_decision()
    obj = envelope(rt.dispatch("kb_lookup",
                               dict(b.nodes[1].args, **{TOKEN_ARG: marked[TOKEN_ARG]})))
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
    token = recovery_token_in(exposed)
    assert exposed.splitlines()[0] == MALFORMED_LITERAL
    assert token
    assert rt.events[-1].state_mutated and rt.events[-1].credited
    assert rt.env.inventory[lot]["quantity"] == before - rt.env.lines[0]["quantity"]
    # a token-bearing replay of the same quote token returns the original
    # reservation and never double-decrements
    rt.begin_decision()
    obj = envelope(rt.dispatch(reserve.tool,
                               dict(reserve.args, **{TOKEN_ARG: token})))
    assert obj["ok"] is True
    assert obj["reservation_token"] == rt.env.lines[0]["reservation_token"]
    assert rt.events[-1].replay and not rt.events[-1].state_mutated
    assert rt.events[-1].token_provided == token
    assert rt.env.inventory[lot]["quantity"] == before - rt.env.lines[0]["quantity"]
    assert len([m for m in rt.env.mutations if m["kind"] == "reserve"]) == 1


# ---------------------------------------------------------------------------
# wrong unit
# ---------------------------------------------------------------------------

def test_wrong_unit_is_the_committed_unit_and_carries_no_token():
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
    text = rt.dispatch("unit_convert", dict(node.args))
    obj = envelope(text)
    assert obj["ok"] is True                 # a VALID result...
    assert obj["unit"] == wrong              # ...for the COMMITTED different unit
    assert obj["value"] == global_unit_convert(
        float(node.args["value"]), node.args["from_unit"], wrong)
    # the trap is not an error envelope, so no token is emitted
    assert TOKEN_ARG not in obj
    assert recovery_token_in(text) is None
    assert rt.events[-1].recovery_token is None
    assert rt.events[-1].requested_unit == str(node.args["to_unit"]).lower()
    assert not rt.events[-1].credited
    # a later correct-target conversion succeeds and feeds the next node
    rt.begin_decision()
    ok = envelope(rt.dispatch("unit_convert", dict(node.args)))
    assert ok["unit"] == node.args["to_unit"]
    assert rt.events[-1].credited


def test_the_runtime_never_re_derives_the_trap_unit():
    """The committed FaultSpec value is the only wrong unit that may appear.

    `SpecRuntime` recomputed one through `wrong_unit_target`, so the unit the
    generator committed and the unit the evaluated model saw could differ. Here
    the committed value is overwritten with a specific one and the runtime must
    expose exactly it.
    """
    from agentlab.suite.faults import wrong_unit_candidates

    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    node = next(n for n in b.nodes if n.node_id == b.spec.faults[0].target_node)
    pinned = wrong_unit_candidates(str(node.args["to_unit"]))[-1]
    spec = dataclasses.replace(
        b.spec, faults=[FaultSpec(fault_type="wrong_unit",
                                  target_node=node.node_id,
                                  params={"wrong_unit": pinned})])
    rt = EpisodeRuntime(spec, b.kb, b.nodes, secret=SECRET)
    rt.begin_decision()
    rt.dispatch("kb_lookup", dict(b.nodes[0].args))
    rt.begin_decision()
    assert envelope(rt.dispatch("unit_convert", dict(node.args)))["unit"] == pinned
