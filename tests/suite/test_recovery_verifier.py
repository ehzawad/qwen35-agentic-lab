"""The ONE verifier: dependency edges, the registered remediation predicate, and
the single certified-success predicate.

The negative tests are the point of this module. A blind retry gets the canonical
value and must NOT be certified; a token echoed after an earlier blind recovery
must not launder it; a wrong token, a token on the wrong call, a same-decision
rate-limit retry, a status-only repair of an ambiguous mutation, a correct answer
with an invalid final fulfillment state, and an over-budget episode must all fail.
"""

from __future__ import annotations

import dataclasses

from agentlab.suite.faults import TOKEN_ARG
from agentlab.suite.runtime import (EpisodeRuntime, parse_observation,
                                    recovery_token_in, run_oracle)
from agentlab.suite.schema import FaultSpec, decision_budget
from agentlab.suite.verify import aggregate_recovery

from .conftest import SECRET, mk_bundle


def rt_for(bundle) -> EpisodeRuntime:
    return EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes, secret=SECRET)


def obj(text: str) -> dict:
    objs = parse_observation(text)["objects"]
    return objs[0] if objs else {}


def boxed(bundle) -> str:
    return f"\\boxed{{{bundle.spec.answer}}}"


def fault_index(bundle) -> int:
    """Where the generator scheduled the fault (never assumed to be node 0)."""
    target = bundle.spec.faults[0].target_node
    return next(i for i, n in enumerate(bundle.nodes) if n.node_id == target)


def drive_to_fault(rt, bundle):
    """Walk the clean prefix, fire the fault once -> (index, node, token)."""
    idx = fault_index(bundle)
    walk(rt, bundle.nodes[:idx])
    node = bundle.nodes[idx]
    rt.begin_decision()
    token = recovery_token_in(rt.dispatch(node.tool, dict(node.args)))
    return idx, node, token


def walk(rt, nodes, *, echo=True):
    """Drive the oracle path one decision per call, echoing every token."""
    for node in nodes:
        pending = None
        for _ in range(4):
            rt.begin_decision()
            args = dict(node.args)
            if pending and echo:
                args[TOKEN_ARG] = pending
            text = rt.dispatch(node.tool, args)
            pending = recovery_token_in(text)
            body = obj(text)
            if pending is None and body.get("ok") and not (
                    node.tool == "unit_convert"
                    and str(body.get("unit", "")).lower()
                    != str(node.args["to_unit"]).lower()):
                break
    return rt


# ---------------------------------------------------------------------------
# the positive path
# ---------------------------------------------------------------------------

def test_registered_remediation_is_certified():
    for fault in ("transient", "rate_limit", "malformed"):
        b = mk_bundle("lookup_chain", 4, [(fault, False)])
        rt, verdict = run_oracle(b.spec, b.kb, b.nodes, secret=SECRET)
        assert verdict.certified_success, (fault, verdict.reasons[:3])
        assert verdict.recovery_reason == "ok"
        assert verdict.recovered and verdict.recovery_success
        report = verdict.fault_reports[0]
        assert report["recovery_call_id"] is not None


# ---------------------------------------------------------------------------
# the negative cases the registered predicate must refuse
# ---------------------------------------------------------------------------

def test_blind_retry_returns_the_canonical_data_but_is_not_certified():
    """The tool recovered, not the agent."""
    b = mk_bundle("lookup_chain", 4, [("transient", False)])
    rt = walk(rt_for(b), b.nodes, echo=False)
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.answer_ok                     # operationally it worked
    assert verdict.task_success                  # every oracle node completed
    assert not verdict.recovered
    assert verdict.recovery_reason == "blind_retry"
    assert not verdict.certified_success
    assert not verdict.recovery_success


def test_a_later_token_echo_cannot_launder_an_earlier_blind_recovery():
    """The qualifying event must BE the event that establishes recovery.

    Computing "a remedial action happened" and "a canonical result happened" as
    two independent event sets is a laundering hole: one blind retry obtains the
    value and a decorative token echo afterwards certifies it.
    """
    b = mk_bundle("lookup_chain", 4, [("transient", False)])
    rt = rt_for(b)
    idx, node, token = drive_to_fault(rt, b)
    assert token
    rt.begin_decision()
    rt.dispatch(node.tool, dict(node.args))               # BLIND retry: canonical
    assert rt.events[-1].credited
    rt.begin_decision()
    rt.dispatch(node.tool, dict(node.args, **{TOKEN_ARG: token}))  # decorative echo
    walk(rt, b.nodes[idx + 1:])
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.task_success
    assert verdict.recovery_reason == "blind_retry"
    assert not verdict.certified_success


def test_a_wrong_token_is_not_remediation():
    b = mk_bundle("lookup_chain", 4, [("transient", False)])
    rt = rt_for(b)
    idx, node, token = drive_to_fault(rt, b)
    rt.begin_decision()
    rt.dispatch(node.tool, dict(node.args, **{TOKEN_ARG: "0" * 32}))
    assert token != "0" * 32
    walk(rt, b.nodes[idx + 1:])
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.recovery_reason == "blind_retry"
    assert not verdict.certified_success


def test_a_token_on_the_wrong_call_is_not_remediation():
    """The token must be echoed on the SAME stripped call identity."""
    b = mk_bundle("lookup_chain", 4, [("transient", False)])
    rt = rt_for(b)
    idx, node, token = drive_to_fault(rt, b)
    assert idx >= 1, "this case needs a fault after the first node"
    assert token
    # the token lands on a DIFFERENT call (an already-completed node) ...
    rt.begin_decision()
    other = b.nodes[idx - 1]
    rt.dispatch(other.tool, dict(other.args, **{TOKEN_ARG: token}))
    # ... and the faulted node is then repaired blind
    walk(rt, b.nodes[idx:], echo=False)
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.recovery_reason == "blind_retry"
    assert not verdict.certified_success


def test_a_same_decision_rate_limit_retry_is_not_remediation():
    b = mk_bundle("lookup_chain", 4, [("rate_limit", False)])
    rt = rt_for(b)
    _idx, node, token = drive_to_fault(rt, b)
    # correct token, correct call, SAME decision: still limited
    still = obj(rt.dispatch(node.tool, dict(node.args, **{TOKEN_ARG: token})))
    assert still["error"] == "rate_limit_active"
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.recovery_reason == "no_remediation"
    assert not verdict.certified_success
    # and the LATER-decision retry with the same token is certified
    rt2 = walk(rt_for(b), b.nodes)
    rt2.begin_decision()
    assert rt2.verify(boxed(b)).certified_success


def test_a_correct_target_conversion_of_the_wrong_operand_is_no_post_fault_result():
    """The registered action happened; the canonical result did not.

    Right unit, wrong operand: the agent DID perform the remediation the contract
    names (a later conversion explicitly requesting the original target unit), and
    still never obtained the node's canonical value. That is its own label, not
    `blind_retry` (nothing recovered) and not `no_remediation` (something was
    tried).
    """
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    rt = rt_for(b)
    _idx, node, _token = drive_to_fault(rt, b)
    rt.begin_decision()
    wrong_operand = dict(node.args, value=float(node.args["value"]) + 1.0)
    rt.dispatch(node.tool, wrong_operand)
    assert not rt.events[-1].exposed_canonical
    assert rt.events[-1].requested_unit == str(node.args["to_unit"]).lower()
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.recovery_reason == "no_post_fault_result"
    assert verdict.fault_reports[0]["attempted"] is True
    assert verdict.fault_reports[0]["recovered"] is False
    assert not verdict.certified_success


def test_giving_up_after_the_fault_is_no_remediation():
    b = mk_bundle("lookup_chain", 4, [("transient", False)])
    rt = rt_for(b)
    drive_to_fault(rt, b)
    rt.begin_decision()
    verdict = rt.verify("I cannot determine this value.")
    assert verdict.recovery_reason == "no_remediation"
    assert not verdict.certified_success


def test_wrong_unit_needs_the_corrected_target_not_a_token():
    """Mental arithmetic must not masquerade as recovery (binding)."""
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))  # trap fires
    rt.begin_decision()
    verdict = rt.verify(boxed(b))                        # correct, no recovery
    assert verdict.answer_ok
    assert verdict.faults_triggered == 1
    assert not verdict.recovered
    assert not verdict.certified_success
    assert not verdict.recovery_success
    # re-requesting the ORIGINAL target unit is the registered repair
    rt2 = walk(rt_for(b), b.nodes)
    rt2.begin_decision()
    ok = rt2.verify(boxed(b))
    assert ok.certified_success and ok.recovery_reason == "ok"


def test_the_committed_wrong_unit_value_never_enters_the_answer():
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    trap = b.spec.faults[0].params["wrong_unit"]
    rt = walk(rt_for(b), b.nodes)
    trap_events = [e for e in rt.events if e.fault_triggered]
    assert len(trap_events) == 1
    assert trap != b.nodes[1].args["to_unit"]
    rt.begin_decision()
    assert rt.verify(boxed(b)).certified_success


def test_status_only_recovery_after_an_ambiguous_mutation_is_not_certified():
    """A status query establishes operational state, not certified remediation.

    Under the registered wording it neither reissues the same call nor echoes the
    token, so it is `blind_retry`. Retaining status-query certification would need
    an explicit protocol amendment, not a verifier that quietly allows it.
    """
    b = mk_bundle("fulfillment", 4, [("malformed", True)])
    fault = b.spec.faults[0]
    rt = rt_for(b)
    idx = next(i for i, n in enumerate(b.nodes) if n.node_id == fault.target_node)
    for node in b.nodes[:idx + 1]:               # ... through the truncated reserve
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    line = rt.env.lines[0]
    rt.begin_decision()
    status = obj(rt.dispatch("warehouse_query",
                             {"resource": "reservation",
                              "token": line["quote_token"]}))
    assert status["ok"] is True and status["status"] == "reserved"
    for node in b.nodes[idx + 1:]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.task_success                  # every node completed, state valid
    assert verdict.unique_valid_nodes == 4
    assert not verdict.recovered
    # The status query is neither the registered remediation NOR a canonical
    # re-observation of the faulted mutation, so it is `no_remediation`; a BLIND
    # reissue of the reserve would have been `blind_retry`. Either way it is
    # uncertified, and certifying it again would need a protocol amendment.
    assert verdict.recovery_reason == "no_remediation"
    assert not verdict.certified_success
    # the blind reissue, for contrast: the canonical replay arrives, uncertified
    blind = rt_for(b)
    for node in b.nodes[:idx + 1]:
        blind.begin_decision()
        blind.dispatch(node.tool, dict(node.args))
    blind.begin_decision()
    blind.dispatch(b.nodes[idx].tool, dict(b.nodes[idx].args))
    for node in b.nodes[idx + 1:]:
        blind.begin_decision()
        blind.dispatch(node.tool, dict(node.args))
    blind.begin_decision()
    blind_verdict = blind.verify(boxed(b))
    assert blind_verdict.recovery_reason == "blind_retry"
    assert not blind_verdict.certified_success


def test_token_bearing_idempotent_replay_after_an_ambiguous_mutation_is_certified():
    b = mk_bundle("fulfillment", 4, [("malformed", True)])
    fault = b.spec.faults[0]
    rt = rt_for(b)
    idx = next(i for i, n in enumerate(b.nodes) if n.node_id == fault.target_node)
    for node in b.nodes[:idx]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    reserve = b.nodes[idx]
    rt.begin_decision()
    token = recovery_token_in(rt.dispatch(reserve.tool, dict(reserve.args)))
    rt.begin_decision()
    replayed = obj(rt.dispatch(reserve.tool,
                               dict(reserve.args, **{TOKEN_ARG: token})))
    assert replayed["ok"] is True
    assert rt.events[-1].replay
    for node in b.nodes[idx + 1:]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.certified_success, verdict.reasons[:3]
    assert verdict.recovery_reason == "ok"


# ---------------------------------------------------------------------------
# the rest of the one success predicate
# ---------------------------------------------------------------------------

def test_missing_oracle_node_fails_even_with_correct_answer():
    b = mk_bundle("lookup_chain", 4)
    rt = rt_for(b)
    for node in (b.nodes[0], b.nodes[2], b.nodes[3]):  # skip n2
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.answer_ok
    assert not verdict.certified_success
    assert verdict.unique_valid_nodes == 1  # progress stops at the gap


def test_a_correct_answer_with_an_invalid_final_fulfillment_state_fails():
    """The certifier used to have ZERO oracle/final-state references."""
    b = mk_bundle("fulfillment", 4)
    rt = rt_for(b)
    for node in b.nodes[:-1]:              # everything except the finalize
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.answer_ok
    assert not verdict.state_ok
    assert not verdict.task_success
    assert not verdict.certified_success
    assert any("status is not complete" in r for r in verdict.reasons)


def test_the_call_budget_is_part_of_the_predicate_and_equality_is_within_it():
    """THE EQUALITY-CAP RULING: exactly max_calls is success, one more is not."""
    b = mk_bundle("lookup_chain", 2)
    spec = dataclasses.replace(b.spec, max_calls=2)
    rt = EpisodeRuntime(spec, b.kb, b.nodes, secret=SECRET)
    for node in b.nodes:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    at_cap = rt.verify(boxed(b))
    assert at_cap.calls == 2 and at_cap.within_budget
    assert not at_cap.runaway
    assert at_cap.certified_success

    over = EpisodeRuntime(dataclasses.replace(b.spec, max_calls=1), b.kb, b.nodes,
                          secret=SECRET)
    for node in b.nodes:
        over.begin_decision()
        over.dispatch(node.tool, dict(node.args))
    over.begin_decision()
    verdict = over.verify(boxed(b))
    assert not verdict.within_budget
    assert verdict.runaway and "over_call_cap" in verdict.runaway_reasons
    assert not verdict.certified_success


def test_a_call_cap_termination_is_runaway_even_at_exactly_the_cap():
    b = mk_bundle("lookup_chain", 2)
    rt = walk(rt_for(b), b.nodes)
    rt.begin_decision()
    assert not rt.verify(boxed(b), termination_reason="answered").runaway
    assert rt.verify(boxed(b), termination_reason="call_cap").runaway
    assert not rt.verify(boxed(b), termination_reason="call_cap").certified_success


def test_a_broken_receipt_chain_vetoes_certified_success():
    b = mk_bundle("lookup_chain", 2)
    rt = walk(rt_for(b), b.nodes)
    rt.begin_decision()
    assert rt.verify(boxed(b)).receipts_ok
    rt.events[0].receipt = "r-" + "0" * 32
    verdict = rt.verify(boxed(b))
    assert not verdict.receipts_ok
    assert not verdict.certified_success


def test_a_committed_value_with_no_validated_source_is_a_hallucination():
    b = mk_bundle("lookup_chain", 2)
    rt = rt_for(b)
    rt.begin_decision()
    verdict = rt.verify("ANSWER: not-a-real-token")
    assert verdict.hallucinated
    assert "answer_value_unvalidated" in verdict.hallucination_reasons
    assert not verdict.certified_success
    # committing nothing is a plain failure, not a fabrication
    quiet = rt_for(b)
    quiet.begin_decision()
    plain = quiet.verify("I cannot determine this value.")
    assert not plain.hallucinated and not plain.certified_success


def test_same_decision_dependency_never_credited():
    b = mk_bundle("lookup_chain", 4)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))  # same batch
    assert not rt.events[-1].credited
    # re-issuing on a LATER decision does count, and the chain completes
    for node in b.nodes[1:]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.certified_success
    d = verdict.node_decisions
    assert d["n2"] > d["n1"] and d["n3"] > d["n2"] and d["n4"] > d["n3"]


def test_repeats_and_extra_reads_accepted_but_never_credited():
    b = mk_bundle("lookup_chain", 2)
    rt = rt_for(b)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))  # idempotent repeat
    assert rt.events[-1].repeat and not rt.events[-1].credited
    rt.begin_decision()
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.certified_success
    assert verdict.unique_valid_nodes == 2
    assert verdict.excess_calls == 1


def test_stress_episodes_require_every_assigned_fault_to_be_certified():
    b = mk_bundle("lookup_chain", 8)
    nodes = b.nodes
    spec = dataclasses.replace(
        b.spec,
        faults=[FaultSpec("transient", nodes[2].node_id, {}),
                FaultSpec("malformed", nodes[5].node_id, {})],
        max_decisions=decision_budget(8, 2))
    # both certified
    both = walk(EpisodeRuntime(spec, b.kb, nodes, secret=SECRET), nodes)
    both.begin_decision()
    good = both.verify(boxed(b))
    assert good.faults_triggered == 2
    assert good.certified_success and good.recovery_reason == "ok"
    # the second one repaired blind: the conjunction fails
    rt = EpisodeRuntime(spec, b.kb, nodes, secret=SECRET)
    walk(rt, nodes[:5])
    walk(rt, nodes[5:6], echo=False)
    walk(rt, nodes[6:])
    rt.begin_decision()
    mixed = rt.verify(boxed(b))
    assert mixed.faults_triggered == 2
    assert mixed.recovery_reason == "blind_retry"
    assert not mixed.certified_success


def test_aggregate_recovery_denominators():
    verdicts = []
    # certified fault arm
    b1 = mk_bundle("lookup_chain", 4, [("transient", False)])
    verdicts.append(run_oracle(b1.spec, b1.kb, b1.nodes, secret=SECRET)[1])
    # triggered but unrecovered fault arm
    b2 = mk_bundle("typed_relay", 2, [("wrong_unit", False)], index=1)
    rt = rt_for(b2)
    rt.begin_decision()
    rt.dispatch(b2.nodes[0].tool, dict(b2.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b2.nodes[1].tool, dict(b2.nodes[1].args))
    rt.begin_decision()
    verdicts.append(rt.verify(boxed(b2)))
    # assigned but never triggered (model never reaches the node)
    b3 = mk_bundle("lookup_chain", 4, [("rate_limit", False)], index=2)
    rt3 = rt_for(b3)
    rt3.begin_decision()
    verdicts.append(rt3.verify("no idea"))
    # clean episode: excluded from every recovery denominator
    b4 = mk_bundle("lookup_chain", 2, index=3)
    verdicts.append(run_oracle(b4.spec, b4.kb, b4.nodes, secret=SECRET)[1])

    agg = aggregate_recovery(verdicts)
    assert agg["n_assigned"] == 3
    assert agg["n_triggered"] == 2
    assert abs(agg["fault_arm_certified_success"] - 1 / 3) < 1e-9
    assert abs(agg["fault_trigger_rate"] - 2 / 3) < 1e-9
    assert abs(agg["recovery_success_over_triggered"] - 1 / 2) < 1e-9


def test_a_same_decision_corrected_conversion_is_not_certified():
    """The stricter reading the claim-bearing certifier has always enforced.

    A corrected conversion issued inside the SAME decision as the trap was
    batched, not read off the trap, so it does not certify recovery even though
    the registered wrong-unit contract names no timing rule.
    """
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    rt = rt_for(b)
    _idx, node, _token = drive_to_fault(rt, b)
    # same decision, correct target unit, canonical result
    same = obj(rt.dispatch(node.tool, dict(node.args)))
    assert same["unit"] == str(node.args["to_unit"]).lower()
    assert rt.events[-1].exposed_canonical
    assert rt.events[-1].decision_id == rt.events[-2].decision_id
    rt.begin_decision()
    verdict = rt.verify(boxed(b))
    assert verdict.recovery_reason == "blind_retry"
    assert not verdict.certified_success
