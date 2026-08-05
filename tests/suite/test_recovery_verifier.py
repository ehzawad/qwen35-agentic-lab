"""Strict verifier: dependency edges, recovery semantics, denominators."""

from __future__ import annotations

import json

from agentlab.suite.runtime import EpisodeRuntime, run_oracle
from agentlab.suite.verify import aggregate_recovery

from .conftest import mk_bundle


def test_recovery_success_requires_trace_not_just_answer():
    """Mental arithmetic must not masquerade as recovery (binding)."""
    b = mk_bundle("typed_relay", 2, [("wrong_unit", False)])
    rt = EpisodeRuntime(b.spec, b.kb, b.nodes)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))  # trap fires
    rt.begin_decision()
    verdict = rt.verify(f"\\boxed{{{b.spec.answer}}}")   # correct, no recovery
    assert verdict.answer_ok
    assert verdict.faults_triggered == 1
    assert not verdict.recovered
    assert not verdict.strict_success
    assert not verdict.recovery_success


def test_missing_oracle_node_fails_even_with_correct_answer():
    b = mk_bundle("lookup_chain", 4)
    rt = EpisodeRuntime(b.spec, b.kb, b.nodes)
    for node in (b.nodes[0], b.nodes[2], b.nodes[3]):  # skip n2
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(f"\\boxed{{{b.spec.answer}}}")
    assert verdict.answer_ok
    assert not verdict.strict_success
    assert verdict.unique_valid_nodes == 1  # progress stops at the gap


def test_same_decision_dependency_never_credited():
    b = mk_bundle("lookup_chain", 4)
    rt = EpisodeRuntime(b.spec, b.kb, b.nodes)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))  # same batch
    assert not rt.events[-1].credited
    # re-issuing on a LATER decision does count, and the chain completes
    for node in b.nodes[1:]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(f"\\boxed{{{b.spec.answer}}}")
    assert verdict.strict_success
    d = verdict.node_decisions
    assert d["n2"] > d["n1"] and d["n3"] > d["n2"] and d["n4"] > d["n3"]


def test_ambiguous_recovery_via_status_query():
    """The second accepted recovery path: reservation-status, then continue."""
    b = mk_bundle("fulfillment", 4, [("malformed", True)])
    fault = b.spec.faults[0]
    rt = EpisodeRuntime(b.spec, b.kb, b.nodes)
    idx = next(i for i, n in enumerate(b.nodes) if n.node_id == fault.target_node)
    for node in b.nodes[:idx + 1]:  # ... through the truncated reserve
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    line = rt.env.lines[0]
    rt.begin_decision()
    status = json.loads(rt.dispatch(
        "warehouse_query",
        {"resource": "reservation", "token": line["quote_token"]}))
    assert status["ok"] is True and status["status"] == "reserved"
    for node in b.nodes[idx + 1:]:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
    rt.begin_decision()
    verdict = rt.verify(f"\\boxed{{{b.spec.answer}}}")
    assert verdict.strict_success
    assert verdict.recovered and verdict.recovery_success
    assert verdict.unique_valid_nodes == 4
    assert verdict.excess_calls == 1  # the status query earned no credit


def test_repeats_and_extra_reads_accepted_but_never_credited():
    b = mk_bundle("lookup_chain", 2)
    rt = EpisodeRuntime(b.spec, b.kb, b.nodes)
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))  # idempotent repeat
    assert rt.events[-1].repeat and not rt.events[-1].credited
    rt.begin_decision()
    rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))
    rt.begin_decision()
    verdict = rt.verify(f"\\boxed{{{b.spec.answer}}}")
    assert verdict.strict_success
    assert verdict.unique_valid_nodes == 2
    assert verdict.excess_calls == 1


def test_aggregate_recovery_denominators():
    verdicts = []
    # recovered fault arm
    b1 = mk_bundle("lookup_chain", 4, [("transient", False)])
    verdicts.append(run_oracle(b1.spec, b1.kb, b1.nodes)[1])
    # triggered but unrecovered fault arm
    b2 = mk_bundle("typed_relay", 2, [("wrong_unit", False)], index=1)
    rt = EpisodeRuntime(b2.spec, b2.kb, b2.nodes)
    rt.begin_decision()
    rt.dispatch(b2.nodes[0].tool, dict(b2.nodes[0].args))
    rt.begin_decision()
    rt.dispatch(b2.nodes[1].tool, dict(b2.nodes[1].args))
    rt.begin_decision()
    verdicts.append(rt.verify(f"\\boxed{{{b2.spec.answer}}}"))
    # assigned but never triggered (model never reaches the node)
    b3 = mk_bundle("lookup_chain", 4, [("rate_limit", False)], index=2)
    rt3 = EpisodeRuntime(b3.spec, b3.kb, b3.nodes)
    rt3.begin_decision()
    verdicts.append(rt3.verify("no idea"))
    # clean episode: excluded from every recovery denominator
    b4 = mk_bundle("lookup_chain", 2, index=3)
    verdicts.append(run_oracle(b4.spec, b4.kb, b4.nodes)[1])

    agg = aggregate_recovery(verdicts)
    assert agg["n_assigned"] == 3
    assert agg["n_triggered"] == 2
    assert abs(agg["fault_arm_strict_success"] - 1 / 3) < 1e-9
    assert abs(agg["fault_trigger_rate"] - 2 / 3) < 1e-9
    assert abs(agg["recovery_success_over_triggered"] - 1 / 2) < 1e-9
