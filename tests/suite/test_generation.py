"""Deterministic generation: byte identity, sizes, budgets, mixtures."""

from __future__ import annotations

import json

from agentlab.suite.generate import (apportion, build_split, build_task,
                                     fault_plan)
from agentlab.suite.schema import CELLS, canon, write_jsonl

from .conftest import SEED, SUITE, mk_bundle


def test_same_labels_same_bytes():
    a = build_task(SUITE, SEED, "eval", "typed_relay", 8, 3, [("wrong_unit", False)])
    b = build_task(SUITE, SEED, "eval", "typed_relay", 8, 3, [("wrong_unit", False)])
    assert canon(a.spec.to_row()) == canon(b.spec.to_row())
    assert canon(a.kb) == canon(b.kb)
    assert canon([n.to_row() for n in a.nodes]) == canon([n.to_row() for n in b.nodes])


def test_different_index_different_task():
    a = mk_bundle("lookup_chain", 4, index=0)
    b = mk_bundle("lookup_chain", 4, index=1)
    assert a.spec.answer != b.spec.answer
    assert not (set(a.kb) & set(b.kb))


def test_declared_horizon_matches_nodes():
    for family, h in CELLS:
        bundle = mk_bundle(family, h)
        assert bundle.spec.horizon == h == len(bundle.nodes)


def test_budgets():
    clean = mk_bundle("fulfillment", 8)
    assert clean.spec.max_decisions == 8 + 3
    assert clean.spec.max_calls == 2 * 8 + 4
    one = mk_bundle("fulfillment", 8, [("transient", False)])
    assert one.spec.max_decisions == 8 + 5
    two = mk_bundle("fulfillment", 8, [("transient", False), ("rate_limit", False)])
    assert two.spec.max_decisions == 8 + 8


def test_apportion_exact():
    assert apportion(100, (34, 33, 33)) == [34, 33, 33]
    assert apportion(20, (34, 33, 33)) == [7, 7, 6]
    assert apportion(100, (1, 1, 1, 1)) == [25, 25, 25, 25]
    assert sum(apportion(7, (1, 1, 1, 1))) == 7


def test_fault_plan_train_mixture_unit_convert_cell():
    plan = fault_plan("oracle_sft", "typed_relay", 8, 400)
    assert plan.count(None) == 200
    types = [e[0][0] for e in plan if e]
    for t in ("transient", "malformed", "wrong_unit", "rate_limit"):
        assert types.count(t) == 50


def test_fault_plan_train_mixture_no_unit_convert_cell():
    plan = fault_plan("distill", "lookup_chain", 4, 200)
    assert plan.count(None) == 100
    types = [e[0][0] for e in plan if e]
    assert types.count("transient") == 34
    assert types.count("malformed") == 33
    assert types.count("rate_limit") == 33
    assert "wrong_unit" not in types


def test_fault_plan_ambiguous_stride():
    plan = fault_plan("grpo_train", "fulfillment", 8, 200)
    malformed = [e[0] for e in plan if e and e[0][0] == "malformed"]
    assert len(malformed) == 25
    ambiguous = [e for e in malformed if e[1]]
    assert len(ambiguous) == 7  # ordinals 0,4,8,12,16,20,24


def test_fault_plan_eval_all_faulted_and_stress_pairs():
    plan = fault_plan("eval", "fulfillment", 14, 100)
    assert all(e and len(e) == 1 for e in plan)
    stress = fault_plan("eval_stress", "typed_relay", 12, 40)
    assert all(e and len(e) == 2 for e in stress)
    for pair in stress:
        assert pair[0][0] != pair[1][0]  # two DISTINCT fault types


def test_stress_faults_target_distinct_nodes():
    b = mk_bundle("fulfillment", 20, [("malformed", True), ("wrong_unit", False)],
                  split="eval_stress")
    assert len(b.spec.faults) == 2
    assert b.spec.faults[0].target_node != b.spec.faults[1].target_node


def test_build_split_tiny_counts():
    result = build_split(SUITE, "dev", 0xA61E0004, 2)
    assert len(result["specs"]) == 2 * len(CELLS)
    stress = build_split(SUITE, "eval_stress", 0xA61E0006, 2)
    assert len(stress["specs"]) == 2 * 7  # seven cells at or above H8


def test_canonical_jsonl(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(str(path), [{"b": 2, "a": 1}])
    raw = path.read_bytes()
    assert raw == b'{"a":1,"b":2}\n'
    assert json.loads(raw)


def test_template_ranges_by_split():
    assert mk_bundle("lookup_chain", 2, split="oracle_sft").spec.template_id in range(0, 8)
    assert mk_bundle("lookup_chain", 2, split="dev").spec.template_id in range(8, 10)
    assert mk_bundle("lookup_chain", 2, split="eval").spec.template_id in range(10, 12)
