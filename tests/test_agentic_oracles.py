"""Oracle replay (S9): reachability, declared horizons, dataflow resolution,
and the redaction / permutation control transforms."""

import copy

from agentic_helpers import chain_spec, relay_spec

from agentlab import provenance


def test_chain_oracle_replays_to_declared_horizon_and_answer():
    spec = chain_spec(1, horizon=4)
    res = provenance.verify_oracle(spec)
    assert res["ok"], res["problems"]
    assert len(res["nodes"]) == 4
    assert res["replayed_answer"] == spec["answer"]


def test_relay_oracle_resolves_dataflow_through_real_tools():
    spec = relay_spec(3)
    res = provenance.verify_oracle(spec)
    assert res["ok"], res["problems"]
    # n3 converted grams -> kg through the real unit_convert
    n3 = res["nodes"][2]
    assert n3["tool"] == "unit_convert"
    grams = float(spec["kb"][f"eval-b-K{3:04d}-spec"]["grams"])
    assert float(n3["envelope"]["value"]) == grams / 1000
    # n4 consumed n3's value inside the calculator expression
    n4 = res["nodes"][3]
    assert str(int(float(n3["envelope"]["value"]))) in n4["args"]["expression"]
    assert str(res["replayed_answer"]) == spec["answer"]


def test_wrong_declared_horizon_is_flagged():
    spec = chain_spec(2, horizon=4)
    spec["horizon"] = 6
    res = provenance.verify_oracle(spec)
    assert not res["ok"]
    assert any("horizon" in p for p in res["problems"])


def test_broken_chain_is_unreachable():
    spec = chain_spec(4, horizon=4)
    # sever the chain: second record no longer points anywhere useful
    first_key = spec["oracle"][0]["args"]["key"]
    spec["kb"][spec["kb"][first_key]["next"]]["next"] = "missing-key"
    res = provenance.verify_oracle(spec)
    assert not res["ok"]


def test_spec_answer_mismatch_is_flagged():
    spec = chain_spec(5, horizon=2)
    spec["answer"] = "deadbeef" * 4
    res = provenance.verify_oracle(spec)
    assert not res["ok"]


def test_redaction_makes_the_oracle_unreachable_but_keeps_the_answer():
    spec = chain_spec(6, horizon=4)
    red = provenance.redact_spec(spec)
    assert red["control"] == "redacted"
    assert red["answer"] == spec["answer"]  # scoring target unchanged
    assert spec["hidden_key"] not in red["kb"]
    res = provenance.execute_oracle(red)
    assert not res["ok"]  # the hidden value is genuinely unavailable


def test_permutation_swaps_answers_between_task_ids():
    specs = [chain_spec(i, horizon=2) for i in range(20)]
    permuted = provenance.permute_hidden_values(specs, seed=0xA61E0008)
    assert len(permuted) == 20
    moved = 0
    for orig, perm in zip(specs, permuted):
        assert perm["task_id"] == orig["task_id"]
        res = provenance.execute_oracle(perm)
        assert res["ok"]
        # the replayed answer must be the DONOR's answer, tracked via the KB
        assert str(res["answer"]) == str(perm["answer"])
        if perm["answer"] != orig["answer"]:
            moved += 1
    assert moved > 0  # the permutation actually moved values


def test_permutation_is_deterministic():
    specs = [chain_spec(i, horizon=2) for i in range(12)]
    a = provenance.permute_hidden_values(copy.deepcopy(specs), seed=7)
    b = provenance.permute_hidden_values(copy.deepcopy(specs), seed=7)
    assert [s["answer"] for s in a] == [s["answer"] for s in b]
