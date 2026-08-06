"""One environment stack, and every consumer provably faithful to it.

A repointed consumer can compile, import and run while producing semantically
DIFFERENT episodes from the ones the verifier scores. These tests refuse to
accept "it imports" as reconciliation. For every family/horizon cell and every
fault type they require:

  * the trajectory a consumer builds replays through the canonical
    `EpisodeRuntime` to IDENTICAL canonical and exposed observation digests;
  * identical credited oracle-node progress (node -> completing decision);
  * an identical verifier verdict row;
  * and the certification layer's independent oracle replay of the same task
    reproduces the canonical node payloads byte for byte.

The tampering tests exist to prove the assertions have teeth: a single altered
argument, decision id, or observation must be caught.
"""

from __future__ import annotations

import copy
import json

import pytest
from rollout_helpers import (TEST_SECRET, BlindRetryPolicy, OraclePolicy,
                             run_engine, token_counter_stub)

from agentlab import provenance
from agentlab.suite import runtime as rt_mod
from agentlab.suite.configio import load_config
from agentlab.suite.generate import build_task, certification_spec
from agentlab.suite.schema import CELLS, FAMILIES, canon, oracle_plan_digest

SUITE = "agentlab-suite-v1"
SEED = 0xA61E0002          # the committed distill seed
# Held-out splits have no seed until L exists; this is the clearly labelled
# sentinel the suite tests use (tests/suite/conftest.py), and using it validates
# the MECHANISM, never the designated held-out realization.
SENTINEL_HELDOUT_SEED = 0x5E4714E1_5E4714E1_5E4714E1_5E4714E1
CFG = load_config()

# One clean and one of each fault type per cell, plus the ambiguous post-mutation
# malformed case where the family supports it.
_FAULT_CASES = [None, [("transient", False)], [("rate_limit", False)],
                [("malformed", False)]]


def _cases_for(family: str, horizon: int) -> list:
    from agentlab.suite.envs import family_module

    cases = list(_FAULT_CASES)
    if family_module(family).has_unit_convert(horizon):
        cases.append([("wrong_unit", False)])
    if family == "fulfillment":
        cases.append([("malformed", True)])
    return cases


def _bundle(family: str, horizon: int, entries, index: int = 0):
    return build_task(SUITE, SEED, "distill", family, horizon, index, entries)


def _all_bundles() -> list:
    out = []
    for i, (family, horizon) in enumerate(CELLS):
        for j, entries in enumerate(_cases_for(family, horizon)):
            out.append(_bundle(family, horizon, entries, index=10 * i + j))
    return out


# ---------------------------------------------------------------------------
# exactly one implementation of each concept
# ---------------------------------------------------------------------------

def test_the_losing_environment_stack_is_gone():
    import importlib

    for name in ("agentlab.suite.environments", "agentlab.suite.scenarios"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_every_consumer_uses_the_canonical_families():
    import agentlab.multidistill as md
    import agentlab.variance as var

    assert FAMILIES == ("lookup_chain", "typed_relay", "fulfillment")
    assert sorted(CFG["mixture"]) == sorted(FAMILIES)
    assert sorted(CFG["acceptance"]["max_redundant_calls"]) == sorted(FAMILIES)
    assert {f for f, _h in CELLS} == set(FAMILIES)
    # the training path reads the mixture by family key, so a stale key would
    # KeyError at rollout time rather than here
    for family, horizon in CELLS:
        assert CFG["mixture"][family]["k"][f"h{horizon}"] >= 4
        assert family in CFG["acceptance"]["call_upper_slack"]
    assert md.rs_split(CFG) in ("oracle_sft", "distill", "grpo_train")
    assert var.load_config is md.load_config


def test_tool_surface_has_one_definition():
    for family in FAMILIES:
        schemas = rt_mod.tool_schemas_for_family(family)
        names = rt_mod.tool_names_for_family(family)
        assert names == [s["function"]["name"] for s in schemas]
        assert {"calculator", "unit_convert", "kb_lookup"} <= set(names)
    assert set(rt_mod.tool_names_for_family("fulfillment")) - set(
        rt_mod.tool_names_for_family("typed_relay")) == {"warehouse_query",
                                                         "warehouse_update"}


def test_canonical_payload_is_shared_with_the_certification_layer():
    """provenance.canonical_dispatch must not be a second tool implementation."""
    bundle = _bundle("typed_relay", 4, None)
    for node in bundle.nodes:
        theirs = provenance.canonical_dispatch(bundle.kb, node.tool, node.args)
        ours, _meta = rt_mod.canonical_payload(node.tool, node.args, kb=bundle.kb)
        assert theirs == ours == node.expect


def test_observations_stay_inside_the_measured_size_ceiling():
    limit = CFG["scenario"]["tool_output_max_chars"]
    worst = 0
    for family, horizon in CELLS:
        for index in range(3):
            bundle = _bundle(family, horizon, None, index=index)
            for node in bundle.nodes:
                worst = max(worst, len(canon(node.expect)))
    assert worst <= limit, f"canonical observation grew to {worst} chars"


# ---------------------------------------------------------------------------
# the training-path consumer: replay parity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rollouts():
    bundles = _all_bundles()
    records = run_engine(bundles, cfg=CFG)
    return {b.spec.task_id: b for b in bundles}, {r["task_id"]: r for r in records}


def test_the_scripted_oracle_reaches_certified_success_everywhere(rollouts):
    bundles, records = rollouts
    failures = {tid: rec["verdict"]["reasons"] for tid, rec in records.items()
                if not rec["verdict"]["certified_success"]}
    assert failures == {}, failures
    # and the fault arms really did fire and really were recovered
    for tid, rec in records.items():
        if bundles[tid].spec.faults:
            assert rec["fault"]["fired"], tid
            assert rec["verdict"]["recovered"], tid
            assert rec["verdict"]["recovery_success"], tid


def test_every_trajectory_replays_to_identical_digests(rollouts):
    """The core faithfulness assertion, for all 12 cells x every fault type."""
    from agentlab.multidistill import replay_record

    bundles, records = rollouts
    for tid, rec in records.items():
        ok, why = replay_record(rec, bundles[tid], secret=TEST_SECRET)
        assert ok, f"{tid}: {why}"


def test_replayed_progress_and_digests_are_recomputed_not_copied(rollouts):
    bundles, records = rollouts
    for tid, rec in records.items():
        bundle = bundles[tid]
        runtime, report = rt_mod.replay_trace(bundle.spec, bundle.kb, bundle.nodes,
                                              rec["calls"], secret=TEST_SECRET)
        assert report["observations"] == rec["parity"]["observations"]
        assert report["progress"] == rec["parity"]["progress"]
        assert report["episode"] == rec["parity"]["episode"]
        # progress is a real map of node -> completing decision, in order
        progress = report["progress"]
        assert list(progress) == [n.node_id for n in bundle.nodes]
        decisions = list(progress.values())
        assert decisions == sorted(decisions)
        assert len(set(decisions)) == len(decisions), "two nodes in one decision"
        assert runtime.verify(rec["final"], transcript=rec["messages"],
                              termination_reason="answered").to_row() == rec["verdict"]


def test_accepted_records_pass_the_full_acceptance_pass(rollouts):
    from agentlab.multidistill import accept_record

    bundles, records = rollouts
    for tid, rec in records.items():
        ok, why = accept_record(rec, CFG, bundles, secret=TEST_SECRET)
        assert ok, f"{tid}: {why}"


@pytest.mark.parametrize("mutation", ["args", "decision", "observation", "extra",
                                      "dropped", "progress"])
def test_a_tampered_trajectory_is_rejected(rollouts, mutation):
    """Every parity assertion must actually be able to fail."""
    from agentlab.multidistill import accept_record, replay_record

    bundles, records = rollouts
    tid = next(t for t, r in records.items()
               if r["family"] == "typed_relay" and r["horizon"] == 4)
    bundle = bundles[tid]
    rec = copy.deepcopy(records[tid])

    if mutation == "args":
        target = next(c for c in rec["calls"] if c["tool"] == "kb_lookup")
        target["args"] = dict(target["args"], key="K" + "A" * 16)
    elif mutation == "decision":
        for call in rec["calls"][1:]:
            call["decision_id"] = 1
    elif mutation == "observation":
        rec["calls"][0]["exposed"] = json.dumps({"ok": True, "record": {"next": "X"}})
    elif mutation == "extra":
        rec["calls"].append(dict(rec["calls"][-1]))
    elif mutation == "dropped":
        rec["calls"] = rec["calls"][:-1]
    elif mutation == "progress":
        rec["parity"]["progress"] = {k: 1 for k in rec["parity"]["progress"]}

    ok, why = replay_record(rec, bundle, secret=TEST_SECRET)
    assert not ok and why.startswith("replay_"), why
    accepted, reason = accept_record(rec, CFG, bundles, secret=TEST_SECRET)
    assert not accepted, reason


def test_a_consumer_cannot_be_handed_the_wrong_task(rollouts):
    from agentlab.multidistill import replay_record

    bundles, records = rollouts
    tid = next(t for t, r in records.items() if r["family"] == "lookup_chain")
    other = next(b for t, b in bundles.items()
                 if t != tid and b.spec.family == "lookup_chain")
    ok, why = replay_record(records[tid], other, secret=TEST_SECRET)
    assert not ok and "replay_wrong_bundle" in why


# ---------------------------------------------------------------------------
# acceptance filters have teeth
# ---------------------------------------------------------------------------

def test_a_missing_oracle_node_is_never_accepted():
    from agentlab.multidistill import accept_record

    bundle = _bundle("lookup_chain", 4, None, index=77)
    records = run_engine([bundle], policy=OraclePolicy([bundle], skip_node=2), cfg=CFG)
    rec = records[0]
    assert not rec["verdict"]["certified_success"]
    ok, why = accept_record(rec, CFG, {bundle.spec.task_id: bundle}, secret=TEST_SECRET)
    assert not ok and why == "verifier_rejected"


def test_a_correct_answer_without_a_recovered_fault_is_not_certified():
    """Mental arithmetic must never masquerade as recovery."""
    from agentlab.multidistill import accept_record

    bundle = _bundle("typed_relay", 2, [("wrong_unit", False)], index=78)
    answer = bundle.spec.answer
    # Stop after the faulted conversion and commit the right answer anyway.
    policy = OraclePolicy([bundle], break_at=1,
                          terminal_text=f"I know it: \\boxed{{{answer}}}")
    rec = run_engine([bundle], policy=policy, cfg=CFG)[0]
    assert rec["verdict"]["answer_ok"] is True
    assert rec["verdict"]["certified_success"] is False
    ok, why = accept_record(rec, CFG, {bundle.spec.task_id: bundle}, secret=TEST_SECRET)
    assert not ok and why == "verifier_rejected"


def test_harmless_extra_read_only_calls_earn_nothing_but_stay_acceptable():
    from agentlab.multidistill import accept_record, replay_record

    bundle = _bundle("typed_relay", 4, None, index=79)
    policy = OraclePolicy([bundle], extra_call=(1, "calculator",
                                                {"expression": "2+2"}))
    rec = run_engine([bundle], policy=policy, cfg=CFG)[0]
    assert rec["verdict"]["certified_success"]
    assert rec["verdict"]["calls"] == bundle.spec.horizon + 1
    assert rec["verdict"]["unique_valid_nodes"] == bundle.spec.horizon
    ok, why = replay_record(rec, bundle, secret=TEST_SECRET)
    assert ok, why
    ok, why = accept_record(rec, CFG, {bundle.spec.task_id: bundle}, secret=TEST_SECRET)
    assert ok, why


def test_a_missing_committed_answer_is_rejected_before_the_verifier():
    """No commitment under the ONE grammar -- not "no \\boxed{}" specifically.

    The bucket is `no_committed_answer` because the filter asks
    `schema.extract_committed_answer`: a trajectory that terminated with the
    preregistered `ANSWER: <value>` form DID commit and must not be dropped
    (tests/test_corpus_completion.py pins that half).
    """
    from agentlab.multidistill import accept_record

    bundle = _bundle("lookup_chain", 2, None, index=80)
    policy = OraclePolicy([bundle], terminal_text="the code is right here")
    rec = run_engine([bundle], policy=policy, cfg=CFG)[0]
    ok, why = accept_record(rec, CFG, {bundle.spec.task_id: bundle}, secret=TEST_SECRET)
    assert not ok and why == "no_committed_answer"


# ---------------------------------------------------------------------------
# the certification consumer: the same tasks, the same canonical observations
# ---------------------------------------------------------------------------

def test_certification_specs_replay_to_the_canonical_node_payloads():
    for family, horizon in CELLS:
        for entries in (None, [("transient", False)]):
            bundle = _bundle(family, horizon, entries, index=5)
            spec = certification_spec(bundle)
            res = provenance.verify_oracle(spec)
            assert res["ok"], (family, horizon, res["problems"])
            assert len(res["nodes"]) == horizon
            for node, got in zip(bundle.nodes, res["nodes"]):
                assert got["envelope"] == node.expect, (family, horizon, node.node_id)
                assert got["args_digest"] == provenance.call_digest(node.tool,
                                                                   node.args)
            assert str(res["replayed_answer"]) == bundle.spec.answer


def test_certification_specs_carry_the_split_isolation_fields():
    from agentlab.suite.splits import check_split_leakage

    groups = {"train": [], "dev": [], "eval": []}
    # The eval row is built with the labelled SENTINEL held-out seed and a sentinel
    # release id: after D3 a held-out certspec cannot be exported at all without a
    # release id, because a row that carries none is an old-seed cached value.
    sentinel_release = "0" * 64
    for family, horizon in CELLS:
        for split, seed, group in (("distill", 0xA61E0002, "train"),
                                   ("dev", 0xA61E0004, "dev"),
                                   ("eval", SENTINEL_HELDOUT_SEED, "eval")):
            bundle = build_task(SUITE, seed, split, family, horizon, 0, None)
            if split == "eval":
                bundle.release_id = sentinel_release
            groups[group].append(certification_spec(bundle))
    for spec in groups["eval"]:
        assert spec["heldout_release_id"] == sentinel_release
    for group in ("train", "dev"):
        for spec in groups[group]:
            assert "heldout_release_id" not in spec
    assert check_split_leakage(groups) == []
    for specs in groups.values():
        for spec in specs:
            assert spec["counterfactual_sensitive"] is True
            assert spec["template_hash"]
            if spec["redactable"]:
                assert spec["hidden_key"] in spec["kb"]
            else:
                # only express fulfillment, whose hidden value is the finalize
                # completion token rather than a KB record
                assert (spec["family"], spec["horizon"]) == ("fulfillment", 4)
                assert spec["hidden_key"] is None


def test_unredactable_specs_are_dropped_from_the_absent_information_arm():
    from agentlab.suite import evaluate

    specs = [certification_spec(_bundle(f, h, None, index=9))
             for f, h in CELLS]
    redacted = evaluate.apply_control(specs, "redacted", permutation_seed=1)
    assert {s["task_id"] for s in redacted} == {
        s["task_id"] for s in specs if s["redactable"]}
    for spec, red in zip([s for s in specs if s["redactable"]], redacted):
        assert spec["hidden_key"] not in red["kb"]


def test_the_certification_runtime_scores_a_generated_task():
    """evaluate.run_episode over an adapted spec, with a scripted chat backend."""
    from agentlab.suite import evaluate

    bundle = _bundle("lookup_chain", 4, None, index=6)
    spec = certification_spec(bundle)
    nodes = list(bundle.nodes)

    def chat_fn(messages, tools):
        assert any(t["function"]["name"] == "kb_lookup" for t in tools)
        assert all("recovery_token" in t["function"]["parameters"]["properties"]
                   for t in tools)
        done = sum(1 for m in messages if m.get("role") == "tool")
        if done >= len(nodes):
            return {"content": f"\\boxed{{{bundle.spec.answer}}}", "tool_calls": []}
        node = nodes[done]
        return {"content": "", "tool_calls": [{"name": node.tool,
                                               "arguments": dict(node.args)}]}

    trace = evaluate.run_episode(
        spec, arm="B0", condition="clean", control="none",
        secret=bytes.fromhex("bb" * 32), fault_seed=1,
        system_prompt="sys", prompt_meta={"path": "-", "sha256": "-"},
        chat_fn=chat_fn, decode={"temperature": 0.0, "top_p": 1.0, "seed": 0,
                                 "max_tokens": 128},
        run_meta={"run_id": "test"})
    assert trace["score"]["raw_success"] is True
    assert trace["score"]["certified_success"] is True
    assert trace["budgets"] == {"max_decisions": bundle.spec.max_decisions,
                                "max_calls": bundle.spec.max_calls}


def test_the_two_consumers_agree_on_the_oracle_plan():
    """The plan digest binds the training path and the certification layer."""
    for family, horizon in CELLS:
        bundle = _bundle(family, horizon, None, index=7)
        spec = certification_spec(bundle)
        replay = provenance.execute_oracle(spec)
        assert replay["ok"], (family, horizon, replay.get("error"))
        rebuilt = [{"node_id": n["node"], "tool": n["tool"],
                    "args": n["args"], "expect": n["envelope"]}
                   for n in replay["nodes"]]
        from agentlab.suite.schema import OracleNode

        assert oracle_plan_digest(
            [OracleNode(node_id=r["node_id"], tool=r["tool"], args=r["args"],
                        expect=r["expect"], match={}) for r in rebuilt]
        ) == oracle_plan_digest(bundle.nodes)


# ---------------------------------------------------------------------------
# the SFT-view consumer
# ---------------------------------------------------------------------------

def test_views_are_single_assistant_completions_from_the_real_transcript():
    from agentlab.suite.datasets import build_views, select_views

    bundle = _bundle("typed_relay", 8, [("transient", False)], index=81)
    rec = run_engine([bundle], cfg=CFG)[0]
    assert rec["verdict"]["certified_success"]
    plan = select_views(rec, CFG)
    kinds = {item["view"] for item in plan}
    assert "terminal" in kinds and "recovery" in kinds and "pivot" in kinds
    terminal = next(i for i in plan if i["view"] == "terminal")
    assert terminal["copies"] == CFG["views"]["terminal_copies"]

    rows, meta, report = build_views([rec], token_counter_stub(), CFG)
    assert report["rows"] == sum(i["copies"] for i in plan)
    for row in rows:
        assert len(row["completion"]) == 1
        assert row["completion"][0]["role"] == "assistant"
        # everything before the completion is masked prompt context, including
        # every tool result and the injected error
        assert all(m["role"] in ("system", "user", "assistant", "tool")
                   for m in row["prompt"])
        assert row["completion"][0] in rec["messages"]
        assert row["chat_template_kwargs"] == {"enable_thinking": False}
    assert {m["task_id"] for m in meta} == {rec["task_id"]}


def test_pivot_views_are_dependency_bearing():
    from agentlab.suite.datasets import consumes_predecessor, select_views

    bundle = _bundle("lookup_chain", 8, None, index=82)
    rec = run_engine([bundle], cfg=CFG)[0]
    plan = select_views(rec, CFG)
    pivots = [i["index"] for i in plan if i["view"] == "pivot"]
    assert pivots, "an 8-hop chain must expose a dependency-bearing pivot"
    for i in pivots:
        assert consumes_predecessor(rec["messages"], i)


def test_view_selection_is_deterministic_for_a_task():
    from agentlab.suite.datasets import select_views

    bundle = _bundle("typed_relay", 8, None, index=83)
    rec = run_engine([bundle], cfg=CFG)[0]
    assert select_views(rec, CFG) == select_views(copy.deepcopy(rec), CFG)


def test_an_oversized_terminal_view_rejects_the_whole_trajectory():
    from agentlab.suite.datasets import build_views

    bundle = _bundle("lookup_chain", 2, None, index=84)
    rec = run_engine([bundle], cfg=CFG)[0]
    huge = token_counter_stub()

    def counter(prompt_msgs, completion_msgs, tools):
        return CFG["acceptance"]["max_view_tokens"] + 1

    rows, _meta, report = build_views([rec], counter, CFG)
    assert rows == []
    assert report["rejected"]["trajectory_over_budget"] == 1
    assert build_views([rec], huge, CFG)[2]["rows"] > 0
