"""The registered claim and control splits actually exist, at their sizes.

Every gate in the preregistration is denominated in tasks. This file checks the
denominators the generator is responsible for -- the six typed_relay order
patterns, the structural-cluster granularity the clustered bootstrap resamples
over, the H8 augmentation, the two controls -- because the alternative failure
mode is silent: a short split does not look broken, it looks like a gate that
turned out "INCONCLUSIVE" after the GPU time was spent.
"""

from __future__ import annotations

import pytest

from agentlab.suite.envs.typed_relay import MT_COMBOS, MT_PATTERNS
from agentlab.suite.generate import (CLEAN_SPLITS, EVAL_FAULT_GROUPS,
                                     MT_MAX_PER_CLUSTER, MT_MIN_CLUSTERS,
                                     REGISTERED_DEV_PER_AXIS,
                                     REGISTERED_H8_PER_FAMILY,
                                     REGISTERED_H8_TOTAL, REGISTERED_TOTALS,
                                     SPLIT_SEED_KEY, SPLITS,
                                     absent_information_problems,
                                     assert_suite_cardinalities, build_split,
                                     build_task, cells_of, cluster_census,
                                     fault_group_census, observation_frontier)
from agentlab.suite.runtime import run_oracle
from agentlab.suite.schema import template_cluster_id, tool_pattern

from .conftest import SEEDS, SUITE


def _split(split: str, per_cell: int | None = None):
    from agentlab.suite.generate import DEFAULT_SIZES

    size = DEFAULT_SIZES[split] if per_cell is None else per_cell
    return build_split(SUITE, split, SEEDS[SPLIT_SEED_KEY[split]], size)


# ---------------------------------------------------------------------------
# structural template clusters: the bootstrap resampling unit
# ---------------------------------------------------------------------------

def test_every_scored_task_carries_a_structural_cluster_id():
    for split in SPLITS:
        for b in _split(split, 2)["bundles"]:
            assert b.spec.template_cluster_id.startswith("tc-")
            assert b.spec.template_cluster_id == template_cluster_id(
                b.spec.family, b.spec.horizon, b.nodes)


def test_cluster_id_is_structural_not_the_paraphrase_id():
    """Same structure, different drawn values and wording -> same cluster."""
    a = build_task(SUITE, SEEDS["eval"], "eval", "lookup_chain", 2, 0, None)
    b = build_task(SUITE, SEEDS["eval"], "eval", "lookup_chain", 2, 1, None)
    assert a.spec.answer != b.spec.answer          # different instantiations
    same_shape = ([sorted(n.expect["record"]) for n in a.nodes]
                  == [sorted(n.expect["record"]) for n in b.nodes])
    assert same_shape == (a.spec.template_cluster_id == b.spec.template_cluster_id)


def test_cluster_id_separates_horizons_and_families():
    ids = {build_task(SUITE, SEEDS["eval"], "eval", f, h, 0, None
                      ).spec.template_cluster_id
           for f, h in (("lookup_chain", 2), ("lookup_chain", 4),
                        ("typed_relay", 2), ("fulfillment", 4),
                        ("fulfillment", 8))}
    assert len(ids) == 5


def test_core_eval_no_longer_collapses_into_two_clusters():
    """The defect this field exists to fix.

    Held-out evaluation renders only paraphrase ids 10 and 11, so clustering the
    bootstrap on `template_id` gave the primary claim exactly two clusters --
    pooled across every family and horizon. Anything at that granularity makes a
    "97.5% clustered lower bound" a statement about two units of information.
    """
    bundles = _split("eval")["bundles"]
    assert len({b.spec.template_id for b in bundles}) == 2
    census = cluster_census(bundles)
    assert census["clusters"] >= len(cells_of("eval"))
    assert census["clusters"] > 100


# ---------------------------------------------------------------------------
# MT: the six registered tool-order patterns
# ---------------------------------------------------------------------------

def test_mt_split_is_six_balanced_registered_patterns():
    bundles = _split("eval_mt")["bundles"]
    assert len(bundles) == REGISTERED_TOTALS["eval_mt"] == 600
    per_pattern: dict = {}
    for b in bundles:
        assert (b.spec.family, b.spec.horizon) == ("typed_relay", 4)
        assert b.spec.pattern_id in range(len(MT_PATTERNS))
        per_pattern[b.spec.pattern_id] = per_pattern.get(b.spec.pattern_id, 0) + 1
    assert per_pattern == {p: 100 for p in range(6)}


def test_the_six_patterns_are_genuinely_distinct_orders():
    """Not one sequence relabelled six times: six different tool ORDERS."""
    orders = {}
    for b in _split("eval_mt")["bundles"]:
        orders.setdefault(b.spec.pattern_id, set()).add(tool_pattern(b.nodes))
    assert len(orders) == 6
    for pid, got in orders.items():
        assert got == {MT_PATTERNS[pid]}
    assert len({next(iter(v)) for v in orders.values()}) == 6


def test_every_mt_pattern_requires_all_three_tools():
    for b in _split("eval_mt", 6)["bundles"]:
        assert {"kb_lookup", "unit_convert", "calculator"} <= {n.tool
                                                               for n in b.nodes}


def test_mt_later_calls_consume_earlier_results():
    """Each node's arguments contain a value only its predecessor produced.

    This is what makes the registered order the ONLY order that can succeed: an
    agent that reorders the calls cannot construct the arguments. Values are
    matched the way `provenance.certify_orchestration` matches them -- an exposed
    number appearing inside a later argument, on a word boundary.
    """
    import json
    import re

    for b in _split("eval_mt", 4)["bundles"]:
        front = observation_frontier(b)
        assert front["broke_at"] is None, b.spec.task_id
        for i in range(1, len(b.nodes)):
            prior = json.loads(front["exposed"][i - 1])
            produced = []
            if "value" in prior:
                produced.append(str(prior["value"]))
            if isinstance(prior.get("record"), dict):
                produced += [str(v) for v in prior["record"].values()]
            consumed = " ".join(str(v) for v in b.nodes[i].args.values())
            assert any(re.search(rf"(?<![\w.]){re.escape(p)}(?![\w.])", consumed)
                       for p in produced if p), (b.spec.task_id, i)


def test_mt_clusters_meet_the_registered_floor_and_ceiling():
    census = cluster_census(_split("eval_mt")["bundles"])
    assert census["clusters"] >= MT_MIN_CLUSTERS
    assert census["max_per_cluster"] <= MT_MAX_PER_CLUSTER
    # the registered allocation: 6 patterns x 50 (conversion, form) templates
    assert census["clusters"] == len(MT_PATTERNS) * len(MT_COMBOS)


def test_the_mt_ceiling_holds_over_the_registered_gated_stratum():
    """MT1's stratum is ["eval", "eval_mt"], and the ceiling binds over BOTH.

    Core eval's all-tools H4 tasks live in the same structural cluster space as
    MT order pattern 0. With their (conversion, form) template drawn at random
    the combined sample put 8 instantiations in one cluster against a registered
    ceiling of 5 -- enough to make MT1 INCONCLUSIVE on a clustering technicality
    while the 600 MT tasks looked perfectly balanced on their own.
    """
    data = {"eval": _split("eval")["bundles"],
            "eval_mt": _split("eval_mt")["bundles"]}
    census = assert_suite_cardinalities(data)
    assert census["mt_stratum_pairs"] == 700
    assert census["mt_stratum_clusters"] >= MT_MIN_CLUSTERS
    assert census["mt_stratum_max_per_cluster"] <= MT_MAX_PER_CLUSTER


def test_every_mt_oracle_replays_to_strict_success():
    for b in _split("eval_mt", 4)["bundles"]:
        _rt, verdict = run_oracle(b.spec, b.kb, b.nodes)
        assert verdict.strict_success, (b.spec.task_id, verdict.reasons[:2])


# ---------------------------------------------------------------------------
# H8 augmentation and the core-eval fault groups
# ---------------------------------------------------------------------------

def test_h8_reaches_four_hundred_pairs_two_hundred_per_family():
    data = {"eval": _split("eval")["bundles"],
            "eval_h8": _split("eval_h8")["bundles"]}
    census = assert_suite_cardinalities(data)
    assert census["h8_pairs"] == REGISTERED_H8_TOTAL == 400
    assert census["h8_per_family"] == {"lookup_chain": REGISTERED_H8_PER_FAMILY,
                                       "typed_relay": REGISTERED_H8_PER_FAMILY}


def test_core_eval_fault_groups_are_exactly_four_hundred_each():
    """Was 685 / 340 / 175: wrong-unit could never reach its registered 400."""
    got = fault_group_census(_split("eval")["bundles"])
    assert got == EVAL_FAULT_GROUPS == {"transient_rate_limit": 400,
                                        "malformed": 400, "wrong_unit": 400}


def test_claim_and_control_splits_carry_no_faults():
    for split in CLEAN_SPLITS:
        for b in _split(split, 2)["bundles"]:
            assert b.spec.faults == []


def test_dev_supplies_three_hundred_per_claim_axis():
    census = assert_suite_cardinalities({"dev": _split("dev")["bundles"]})
    for axis in ("recovery", "orchestration", "h8"):
        assert census[f"dev_{axis}_axis"] >= REGISTERED_DEV_PER_AXIS


# ---------------------------------------------------------------------------
# the absent-information control
# ---------------------------------------------------------------------------

def test_absent_control_covers_all_three_families_per_arm():
    bundles = _split("eval_absent")["bundles"]
    counts: dict = {}
    for b in bundles:
        counts[b.spec.family] = counts.get(b.spec.family, 0) + 1
    assert counts == {"lookup_chain": 200, "typed_relay": 200,
                      "fulfillment": 200}


@pytest.mark.parametrize("family", ["lookup_chain", "typed_relay", "fulfillment"])
def test_absent_redaction_is_family_specific_and_effective(family):
    """A KB deletion does nothing for two of the three families.

    Express fulfillment reads no KB record at all, and a typed_relay numeric
    terminal is computed rather than retrieved -- so each family gets the
    redaction that actually withholds ITS hidden value, and the proof is against
    the unredacted twin rather than against the descriptor's own claim.
    """
    cell = {c.family: c for c in cells_of("eval_absent")}[family]
    seed = SEEDS[SPLIT_SEED_KEY["eval_absent"]]
    for index in range(4):
        bundle = build_task(SUITE, seed, "eval_absent", cell.family,
                            cell.horizon, index, None, variant=cell.variant)
        twin = build_task(SUITE, seed, "eval_absent", cell.family, cell.horizon,
                          index, None, variant=cell.variant,
                          apply_control=False)
        assert bundle.spec.control == "redacted"
        assert absent_information_problems(bundle, twin) == []
        assert bundle.spec.control_meta["hidden_entropy_bits"] >= 48


def test_absent_typed_relay_terminal_keeps_forty_eight_bits():
    """The numeric terminal is unguessable, not merely unretrieved."""
    bundles = [b for b in _split("eval_absent", 8)["bundles"]
               if b.spec.family == "typed_relay"]
    assert bundles
    for b in bundles:
        assert b.spec.answer_kind == "integer"
        assert int(b.spec.answer) > (1 << 48)
        assert b.spec.control_meta["hidden_entropy_bits"] == 48


def test_absent_fulfillment_withholds_the_completion_token():
    bundles = [b for b in _split("eval_absent", 4)["bundles"]
               if b.spec.family == "fulfillment"]
    assert bundles
    for b in bundles:
        assert b.spec.env["completion_token"] is None
        assert b.spec.answer.startswith("CMP-")
        assert b.spec.answer not in b.spec.secret_tokens
        front = observation_frontier(b)
        # the order still completes; the token is simply not in the envelope
        assert front["broke_at"] is None
        assert "completion_token" not in front["exposed"][-1]
        assert all(b.spec.answer not in p for p in front["exposed"])


# ---------------------------------------------------------------------------
# the counterfactual permutation control
# ---------------------------------------------------------------------------

def test_permutation_is_exactly_one_hundred_and_a_derangement():
    result = _split("eval_perm")
    bundles = result["bundles"]
    assert len(bundles) == REGISTERED_TOTALS["eval_perm"] == 100
    donors = result["permutation"]
    assert len(donors) == 100
    assert len(set(donors.values())) == 100          # a bijection
    for b in bundles:
        meta = b.spec.control_meta
        assert meta["donor_task_id"] != b.spec.task_id     # no fixed point
        assert meta["original_answer"] != meta["permuted_answer"]
        assert b.spec.answer == meta["permuted_answer"]


def test_permuted_task_scores_against_the_returned_value_not_the_original():
    for b in _split("eval_perm", 6)["bundles"]:
        meta = b.spec.control_meta
        rt, verdict = run_oracle(b.spec, b.kb, b.nodes)
        assert verdict.strict_success, b.spec.task_id
        front = observation_frontier(b)
        assert any(meta["permuted_answer"] in p for p in front["exposed"])
        assert all(meta["original_answer"] not in p for p in front["exposed"])
        assert meta["original_answer"] not in b.spec.prompt


def test_permutation_values_stay_globally_unique_tokens():
    """A permutation is a bijection, so the split's tokens are still distinct."""
    bundles = _split("eval_perm")["bundles"]
    answers = [b.spec.answer for b in bundles]
    originals = [b.spec.control_meta["original_answer"] for b in bundles]
    assert len(set(answers)) == len(answers)
    assert set(answers) == set(originals)


# ---------------------------------------------------------------------------
# a shortfall must fail LOUDLY, at generation time
# ---------------------------------------------------------------------------

def test_a_short_mt_split_raises_instead_of_being_written():
    """The whole point of the assertions: shortfalls are not discovered later.

    A generator misconfigured to 80 tasks per pattern would previously have
    written a 480-task MT split, and the shortfall would first have surfaced as
    an "INCONCLUSIVE" MT1 gate after the GPU budget was already spent.
    """
    from agentlab.suite.generate import assert_split_cardinalities

    result = _split("eval_mt", 80)
    with pytest.raises(AssertionError, match="registered"):
        assert_split_cardinalities("eval_mt", result, 100)


def test_a_dev_split_too_small_for_the_tournament_raises():
    small = _split("dev", 20)["bundles"]
    # 20 per cell is what the tree carried before: the orchestration axis is a
    # single cell, so the tournament had 20 of the 300 registered instances.
    counts = {}
    for b in small:
        counts[(b.spec.family, b.spec.horizon)] = counts.get(
            (b.spec.family, b.spec.horizon), 0) + 1
    assert counts[("typed_relay", 4)] == 20
    # the cross-split assertion only binds at the registered total, so state the
    # arithmetic directly: 100 round-one + 200 round-two instances need 300.
    assert counts[("typed_relay", 4)] < REGISTERED_DEV_PER_AXIS


def test_registered_tables_are_self_consistent():
    from agentlab.suite.generate import (DEFAULT_SIZES, EVAL_FAULT_ORDER,
                                         EVAL_FAULT_WEIGHTS)

    for split in SPLITS:
        assert (DEFAULT_SIZES[split] * len(cells_of(split))
                == REGISTERED_TOTALS[split]), split
    groups = {"transient_rate_limit": 0, "malformed": 0, "wrong_unit": 0}
    for cell, weights in EVAL_FAULT_WEIGHTS.items():
        assert sum(weights) == DEFAULT_SIZES["eval"], cell
        for name, w in zip(EVAL_FAULT_ORDER, weights):
            key = ("transient_rate_limit"
                   if name in ("transient", "rate_limit") else name)
            groups[key] += w
    assert groups == EVAL_FAULT_GROUPS


def test_manifest_records_the_registered_numbers(tmp_path):
    from agentlab.suite.generate import generate_all, load_suite_config

    cfg = load_suite_config(str(__import__("pathlib").Path(
        __file__).resolve().parents[2] / "configs" / "suite_v1.toml"))
    cfg = dict(cfg, sizes={k: 2 for k in cfg["sizes"]})
    manifest = generate_all(cfg, str(tmp_path / "v1"))
    assert manifest["registered_totals"] == REGISTERED_TOTALS
    assert manifest["registered_fault_groups"] == EVAL_FAULT_GROUPS
    assert manifest["registered_h8_pairs"] == REGISTERED_H8_TOTAL
    assert manifest["registered_mt_clusters"] == {
        "min_clusters": MT_MIN_CLUSTERS, "max_per_cluster": MT_MAX_PER_CLUSTER}
    assert manifest["splits"]["eval_mt"]["patterns"] == {
        str(p): MT_PATTERNS[p] for p in range(6)}
    assert manifest["config"]["sha256"] == cfg["config_sha256"]
    # every split's payload is listed, with POSIX separators
    assert all("/" in rel or rel == "manifest.json" for rel in manifest["files"])
    for split in SPLITS:
        assert f"specs/{split}.jsonl" in manifest["files"]
