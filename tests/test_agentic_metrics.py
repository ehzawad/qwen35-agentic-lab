"""Preregistered statistics: exact McNemar, Holm, the template-clustered
deterministic bootstrap, and the descriptive horizon curves."""

import math

from agentlab.suite.stats import (cluster_bootstrap_lb, holm, mcnemar_exact,
                                  wilson, _quantile_lower)


# ---------------------------------------------------------------------------
# exact McNemar
# ---------------------------------------------------------------------------

def test_mcnemar_exact_known_value():
    # b=1, c=9: two-sided exact = 2 * P(X <= 1 | n=10, p=1/2) = 22/1024
    res = mcnemar_exact(1, 9)
    assert math.isclose(res["p_two_sided"], 22 / 1024)
    assert math.isclose(res["p_one_sided"], (10 + 1) / 1024)


def test_mcnemar_exact_no_discordance_is_uninformative():
    res = mcnemar_exact(0, 0)
    assert res["p_two_sided"] == 1.0 and res["n_discordant"] == 0


def test_mcnemar_exact_stays_exact_at_large_n():
    # 40 vs 60 discordant: exact binomial, NOT a normal approximation
    res = mcnemar_exact(40, 60)
    exact_tail = sum(math.comb(100, i) for i in range(0, 41)) / 2**100
    assert math.isclose(res["p_two_sided"], min(1.0, 2 * exact_tail))


def test_mcnemar_symmetry():
    a, b = mcnemar_exact(3, 12), mcnemar_exact(12, 3)
    assert math.isclose(a["p_two_sided"], b["p_two_sided"])


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------

def test_holm_adjustment_ordering_and_monotonicity():
    adj = holm([0.01, 0.04, 0.03])
    assert math.isclose(adj[0], 0.03)          # 3 * 0.01
    assert adj[1] >= adj[2] - 1e-12            # monotone in the sorted order
    assert all(a <= 1.0 for a in adj)
    assert holm([0.5]) == [0.5]


# ---------------------------------------------------------------------------
# clustered bootstrap
# ---------------------------------------------------------------------------

def _mixed_data(n_clusters=40, per_cluster=10, delta=0.2):
    # clusters are deliberately heterogeneous (cluster c has (c % 5)/5 of the
    # mass shifted): a homogeneous fixture would make every resample identical
    diffs, clusters = [], []
    for c in range(n_clusters):
        k = round(per_cluster * delta * 2 * (c % 5) / 4)  # mean over c: delta
        for i in range(per_cluster):
            diffs.append(1.0 if i < k else 0.0)
            clusters.append(f"tpl{c:03d}")
    return diffs, clusters


def test_bootstrap_is_deterministic_for_a_committed_seed():
    diffs, clusters = _mixed_data()
    a = cluster_bootstrap_lb(diffs, clusters, seed=0xA61E000B, label="ER2",
                             replicates=5000)
    b = cluster_bootstrap_lb(diffs, clusters, seed=0xA61E000B, label="ER2",
                             replicates=5000)
    assert a["lb"] == b["lb"] and a["point"] == b["point"]


def test_bootstrap_stream_depends_on_seed_and_label():
    # the LB itself is a discrete order statistic and may coincide across
    # streams; the committed resampling STREAM must not.
    from agentlab.suite.rng import choice_indices

    base = choice_indices(0xA61E000B, "ER2:chunk0", 1000, 40)
    assert list(base) != list(choice_indices(0xA61E000B + 1, "ER2:chunk0", 1000, 40))
    assert list(base) != list(choice_indices(0xA61E000B, "MT1:chunk0", 1000, 40))


def test_bootstrap_constant_diffs_give_the_constant():
    diffs = [0.5] * 100
    clusters = [f"c{i % 10}" for i in range(100)]
    res = cluster_bootstrap_lb(diffs, clusters, seed=7, label="x", replicates=2000)
    assert math.isclose(res["lb"], 0.5) and math.isclose(res["point"], 0.5)


def test_bootstrap_lower_bound_is_below_the_point():
    diffs, clusters = _mixed_data(delta=0.3)
    res = cluster_bootstrap_lb(diffs, clusters, seed=3, label="x", replicates=5000)
    assert res["lb"] <= res["point"]
    assert math.isclose(res["point"], 0.3, abs_tol=1e-9)


def test_bootstrap_single_cluster_is_degenerate():
    res = cluster_bootstrap_lb([1.0, 0.0], ["t", "t"], seed=1, label="x",
                               replicates=100)
    assert res["degenerate"]


def test_bootstrap_empty_is_degenerate():
    res = cluster_bootstrap_lb([], [], seed=1, label="x", replicates=100)
    assert res["degenerate"]


def test_quantile_rule_is_the_committed_order_statistic():
    vals = sorted(range(100))  # 0..99
    assert _quantile_lower(vals, 0.025) == 2  # floor(0.025*100) = index 2
    assert _quantile_lower(vals, 0.0) == 0
    assert _quantile_lower(vals, 0.999999) == 99


def test_wilson_reexport_matches_analyze():
    from agentlab.analyze import wilson as w2

    assert wilson is w2


# ---------------------------------------------------------------------------
# horizon curves (descriptive; censored H50)
# ---------------------------------------------------------------------------

def _ep(family, horizon, ok):
    return {"family": family, "horizon": horizon, "template": "t", "pattern_id": None,
            "all_tools_required": False, "task_id": f"{family}{horizon}{ok}",
            "rep": {"certified_success": ok, "raw_success": ok, "n_calls": horizon,
                    "runaway": {"runaway": False, "reasons": []},
                    "hallucination": {"hallucinated": False, "reasons": []},
                    "receipts_ok": True, "answer_event_call_id": 1,
                    "n_decisions": horizon, "excess_calls": 0,
                    "n_events": horizon, "n_valid_events": horizon,
                    "distinct_decisions_with_calls": horizon,
                    "answer_extracted": "x"},
            "trace": {"runner": {"termination_reason": "answered"}}}


def test_h50_censoring_rules():
    import pathlib

    from agentlab.analyze import horizon_curves, load_preregister

    # the H50 crossing rate is a REGISTERED reporting rule, not a literal
    prereg = load_preregister(pathlib.Path(__file__).resolve().parents[1]
                              / "configs" / "agentic_preregister.json")
    assert prereg["machine"]["curves"]["h50_crossing_rate"] == 0.5

    eps = {("TP", "clean", "none"): {}}
    tasks = eps[("TP", "clean", "none")]
    i = 0
    for h, rate in [(2, 0.9), (4, 0.8), (8, 0.2)]:
        for j in range(10):
            e = _ep("fam", h, j < rate * 10)
            e["task_id"] = f"t{i}"
            i += 1
            tasks[e["task_id"]] = e
    curves = horizon_curves(eps, arms=("TP",), prereg=prereg)
    cur = curves["TP/clean/fam"]
    assert "crossed in [H4, H8]" == cur["H50"]
    # all-high curve: right-censored, no extrapolated number
    eps2 = {("TP", "clean", "none"): {f"x{j}": _ep("fam", h, True)
            for j, h in enumerate([2, 2, 4, 4, 8, 8])}}
    for j, (tid, e) in enumerate(eps2[("TP", "clean", "none")].items()):
        e["task_id"] = tid
    curves2 = horizon_curves(eps2, arms=("TP",), prereg=prereg)
    assert curves2["TP/clean/fam"]["H50"].startswith("right-censored")
