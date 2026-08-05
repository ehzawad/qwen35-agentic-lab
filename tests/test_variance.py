"""The GRPO grip probe: reward ordering, group variance, and the four gates.

The probe exists because the previous null was a variance failure, not a learning
failure: 56% of training groups had zero reward variance. So these tests pin the
two things that could quietly make the probe useless -- a reward that lets a
failure outscore a success, and a "variance" measure that counts shaping spread
as terminal disagreement -- and they check that every eligibility threshold in
configs/multifaceted.yaml is computable from the probe's own output alone.
"""

from __future__ import annotations

import pytest
from rollout_helpers import OraclePolicy, run_engine

from agentlab import variance
from agentlab.suite.configio import load_config
from agentlab.suite.generate import build_task, cell_slice, group_by_cell
from agentlab.suite.schema import CELLS, FAMILIES

CFG = load_config()
SUITE = "agentlab-suite-v1"
SEED = 0xA61E0003          # the committed grpo_train seed


def _rec(*, success, milestone=1.0, calls=4, horizon=4, family="typed_relay",
         final="\\boxed{7}", truncated=False, exhausted=False, unsafe=False,
         recovery=False, faults=(), task_id="t", consistent=True):
    return {
        "task_id": task_id, "family": family, "horizon": horizon,
        "fault_types": list(faults), "final": final, "truncated": truncated,
        "exhausted": exhausted, "milestone_fraction": milestone,
        "verdict": {"strict_success": success, "recovery_success": recovery,
                    "calls": calls, "unsafe_mutation": unsafe,
                    "consistent": consistent},
    }


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

def test_no_failure_can_outscore_any_success_in_any_cell():
    """The lexicographic property, checked at the reachable boundary.

    A success cannot be arbitrarily wasteful: the verifier refuses success past
    `call_budget(H) = 2H+4`, so the worst-scoring success in a cell is one that
    spends every allowed call. Compare that against the best-scoring failure
    (full milestone credit and a certified recovery) for each of the 12 cells.
    """
    from agentlab.suite.schema import call_budget

    lo, hi = CFG["grpo"]["reward"]["clamp"]
    for family, horizon in CELLS:
        best_failure = variance.lexicographic_reward(
            _rec(success=False, milestone=1.0, horizon=horizon,
                 calls=horizon, recovery=True, family=family), CFG)
        worst_success = variance.lexicographic_reward(
            _rec(success=True, milestone=1.0, horizon=horizon,
                 calls=call_budget(horizon), family=family), CFG)
        assert best_failure < worst_success, (family, horizon)
        assert lo <= best_failure <= hi and lo <= worst_success <= hi


def test_reward_is_clamped_as_a_total():
    lo, hi = CFG["grpo"]["reward"]["clamp"]
    ceiling = variance.lexicographic_reward(
        _rec(success=True, milestone=1.0, calls=4, recovery=True), CFG)
    floor = variance.lexicographic_reward(
        _rec(success=False, milestone=0.0, calls=400, final="", unsafe=True), CFG)
    assert ceiling <= hi and floor >= lo


def test_partial_credit_is_bounded_and_monotone():
    weights = CFG["grpo"]["reward"]["weights"]
    zero = variance.lexicographic_reward(_rec(success=False, milestone=0.0, calls=4), CFG)
    half = variance.lexicographic_reward(_rec(success=False, milestone=0.5, calls=4), CFG)
    full = variance.lexicographic_reward(_rec(success=False, milestone=1.0, calls=4), CFG)
    assert zero < half < full
    assert full - zero == pytest.approx(weights["verified_milestone_fraction"])


def test_excess_calls_and_missing_commitment_are_penalised():
    base = variance.lexicographic_reward(_rec(success=True, calls=4, horizon=4), CFG)
    wasteful = variance.lexicographic_reward(_rec(success=True, calls=9, horizon=4), CFG)
    silent = variance.lexicographic_reward(
        _rec(success=False, milestone=1.0, calls=4, final=""), CFG)
    committed = variance.lexicographic_reward(
        _rec(success=False, milestone=1.0, calls=4), CFG)
    assert wasteful < base
    assert silent < committed


def test_clipping_counts_both_truncation_and_horizon_exhaustion():
    assert variance.is_clipped(_rec(success=False, truncated=True))
    assert variance.is_clipped(_rec(success=False, exhausted=True))
    assert not variance.is_clipped(_rec(success=True))


# ---------------------------------------------------------------------------
# group statistics: shaping spread is NOT terminal disagreement
# ---------------------------------------------------------------------------

def test_shaping_spread_alone_is_not_terminal_disagreement():
    group = [_rec(success=False, milestone=m / 4.0) for m in range(4)] * 2
    stats = variance.group_stats(group, CFG)
    assert stats["nonzero_reward_sd"] is True     # rewards differ
    assert stats["terminal_disagreement"] is False  # but nobody succeeded
    assert stats["mean_success"] == 0.0
    assert stats["milestone_sd"] > 0.0


def test_a_mixed_group_disagrees_terminally():
    group = [_rec(success=True)] * 3 + [_rec(success=False, milestone=0.5)] * 5
    stats = variance.group_stats(group, CFG)
    assert stats["nonzero_reward_sd"] and stats["terminal_disagreement"]
    assert stats["n_success"] == 3
    assert stats["mean_success"] == pytest.approx(3 / 8)


def test_a_saturated_group_has_no_variance_at_all():
    stats = variance.group_stats([_rec(success=True)] * 8, CFG)
    assert stats["reward_sd"] == 0.0
    assert stats["nonzero_reward_sd"] is False
    assert stats["terminal_disagreement"] is False


# ---------------------------------------------------------------------------
# the four eligibility gates, computed from the summary alone
# ---------------------------------------------------------------------------

def _synthetic_rows(n_groups: int, n_mixed: int, *, family="typed_relay",
                    horizon=4, clipped=0, generations=8) -> list:
    rows = []
    for g in range(n_groups):
        mixed = g < n_mixed
        for i in range(generations):
            success = mixed and i < generations // 2
            rows.append(_rec(task_id=f"{family}-{horizon}-{g:03d}",
                             family=family, horizon=horizon,
                             success=success,
                             milestone=1.0 if success else 0.5,
                             truncated=(g * generations + i) < clipped))
    return rows


def test_gate_report_uses_only_the_summary_and_the_config():
    gates = CFG["variance_probe"]["gates"]
    rows = _synthetic_rows(10, 10)
    report = variance.gate_report(variance.summarize(rows, CFG), CFG)
    checks = report["overall"]["checks"]
    assert checks["nonzero_reward_sd"]["threshold"] == gates["min_nonzero_reward_sd_frac"]
    assert checks["terminal_disagreement"]["threshold"] == gates["min_terminal_disagreement_frac"]
    assert checks["mean_success"]["range"] == gates["mean_success_range"]
    assert checks["clip_frac"]["threshold"] == gates["max_clip_frac"]
    assert set(checks) == {"nonzero_reward_sd", "terminal_disagreement",
                           "mean_success", "clip_frac", "no_reward_leakage"}
    assert report["overall"]["eligible"] is True
    assert report["grpo_enabled"] is True


def test_a_zero_variance_family_is_skipped_not_run_ceremonially():
    rows = _synthetic_rows(10, 0)            # every group all-failure
    report = variance.gate_report(variance.summarize(rows, CFG), CFG)
    checks = report["overall"]["checks"]
    assert checks["terminal_disagreement"]["value"] == 0.0
    assert checks["mean_success"]["ok"] is False   # 0.0 is below the 0.15 floor
    assert report["overall"]["eligible"] is False
    assert report["eligible_cells"] == []
    assert report["skipped_cells"] == ["typed_relay-h4"]
    assert report["grpo_enabled"] is False


def test_shaping_only_variance_does_not_open_the_gate():
    """60% reward-SD but no terminal disagreement must still be ineligible."""
    rows = []
    for g in range(10):
        for i in range(8):
            rows.append(_rec(task_id=f"g{g:02d}", success=False,
                             milestone=(i % 4) / 4.0 if g < 7 else 0.5))
    summary = variance.summarize(rows, CFG)
    assert summary["overall"]["nonzero_reward_sd_frac"] == pytest.approx(0.7)
    assert summary["overall"]["terminal_disagreement_frac"] == 0.0
    assert summary["overall"]["shaping_only_frac"] == pytest.approx(0.7)
    report = variance.gate_report(summary, CFG)
    assert report["overall"]["checks"]["nonzero_reward_sd"]["ok"] is True
    assert report["overall"]["checks"]["terminal_disagreement"]["ok"] is False
    assert report["overall"]["eligible"] is False


def test_too_much_clipping_closes_the_gate():
    rows = _synthetic_rows(10, 10, clipped=8)     # 8 of 80 generations = 10%
    summary = variance.summarize(rows, CFG)
    assert summary["overall"]["clip_frac"] == pytest.approx(0.1)
    report = variance.gate_report(summary, CFG)
    assert report["overall"]["checks"]["clip_frac"]["ok"] is False
    assert report["overall"]["eligible"] is False


def test_a_replay_or_consistency_failure_vetoes_eligibility():
    rows = _synthetic_rows(10, 10)
    rows[0]["replay_ok"] = False
    rows[1]["verdict"] = dict(rows[1]["verdict"], consistent=False)
    report = variance.gate_report(variance.summarize(rows, CFG), CFG)
    leak = report["overall"]["checks"]["no_reward_leakage"]
    assert leak["replay_failures"] == 1 and leak["inconsistent_groups"] == 1
    assert leak["ok"] is False
    assert report["overall"]["eligible"] is False


def test_summary_reports_per_family_and_per_cell():
    rows = (_synthetic_rows(5, 5, family="typed_relay", horizon=4)
            + _synthetic_rows(5, 0, family="lookup_chain", horizon=8))
    summary = variance.summarize(rows, CFG)
    assert set(summary["per_family"]) == {"typed_relay", "lookup_chain"}
    assert set(summary["per_cell"]) == {"typed_relay-h4", "lookup_chain-h8"}
    report = variance.gate_report(summary, CFG)
    assert report["eligible_cells"] == ["typed_relay-h4"]
    assert report["skipped_cells"] == ["lookup_chain-h8"]


# ---------------------------------------------------------------------------
# probe geometry: 48 disjoint groups per family x 8 generations
# ---------------------------------------------------------------------------

def test_probe_geometry_matches_the_council_design():
    vp = CFG["variance_probe"]
    cells_per_family = len([1 for f, _h in CELLS if f == FAMILIES[0]])
    assert cells_per_family == 4
    assert vp["groups_per_cell"] * cells_per_family == vp["groups_per_family"] == 48
    assert vp["generations_per_group"] == 8


def test_probe_and_grpo_pools_are_disjoint_windows_of_one_split():
    bundles = [build_task(SUITE, SEED, "grpo_train", family, horizon, i, None)
               for family, horizon in CELLS for i in range(20)]
    vp, grpo = CFG["variance_probe"], CFG["grpo"]
    probe = cell_slice(bundles, vp["groups_per_cell"])
    pool = cell_slice(bundles, 8, offset=grpo["pool_offset_per_cell"])
    assert grpo["pool_offset_per_cell"] == vp["groups_per_cell"]
    probe_ids = {b.spec.task_id for b in probe}
    pool_ids = {b.spec.task_id for b in pool}
    assert probe_ids and pool_ids
    assert not (probe_ids & pool_ids)
    assert len(group_by_cell(probe)) == len(CELLS)
    for _cell, block in group_by_cell(probe).items():
        assert len(block) == vp["groups_per_cell"]


def test_cell_slice_refuses_to_shrink_a_sample_silently():
    bundles = [build_task(SUITE, SEED, "grpo_train", "lookup_chain", 2, i, None)
               for i in range(3)]
    with pytest.raises(ValueError, match="cannot take"):
        cell_slice(bundles, 4)


# ---------------------------------------------------------------------------
# end to end on the real engine: a group is eight runs of ONE committed task
# ---------------------------------------------------------------------------

def test_every_generation_of_a_group_faces_the_identical_task():
    bundle = build_task(SUITE, SEED, "grpo_train", "typed_relay", 4,
                        0, [("transient", False)])
    records = run_engine([bundle], cfg=CFG, generations=8)
    assert len(records) == 8
    assert {r["task_id"] for r in records} == {bundle.spec.task_id}
    # identical task => identical observation digests and oracle progress
    assert len({r["parity"]["episode"] for r in records}) == 1
    assert len({r["milestone_fraction"] for r in records}) == 1
    stats = variance.group_stats(records, CFG)
    assert stats["n"] == 8
    assert stats["reward_sd"] == 0.0
    assert stats["terminal_disagreement"] is False
    assert stats["mean_success"] == 1.0


def test_a_group_that_genuinely_disagrees_is_detected_end_to_end():
    bundle = build_task(SUITE, SEED, "grpo_train", "lookup_chain", 4, 1, None)
    good = run_engine([bundle], cfg=CFG, generations=4)
    bad = run_engine([bundle], cfg=CFG, generations=4,
                     policy=OraclePolicy([bundle], break_at=2))
    group = good + bad
    stats = variance.group_stats(group, CFG)
    assert stats["terminal_disagreement"] is True
    assert stats["nonzero_reward_sd"] is True
    assert stats["mean_success"] == pytest.approx(0.5)
    summary = variance.summarize(group, CFG)
    assert summary["overall"]["mean_success"] == pytest.approx(0.5)
    report = variance.gate_report(summary, CFG)
    assert report["overall"]["checks"]["mean_success"]["ok"] is True


def test_probe_rows_keep_every_field_the_gates_need():
    bundle = build_task(SUITE, SEED, "grpo_train", "fulfillment", 8, 2, None)
    records = run_engine([bundle], cfg=CFG, generations=2)
    for rec in records:
        rec["replay_ok"] = True
        rec["replay_reason"] = ""
    slim = [variance._slim(r) for r in records]
    summary = variance.summarize(slim, CFG)
    report = variance.gate_report(summary, CFG)
    assert report["overall"]["n_groups"] == 1
    assert summary["overall"]["n_generations"] == 2
