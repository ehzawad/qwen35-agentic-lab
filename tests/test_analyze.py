"""The analyzer must separate 'model is weak' from 'harness is broken'.

Two kinds of test:
  * statistics -- the formulas do what their names claim, including edges;
  * retrodiction -- every sanity check fires on a synthetic version of a bug
    this session actually hit, and stays SILENT on real healthy data and on the
    real pathological-but-honestly-measured checkpoint. A weak model is a
    result; only a broken harness is a defect.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agentlab.analyze import mde, sanity_checks, two_prop_test, wilson


class TestWilson:
    def test_matches_known_value(self):
        # 40/50: canonical Wilson bounds ~ [0.670, 0.888]
        p, lo, hi = wilson(40, 50)
        assert p == pytest.approx(0.8)
        assert lo == pytest.approx(0.6699, abs=2e-3)
        assert hi == pytest.approx(0.8884, abs=2e-3)

    def test_edges_stay_in_unit_interval(self):
        for k, n in ((0, 20), (20, 20), (1, 1), (0, 1)):
            p, lo, hi = wilson(k, n)
            assert 0.0 <= lo <= p <= hi <= 1.0

    def test_zero_n_is_uninformative_not_a_crash(self):
        assert wilson(0, 0) == (0.0, 0.0, 1.0)

    def test_interval_narrows_with_n(self):
        _, lo1, hi1 = wilson(8, 10)
        _, lo2, hi2 = wilson(160, 200)
        assert (hi2 - lo2) < (hi1 - lo1)


class TestTwoProp:
    def test_no_difference_gives_p_one(self):
        z, p = two_prop_test(40, 50, 160, 200)  # both 0.8
        assert z == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0)

    def test_huge_difference_is_significant(self):
        z, p = two_prop_test(1, 20, 40, 50)  # 0.05 vs 0.80
        assert p < 1e-6 and z < 0

    def test_symmetry(self):
        z1, p1 = two_prop_test(30, 50, 20, 50)
        z2, p2 = two_prop_test(20, 50, 30, 50)
        assert z1 == pytest.approx(-z2)
        assert p1 == pytest.approx(p2)

    def test_degenerate_inputs_do_not_crash(self):
        assert two_prop_test(0, 0, 5, 10) == (0.0, 1.0)
        assert two_prop_test(0, 10, 0, 10) == (0.0, 1.0)

    def test_mde_shrinks_with_n(self):
        assert mde(200, 200, 0.8) < mde(50, 50, 0.8) < mde(20, 20, 0.8)


# ---------------------------------------------------------------------------
# retrodiction: bug signatures from this session, as fixtures
# ---------------------------------------------------------------------------

def _row(n, episodes, accuracy=None, tool_use=None):
    ok_k = sum(1 for e in episodes if e.get("ok"))
    acc = accuracy if accuracy is not None else (ok_k / len(episodes) if episodes else 0)
    used = sum(1 for e in episodes if e.get("n_calls", 0) > 0)
    tu = tool_use if tool_use is not None else (used / len(episodes) if episodes else 0)
    return {
        "n": n, "acc_k": round(acc * n),
        "summary": {"n": n, "accuracy": acc, "tool_use_rate": tu},
        "_episodes": episodes,
    }


def _ep(i, ok=True, final=r"the answer is \boxed{5}", n_calls=2, predicted=5.0, expected=5.0):
    return {"index": i, "ok": ok, "final": final, "n_calls": n_calls,
            "rewards": {"predicted": predicted, "expected": expected}}


def _codes(issues, level=None):
    return [c for lv, c, _ in issues if level is None or lv == level]


class TestSanitySignatures:
    def test_healthy_run_is_clean(self):
        # Finals must be distinct per problem, as they are in a real run --
        # identical finals across a run is itself one of the bug signatures.
        eps = [_ep(i, final=f"the answer is \\boxed{{{i}}}", predicted=float(i), expected=float(i))
               for i in range(30)]
        eps += [_ep(i, ok=False, final=f"went wrong at step {i}") for i in range(30, 40)]
        assert sanity_checks(_row(40, eps)) == []

    def test_weak_model_is_NOT_flagged(self):
        # 5% accuracy with working parsing/scoring: a result, not a defect.
        eps = [_ep(0)] + [_ep(i, ok=False, final=f"wandering text {i}", n_calls=50)
                          for i in range(1, 20)]
        assert _codes(sanity_checks(_row(20, eps)), "BUG") == []

    def test_S1_dropped_episodes(self):
        eps = [_ep(i) for i in range(10)]
        assert "S1" in _codes(sanity_checks(_row(50, eps, accuracy=1.0)), "BUG")

    def test_S2_scoring_paths_disagree(self):
        eps = [_ep(i) for i in range(20)]                       # trace says 100%
        assert "S2" in _codes(sanity_checks(_row(20, eps, accuracy=0.5)), "BUG")

    def test_S3_scorer_blind(self):
        # The GRPO-zero shape: boxes present, accuracy exactly 0.
        eps = [_ep(i, ok=False) for i in range(20)]             # finals DO contain boxes
        assert "S3" in _codes(sanity_checks(_row(20, eps, accuracy=0.0)), "BUG")

    def test_S3_warns_when_no_boxes_at_all(self):
        eps = [_ep(i, ok=False, final="rambling") for i in range(20)]
        issues = sanity_checks(_row(20, eps, accuracy=0.0))
        assert "S3" in _codes(issues, "WARN") and "S3" not in _codes(issues, "BUG")

    def test_S4_parser_blind(self):
        # The XML-vs-JSON shape: tools offered, zero calls observed.
        eps = [_ep(i, n_calls=0) for i in range(20)]
        assert "S4" in _codes(sanity_checks(_row(20, eps, tool_use=0.0)), "BUG")

    def test_S5_ok_with_empty_final(self):
        eps = [_ep(0, final="   ")] + [_ep(i) for i in range(1, 12)]
        assert "S5" in _codes(sanity_checks(_row(12, eps)), "BUG")

    def test_S5_ok_but_numbers_disagree(self):
        eps = [_ep(0, predicted=5.0, expected=7.0)] + [_ep(i) for i in range(1, 12)]
        assert "S5" in _codes(sanity_checks(_row(12, eps)), "BUG")

    def test_S6_mass_duplicate_finals(self):
        eps = [_ep(i, final=r"identical \boxed{5}") for i in range(15)] + \
              [_ep(i, final=f"unique {i} \\boxed{{5}}") for i in range(15, 20)]
        assert "S6" in _codes(sanity_checks(_row(20, eps)), "BUG")

    def test_S7_summary_trace_tool_use_mismatch(self):
        eps = [_ep(i, n_calls=2) for i in range(20)]            # trace: 100% use
        assert "S7" in _codes(sanity_checks(_row(20, eps, tool_use=0.5)), "BUG")


class TestRetrodictionOnRealData:
    """The checks must be quiet on the real runs already on disk."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def _load(self, tag):
        from agentlab.analyze import load_tag

        row = load_tag(tag, self.ROOT / "out", [self.ROOT / "out/comparison", self.ROOT / "out/chain"])
        if row is None or not row.get("_episodes"):
            pytest.skip(f"no local data for {tag}")
        return row

    def test_real_base_run_is_clean(self):
        assert _codes(sanity_checks(self._load("base")), "BUG") == []

    def test_real_broken_sft_run_is_weak_not_buggy(self):
        # The 0.050 checkpoint: catastrophically bad AND honestly measured.
        assert _codes(sanity_checks(self._load("sft")), "BUG") == []
