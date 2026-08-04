"""The analyzer must separate 'model is weak' from 'harness is broken'.

Three kinds of test:
  * statistics -- the formulas do what their names claim, including edges;
  * discrimination -- each sanity check fires on a synthetic version of a bug
    that actually happened, and stays SILENT on a genuinely weak-but-honest
    model. A weak model is a result; only a broken harness is a defect;
  * retrodiction -- the checks are quiet on the real runs already on disk.
"""

from __future__ import annotations

import pathlib

import pytest

from agentlab.analyze import (
    _dedupe,
    mcnemar,
    mde,
    paired_compare,
    sanity_checks,
    two_prop_test,
    wilson,
)


class TestWilson:
    def test_matches_known_value(self):
        p, lo, hi = wilson(40, 50)
        assert p == pytest.approx(0.8)
        assert lo == pytest.approx(0.6699, abs=2e-3)
        assert hi == pytest.approx(0.8884, abs=2e-3)

    def test_edges_stay_in_unit_interval(self):
        # (10,10) and (50,50) previously returned hi one ulp BELOW p.
        for k, n in ((0, 20), (20, 20), (1, 1), (0, 1), (10, 10), (50, 50), (0, 5)):
            p, lo, hi = wilson(k, n)
            assert 0.0 <= lo <= p <= hi <= 1.0
        assert wilson(10, 10)[2] == 1.0
        assert wilson(0, 5)[1] == 0.0

    def test_zero_n_is_uninformative_not_a_crash(self):
        assert wilson(0, 0) == (0.0, 0.0, 1.0)

    def test_interval_narrows_with_n(self):
        _, lo1, hi1 = wilson(8, 10)
        _, lo2, hi2 = wilson(160, 200)
        assert (hi2 - lo2) < (hi1 - lo1)


class TestTwoProp:
    def test_no_difference_gives_p_one(self):
        z, p = two_prop_test(40, 50, 160, 200)
        assert z == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0)

    def test_huge_difference_is_significant(self):
        z, p = two_prop_test(1, 20, 40, 50)
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


class TestMcNemar:
    def test_no_discordance_is_no_evidence(self):
        assert mcnemar(0, 0) == (0.0, 1.0)

    def test_resolves_what_the_unpaired_test_cannot(self):
        # The regime this experiment actually sits in: n=200 per arm, base
        # 160/200 (0.800) vs 170/200 (0.850), with the improvement concentrated
        # in 12 discordant pairs (b=1, c=11). Paired resolves it at p=0.006;
        # the unpaired test sees nothing at p=0.19 and would report "no
        # significant difference" for a real +5pp gain.
        _, p_paired = mcnemar(1, 11)
        _, p_unpaired = two_prop_test(170, 200, 160, 200)
        assert p_paired < 0.05 <= p_unpaired, (p_paired, p_unpaired)

    def test_exact_branch_is_more_conservative_than_the_normal_approx(self):
        # b=1,c=7: normal approx gives ~0.034, the exact binomial ~0.070. The
        # exact one is used below n=25 precisely because it does not overstate.
        import math

        _, p_exact = mcnemar(1, 7)
        z_norm = (7 - 1) / math.sqrt(8)
        p_norm = math.erfc(abs(z_norm) / math.sqrt(2))
        assert p_exact > p_norm

    def test_direction_follows_c_minus_b(self):
        assert mcnemar(1, 9)[0] > 0   # second condition better
        assert mcnemar(9, 1)[0] < 0

    def test_exact_and_normal_branches_agree_in_sign(self):
        assert mcnemar(2, 20)[0] > 0 and mcnemar(5, 40)[0] > 0


def _row(n, episodes, accuracy=None, tool_use=None, dupes=0):
    ok_k = sum(1 for e in episodes if e.get("ok"))
    acc = accuracy if accuracy is not None else (ok_k / len(episodes) if episodes else 0)
    used = sum(1 for e in episodes if e.get("n_calls", 0) > 0)
    tu = tool_use if tool_use is not None else (used / len(episodes) if episodes else 0)
    return {
        "n": n, "acc_k": round(acc * n), "dupes": dupes,
        "summary": {"n": n, "accuracy": acc, "tool_use_rate": tu},
        "_episodes": episodes,
    }


def _ep(i, ok=True, final=None, n_calls=2, predicted=None, expected=None, gt=None):
    """A self-consistent episode for problem i, unless overridden."""
    gt = gt if gt is not None else str(i)
    val = float(gt)
    if final is None:
        final = f"the answer is \\boxed{{{gt if ok else int(val) + 1}}}"
    pred = predicted if predicted is not None else (val if ok else val + 1)
    exp = expected if expected is not None else val
    return {"index": i, "ok": ok, "final": final, "n_calls": n_calls,
            "ground_truth": gt, "rewards": {"predicted": pred, "expected": exp}}


def _codes(issues, level=None):
    return [c for lv, c, _ in issues if level is None or lv == level]


class TestDiscrimination:
    def test_healthy_run_is_clean(self):
        eps = [_ep(i) for i in range(30)]
        eps += [_ep(i, ok=False, final=f"went wrong at step {i}") for i in range(30, 40)]
        assert sanity_checks(_row(40, eps)) == []

    def test_weak_model_is_NOT_flagged(self):
        # 5% accuracy, working parsing and scoring: a result, not a defect.
        eps = [_ep(0)] + [_ep(i, ok=False, final=f"wandering text {i}", n_calls=50)
                          for i in range(1, 20)]
        assert _codes(sanity_checks(_row(20, eps)), "BUG") == []

    # --- S0/S1: trace integrity -------------------------------------------
    def test_S0_duplicate_indices_warn_and_dedupe_keeps_last(self):
        eps = [_ep(0, n_calls=1), _ep(0, n_calls=3)] + [_ep(i) for i in range(1, 20)]
        deduped, dupes = _dedupe(eps)
        assert dupes == 1 and len(deduped) == 20
        assert deduped[0]["n_calls"] == 3, "the rerun supersedes the killed run"
        issues = sanity_checks(_row(20, deduped, dupes=dupes))
        assert "S0" in _codes(issues) and _codes(issues, "BUG") == []

    def test_S1_dropped_episodes(self):
        eps = [_ep(i) for i in range(10)]
        assert "S1" in _codes(sanity_checks(_row(50, eps, accuracy=1.0)), "BUG")

    def test_S1_off_by_one_now_fires(self):
        # The old +/-1 slack admitted exactly the duplicate-append artifact.
        eps = [_ep(i) for i in range(49)]
        assert "S1" in _codes(sanity_checks(_row(50, eps, accuracy=1.0)), "BUG")

    # --- S2: scoring paths must agree -------------------------------------
    def test_S2_scoring_paths_disagree(self):
        eps = [_ep(i) for i in range(20)]
        assert "S2" in _codes(sanity_checks(_row(20, eps, accuracy=0.5)), "BUG")

    # --- S3: scorer-blind, at ANY accuracy --------------------------------
    def test_S3_fires_when_boxed_equals_gt_but_scored_wrong(self):
        eps = [_ep(i) for i in range(14)]
        eps += [_ep(i, ok=False, final=f"the answer is \\boxed{{{i}}}",
                    predicted=float(i), expected=float(i)) for i in range(14, 20)]
        issues = sanity_checks(_row(20, eps))
        assert "S3" in _codes(issues, "BUG") or "S5" in _codes(issues, "BUG")

    def test_S3_honest_zero_scorer_is_not_a_bug(self):
        # Always answers 42, always wrong: catastrophic RESULT, no harness bug.
        eps = [_ep(i, ok=False, final=r"the answer is \boxed{42}", gt=str(1000 + i),
                   predicted=42.0, expected=float(1000 + i)) for i in range(20)]
        issues = sanity_checks(_row(20, eps, accuracy=0.0))
        assert _codes(issues, "BUG") == []
        assert "S3" in _codes(issues, "WARN")

    def test_S3_no_boxes_at_zero_warns_only(self):
        eps = [_ep(i, ok=False, final=f"rambling {i}") for i in range(20)]
        issues = sanity_checks(_row(20, eps, accuracy=0.0))
        assert "S3" in _codes(issues, "WARN") and "S3" not in _codes(issues, "BUG")

    # --- S4: parser-blind needs corroboration -----------------------------
    def test_S4_fires_only_with_visible_tool_syntax(self):
        eps = [_ep(i, ok=False, n_calls=0,
                   final=f"<tool_call>\n<function=calculator>\n{i}") for i in range(20)]
        assert "S4" in _codes(sanity_checks(_row(20, eps, tool_use=0.0)), "BUG")

    def test_S4_honest_no_tool_collapse_is_a_warn(self):
        eps = [_ep(i, ok=False, n_calls=0, final=f"just prose {i}") for i in range(20)]
        issues = sanity_checks(_row(20, eps, tool_use=0.0))
        assert "S4" in _codes(issues, "WARN") and "S4" not in _codes(issues, "BUG")

    # --- S5: impossible states, both directions ---------------------------
    def test_S5_ok_with_empty_final(self):
        eps = [_ep(0, final="   ")] + [_ep(i) for i in range(1, 12)]
        assert "S5" in _codes(sanity_checks(_row(12, eps)), "BUG")

    def test_S5_ok_but_numbers_disagree(self):
        eps = [_ep(0, predicted=5.0, expected=7.0)] + [_ep(i) for i in range(1, 12)]
        assert "S5" in _codes(sanity_checks(_row(12, eps)), "BUG")

    def test_S5_inverse_not_ok_but_numbers_agree(self):
        eps = [_ep(0, ok=False, final="text with no box", predicted=5.0, expected=5.0)]
        eps += [_ep(i) for i in range(1, 12)]
        assert "S5" in _codes(sanity_checks(_row(12, eps)), "BUG")

    # --- S6/S7 ------------------------------------------------------------
    def test_S6_mass_duplicate_finals_warns(self):
        # WARN not BUG: a collapsed policy and an indexing fault look identical
        # in the trace, so this flags for inspection rather than vetoing.
        eps = [_ep(i, final=r"identical \boxed{5}") for i in range(15)]
        eps += [_ep(i, final=f"unique {i} \\boxed{{{i}}}") for i in range(15, 20)]
        assert "S6" in _codes(sanity_checks(_row(20, eps)), "WARN")

    def test_S7_summary_trace_tool_use_mismatch(self):
        eps = [_ep(i, n_calls=2) for i in range(20)]
        assert "S7" in _codes(sanity_checks(_row(20, eps, tool_use=0.5)), "BUG")


class TestPairedCompare:
    def test_joins_on_index_and_counts_discordance(self):
        a = [_ep(i, ok=(i < 8)) for i in range(20)]
        b = [_ep(i, ok=(i < 14)) for i in range(20)]
        out = paired_compare(a, b)
        assert out["n_pairs"] == 20 and out["b"] == 0 and out["c"] == 6
        assert out["z"] > 0

    def test_returns_none_when_too_little_overlap(self):
        assert paired_compare([_ep(i) for i in range(5)], [_ep(i) for i in range(5)]) is None


class TestReportSemantics:
    """A skipped gate is not a failed gate, and a failed gate is not success."""

    def _write(self, tmp_path, tag, n, acc, episodes=None, tool_use=1.0):
        import json

        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / f"eval-{tag}.json").write_text(json.dumps({
            "n": n, "accuracy": acc, "tool_use_rate": tool_use,
            "tool_error_rate": 0.0, "mean_turns": 2.5, "seconds": 1.0, "tag": tag,
        }))
        if episodes is not None:
            (tmp_path / "traces").mkdir(exist_ok=True)
            with (tmp_path / "traces" / f"trace-{tag}.jsonl").open("w") as fh:
                for e in episodes:
                    fh.write(json.dumps({"kind": "episode", **e}) + "\n")

    def test_failed_gate_never_prints_the_success_line(self, tmp_path):
        from agentlab.analyze import report

        base_eps = [_ep(i, ok=(i < 40)) for i in range(50)]
        # rssft below the 0.800 gate but behaviourally healthy
        rs_eps = [_ep(i, ok=(i < 37), n_calls=3) for i in range(50)]
        self._write(tmp_path, "base", 50, 0.80, base_eps)
        self._write(tmp_path, "rssft", 50, 0.74, rs_eps)
        out = report(["base", "rssft"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "FAIL  G1" in out
        assert "restored it" not in out, "success narrative printed despite a failed gate"

    def test_missing_trace_is_skipped_not_failed(self, tmp_path):
        from agentlab.analyze import report

        self._write(tmp_path, "rssft", 200, 0.84, episodes=None)
        out = report(["rssft"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "SKIP" in out
        assert "0 failed" in out
        assert "INCOMPLETE DATA" in out
        assert "NOT supported" not in out, "a missing file was reported as a model result"

    def test_all_gates_passing_prints_success(self, tmp_path):
        from agentlab.analyze import report

        base_eps = [_ep(i, ok=(i < 40), n_calls=3) for i in range(50)]
        base_eps = [{**e, "final": e["final"] if e["ok"] else "no box"} for e in base_eps]
        rs_eps = [_ep(i, ok=(i < 44), n_calls=3) for i in range(50)]
        self._write(tmp_path, "base", 50, 0.80, base_eps)
        self._write(tmp_path, "rssft", 50, 0.88, rs_eps)
        out = report(["base", "rssft"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "0 failed" in out and "0 skipped" in out
        assert "restored it" in out


class TestRetrodictionOnRealData:
    """The checks must be quiet on the real runs already on disk."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def _load(self, tag):
        from agentlab.analyze import load_tag

        # chain-first: the production verdict's precedence. eval-base.json is
        # the n=200 rerun, and out/chain holds its matching trace.
        row = load_tag(tag, self.ROOT / "out",
                       [self.ROOT / "out/chain", self.ROOT / "out/comparison"])
        if row is None or not row.get("_episodes"):
            pytest.skip(f"no local data for {tag}")
        return row

    def test_real_base_run_has_no_harness_bug(self):
        assert _codes(sanity_checks(self._load("base")), "BUG") == []

    def test_real_broken_sft_run_is_weak_not_buggy(self):
        # The 0.050 checkpoint: catastrophically bad AND honestly measured.
        assert _codes(sanity_checks(self._load("sft")), "BUG") == []


class TestNotationRescoring:
    """A right answer in the wrong notation must not score wrong."""

    def test_normalizer_accepts_model_notation(self):
        from agentlab.chat import numeric_answer

        for s, want in ((r"24\%", 24.0), ("24%", 24.0), (r"\$50", 50.0),
                        ("1,234", 1234.0), (r"\text{42}", 42.0), ("abc", None)):
            assert numeric_answer(s) == want, s

    def test_real_base_episode_38_is_corrected(self):
        # The episode the referee found: gt='24', boxed='24\%', scored wrong.
        from agentlab.analyze import load_tag

        root = pathlib.Path(__file__).resolve().parents[1]
        row = load_tag("base", root / "out", [root / "out/comparison", root / "out/chain"])
        if row is None or not row.get("_episodes"):
            pytest.skip("no local base data")
        ep = {e["index"]: e for e in row["_episodes"]}.get(38)
        if ep is None or "24" not in str(ep.get("final", "")):
            pytest.skip("trace does not contain the episode")
        assert ep.get("ok") is False and ep["_ok_rescored"] is True
        assert row["corrections"] >= 1
        assert row["rescored_k"] == sum(1 for e in row["_episodes"] if e["_ok_rescored"])

    def test_paired_compare_uses_rescored_flags(self):
        eps_a = [dict(_ep(i), _ok_rescored=(i < 10)) for i in range(20)]
        eps_b = [dict(_ep(i), _ok_rescored=(i < 15)) for i in range(20)]
        out = paired_compare(eps_a, eps_b)
        assert out["b"] == 0 and out["c"] == 5

    def test_tool_compliant_accuracy_counts_only_tool_using_successes(self):
        from agentlab.analyze import load_tag  # noqa: F401  (shape covered above)

        eps = [_ep(0, n_calls=0), _ep(1, n_calls=2), _ep(2, ok=False, n_calls=2)]
        for e in eps:
            e["_ok_rescored"] = bool(e["ok"])
        tools_ok = sum(1 for e in eps if e["_ok_rescored"] and e["n_calls"] > 0)
        assert tools_ok == 1  # the no-tool success does not count


class TestG5EvidenceRequirement:
    """G5 must never evaluate off a self-reported summary alone.

    The dress rehearsal produced a full false-success from eval-rsgrpo.json
    with no trace, and reproduced the silent variant (rsgrpo absent, gate never
    mentioned, '0 skipped') against the real output directory.
    """

    def _write(self, tmp_path, tag, n, acc, episodes=None):
        import json

        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / f"eval-{tag}.json").write_text(json.dumps({
            "n": n, "accuracy": acc, "tool_use_rate": 1.0,
            "tool_error_rate": 0.0, "mean_turns": 2.0, "seconds": 1.0, "tag": tag,
        }))
        if episodes is not None:
            (tmp_path / "traces").mkdir(exist_ok=True)
            with (tmp_path / "traces" / f"trace-{tag}.jsonl").open("w") as fh:
                for e in episodes:
                    fh.write(json.dumps({"kind": "episode", **e}) + "\n")

    def _healthy(self, n, k_ok, nobox_failures=True):
        eps = [_ep(i, n_calls=2) for i in range(k_ok)]
        # Failures must look like the real base failure mode (truncated, no box)
        # or G4 (rssft no-box < base no-box) can never pass in a fixture.
        fail_final = "ran out of budget mid-derivation" if nobox_failures else None
        eps += [_ep(i, ok=False, n_calls=2, final=fail_final) for i in range(k_ok, n)]
        return eps

    def test_summary_without_trace_is_a_skip_not_a_pass(self, tmp_path):
        from agentlab.analyze import report

        self._write(tmp_path, "base", 50, 0.80, self._healthy(50, 40))
        self._write(tmp_path, "rssft", 50, 0.88, self._healthy(50, 44, nobox_failures=False))
        self._write(tmp_path, "rsgrpo", 50, 0.93, episodes=None)  # json only
        out = report(["base", "rssft", "rsgrpo"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "SKIP  G5" in out and "1 skipped" in out
        assert "INCOMPLETE DATA" in out
        assert "restored it" not in out

    def test_missing_rsgrpo_entirely_is_still_a_visible_skip(self, tmp_path):
        from agentlab.analyze import report

        self._write(tmp_path, "base", 50, 0.80, self._healthy(50, 40))
        self._write(tmp_path, "rssft", 50, 0.88, self._healthy(50, 44, nobox_failures=False))
        out = report(["base", "rssft", "rsgrpo"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "SKIP  G5" in out and "no eval json" in out
        assert "0 skipped" not in out
        assert "restored it" not in out

    def test_rsgrpo_not_requested_means_no_g5_bookkeeping(self, tmp_path):
        from agentlab.analyze import report

        self._write(tmp_path, "base", 50, 0.80, self._healthy(50, 40))
        self._write(tmp_path, "rssft", 50, 0.88, self._healthy(50, 44, nobox_failures=False))
        out = report(["base", "rssft"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "G5" not in out and "0 skipped" in out
        assert "restored it" in out  # 4-gate experiment may still succeed

    def test_g5_only_failure_is_a_grpo_null_not_an_rssft_indictment(self, tmp_path):
        from agentlab.analyze import report

        self._write(tmp_path, "base", 50, 0.80, self._healthy(50, 40))
        self._write(tmp_path, "rssft", 50, 0.88, self._healthy(50, 44, nobox_failures=False))
        self._write(tmp_path, "rsgrpo", 50, 0.84, self._healthy(50, 42, nobox_failures=False))  # below rssft
        out = report(["base", "rssft", "rsgrpo"], str(tmp_path / "out"), [str(tmp_path / "traces")])
        assert "FAIL  G5" in out
        assert "restoration result stands" in out
        assert "NOT supported" not in out
        assert "restored it" not in out

    def test_stale_trace_pairing_is_caught_not_silent(self):
        # Pairing the n=200 summary with the old n=50 trace (comparison-first
        # ordering) MUST trip S1. This mispairing happened for real when the
        # base rerun overwrote eval-base.json; the check is the defence.
        from agentlab.analyze import load_tag, sanity_checks

        row = load_tag("base", self.ROOT / "out",
                       [self.ROOT / "out/comparison", self.ROOT / "out/chain"])
        if row is None or not row.get("_episodes") or row["n"] == row.get("trace_n"):
            pytest.skip("stale pairing not present on this checkout")
        assert "S1" in [c for _, c, _ in sanity_checks(row)]

