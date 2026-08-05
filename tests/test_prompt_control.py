"""The elicitation control is the PREREGISTERED prompt set, and only that.

Round 1 of this lab lost its headline to a one-sentence prompt, so the control is
the part of the design most worth protecting from drift. These tests pin:

  * the candidate set is the eight hash-committed files in prompts/agentic, and
    an edited file stops the tournament instead of silently changing the control;
  * the neutral arm's bytes come from the preregistered neutral default, with no
    second copy anywhere in the project;
  * the three tournament axes are the three CLAIM axes, drawn from dev data only;
  * the winner rule is the preregistered one, ties to the shorter file then the
    lower index;
  * a dev split too small for the preregistered per-axis sizes is reported, not
    silently sampled down.
"""

from __future__ import annotations

import json

import pytest

from agentlab import prompt_control as pc
from agentlab.suite.configio import ROOT, load_config
from agentlab.suite.generate import build_task
from agentlab.suite.schema import CELLS, file_sha256

CFG = load_config()
SUITE = "agentlab-suite-v1"
DEV_SEED = 0xA61E0004


def _dev_bundles(per_cell: int = 4) -> list:
    """A stand-in dev split: every cell, one assigned fault each (dev design)."""
    return [build_task(SUITE, DEV_SEED, "dev", family, horizon, i,
                       [("transient", False)])
            for family, horizon in CELLS for i in range(per_cell)]


# ---------------------------------------------------------------------------
# the frozen candidate set
# ---------------------------------------------------------------------------

def test_candidates_are_the_eight_preregistered_files():
    cands = pc.verify_frozen()
    assert len(cands) == 8
    prereg = pc.preregistration()["prompt_candidates"]
    assert {c["id"] for c in cands} == set(prereg["sha256"])
    assert sum(c["neutral"] for c in cands) == 1
    for cand in cands:
        assert cand["path"].exists()
        assert file_sha256(str(cand["path"])) == cand["committed_sha256"]


def test_an_edited_candidate_stops_the_tournament():
    prereg = pc.preregistration()
    tampered = json.loads(json.dumps(prereg))
    name = sorted(tampered["prompt_candidates"]["sha256"])[3]
    tampered["prompt_candidates"]["sha256"][name] = "00" * 32
    with pytest.raises(SystemExit, match="do not match their preregistered"):
        pc.verify_frozen(tampered)


def test_an_added_or_removed_candidate_stops_the_tournament():
    prereg = pc.preregistration()
    tampered = json.loads(json.dumps(prereg))
    tampered["prompt_candidates"]["sha256"].pop(
        sorted(tampered["prompt_candidates"]["sha256"])[0])
    with pytest.raises(SystemExit, match="candidate set is frozen"):
        pc.verify_frozen(tampered)


def test_the_neutral_prompt_has_exactly_one_source_of_bytes():
    prereg = pc.preregistration()["prompt_candidates"]
    path = ROOT / prereg["directory"] / prereg["neutral_default"]
    assert pc.CANONICAL_SYSTEM == path.read_text(encoding="utf-8").strip()
    assert pc.neutral_prompt() == pc.CANONICAL_SYSTEM
    # and the rollout engine serves exactly those bytes for the neutral variant
    from agentlab.multidistill import RolloutEngine

    engine = RolloutEngine(CFG, lambda m, s: m, lambda p: [])
    assert engine.system_prompt("canonical") == pc.CANONICAL_SYSTEM
    engine.frozen = "WINNER PROMPT"
    assert engine.system_prompt("frozen") == "WINNER PROMPT"
    assert engine.system_prompt("canonical") == pc.CANONICAL_SYSTEM


def test_no_second_candidates_file_is_referenced():
    assert "candidates_file" not in CFG["prompt_control"]
    assert not (ROOT / "configs" / "prompt_candidates.json").exists()


def test_an_unfrozen_control_refuses_production_rollouts():
    from agentlab.multidistill import load_frozen_prompt

    if pc.frozen_file(CFG).exists():
        pytest.skip("a tournament winner is already frozen in this tree")
    with pytest.raises(SystemExit, match="must be frozen BEFORE"):
        load_frozen_prompt(CFG)


# ---------------------------------------------------------------------------
# the three claim axes
# ---------------------------------------------------------------------------

def test_axes_are_the_three_claim_axes():
    assert pc.AXES == ("recovery", "orchestration", "h8")
    bundles = _dev_bundles()

    recovery = pc.axis_pool(bundles, "recovery")
    assert recovery and all(b.spec.faults for b in recovery)

    orchestration = pc.axis_pool(bundles, "orchestration")
    assert orchestration
    for b in orchestration:
        assert (b.spec.family, b.spec.horizon) == ("typed_relay", 4)
        assert {"kb_lookup", "unit_convert", "calculator"} <= {n.tool for n in b.nodes}
        assert b.spec.faults == []          # the paired CLEAN arm

    h8 = pc.axis_pool(bundles, "h8")
    assert h8
    for b in h8:
        assert b.spec.horizon == 8
        assert b.spec.family in ("lookup_chain", "typed_relay")
        assert b.spec.faults == []


def test_clean_axis_arms_keep_the_task_but_drop_the_fault():
    bundles = _dev_bundles(per_cell=1)
    faulted = next(b for b in bundles
                   if (b.spec.family, b.spec.horizon) == ("typed_relay", 4))
    clean = pc.axis_pool(bundles, "orchestration")[0]
    assert clean.spec.task_id == faulted.spec.task_id
    assert clean.spec.answer == faulted.spec.answer
    assert clean.kb == faulted.kb
    assert clean.spec.max_decisions < faulted.spec.max_decisions


def test_axis_samples_are_balanced_and_disjoint_between_rounds():
    bundles = _dev_bundles(per_cell=4)
    r1 = pc.axis_bundles(bundles, "recovery", 12, offset=0)
    r2 = pc.axis_bundles(bundles, "recovery", 12, offset=2)
    assert len(r1) == len(r2) == 12
    assert not ({b.spec.task_id for b in r1} & {b.spec.task_id for b in r2})
    cells = {(b.spec.family, b.spec.horizon) for b in r1}
    assert len(cells) == len(CELLS)      # one per cell at n=12 over 12 cells


def test_an_undersized_axis_raises_instead_of_sampling_less():
    bundles = _dev_bundles(per_cell=1)
    with pytest.raises(ValueError, match="cannot take|can supply"):
        pc.axis_bundles(bundles, "orchestration", 100)


# ---------------------------------------------------------------------------
# the preregistered winner rule
# ---------------------------------------------------------------------------

def _rows(candidate: str, rates: dict, n: int = 10) -> list:
    out = []
    for axis, rate in rates.items():
        for i in range(n):
            out.append({"candidate": candidate, "axis": axis,
                        "success": i < round(rate * n)})
    return out


def test_combined_score_weights_the_three_axes_equally():
    rows = _rows("p1_minimal.txt", {"recovery": 0.0, "orchestration": 0.6, "h8": 0.9})
    rates = pc.axis_rates(rows)["p1_minimal.txt"]
    assert rates["combined"] == pytest.approx((0.0 + 0.6 + 0.9) / 3, abs=1e-4)


def test_the_winner_is_the_highest_combined_score():
    rows = (_rows("p1_minimal.txt", {"recovery": 0.2, "orchestration": 0.2, "h8": 0.2})
            + _rows("p4_error_repair.txt", {"recovery": 0.8, "orchestration": 0.5,
                                            "h8": 0.5}))
    verdict = pc.pick_winner(rows, CFG)
    assert verdict["winner"]["candidate"] == "p4_error_repair.txt"
    assert verdict["ranking"][0] == "p4_error_repair.txt"
    assert verdict["round2_candidates"] == ["p4_error_repair.txt", "p1_minimal.txt"]
    assert "best of eight" in verdict["honest_description"]


def test_ties_break_to_the_shorter_file_then_the_lower_index():
    cands = pc.candidates()
    sizes = {c["id"]: c["path"].stat().st_size for c in cands}
    a, b = sorted(sizes, key=lambda k: sizes[k])[:2]
    assert sizes[a] < sizes[b], "the fixture needs two differently sized prompts"
    rows = (_rows(a, {"recovery": 0.5, "orchestration": 0.5, "h8": 0.5})
            + _rows(b, {"recovery": 0.5, "orchestration": 0.5, "h8": 0.5}))
    verdict = pc.pick_winner(rows, CFG)
    assert verdict["winner"]["candidate"] == a
    # the cheaper control is the stronger baseline: a longer prompt must EARN it
    assert verdict["ranking"] == [a, b]


def test_the_winner_record_pins_the_file_hash():
    rows = _rows("p8_combined.txt", {"recovery": 0.9, "orchestration": 0.9, "h8": 0.9})
    verdict = pc.pick_winner(rows, CFG)
    cand = next(c for c in pc.candidates() if c["id"] == "p8_combined.txt")
    assert verdict["winner"]["sha256"] == cand["committed_sha256"]


def test_h8_feasibility_is_reported_against_the_preregistered_floor():
    floor = CFG["prompt_control"]["h8_feasibility_min"]
    low = pc.pick_winner(_rows("p1_minimal.txt", {"recovery": 0.5,
                                                  "orchestration": 0.5, "h8": 0.0}), CFG)
    assert low["h8"]["measured_only"] is True
    assert low["h8"]["feasibility_min"] == floor
    high = pc.pick_winner(_rows("p1_minimal.txt", {"recovery": 0.5,
                                                   "orchestration": 0.5, "h8": 0.5}), CFG)
    assert high["h8"]["measured_only"] is False


# ---------------------------------------------------------------------------
# the preregistered sample sizes
# ---------------------------------------------------------------------------

def test_the_preregistered_per_axis_sizes_are_the_configured_ones():
    pcfg = CFG["prompt_control"]
    assert (pcfg["round1_per_axis"], pcfg["round2_per_axis"]) == (100, 200)
    assert pcfg["n_candidates"] == len(pc.candidates()) == 8


def test_the_committed_dev_split_cannot_yet_supply_them():
    """A recorded, reproducible shortfall -- not a silently shrunken tournament.

    The dev split is 20 specs per cell, so the orchestration axis (typed_relay H4
    only) has 20 instances against a requirement of 300. The tournament stage must
    enlarge the dev split in configs/suite_v1.toml and regenerate, or amend; this
    test exists so that decision is made deliberately rather than discovered on
    the card.
    """
    bundles = _dev_bundles(per_cell=20)          # the committed dev per-cell size
    need = CFG["prompt_control"]["round1_per_axis"] + CFG["prompt_control"]["round2_per_axis"]
    assert need == 300
    assert len(pc.axis_pool(bundles, "orchestration")) == 20
    assert len(pc.axis_pool(bundles, "h8")) == 40
    assert len(pc.axis_pool(bundles, "recovery")) == 20 * len(CELLS)
    with pytest.raises(ValueError):
        pc.axis_bundles(bundles, "orchestration", need)
