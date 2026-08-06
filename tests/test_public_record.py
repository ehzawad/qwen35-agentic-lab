"""The public record must not drift back into claims the study has outgrown.

These tests read files only -- no git, no GPU, no network, no live repository
state. That is deliberate: the one test this suite currently fails
(`test_lock_prompt_refuses_before_P_exists`) fails precisely because it asserts
against the live repository, so a guard written to protect the README must not
repeat that mistake.

What is pinned here is the SHAPE of the honest status block, not its numbers. The
test count and the GPU-hour figure move every time the suite or the ledger moves;
asserting them would make this file a maintenance tax that gets deleted. What
cannot move without a lie is the set of retired claims, and the reporting rules
for the saturated secondary endpoints.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
INTERPRETATION = ROOT / "docs" / "INTERPRETATION.md"
DEFERRED = ROOT / "docs" / "DEFERRED_REPAIRS.md"

# Every one of these was true of some earlier commit and is false now. A README
# that says any of them again is a materially false account of study state.
RETIRED_README_CLAIMS = (
    "preregistration NOT yet finalized",
    "zero GPU hours spent",
    "645 passing tests",
    "S18 delivers test-set commitment",
    "Until the finalization marker exists the preregistration is a **draft**",
)


def _read(p: pathlib.Path) -> str:
    assert p.exists(), f"{p.relative_to(ROOT)} is part of the public record"
    return p.read_text(encoding="utf-8")


def _flat(p: pathlib.Path) -> str:
    """The file with markdown blockquote markers dropped and whitespace collapsed.

    Prose in these files is hard-wrapped at 88 columns and the verbatim required
    wording is a `>` blockquote, so a sentence-level assertion against the raw
    bytes breaks the moment a word moves across a line boundary. That would punish
    rewrapping instead of protecting the claim, so compare the words.
    """
    lines = [line.lstrip().removeprefix(">").strip()
             for line in _read(p).splitlines()]
    return " ".join(" ".join(lines).split())


@pytest.mark.parametrize("claim", RETIRED_README_CLAIMS)
def test_readme_does_not_restate_a_retired_claim(claim: str) -> None:
    assert claim not in _flat(README), (
        f"README restates a retired claim: {claim!r}. The preregistration is "
        f"finalized, GPU hours are on the ledger, and the S18 defect closed.")


def test_readme_states_the_true_lock_state() -> None:
    text = _flat(README)
    # P exists and is named by its commit; L does not exist and was refused.
    assert "5844a97" in text, "the README must name P by its commit"
    assert "p2_plan_state_act" in text, "the README must name the locked winner"
    for phrase in ("does not exist", "REFUSED"):
        assert phrase in text, f"the README must say L {phrase!r}"


def test_readme_denies_a_capability_result() -> None:
    text = _flat(README).upper()
    assert "NO CAPABILITY RESULT EXISTS YET" in text, (
        "no trained checkpoint and no held-out bytes exist; the README must say "
        "so in the status block, not merely omit a results table")


def test_readme_points_at_the_reporting_rules_and_the_ledger() -> None:
    text = _flat(README)
    assert "docs/INTERPRETATION.md" in text
    assert "docs/DEFERRED_REPAIRS.md" in text


# --- the referee's reporting rules ----------------------------------------

REQUIRED_WORDING_FRAGMENTS = (
    "295/300 (0.9833) certified successes on the development H4 all-tools "
    "orchestration axis",
    "296/300 (0.9867) on the development H8 execution axis",
    "could improve the respective rates by only 0.0167 and 0.0133, both below "
    "the preregistered +0.05 superiority margin",
    "they are not held-out estimates and do not determine the registered gate "
    "statuses",
)

PER_GATE_TEMPLATE_FRAGMENTS = (
    "We nevertheless evaluated the unchanged preregistered contrast",
    "structural-cluster bootstrap lower bound",
    "[PASS/FAIL/INCONCLUSIVE]",
    "It must not be interpreted as evidence that the training intervention had "
    "zero effect",
)

BANNED_PHRASES = (
    "Training failed.",
    "Training had nothing to add.",
    "No training effect.",
    "Both claims were inconclusive by saturation",
)


@pytest.mark.parametrize("fragment", REQUIRED_WORDING_FRAGMENTS)
def test_required_results_wording_is_recorded_verbatim(fragment: str) -> None:
    assert fragment in _flat(INTERPRETATION), (
        "the wording that must precede the held-out secondary results is "
        "recorded verbatim, so it cannot be softened later")


@pytest.mark.parametrize("fragment", PER_GATE_TEMPLATE_FRAGMENTS)
def test_per_gate_reporting_template_is_recorded(fragment: str) -> None:
    assert fragment in _flat(INTERPRETATION)


@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_banned_phrases_are_listed_as_banned(phrase: str) -> None:
    assert phrase in _flat(INTERPRETATION), (
        f"{phrase!r} must be listed among the phrases to avoid")


def test_interpretation_keeps_fail_distinct_from_inconclusive() -> None:
    text = _flat(INTERPRETATION)
    assert "MT1` normally becomes FAIL" in text or "MT1 normally becomes FAIL" in text
    # HR1's escape hatch is a REGISTERED condition, and must be named as one.
    assert "20 discordant" in text
    assert "Never convert a machine `FAIL` to `INCONCLUSIVE` in prose" in text


def test_interpretation_bounds_the_saturation_claim_to_the_dev_samples() -> None:
    text = _flat(INTERPRETATION)
    assert "not logically true of unseen held-out data" in text
    assert "establish that training has zero effect" in text


def test_interpretation_records_the_deep_horizon_counterpoint() -> None:
    text = _flat(INTERPRETATION)
    for fragment in ("1/26", "7/26", "244/248"):
        assert fragment in text, (
            "the floor at fulfillment H14/H20 is why harder strata are a bad "
            "trade; the numbers must survive in the record")


# --- the deferred-work ledger ---------------------------------------------

def test_deferred_ledger_names_the_tournament_receipt_as_pre_L() -> None:
    text = _flat(DEFERRED)
    assert "configs/frozen_prompt.json" in text
    assert "before `L`" in text, (
        "the tournament receipt correction is ordered before L, and the ledger "
        "must say so")
    assert "round2_candidates" in text


@pytest.mark.parametrize("path", (
    "scripts/run_multifaceted_chain.sh",
    "scripts/run_stage.sh",
    "tests/test_chain.py",
    "verify-heldout-release",
))
def test_deferred_ledger_names_the_file_each_repair_touches(path: str) -> None:
    assert path in _flat(DEFERRED), (
        "a deferral without the file it touches is not a ledger entry")


def test_deferred_ledger_protects_the_s18_receipts() -> None:
    text = _flat(DEFERRED)
    assert "results/agentic/locks.json" in text
    assert "results/agentic/seed_reveal.json" in text
    assert "dedicated" in text
