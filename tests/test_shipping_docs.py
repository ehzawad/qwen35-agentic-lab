"""The shipping documents must keep their promises without a human rereading them.

Three files are pinned here:

* `docs/RESULTS.md`   -- the results skeleton, whose verbatim required wording must
  survive and whose numeric slots must stay UNFILLED until the analyzer fills them.
* `docs/USAGE.md`     -- the one usage note / model card.
* `docs/AMENDED_REPLICATION_NOT_RUN.md` -- the record that one unregistered optional
  replication is not being run, that this is not a deviation, and that the registered
  7,800-episode evaluation is being completed.

Like `tests/test_public_record.py`, these tests read FILES ONLY -- no git, no GPU, no
network, no live repository state -- and they pin the SHAPE and the load-bearing
sentences, not figures that legitimately move. The distill snapshot counts ARE pinned,
because the referee requires those exact numerators and denominators: if the corpus is
resnapshotted, the new counts land in `docs/RESULTS.md` section 10 as a separate labelled
row and these constants are updated deliberately, never silently.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "docs" / "RESULTS.md"
USAGE = ROOT / "docs" / "USAGE.md"
SCOPE = ROOT / "docs" / "AMENDED_REPLICATION_NOT_RUN.md"


def _flat(p: pathlib.Path) -> str:
    """The file with blockquote markers dropped and whitespace collapsed.

    The verbatim wording is hard-wrapped inside `>` blockquotes, so asserting against
    raw bytes would punish rewrapping instead of protecting the claim.
    """
    assert p.exists(), f"{p.name} is part of the public record"
    lines = [ln.lstrip().removeprefix(">").strip()
             for ln in p.read_text(encoding="utf-8").splitlines()]
    return " ".join(" ".join(lines).split())


# --- the referee's verbatim wording, all five blocks ------------------------

STUDY_STATUS = (
    "The preregistered 7,800-episode evaluation was not completed. This study is "
    "reported as a deliberate post-registration deviation and partial completion. No "
    "preregistered study-level winner is claimed."
)

SATURATED_SECONDARIES = (
    "the selected prompt-only control achieved 295/300 (0.9833) certified successes on "
    "the development H4 all-tools orchestration axis and 296/300 (0.9867) on the "
    "development H8 execution axis"
)

PER_GATE = (
    "We evaluated the unchanged preregistered contrast on the observations available. "
    "BP achieved [kBP/n], TP achieved [kTP/n], the paired difference was [Δ], and the "
    "one-sided 97.5% structural-cluster bootstrap lower bound was [LB] over [G] "
    "clusters, with [d] discordant pairs. The preregistered gate status was "
    "[PASS/FAIL/INCONCLUSIVE], for the following registered reason: [reason]."
)

UNDERPOWERED_PRIMARY = (
    "The preregistered primary status was [PASS/FAIL/INCONCLUSIVE], with [|C|] "
    "common-clean pairs across [G] structural clusters and a TP−BP certified-recovery "
    "difference of [Δ], lower bound [LB]."
)

DESCRIPTIVE_PREFIX = (
    "**Descriptive only; not a preregistered claim:** [metric and exact "
    "numerator/denominator]. This estimate carries no registered decision threshold, "
    "does not change or replace any original gate, and must not be read as a "
    "confirmatory claim about training efficacy or general agentic capability."
)


@pytest.mark.parametrize("block", (
    STUDY_STATUS, SATURATED_SECONDARIES, PER_GATE, UNDERPOWERED_PRIMARY,
    DESCRIPTIVE_PREFIX,
), ids=("study_status", "saturated_secondaries", "per_gate",
        "underpowered_primary", "descriptive_prefix"))
def test_results_carries_the_required_wording_verbatim(block: str) -> None:
    assert block in _flat(RESULTS), (
        "the required wording is recorded before the numbers exist precisely so that "
        "it cannot be softened once they do")


def test_results_records_that_the_tripwire_actually_fired() -> None:
    """This test used to assert the study status was still CONDITIONAL.

    It is inverted deliberately. On 2026-08-06 rejection sampling was stopped at 15
    shards and the training, L, R, held-out and verdict stages were never run, so the
    registered 7,800-episode evaluation was not completed and the trigger the previous
    version of this test described as hypothetical became today's status. The paragraph
    is no longer a template here: it is the finding, and the assertions below make it
    impossible to quietly revert to the conditional phrasing while the deviation stands.
    """
    text = _flat(RESULTS)
    assert "THE TRIPWIRE FIRED ON 2026-08-06" in text
    assert "DEVIATION_2026-08-06.md" in text
    assert "No reduced evaluation was substituted" in text
    assert "a gate that was never evaluated is not a FAIL" in text
    # the conditional framing is preserved as history, explicitly labelled as such
    assert "The original conditional text, kept for the record" in text


@pytest.mark.parametrize("slot", (
    "[kBP/n]", "[kTP/n]", "[Δ]", "[LB]", "[G]", "[d]",
    "[PASS/FAIL/INCONCLUSIVE]", "[|C|]", "[reason]",
))
def test_numeric_slots_are_still_unfilled(slot: str) -> None:
    assert slot in _flat(RESULTS), (
        f"slot {slot!r} disappeared: a slot is filled only from the analyzer's "
        f"emitted output, and the template must survive being filled")


def test_results_declares_itself_a_skeleton_while_no_verdict_exists() -> None:
    text = _flat(RESULTS)
    assert "UNFILLED" in text
    assert "EMPTY BY DESIGN" in text


# --- the three training-side findings, exactly as the referee requires -----

TRAINING_SIDE_COUNTS = (
    "1,174/1,200",   # fulfillment-h4 certified
    "1,509/1,600",   # fulfillment-h8 certified
    "1,302/1,752",   # token-bearing recovery predicate met
    "59/1,152",      # fulfillment-h14 certified -- the cliff
    "2,742/3,952",   # pooled certified over the 13-shard snapshot
)


@pytest.mark.parametrize("count", TRAINING_SIDE_COUNTS)
def test_results_reports_exact_numerators_and_denominators(count: str) -> None:
    assert count in _flat(RESULTS), (
        "the referee requires exact counts, not a rounded percentage band")


def test_training_side_section_is_headed_as_not_held_out() -> None:
    text = _flat(RESULTS)
    assert "Training-side descriptive observations (not held-out)" in text


@pytest.mark.parametrize("caveat", (
    "NOT the adapter",
    "NOT an independent task sample",
    "NOT evidence about arbitrary multi-tool use",
    "mixture over the observed distillation cells",
    "NOT BP-on-`C`",
    "NOT a TP−BP contrast",
    "NOT recovery from arbitrary production failures",
    "NOT proof that horizon alone caused the collapse",
    "NOT proof that every deep task fails",
    "NOT evidence that the adapter will retain the same rate",
))
def test_each_finding_carries_its_required_caveat(caveat: str) -> None:
    assert caveat in _flat(RESULTS), (
        "a finding published without the caveat the referee attached to it is a "
        "different claim from the one the evidence supports")


def test_no_binomial_intervals_on_repeated_rollouts_and_the_reason_is_given() -> None:
    text = _flat(RESULTS)
    assert "repeated rollouts for the same tasks" in text
    assert "Ordinary binomial intervals treating all rollouts as independent would be" in text
    assert "exact numerators and denominators without inferential language" in text


def test_results_lists_the_banned_phrases_as_banned() -> None:
    text = _flat(RESULTS)
    for phrase in ("Training failed.", "Training had nothing to add.",
                   "No training effect."):
        assert phrase in text, f"{phrase!r} must be listed among the phrases to avoid"
    assert "must not appear" in text


def test_results_keeps_the_two_snapshots_apart() -> None:
    text = _flat(RESULTS)
    assert "13" in text and "3,952" in text
    assert "4,400" in text, "the later sealed-corpus count must be shown as its own row"
    assert "never silently mixed" in text


# --- the usage note / model card -------------------------------------------

MINIMUM_DISCLOSURE = (
    "This release is a tool-loop configuration for a synthetic five-tool suite, not a "
    "standalone autonomous system."
)


def test_usage_carries_the_minimum_disclosure_verbatim() -> None:
    text = _flat(USAGE)
    assert MINIMUM_DISCLOSURE in text
    for fragment in (
        "certified recovery was 1,302/1,752 faulted rollouts (74.3%)",
        "rate-limit recovery must occur on a later assistant decision",
        "not evidence of 74.3% recovery from arbitrary API, network, or infrastructure "
        "failures",
        "H14/H20 are outside the supported reliability envelope",
        "is not claimed to outperform the base-plus-prompt configuration",
    ):
        assert fragment in text, fragment


@pytest.mark.parametrize("disclosure", (
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",   # pinned base revision
    "5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d",  # prompt hash
    "apache-2.0",                                  # base-model licence identifier
    "Apache License 2.0",                          # this repository's licence
    "rank 32",                                     # LoRA rank
    "GRPO_NOT_RUN_HARDWARE_INFEASIBLE",            # GRPO was not run
    "text only",                                   # modality limit
    "RTX A5000",                                   # the only tested card
))
def test_usage_discloses_what_a_user_must_know(disclosure: str) -> None:
    assert disclosure in _flat(USAGE), disclosure


def test_usage_states_the_serving_contract_and_both_model_ids() -> None:
    text = _flat(USAGE)
    assert "bash scripts/serve.sh Qwen/Qwen3.5-4B" in text
    assert "--lora-modules trained=out/multiface/rssft-lora" in text
    assert "--max-lora-rank 32" in text
    assert "is not the shipped configuration" in text, (
        "a request without the frozen system prompt is not the shipped configuration "
        "and the usage note must say so")


def test_usage_leads_with_the_deep_horizon_collapse() -> None:
    text = _flat(USAGE)
    assert "The deep-horizon collapse" in text
    assert "The supported reliability envelope stops at H8" in text


def test_usage_licence_caveat_is_not_overstated() -> None:
    text = _flat(USAGE)
    # the identifier came from Hub metadata, not from the cached bytes
    assert "not verified from the cached bytes" in text
    assert "snapshot carries no `LICENSE` file" in text


# --- the scope decision ----------------------------------------------------

def test_scope_record_says_what_is_not_run_and_why_it_is_not_a_deviation() -> None:
    text = _flat(SCOPE)
    assert "cluster-balanced" in text
    assert "was never registered, and will not be run" in text
    assert "not a deviation" in text


def test_scope_record_commits_to_completing_the_registered_evaluation() -> None:
    text = _flat(SCOPE)
    assert "7,800" in text
    assert "177.590" in text or "177.6" in text
    assert "6.4" in text
    assert "projection from a measured rate" in text, (
        "6.4 GPU-h is projected from a measured rate; calling it a measured cost "
        "would be the kind of overstatement this repository refuses")
    assert "may not be entered into the ledger" in text


def test_scope_record_carries_the_reduced_size_rule_and_the_tripwire() -> None:
    text = _flat(SCOPE)
    assert "A permissible reduction requires all of the following before `L`" in text
    assert "It must not use the 0.9833, 0.9867, 0.743, or 0.051 rates to solve for" in text
    assert "docs/DEVIATION_<date>.md" in text


# --- the README points at all three, and stays honest ----------------------

@pytest.mark.parametrize("link", (
    "docs/RESULTS.md",
    "docs/USAGE.md",
    "docs/AMENDED_REPLICATION_NOT_RUN.md",
))
def test_readme_points_at_the_shipping_documents(link: str) -> None:
    assert link in _flat(README), link


def test_readme_carries_the_deep_horizon_warning_and_labels_every_row() -> None:
    text = _flat(README)
    assert "Deep horizons collapse" in text
    assert "59/1,152 (5.1%)" in text
    # every headline row is labelled dev or distill with a snapshot
    assert "distill | 13 shards" in text
    assert "dev | prompt tournament" in text
    assert "Descriptive only; not a preregistered claim" in text


def test_readme_states_the_registered_evaluation_was_not_completed() -> None:
    """The premise this test guarded changed on 2026-08-06, so the guard changed with it.

    It used to assert the README said the 7,800-episode census "is being completed".
    Rejection sampling was then stopped at 15 shards and the training, L, R, held-out and
    verdict stages were never run, so that sentence became false and this test failed --
    which is the guard doing its job, not a broken test. The assertion now enforces the
    replacement claim, so the README cannot quietly drift back to the completed story and
    cannot go silent about the deviation either.
    """
    text = _flat(README)
    assert "NOT COMPLETED" in text
    assert "DEVIATION_2026-08-06.md" in text
    assert "no registered study-level winner" in text.lower()
    # the old, now-false claim must be gone
    assert "is being completed" not in text


def test_the_deviation_notice_exists_and_carries_the_required_paragraph() -> None:
    """The tripwire in RESULTS.md promises a dated canonical notice; this is its receipt."""
    notice = ROOT / "docs" / "DEVIATION_2026-08-06.md"
    assert notice.exists(), "the tripwire fired but no dated notice was published"
    text = _flat(notice)
    assert "The preregistered 7,800-episode evaluation was not completed" in text
    assert "deliberate post-registration deviation and partial completion" in text
    # a gate that was never evaluated must not be reported as a failure
    assert "not a FAIL" in text or "is not a FAIL" in text
    # and it must say plainly that the train-vs-prompt comparison never ran
    assert "never run" in text
