"""The registered size ceilings must follow the COMMITTED census, not a memory.

`scenario.tool_output_max_tokens` was 208 -- a number measured against the
tokenless payloads: no recovery token, no remediation text, no receipt line. Once
the fault contract was unified it was silently wrong, and nothing in the repository
would have said so, because no consumer read it and no test compared it to a
measurement.

These tests bind the caps to `results/agentic/token_census.json`, the exhaustive
measurement the preregistration amendment pins by hash: 78,100 episodes over all
four committed train/dev splits, all twelve family/horizon cells, the clean case,
every eligible fault class, the same-decision rate-limit repeat and the ambiguous
malformed mutation, with 742,500 model-visible observations and 624,800 rendered
terminal views across all eight preregistered prompt candidates.

A cap below the measured maximum would reject legitimate episodes; a census taken
under a different environment contract would not describe this environment. Both
are failures here.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agentlab.suite.configio import ROOT, load_config
from agentlab.suite.contract import environment_contract_sha256
from agentlab.suite.schema import digest_text

CENSUS_PATH = ROOT / "results" / "agentic" / "token_census.json"
# The hash the preregistration amendment records. Editing the artifact without
# editing the amendment (or vice versa) fails here.
CENSUS_SHA256 = "98379c42540a30d7c6c29fea53193a169714351f184f9c58b516484dc5896fa8"

# What the census measured, restated so a reader sees the numbers the caps follow
# without opening a 476 KB artifact. Asserted against the artifact below.
MEASURED_TOOL_RESULT_TOKENS = 231
MEASURED_TOOL_RESULT_CHARS = 474
MEASURED_TERMINAL_VIEW_TOKENS = 4960


@pytest.fixture(scope="module")
def census() -> dict:
    assert CENSUS_PATH.exists(), (
        f"{CENSUS_PATH} is missing. The size ceilings are denominated in a "
        f"measurement, so the measurement is committed: run `make token-census` "
        f"and copy the artifact here.")
    return json.loads(CENSUS_PATH.read_text(encoding="utf-8"))


def test_the_committed_census_is_the_one_the_amendment_pins(census):
    raw = CENSUS_PATH.read_bytes()
    assert digest_text(raw.decode("utf-8")) == CENSUS_SHA256
    stated = (CENSUS_PATH.parent / (CENSUS_PATH.name + ".sha256"))
    if stated.exists():
        assert stated.read_text().split()[0] == CENSUS_SHA256
    assert census["kind"] == "token_census"


def test_the_census_describes_THIS_environment(census):
    """A census taken under a different contract measures a different environment."""
    assert census["environment_contract_sha256"] == environment_contract_sha256()


def test_the_census_is_exhaustive_over_the_registered_strata(census):
    assert sorted(census["splits"]) == ["dev", "distill", "grpo_train", "oracle_sft"]
    assert census["limit_per_split_per_cell"] is None, "a sampled census is not this one"
    assert set(census["variants"]) == {
        "clean", "transient", "rate_limit", "rate_limit_same_decision",
        "malformed", "malformed_ambiguous", "wrong_unit"}
    assert len(census["prompt_candidates"]) == 8
    assert census["episodes_measured"] >= 78100
    assert census["tool_result"]["observations_measured"] >= 742500
    assert census["rendered_terminal_view"]["views_measured"] >= 624800
    # every one of the twelve cells is present in both strata
    from agentlab.suite.schema import CELLS

    for family, horizon in CELLS:
        cell = f"{family}-h{horizon}"
        assert any(cell in label for label in census["tool_result"]["per_stratum"]), cell
        assert any(cell in label
                   for label in census["rendered_terminal_view"]["per_stratum"]), cell


def test_the_measured_maxima_are_what_this_module_states(census):
    assert census["tool_result"]["worst"]["max_tokens"] == MEASURED_TOOL_RESULT_TOKENS
    assert census["tool_result"]["worst_chars"]["chars"] == MEASURED_TOOL_RESULT_CHARS
    assert (census["rendered_terminal_view"]["worst"]["max_tokens"]
            == MEASURED_TERMINAL_VIEW_TOKENS)


def test_the_tool_result_caps_are_at_or_above_the_measured_maximum(census):
    cfg = load_config()
    tokens = cfg["scenario"]["tool_output_max_tokens"]
    chars = cfg["scenario"]["tool_output_max_chars"]
    assert tokens >= census["tool_result"]["worst"]["max_tokens"], (
        f"tool_output_max_tokens {tokens} is below the measured maximum "
        f"{census['tool_result']['worst']['max_tokens']} at "
        f"{census['tool_result']['worst']['stratum']} -- the cap would reject a "
        f"legitimate observation")
    assert chars >= census["tool_result"]["worst_chars"]["chars"]
    # the retired value, measured against the tokenless payloads, is now too small
    assert tokens > 208


def test_the_view_budget_is_at_or_above_the_measured_maximum(census):
    cfg = load_config()
    view = cfg["acceptance"]["max_view_tokens"]
    measured = census["rendered_terminal_view"]["worst"]["max_tokens"]
    assert view >= measured, (
        f"acceptance.max_view_tokens {view} is below the measured maximum "
        f"{measured} at {census['rendered_terminal_view']['worst']['stratum']} -- "
        f"the registered wire format would structurally exclude valid H20 "
        f"trajectories that fit under the retired tokenless treatment")
    assert view > 4096


def test_the_view_budget_and_the_trainer_length_move_together():
    """A view the builder accepts and the trainer truncates is a different signal."""
    cfg = load_config()
    assert cfg["sft"]["max_length"] == cfg["acceptance"]["max_view_tokens"]


def test_the_serving_context_still_covers_the_longest_view(census):
    from agentlab.suite.configio import engine_contract

    cfg = load_config()
    contract = engine_contract(cfg)
    longest = census["rendered_terminal_view"]["worst"]["max_tokens"]
    per_decision = int(cfg["eval_decoding"]["max_tokens_per_decision"])
    assert longest + per_decision <= contract["max_model_len"], (
        "the 8,192-token serving context must still hold the longest rendered "
        "view plus one decision; if it does not, the engine contract moves and "
        "that is a hardware decision, not a cap edit")


def test_the_decision_budgets_did_not_move():
    """Recovery costs one call and one decision; observations are INPUT tokens.

    The census raised an input-length cap. It must not have been used as cover to
    change a completion cap or a call/decision budget -- those are registered
    numbers and no measurement here bears on them.
    """
    from agentlab.suite.schema import call_budget, decision_budget

    cfg = load_config()
    assert cfg["decoding"]["max_tokens_per_decision"] == 384
    assert cfg["eval_decoding"]["max_tokens_per_decision"] == 1024
    assert (decision_budget(8, 0), decision_budget(8, 1), decision_budget(8, 2)) == \
        (11, 13, 16)
    assert call_budget(8) == 20


def test_the_a5000_sft_arithmetic_holds_at_the_new_length(census):
    """5,120 tokens must still fit the registered card, checked not assumed."""
    cfg = load_config()
    length = int(cfg["sft"]["max_length"])
    vocab = 248320                      # the padded embedding size in the config
    static_gib = 9.423                  # bf16 policy 8.455 + LoRA fp32 state 0.968
    card_gib = 25282805760 / 2 ** 30    # the registered CUDA-visible bytes
    logits_gib = length * vocab * 2 / 2 ** 30
    train_peak = static_gib + int(cfg["sft"]["bsz"]) * logits_gib
    eval_peak = static_gib + int(cfg["sft"]["eval_bsz"]) * logits_gib
    assert train_peak < card_gib - 8.0, (
        f"train peak {train_peak:.2f} GiB leaves under 8 GiB of the "
        f"{card_gib:.1f} GiB card for activations")
    assert eval_peak < card_gib - 10.0
    # the dangerous Hugging Face default still does not fit, which is why
    # eval_bsz is pinned at 1
    assert static_gib + 8 * logits_gib > card_gib
    assert int(cfg["sft"]["eval_bsz"]) == 1
    assert cfg["sft"]["prediction_loss_only"] is True


def test_no_observation_in_any_committed_cell_exceeds_the_char_proxy(census):
    """The acceptance filter enforces CHARS, so every stratum must clear it."""
    cfg = load_config()
    limit = cfg["scenario"]["tool_output_max_chars"]
    over = {label: row["max_chars"]
            for label, row in census["tool_result"]["per_stratum"].items()
            if row["max_chars"] > limit}
    assert over == {}, f"strata above the {limit}-char acceptance proxy: {over}"
