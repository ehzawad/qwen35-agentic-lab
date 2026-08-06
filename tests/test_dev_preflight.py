"""The dev-only preflight, as a CPU test: probes 1 and 2 stay permanently runnable.

`scripts/preflight_dev.py` is the council's smallest credible pre-production
coherence check (round 5). Its two CPU probes are the discriminating ones and
they need neither a GPU nor a live run, so they are pinned here as well:

  probe 1   the exact 12-row extractor rescore -- 4/12 -> 11/12, the seven named
            rows and no others, and the false hallucination labels clearing
  probe 2   the oracle-driven fault parity matrix over the six committed dev
            tasks: the evaluation path and the canonical training path must
            agree byte for byte on envelopes, the whole event ledger, token
            arguments, budgets, progress and the verdict, in all four fault
            classes, and a BARE RETRY must never be certified

The last section of this module was an `xfail(strict=True)` recording the OPEN
DEFECT the preflight found (`results/agentic/preflight/probe2.json`): S17 read a
legitimate strict refusal as a harness BUG. That seam is now CLOSED and the
marker is gone -- the tests are positive assertions, and both mechanisms the
probe reproduced (a blind retry in the fault arm; a fault-free episode that
batches both hops into one decision) are kept as regression tests.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "preflight_dev.py"
CERTSPECS = REPO / "data" / "suite" / "v1" / "certspecs" / "dev.jsonl"


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location("preflight_dev", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the six tasks and the derived manifest
# ---------------------------------------------------------------------------

COUNCIL_TABLE = {
    "dev-lookup_chain-h2-0000": ("lookup_chain", 2, "transient"),
    "dev-lookup_chain-h12-0102": ("lookup_chain", 12, "malformed"),
    "dev-typed_relay-h2-0150": ("typed_relay", 2, "wrong_unit"),
    "dev-typed_relay-h12-0225": ("typed_relay", 12, "rate_limit"),
    "dev-fulfillment-h4-0102": ("fulfillment", 4, "malformed"),
    "dev-fulfillment-h20-0225": ("fulfillment", 20, "rate_limit"),
}


def test_the_preflight_uses_exactly_the_councils_six_tasks(pf):
    assert dict(pf.SIX) == {t: c for t, (_f, _h, c) in COUNCIL_TABLE.items()}
    assert len(pf.SIX) == 6
    assert len(pf.SIX_IDS) * len(pf.CONDITIONS) == 12


@pytest.mark.skipif(not CERTSPECS.exists(),
                    reason="the derived certification specs are regenerated, "
                           "not committed (scripts/export_eval_specs.py)")
def test_the_six_tasks_are_committed_with_the_registered_fault_classes(pf):
    """Three families x clean/faulted x low/high horizon, and nothing invented."""
    rows = {}
    for line in CERTSPECS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task_id"] in COUNCIL_TABLE:
            rows[row["task_id"]] = row
    assert set(rows) == set(COUNCIL_TABLE)
    for task_id, (family, horizon, fault) in COUNCIL_TABLE.items():
        row = rows[task_id]
        assert (row["family"], row["horizon"]) == (family, horizon)
        assert [f["fault_type"] for f in row["spec_row"]["faults"]] == [fault]
    families = {v[0] for v in COUNCIL_TABLE.values()}
    assert families == {"lookup_chain", "typed_relay", "fulfillment"}
    assert {v[2] for v in COUNCIL_TABLE.values()} == {
        "transient", "rate_limit", "malformed", "wrong_unit"}


# ---------------------------------------------------------------------------
# probe 1 -- the exact extractor rescore
# ---------------------------------------------------------------------------

def test_the_defective_grammar_scored_4_of_12(pf):
    from agentlab import provenance

    fixture = pf._load_fixture_rows()
    correct = {t for t, (ans, final) in fixture.items()
               if pf.pre_fix_extract(final) == ans}
    assert len(correct) == 4 and correct == set(pf.ALREADY_CORRECT)
    assert all(pf.pre_fix_extract(final) == "\\boxed{%s}" % ans
               for t, (ans, final) in fixture.items() if t in pf.RESCUED)
    assert provenance.extract_final_answer is not None


def test_the_repaired_grammar_scores_11_of_12(pf):
    from agentlab.suite.schema import extract_committed_answer

    fixture = pf._load_fixture_rows()
    correct = {t for t, (ans, final) in fixture.items()
               if extract_committed_answer(final) == ans}
    assert correct == set(pf.ALREADY_CORRECT) | set(pf.RESCUED)
    assert len(correct) == 11
    for task_id in pf.STILL_WRONG:
        assert extract_committed_answer(fixture[task_id][1]) is None


@pytest.mark.skipif(not pathlib.Path(REPO / "out" / "verify-a5000" / "traces"
                                    / "B0.clean.none.jsonl").exists(),
                    reason="the recorded 12-episode run is under out/, which is "
                           "not committed")
def test_the_seven_false_hallucination_labels_clear(pf):
    """The literal `\\boxed{x}` has no source in any observation; `x` does."""
    rows = pf.read_jsonl(pf.D1_TRACE)
    rescored = {r["task_id"]: r for r in pf.rescore_d1_rows(rows)["rows"]}
    assert len(rescored) == 12
    for task_id in pf.RESCUED:
        row = rescored[task_id]
        assert row["recorded_hallucinated"] is True
        assert row["before_sourced"] is False
        assert row["after_sourced"] is True
        assert row["after_correct"] is True
    for task_id in pf.ALREADY_CORRECT:
        assert rescored[task_id]["recorded_hallucinated"] is False
    # the recorded tally IS the defective grammar's reading, not a recollection
    assert all(r["before_correct"] == r["recorded_raw_success"]
               for r in rescored.values())


# ---------------------------------------------------------------------------
# probe 2 -- the oracle-driven fault parity matrix
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matrix(pf):
    """18 episode pairs: six tasks x (clean, faulted, faulted-with-a-bare-retry).

    The rendered-prefix token ids are deliberately not compared here (no
    tokenizer): `tests/test_environment_parity.py` and the probe itself own that
    surface. What is compared is every byte the model read, the whole hidden
    event ledger, the token arguments, the budgets, the credited progress and the
    verdict row.
    """
    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    bundles = pf.dev_bundles()
    cases = ([(t, "clean", "remediation") for t in pf.SIX_IDS]
             + [(t, "faulted", "remediation") for t in pf.SIX_IDS]
             + [(t, "faulted", "bare_retry") for t in pf.SIX_IDS])
    out = []
    for task_id, condition, mode in cases:
        bundle = pf.bundle_for(bundles[task_id], condition)
        train = pf.training_side(
            bundle, pf.ScriptedPolicy(bundle.spec, bundle.kb, bundle.nodes,
                                      secret, mode=mode),
            None, condition, secret)
        ev = pf.evaluation_side(
            specs[task_id], pf.ScriptedPolicy(bundle.spec, bundle.kb, bundle.nodes,
                                              secret, mode=mode),
            None, condition, secret)
        out.append({"task_id": task_id, "condition": condition, "mode": mode,
                    "fault_class": pf.FAULT_OF[task_id] if condition == "faulted"
                                   else None,
                    "train": train, "eval": ev})
    return out


def test_the_two_paths_face_the_identical_environment(pf, matrix):
    for case in matrix:
        for field in pf.PARITY_FIELDS:
            assert case["train"][field] == case["eval"][field], (
                f"{case['task_id']}/{case['condition']}/{case['mode']}: "
                f"{field} differs")
        assert case["eval"]["termination"] != "spec_error"


def test_one_success_predicate_across_both_paths(matrix):
    for case in matrix:
        assert (case["train"]["verdict"]["certified_success"]
                == case["eval"]["verdict"]["certified_success"]
                == case["eval"]["trace"]["score"]["certified_success"])


def test_every_scheduled_fault_fires_exactly_once(matrix):
    faulted = [c for c in matrix if c["condition"] == "faulted"]
    assert len(faulted) == 12
    for case in faulted:
        verdict = case["train"]["verdict"]
        assert verdict["fault_assigned"] == 1
        assert verdict["faults_triggered"] == 1
        assert set(verdict["fault_fire_counts"].values()) == {1}
    for case in matrix:
        if case["condition"] == "clean":
            assert case["train"]["verdict"]["fault_assigned"] == 0
            assert case["train"]["verdict"]["faults_triggered"] == 0


def test_the_registered_remediation_is_certified_in_all_four_classes(matrix):
    remed = [c for c in matrix
             if c["condition"] == "faulted" and c["mode"] == "remediation"]
    assert {c["fault_class"] for c in remed} == {"transient", "rate_limit",
                                                 "malformed", "wrong_unit"}
    for case in remed:
        assert case["train"]["verdict"]["recovery_reason"] == "ok"
        assert case["train"]["verdict"]["certified_success"]
        rec = case["eval"]["trace"]["score"]["recovery"]
        assert rec["certified_recovery"] is True and rec["reason"] == "ok"


def test_bare_retries_are_never_certified(matrix):
    """The token contract is the whole point: operationally fine, never credited.

    The label comes from the frozen non-recovery precedence. Five of the six are
    `blind_retry` -- the tool answered on the untokened re-issue. The wrong-unit
    arm is different by construction: accepting the trapped value means the
    committed answer appears in NO validated observation, and the frozen
    precedence puts that (ER7, `hallucinated`) above the remediation label.
    """
    from agentlab import provenance
    from agentlab.suite import verify

    registered = ((set(provenance.NON_RECOVERY_PRECEDENCE)
                   | set(verify.RECOVERY_REASON_ORDER)) - {"ok"})
    blind = [c for c in matrix if c["mode"] == "bare_retry"]
    assert len(blind) == 6
    for case in blind:
        assert case["train"]["verdict"]["recovery_reason"] != "ok"
        assert not case["train"]["verdict"]["certified_success"]
        rec = case["eval"]["trace"]["score"]["recovery"]
        assert rec["certified_recovery"] is False
        assert rec["reason"] != "ok"
        assert rec["reason"] in registered
    by_class = {c["fault_class"]: c["eval"]["trace"]["score"]["recovery"]["reason"]
                for c in blind}
    assert by_class["transient"] == "blind_retry"
    assert by_class["rate_limit"] == "blind_retry"
    assert by_class["malformed"] == "blind_retry"
    assert by_class["wrong_unit"] == "hallucinated"


def test_clean_oracle_episodes_are_certified(matrix):
    clean = [c for c in matrix if c["condition"] == "clean"]
    assert len(clean) == 6
    for case in clean:
        assert case["train"]["verdict"]["certified_success"], \
            case["train"]["verdict"]["reasons"]


# ---------------------------------------------------------------------------
# the cross-predicate seam probe 2 found, now closed
# ---------------------------------------------------------------------------

def test_S17_accepts_the_registered_oracle_episodes(pf, matrix):
    """The positive control: the analyzer's harness veto is happy with 12 clean
    and remediated episodes, and reproduces every verdict by canonical replay."""
    from agentlab import analyze

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    traces = [c["eval"]["trace"] for c in matrix if c["mode"] == "remediation"]
    veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes(traces, specs, secret))
    assert veto["status"] == "OK", veto["detail"]
    assert veto["numbers"]["replayed"] == 12
    # all twelve certify, so the repaired veto is not passing by leniency
    assert veto["numbers"]["strict_refusals"] == 0


@pytest.fixture(scope="module")
def batched_episode(pf):
    """Mechanism (b): fault-free, correct, both hops batched into ONE decision."""
    return pf._same_decision_episode(
        {r["task_id"]: r for r in pf.manifest_rows()}, pf.dev_bundles(),
        pf.secret_bytes())


def test_S17_does_not_read_a_legitimate_non_certified_episode_as_a_bug(
        pf, matrix, batched_episode):
    """The seam probe 2 pinned, in both reproduced mechanisms.

    `certify_episode` used to set `verdict_agrees = (certified_success ==
    ledger_ok)` -- an EQUALITY between the canonical predicate and a strictly
    WEAKER transcript-only one -- and S17 turned any inequality into a harness BUG
    that vetoed every gate, claim and the winner. Both mechanisms below are
    episodes the strict verifier is SUPPOSED to refuse and whose transcripts have
    nothing to object to, so the two predicates legitimately differ.
    """
    from agentlab import analyze

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}

    # (a) the fault arm: a blind retry, 5 of the 6 tasks (the wrong-unit arm
    #     commits a trapped value, so its transcript IS objectionable -- ER7)
    blind = [c["eval"]["trace"] for c in matrix if c["mode"] == "bare_retry"]
    assert len(blind) == 6
    blind_veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes(blind, specs, secret))
    # (b) fault-free: a clean episode that batches both hops into one decision
    batched_veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([batched_episode], specs, secret))

    assert not batched_episode["verdict"]["certified_success"]
    assert batched_episode["score"]["raw_success"]
    assert blind_veto["status"] != "BUG", blind_veto["detail"]
    assert batched_veto["status"] != "BUG", batched_veto["detail"]
    # and the veto is positively OK, not merely "not a BUG": the canonical
    # verdicts were reproduced field-for-field by canonical replay
    assert blind_veto["status"] == "OK", blind_veto["detail"]
    assert batched_veto["status"] == "OK", batched_veto["detail"]
    assert blind_veto["numbers"]["replayed"] == 6
    assert batched_veto["numbers"]["replayed"] == 1
    # the disagreement is COUNTED, not erased: 5 of the 6 blind retries and the
    # one batched episode are strict refusals with clean transcripts
    assert blind_veto["numbers"]["strict_refusals"] == 5
    assert batched_veto["numbers"]["strict_refusals"] == 1


def test_the_two_predicates_legitimately_disagree_and_are_counted_as_such(
        pf, matrix, batched_episode):
    """The disagreement is real, is the right shape, and is REPORTED not vetoed."""
    from agentlab import provenance

    secret = pf.secret_bytes()
    blind = [c["eval"]["trace"] for c in matrix if c["mode"] == "bare_retry"]
    reps = {t["task_id"]: provenance.certify_episode(t, secret)
            for t in blind if pf.FAULT_OF[t["task_id"]] != "wrong_unit"}
    reps["batched"] = provenance.certify_episode(batched_episode, secret)
    assert len(reps) == 6      # the five blind retries plus the batched episode

    for name, rep in reps.items():
        # the strictly weaker predicate holds; the strict one does not
        assert rep["ledger_ok"] is True, name
        assert rep["verdict_certified_success"] is False, name
        # so the two are NOT equal -- and that is a strict refusal, not a defect
        assert rep["strict_refusal"] is True, name
        assert rep["verdict_agrees"] is True, name
        assert rep["verdict_shared_mismatches"] == [], name
        assert rep["ledger_contradiction"] is None, name
        # a refusal must always say why: an unexplained one is the only way a
        # genuine strict-side defect could hide behind a legitimate refusal
        assert rep["unexplained_refusal"] is False, name
        assert rep["strict_refusal_reasons"], name
        # the certification the run may claim is still the strict one
        assert rep["certified_success"] is False, name

    # the wrong-unit bare retry is the OTHER shape: its transcript does object
    wrong_unit = next(t for t in blind
                      if pf.FAULT_OF[t["task_id"]] == "wrong_unit")
    rep = provenance.certify_episode(wrong_unit, secret)
    assert rep["ledger_ok"] is False and rep["strict_refusal"] is False
    assert rep["hallucination"]["hallucinated"] is True
    assert rep["verdict_agrees"] is True and rep["ledger_contradiction"] is None


def test_S17_still_vetoes_a_certification_the_ledger_cannot_support(pf, matrix):
    """The direction that IS a defect: certified_success without ledger support.

    Every ledger conjunct is also a conjunct of the canonical predicate, so the
    strict verdict must IMPLY the ledger conditions. Tampering with the recorded
    observation bytes breaks the receipt chain while the recorded verdict still
    claims certification, and that must still be a BUG.
    """
    import copy

    from agentlab import analyze, provenance

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    good = next(c["eval"]["trace"] for c in matrix
                if c["condition"] == "clean"
                and c["eval"]["trace"]["verdict"]["certified_success"])

    tampered = copy.deepcopy(good)
    ev = tampered["events"][0]
    ev["exposed_text"] = ev["exposed_text"][:-1] + " "
    rep = provenance.certify_episode(tampered, secret)
    assert rep["receipts_ok"] is False and rep["ledger_ok"] is False
    assert rep["verdict_certified_success"] is True
    assert rep["ledger_contradiction"] is not None
    assert rep["verdict_agrees"] is False
    assert rep["strict_refusal"] is False
    veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([tampered], specs, secret))
    assert veto["status"] == "BUG", veto["detail"]


def test_S17_still_vetoes_a_mis_read_correct_answer(pf, matrix):
    """The property S17 exists for: one side reads the commitment differently.

    `raw_success` is the SAME predicate on both sides, so a canonical verdict that
    calls a correct answer wrong (or a wrong answer correct) while the ledger
    recomputation of the very same grammar disagrees is a scorer defect, and no
    amount of legitimate strict refusal may absorb it.
    """
    import copy

    from agentlab import analyze, provenance

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    good = next(c["eval"]["trace"] for c in matrix
                if c["condition"] == "clean"
                and c["eval"]["trace"]["verdict"]["certified_success"])

    mis_scored = copy.deepcopy(good)
    # the scorer mis-read a correct answer: the strict side says the commitment is
    # wrong, the transcript says it is right
    mis_scored["verdict"]["raw_success"] = False
    mis_scored["verdict"]["answer_ok"] = False
    mis_scored["verdict"]["certified_success"] = False
    mis_scored["verdict"]["reasons"] = ["final answer missing or wrong"]
    mis_scored["score"] = dict(mis_scored["score"], certified_success=False)
    rep = provenance.certify_episode(mis_scored, secret)
    assert rep["raw_success"] is True
    assert rep["verdict_shared_mismatches"], rep
    assert any(d.startswith("raw_success") for d in rep["verdict_shared_mismatches"])
    assert rep["verdict_agrees"] is False
    veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([mis_scored], specs, secret))
    assert veto["status"] == "BUG", veto["detail"]


def test_S17_still_vetoes_a_refusal_that_records_no_reason(pf, matrix):
    """A silent refusal is the only place a strict-side defect could hide."""
    import copy

    from agentlab import analyze, provenance

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    blind = copy.deepcopy(next(c["eval"]["trace"] for c in matrix
                               if c["mode"] == "bare_retry"
                               and pf.FAULT_OF[c["task_id"]] != "wrong_unit"))
    assert not blind["verdict"]["certified_success"]
    assert blind["verdict"]["reasons"], "the legitimate refusal DOES say why"
    blind["verdict"]["reasons"] = []
    rep = provenance.certify_episode(blind, secret)
    assert rep["unexplained_refusal"] is True
    veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([blind], specs, secret))
    assert veto["status"] == "BUG", veto["detail"]


def test_S17_still_vetoes_a_verdict_canonical_replay_cannot_reproduce(pf, matrix):
    """Canonical against canonical replay, field for field -- the primary check."""
    import copy

    from agentlab import analyze

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    good = copy.deepcopy(next(c["eval"]["trace"] for c in matrix
                              if c["condition"] == "clean"))
    # a field only the canonical replay can speak to: oracle progress
    good["verdict"]["unique_valid_nodes"] = good["verdict"]["unique_valid_nodes"] + 7
    veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([good], specs, secret))
    assert veto["status"] == "BUG", veto["detail"]
    assert "unique_valid_nodes" in veto["detail"]
