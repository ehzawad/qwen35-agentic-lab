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

The last test in this module documents an OPEN DEFECT the preflight found (see
`results/agentic/preflight/probe2.json`) and is `xfail(strict=True)`: it asserts
the behaviour the harness must have, so the day the seam is repaired the marker
fires and forces this test to be flipped into a positive assertion rather than
left behind.
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
# the open defect probe 2 found
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


@pytest.mark.xfail(strict=True, reason=(
    "OPEN DEFECT found by the dev-only preflight (probe 2, "
    "results/agentic/preflight/probe2.json). `provenance.certify_episode` sets "
    "`verdict_agrees = (verdict.certified_success == ledger_ok)`, an EQUALITY "
    "between the canonical verdict and a strictly weaker transcript-only "
    "predicate, and `analyze.veto_s17_trace_summary` turns any inequality into a "
    "harness BUG -- which vetoes every gate, claim and the winner. So a "
    "legitimately non-certified episode with a clean transcript (a blind retry "
    "in the fault arm; both hops batched into one decision in the clean arm) "
    "reads as a broken harness. When the seam is repaired this xfail turns into "
    "an unexpected pass and must be flipped into a positive assertion."))
def test_S17_does_not_read_a_legitimate_non_certified_episode_as_a_bug(pf, matrix):
    from agentlab import analyze

    secret = pf.secret_bytes()
    specs = {r["task_id"]: r for r in pf.manifest_rows()}
    bundles = pf.dev_bundles()

    blind = [c["eval"]["trace"] for c in matrix if c["mode"] == "bare_retry"]
    blind_veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes(blind, specs, secret))
    batched = pf._same_decision_episode(specs, bundles, secret)
    batched_veto = analyze.veto_s17_trace_summary(
        pf._analyzer_episodes([batched], specs, secret))

    # every one of these episodes is correctly REFUSED certification, and the
    # transcript-only recomputation has nothing to object to
    assert not batched["verdict"]["certified_success"]
    assert batched["score"]["raw_success"]
    assert blind_veto["status"] != "BUG", blind_veto["detail"]
    assert batched_veto["status"] != "BUG", batched_veto["detail"]
