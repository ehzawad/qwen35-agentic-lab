"""End-to-end machine verdict: S8-S18 vetoes, ER/MT/HR gates, floors, winner.

The heavy fixture drives the REAL episode loop (scripted policies, real fault
injection, real receipts) at preregistered scale: >= 900 assigned fault pairs,
|C| >= 500, 100k-replicate deterministic bootstraps from the frozen
preregistration file. Small fixtures then break the harness on purpose and
assert that a BUG vetoes everything.
"""

import hashlib
import json
import pathlib

import pytest
from agentic_helpers import (SECRET, Guesser, ScriptedOracle, chain_spec,
                             make_arm_policy, run)

from agentlab import provenance
from agentlab.analyze import agentic_verdict, render_agentic_verdict

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "agentic_preregister.json"
P8 = REPO / "prompts" / "agentic" / "p8_combined.txt"
P8_SHA = hashlib.sha256(P8.read_bytes()).hexdigest()

N_TASKS = 920
N_REDACTED = 200   # per arm; one family -> 400 total >= 200*2
N_PERMUTED = 60    # per arm; 120 total >= 100


def _write(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _locks_and_reveal(results_dir: pathlib.Path):
    locks = {"checkpoint": {"path": "out/qwen35-4b-rssft-lora",
                            "locked_at": "2026-08-05T00:00:00Z", "commit": "aaa111"},
             "prompt_winner": {"file": "prompts/agentic/p8_combined.txt",
                               "sha256": P8_SHA,
                               "locked_at": "2026-08-05T00:00:00Z", "commit": "aaa111"}}
    seed = int.from_bytes(hashlib.sha256(
        b"deadbeef:agentic-heldout-v1").digest()[:8], "big")
    reveal = {"revealed_at": "2026-08-05T01:00:00Z",
              "preregistration_commit": "deadbeef", "heldout_seed": seed}
    (results_dir / "locks.json").write_text(json.dumps(locks))
    (results_dir / "seed_reveal.json").write_text(json.dumps(reveal))


def _run_arm_traces(specs, arm, out_dir, *, recover_pct=None):
    clean_rows, fault_rows = [], []
    for spec in specs:
        clean_rows.append(run(spec, ScriptedOracle(spec), arm=arm,
                              condition="clean", prompt_sha=P8_SHA))
        fault_rows.append(run(spec, make_arm_policy(arm, spec,
                                                    recover_pct=recover_pct),
                              arm=arm, condition="faulted", prompt_sha=P8_SHA))
    _write(out_dir / f"{arm}.clean.none.jsonl", clean_rows)
    _write(out_dir / f"{arm}.faulted.none.jsonl", fault_rows)


@pytest.fixture(scope="module")
def full_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("agentic-verdict")
    traces = root / "traces"
    specs = [chain_spec(i, horizon=4) for i in range(N_TASKS)]

    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces)
        red_rows = [run(provenance.redact_spec(s), ScriptedOracle(
                        provenance.redact_spec(s)), arm=arm, condition="clean",
                        control="redacted", prompt_sha=P8_SHA)
                    for s in specs[:N_REDACTED]]
        _write(traces / f"{arm}.clean.redacted.jsonl", red_rows)
        permuted = provenance.permute_hidden_values(specs, seed=2786983944)
        perm_rows = [run(p, ScriptedOracle(p), arm=arm, condition="clean",
                         control="permuted", prompt_sha=P8_SHA)
                     for p in permuted[:N_PERMUTED]]
        _write(traces / f"{arm}.clean.permuted.jsonl", perm_rows)

    specs_path = root / "eval.jsonl"
    _write(specs_path, specs)
    train_path, dev_path = root / "train.jsonl", root / "dev.jsonl"
    _write(train_path, [chain_spec(i, split="train", ns="train-a") for i in range(20)])
    _write(dev_path, [chain_spec(i, split="dev", ns="dev-a") for i in range(20)])
    secret_path = root / "secret.hex"
    secret_path.write_text(SECRET.hex())
    _locks_and_reveal(root)

    verdict = agentic_verdict(
        str(traces), str(PREREG), str(secret_path), specs_path=str(specs_path),
        split_manifests={"train": str(train_path), "dev": str(dev_path),
                         "eval": str(specs_path)},
        results_dir=str(root))
    return verdict


def test_happy_path_vetoes_all_clean(full_results):
    v = full_results
    for name, res in v["vetoes"].items():
        assert res["status"] == "OK", (name, res)
    assert not v["any_bug"]


def test_happy_path_primary_claim_passes(full_results):
    v = full_results
    for g in ("ER1", "ER2", "ER3", "ER4", "ER5", "ER6", "ER7", "ER8"):
        assert v["gates"][g]["status"] == "PASS", (g, v["gates"][g])
    assert v["claims"]["primary_certified_error_recovery"] == "PASS"
    # the bound really is the preregistered clustered LB above the margin
    assert v["gates"]["ER2"]["numbers"]["lb"] > 0.05
    assert v["gates"]["ER2"]["numbers"]["n_pairs"] >= 500
    assert v["gates"]["ER5"]["numbers"]["n"] >= 900


def test_happy_path_secondaries_are_inconclusive_without_their_samples(full_results):
    v = full_results
    # no H4 all-tools pairs and no H8 pairs in this fixture: the analyzer must
    # say INCONCLUSIVE, never PASS/FAIL on absent evidence.
    assert v["gates"]["MT1"]["status"] == "INCONCLUSIVE"
    assert v["gates"]["HR1"]["status"] == "INCONCLUSIVE"
    assert v["claims"]["secondary_all_tools_orchestration_H4"] == "INCONCLUSIVE"
    assert v["claims"]["secondary_H8_execution_reliability"] == "INCONCLUSIVE"


def test_happy_path_winner_is_trained_arm(full_results):
    v = full_results
    for name, res in v["floors"]["TP"].items():
        assert res["status"] == "PASS", (name, res)
    assert v["winner"].startswith("TP")


def test_happy_path_render_mentions_rejected_claims(full_results):
    text = render_agentic_verdict(full_results)
    assert "Winner: TP" in text
    assert "general agentic competence" in text
    assert "H50" in text


def test_curves_are_present_and_censored_not_fitted(full_results):
    cur = full_results["curves"]["TP/clean/lookup_chain"]
    assert cur["points"][0]["n"] == N_TASKS
    assert cur["H50"].startswith("right-censored")


# ---------------------------------------------------------------------------
# small adversarial fixtures: BUG vetoes everything
# ---------------------------------------------------------------------------

def _mini(tmp_path, mutate=None, arms=("BP", "TP"), n=12):
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(n)]
    rows_by_file = {}
    for arm in arms:
        rows_by_file[f"{arm}.clean.none.jsonl"] = [
            run(s, ScriptedOracle(s), arm=arm, condition="clean",
                prompt_sha=P8_SHA) for s in specs]
        rows_by_file[f"{arm}.faulted.none.jsonl"] = [
            run(s, ScriptedOracle(s), arm=arm, condition="faulted",
                prompt_sha=P8_SHA) for s in specs]
    if mutate:
        mutate(rows_by_file, specs)
    for name, rows in rows_by_file.items():
        _write(traces / name, rows)
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    return agentic_verdict(str(traces), str(PREREG), str(secret_path),
                           results_dir=str(tmp_path))


def test_underpowered_common_clean_is_inconclusive_not_pass(tmp_path):
    v = _mini(tmp_path)
    assert v["gates"]["ER1"]["status"] == "INCONCLUSIVE"
    assert v["gates"]["ER2"]["status"] == "INCONCLUSIVE"
    assert v["claims"]["primary_certified_error_recovery"] == "INCONCLUSIVE"


def test_s8_pairing_mismatch_is_a_bug(tmp_path):
    def mutate(rows, specs):
        rows["TP.clean.none.jsonl"] = rows["TP.clean.none.jsonl"][:-1]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S8"]["status"] == "BUG"
    assert v["any_bug"]
    assert all(s == "BUG" for s in v["claims"].values())
    assert v["winner"].startswith("NO VERDICT")


def test_s11_redacted_success_is_a_bug(tmp_path):
    def mutate(rows, specs):
        red = provenance.redact_spec(specs[0])
        leak = run(red, Guesser(specs[0]["answer"]), arm="TP", condition="clean",
                   control="redacted", prompt_sha=P8_SHA)
        rows["TP.clean.redacted.jsonl"] = [leak]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S11"]["status"] == "BUG"
    assert v["any_bug"] and v["winner"].startswith("NO VERDICT")


def test_s11_unexecuted_redacted_control_is_a_bug_not_a_pass(tmp_path):
    """A control that never ran must never read as a clean control.

    Before the fix every redacted episode looked like this row: the runtime
    rejected the deliberately-broken oracle, so the policy made zero decisions and
    S11 reported "zero raw and certified success" over nothing at all.
    """
    def mutate(rows, specs):
        red = provenance.redact_spec(specs[0])
        stub = run(red, Guesser("whatever"), arm="TP", condition="clean",
                   control="redacted", prompt_sha=P8_SHA)
        stub["messages"], stub["events"] = [], []
        stub["runner"] = {"n_decisions": 0, "n_calls": 0,
                          "termination_reason": "spec_error", "wall_s": 0.0}
        stub["score"] = {"raw_success": False, "certified_success": False,
                         "runaway": False, "hallucinated": False}
        rows["TP.clean.redacted.jsonl"] = [stub]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S11"]["status"] == "BUG", v["vetoes"]["S11"]
    assert "never ran" in v["vetoes"]["S11"]["detail"]
    assert v["any_bug"] and v["winner"].startswith("NO VERDICT")


def test_a_harness_bug_vetoes_every_gate_and_floor_without_erasing_them(tmp_path):
    """A BUG vetoes downstream model-level gates AND floors, relabel not delete."""
    def mutate(rows, specs):
        rows["TP.clean.none.jsonl"] = rows["TP.clean.none.jsonl"][:-1]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S8"]["status"] == "BUG"
    assert {g["status"] for g in v["gates"].values()} == {"BUG"}
    assert {f["status"] for arm in v["floors"].values() for f in arm.values()} == {"BUG"}
    # nothing is hidden: what each gate measured survives for the record
    assert all("measured_status" in g for g in v["gates"].values())
    assert all("S8" in g["detail"] for g in v["gates"].values())
    assert v["winner"].startswith("NO VERDICT")


def test_no_gate_silently_degrades_a_fail_into_inconclusive(tmp_path):
    """Underpowered gates say INCONCLUSIVE, but a measured FAIL stays on the record.

    Also pins the companion rule: an interval gate whose threshold is unreachable
    at the observed n is INCONCLUSIVE (a sample-size statement), never a FAIL
    blamed on the policy.
    """
    v = _mini(tmp_path)  # n=12: far below every preregistered sample size
    for name in ("ER2", "ER3", "ER6", "ER7"):
        assert v["gates"][name]["status"] == "INCONCLUSIVE", (name, v["gates"][name])
    # ER6/ER7 are unmeasurable at n=24, and they say so with the arithmetic
    assert "unmeasurable at n=" in v["gates"]["ER7"]["detail"]
    assert v["gates"]["ER7"]["numbers"]["best_possible_ub"] > 0.01
    # a downgrade never deletes the measurement it downgraded
    for name in ("ER2", "ER3"):
        g = v["gates"][name]
        if "measured_status" in g:
            assert g["measured_status"] in ("PASS", "FAIL")
            assert "measured anyway for the record" in g["detail"]
    # and INCONCLUSIVE never reads as support
    assert v["claims"]["primary_certified_error_recovery"] == "INCONCLUSIVE"
    assert not v["winner"].startswith("TP")


def test_a_real_fail_outranks_missing_evidence_in_a_claim(tmp_path):
    """One refuted gate refutes the claim even when other gates are unmeasured."""
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(40)]
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces, recover_pct={"BP": 90, "TP": 10})
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(secret_path),
                        results_dir=str(tmp_path))
    assert not v["any_bug"]
    # the trained arm recovers far worse than the prompted base: ER8 must FAIL
    assert v["gates"]["ER8"]["status"] == "FAIL", v["gates"]["ER8"]
    assert any(g["status"] == "INCONCLUSIVE" for g in v["gates"].values())
    assert v["claims"]["primary_certified_error_recovery"] == "FAIL"
    assert not v["winner"].startswith("TP")


def test_s13_forged_receipt_is_a_bug(tmp_path):
    def mutate(rows, specs):
        row = rows["TP.clean.none.jsonl"][0]
        row["events"][0]["receipt"] = "r-" + "0" * 32
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S13"]["status"] == "BUG"
    assert v["any_bug"]


def test_s17_score_tampering_is_a_bug(tmp_path):
    def mutate(rows, specs):
        row = rows["TP.faulted.none.jsonl"][0]
        row["score"]["certified_success"] = not row["score"]["certified_success"]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S17"]["status"] == "BUG"
    assert v["any_bug"]


def test_s16_adapter_in_prompt_only_arm_is_a_bug(tmp_path):
    def mutate(rows, specs):
        for row in rows["BP.clean.none.jsonl"]:
            row["provenance"]["adapter"] = "out/some-adapter"
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S16"]["status"] == "BUG"


def test_s18_reveal_before_lock_is_a_bug(tmp_path):
    def no_mutate(rows, specs):
        pass
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(4)]
    for arm in ("BP", "TP"):
        _write(traces / f"{arm}.clean.none.jsonl",
               [run(s, ScriptedOracle(s), arm=arm, condition="clean",
                    prompt_sha=P8_SHA) for s in specs])
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    _locks_and_reveal(tmp_path)
    reveal = json.loads((tmp_path / "seed_reveal.json").read_text())
    reveal["revealed_at"] = "2025-01-01T00:00:00Z"  # before the locks
    (tmp_path / "seed_reveal.json").write_text(json.dumps(reveal))
    v = agentic_verdict(str(traces), str(PREREG), str(secret_path),
                        results_dir=str(tmp_path))
    assert v["vetoes"]["S18"]["status"] == "BUG"


def test_winner_falls_back_to_prompted_base_when_recovery_gate_fails(tmp_path):
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(40)]
    # both arms recover identically (50/50 by task hash): no trained edge,
    # but both clear the launch floors.
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces,
                        recover_pct={"BP": 50, "TP": 50})
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(secret_path),
                        results_dir=str(tmp_path))
    assert not v["any_bug"]
    assert v["gates"]["ER2"]["status"] != "PASS"  # INCONCLUSIVE at this n
    assert v["winner"].startswith("BP")
