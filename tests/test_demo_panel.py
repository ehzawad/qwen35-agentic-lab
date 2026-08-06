"""The shipped demo must show the machinery, and must not be able to cherry-pick.

These tests run on CPU with no server and no GPU: they drive
`scripts/demo_agentic.py`'s printer with a scripted policy over the REAL
committed dev spec that carries the registered `rate_limit` fault, and they check
the two properties that make the demo honest rather than promotional:

  1. the panel is a consequence of declared rules over the committed manifest,
     so it cannot have been chosen after seeing outcomes -- and the derivation
     REFUSES when the manifest disagrees;
  2. the printed episode really exposes the injected fault envelope, its recovery
     token, the remediation the registered contract demands, the remedial call
     and the verifier's per-fault verdict -- and the deep-horizon cliff is
     printed whether that episode passed or failed.

The serving document is checked against `configio.engine_contract()` for the same
reason every stage reads that one copy: a page that drifts from the served engine
is worse than no page.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import pytest

import agentic_helpers as H
from agentlab.suite import configio, evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_SPECS = ROOT / "data/suite/v1/certspecs/dev.jsonl"


def _load_demo():
    path = ROOT / "scripts" / "demo_agentic.py"
    spec = importlib.util.spec_from_file_location("demo_agentic", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_agentic"] = mod
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo()


@pytest.fixture(scope="module")
def dev_specs():
    if not DEV_SPECS.exists():
        pytest.skip(f"{DEV_SPECS} not generated in this tree")
    return evaluate.load_specs(DEV_SPECS)


@pytest.fixture(scope="module")
def panel(dev_specs):
    return demo.derive_panel(dev_specs)


# ---------------------------------------------------------------------------
# the panel cannot be outcome-selected
# ---------------------------------------------------------------------------

def test_panel_ids_derive_from_declared_rules(panel):
    """Every id follows from a cell/fault-class rule over the committed manifest."""
    assert [row["task_id"] for row, _ in panel] == [
        r["task_id"] for r in demo.PANEL]
    assert len(panel) == 5


def test_panel_has_one_real_injected_fault_and_a_deep_task(panel):
    faulted = [(row, spec) for row, spec in panel if row["condition"] != "clean"]
    assert len(faulted) == 1, "a clean-only demo cannot support a recovery claim"
    row, spec = faulted[0]
    committed = (spec["spec_row"]["faults"] or [])[0]
    assert committed["fault_type"] == "rate_limit"
    assert committed["params"]["retry_after_turns"] == 1
    deep = [row for row, _ in panel
            if row["family"] == "fulfillment" and row["horizon"] >= 14]
    assert deep, "the demo must show the deep-horizon boundary, not hide it"


def test_derive_panel_refuses_a_manifest_that_disagrees(dev_specs):
    """Drop the first task of a declared cell: the rule now selects another id,
    and the demo must refuse rather than demonstrate some other task."""
    dropped = [s for s in dev_specs if s["task_id"] != "dev-typed_relay-h4-0000"]
    with pytest.raises(SystemExit) as exc:
        demo.derive_panel(dropped)
    assert "does not satisfy its declared selection rule" in str(exc.value)


def test_derive_panel_refuses_a_condition_with_no_committed_fault(dev_specs):
    thinned = []
    for s in dev_specs:
        if s["task_id"] == "dev-fulfillment-h8-0225":
            s = json.loads(json.dumps(s))
            s["spec_row"]["faults"] = []
            # keep the fault CLASS visible to the selection rule, so the failure
            # under test is the condition check and not the rule check
            s["spec_row"]["faults"] = [{"fault_type": "rate_limit", "params": {},
                                        "target_node": "n8"}]
            thinned.append(s)
            continue
        thinned.append(s)
    # the rule still resolves; now remove the fault the CONDITION needs
    for s in thinned:
        if s["task_id"] == "dev-fulfillment-h8-0225":
            s["spec_row"]["faults"] = []
    with pytest.raises(SystemExit) as exc:
        demo.derive_panel(thinned)
    assert "committed faults" in str(exc.value) or "selection rule" in str(exc.value)


def test_frozen_prompt_is_pinned_by_hash(tmp_path):
    prompt, sha = demo.frozen_prompt(ROOT)
    assert sha == demo.FROZEN_PROMPT_SHA256
    assert prompt.strip() == prompt and prompt
    fake = tmp_path / demo.FROZEN_PROMPT
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("not the frozen winner\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        demo.frozen_prompt(tmp_path)
    assert "frozen winner" in str(exc.value)


# ---------------------------------------------------------------------------
# the printer really shows the machinery
# ---------------------------------------------------------------------------

def _run_scripted(spec: dict, condition: str, *, recovering: bool):
    policy = H.ScriptedOracle(spec, abandon_on_error=not recovering)
    return evaluate.run_episode(
        spec, arm="BP", condition=condition, control="none", secret=H.SECRET,
        fault_seed=evaluate.DEFAULT_FAULT_SEED, system_prompt="frozen prompt",
        prompt_meta={"path": demo.FROZEN_PROMPT,
                     "sha256": demo.FROZEN_PROMPT_SHA256},
        chat_fn=policy,
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 2786983945,
                "max_tokens": 1024, "enable_thinking": False},
        run_meta={"git_sha": "test", "server_model": "m",
                  "base_id": demo.BASE_MODEL, "adapter": None, "demo": True})


@pytest.fixture(scope="module")
def faulted_episode(panel):
    row, spec = next((r, s) for r, s in panel if r["condition"] == "faulted")
    trace = _run_scripted(spec, row["condition"], recovering=True)
    return row, trace


def test_injected_fault_envelope_and_recovery_are_printed(faulted_episode, capsys):
    row, trace = faulted_episode
    fired = [e for e in trace["events"] if e["fault_triggered"]]
    assert fired, "the scripted oracle must have reached the fault's target node"
    assert fired[0]["fault_type"] == "rate_limit"
    assert fired[0]["recovery_token"]

    demo.print_transcript(trace)
    demo.print_verdict(trace, row)
    out = capsys.readouterr().out

    # the fault, as the model saw it
    assert "FAULT INJECTED HERE: class=rate_limit" in out
    assert fired[0]["recovery_token"] in out, "the emitted token must be visible"
    assert "later_decision_required" in out, \
        "the registered remediation for rate_limit must be stated"
    # the remedial call and the verdict on it
    assert "recovery_token echoed by the model" in out
    assert "fault report: rate_limit@" in out
    assert "recovery_attempted=" in out and "recovery_success=" in out
    assert "certified_success =" in out
    # the exact tool calls and the observation bytes, not a summary
    assert "-> CALL " in out
    assert "receipt: " in out


def test_a_failed_recovery_is_printed_as_a_failure(panel, capsys):
    """The abandoning policy must read as a NON-recovery, with the reason."""
    row, spec = next((r, s) for r, s in panel if r["condition"] == "faulted")
    trace = _run_scripted(spec, row["condition"], recovering=False)
    assert trace["verdict"]["certified_success"] is False
    demo.print_verdict(trace, row)
    out = capsys.readouterr().out
    assert "certified_success = False" in out
    assert "recovered=False" in out
    assert trace["verdict"]["recovery_reason"] in out


@pytest.mark.parametrize("certified", [True, False])
def test_deep_horizon_cliff_is_printed_pass_or_fail(certified, capsys):
    """Whether the single H14 episode passes or fails, the measured aggregate
    cliff is printed next to it: no substitution to manufacture a failure, no
    omission to hide one."""
    row = next(r for r in demo.PANEL
               if r["family"] == "fulfillment" and r["horizon"] >= 14)
    trace = {"verdict": {"certified_success": certified, "fault_reports": [],
                         "fault_assigned": 0, "reasons": []},
             "score": {}, "runner": {"termination_reason": "answered"}}
    demo.print_verdict(trace, row)
    out = capsys.readouterr().out
    assert "Descriptive only; not a preregistered claim" in out
    assert "59/1,152 (5.1%)" in out
    assert "no task was substituted" in out


def test_demo_refuses_to_mint_the_run_secret(tmp_path):
    """`load_or_create_secret` CREATES a secret when the file is absent, and the
    next study stage would adopt it. A demonstration may not author that."""
    ns = argparse.Namespace(
        server="http://127.0.0.1:59999", model=None, base_id=demo.BASE_MODEL,
        adapter=None, secret_file="out/agentic/does-not-exist.hex",
        episode_wall_s=1.0, root=str(ROOT))
    with pytest.raises(SystemExit) as exc:
        demo.run(ns)
    assert "will not MINT the run secret" in str(exc.value)
    assert not (ROOT / "out/agentic/does-not-exist.hex").exists()


def test_banner_states_what_is_not_demonstrated():
    banner = demo.BANNER
    for required in ("NOT a benchmark", "NOT held-out evidence",
                     "NOT reliable execution beyond H8",
                     "NOT adapter superiority",
                     "NOT recovery from network outages",
                     "NOT a 100% rate"):
        assert required in banner, required


# ---------------------------------------------------------------------------
# the serving document must not drift from the served engine
# ---------------------------------------------------------------------------

def test_serving_doc_carries_the_registered_engine_contract():
    doc = (ROOT / "docs" / "SERVING.md").read_text(encoding="utf-8")
    engine = configio.engine_contract()
    for needle in (str(engine["max_model_len"]),
                   str(engine["gpu_memory_utilization"]),
                   str(engine["max_num_seqs"]),
                   str(engine["max_num_batched_tokens"]),
                   "bfloat16",
                   "--tool-call-parser qwen3_coder",
                   "--reasoning-parser qwen3",
                   '{"enable_thinking":false}',
                   '{"image":0,"video":0}',
                   "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                   "CUDA_VISIBLE_DEVICES=0",
                   "EXPECT_GPU=A5000",
                   demo.FROZEN_PROMPT_SHA256,
                   "--max-lora-rank 32",
                   "out/multiface/rssft-lora",
                   "scripts/demo_agentic.py"):
        assert needle in doc, needle
    assert engine["enforce_eager"] is False
    assert engine["enable_thinking"] is False
    assert "no `--enforce-eager`" in doc
    assert "thinking is disabled" in doc.lower()


def test_serving_doc_serves_both_ids_and_requires_the_system_prompt():
    doc = (ROOT / "docs" / "SERVING.md").read_text(encoding="utf-8")
    assert "bash scripts/serve.sh Qwen/Qwen3.5-4B" in doc
    assert "--lora-modules trained=out/multiface/rssft-lora" in doc
    assert "shipped BP configuration" in doc
    assert "<exact contents of p2_plan_state_act.txt>" in doc
