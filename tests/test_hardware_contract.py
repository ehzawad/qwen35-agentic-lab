"""The single-card A5000 contract: engine settings, thinking, provenance, ledger.

Every test here is CPU-only, starts no server, and never probes a card: the
fingerprint helpers are written so that an empty CUDA_VISIBLE_DEVICES means "no
GPU information", not "go and look at whatever is installed".

What is being defended:

  * ONE copy of the engine contract. Two copies is how a study silently compares
    two engines, and the analyzer's copy in configs/agentic_preregister.json is
    cross-checked against the operational one here.
  * thinking DISABLED over the wire. This checkpoint thinks by default and the
    offline sampler renders with thinking off, so an HTTP request that omits
    `chat_template_kwargs` runs a different policy and reads as "the model never
    committed an answer".
  * provenance on the ROW. A trace or ledger row that cannot say which card and
    which engine produced it cannot support a same-card claim (S19).
  * refusals that cost nothing to check: a foreign-card resume, a second physical
    GPU in one run, a base-model variance probe, a 45,056 MiB free-memory guard
    on a 24 GB card.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import types

import pytest

from agentlab.suite import configio

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "run_multifaceted_chain.sh"
SERVE = REPO / "scripts" / "serve.sh"
PREREG = json.loads((REPO / "configs" / "agentic_preregister.json").read_text())

CERTSPECS = REPO / "data" / "suite" / "v1" / "certspecs"


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Nothing in this file may probe a card, pinned or otherwise."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    configio.gpu_fingerprint.__wrapped__ if hasattr(
        configio.gpu_fingerprint, "__wrapped__") else None


def _cfg(tmp_path, **overrides):
    """A deep copy of the real config with the ledger and lock redirected.

    The real results/agentic/gpu_ledger.jsonl is the run's accounting record and
    must never be written by a test.
    """
    cfg = copy.deepcopy(configio.load_config())
    cfg["budget"]["ledger"] = str(tmp_path / "gpu_ledger.jsonl")
    cfg["hardware"]["lock"] = str(tmp_path / "hardware.json")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


# ---------------------------------------------------------------------------
# one engine contract, read from one place
# ---------------------------------------------------------------------------

def test_engine_contract_is_the_registered_one():
    c = configio.engine_contract()
    assert c == {"dtype": "bfloat16", "gpu_memory_utilization": 0.80,
                 "max_model_len": 8192, "max_num_seqs": 8,
                 "max_num_batched_tokens": 8192, "enforce_eager": False,
                 "enable_thinking": False, "multimodal_inputs": "REJECTED",
                 "tensor_parallel_size": 1}


def test_the_analyzer_copy_and_the_operational_copy_agree():
    """The seam: the analyzer reads machine.engine_contract, stages read `engine:`.

    They are two files by necessity (one is frozen, one is operational), so the
    only thing that keeps them one contract is this check.
    """
    registered = PREREG["machine"]["engine_contract"]
    ours = configio.engine_contract()
    for key, value in ours.items():
        assert registered[key] == value, key


def test_a_missing_or_mistyped_engine_key_refuses(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["engine"]["max_num_seqs"]
    with pytest.raises(SystemExit, match="max_num_seqs"):
        configio.engine_contract(cfg)
    cfg = _cfg(tmp_path)
    cfg["engine"]["max_model_len"] = "8192"
    with pytest.raises(SystemExit, match="max_model_len"):
        configio.engine_contract(cfg)


def test_engine_fingerprint_carries_versions_and_the_contract():
    fp = configio.engine_fingerprint()
    for key in configio.ENGINE_KEYS:
        assert key in fp
    for pkg in ("vllm", "torch", "transformers"):
        assert fp[pkg], pkg
    # identical twice: S19 treats two engine fingerprints in one trace set as a BUG
    assert fp == configio.engine_fingerprint()


def test_utilization_is_080_not_085_and_not_compensated_upward():
    # 0.8725 would "recover" the KV the drop from 0.85 gives up, which is exactly
    # the compensation the pivot forbids.
    assert configio.engine_contract()["gpu_memory_utilization"] == 0.80


def test_serve_sh_reads_the_contract_and_drops_the_32768_default():
    text = SERVE.read_text(encoding="utf-8")
    assert "32768" not in text
    assert "engine_contract" in text
    assert '--default-chat-template-kwargs' in text
    assert '"enable_thinking"' in text
    assert '--limit-mm-per-prompt' in text and '"image":0' in text
    assert "--max-num-seqs" in text and "--max-num-batched-tokens" in text
    # no literal engine numbers: they would be a second source of truth
    assert "0.85" not in text and "--gpu-memory-utilization 0.8" not in text


def test_serve_sh_refuses_an_off_contract_context():
    r = subprocess.run(["bash", str(SERVE)], cwd=REPO, capture_output=True,
                       text=True, timeout=120,
                       env=dict(os.environ, MAXLEN="32768",
                                CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 2
    assert "REFUSED" in r.stderr and "8192" in r.stderr


def test_no_stage_keeps_its_own_engine_knobs():
    """--gpu-frac / --max-model-len / --enforce-eager are gone from the stages."""
    for rel in ("src/agentlab/multidistill.py", "src/agentlab/variance.py",
                "src/agentlab/prompt_control.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        for knob in ('"--gpu-frac"', '"--max-model-len"', '"--enforce-eager"'):
            assert knob not in text, f"{rel} still offers {knob}"
        assert "engine_contract" in text or "_vllm_engine" in text


def test_multimodal_input_is_refused_not_ignored():
    text = [{"role": "user", "content": "how many grams?"}]
    configio.reject_multimodal(text)          # text is fine
    with pytest.raises(SystemExit, match="REJECTS"):
        configio.reject_multimodal([
            {"role": "user",
             "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


# ---------------------------------------------------------------------------
# thinking, over the wire
# ---------------------------------------------------------------------------

def test_the_http_evaluator_disables_thinking_in_every_request(monkeypatch):
    import requests

    from agentlab.suite import evaluate

    seen = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "done"}}]}

    def fake_post(url, json=None, timeout=None):
        seen.update(json or {})
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    chat = evaluate.make_http_chat("http://127.0.0.1:1", "Qwen/Qwen3.5-4B",
                                   {"temperature": 0.0, "top_p": 1.0, "seed": 1,
                                    "max_tokens": 1024, "enable_thinking": False})
    chat([{"role": "user", "content": "hi"}], [])
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["temperature"] == 0.0 and seen["max_tokens"] == 1024


def test_the_eval_decoding_block_is_the_frozen_one():
    """The evaluator's defaults now come from the config, so they must match."""
    ours = configio.load_config()["eval_decoding"]
    frozen = PREREG["decoding"]
    for key in ("temperature", "top_p", "seed", "max_tokens_per_decision",
                "enable_thinking"):
        assert ours[key] == frozen[key], key
    assert ours["concurrency"] == configio.engine_contract()["max_num_seqs"]


def test_the_offline_renderer_and_the_server_agree_on_thinking():
    """Both paths take thinking from the same contract key."""
    assert configio.engine_contract()["enable_thinking"] is False
    text = (REPO / "src" / "agentlab" / "multidistill.py").read_text(encoding="utf-8")
    assert 'thinking = contract["enable_thinking"]' in text


# ---------------------------------------------------------------------------
# per-trace provenance and the resume guard
# ---------------------------------------------------------------------------

def _shard_args(tmp_path, specs, **kw):
    args = types.SimpleNamespace(
        specs=str(specs), control="none", permutation_seed=7, num_shards=1,
        shard=0, out=str(tmp_path / "traces"), limit=2,
        secret_file=str(tmp_path / "secret.hex"),
        prompt=str(REPO / "prompts" / "agentic" / "p1_minimal.txt"),
        arm="BP", condition="clean", model="Qwen/Qwen3.5-4B",
        base_id="Qwen/Qwen3.5-4B", adapter=None, served_adapter_name="trained",
        run_id="agentic-test", decode_seed=0xA61E0009, fault_seed=0xA61E0007,
        max_tokens=1024, concurrency=2, time_budget_s=60.0,
        episode_wall_s=30.0, server="http://127.0.0.1:1")
    for key, value in kw.items():
        setattr(args, key, value)
    return args


@pytest.mark.skipif(not (CERTSPECS / "dev.jsonl").exists(),
                    reason="run scripts/export_eval_specs.py first")
def test_every_trace_row_carries_the_frozen_fingerprint(tmp_path):
    from agentlab.suite import evaluate

    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl")
    status = evaluate.run_shard(args, chat_fn=lambda m, t: {"content": "\\boxed{1}",
                                                            "tool_calls": []})
    rows = [json.loads(l) for l in
            pathlib.Path(status["out"]).read_text().splitlines() if l.strip()]
    assert rows
    for row in rows:
        prov = row["provenance"]
        for field in configio.FINGERPRINT_FIELDS:
            assert field in prov, field
        assert prov["run_id"] == "agentic-test"
        assert prov["enable_thinking_effective"] is False
        assert prov["engine_fingerprint"] == configio.engine_fingerprint()
        assert prov["config_hash"] == configio.config_hash()
    assert status["enable_thinking_effective"] is False


@pytest.mark.skipif(not (CERTSPECS / "dev.jsonl").exists(),
                    reason="run scripts/export_eval_specs.py first")
def test_resume_refuses_a_trace_file_produced_on_another_card(tmp_path):
    from agentlab.suite import evaluate

    out = tmp_path / "traces"
    out.mkdir()
    foreign = configio.fingerprint("agentic-test")
    foreign.update(gpu_name="NVIDIA RTX A6000", gpu_uuid="GPU-not-this-card",
                   cuda_visible_bytes=51522830336, driver_version="610.43.02")
    (out / "BP.clean.none.jsonl").write_text(
        json.dumps({"kind": "episode", "task_id": "x", "provenance": foreign}) + "\n",
        encoding="utf-8")
    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl")
    with pytest.raises(SystemExit, match="different runtime fingerprint"):
        evaluate.run_shard(args, chat_fn=lambda m, t: {"content": "", "tool_calls": []})


def test_a_pre_fingerprint_row_does_not_block_resume(tmp_path):
    from agentlab.suite import evaluate

    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"kind": "episode", "task_id": "x",
                             "provenance": {"git_sha": "abc"}}) + "\n",
                 encoding="utf-8")
    assert evaluate.require_same_fingerprint(p, configio.fingerprint()) == 1


def test_a_trained_arm_must_request_the_served_adapter_alias(tmp_path):
    from agentlab.suite import evaluate

    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl",
                       adapter="out/multiface/rssft-lora", served_adapter_name="")
    with pytest.raises(SystemExit, match="evaluates the BASE"):
        evaluate.run_shard(args, chat_fn=lambda m, t: {"content": "", "tool_calls": []})


# ---------------------------------------------------------------------------
# the ledger is a receipt
# ---------------------------------------------------------------------------

def test_ledger_rows_carry_timestamps_gpu_identity_git_sha_and_work(tmp_path):
    cfg = _cfg(tmp_path)
    configio.ledger_append("eval", 12.5, cfg, kind="server_session",
                           work={"unit": "episodes", "count": 40})
    rows = configio.ledger_rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    for field in configio.FINGERPRINT_FIELDS:
        assert field in row, field
    assert row["kind"] == "server_session"
    assert row["work"] == {"unit": "episodes", "count": 40}
    assert row["started_at_utc"] and row["ended_at_utc"]
    assert row["minutes"] == 12.5
    assert configio.ledger_hours(cfg) == pytest.approx(12.5 / 60.0)


def test_a_second_physical_card_in_one_run_is_fatal_to_the_ledger(tmp_path):
    cfg = _cfg(tmp_path)
    pathlib.Path(cfg["budget"]["ledger"]).write_text(
        json.dumps({"stage": "distill", "minutes": 3.0,
                    "gpu_uuid": "GPU-first-card"}) + "\n", encoding="utf-8")
    pathlib.Path(cfg["hardware"]["lock"]).write_text(
        json.dumps({"gpu_uuid": "GPU-second-card", "gpu_name": "NVIDIA RTX A5000",
                    "cuda_visible_bytes": 25282805760,
                    "driver_version": "610.43.02"}), encoding="utf-8")
    assert configio.ledger_bound_uuid(cfg) == "GPU-first-card"
    with pytest.raises(SystemExit, match="two physical cards"):
        configio.ledger_append("eval", 1.0, cfg)


def test_the_ledger_ceiling_is_the_registered_120_hours():
    assert float(configio.load_config()["budget"]["gpu_hours_ceiling"]) == 120.0


def test_the_startup_ledger_records_the_persistent_engine_saving():
    starts = configio.load_config()["budget"]["engine_starts"]
    assert starts["registered"] == 6 and starts["pre_pivot"] == 85
    per = configio.engine_startup_s() / 3600.0
    assert starts["hours_each"] == pytest.approx(per, abs=5e-5)
    saved = (starts["pre_pivot"] - starts["registered"]) * per
    assert starts["hours_saved"] == pytest.approx(saved, abs=0.01)


# ---------------------------------------------------------------------------
# the chain driver
# ---------------------------------------------------------------------------

def test_the_free_memory_guard_fits_a_24_gb_card():
    text = CHAIN.read_text(encoding="utf-8")
    assert "45056" not in text, "a 48 GB guard can never pass on this card"
    assert configio.hardware_contract()["min_free_mib"] == 23500


def test_the_chain_pins_the_registered_card():
    text = CHAIN.read_text(encoding="utf-8")
    assert 'EXPECT_GPU="${EXPECT_GPU:-A5000}"' in text
    assert "PCI_BUS_ID" in text
    assert "EXPECT_GPU=A6000" not in text
    hw = configio.hardware_contract()
    assert hw["registered_gpu"] == "A5000"
    assert hw["cuda_visible_devices"] == "0"
    assert hw["cuda_visible_bytes"] == 25282805760


def test_one_engine_per_stage_not_one_per_shard():
    text = CHAIN.read_text(encoding="utf-8")
    # the per-shard budget-minutes loops are gone
    assert "--budget-minutes 7" not in text
    assert "session_begin" in text and "session_end" in text
    # and the modules expose the persistent loop
    from agentlab.multidistill import run_units

    calls = []
    done = set()

    def run_one(unit):
        calls.append(unit)
        done.add(unit)

    status = run_units(None, [1, 2, 3], run_one=run_one,
                       is_done=lambda u: u in done, budget_minutes=10.0,
                       label="t", work_unit="unit")
    assert calls == [1, 2, 3] and status["complete"] is True
    # already-done units are skipped, so re-invoking is the resume mechanism
    calls.clear()
    status = run_units(None, [1, 2, 3], run_one=run_one,
                       is_done=lambda u: u in done, budget_minutes=10.0, label="t")
    assert calls == [] and status["units_done"] == 0


def test_the_eval_stage_runs_the_registered_mandatory_census():
    """MT and H8 were registered and never invoked; the controls used core specs."""
    r = subprocess.run(["bash", "-c",
                        f'source <(sed -n "/^eval_units()/,/^}}/p" {CHAIN}); '
                        f'PY={REPO}/.venv/bin/python; eval_units'],
                       cwd=REPO, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONPATH="src",
                                CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr
    units = [ln.split() for ln in r.stdout.splitlines() if ln.strip()]
    cfg = configio.load_config()
    sizes = {m["name"]: m["tasks"] for m in cfg["eval"]["manifests"]}
    mandatory = [u for u in units if u[5] == "True"]
    episodes = sum(sizes[u[1]] for u in mandatory)
    assert episodes == cfg["eval"]["mandatory_episodes"] == 7800
    names = {(u[0], u[1], u[3], u[4]) for u in mandatory}
    for arm in ("BP", "TP"):
        assert (arm, "core", "clean", "none") in names
        assert (arm, "core", "faulted", "none") in names
        assert (arm, "mt", "clean", "none") in names
        assert (arm, "h8", "clean", "none") in names
        assert (arm, "absent", "clean", "redacted") in names
        assert (arm, "perm", "clean", "permuted") in names
    # the controls run against their OWN manifests, not the core one
    for arm, name, specs, condition, control, _m in units:
        if control == "redacted":
            assert specs == "eval_absent.jsonl"
        if control == "permuted":
            assert specs == "eval_perm.jsonl"
    # R0/RP are absent by design, never scheduled
    assert not [u for u in units if u[0] in ("R0", "RP")]
    # the descriptive arms are scheduled after every mandatory unit
    last_mandatory = max(i for i, u in enumerate(units) if u[5] == "True")
    first_descriptive = min(i for i, u in enumerate(units) if u[0] in ("B0", "T0"))
    assert first_descriptive > last_mandatory


def test_the_lock_stage_requires_the_grpo_disposition_and_names_rs_sft():
    text = CHAIN.read_text(encoding="utf-8")
    assert "grpo.disposition_artifact" in text
    # the label is READ from the registered config, never retyped in the driver,
    # and the receipt itself is written/checked by the receipts script
    assert "cfg_get grpo.stage_disposition" in text
    assert "agentic_locks.py grpo-disposition" in text
    assert "grpo-disposition --verify" in text
    # the silent "prefer whichever adapter exists" fallback is gone
    assert 'stage_name="grpo"' not in text
    assert 'candidate="$RSSFT" stage_name="rs_sft"' in text
    assert "finalize-prereg" in text and "--require-finalization" in text


def test_the_disposition_receipt_carries_every_required_field(tmp_path):
    """The artifact the analyzer requires whenever no GRPO checkpoint exists."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "agentic_locks", REPO / "scripts" / "agentic_locks.py")
    locks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(locks)
    cfg = configio.load_config()
    payload = locks.grpo_disposition_payload(cfg)
    reg = PREREG["machine"]["grpo_disposition"]
    for field in reg["required_artifact_fields"]:
        assert field in payload, field
    assert payload["branch"] == reg["expected_branch"]
    infeasible = reg["hardware_infeasible"]
    assert payload["outcome"] == infeasible["outcome"]
    assert payload["variance_gate"] == infeasible["variance_gate"]
    assert payload["trainer_feasibility"] == infeasible["trainer_feasibility"]
    assert payload["optimizer_steps"] == infeasible["optimizer_steps"]
    assert payload["gpu_name"] == PREREG["machine"]["hardware_integrity"][
        "expected_gpu_name"]
    assert payload["cuda_visible_bytes"] == PREREG["machine"][
        "hardware_integrity"]["expected_cuda_visible_bytes"]
    for field in infeasible["arithmetic_fields"]:
        assert field in payload["arithmetic"], field
    assert (payload["arithmetic"]["vllm_allocation_gib"]
            < payload["arithmetic"]["vllm_policy_copy_gib"])
    assert payload["not_interchangeable_with"] == "GRPO_NOT_RUN_VARIANCE_GATE_CLOSED"
    for forbidden in reg["forbidden_substitutions"]:
        assert forbidden in payload["forbidden_substitutions"], forbidden

    # and a mislabelled artifact is refused rather than accepted
    p = tmp_path / "grpo_disposition.json"
    bad = dict(payload, variance_gate="GRPO_NOT_RUN_VARIANCE_GATE_CLOSED")
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit, match="144-group probe"):
        locks.require_grpo_disposition(cfg, p)
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert locks.require_grpo_disposition(cfg, p)["outcome"] == payload["outcome"]
    with pytest.raises(SystemExit, match="not a silent fallback"):
        locks.require_grpo_disposition(cfg, tmp_path / "absent.json")


def test_the_probe_records_not_evaluated_rather_than_a_missing_artifact(
        tmp_path, monkeypatch):
    from agentlab import variance

    monkeypatch.setattr(variance, "REPORT_PATH", tmp_path / "variance_report.json")
    out = variance.write_disposition_report()
    assert out["stage_disposition"] == "NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT"
    assert out["probe_run"] is False
    # gates and summary are explicitly null: there is no measurement to report,
    # and an empty-but-shaped gate block would read as "evaluated and failed"
    assert out["gates"] is None and out["summary"] is None
    assert out["rollouts"] == 0 and out["cells_expected"] == 12
    for field in configio.FINGERPRINT_FIELDS:
        assert field in out["provenance"], field
    assert json.loads((tmp_path / "variance_report.json").read_text())[
        "stage_disposition"] == "NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT"


def test_the_registered_dispositions_are_the_council_labels():
    cfg = configio.load_config()
    assert cfg["grpo"]["stage_disposition"] == "GRPO_NOT_RUN_HARDWARE_INFEASIBLE"
    assert (cfg["variance_probe"]["stage_disposition"]
            == "NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT")
    # a disposition is not a gate state
    assert "GRPO_NOT_RUN_HARDWARE_INFEASIBLE" not in PREREG["outcome_states"]
    reg = PREREG["machine"]["grpo_disposition"]
    assert cfg["grpo"]["stage_disposition"] in reg["allowed_dispositions"]
    assert reg["registered_disposition_for_this_run"] == cfg["grpo"]["stage_disposition"]
    assert cfg["grpo"]["disposition_artifact"] == reg["artifact_path"]
    arith = cfg["grpo"]["disposition_arithmetic"]
    # the arithmetic must SHOW the shortfall, or it establishes nothing
    assert arith["vllm_allocation_gib"] < arith["vllm_policy_copy_gib"]
    assert arith["vllm_allocation_gib"] == reg["hardware_infeasible"]["vllm_allocation_gib"]
    assert arith["vllm_policy_copy_gib"] == reg["hardware_infeasible"]["vllm_policy_copy_gib"]


# ---------------------------------------------------------------------------
# the pin itself
# ---------------------------------------------------------------------------

def test_expect_gpu_has_no_off_switch(monkeypatch):
    import importlib

    from agentlab import env as labenv

    assert labenv.REGISTERED_GPU == "A5000"
    assert labenv.EXPECT_GPU == "A5000"
    # "unset it to make the error go away" must not work any more
    monkeypatch.setenv("EXPECT_GPU", "")
    reloaded = importlib.reload(labenv)
    try:
        assert reloaded.EXPECT_GPU == "A5000"
    finally:
        monkeypatch.delenv("EXPECT_GPU", raising=False)
        importlib.reload(labenv)


def test_a_study_stage_refuses_fastest_first_ordering(monkeypatch):
    from agentlab import env as labenv

    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    with pytest.raises(SystemExit, match="PCI_BUS_ID"):
        labenv.require_pci_bus_order()
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    labenv.require_pci_bus_order()


def test_a_study_stage_refuses_an_unregistered_index(monkeypatch):
    from agentlab import env as labenv

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(SystemExit, match="SEPARATE registered run"):
        labenv.require_registered_index()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(SystemExit, match="explicit pin"):
        labenv.require_registered_index()


def test_no_card_is_probed_when_nothing_is_pinned(monkeypatch):
    """On a shared box, looking must cost nothing -- and must never touch a card
    this run does not own."""
    calls = []
    monkeypatch.setattr(configio.subprocess, "run",
                        lambda *a, **k: calls.append(a) or types.SimpleNamespace(
                            stdout=""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    row = configio.gpu_fingerprint(_cfg_no_lock())
    assert row == {k: None for k in configio.GPU_FIELDS}
    assert not calls


def _cfg_no_lock():
    cfg = copy.deepcopy(configio.load_config())
    cfg["hardware"]["lock"] = "/nonexistent/hardware.json"
    return cfg


# ---------------------------------------------------------------------------
# the variance probe: right policy, or none at all
# ---------------------------------------------------------------------------

def test_the_probe_refuses_while_it_is_short_circuited():
    from agentlab import variance

    with pytest.raises(SystemExit, match="NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT"):
        variance.refuse_if_short_circuited()


def test_the_probe_refuses_to_fall_back_to_the_base_model(tmp_path):
    from agentlab import variance

    args = types.SimpleNamespace(adapter=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="BASE"):
        variance.probe_adapter(args)


def test_the_probe_engine_would_load_the_adapter(tmp_path):
    from agentlab import variance

    adapter = tmp_path / "rssft-lora"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\0")
    args = types.SimpleNamespace(adapter=str(adapter))
    assert variance.probe_adapter(args) == str(adapter)
    text = (REPO / "src" / "agentlab" / "variance.py").read_text(encoding="utf-8")
    assert "_vllm_engine(cfg, args, frozen=None, adapter=adapter)" in text


def test_probe_rows_keep_hardware_provenance():
    from agentlab import variance

    row = {k: None for k in ("task_id", "family", "horizon", "split", "fault_types",
                             "sample_index", "prompt_variant", "final",
                             "decisions_used", "truncated", "exhausted",
                             "unknown_tool", "arg_error", "call_cap",
                             "milestone_fraction", "verdict", "replay_ok",
                             "replay_reason")}
    row["provenance"] = {"gpu_uuid": "GPU-x"}
    assert variance._slim(row)["provenance"] == {"gpu_uuid": "GPU-x"}


# ---------------------------------------------------------------------------
# the SFT contract
# ---------------------------------------------------------------------------

def test_the_sft_contract_is_the_registered_one():
    from agentlab import sft

    cfg = configio.load_config()["sft"]
    args = sft.build_parser(cfg).parse_args([])
    args.out = "/tmp/_t"
    kwargs = sft.sft_config_kwargs(args)
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["gradient_accumulation_steps"] == 8
    # the dangerous default: 8 x 2.03 GiB of logits does not fit on this card
    assert kwargs["per_device_eval_batch_size"] == 1
    assert kwargs["eval_accumulation_steps"] == 1
    assert kwargs["prediction_loss_only"] is True
    assert kwargs["max_length"] == 4096
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["packing"] is False
    assert kwargs["bf16"] is True


def test_lora_alpha_and_dropout_reach_the_trainer():
    from agentlab import sft
    from agentlab.peft_cfg import lora_config

    cfg = configio.load_config()["sft"]
    args = sft.build_parser(cfg).parse_args([])
    assert args.lora_alpha == cfg["lora_alpha"] == 64
    assert args.lora_dropout == cfg["lora_dropout"] == 0.05
    # and they are real flags, so a config edit cannot miss the trainer
    over = sft.build_parser(cfg).parse_args(["--lora-alpha", "16",
                                             "--lora-dropout", "0.1"])
    peft = lora_config(r=over.rank, alpha=over.lora_alpha, dropout=over.lora_dropout)
    assert peft.lora_alpha == 16 and peft.lora_dropout == 0.1


def test_the_sft_stage_passes_every_contract_flag():
    text = CHAIN.read_text(encoding="utf-8")
    for flag in ("--lora-alpha", "--lora-dropout", "--eval-bsz",
                 "--eval-accumulation-steps", "--prediction-loss-only",
                 "--gradient-checkpointing", "--packing"):
        assert flag in text, flag
