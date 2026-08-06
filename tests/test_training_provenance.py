"""The training-side provenance chain: rollout -> view -> trainer -> locked bytes.

The chain used to stop three times over. Rejection-sampling rows carried a
fingerprint but were allowed to carry `None` instead; the SFT view metadata
dropped provenance entirely; the trainer wrote no receipt at all; and the
checkpoint lock recorded a mutable PATH. So a locked "study candidate" could not
name the card, the engine, the corpus or the trajectories behind it, and nothing
detected an adapter whose bytes changed after the lock was taken.

These tests pin the repaired chain end to end, on CPU:

  * a row may never carry a null producer, and a CPU producer must SAY it is one
    (card identity explicitly null) rather than inheriting a card from an old
    hardware lock;
  * every SFT view carries its trajectory's producer snapshot, that trajectory's
    content digest and `environment_contract_sha256` -- and a view missing any of
    them is REFUSED, not trained on;
  * the trainer's receipt pins the checkpoint's byte digest, the corpus digests,
    the attested training hardware and the ledger's card;
  * the checkpoint lock refuses to exist without that receipt, and a PATH-ONLY
    lock (the old shape) is rejected rather than accepted as legacy.

Nothing here touches a GPU: every producer attestation is a fabricated but
structurally complete runtime manifest, which is exactly what the real seam
hands the consumers.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib

import pytest
from rollout_helpers import TEST_SECRET, run_engine, token_counter_stub

from agentlab import multidistill, variance
from agentlab.suite import configio
from agentlab.suite import contract as contract_mod
from agentlab.suite import datasets
from agentlab.suite.generate import build_task

REPO = pathlib.Path(__file__).resolve().parents[1]
LOCKS_SCRIPT = REPO / "scripts" / "agentic_locks.py"
CFG = configio.load_config()
SUITE = "agentlab-suite-v1"
SEED = 0xA61E0001          # the committed distill seed

FAKE_UUID = "GPU-00000000-1111-2222-3333-444444444444"
OTHER_UUID = "GPU-99999999-8888-7777-6666-555555555555"


# ---------------------------------------------------------------------------
# fixtures: a structurally complete producer attestation, without a card
# ---------------------------------------------------------------------------

def _manifest_record(*, stage: str, uuid: str = FAKE_UUID, adapter=None) -> dict:
    """Every registered manifest field, filled the way a real producer fills it."""
    contract = configio.engine_contract(CFG)
    rec = {
        "run_id": "provenance-test", "session_id": f"{stage}-test-1234-abcd",
        "stage": stage,
        "gpu_name": "NVIDIA RTX A5000", "gpu_uuid": uuid,
        "cuda_visible_bytes": int(CFG["hardware"]["cuda_visible_bytes"]),
        "driver_version": "560.35.03", "pci_bus_id": "00000000:01:00.0",
        "compute_capability": "8.6", "visible_ordinal": 0,
        "cuda_device_order": "PCI_BUS_ID", "cuda_visible_devices": "0",
        "git_sha": configio.git_sha(), "config_hash": configio.config_hash(),
        "engine_fingerprint": configio.engine_fingerprint(CFG),
        "enable_thinking_effective": bool(contract["enable_thinking"]),
        "model": "Qwen/Qwen3.5-4B", "adapter": adapter,
        "adapter_sha256": None, "served_adapter_name": None,
        "host": "testhost", "boot_id": None, "pid": 4321, "process_start": None,
        "port": None, "server_url": None,
        "captured_at_utc": configio.now_utc(), "ready_at_utc": configio.now_utc(),
    }
    return rec


def _producer(tmp_path, *, stage="rs_sft", uuid=FAKE_UUID):
    """(path, record) of a written, self-hashed producer runtime manifest."""
    path = tmp_path / "manifests" / f"{stage}.json"
    return configio.write_runtime_manifest(_manifest_record(stage=stage, uuid=uuid),
                                           path, CFG)


def _gpu_provenance(rec: dict) -> dict:
    """Exactly what `_vllm_engine` copies onto every row it produces."""
    fp = configio.fingerprint_from_manifest(rec, CFG)
    fp.update({multidistill.GPU_EXECUTION: True,
               "producer": rec["stage"],
               "runtime_manifest_sha256": rec[configio.MANIFEST_HASH_FIELD],
               "session_id": rec["session_id"],
               "producer_pid": rec["pid"],
               "adapter": rec["adapter"],
               "adapter_sha256": rec["adapter_sha256"],
               "served_model": rec["model"]})
    return fp


def _attested_records(uuid: str = FAKE_UUID, *, index: int = 11,
                      family: str = "typed_relay", horizon: int = 4,
                      faults=(("transient", False),), tmp_path=None):
    """Real scripted rollouts, relabelled with a GPU producer's snapshot.

    The rollout bytes come from the canonical runtime (so the views are real
    views); the provenance block is the one the attested vLLM engine would have
    copied from its manifest, so the chain under test is the real grammar.
    """
    bundle = build_task(SUITE, SEED, "distill", family, horizon, index,
                        list(faults) or None)
    records = run_engine([bundle], cfg=CFG, secret=TEST_SECRET)
    _path, manifest = _producer(tmp_path, stage="multidistill", uuid=uuid)
    prov = _gpu_provenance(manifest)
    for rec in records:
        rec["provenance"] = dict(prov)
    return records


def _cfg(tmp_path, *, ledger_uuid: str | None = FAKE_UUID) -> dict:
    """A hermetic config: its hardware lock and GPU ledger live under tmp_path."""
    cfg = copy.deepcopy(CFG)
    cfg["hardware"]["lock"] = str(tmp_path / "hardware.json")
    cfg["budget"]["ledger"] = str(tmp_path / "gpu_ledger.jsonl")
    if ledger_uuid:
        row = {"stage": "multidistill", "minutes": 1.0, "gpu_uuid": ledger_uuid}
        (tmp_path / "gpu_ledger.jsonl").write_text(json.dumps(row) + "\n",
                                                   encoding="utf-8")
    return cfg


def _checkpoint(tmp_path, *, name="rssft-lora", weights=b"\x01\x02\x03") -> pathlib.Path:
    out = tmp_path / "out" / name
    out.mkdir(parents=True)
    (out / "adapter_model.safetensors").write_bytes(weights)
    (out / "adapter_config.json").write_text('{"r": 32}\n', encoding="utf-8")
    return out


def _locks_module():
    spec = importlib.util.spec_from_file_location("agentic_locks_prov", LOCKS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. the producer end: no row may be unattributable
# ---------------------------------------------------------------------------

def test_a_scripted_engine_says_it_is_not_a_gpu_rather_than_saying_nothing():
    """`provenance: None` is gone; a CPU producer names itself and nulls the card.

    The council's rule for stages with no visible device: do not invent a GPU
    execution fingerprint, and do not let an old hardware lock make a CPU-only
    transformation look like a measured run.
    """
    engine = multidistill.RolloutEngine(CFG, lambda m, s: m, lambda p: [],
                                        secret=TEST_SECRET)
    prov = engine.provenance
    assert prov[multidistill.GPU_EXECUTION] is False
    assert prov["producer"] == "scripted-cpu-policy"
    for field in ("gpu_name", "gpu_uuid", "cuda_visible_bytes", "driver_version"):
        assert prov[field] is None, field
    # the frozen S19 field set is still complete as a KEY set, so one reader
    # handles GPU and CPU blocks alike
    for field in configio.FINGERPRINT_FIELDS:
        assert field in prov, field
    assert multidistill.provenance_gaps(prov) == []


@pytest.mark.parametrize("block,expect", [
    (None, "provenance_absent"),
    ({}, "provenance_absent"),
    ({"run_id": "r", "gpu_uuid": "GPU-x"}, "gpu_execution"),
])
def test_a_row_with_no_producer_is_refused_at_the_write(block, expect):
    assert expect in multidistill.provenance_gaps(block)
    with pytest.raises(SystemExit, match="REFUSED"):
        multidistill.require_row_provenance(block, "a row")


def test_a_half_recorded_gpu_producer_is_refused(tmp_path):
    _path, manifest = _producer(tmp_path, stage="multidistill")
    prov = _gpu_provenance(manifest)
    assert multidistill.provenance_gaps(prov) == []

    # a null card identity is the exact D1 shape: it reads as "unattributed"
    holed = dict(prov, gpu_uuid=None)
    assert "gpu_uuid" in multidistill.provenance_gaps(holed)
    with pytest.raises(SystemExit, match="COMPLETE S19 fingerprint"):
        multidistill.require_row_provenance(holed, "a row")

    # and a GPU block that cannot point at its session attestation is not one
    for field in ("runtime_manifest_sha256", "session_id"):
        assert field in multidistill.provenance_gaps(dict(prov, **{field: None}))


def test_a_cpu_attestation_may_not_carry_a_card(tmp_path):
    """The inverse failure: a CPU transformation borrowing the run's GPU lock."""
    cpu = multidistill.cpu_provenance("cpu-transformation", CFG)
    assert multidistill.provenance_gaps(cpu) == []
    borrowed = dict(cpu, gpu_uuid=FAKE_UUID, gpu_name="NVIDIA RTX A5000")
    gaps = multidistill.provenance_gaps(borrowed)
    assert "gpu_uuid_on_a_cpu_attestation" in gaps
    with pytest.raises(SystemExit, match="REFUSED"):
        multidistill.require_row_provenance(borrowed, "a CPU row")


def test_rollout_rows_carry_the_engine_snapshot(tmp_path):
    records = _attested_records(tmp_path=tmp_path)
    for rec in records:
        assert multidistill.provenance_gaps(rec["provenance"]) == []
        assert rec["provenance"]["gpu_uuid"] == FAKE_UUID
        assert contract_mod.is_current(rec)


def test_one_corpus_may_not_mix_two_producers(tmp_path):
    a = _attested_records(FAKE_UUID, tmp_path=tmp_path / "a")
    b = _attested_records(OTHER_UUID, index=12, tmp_path=tmp_path / "b")
    assert len(multidistill.distinct_producers(a + b)) == 2
    with pytest.raises(SystemExit, match="FATAL"):
        multidistill.require_one_producer(a + b, "the accepted RS corpus")
    # two SESSIONS of one run on one card are the resumable design and must pool
    c = _attested_records(FAKE_UUID, index=13, tmp_path=tmp_path / "c")
    assert len(multidistill.distinct_producers(a + c)) == 1


def test_an_unattributable_shard_is_never_written(tmp_path):
    records = _attested_records(tmp_path=tmp_path)
    good = tmp_path / "shard-0000.jsonl"
    multidistill.write_attested_jsonl(good, records, "a shard")
    assert good.exists()

    stripped = [dict(r, provenance=None) for r in records]
    bad = tmp_path / "shard-0001.jsonl"
    with pytest.raises(SystemExit, match="REFUSED"):
        multidistill.write_attested_jsonl(bad, stripped, "a shard")
    assert not bad.exists(), "a refused write must leave nothing behind"


def test_acceptance_drops_a_row_that_cannot_say_what_produced_it(tmp_path):
    rec = _attested_records(tmp_path=tmp_path)[0]
    ok, why = multidistill.accept_record(dict(rec, provenance=None), CFG,
                                         skip_replay=True, secret=TEST_SECRET)
    assert ok is False and why.startswith("missing_provenance")


# ---------------------------------------------------------------------------
# 2. the SFT views: every view carries its trajectory's provenance
# ---------------------------------------------------------------------------

def _views(tmp_path, records=None):
    records = records or _attested_records(tmp_path=tmp_path)
    rows, meta, report = datasets.build_views(records, token_counter_stub(), CFG)
    assert rows and meta
    return records, rows, meta, report


def test_every_sft_view_carries_the_provenance_of_its_trajectory(tmp_path):
    records, rows, meta, report = _views(tmp_path)
    by_task = {r["task_id"]: r for r in records}
    assert len(meta) == len(rows)
    for i, m in enumerate(meta):
        source = by_task[m["task_id"]]
        assert m["source_provenance"] == source["provenance"], i
        assert m["source_row_sha256"] == multidistill.row_digest(source)
        assert m[contract_mod.STAMP_FIELD] == \
            contract_mod.environment_contract_sha256()
        assert m["runtime_manifest_sha256"] == \
            source["provenance"]["runtime_manifest_sha256"]
        assert m["session_id"] == source["provenance"]["session_id"]
        assert m["gpu_execution"] is True
        datasets.require_view_chain(m, f"view {i}")
    # the report summarizes the same chain
    assert report["source_provenance"]["gpu_uuid"] == FAKE_UUID
    assert report["gpu_execution"] is True
    assert report["source_trajectories"] == len(records)
    assert report["source_runtime_manifests"] == \
        [records[0]["provenance"]["runtime_manifest_sha256"]]


def test_the_four_trl_columns_are_unchanged_by_the_chain(tmp_path):
    """Provenance goes to the metadata: the trainer must see no stray columns."""
    _records, rows, _meta, _report = _views(tmp_path)
    for row in rows:
        assert set(row) == {"prompt", "completion", "tools", "chat_template_kwargs"}
        assert len(row["completion"]) == 1
        assert row["completion"][0]["role"] == "assistant"


def test_a_contract_less_view_is_rejected(tmp_path):
    """THE required refusal: a view with no environment contract is not trainable.

    Three shapes, one rule. A view that cannot name the model-visible environment
    it was built under, the trajectory it came from, or the producer of that
    trajectory, is refused -- never accepted as an unlabelled legacy row.
    """
    _records, rows, meta, report = _views(tmp_path)

    contractless = dict(meta[0])
    contractless.pop(contract_mod.STAMP_FIELD)
    with pytest.raises(SystemExit, match=contract_mod.STAMP_FIELD):
        datasets.require_view_chain(contractless, "a contract-less view")

    stale = dict(meta[0], **{contract_mod.STAMP_FIELD: "0" * 64})
    with pytest.raises(SystemExit, match="REFUSED"):
        datasets.require_view_chain(stale, "a stale view")

    sourceless = dict(meta[0])
    sourceless["source_provenance"] = None
    with pytest.raises(SystemExit, match="source_provenance"):
        datasets.require_view_chain(sourceless, "a sourceless view")

    unattributed = dict(meta[0])
    unattributed["source_provenance"] = dict(meta[0]["source_provenance"],
                                             gpu_uuid=None)
    with pytest.raises(SystemExit, match="not evidence"):
        datasets.require_view_chain(unattributed, "an unattributed view")

    # and the corpus-level gate refuses the whole build, not merely one row
    with pytest.raises(SystemExit, match=contract_mod.STAMP_FIELD):
        datasets.require_views_chain(rows, [contractless] + list(meta[1:]), report)


def test_a_trajectory_with_no_provenance_never_becomes_a_view(tmp_path):
    records = _attested_records(tmp_path=tmp_path)
    stripped = [dict(r, provenance=None) for r in records]
    rows, meta, report = datasets.build_views(stripped, token_counter_stub(), CFG)
    assert rows == [] and meta == []
    assert report["rejected"]["missing_source_provenance"] == len(records)


def test_the_view_metadata_must_be_one_to_one_with_the_training_rows(tmp_path):
    _records, rows, meta, report = _views(tmp_path)
    datasets.require_views_chain(rows, meta, report)

    with pytest.raises(SystemExit, match="one-to-one"):
        datasets.require_views_chain(rows, meta[:-1], report)
    duped = list(meta[:-1]) + [dict(meta[-1], row_id=meta[0]["row_id"])]
    with pytest.raises(SystemExit, match="row_id"):
        datasets.require_views_chain(rows, duped, report)
    # a report that describes another build is refused too
    with pytest.raises(SystemExit, match="another build"):
        datasets.require_views_chain(rows, meta, dict(report, rows=len(rows) + 1))


def test_a_cpu_built_corpus_is_a_fixture_not_a_training_corpus(tmp_path):
    """A scripted corpus is honest, and still not something a candidate is trained on."""
    bundle = build_task(SUITE, SEED, "distill", "typed_relay", 4, 21, None)
    records = run_engine([bundle], cfg=CFG, secret=TEST_SECRET)
    rows, meta, report = datasets.build_views(records, token_counter_stub(), CFG)
    assert meta and all(m["gpu_execution"] is False for m in meta)
    # the chain itself is intact ...
    datasets.require_views_chain(rows, meta, report, require_gpu_source=False)
    # ... but the trainer's gate says out loud that no card attested it
    with pytest.raises(SystemExit, match="no card attested"):
        datasets.require_views_chain(rows, meta, report)


# ---------------------------------------------------------------------------
# 3. the trainer's receipt
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path, rows, meta, report) -> pathlib.Path:
    views = tmp_path / "sft_views.jsonl"
    views.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows),
                     encoding="utf-8")
    views.with_suffix(".meta.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8")
    views.with_suffix(".report.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
    return views


def _receipt(tmp_path, *, cfg=None, checkpoint=None, uuid=FAKE_UUID, steps=12):
    """Build and write a training manifest the way `sft.main` does."""
    from agentlab import sft

    cfg = cfg or _cfg(tmp_path)
    _records, rows, meta, report = _views(tmp_path)
    views = _write_corpus(tmp_path, rows, meta, report)
    bundle = sft.load_views_metadata(views)
    inputs = sft.require_training_inputs(bundle)
    _path, producer = _producer(tmp_path / "trainer", stage="rs_sft", uuid=uuid)
    ckpt = checkpoint or _checkpoint(tmp_path)
    payload = sft.build_training_manifest(
        producer=producer, producer_path=_path, inputs=inputs,
        hyperparameters={"lora_rank": 32, "bsz": 2}, optimizer_steps=steps,
        train_rows=len(rows), eval_rows=1,
        started_at_utc=configio.now_utc(), checkpoint_path=ckpt, cfg=cfg)
    receipt = sft.write_training_manifest(ckpt, payload)
    return sft, cfg, ckpt, receipt, payload


def test_the_trainer_receipt_records_the_whole_chain(tmp_path):
    sft, cfg, ckpt, receipt, payload = _receipt(tmp_path)
    assert receipt == ckpt.parent / (ckpt.name + sft.TRAINING_MANIFEST_SUFFIX)
    assert receipt.exists()
    # written BESIDE the adapter: a file inside it would change the very digest
    # the lock pins
    assert receipt.parent == ckpt.parent and receipt.name != ckpt.name

    for field in sft.TRAINING_MANIFEST_FIELDS:
        assert payload.get(field) is not None, field
    assert payload["hardware"]["gpu_uuid"] == FAKE_UUID
    assert payload["ledger"]["gpu_uuid"] == FAKE_UUID
    assert payload["engine_fingerprint"] == configio.engine_fingerprint(CFG)
    assert payload["config_hash"] == configio.config_hash()
    assert payload["git_sha"] == configio.git_sha()
    assert payload[contract_mod.STAMP_FIELD] == \
        contract_mod.environment_contract_sha256()
    assert payload["optimizer_steps"] == 12
    assert payload["inputs"]["source_provenance"]["gpu_uuid"] == FAKE_UUID
    assert payload["inputs"]["views_sha256"] and payload["inputs"]["meta_sha256"]
    # the checkpoint is pinned by CONTENT
    assert payload["checkpoint"]["checkpoint_sha256"] == \
        configio.checkpoint_tree_sha256(ckpt)
    assert sft.require_training_manifest(receipt, checkpoint_path=ckpt, cfg=cfg,
                                        stage="rs_sft")["optimizer_steps"] == 12


def test_the_receipt_is_refused_once_the_checkpoint_bytes_change(tmp_path):
    sft, cfg, ckpt, receipt, _payload = _receipt(tmp_path)
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x09\x09\x09")
    with pytest.raises(SystemExit, match="A lock pins BYTES"):
        sft.require_training_manifest(receipt, checkpoint_path=ckpt, cfg=cfg)


def test_an_edited_receipt_is_not_evidence(tmp_path):
    sft, cfg, ckpt, receipt, payload = _receipt(tmp_path)
    tampered = dict(payload, optimizer_steps=999)
    receipt.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not hash to its recorded digest"):
        sft.require_training_manifest(receipt, checkpoint_path=ckpt, cfg=cfg)


def test_a_receipt_with_no_card_is_refused(tmp_path):
    sft, cfg, ckpt, _path, payload = _receipt(tmp_path)
    holed = dict(payload)
    holed["hardware"] = dict(payload["hardware"], gpu_uuid=None)
    holed[configio.MANIFEST_HASH_FIELD] = configio.manifest_digest(holed)
    path = tmp_path / "holed.json"
    path.write_text(json.dumps(holed), encoding="utf-8")
    with pytest.raises(SystemExit, match="no gpu_uuid"):
        sft.require_training_manifest(path, checkpoint_path=ckpt, cfg=cfg)


def test_the_ledger_and_the_trainer_must_agree_on_the_card(tmp_path):
    """Hours charged to one card cannot have produced a checkpoint on another."""
    cfg = _cfg(tmp_path, ledger_uuid=OTHER_UUID)
    sft, cfg, ckpt, receipt, _payload = _receipt(tmp_path, cfg=cfg)
    with pytest.raises(SystemExit, match="ledger is bound to"):
        sft.require_training_manifest(receipt, checkpoint_path=ckpt, cfg=cfg)


def test_the_trainer_refuses_a_corpus_with_no_metadata(tmp_path):
    from agentlab import sft

    _records, rows, meta, report = _views(tmp_path)
    views = _write_corpus(tmp_path, rows, meta, report)
    views.with_suffix(".meta.jsonl").unlink()
    with pytest.raises(SystemExit, match="view metadata"):
        sft.load_views_metadata(views)


def test_the_trainer_refuses_a_stale_corpus(tmp_path):
    from agentlab import sft

    _records, rows, meta, report = _views(tmp_path)
    stale_meta = [dict(m, **{contract_mod.STAMP_FIELD: "0" * 64}) for m in meta]
    views = _write_corpus(tmp_path, rows, stale_meta, report)
    with pytest.raises(SystemExit, match="REFUSED"):
        sft.require_training_inputs(sft.load_views_metadata(views))


# ---------------------------------------------------------------------------
# 4. the checkpoint lock: bytes, not a path
# ---------------------------------------------------------------------------

def _lock_module_for(tmp_path, mod=None):
    mod = mod or _locks_module()
    mod.ROOT = tmp_path
    mod.RESULTS = tmp_path / "results"
    mod.LOCKS = mod.RESULTS / "locks.json"
    mod.REVEAL = mod.RESULTS / "seed_reveal.json"
    mod._git = lambda *a: "0" * 40
    return mod


def test_a_path_only_checkpoint_lock_is_rejected(tmp_path):
    """THE required refusal: the OLD lock shape pins nothing and is refused.

    {path, stage, locked_at, commit} is exactly what this script used to write. A
    path is mutable, so such a lock neither fixes the bytes nor ties them to the
    run -- and a verdict citing "the locked checkpoint" would cite nothing.
    """
    mod = _lock_module_for(tmp_path)
    path_only = {"checkpoint": {"path": "out/multiface/rssft-lora",
                                "stage": "rs_sft", "locked_at": "2026-08-05T00:00:00Z",
                                "commit": "0" * 40}}
    with pytest.raises(SystemExit) as exc:
        mod.verify_checkpoint_lock(path_only, cfg=_cfg(tmp_path))
    message = str(exc.value)
    assert "checkpoint_sha256" in message and "training_manifest" in message
    assert "records only a PATH pins nothing" in message

    with pytest.raises(SystemExit, match="no checkpoint is locked"):
        mod.verify_checkpoint_lock({}, cfg=_cfg(tmp_path))


def test_locking_a_checkpoint_with_no_receipt_is_refused(tmp_path):
    """No trainer manifest, no lock: the chain is required, not preferred."""
    import argparse

    mod = _lock_module_for(tmp_path)
    ckpt = _checkpoint(tmp_path)
    args = argparse.Namespace(path=str(ckpt.relative_to(tmp_path)), stage="rs_sft",
                              training_manifest=None)
    with pytest.raises(SystemExit, match="no training manifest"):
        mod.cmd_lock_checkpoint(args)
    assert not mod.LOCKS.exists(), "a refused lock must leave no locks.json"


def test_the_lock_pins_the_checkpoint_byte_digest(tmp_path):
    import argparse

    sft, cfg, ckpt, receipt, payload = _receipt(tmp_path)
    mod = _lock_module_for(tmp_path)
    mod._load_cfg = lambda: cfg
    args = argparse.Namespace(path=str(ckpt.relative_to(tmp_path)), stage="rs_sft",
                              training_manifest=str(receipt.relative_to(tmp_path)))
    assert mod.cmd_lock_checkpoint(args) == 0

    rec = json.loads(mod.LOCKS.read_text(encoding="utf-8"))["checkpoint"]
    for field in mod.CHECKPOINT_LOCK_FIELDS:
        assert rec.get(field) not in (None, ""), field
    assert rec["checkpoint_sha256"] == configio.checkpoint_tree_sha256(ckpt)
    assert rec["training_manifest_sha256"] == payload[configio.MANIFEST_HASH_FIELD]
    assert rec["gpu_uuid"] == FAKE_UUID
    assert rec["views_sha256"] == payload["inputs"]["views_sha256"]
    assert rec["source_provenance"]["gpu_uuid"] == FAKE_UUID
    assert rec["environment_contract_sha256"] == \
        contract_mod.environment_contract_sha256()
    # the lock verifies, and stops verifying the moment the bytes move
    mod.verify_checkpoint_lock({"checkpoint": rec}, cfg=cfg)
    (ckpt / "adapter_model.safetensors").write_bytes(b"\xff")
    with pytest.raises(SystemExit, match="A lock pins BYTES"):
        mod.verify_checkpoint_lock({"checkpoint": rec}, cfg=cfg)


def test_a_checkpoint_trained_on_an_unattested_corpus_cannot_be_locked(tmp_path):
    """The chain is transitive: a fixture-trained adapter is not lockable."""
    from agentlab import sft

    cfg = _cfg(tmp_path)
    bundle = build_task(SUITE, SEED, "distill", "typed_relay", 4, 22, None)
    records = run_engine([bundle], cfg=CFG, secret=TEST_SECRET)
    rows, meta, report = datasets.build_views(records, token_counter_stub(), CFG)
    views = _write_corpus(tmp_path, rows, meta, report)
    inputs = sft.require_training_inputs(sft.load_views_metadata(views),
                                        require_gpu_source=False)
    _path, producer = _producer(tmp_path / "trainer", stage="rs_sft")
    ckpt = _checkpoint(tmp_path)
    payload = sft.build_training_manifest(
        producer=producer, producer_path=_path, inputs=inputs,
        hyperparameters={"lora_rank": 32}, optimizer_steps=1, train_rows=len(rows),
        eval_rows=1, started_at_utc=configio.now_utc(), checkpoint_path=ckpt,
        cfg=cfg)
    receipt = sft.write_training_manifest(ckpt, payload)
    with pytest.raises(SystemExit, match="gpu_execution"):
        sft.require_training_manifest(receipt, checkpoint_path=ckpt, cfg=cfg)


# ---------------------------------------------------------------------------
# 5. stages with no visible device must not look like GPU runs
# ---------------------------------------------------------------------------

def test_the_short_circuited_probe_does_not_look_like_a_gpu_run(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(variance, "REPORT_PATH", tmp_path / "variance_report.json")
    out = variance.write_disposition_report()
    prov = out["provenance"]
    assert prov[multidistill.GPU_EXECUTION] is False
    assert prov["producer"] == "variance-probe-disposition"
    assert prov["gpu_uuid"] is None and prov["gpu_name"] is None
    # the frozen S19 key set is still whole, so a reader never has to guess
    for field in configio.FINGERPRINT_FIELDS:
        assert field in prov, field
    assert multidistill.provenance_gaps(prov) == []
