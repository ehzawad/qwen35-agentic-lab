"""Supervised fine-tuning (LoRA) on verified distilled trajectories.

The sole supported input is the corpus written by `agentlab.distill`: the
model's own multi-turn episodes against the real tools, kept only when they end
in a correct committed answer. Every training row therefore demonstrates the
full loop -- call a tool, read the result, terminate -- and GRPO downstream
assumes this already works: if the policy never produces a parseable call and a
committed answer, every rollout scores zero and the gradient is noise.

THE A5000 TRAINING CONTRACT (configs/multifaceted.yaml `sft:`). Static training
memory is 8.455 GiB of bf16 base policy plus 0.968 GiB of LoRA fp32 parameters,
gradients and two Adam moments = 9.423 GiB. One 4,096-token sequence of logits
over this 248,320-token vocabulary is about 2.03 GiB, so train batch 2 peaks near
13.5 GiB and gradient checkpointing holds the working peak around 14-15 GiB of
23.3 GiB.

`per_device_eval_batch_size` is the dangerous one and is therefore explicit.
Hugging Face defaults it to EIGHT, which is 8 x 2.03 = 16.24 GiB of logits on top
of the 9.423 GiB static footprint -- 25.66 GiB, which cannot fit on this card, and
which would fail at the first evaluation rather than at start-up. At batch 1 the
peak is about 11.45 GiB, and `prediction_loss_only` stops the trainer gathering
logits at all.

THE TRAINING RECEIPT. This stage is the point where the provenance chain used to
stop: an adapter appeared in a directory, and the checkpoint lock recorded its
PATH. A path is not evidence -- the bytes behind it can change, and nothing tied
them to the trajectories, the card or the run. So the trainer now:

  * REFUSES a view corpus that cannot name its producer (see
    `suite.datasets.require_views_chain`), before it touches an optimizer;
  * captures its own producer runtime manifest (the trainer process owns the
    card, so the trainer attests the training hardware);
  * writes `<adapter>.agentlab_training_manifest.json` -- hardware, engine
    fingerprint, config hash, git sha, input digests, source provenance,
    hyperparameters, optimizer steps, timestamps, ledger UUID and the final
    CHECKPOINT TREE DIGEST -- and `scripts/agentic_locks.py lock-checkpoint`
    refuses to lock a checkpoint without it.

The receipt lives BESIDE the adapter directory rather than inside it, and that is
deliberate: the digest it pins is the hash of the directory's bytes, so a file
written into that directory after hashing would make the recorded digest
unverifiable for ever.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from . import env
from .data import build_distill_sft
from .peft_cfg import describe, lora_config
from .suite import contract as contract_mod
from .suite.configio import load_config


def _flag(text: str) -> bool:
    return str(text).strip().lower() not in ("false", "0", "no")


def sft_config_kwargs(args) -> dict:
    """Every SFTConfig keyword this stage passes, as data.

    A pure function so the contract can be asserted on CPU: constructing a real
    SFTConfig requires a bf16-capable GPU, and "the eval batch is 1" is exactly
    the kind of claim that must not depend on a GPU being present to check.
    """
    return dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        # Registered evaluation safety: batch 1, no logit gathering, no
        # accumulation across eval steps.
        per_device_eval_batch_size=args.eval_bsz,
        eval_accumulation_steps=args.eval_accumulation_steps,
        prediction_loss_only=args.prediction_loss_only,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        # 262K-context model on a 24 GB card: cap the sequence and leave packing
        # off, so a single long trajectory row cannot blow the step.
        packing=args.packing,
    )


# --------------------------------------------------------------------------
# the training manifest: the trainer's link in the provenance chain
# --------------------------------------------------------------------------

TRAINING_MANIFEST_SUFFIX = ".agentlab_training_manifest.json"

TRAINING_MANIFEST_KIND = "agentlab_training_manifest"
TRAINING_MANIFEST_VERSION = 1

# Every one of these must be present and non-null in a manifest that may be
# locked against. `adapter` is not among them because a manifest describes the
# adapter it just wrote, which is `checkpoint`.
TRAINING_MANIFEST_FIELDS = (
    "kind", "schema_version", "run_id", "stage", "git_sha", "config_hash",
    contract_mod.STAMP_FIELD, "hardware", "runtime_manifest_sha256", "session_id",
    "engine_fingerprint", "inputs", "hyperparameters", "optimizer_steps",
    "train_rows", "started_at_utc", "finished_at_utc", "checkpoint", "ledger",
)


def training_manifest_path(out_dir) -> pathlib.Path:
    """`<adapter>.agentlab_training_manifest.json`, beside the adapter directory.

    Beside rather than inside: the manifest records the checkpoint TREE digest,
    and a file added to that tree after it was hashed would make the recorded
    digest impossible to reproduce.
    """
    p = pathlib.Path(str(out_dir).rstrip("/"))
    return p.parent / (p.name + TRAINING_MANIFEST_SUFFIX)


def load_views_metadata(views_path, meta_path=None, report_path=None) -> dict:
    """The three files the view builder wrote, or a refusal naming the missing one.

    The metadata and the report are DERIVED from the corpus path by default, so a
    trainer invocation cannot accidentally pair one build's rows with another
    build's provenance.
    """
    views = pathlib.Path(views_path)
    meta = pathlib.Path(meta_path) if meta_path else views.with_suffix(".meta.jsonl")
    report = (pathlib.Path(report_path) if report_path
              else views.with_suffix(".report.json"))
    for label, path in (("training rows", views), ("view metadata", meta),
                        ("view report", report)):
        if not path.exists():
            raise SystemExit(
                f"REFUSED: no {label} at {path}. The trainer will not train on a "
                f"corpus whose provenance is missing: build it with `python -m "
                f"agentlab.suite.datasets --accepted data/multiface/accepted.jsonl`, "
                f"which writes the rows, the per-row metadata and the report "
                f"together.")
    rows = [json.loads(line) for line in
            views.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_rows = [json.loads(line) for line in
                 meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    report_rec = json.loads(report.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"REFUSED: {views} is empty; there is nothing to train on.")
    return {"views_path": views, "meta_path": meta, "report_path": report,
            "rows": rows, "meta": meta_rows, "report": report_rec}


def file_sha256(path) -> str:
    """SHA-256 of one file's bytes (the corpus files the receipt points back at)."""
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def require_training_inputs(bundle: dict, *, require_gpu_source: bool = True) -> dict:
    """Verify the corpus chain and return the input digests the receipt records."""
    from .suite.datasets import require_views_chain

    checked = require_views_chain(bundle["rows"], bundle["meta"], bundle["report"],
                                 require_gpu_source=require_gpu_source)
    meta = bundle["meta"]
    return {
        "views_path": str(bundle["views_path"]),
        "views_sha256": file_sha256(bundle["views_path"]),
        "views_rows": len(bundle["rows"]),
        "meta_path": str(bundle["meta_path"]),
        "meta_sha256": file_sha256(bundle["meta_path"]),
        "report_path": str(bundle["report_path"]),
        "report_sha256": file_sha256(bundle["report_path"]),
        "source_provenance": checked["source_provenance"],
        "source_trajectories": len({m["source_row_sha256"] for m in meta}),
        "source_sessions": sorted({m["session_id"] for m in meta
                                   if m.get("session_id")}),
        "source_runtime_manifests": sorted({m["runtime_manifest_sha256"]
                                            for m in meta
                                            if m.get("runtime_manifest_sha256")}),
        "view_counts": {v: sum(1 for m in meta if m["view"] == v)
                        for v in sorted({m["view"] for m in meta})},
    }


def build_training_manifest(*, producer: dict, inputs: dict, hyperparameters: dict,
                            optimizer_steps: int, train_rows: int, eval_rows: int,
                            started_at_utc: str, checkpoint_path,
                            producer_path=None, cfg: dict | None = None) -> dict:
    """The receipt, as data. Pure, so a CPU test can assert the whole grammar.

    `producer` is the trainer's own runtime manifest (written by
    `env.capture_runtime_manifest`): the training hardware is attested by the
    process that owned the card, never re-measured by a later reader.
    """
    from .suite import configio

    cfg = cfg or load_config()
    ckpt = pathlib.Path(checkpoint_path)
    tree = configio.checkpoint_tree_sha256(ckpt)
    if not tree:
        raise SystemExit(
            f"REFUSED: there is nothing to hash at {ckpt}, so the trainer cannot "
            f"record which bytes it produced. A manifest for a checkpoint that "
            f"does not exist would let a lock pin a path with no content.")
    files = sorted(str(q.relative_to(ckpt)) for q in ckpt.rglob("*")
                   if q.is_file()) if ckpt.is_dir() else [ckpt.name]
    rec = {
        "kind": TRAINING_MANIFEST_KIND,
        "schema_version": TRAINING_MANIFEST_VERSION,
        "run_id": producer["run_id"],
        "stage": producer["stage"],
        "git_sha": configio.git_sha(),
        "config_hash": configio.config_hash(),
        contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
        # The trainer's OWN hardware, copied from its producer manifest.
        "hardware": {k: producer[k] for k in
                     ("gpu_name", "gpu_uuid", "cuda_visible_bytes",
                      "driver_version", "pci_bus_id", "compute_capability",
                      "visible_ordinal", "cuda_device_order",
                      "cuda_visible_devices")},
        "runtime_manifest_path": str(producer_path) if producer_path else None,
        "runtime_manifest_sha256": producer[configio.MANIFEST_HASH_FIELD],
        "session_id": producer["session_id"],
        "producer_pid": producer["pid"],
        "engine_fingerprint": producer["engine_fingerprint"],
        "enable_thinking_effective": producer["enable_thinking_effective"],
        "base_model": producer["model"],
        "inputs": inputs,
        "hyperparameters": hyperparameters,
        "optimizer_steps": int(optimizer_steps),
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "started_at_utc": started_at_utc,
        "finished_at_utc": configio.now_utc(),
        "checkpoint": {"path": str(ckpt),
                       "checkpoint_sha256": tree,
                       "files": len(files),
                       "file_names": files},
        # Which card the run's ledger is bound to. The lock refuses a manifest
        # whose training hardware is not the card the hours were charged to.
        "ledger": {"gpu_uuid": configio.ledger_bound_uuid(cfg),
                   "hours_at_finish": round(configio.ledger_hours(cfg), 4)},
    }
    rec[configio.MANIFEST_HASH_FIELD] = configio.manifest_digest(rec)
    return rec


def write_training_manifest(out_dir, payload: dict) -> pathlib.Path:
    """Atomically write the receipt beside the adapter and return its path."""
    from .suite import configio

    return configio.write_json_atomic(training_manifest_path(out_dir), payload)


def read_training_manifest(path) -> dict:
    """Parse a receipt and verify its self-hash. Never re-measures anything."""
    from .suite import configio

    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(
            f"REFUSED: no training manifest at {p}. A checkpoint whose training "
            f"was never attested cannot be locked: the lock would record a PATH "
            f"and nothing about the bytes, the card, the corpus or the run that "
            f"produced them. Re-run the sft stage on this build.")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"REFUSED: training manifest {p} is unreadable: {exc}")
    if not isinstance(rec, dict) or rec.get("kind") != TRAINING_MANIFEST_KIND:
        raise SystemExit(f"REFUSED: {p} is not an {TRAINING_MANIFEST_KIND}.")
    got = rec.get(_manifest_hash_field())
    want = _manifest_digest(rec)
    if got != want:
        raise SystemExit(
            f"REFUSED: training manifest {p} does not hash to its recorded digest "
            f"({got} != {want}); it was edited after the training run, so it is "
            f"not evidence.")
    return rec


def _manifest_hash_field() -> str:
    from .suite import configio

    return configio.MANIFEST_HASH_FIELD


def _manifest_digest(rec: dict) -> str:
    from .suite import configio

    return configio.manifest_digest(rec)


def require_training_manifest(path, *, checkpoint_path, cfg: dict | None = None,
                              stage: str | None = None) -> dict:
    """The fail-closed gate the checkpoint lock runs. Verifies the WHOLE chain.

    In order, terminating on the first failure: the receipt exists and hashes to
    itself; every registered field is present and non-null; it was written under
    THIS environment contract; the checkpoint bytes on disk hash to the digest it
    recorded; the training hardware is attested (non-null UUID) and is the card
    this run is bound to and the card the ledger charged; and the corpus it names
    was itself produced by an attested GPU session.
    """
    from .suite import configio

    cfg = cfg or load_config()
    rec = read_training_manifest(path)
    missing = [k for k in TRAINING_MANIFEST_FIELDS
               if rec.get(k) is None or rec.get(k) == ""]
    if missing:
        raise SystemExit(
            f"REFUSED: training manifest {path} is incomplete "
            f"({', '.join(missing)}). A receipt with holes in it is not a chain.")
    contract_mod.require_current(rec, f"the training manifest {path}")
    if stage is not None and rec["stage"] != stage:
        raise SystemExit(
            f"REFUSED: training manifest {path} was written by stage "
            f"{rec['stage']!r} and this lock selects {stage!r}.")

    ckpt = pathlib.Path(checkpoint_path)
    got = configio.checkpoint_tree_sha256(ckpt)
    want = rec["checkpoint"]["checkpoint_sha256"]
    if not got:
        raise SystemExit(f"REFUSED: there is nothing to hash at {ckpt}.")
    if got != want:
        raise SystemExit(
            f"REFUSED: {ckpt} hashes {got} and the training manifest recorded "
            f"{want}. The bytes changed after training, so the receipt describes a "
            f"different checkpoint. A lock pins BYTES, not a path -- that is the "
            f"whole point of recording the digest.")

    uuid = rec["hardware"].get("gpu_uuid")
    if not uuid:
        raise SystemExit(
            f"REFUSED: training manifest {path} records no gpu_uuid, so nothing "
            f"attests which card trained this adapter (S19). A trainer that could "
            f"not measure its card must fail before writing a checkpoint.")
    lock = configio.hardware_lock_path(cfg)
    if lock.exists():
        binding = json.loads(lock.read_text(encoding="utf-8"))
        for key in ("gpu_uuid", "gpu_name", "cuda_visible_bytes"):
            wanted, seen = binding.get(key), rec["hardware"].get(key)
            if wanted is not None and seen is not None and str(wanted) != str(seen):
                raise SystemExit(
                    f"REFUSED: the adapter was trained with {key}={seen!r} and "
                    f"this run is bound to {wanted!r}. Neither side wins: the "
                    f"lock aborts as a hardware-integrity failure (S19).")
    bound = configio.ledger_bound_uuid(cfg)
    if bound and bound != uuid:
        raise SystemExit(
            f"REFUSED: the GPU ledger is bound to {bound} and this adapter was "
            f"trained on {uuid}. The hours charged to this run were not spent on "
            f"the card that produced the checkpoint being locked (S19).")

    source = rec["inputs"].get("source_provenance") or {}
    if not source.get("gpu_execution"):
        raise SystemExit(
            f"REFUSED: the corpus this adapter was trained on was produced by "
            f"{source.get('producer')!r} with gpu_execution="
            f"{source.get('gpu_execution')!r}. A checkpoint trained on rollouts no "
            f"card attested cannot be locked as a study candidate.")
    return rec


def build_parser(sft: dict) -> argparse.ArgumentParser:
    """Defaults come from configs/multifaceted.yaml, not from CLI literals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--distill-path", default="data/distill.jsonl",
                    help="verified trajectories from `make distill`")
    ap.add_argument("--epochs", type=float, default=float(sft["epochs"]))
    ap.add_argument("--bsz", type=int, default=int(sft["bsz"]))
    ap.add_argument("--accum", type=int, default=int(sft["accum"]))
    ap.add_argument("--lr", type=float, default=float(sft["lr"]))
    ap.add_argument("--rank", type=int, default=int(sft["lora_rank"]))
    # Wired through as real flags. They were configured in
    # configs/multifaceted.yaml and never reached the trainer, so a config edit
    # silently changed nothing about the adapter that actually got trained.
    ap.add_argument("--lora-alpha", type=int, default=int(sft["lora_alpha"]))
    ap.add_argument("--lora-dropout", type=float, default=float(sft["lora_dropout"]))
    ap.add_argument("--max-length", type=int, default=int(sft["max_length"]))
    ap.add_argument("--eval-bsz", type=int, default=int(sft["eval_bsz"]))
    ap.add_argument("--eval-accumulation-steps", type=int,
                    default=int(sft["eval_accumulation_steps"]))
    ap.add_argument("--prediction-loss-only", type=_flag,
                    default=bool(sft["prediction_loss_only"]))
    ap.add_argument("--gradient-checkpointing", type=_flag,
                    default=bool(sft["gradient_checkpointing"]))
    ap.add_argument("--packing", type=_flag, default=bool(sft["packing"]))
    ap.add_argument("--out", default=None)
    # The provenance seam. Both default to the corpus path, so the rows and the
    # metadata a receipt describes always come from ONE view build.
    ap.add_argument("--views-meta", default=None,
                    help="per-row view metadata (default: <corpus>.meta.jsonl)")
    ap.add_argument("--views-report", default=None,
                    help="view build report (default: <corpus>.report.json)")
    ap.add_argument("--run-id", default=None,
                    help="S19 run_id for this trainer session (default AGENTIC_RUN_ID)")
    return ap


def hyperparameters_of(args, kwargs: dict) -> dict:
    """Every knob that shaped the adapter, as data, for the receipt."""
    return dict(kwargs, lora_rank=args.rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, model=args.model,
                eval_test_size=0.02, split_seed=0)


def main() -> None:
    cfg = load_config()
    args = build_parser(cfg["sft"]).parse_args()

    from .suite import configio

    args.out = args.out or str(env.SFT_DIR)
    started = configio.now_utc()
    t0 = time.time()

    # (1) THE CORPUS CHAIN, before the card is touched. A corpus whose rows
    #     cannot name the trajectory, the environment contract and the producer
    #     session behind them is refused here: training on it would produce an
    #     adapter no lock could attribute, and that is the seam this stage exists
    #     to close.
    bundle = load_views_metadata(args.distill_path, args.views_meta,
                                 args.views_report)
    inputs = require_training_inputs(bundle)
    print(f"[sft] corpus chain OK: {inputs['views_rows']} rows from "
          f"{inputs['source_trajectories']} attested trajectories, "
          f"{len(inputs['source_sessions'])} producer session(s)")

    # (2) THE TRAINER'S OWN ATTESTATION. This process owns the card, so it -- not
    #     a later reader -- records which card trained the adapter. The full
    #     hardware veto runs inside `capture_runtime_manifest`.
    manifest_path, producer = env.capture_runtime_manifest(
        stage="rs_sft", cfg=cfg, run_id=args.run_id, model=args.model)
    producer = configio.mark_manifest_ready(manifest_path, run_id=producer["run_id"],
                                           cfg=cfg, stage="rs_sft")

    from trl import SFTConfig, SFTTrainer

    ds = build_distill_sft(args.distill_path)
    print(f"[data] source={args.distill_path}")
    split = ds.train_test_split(test_size=0.02, seed=0)
    print(f"[data] train={len(split['train'])} eval={len(split['test'])} cols={ds.column_names}")

    kwargs = sft_config_kwargs(args)
    print(f"[sft] train bsz {kwargs['per_device_train_batch_size']} x accum "
          f"{kwargs['gradient_accumulation_steps']} (effective "
          f"{kwargs['per_device_train_batch_size'] * kwargs['gradient_accumulation_steps']}), "
          f"eval bsz {kwargs['per_device_eval_batch_size']}, "
          f"max_length {kwargs['max_length']}, "
          f"grad-ckpt {kwargs['gradient_checkpointing']}, packing {kwargs['packing']}")
    print(f"[sft] LoRA r={args.rank} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout}")

    trainer = SFTTrainer(
        model=args.model,
        args=SFTConfig(**kwargs),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora_config(r=args.rank, alpha=args.lora_alpha,
                                dropout=args.lora_dropout),
    )
    describe(trainer.model)

    trainer.train()
    trainer.save_model(args.out)

    # (3) THE RECEIPT. Written AFTER the checkpoint, because it pins the digest of
    #     the bytes that were written; and beside the directory, because a file
    #     inside it would change that digest.
    steps = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    payload = build_training_manifest(
        producer=producer, producer_path=manifest_path, inputs=inputs,
        hyperparameters=hyperparameters_of(args, kwargs),
        optimizer_steps=steps, train_rows=len(split["train"]),
        eval_rows=len(split["test"]), started_at_utc=started,
        checkpoint_path=args.out, cfg=cfg)
    receipt = write_training_manifest(args.out, payload)
    print(f"[done] adapter -> {args.out}")
    print(f"[sft] {steps} optimizer steps in "
          f"{(time.time() - t0) / 60.0:.1f} min")
    print(f"[sft] checkpoint sha256 {payload['checkpoint']['checkpoint_sha256']}")
    print(f"[sft] training manifest -> {receipt}")
    print(f"[sft] runtime manifest  -> {manifest_path}")
    # Re-read the receipt through the same gate the checkpoint lock uses, so a
    # broken chain fails HERE, while the operator is still looking at the trainer,
    # rather than at lock time after the GPU hours are spent.
    require_training_manifest(receipt, checkpoint_path=args.out, cfg=cfg,
                              stage="rs_sft")
    print("[sft] provenance chain verified: corpus -> trainer manifest -> "
          "checkpoint bytes")


if __name__ == "__main__":
    main()
