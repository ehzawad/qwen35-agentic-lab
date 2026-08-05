"""Loader for configs/multifaceted.yaml, the single source of every count and gate.

Lives in its own module (not suite/__init__.py) so the training-path modules and
the measurement modules can evolve without editing each other's files.

It also owns three things that must have exactly ONE definition in this repo,
because two definitions is how a study silently compares two different runs:

  engine_contract()   the registered vLLM settings every inference stage uses
  fingerprint()       the S19 HARDWARE-INTEGRITY row every claim-bearing trace
                      and every ledger row carries
  ledger_append()     the GPU ledger, whose rows are now full receipts rather
                      than "stage, minutes, cumulative"
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "multifaceted.yaml"

DEFAULT_RUN_ID = os.environ.get("AGENTIC_RUN_ID") or "agentic-v1"


@functools.lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    """Parse the multifaceted config once per path; callers must not mutate it."""
    import yaml

    p = pathlib.Path(path) if path else CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=4)
def config_hash(path: str | None = None) -> str:
    """SHA-256 of the config bytes: what `config_hash` means in every trace row."""
    p = pathlib.Path(path) if path else CONFIG_PATH
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@functools.lru_cache(maxsize=1)
def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=10,
                             capture_output=True, text=True).stdout.strip()
        return out or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# the registered engine contract -- ONE copy, read by every inference stage
# --------------------------------------------------------------------------

# FROZEN key set. configs/agentic_preregister.json machine.engine_contract is the
# analyzer's copy of the same values and a test cross-checks the two, so a drift
# in either file fails on CPU rather than after a GPU-hour.
ENGINE_KEYS = ("dtype", "gpu_memory_utilization", "max_model_len", "max_num_seqs",
               "max_num_batched_tokens", "enforce_eager", "enable_thinking",
               "multimodal_inputs", "tensor_parallel_size")

_ENGINE_TYPES = {"dtype": str, "gpu_memory_utilization": float, "max_model_len": int,
                 "max_num_seqs": int, "max_num_batched_tokens": int,
                 "enforce_eager": bool, "enable_thinking": bool,
                 "multimodal_inputs": str, "tensor_parallel_size": int}


def engine_contract(cfg: dict | None = None) -> dict:
    """The registered engine settings. Every stage builds its engine from THIS.

    Refuses a config that is missing a key or carries the wrong type: a stage
    that silently fell back to a vLLM default is exactly the drift the single
    copy exists to prevent.
    """
    cfg = cfg or load_config()
    block = cfg.get("engine")
    if not isinstance(block, dict):
        raise SystemExit("configs/multifaceted.yaml has no `engine:` block; the "
                         "registered engine contract has ONE home and this is it")
    out = {}
    for key in ENGINE_KEYS:
        if key not in block:
            raise SystemExit(f"engine contract is missing `{key}`; refusing to "
                             f"let vLLM pick a default for a registered setting")
        want = _ENGINE_TYPES[key]
        val = block[key]
        if want is float and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)
        if isinstance(val, bool) is not (want is bool) or not isinstance(val, want):
            raise SystemExit(f"engine contract `{key}` is {val!r} "
                             f"({type(val).__name__}), expected {want.__name__}")
        out[key] = val
    if out["multimodal_inputs"] != "REJECTED":
        raise SystemExit("engine contract multimodal_inputs must be REJECTED: "
                         "image/video inputs are refused, not merely unused")
    return out


def multimodal_rejected(cfg: dict | None = None) -> bool:
    return engine_contract(cfg)["multimodal_inputs"] == "REJECTED"


def reject_multimodal(messages: list[dict], cfg: dict | None = None) -> None:
    """Refuse any non-text message content before it can reach the engine.

    The contract says multimodal inputs are REJECTED rather than merely unused.
    Convention is not a mechanism: this is the mechanism. Qwen3.5 is a natively
    multimodal checkpoint, so an image part would be accepted by the server and
    would silently produce an episode no registered claim describes.
    """
    if not multimodal_rejected(cfg):
        return
    for i, msg in enumerate(messages or ()):
        content = msg.get("content")
        if content is None or isinstance(content, str):
            continue
        raise SystemExit(
            f"REFUSED: message {i} ({msg.get('role')}) carries non-text content "
            f"({type(content).__name__}). The registered engine contract REJECTS "
            f"image/video inputs; a multimodal episode is not a registered claim.")


def engine_startup_s(cfg: dict | None = None) -> float:
    cfg = cfg or load_config()
    return float(cfg["engine"]["measured"]["startup_s"])


def hardware_contract(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    hw = cfg.get("hardware")
    if not isinstance(hw, dict):
        raise SystemExit("configs/multifaceted.yaml has no `hardware:` block")
    return hw


def hardware_lock_path(cfg: dict | None = None) -> pathlib.Path:
    return ROOT / hardware_contract(cfg).get("lock", "results/agentic/hardware.json")


@functools.lru_cache(maxsize=8)
def _package_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in ("vllm", "torch", "transformers", "trl", "peft"):
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return dict(out)


def engine_fingerprint(cfg: dict | None = None) -> dict:
    """The value S19 compares across every trace: stack versions + the contract.

    Deliberately contains nothing per-episode, per-shard or per-clock, so two
    traces from one run compare equal and two traces from different engines never
    do.
    """
    fp = dict(_package_versions())
    fp.update(engine_contract(cfg))
    return fp


# --------------------------------------------------------------------------
# hardware fingerprint (never allocates, never probes an unpinned card)
# --------------------------------------------------------------------------

GPU_FIELDS = ("gpu_name", "gpu_uuid", "cuda_visible_bytes", "driver_version",
              "pci_bus_id", "compute_capability", "visible_ordinal")


def _empty_gpu() -> dict:
    return {k: None for k in GPU_FIELDS}


def _nvidia_smi_identity(index: str) -> dict:
    """name/uuid/driver/bus of ONE pinned index. Read-only, no CUDA context."""
    row = _empty_gpu()
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", index,
             "--query-gpu=name,uuid,driver_version,pci.bus_id,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return row
    if not out:
        return row
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    keys = ("gpu_name", "gpu_uuid", "driver_version", "pci_bus_id",
            "compute_capability")
    for key, val in zip(keys, parts):
        row[key] = val or None
    row["visible_ordinal"] = 0
    return row


# Public alias: agentlab.env reads the same identity when it binds the card.
nvidia_smi_identity = _nvidia_smi_identity


def gpu_fingerprint(cfg: dict | None = None) -> dict:
    """GPU identity for the pinned card, or all-None off-GPU.

    Preference order:
      1. the run's hardware lock, written by the first GPU stage. It is the only
         source of `cuda_visible_bytes`, which is torch's measured total_memory
         -- NOT the board total nvidia-smi prints -- and it is what lets a
         pure-HTTP client process (the evaluator) carry the same fingerprint as
         the engine process without opening a CUDA context of its own.
      2. a read-only nvidia-smi identity query on the pinned index.
      3. nothing. Missing provenance is missing evidence (S19 -> INCONCLUSIVE);
         it is never backfilled from the registered expectation, because an
         assumed A5000 is exactly the assumption S19 exists to refuse.

    With CUDA_VISIBLE_DEVICES empty or unset nothing is probed at all: no
    nvidia-smi call, and in particular never a card this run does not own.
    """
    lock = hardware_lock_path(cfg)
    if lock.exists():
        try:
            rec = json.loads(lock.read_text(encoding="utf-8"))
            row = {k: rec.get(k) for k in GPU_FIELDS}
            if row["gpu_uuid"]:
                return row
        except Exception:
            pass
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not visible:
        return _empty_gpu()
    return _nvidia_smi_identity(visible.split(",")[0].strip())


FINGERPRINT_FIELDS = ("run_id", "git_sha", "config_hash", "gpu_name", "gpu_uuid",
                      "cuda_visible_bytes", "driver_version", "engine_fingerprint",
                      "enable_thinking_effective", "timestamp_utc")

# What must not change between two shards of one trace file, or between the two
# members of a paired comparison. git_sha is excluded on purpose: a commit
# between two shards is legitimate, and S19 does not pair on it.
FINGERPRINT_IDENTITY_FIELDS = ("run_id", "config_hash", "gpu_name", "gpu_uuid",
                               "cuda_visible_bytes", "driver_version",
                               "engine_fingerprint", "enable_thinking_effective")


def fingerprint(run_id: str | None = None, cfg: dict | None = None,
                enable_thinking: bool | None = None) -> dict:
    """The frozen S19 fingerprint. Exactly FINGERPRINT_FIELDS, nothing implied."""
    cfg = cfg or load_config()
    contract = engine_contract(cfg)
    gpu = gpu_fingerprint(cfg)
    thinking = (contract["enable_thinking"] if enable_thinking is None
                else bool(enable_thinking))
    return {"run_id": run_id or DEFAULT_RUN_ID,
            "git_sha": git_sha(),
            "config_hash": config_hash(),
            "gpu_name": gpu["gpu_name"],
            "gpu_uuid": gpu["gpu_uuid"],
            "cuda_visible_bytes": gpu["cuda_visible_bytes"],
            "driver_version": gpu["driver_version"],
            "engine_fingerprint": engine_fingerprint(cfg),
            "enable_thinking_effective": thinking,
            "timestamp_utc": now_utc()}


def fingerprint_identity(row: dict) -> dict:
    return {k: row.get(k) for k in FINGERPRINT_IDENTITY_FIELDS}


def fingerprint_conflict(a: dict, b: dict) -> list[str]:
    """Which identity fields two fingerprints disagree on (empty = compatible)."""
    return [k for k in FINGERPRINT_IDENTITY_FIELDS if a.get(k) != b.get(k)]


# --------------------------------------------------------------------------
# GPU ledger: one full receipt per completed chunk, hard ceiling for the run
# --------------------------------------------------------------------------

def ledger_path(cfg: dict | None = None) -> pathlib.Path:
    cfg = cfg or load_config()
    return ROOT / cfg["budget"]["ledger"]


def ledger_rows(cfg: dict | None = None) -> list[dict]:
    p = ledger_path(cfg)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ledger_hours(cfg: dict | None = None) -> float:
    """Cumulative measured GPU hours so far (0.0 when no ledger exists yet)."""
    return sum(float(r.get("minutes", 0.0)) for r in ledger_rows(cfg)) / 60.0


def ledger_bound_uuid(cfg: dict | None = None) -> str | None:
    """The GPU UUID the first ledger row bound this run to, if any."""
    for row in ledger_rows(cfg):
        if row.get("gpu_uuid"):
            return row["gpu_uuid"]
    return None


def ledger_append(stage: str, minutes: float, cfg: dict | None = None, *,
                  kind: str = "stage", work: dict | None = None,
                  run_id: str | None = None, started_at: str | None = None,
                  enable_thinking: bool | None = None) -> float:
    """Append one completed chunk; returns cumulative hours after the append.

    The row is a receipt, not a tally. The protocol promises timestamps, GPU
    identity, git SHA and work counts on every ledger row, and the earlier
    implementation wrote {stage, minutes, cumulative_h}: a ledger that cannot say
    which card spent the hours cannot support a same-card claim.

    A second physical GPU inside one run is fatal here rather than at analysis
    time, because mixing two cards inside one claim is the S19 failure itself.
    """
    cfg = cfg or load_config()
    fp = fingerprint(run_id, cfg, enable_thinking=enable_thinking)
    bound = ledger_bound_uuid(cfg)
    if bound and fp["gpu_uuid"] and fp["gpu_uuid"] != bound:
        raise SystemExit(
            f"FATAL: the GPU ledger is bound to {bound} and this stage is on "
            f"{fp['gpu_uuid']}. One run may not span two physical cards (S19). "
            f"A different card is a NEW run with its own run_id, locks, seeds "
            f"and ledger.")
    p = ledger_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cumulative = ledger_hours(cfg) + minutes / 60.0
    row = {"stage": stage, "kind": kind,
           "minutes": round(float(minutes), 2),
           # The protocol enforces the ceiling against the sum of MEASURED
           # seconds, so the seconds are recorded, not only their rounded minutes.
           "elapsed_s": round(float(minutes) * 60.0, 1),
           "cumulative_h": round(cumulative, 3),
           "started_at_utc": started_at or fp["timestamp_utc"],
           "ended_at_utc": now_utc(),
           "work": work or {}}
    row.update(fp)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return cumulative


def ledger_guard(stage: str, projected_minutes: float, cfg: dict | None = None) -> None:
    """Refuse to start GPU work whose projection would cross the hard ceiling."""
    cfg = cfg or load_config()
    ceiling = float(cfg["budget"]["gpu_hours_ceiling"])
    projected = ledger_hours(cfg) + projected_minutes / 60.0
    if projected > ceiling:
        raise SystemExit(
            f"BUDGET: {stage} projects {projected:.1f} GPU-hours against the "
            f"{ceiling:.0f}h ceiling; stopping before touching the card.")
