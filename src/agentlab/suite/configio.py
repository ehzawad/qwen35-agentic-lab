"""Loader for configs/multifaceted.yaml, the single source of every count and gate.

Lives in its own module (not suite/__init__.py) so the training-path modules and
the measurement modules can evolve without editing each other's files.

It also owns four things that must have exactly ONE definition in this repo,
because two definitions is how a study silently compares two different runs:

  engine_contract()   the registered vLLM settings every inference stage uses
  fingerprint()       the S19 HARDWARE-INTEGRITY row every claim-bearing trace
                      and every ledger row carries
  runtime manifests   the PRODUCER-side attestation: the process that actually
                      owned the card writes what it measured, and every consumer
                      (the pure-HTTP evaluator, the ledger) COPIES it instead of
                      guessing from its own environment
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
import socket
import subprocess
import tempfile
import urllib.parse

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


def fingerprint_gaps(fp: dict) -> list[str]:
    """Which S19 fields are absent or null. Empty means the fingerprint is whole."""
    return [k for k in FINGERPRINT_FIELDS
            if fp.get(k) is None or fp.get(k) == ""]


def require_complete_fingerprint(fp: dict, what: str = "a claim-bearing row") -> dict:
    """Refuse to SERIALIZE an incomplete S19 fingerprint.

    The D1 failure was not that the fingerprint could be incomplete -- off-GPU it
    legitimately is -- but that an incomplete one was still written into a
    claim-bearing artifact, where "gpu_uuid: null" is indistinguishable from a
    run nobody can attribute to a card. Missing evidence stops the write.
    """
    gaps = fingerprint_gaps(fp)
    if gaps:
        raise SystemExit(
            f"REFUSED: {what} would carry a null S19 fingerprint "
            f"({', '.join(gaps)}). The producer process that owned the card is "
            f"the authority on which card produced the tokens; a consumer may "
            f"COPY its runtime manifest but may never synthesize the values "
            f"from the registered expectation, and it may never write nulls.")
    return fp


# --------------------------------------------------------------------------
# producer-side runtime manifests
# --------------------------------------------------------------------------
#
# A run binding (results/agentic/hardware.json, written by env.lock_hardware) is
# the run's IMMUTABLE physical identity. A runtime manifest is per GPU-owning
# process/session: it records what that process measured, which process it was,
# and what it served. The rule the manifests exist to enforce:
#
#   the process that produced the model output is authoritative about the
#   hardware that produced it.
#
# So the HTTP evaluator never probes a card and never reads the run lock as if it
# were evidence about the server; it reads the SERVER's manifest. If the manifest
# and the run binding disagree, neither side wins -- the stage aborts.

MANIFEST_HASH_FIELD = "manifest_sha256"

MANIFEST_FIELDS = (
    # identity of the run and of this producer session
    "run_id", "session_id", "stage",
    # physical card, as MEASURED by the producer process
    "gpu_name", "gpu_uuid", "cuda_visible_bytes", "driver_version",
    "pci_bus_id", "compute_capability", "visible_ordinal",
    # how the card was made visible to it
    "cuda_device_order", "cuda_visible_devices",
    # the code and configuration it ran
    "git_sha", "config_hash", "engine_fingerprint", "enable_thinking_effective",
    # what it served
    "model", "adapter", "adapter_sha256", "served_adapter_name",
    # which OS process it was, and where it listened
    "host", "boot_id", "pid", "process_start", "port", "server_url",
    # when it measured, and when the driver saw it answer
    "captured_at_utc", "ready_at_utc",
)

# The only fields a COMPLETE manifest may leave null: a stage with no server has
# no port/url/alias, a base-model stage has no adapter, `ready_at_utc` is stamped
# by the driver once the producer answers /v1/models, and boot_id/process_start
# are absent on a kernel that does not expose them.
MANIFEST_NULLABLE = ("adapter", "adapter_sha256", "served_adapter_name",
                     "port", "server_url", "ready_at_utc", "boot_id",
                     "process_start")

_LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def manifest_dir(cfg: dict | None = None) -> pathlib.Path:
    """Beside the run binding, so one config key locates both."""
    return hardware_lock_path(cfg).parent / "manifests"


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    return "".join(keep).strip("-") or "x"


def manifest_path(stage: str, run_id: str | None = None,
                  session_id: str | None = None,
                  cfg: dict | None = None) -> pathlib.Path:
    """A UNIQUE path per producer session; never one shared "current" file."""
    run_id = run_id or DEFAULT_RUN_ID
    session_id = session_id or new_session_id(stage)
    return manifest_dir(cfg) / f"{_slug(stage)}.{_slug(run_id)}.{_slug(session_id)}.json"


def new_session_id(stage: str = "session") -> str:
    """Unique per process/session: stage, UTC second, pid and 4 random bytes."""
    import secrets

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(stage)}-{stamp}-{os.getpid()}-{secrets.token_hex(4)}"


def _boot_id() -> str | None:
    try:
        return pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except Exception:
        return None


def _process_start(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat: start time in clock ticks since boot.

    (pid, boot_id, start_ticks) is what makes a manifest's process identity
    unforgeable by PID reuse: a recycled PID has a different start time, so a
    stale manifest cannot pass as a live server.
    """
    try:
        raw = pathlib.Path(f"/proc/{int(pid)}/stat").read_text()
    except Exception:
        return None
    try:  # comm can contain spaces and parentheses; everything after ') ' is safe
        tail = raw[raw.rindex(") ") + 2:].split()
        return int(tail[19])
    except Exception:
        return None


def process_identity(pid: int | None = None) -> dict:
    pid = int(pid if pid is not None else os.getpid())
    return {"host": socket.gethostname(), "boot_id": _boot_id(), "pid": pid,
            "process_start": _process_start(pid)}


def process_is_current(rec: dict) -> tuple[bool, str]:
    """Is the process this manifest describes still the live one? (ok, reason)."""
    host = socket.gethostname()
    if rec.get("host") and rec["host"] != host:
        return False, (f"manifest host {rec['host']!r} is not this host {host!r}")
    boot = _boot_id()
    if rec.get("boot_id") and boot and rec["boot_id"] != boot:
        return False, ("the manifest was written before a reboot "
                       f"(boot_id {rec['boot_id']} != {boot})")
    pid = rec.get("pid")
    if not pid:
        return False, "the manifest records no producer pid"
    start = _process_start(int(pid))
    if start is None:
        return False, f"producer pid {pid} is no longer running"
    if rec.get("process_start") is not None and int(rec["process_start"]) != start:
        return False, (f"pid {pid} was recycled: process start {start} != the "
                       f"manifest's {rec['process_start']}")
    return True, "live"


def manifest_digest(rec: dict) -> str:
    """SHA-256 over the canonical manifest payload, excluding its own digest."""
    payload = {k: v for k, v in rec.items() if k != MANIFEST_HASH_FIELD}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict) -> pathlib.Path:
    """Write-then-rename, so a reader never sees a half-written attestation."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                              suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_runtime_manifest(record: dict, path: str | pathlib.Path | None = None,
                           cfg: dict | None = None) -> tuple[pathlib.Path, dict]:
    """Hash and atomically write one producer session's attestation."""
    missing = [k for k in MANIFEST_FIELDS if k not in record]
    if missing:
        raise SystemExit(f"REFUSED: a runtime manifest is missing "
                         f"{', '.join(missing)}; the manifest key set is frozen")
    extra = [k for k in record if k not in MANIFEST_FIELDS
             and k != MANIFEST_HASH_FIELD]
    if extra:
        raise SystemExit(f"REFUSED: a runtime manifest carries unregistered "
                         f"fields {', '.join(sorted(extra))}")
    rec = {k: record[k] for k in MANIFEST_FIELDS}
    rec[MANIFEST_HASH_FIELD] = manifest_digest(rec)
    p = pathlib.Path(path) if path else manifest_path(
        rec["stage"], rec["run_id"], rec["session_id"], cfg)
    write_json_atomic(p, rec)
    return p, rec


def read_runtime_manifest(path: str | pathlib.Path) -> dict:
    """Parse one manifest and verify its self-hash. Never probes a card."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(
            f"REFUSED: no producer runtime manifest at {p}. A claim-bearing "
            f"stage may not attest its own hardware: the process that produced "
            f"the tokens writes the manifest, and this stage COPIES it.")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"REFUSED: runtime manifest {p} is unreadable: {exc}")
    if not isinstance(rec, dict):
        raise SystemExit(f"REFUSED: runtime manifest {p} is not an object")
    got = rec.get(MANIFEST_HASH_FIELD)
    want = manifest_digest(rec)
    if got != want:
        raise SystemExit(
            f"REFUSED: runtime manifest {p} does not hash to its recorded "
            f"{MANIFEST_HASH_FIELD} ({got} != {want}); it was edited after "
            f"capture, so it is not evidence.")
    return rec


def manifest_gaps(rec: dict) -> list[str]:
    """Registered manifest fields that are absent or null (nullable ones aside)."""
    return [k for k in MANIFEST_FIELDS
            if k not in MANIFEST_NULLABLE
            and (rec.get(k) is None or rec.get(k) == "")]


def _refuse(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    parsed = urllib.parse.urlsplit(url if "//" in url else f"//{url}")
    if parsed.port:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def require_runtime_manifest(path: str | pathlib.Path | None, *,
                             run_id: str | None = None,
                             cfg: dict | None = None,
                             stage: str | None = None,
                             server: str | None = None,
                             model: str | None = None,
                             adapter: str | None = None,
                             served_adapter_name: str | None = None,
                             require_ready: bool = True,
                             require_live: bool = True) -> dict:
    """The fail-closed gate every claim-bearing consumer runs BEFORE any output.

    Verifies, in this order, and terminates on the first failure:
      the manifest exists and hashes to itself; every non-nullable field is
      non-null; the run_id is this run's; the measured card equals the run's
      immutable binding; the visibility settings are the registered PCI-ordered
      ones; config hash, engine fingerprint and effective thinking mode equal the
      registered contract; the producer stage, port, served model and adapter are
      the ones this consumer is about to talk to; the producer process is still
      the live one rather than a stale manifest; and the driver has countersigned
      that the producer answered.
    """
    cfg = cfg or load_config()
    rec = read_runtime_manifest(path)
    p = pathlib.Path(path)

    gaps = manifest_gaps(rec)
    if gaps:
        _refuse(f"runtime manifest {p} is incomplete ({', '.join(gaps)}). A "
                f"producer that could not measure its card must fail before it "
                f"produces output, not write nulls that read as 'unattributed'.")

    if run_id is not None and rec["run_id"] != run_id:
        _refuse(f"runtime manifest {p} belongs to run_id {rec['run_id']!r} and "
                f"this stage is {run_id!r}. A manifest from another run attests "
                f"nothing about this one.")

    hw = hardware_contract(cfg)
    lock = hardware_lock_path(cfg)
    if not lock.exists():
        _refuse(f"this run has no hardware binding at {lock}, so the producer "
                f"manifest {p} cannot be checked against the run's physical "
                f"identity. Bind the card first (agentlab.env pin).")
    try:
        binding = json.loads(lock.read_text(encoding="utf-8"))
    except Exception as exc:
        _refuse(f"the run binding {lock} is unreadable: {exc}")
    for key in ("gpu_uuid", "gpu_name", "cuda_visible_bytes"):
        want, got = binding.get(key), rec.get(key)
        if want is not None and got is not None and str(want) != str(got):
            _refuse(f"HARDWARE INTEGRITY: the producer measured {key}={got!r} and "
                    f"this run is bound to {want!r}. Neither side wins: the "
                    f"stage aborts (S19). A different card is a NEW run with its "
                    f"own run_id, locks, seeds, ledger and declaration.")

    if str(hw.get("registered_gpu", "")).lower() not in str(rec["gpu_name"]).lower():
        _refuse(f"the producer served on {rec['gpu_name']!r}, and this study is "
                f"registered on one {hw.get('expected_name')}.")
    if hw.get("cuda_visible_bytes") and int(hw["cuda_visible_bytes"]) != int(
            rec["cuda_visible_bytes"]):
        _refuse(f"the producer measured {rec['cuda_visible_bytes']} CUDA-visible "
                f"bytes and the registered hardware declares "
                f"{int(hw['cuda_visible_bytes'])} (known-wrong card, S19).")
    if rec["cuda_device_order"] != hw.get("cuda_device_order", "PCI_BUS_ID"):
        _refuse(f"the producer ran under CUDA_DEVICE_ORDER="
                f"{rec['cuda_device_order']!r}; the registered contract is "
                f"{hw.get('cuda_device_order', 'PCI_BUS_ID')!r}.")
    if str(rec["cuda_visible_devices"]).split(",")[0].strip() != str(
            hw.get("cuda_visible_devices", "0")):
        _refuse(f"the producer ran with CUDA_VISIBLE_DEVICES="
                f"{rec['cuda_visible_devices']!r}; this run is registered on "
                f"index {hw.get('cuda_visible_devices')}.")

    if rec["config_hash"] != config_hash():
        _refuse(f"the producer ran config_hash {rec['config_hash']} and this "
                f"stage reads {config_hash()}; the counts and gates changed "
                f"between the two.")
    contract = engine_contract(cfg)
    if _canonical(rec["engine_fingerprint"]) != _canonical(engine_fingerprint(cfg)):
        _refuse(f"the producer's engine fingerprint is not the registered one:\n"
                f"  producer: {json.dumps(rec['engine_fingerprint'], sort_keys=True)}\n"
                f"  registered: "
                f"{json.dumps(engine_fingerprint(cfg), sort_keys=True)}")
    if bool(rec["enable_thinking_effective"]) != bool(contract["enable_thinking"]):
        _refuse(f"the producer rendered with enable_thinking="
                f"{rec['enable_thinking_effective']!r}; the contract is "
                f"{contract['enable_thinking']!r}. That is a different policy.")

    if stage is not None and rec["stage"] != stage:
        _refuse(f"runtime manifest {p} was captured by stage {rec['stage']!r}, "
                f"not {stage!r}.")
    if server is not None:
        want_port = _port_of(server)
        host = urllib.parse.urlsplit(server).hostname
        if host not in _LOOPBACK:
            _refuse(f"--server {server} is not loopback; the producer manifest "
                    f"can only attest a server this box owns.")
        if rec.get("port") is None or int(rec["port"]) != int(want_port):
            _refuse(f"the producer listens on port {rec.get('port')!r} and this "
                    f"stage would talk to {want_port!r}. A manifest for another "
                    f"server attests nothing about the one answering.")
    if model is not None and rec["model"] != model:
        _refuse(f"the producer serves base model {rec['model']!r} and this stage "
                f"declares {model!r}.")
    if adapter:
        if not rec.get("adapter"):
            _refuse(f"this stage evaluates the adapter {adapter}, and the "
                    f"producer registered no LoRA at all: it would serve the "
                    f"BASE weights and the trace would label them trained.")
        if str(pathlib.Path(rec["adapter"]).resolve()) != str(
                pathlib.Path(adapter).resolve()):
            _refuse(f"the producer registered adapter {rec['adapter']!r} and this "
                    f"stage declares {adapter!r}.")
        if served_adapter_name and rec.get("served_adapter_name") != served_adapter_name:
            _refuse(f"the producer registered the LoRA under alias "
                    f"{rec.get('served_adapter_name')!r} and this stage requests "
                    f"{served_adapter_name!r}; the request would fall back to the "
                    f"base weights.")
    # A base arm against a LoRA-enabled server is legitimate and deliberately not
    # refused: the S8 pairing requirement is that both arms face the SAME server.

    if require_live:
        ok, why = process_is_current(rec)
        if not ok:
            _refuse(f"the producer session in {p} is not current: {why}. A stale "
                    f"manifest would attribute this stage's episodes to a process "
                    f"that no longer exists.")
    if require_ready and not rec.get("ready_at_utc"):
        _refuse(f"runtime manifest {p} has no ready_at_utc: the driver never "
                f"countersigned that this producer answered /v1/models, so "
                f"nothing has confirmed the attested engine is the one serving.")
    return rec


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def mark_manifest_ready(path: str | pathlib.Path, *, run_id: str | None = None,
                        cfg: dict | None = None, stage: str | None = None,
                        server: str | None = None) -> dict:
    """The driver's countersignature, after the producer answered /v1/models.

    Re-verifies the whole manifest (the producer is still the live process, the
    card still matches the run binding), stamps ready_at_utc and re-hashes, so
    the digest every trace row carries is the READY manifest's.
    """
    rec = require_runtime_manifest(path, run_id=run_id, cfg=cfg, stage=stage,
                                   server=server, require_ready=False)
    rec = dict(rec, ready_at_utc=rec.get("ready_at_utc") or now_utc())
    _, rec = write_runtime_manifest(rec, path, cfg)
    return rec


def fingerprint_from_manifest(rec: dict, cfg: dict | None = None) -> dict:
    """The S19 fingerprint COPIED from the producer, not measured by the caller.

    Deliberately a copy: the evaluator opens no CUDA context, reads no
    nvidia-smi, and never substitutes the registered A5000 expectation for a
    measurement. Whatever the producer measured is what the row claims.
    """
    fp = {"run_id": rec["run_id"],
          "git_sha": rec.get("git_sha"),
          "config_hash": rec.get("config_hash"),
          "gpu_name": rec.get("gpu_name"),
          "gpu_uuid": rec.get("gpu_uuid"),
          "cuda_visible_bytes": rec.get("cuda_visible_bytes"),
          "driver_version": rec.get("driver_version"),
          "engine_fingerprint": rec.get("engine_fingerprint"),
          "enable_thinking_effective": rec.get("enable_thinking_effective"),
          "timestamp_utc": now_utc()}
    return require_complete_fingerprint(fp, "a fingerprint copied from a manifest")


def checkpoint_tree_sha256(path: str | pathlib.Path | None) -> str | None:
    """Content hash of an adapter/checkpoint directory (or file), or None.

    Sorted relative path + file digest, so two directories with the same bytes
    hash the same and a silently swapped adapter cannot pass as the locked one.
    """
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(
        q for q in p.rglob("*") if q.is_file())
    for q in files:
        rel = q.name if p.is_file() else str(q.relative_to(p))
        h.update(rel.encode("utf-8") + b"\0")
        h.update(hashlib.sha256(q.read_bytes()).hexdigest().encode("ascii") + b"\0")
    return h.hexdigest()


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
                  enable_thinking: bool | None = None,
                  manifest: str | pathlib.Path | None = None) -> float:
    """Append one completed chunk; returns cumulative hours after the append.

    The row is a receipt, not a tally. The protocol promises timestamps, GPU
    identity, git SHA and work counts on every ledger row, and the earlier
    implementation wrote {stage, minutes, cumulative_h}: a ledger that cannot say
    which card spent the hours cannot support a same-card claim.

    `manifest` is the producer session that actually spent the minutes. When it is
    given the row COPIES that session's snapshot (and records its digest and
    session id) instead of recomputing a fingerprint at append time -- which is
    how a served stage's receipt ends up carrying the same UUID as its traces
    rather than whatever the appending process can see. The process need not still
    be alive: a session is charged after its server has been stopped.

    A second physical GPU inside one run is fatal here rather than at analysis
    time, because mixing two cards inside one claim is the S19 failure itself.
    """
    cfg = cfg or load_config()
    session = None
    if manifest:
        rec = read_runtime_manifest(manifest)
        gaps = manifest_gaps(rec)
        if gaps:
            raise SystemExit(f"REFUSED: the producer manifest {manifest} is "
                             f"incomplete ({', '.join(gaps)}); a ledger row may "
                             f"not charge hours to an unattributed card.")
        fp = fingerprint_from_manifest(rec, cfg)
        session = {"runtime_manifest_sha256": rec[MANIFEST_HASH_FIELD],
                   "session_id": rec["session_id"],
                   "producer_pid": rec["pid"]}
    else:
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
    if session:
        row.update(session)
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
