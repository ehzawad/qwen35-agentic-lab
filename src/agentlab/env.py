"""Shared config: model resolution, GPU pinning, paths, model/processor loading.

Everything in this lab funnels through here so that a stage never disagrees with
another stage about which model or which GPU it is using.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
DATA = ROOT / "data"

# Qwen3.5 small series (Apache-2.0, released 2026-03-02). All four sizes share the
# same hybrid Gated-DeltaNet/Gated-Attention stack, 262K native context, thinking
# mode, and tool calling -- so the pipeline is size-agnostic. Override with
# QWEN_MODEL=Qwen/Qwen3.5-2B for faster iteration.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
MODEL = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)

# Short slug used to namespace adapter/checkpoint dirs, so switching model size
# does not silently overwrite another size's artifacts.
SLUG = MODEL.split("/")[-1].replace(".", "").lower()

SFT_DIR = OUT / f"{SLUG}-sft-lora"
GRPO_DIR = OUT / f"{SLUG}-grpo-lora"
MERGED_DIR = OUT / f"{SLUG}-merged"


# This study is preregistered as a SINGLE-CARD NVIDIA RTX A5000 run, so the card
# check is NOT optional any more. EXPECT_GPU defaults to the registered card and
# an empty EXPECT_GPU no longer disables the check -- "unset it to make the error
# go away" is how a run ends up measured on a card nobody declared. Pointing it
# at a different card is possible, but a study stage refuses that anyway
# (require_registered_gpu): another card is a separate registered run with its
# own run_id, locks, seeds, ledger and hardware declaration.
REGISTERED_GPU = "A5000"
EXPECT_GPU = (os.environ.get("EXPECT_GPU") or "").strip() or REGISTERED_GPU


def _hardware(cfg=None) -> dict:
    from agentlab.suite.configio import hardware_contract

    return hardware_contract(cfg)


def require_pci_bus_order() -> None:
    """Refuse a run whose CUDA ordinals need not agree with nvidia-smi's.

    Without PCI_BUS_ID, `CUDA_VISIBLE_DEVICES=0` means "the fastest card", which
    on a heterogeneous box is not the card the operator read off nvidia-smi. The
    registered contract names PCI ordering, so this is a refusal, not a warning.
    """
    order = os.environ.get("CUDA_DEVICE_ORDER", "")
    if order != "PCI_BUS_ID":
        sys.exit(
            f"REFUSED: CUDA_DEVICE_ORDER is {order or 'unset (FASTEST_FIRST)'}, "
            f"and the registered hardware contract is PCI_BUS_ID. Under "
            f"FASTEST_FIRST the index you pin need not be the card nvidia-smi "
            f"shows.\n  Fix: export CUDA_DEVICE_ORDER=PCI_BUS_ID")


def require_registered_index(cfg=None) -> str:
    """The pinned index must be the registered one, and exactly one card."""
    hw = _hardware(cfg)
    want = str(hw.get("cuda_visible_devices", "0"))
    got = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not got:
        sys.exit("REFUSED: a study stage needs an explicit pin, e.g. "
                 f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={want} "
                 f"EXPECT_GPU={REGISTERED_GPU}")
    if got != want:
        sys.exit(
            f"REFUSED: CUDA_VISIBLE_DEVICES={got}, and this run is registered on "
            f"index {want} under PCI ordering ({hw.get('expected_name')}). "
            f"Another card is a SEPARATE registered run with its own run_id, "
            f"locks, seeds, ledger and hardware declaration -- not an "
            f"opportunistic branch of this one.")
    return got


def require_registered_gpu(cfg=None, run_id: str | None = None) -> dict:
    """The full hardware veto every STUDY stage runs before it spends a GPU-hour.

    Order, index, card name, exclusivity, and then the run's UUID lock. Returns
    the hardware fingerprint it wrote, so the ledger and every trace row carry
    the same identity rather than each layer guessing.
    """
    from agentlab.suite.configio import hardware_contract

    hw = hardware_contract(cfg)
    require_pci_bus_order()
    require_registered_index(cfg)
    name = require_single_gpu(hw.get("registered_gpu", REGISTERED_GPU))
    return lock_hardware(cfg, run_id=run_id, name=name)


def lock_hardware(cfg=None, run_id: str | None = None,
                  name: str | None = None) -> dict:
    """Measure and bind this run's physical card. Requires an initialized CUDA.

    `cuda_visible_bytes` comes from torch's measured total_memory, which is the
    number the registered contract and S19 use (25,282,805,760 on this card).
    nvidia-smi reports the 24,564 MiB BOARD total instead, so the two must never
    be substituted for one another.

    The first GPU stage writes the lock; every later stage -- including the
    pure-HTTP evaluator, which never opens a CUDA context -- reads it and must
    agree with it. A second physical card inside one run is fatal here.
    """
    import torch

    from agentlab.suite import configio

    cfg = cfg or configio.load_config()
    hw = configio.hardware_contract(cfg)
    props = torch.cuda.get_device_properties(0)
    smi = configio.nvidia_smi_identity(
        (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip())
    record = {
        "gpu_name": name or torch.cuda.get_device_name(0),
        "gpu_uuid": smi["gpu_uuid"],
        "cuda_visible_bytes": int(props.total_memory),
        "driver_version": smi["driver_version"],
        "pci_bus_id": smi["pci_bus_id"],
        "compute_capability": f"{props.major}.{props.minor}",
        "visible_ordinal": 0,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "run_id": run_id or configio.DEFAULT_RUN_ID,
        "locked_at_utc": configio.now_utc(),
    }
    expected_bytes = hw.get("cuda_visible_bytes")
    if expected_bytes and int(expected_bytes) != record["cuda_visible_bytes"]:
        sys.exit(
            f"REFUSED: the pinned card exposes {record['cuda_visible_bytes']} "
            f"CUDA-visible bytes, the registered hardware declares "
            f"{int(expected_bytes)}. A known-wrong card is a BUG (S19), not a "
            f"tolerance.")
    path = configio.hardware_lock_path(cfg)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for key in ("gpu_uuid", "gpu_name", "cuda_visible_bytes"):
            if prior.get(key) and record.get(key) and prior[key] != record[key]:
                sys.exit(
                    f"REFUSED: this run is locked to {key}={prior[key]!r} and "
                    f"the pinned card reports {record[key]!r}. One run may not "
                    f"span two physical cards (S19).")
        return prior
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"[gpu] locked {record['gpu_name']} {record['gpu_uuid']} "
          f"({record['cuda_visible_bytes']} bytes) -> {path}")
    return record


def require_single_gpu(expect: str | None = None) -> str:
    """Assert exactly one GPU is visible, that it is the intended one, and name it.

    Worth being strict here. CUDA_VISIBLE_DEVICES indexes in *CUDA* order, which
    defaults to FASTEST_FIRST -- NOT the PCI order that nvidia-smi prints. On a
    heterogeneous multi-GPU machine those two orderings can disagree, so the
    index you read off nvidia-smi can select a different card than you meant,
    and a run sized for the bigger card then OOMs on the smaller one with no
    hint as to why. Export CUDA_DEVICE_ORDER=PCI_BUS_ID to make them agree.

    `expect` defaults to EXPECT_GPU, which defaults to the REGISTERED card. There
    is no way to switch the check off from the environment: that is deliberate.
    """
    import torch

    expect = (expect or EXPECT_GPU).strip() or REGISTERED_GPU

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible -- check CUDA_VISIBLE_DEVICES")
    n = torch.cuda.device_count()
    if n != 1:
        sys.exit(
            f"expected exactly 1 visible GPU, found {n}.\n"
            f"  Expose one, e.g.: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index> make smoke"
        )

    warn_fast_path()

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST (default)")
    print(
        f"[gpu] {name}  ({total:.1f} GiB)  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  "
        f"CUDA_DEVICE_ORDER={order}"
    )

    if expect.lower() not in name.lower():
        sys.exit(
            f"pinned the wrong GPU: wanted {expect}, got {name}.\n"
            f"  CUDA_VISIBLE_DEVICES indexes in CUDA order, not nvidia-smi order.\n"
            f"  Fix: export CUDA_DEVICE_ORDER=PCI_BUS_ID and pick the right index.\n"
            f"  This check has no off switch: this study is registered on one "
            f"{REGISTERED_GPU} and a result from another card belongs to another run."
        )
    return name


def warn_fast_path() -> bool:
    """Warn if the Gated DeltaNet fast-path kernels are installed.

    transformers advertises them in a log line that reads like an invitation, but
    flash-linear-attention 0.5.2 + causal-conv1d 1.6.2 segfault the forward pass
    on torch 2.11.0+cu130 -- exit 139, no traceback, indistinguishable from an
    OOM. The torch fallback is slower and correct. Retest before re-adding them.
    """
    import importlib.util

    present = [m for m in ("fla", "causal_conv1d") if importlib.util.find_spec(m)]
    if present:
        print(
            f"[warn] fast-path kernels installed ({', '.join(present)}). These segfaulted\n"
            f"       the forward pass on this stack. If training dies with exit 139, run:\n"
            f"       uv pip uninstall --python .venv flash-linear-attention fla-core causal-conv1d"
        )
    return bool(present)


def load_processor(model_id: str = MODEL):
    """Load the tokenizer/processor.

    Qwen3.5 ships as a natively multimodal checkpoint, so the repo carries a
    processor. We only ever feed it text, but the chat template (thinking mode,
    tool schemas) lives on the tokenizer either way.
    """
    from transformers import AutoProcessor, AutoTokenizer

    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        # Expose .apply_chat_template / .pad_token uniformly.
        if not hasattr(proc, "apply_chat_template") and hasattr(proc, "tokenizer"):
            return proc.tokenizer
        return proc
    except Exception:
        return AutoTokenizer.from_pretrained(model_id)


def get_tokenizer(proc):
    """Return the underlying tokenizer whether given a processor or a tokenizer."""
    return getattr(proc, "tokenizer", proc)


def load_model(model_id: str = MODEL, **kw):
    """Load the policy model, probing for the right auto-class.

    Qwen3.5 is registered as image-text-to-text (AutoModelForMultimodalLM), which
    is not the class a text-only script would reach for by default. Probe in
    order rather than hardcoding, so this keeps working if the registration
    changes or a text-only sibling is used instead.
    """
    import torch
    import transformers

    kw.setdefault("dtype", torch.bfloat16)
    kw.setdefault("device_map", None)

    candidates = [
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
    ]
    last = None
    for cls_name in candidates:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kw)
            print(f"[model] {model_id} via {cls_name}")
            return model
        except Exception as e:  # wrong class for this checkpoint -- try the next
            last = e
    raise RuntimeError(f"could not load {model_id}; last error: {last}")


def model_auto_class(model_id: str = MODEL) -> str:
    """Name of the auto-class that matches this checkpoint, without loading weights."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    archs = getattr(cfg, "architectures", None) or []
    return archs[0] if archs else type(cfg).__name__


# Sampling presets straight from the Qwen3.5 model card. Thinking mode is the
# default for this checkpoint; the non-thinking numbers are for enable_thinking=False.
SAMPLING_THINKING = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
SAMPLING_THINKING_CODE = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0)
SAMPLING_INSTRUCT = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)


def main() -> None:
    """`python -m agentlab.env pin` -- bind this run's card; `... fingerprint`.

    `pin` is what a served (HTTP) stage runs before its server starts: the
    evaluator process never opens a CUDA context, so without this the card that
    produced its episodes would be unrecorded, and S19 treats missing provenance
    as INCONCLUSIVE rather than as an assumed A5000.
    """
    import argparse

    from agentlab.suite import configio

    ap = argparse.ArgumentParser(description="hardware pin / fingerprint")
    ap.add_argument("cmd", choices=("pin", "fingerprint", "contract"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.cmd == "pin":
        record = require_registered_gpu(run_id=args.run_id)
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.cmd == "fingerprint":
        print(json.dumps(configio.fingerprint(args.run_id), indent=2, sort_keys=True))
    else:
        print(json.dumps(configio.engine_contract(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
