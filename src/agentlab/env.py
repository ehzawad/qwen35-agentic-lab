"""Shared config: model resolution, GPU pinning, paths, model/processor loading.

Everything in this lab funnels through here so that a stage never disagrees with
another stage about which model or which GPU it is using.
"""

from __future__ import annotations

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
DPO_DIR = OUT / f"{SLUG}-dpo-lora"
GRPO_DIR = OUT / f"{SLUG}-grpo-lora"
MERGED_DIR = OUT / f"{SLUG}-merged"


# The box is shared and heterogeneous: A5000 (24 GB, bus 3B:00.0) and A6000
# (48 GB, bus AF:00.0). We want the A6000.
EXPECT_GPU = os.environ.get("EXPECT_GPU", "A6000")


def require_single_gpu(expect: str | None = EXPECT_GPU) -> str:
    """Assert exactly one GPU is visible, that it is the intended one, and name it.

    Worth being strict here. CUDA_VISIBLE_DEVICES indexes in *CUDA* order, which
    defaults to FASTEST_FIRST -- not the PCI order nvidia-smi prints. On this box
    that inverts the two cards, so CUDA_VISIBLE_DEVICES=1 lands on the A5000 and
    a 48 GB-shaped run OOMs at 24 GB with no hint as to why. Setting
    CUDA_DEVICE_ORDER=PCI_BUS_ID makes the two orderings agree.
    """
    import torch

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible -- check CUDA_VISIBLE_DEVICES")
    n = torch.cuda.device_count()
    if n != 1:
        sys.exit(
            f"expected exactly 1 visible GPU, found {n}. "
            f"Set CUDA_VISIBLE_DEVICES=1 (with CUDA_DEVICE_ORDER=PCI_BUS_ID) to pin the A6000."
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

    if expect and expect.lower() not in name.lower():
        sys.exit(
            f"pinned the wrong GPU: wanted {expect}, got {name}.\n"
            f"  CUDA_VISIBLE_DEVICES indexes in CUDA order, not nvidia-smi order.\n"
            f"  Fix: export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1\n"
            f"  Or set EXPECT_GPU= to disable this check."
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
