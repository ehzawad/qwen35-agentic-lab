"""Shared LoRA configuration.

One definition so the SFT adapter, the DPO adapter and the GRPO adapter are
rank-compatible and can be stacked or compared without surprises.
"""

from __future__ import annotations

import os

# Qwen3.5 is a hybrid stack: Gated DeltaNet layers (in_proj/out_proj-style names)
# interleaved 3:1 with Gated Attention layers (q/k/v/o_proj). "all-linear" covers
# both without hardcoding names that differ between the two layer types -- but it
# would also reach into the vision tower, which we never train here, so the
# vision modules are excluded explicitly.
DEFAULT_TARGETS = os.environ.get("LORA_TARGETS", "all-linear")
EXCLUDE = ["vision", "visual", "image_", "patch_embed", "merger", "lm_head"]


def lora_config(r: int = 32, alpha: int | None = None, dropout: float = 0.05,
                targets: str | list[str] = DEFAULT_TARGETS, task_type: str = "CAUSAL_LM"):
    from peft import LoraConfig

    if isinstance(targets, str) and targets != "all-linear":
        targets = [t.strip() for t in targets.split(",") if t.strip()]

    return LoraConfig(
        r=r,
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        target_modules=targets,
        exclude_modules=EXCLUDE if targets == "all-linear" else None,
        bias="none",
        task_type=task_type,
    )


def describe(model) -> None:
    """Print the trainable-parameter split, so a silently-frozen run is obvious."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M  ({100*trainable/total:.3f}%)")
