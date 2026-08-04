"""Shared LoRA configuration.

One definition so the SFT adapter and the GRPO adapter are rank-compatible and
can be stacked or compared without surprises.
"""

from __future__ import annotations

import os

# Qwen3.5 is a hybrid stack: Gated DeltaNet layers (in_proj/out_proj-style names)
# interleaved 3:1 with Gated Attention layers (q/k/v/o_proj). "all-linear" covers
# both without hardcoding names that differ between the two layer types -- but it
# would also reach into the vision tower, which we never train here, so the
# vision modules are excluded explicitly.
DEFAULT_TARGETS = os.environ.get("LORA_TARGETS", "all-linear")

# MUST be a regex string, not a list. PEFT (tuners_utils._check_exclude, ~L1907)
# matches a *list* only by exact key or `key.endswith(f".{entry}")`, so "visual"
# never matches `model.visual.blocks.0.attn.proj` and the whole vision tower was
# silently getting LoRA. A *string* is applied with re.fullmatch instead, which
# is the only form that can express "anything under the vision tower".
#
# This was invisible because PEFT only warns when exclude_modules matched
# NOTHING -- and "lm_head" did match, so the warning never fired while 16.7% of
# every adapter (196 of 692 tensors, 13.1M params) trained on a vision stack
# that never sees an image in this pipeline.
EXCLUDE = r".*\.(visual|vision_tower|vision_model)\..*|.*\.(patch_embed|merger)\..*|lm_head"


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
        # LoRA on a vision tower that never sees an image would also be merged
        # into the served checkpoint, so this is wasted VRAM at inference too.
        bias="none",
        task_type=task_type,
    )


def policy_and_peft(model_id: str, adapter: str | None, rank: int = 32):
    """Return the (model, peft_config) pair a TRL trainer should be given.

    This is what makes the stages an actual pipeline rather than four
    independent runs from base.

      adapter is None -> (model_id, LoraConfig)   TRL attaches a fresh adapter
      adapter given   -> (PeftModel, None)        continue training that adapter

    The second form must pass `peft_config=None`. TRL 1.9.2 raises outright if a
    PeftModel arrives alongside a peft_config:

        "You passed a `PeftModel` instance together with a `peft_config` to the
         trainer. Please first merge and unload the existing adapter, save the
         resulting base model, and then pass that base model along with the new
         `peft_config` to the trainer."

    So continuing the same adapter (here) and merge-then-new-adapter (via
    `merge.py`) are the only two legal chainings. Continuing is the default
    because it keeps one lineage of weights and costs no 8.5 GB merge per stage;
    merge first if you need to change the rank between stages.

    `is_trainable=True` is load-bearing -- PeftModel.from_pretrained defaults it
    to False, which loads the adapter frozen and would train nothing at all.
    """
    if adapter is None:
        return model_id, lora_config(r=rank)

    from peft import PeftModel

    from . import env

    base = env.load_model(model_id)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            f"adapter {adapter} loaded with zero trainable parameters -- "
            f"is_trainable did not take effect, the run would be a no-op"
        )
    print(f"[chain] continuing adapter {adapter} ({trainable/1e6:.1f}M trainable)")
    return model, None


def describe(model) -> None:
    """Print the trainable-parameter split, so a silently-frozen run is obvious."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M  ({100*trainable/total:.3f}%)")
