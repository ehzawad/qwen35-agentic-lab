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
"""

from __future__ import annotations

import argparse

from . import env
from .data import build_distill_sft
from .peft_cfg import describe, lora_config
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
    return ap


def main() -> None:
    cfg = load_config()
    args = build_parser(cfg["sft"]).parse_args()

    env.require_registered_gpu(cfg)
    args.out = args.out or str(env.SFT_DIR)

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
    print(f"[done] adapter -> {args.out}")


if __name__ == "__main__":
    main()
