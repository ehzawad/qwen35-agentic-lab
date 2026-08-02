"""Stage 1 -- supervised fine-tuning on tool calls (LoRA).

Teaches the one thing RL cannot bootstrap cheaply: emitting a syntactically valid
call against the tool list it was actually given, instead of inventing a function
or answering in prose. GRPO in stage 3 assumes this already works -- if the policy
never produces a parseable call, every rollout scores zero and the gradient is
noise.
"""

from __future__ import annotations

import argparse

from . import env
from .data import build_sft
from .peft_cfg import describe, lora_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--n", type=int, default=4000, help="training examples")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env.require_single_gpu()
    out_dir = args.out or str(env.SFT_DIR)

    from trl import SFTConfig, SFTTrainer

    ds = build_sft(n=args.n)
    split = ds.train_test_split(test_size=0.02, seed=0)
    print(f"[data] train={len(split['train'])} eval={len(split['test'])} cols={ds.column_names}")

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        # 262K-context model on a 48 GB card: cap the sequence and let packing off,
        # so a single long xlam row cannot blow the step.
        packing=False,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora_config(r=args.rank),
    )
    describe(trainer.model)

    trainer.train()
    trainer.save_model(out_dir)
    print(f"[done] adapter -> {out_dir}")


if __name__ == "__main__":
    main()
