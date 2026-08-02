"""Stage 2 -- preference alignment with DPO (LoRA).

This is the RLHF leg of the path. DPO fits the policy to pairwise human
preferences through a classification loss, so it needs no reward model and no
sampling loop -- the cheap way to see preference optimisation end to end.

Run `reward.py` alongside it if you want the explicit reward-model step that
classical RLHF (reward model -> PPO/GRPO) would use instead.
"""

from __future__ import annotations

import argparse

from . import env
from .data import build_preference
from .peft_cfg import describe, lora_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--adapter", default=None, help="SFT adapter to continue from")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1, help="KL strength; higher = stays closer to ref")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env.require_single_gpu()
    out_dir = args.out or str(env.DPO_DIR)

    from trl import DPOConfig, DPOTrainer

    ds = build_preference(n=args.n, explicit_prompt=True)
    split = ds.train_test_split(test_size=0.02, seed=0)
    print(f"[data] train={len(split['train'])} eval={len(split['test'])} cols={ds.column_names}")

    cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        beta=args.beta,
        # TRL v1 dropped max_prompt_length -- max_length now bounds the whole
        # sequence. truncation_mode is left at its default: 'keep_end' is
        # deprecated for removal in v2.0.0, so setting it just buys a warning.
        max_length=1536,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    # With a PEFT policy, TRL uses the adapter-disabled base as the implicit
    # reference model -- so no second full copy of the weights on the card.
    trainer = DPOTrainer(
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
