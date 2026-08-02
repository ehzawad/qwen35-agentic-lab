"""Stage 2b -- train an explicit reward model (the classical RLHF leg).

DPO folds the reward into the policy loss. The older RLHF recipe keeps them
separate: fit a scalar reward model on preferences, then optimise the policy
against it with PPO/GRPO. That separation is worth seeing, because it is what
lets you reuse one reward model across many policies -- and it is where reward
hacking becomes visible.

Reward models are conventionally smaller than the policy, so this defaults to the
0.8B sibling rather than the 4B policy.
"""

from __future__ import annotations

import argparse

from . import env
from .data import build_preference
from .peft_cfg import describe, lora_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env.require_single_gpu()
    out_dir = args.out or str(env.OUT / f"{args.model.split('/')[-1].lower()}-reward")

    from trl import RewardConfig, RewardTrainer

    # RewardTrainer wants the implicit-prompt form: chosen/rejected as complete
    # conversations, scored end to end.
    ds = build_preference(n=args.n, explicit_prompt=False)
    split = ds.train_test_split(test_size=0.02, seed=0)
    print(f"[data] train={len(split['train'])} eval={len(split['test'])} cols={ds.column_names}")

    cfg = RewardConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    # Build the scoring model explicitly rather than passing a model id. A
    # sequence-classification head carries no pad_token_id of its own, and
    # without one the trainer cannot tell where a sequence ends:
    #   ValueError: Cannot handle batch sizes > 1 if no padding token is defined.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
        print(f"[reward] set pad_token_id = {model.config.pad_token_id}")

    trainer = RewardTrainer(
        model=model,
        args=cfg,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
        peft_config=lora_config(r=args.rank, task_type="SEQ_CLS"),
    )
    describe(trainer.model)

    trainer.train()
    trainer.save_model(out_dir)
    print(f"[done] reward model -> {out_dir}")
    print("       feed it to GRPO with: reward_funcs=[<this model>] instead of the verifiable rewards")


if __name__ == "__main__":
    main()
