"""Stage 4a -- fold a LoRA adapter into the base weights.

vLLM can serve adapters directly, but a merged checkpoint is the thing you ship:
no adapter plumbing at inference, and no chance of serving the base by accident
because an adapter path was wrong.
"""

from __future__ import annotations

import argparse
import shutil

from . import env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or str(env.MERGED_DIR)

    from peft import PeftModel

    print(f"[base]    {args.model}")
    print(f"[adapter] {args.adapter}")

    # Merge on CPU: a 4B merge needs no GPU, and this keeps the card free for a
    # training run happening at the same time.
    model = env.load_model(args.model, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)

    # Save the processor *and* the tokenizer under it. The chat template lives
    # with the tokenizer; without it the merged repo renders no tool schemas and
    # every tool call silently disappears at serve time.
    proc = env.load_processor(args.model)
    proc.save_pretrained(out_dir)
    tok = env.get_tokenizer(proc)
    if tok is not proc and getattr(tok, "chat_template", None):
        tok.save_pretrained(out_dir)

    print(f"[done] merged -> {out_dir}")
    print(f"       serve with: bash scripts/serve.sh {out_dir}")


if __name__ == "__main__":
    main()
