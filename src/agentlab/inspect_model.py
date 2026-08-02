"""Print the module tree so LoRA targets are chosen from fact, not guesswork.

Qwen3.5 interleaves Gated DeltaNet and Gated Attention layers, which do not share
projection names. Run this once per new checkpoint before trusting a target list.
"""

from __future__ import annotations

import argparse
import collections

from . import env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    print(f"[config] {type(cfg).__name__}  model_type={getattr(cfg, 'model_type', '?')}")
    for k in ("num_hidden_layers", "hidden_size", "full_attention_interval",
              "linear_num_value_heads", "num_attention_heads", "max_position_embeddings"):
        v = getattr(cfg, k, None)
        if v is None and hasattr(cfg, "text_config"):
            v = getattr(cfg.text_config, k, None)
        if v is not None:
            print(f"         {k} = {v}")

    model = env.load_model(args.model, dtype="auto", device_map="meta")

    import torch.nn as nn

    leaf = collections.Counter()
    vision = collections.Counter()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        short = name.split(".")[-1]
        if any(tag in name.lower() for tag in ("vision", "visual", "patch_embed", "merger")):
            vision[short] += 1
        else:
            leaf[short] += 1

    print("\n[linear modules -- language stack]")
    for name, n in leaf.most_common():
        print(f"  {n:5d}  {name}")
    if vision:
        print("\n[linear modules -- vision tower (excluded from LoRA)]")
        for name, n in vision.most_common():
            print(f"  {n:5d}  {name}")

    print(f"\ntotal language linears: {sum(leaf.values())}, vision linears: {sum(vision.values())}")


if __name__ == "__main__":
    main()
