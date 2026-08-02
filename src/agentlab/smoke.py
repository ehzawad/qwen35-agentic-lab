"""Stage 0 -- prove the base checkpoint works before spending a GPU-hour on it.

Three checks, in increasing order of what they can break:
  1. plain generation, thinking mode on and off
  2. the chat template actually renders our tool schemas
  3. a real multi-turn agent loop: model calls a tool, we execute it, it answers

Run this first. If stage 0 is wrong, every reward number downstream is fiction.
"""

from __future__ import annotations

import argparse
import time

from . import env
from .chat import render, run_agent_loop, strip_thinking, extract_thinking
from .tools import TOOLS, tool_schemas


def check_generation(model, proc) -> None:
    import torch

    print("\n=== 1. generation ===")
    for thinking in (True, False):
        msgs = [{"role": "user", "content": "In one sentence: why does GRPO not need a value network?"}]
        text = render(proc, msgs, enable_thinking=thinking)
        tok = env.get_tokenizer(proc)
        inputs = tok([text], return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                **({"temperature": 1.0, "top_p": 0.95, "top_k": 20} if thinking
                   else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}),
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = tok.decode(gen, skip_special_tokens=True)
        dt = time.time() - t0
        ntok = len(gen)
        think = extract_thinking(raw)
        print(f"\n-- thinking={thinking}  {ntok} tok in {dt:.1f}s ({ntok/dt:.1f} tok/s)")
        if think:
            print(f"   <think> {len(think)} chars")
        print(f"   {strip_thinking(raw)[:400]}")


def check_template(proc) -> None:
    print("\n=== 2. tool schemas in the chat template ===")
    schemas = tool_schemas()
    for s in schemas:
        fn = s["function"]
        print(f"  {fn['name']}({', '.join(fn['parameters']['properties'])})")
    msgs = [{"role": "user", "content": "What is 17 * 23?"}]
    text = render(proc, msgs, tools=schemas)
    for name in (f["function"]["name"] for f in schemas):
        assert name in text, f"tool {name!r} did not reach the rendered prompt"
    print(f"  rendered prompt: {len(text)} chars, all {len(schemas)} tools present")


def check_agent_loop(model, proc) -> None:
    print("\n=== 3. agent loop (model -> tool -> model) ===")
    schemas = tool_schemas()
    tasks = [
        "What is 4871 * 209, minus 1337? Use the calculator.",
        "How many gigabytes of VRAM does the A6000 have? Look it up.",
        "Convert 26.2 miles to kilometres.",
    ]
    for q in tasks:
        print(f"\n-- {q}")
        # Thinking mode spends most of a small budget before the call is emitted,
        # so give the loop real headroom rather than truncating mid-tool-call.
        _, final = run_agent_loop(
            model, proc, [{"role": "user", "content": q}], schemas,
            max_turns=4, max_new_tokens=1024,
            temperature=0.7, top_p=0.8, top_k=20,
        )
        print(f"  final: {final[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir to load on top")
    ap.add_argument("--skip", nargs="*", default=[], choices=["gen", "template", "agent"])
    args = ap.parse_args()

    env.require_single_gpu()
    print(f"[arch] {env.model_auto_class(args.model)}")

    proc = env.load_processor(args.model)
    model = env.load_model(args.model)
    model = model.to("cuda").eval()

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"[adapter] {args.adapter}")

    if "template" not in args.skip:
        check_template(proc)
    if "gen" not in args.skip:
        check_generation(model, proc)
    if "agent" not in args.skip:
        check_agent_loop(model, proc)

    print(f"\nsmoke ok -- {len(TOOLS)} tools, model {args.model}")


if __name__ == "__main__":
    main()
