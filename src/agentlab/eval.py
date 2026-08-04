"""Stage 4b -- measure whether any of the training actually helped.

Runs the real agent loop over held-out GSM8K and reports four numbers:

  accuracy        boxed answer matches the reference
  tool_use_rate   the rollout called at least one tool
  tool_error_rate a call was made but the tool rejected it (bad schema/args)
  turns           mean assistant turns per episode

accuracy alone is not enough: a policy that stops calling tools and guesses can
hold accuracy steady while the agentic behaviour you trained for quietly dies.
"""

from __future__ import annotations

import argparse
import json
import time

from . import env, trace
from .chat import boxed_answer, run_agent_loop
from .grpo import (
    REWARD_NAMES,
    REWARD_WEIGHTS,
    correctness_reward,
    format_reward,
    tool_use_reward,
)
from .data import build_eval
from .tools import tool_schemas


def _numeric(s):
    from .chat import numeric_answer

    return numeric_answer(s)



def evaluate(model, proc, n: int, max_turns: int, max_new_tokens: int,
             enable_thinking: bool, verbose: bool) -> dict:
    ds = build_eval(n=n)
    schemas = tool_schemas()

    correct = used_tool = tool_error = 0
    turns_total = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        episode = (
            trace.Episode(ex["prompt"][-1]["content"], index=i, ground_truth=ex["ground_truth"])
            if trace.enabled()
            else None
        )
        convo, final = run_agent_loop(
            model, proc, list(ex["prompt"]), schemas,
            max_turns=max_turns, max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking, verbose=False, episode=episode,
            temperature=0.7, top_p=0.8, top_k=20,
        )

        got, want = _numeric(boxed_answer(final or "")), _numeric(ex["ground_truth"])
        ok = got is not None and want is not None and abs(got - want) < 1e-4
        correct += ok
        if episode is not None:
            # Score with the ACTUAL GRPO reward functions rather than stuffing
            # predicted/expected into a field labelled "rewards". This is what
            # makes an eval trace comparable with a training trace.
            comp = [m for m in convo if m.get("role") in ("assistant", "tool")]
            # ONE list object for all three calls. grpo._record keys its buffer
            # on id(completions), so three separate `[comp]` literals would never
            # complete a triple -- and would emit sporadic records with mismatched
            # scores whenever CPython recycled a freed id.
            batch = [comp]
            rewards = {
                "correctness": correctness_reward(batch, [ex["ground_truth"]])[0],
                "tool_use": tool_use_reward(batch)[0],
                "format": format_reward(batch)[0],
            }
            rewards["total"] = round(
                sum(w * rewards[n] for n, w in zip(REWARD_NAMES, REWARD_WEIGHTS)), 4
            )
            episode.finish(final or "", ok=bool(ok), predicted=got, expected=want, **rewards)

        calls = [m for m in convo if m.get("role") == "assistant" and m.get("tool_calls")]
        results = [m for m in convo if m.get("role") == "tool"]
        used_tool += bool(calls)
        tool_error += any(str(m.get("content", "")).startswith("error:") for m in results)
        turns_total += sum(1 for m in convo if m.get("role") == "assistant")

        if verbose and i < 5:
            print(f"  [{i}] {'OK ' if ok else 'MISS'} got={got} want={want} calls={len(calls)}")

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(ds)}  acc={correct/(i+1):.3f}  tool={used_tool/(i+1):.3f}")

    n_done = len(ds)
    return {
        "n": n_done,
        "accuracy": correct / n_done,
        "tool_use_rate": used_tool / n_done,
        "tool_error_rate": tool_error / n_done,
        "mean_turns": turns_total / n_done,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to evaluate; omit for the base model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--tag", default=None, help="label for the results file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    env.require_single_gpu()
    proc = env.load_processor(args.model)
    model = env.load_model(args.model).to("cuda").eval()

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"[adapter] {args.adapter}")

    tag = args.tag or ("base" if not args.adapter else args.adapter.rstrip("/").split("/")[-1])
    print(f"[eval] {tag}: {args.n} held-out GSM8K problems, thinking={not args.no_thinking}")

    res = evaluate(model, proc, args.n, args.max_turns, args.max_new_tokens,
                   not args.no_thinking, not args.quiet)
    res["tag"] = tag
    res["model"] = args.model
    res["adapter"] = args.adapter

    print("\n" + json.dumps(res, indent=2))

    env.OUT.mkdir(parents=True, exist_ok=True)
    path = env.OUT / f"eval-{tag}.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
