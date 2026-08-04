"""Stage 1b -- build SFT data the model can actually learn termination from.

Why this exists: training on `xlam-function-calling-60k` degraded the model 16x
(0.800 -> 0.050). Every xlam target is a bare tool call, so nothing in it ever
shows a model receiving a tool result and *concluding*. The policy learned to
emit calls and never stop -- 50 calls per episode, 19 of 20 episodes with no
final answer at all.

This is rejection sampling / expert iteration (STaR, RFT): let the base model
attempt the real task with the real tools, keep only the trajectories that end in
a *correct committed answer*, and train on those. Every accepted example
therefore demonstrates the exact behaviour xlam lacked -- call a tool, read the
result, and terminate.

The honest limit: this distils the base model into itself and cannot exceed base
on reasoning. It is aimed at the failure base actually has -- 9 of its 10 errors
were "ran out of tokens before \\boxed{}", not wrong arithmetic -- and the
acceptance filter selects precisely against that.

Generation runs through vLLM in batch. The repo's own agent loop is ~25 s per
episode, which would be ~10 h for a useful corpus; batched vLLM does the same
work in tens of minutes.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from . import env
from .chat import boxed_answer, parse_tool_calls

# strips the tool-call block so the assistant's prose can be kept separately
_TOOL_CALL_BLOCK = __import__('re').compile(r'<tool_call>.*?(?:</tool_call>|$)', __import__('re').DOTALL)
from .data import build_grpo
from .tools import call_tool, tool_schemas


def _save(convos, path: str) -> None:
    """Snapshot rollout state; cheap relative to generation, and it is insurance."""
    import pathlib as _pl
    p = _pl.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for c in convos:
            fh.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)  # atomic, so a crash mid-write cannot corrupt the snapshot


def _numeric(s):
    try:
        return float(str(s).strip().replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def generate(model_id: str, n_problems: int, k: int, max_turns: int,
             max_new_tokens: int, temperature: float, top_p: float,
             gpu_frac: float, max_model_len: int, seed: int = 0,
             offset: int = 0, checkpoint: str | None = None) -> list[dict]:
    """Roll out k attempts per problem with vLLM, executing tools for real."""
    from vllm import LLM, SamplingParams

    proc = env.load_processor(model_id)
    tok = env.get_tokenizer(proc)
    schemas = tool_schemas()

    ds = build_grpo(n=n_problems, split="train", seed=seed, offset=offset)
    # k independent attempts per problem; diversity is what makes rejection
    # sampling work, so these are sampled, not greedy.
    convos = []
    for i, row in enumerate(ds):
        for j in range(k):
            convos.append({
                "pid": i, "sample": j,
                "gt": row["ground_truth"],
                "question": row["prompt"][-1]["content"],
                "messages": list(row["prompt"]),
                "done": False, "final": "", "n_calls": 0, "tool_error": False,
                "tools_used": [], "truncated": False, "exhausted": False,
            })

    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_frac,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=20,
                        max_tokens=max_new_tokens, seed=None)

    for turn in range(max_turns):
        active = [c for c in convos if not c["done"]]
        if not active:
            break
        prompts = [
            tok.apply_chat_template(c["messages"], tools=schemas, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
            for c in active
        ]
        print(f"[gen] turn {turn}: {len(active)} active rollouts", flush=True)
        try:
            outs = llm.generate(prompts, sp)
        except Exception as e:  # noqa: BLE001 - a 6000-rollout run must not vanish
            print(f"[gen] turn {turn} FAILED ({type(e).__name__}: {e}); "
                  f"keeping {sum(1 for c in convos if c['done'])} finished rollouts", flush=True)
            break

        for c, out in zip(active, outs):
            comp = out.outputs[0]
            text = comp.text
            calls = parse_tool_calls(text)
            if not calls:
                # finish_reason "length" means the cap cut it off mid-sentence.
                # Recording that as a committed answer would distil truncation --
                # the very habit that causes most base failures -- back in.
                c["done"] = True
                c["truncated"] = comp.finish_reason == "length"
                c["final"] = "" if c["truncated"] else text.strip()
                c["messages"].append({"role": "assistant", "content": text.strip()})
                continue
            # Keep the model's own prose next to the call. Dropping it would make
            # every intermediate turn byte-identical to the xlam shape this run
            # exists to replace, and the conditioning context the terminal row is
            # trained against would then not match what the model emits at
            # inference. `content` is the text with the tool-call block removed.
            prose = _TOOL_CALL_BLOCK.sub("", text).strip()
            msg = {"role": "assistant",
                   "tool_calls": [{"type": "function",
                                   "function": {"name": x["name"], "arguments": x["arguments"]}}
                                  for x in calls]}
            if prose:
                msg["content"] = prose
            c["messages"].append(msg)
            for x in calls:
                result = call_tool(x["name"], x["arguments"])
                c["n_calls"] += 1
                c["tools_used"].append(x["name"])
                if str(result).startswith("error:"):
                    c["tool_error"] = True
                c["messages"].append({"role": "tool", "name": x["name"], "content": result})

        # Persist after every turn: without this one bad rollout at turn 4 of 4
        # throws away the entire generation run.
        if checkpoint:
            _save(convos, checkpoint)

    # Anything still running hit max_turns without committing. Flagged distinctly
    # so the rejection histogram can tell "ran out of turns" from "said nothing".
    for c in convos:
        if not c["done"]:
            c["final"] = ""
            c["exhausted"] = True
    return convos


def accept(traj: dict, max_calls: int, max_final_tokens: int = 128,
           allowed_tools: tuple[str, ...] = ("calculator", "unit_convert"),
           allow_tool_error: bool = False) -> tuple[bool, str]:
    """Acceptance filter. Returns (accepted, reason_when_rejected).

    Each condition removes a behaviour we do NOT want to teach:
      correct     -- never train on a wrong answer
      committed   -- the whole point; must end in a parseable boxed answer
      used a tool -- otherwise we teach ignoring the tools
      bounded     -- reject the runaway pattern that broke the last adapter
      clean tools -- do not teach malformed calls
    """
    if traj.get("exhausted"):
        return False, "max_turns_exhausted"
    if traj.get("truncated"):
        return False, "truncated_final"
    if not traj["final"]:
        return False, "no_final"
    got, want = _numeric(boxed_answer(traj["final"]) or ""), _numeric(traj["gt"])
    if got is None:
        return False, "no_box"
    if want is None or abs(got - want) > 1e-4:
        return False, "wrong"
    if traj["n_calls"] == 0:
        return False, "no_tool"
    if traj["n_calls"] > max_calls:
        return False, "too_many_calls"
    # Long derivations are exactly what makes base run out of tokens before the
    # box, so select against them rather than distilling the habit back in.
    if len(traj["final"].split()) > max_final_tokens:
        return False, "final_too_long"
    # kb_lookup has nothing to offer a GSM8K problem; training on it teaches
    # irrelevant tool selection.
    if any(n not in allowed_tools for n in traj.get("tools_used", [])):
        return False, "irrelevant_tool"
    # A trajectory that hit a tool error and still reached the right answer is
    # an error-RECOVERY demonstration. Banning them outright leaves the corpus
    # with zero examples of the behaviour the eval measures as tool_error_rate,
    # so they are admitted under a quota instead (see main).
    if traj["tool_error"] and not allow_tool_error:
        return False, "tool_error"
    return True, ""


def to_sft_rows(traj: dict) -> list[dict]:
    """ONE row per trajectory: the terminal turn only.

    The obvious encoding -- a row per assistant turn -- is wrong for this
    objective even though it correctly avoids training on tool results. A typical
    four-turn trajectory has three tool-call targets and one termination target,
    so training all of them recreates the call-heavy target distribution that
    broke the last adapter. The base model already knows how to call the
    calculator; the missing conditional is what to do *after* a result arrives.

    So: everything up to and including the last tool result is the (masked)
    prompt, and the committed final answer is the only completion. Loss lands on
    the termination decision and nothing else.

    Balancing the accepted set across call counts is what stops this teaching
    "always stop after exactly one result" -- see `main`.
    """
    msgs = traj["messages"]
    # index of the final assistant turn (the committed answer)
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    return [{
        "prompt": msgs[:last],
        "completion": [msgs[last]],
        "tools": tool_schemas(),
        "chat_template_kwargs": {"enable_thinking": False},
    }]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env.MODEL)
    ap.add_argument("--n-problems", type=int, default=1500)
    ap.add_argument("--k", type=int, default=4, help="attempts per problem")
    ap.add_argument("--max-turns", type=int, default=4,
                    help="assistant decisions: up to 3 tool turns plus the terminal answer")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-calls", type=int, default=6, help="reject runaway trajectories")
    ap.add_argument("--per-problem", type=int, default=1,
                    help="keep at most this many accepted samples per problem")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", default="data/distill.jsonl")
    ap.add_argument("--offset", type=int, default=0,
                    help="start index into the shuffled train split; GRPO must use a DISJOINT slice")
    ap.add_argument("--checkpoint", default="data/distill-raw.jsonl")
    ap.add_argument("--tool-error-frac", type=float, default=0.2,
                    help="max share of accepted trajectories that recovered from a tool error")
    args = ap.parse_args()

    env.require_single_gpu()
    trajs = generate(args.model, args.n_problems, args.k, args.max_turns,
                     args.max_new_tokens, args.temperature, args.top_p,
                     args.gpu_frac, args.max_model_len,
                     offset=args.offset, checkpoint=args.checkpoint)

    kept, reasons, per_pid = [], {}, {}
    err_budget = int(args.tool_error_frac * args.n_problems * args.per_problem)
    err_used = 0
    for t in trajs:
        allow_err = t["tool_error"] and err_used < err_budget
        ok, why = accept(t, args.max_calls, allow_tool_error=allow_err)
        if ok and t["tool_error"]:
            err_used += 1
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        # Cap per problem so easy questions cannot dominate the corpus.
        if per_pid.get(t["pid"], 0) >= args.per_problem:
            reasons["over_quota"] = reasons.get("over_quota", 0) + 1
            continue
        per_pid[t["pid"]] = per_pid.get(t["pid"], 0) + 1
        kept.append(t)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for t in kept for r in to_sft_rows(t)]
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_att = len(trajs)
    print(f"\n[distill] attempts        {n_att}")
    print(f"[distill] accepted        {len(kept)}  ({len(kept)/max(n_att,1):.1%})")
    print(f"[distill] problems solved {len(per_pid)} / {args.n_problems}")
    print(f"[distill] SFT rows        {len(rows)}  -> {out}")
    print(f"[distill] mean calls/kept {sum(t['n_calls'] for t in kept)/max(len(kept),1):.2f}")
    print(f"[distill] recovery kept   {err_used} (budget {err_budget})")
    calls = [t["n_calls"] for t in kept]
    if calls:
        from collections import Counter
        print("[distill] call-count dist " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(calls).items())))
    print("[distill] rejections      " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))


if __name__ == "__main__":
    main()
