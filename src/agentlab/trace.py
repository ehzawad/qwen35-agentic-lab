"""Structured traces of what actually happened inside a rollout.

`exit 0` is not evidence. A rollout can exit 0 while the model never called a
tool, or called one that errored, or answered correctly by guessing with the
tool ignored. This records the whole episode -- prompt, thinking, every call,
every result, the final answer, and each reward function's contribution -- as
one JSON object per line.

Off unless AGENTLAB_TRACE is set, so training pays nothing for it:

    AGENTLAB_TRACE=out/trace.jsonl make eval-base
    make trace-view FILE=out/trace.jsonl
"""

from __future__ import annotations

import json
import os
import pathlib
import threading

_LOCK = threading.Lock()
_PATH_ENV = "AGENTLAB_TRACE"


def enabled() -> bool:
    return bool(os.environ.get(_PATH_ENV))


def path() -> pathlib.Path | None:
    p = os.environ.get(_PATH_ENV)
    return pathlib.Path(p) if p else None


def emit(kind: str, **fields) -> None:
    """Append one trace record. Never raises -- tracing must not break a run."""
    p = path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"kind": kind, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        # GRPO reward functions are called from multiple workers; without the
        # lock, records interleave mid-line and the file will not parse.
        with _LOCK, p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


class Episode:
    """Accumulates one agent episode, then writes it as a single record.

    One line per episode rather than per event: the unit a human wants to read
    is "what happened on this problem", and per-event lines would have to be
    re-joined before they mean anything.
    """

    def __init__(self, task: str, index: int | None = None, ground_truth: str | None = None):
        self.rec = {
            "task": task,
            "index": index,
            "ground_truth": ground_truth,
            "turns": [],
            "final": None,
            "rewards": {},
            "ok": None,
        }

    def turn(self, thinking: str = "", calls: list[dict] | None = None,
             results: list[str] | None = None, text: str = "") -> None:
        self.rec["turns"].append(
            {
                "thinking": thinking,
                "thinking_chars": len(thinking),
                "calls": calls or [],
                "results": results or [],
                "text": text,
            }
        )

    def finish(self, final: str, ok: bool | None = None, **rewards) -> None:
        self.rec["final"] = final
        self.rec["ok"] = ok
        self.rec["rewards"] = rewards
        self.rec["n_turns"] = len(self.rec["turns"])
        self.rec["n_calls"] = sum(len(t["calls"]) for t in self.rec["turns"])
        self.rec["n_tool_errors"] = sum(
            1 for t in self.rec["turns"] for r in t["results"] if str(r).startswith("error:")
        )
        emit("episode", **self.rec)


def render(file: str | pathlib.Path, limit: int = 5) -> str:
    """Human-readable view of a trace file."""
    p = pathlib.Path(file)
    if not p.exists():
        return f"no trace at {p}"

    episodes, out = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "episode":
            episodes.append(rec)

    if not episodes:
        return f"{p}: no episodes recorded"

    for ep in episodes[:limit]:
        mark = {True: "PASS", False: "FAIL", None: "----"}[ep.get("ok")]
        out.append("=" * 78)
        out.append(f"[{mark}] episode {ep.get('index')}   ground_truth={ep.get('ground_truth')}")
        out.append(f"task: {str(ep.get('task', ''))[:300]}")
        for i, t in enumerate(ep.get("turns", [])):
            out.append(f"  -- turn {i}")
            if t.get("thinking_chars"):
                out.append(f"     thinking: {t['thinking_chars']} chars")
                out.append(f"       {str(t.get('thinking',''))[:200].replace(chr(10),' ')}")
            for c, r in zip(t.get("calls", []), t.get("results", []) or [None] * len(t.get("calls", []))):
                out.append(f"     call  {c.get('name')}({json.dumps(c.get('arguments', {}))})")
                out.append(f"     ->    {r}")
            if t.get("text"):
                out.append(f"     text: {str(t['text'])[:200]}")
        out.append(f"  final: {str(ep.get('final',''))[:300]}")
        if ep.get("rewards"):
            parts = ", ".join(f"{k}={v}" for k, v in ep["rewards"].items())
            out.append(f"  rewards: {parts}")
        out.append(f"  turns={ep.get('n_turns')} calls={ep.get('n_calls')} tool_errors={ep.get('n_tool_errors')}")

    # The aggregate is the part that answers "is the pipeline working".
    n = len(episodes)
    scored = [e for e in episodes if e.get("ok") is not None]
    used = sum(1 for e in episodes if e.get("n_calls"))
    errs = sum(1 for e in episodes if e.get("n_tool_errors"))
    out.append("=" * 78)
    out.append(f"episodes        {n}" + (f"  (showing first {limit})" if n > limit else ""))
    if scored:
        out.append(f"accuracy        {sum(1 for e in scored if e['ok'])/len(scored):.3f}  (n={len(scored)})")
    out.append(f"tool_use_rate   {used/n:.3f}")
    out.append(f"tool_error_rate {errs/n:.3f}")
    out.append(f"mean_turns      {sum(e.get('n_turns',0) for e in episodes)/n:.2f}")
    out.append(f"mean_calls      {sum(e.get('n_calls',0) for e in episodes)/n:.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "out/trace.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(render(target, limit))
