"""Completion-only SFT view construction from accepted trajectories.

Training all assistant turns of a successful eight-call transcript would
recreate the old action-heavy failure: eight call targets for every one
termination target. Each accepted trajectory is instead converted into a small
set of completion-only views (per configs/multifaceted.yaml `views`):

  terminal   the committed final answer, emitted TWICE;
  pivot      ONE dependency-bearing intermediate action (TWO for H8): a tool
             call whose arguments consume a value produced by an earlier tool
             result, position-stratified across the corpus by seed hash;
  recovery   the first assistant decision after the injected fault result,
             emitted TWICE; when recovery genuinely took two distinct
             decisions, both are emitted once (combined weight stays 2).

Masking is structural: every view is a {prompt, completion} pair where the
completion is exactly ONE assistant message and everything before it -- all
earlier assistant turns and ALL tool outputs, including the injected error --
is prompt context that TRL's completion-only loss never trains on. The action
that caused an unscheduled tool error is never a supervised target.

Views that do not fit `acceptance.max_view_tokens` (4096) are REJECTED, never
truncated: a silently truncated completion is exactly the termination-free
supervision this pipeline exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

from agentlab.suite import rng
from agentlab.suite.configio import load_config
from agentlab.suite.runtime import tool_schemas_for_family

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
# Opaque identifiers the suite generators mint, all uppercase and high-entropy:
#   lookup_chain / typed_relay start keys   K + 16 base32 chars
#   typed_relay derived keys                6 base32 chars + "-" + digits
#   fulfillment tokens                      ORD-/LIN-/QTE-/RSV-/FIN-/CMP-/SPEC-...
# The dependency test below asks whether such a token crossed from a tool result
# into a later call's arguments, so the pattern must match the keys this suite
# actually generates -- the previous four-groups-of-four pattern matched none of
# them, which silently made every lookup hop look dependency-free.
_KEY_RE = re.compile(r"[A-Z][A-Z2-7]{5,}|[A-Z2-7]{3,}-[A-Z0-9]{3,}")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


# --------------------------------------------------------------------------
# transcript inspection helpers (shared with multidistill's acceptance filters)
# --------------------------------------------------------------------------

def assistant_indices(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def floats_in(text: str) -> set[float]:
    out = set()
    for m in _NUM_RE.findall(text):
        try:
            out.add(float(m))
        except ValueError:
            pass
    return out


def keys_in(text: str) -> set[str]:
    return set(_KEY_RE.findall(text))


def _call_args_text(msg: dict) -> str:
    parts = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        parts.append(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
    return "\n".join(parts)


def consumes_predecessor(messages: list, i: int) -> bool:
    """True when assistant turn i's call arguments consume an earlier tool value.

    "Consume" means: a numeric value or an opaque record key that appeared in a
    PRIOR tool result appears in this turn's call arguments. This is the
    dependency-bearing test both the pivot selector and the compositional
    acceptance filter use.
    """
    msg = messages[i]
    if not msg.get("tool_calls"):
        return False
    prior = "\n".join(str(m.get("content", "")) for m in messages[:i]
                      if m.get("role") == "tool")
    if not prior:
        return False
    args_text = _call_args_text(msg)
    if keys_in(prior) & keys_in(args_text):
        return True
    return bool(floats_in(prior) & floats_in(args_text))


def followed_by_unscheduled_error(messages: list, i: int, fault_result_index) -> bool:
    """True when turn i's calls produced a tool error that was NOT the injection."""
    for j in range(i + 1, len(messages)):
        m = messages[j]
        if m.get("role") == "assistant":
            break
        if m.get("role") != "tool":
            continue
        if j == fault_result_index:
            continue
        content = str(m.get("content", ""))
        if '"ok":false' in content.replace(" ", "").lower():
            return True
    return False


# --------------------------------------------------------------------------
# view selection
# --------------------------------------------------------------------------

def _pivot_seed(task_id: str) -> int:
    """A 63-bit deterministic seed from a committed task id."""
    from agentlab.suite.schema import digest_text

    return int(digest_text(f"pivot-seed|{task_id}")[:16], 16) >> 1


def select_views(record: dict, cfg: dict | None = None) -> list[dict]:
    """The (msg_index, view, copies) plan for one accepted trajectory."""
    cfg = cfg or load_config()
    v = cfg["views"]
    messages = record["messages"]
    a_idx = assistant_indices(messages)
    if not a_idx:
        return []
    terminal = a_idx[-1]
    if not _BOXED_RE.search(str(messages[terminal].get("content", ""))):
        # No committed final answer: nothing here is worth supervising.
        return []

    plan = [{"index": terminal, "view": "terminal", "copies": v["terminal_copies"]}]
    fault = record.get("fault") or {}
    fault_result = fault.get("result_msg_index")

    recovery_indices: list[int] = []
    if fault.get("fired") and fault_result is not None:
        post = [i for i in a_idx if i > fault_result and i != terminal]
        if post:
            first_post = post[0]
            certified = fault.get("recovery_msg_index")
            if certified is not None and certified != first_post and certified in post:
                # Two genuinely distinct decisions: each once, combined weight 2.
                recovery_indices = [first_post, certified]
                copies = v["recovery_two_decision_copies"]
            else:
                recovery_indices = [first_post]
                copies = v["recovery_copies"]
            for i in recovery_indices:
                plan.append({"index": i, "view": "recovery", "copies": copies})

    # Pivots: dependency-bearing intermediate actions, excluding the terminal,
    # the recovery decisions (already supervised), and any turn that caused an
    # unscheduled tool error (never a supervised target).
    eligible = [i for i in a_idx
                if i != terminal
                and i not in recovery_indices
                and consumes_predecessor(messages, i)
                and not followed_by_unscheduled_error(messages, i, fault_result)]
    n_pivots = v["pivot_per_trajectory_h8"] if record["horizon"] >= 8 else v["pivot_per_trajectory"]
    if eligible:
        # Position stratification: a seed-keyed draw, uniform over the eligible
        # positions, so early and late transitions are covered corpus-wide. The
        # key is the committed task id (the suite's task identity), so the same
        # accepted trajectory always yields the same pivot positions.
        draws = rng.stream_u64(_pivot_seed(record["task_id"]),
                               f"pivot|{record['task_id']}", 2)
        picked = [eligible[int(draws[0]) % len(eligible)]]
        if n_pivots > 1 and len(eligible) > 1:
            second = eligible[int(draws[1]) % len(eligible)]
            if second == picked[0]:
                second = eligible[(eligible.index(second) + 1) % len(eligible)]
            picked.append(second)
        for i in sorted(set(picked)):
            plan.append({"index": i, "view": "pivot", "copies": 1})
    return plan


# --------------------------------------------------------------------------
# row construction
# --------------------------------------------------------------------------

def default_token_counter():
    """Chat-template token counter over (prompt, completion, tools)."""
    from agentlab import env as labenv

    proc = labenv.load_processor()
    tok = labenv.get_tokenizer(proc)

    def count(prompt_msgs: list, completion_msgs: list, tools: list) -> int:
        text = tok.apply_chat_template(
            list(prompt_msgs) + list(completion_msgs), tools=tools, tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        return len(tok(text)["input_ids"])

    return count


def build_views(records: list, token_counter, cfg: dict | None = None):
    """Accepted trajectories -> (SFT rows, build report).

    Rows carry exactly the four keys TRL's completion-only SFT path consumes
    (prompt, completion, tools, chat_template_kwargs); provenance goes to the
    parallel report so the trainer sees no stray columns.
    """
    cfg = cfg or load_config()
    max_tokens = cfg["acceptance"]["max_view_tokens"]
    schema_cache: dict[str, list] = {}
    rows, meta = [], []
    counts = {"terminal": 0, "pivot": 0, "recovery": 0}
    rejected = {"over_token_budget": 0, "no_terminal": 0, "trajectory_over_budget": 0}

    for rec in records:
        family = rec["family"]
        tools = schema_cache.setdefault(family, tool_schemas_for_family(family))
        plan = select_views(rec, cfg)
        if not plan:
            rejected["no_terminal"] += 1
            continue

        # The terminal view has the longest prompt; if IT does not fit, the
        # whole trajectory is rejected (universal filter: fit 4096 or reject).
        candidate_rows, over = [], 0
        for item in plan:
            i = item["index"]
            prompt = rec["messages"][:i]
            completion = [rec["messages"][i]]
            n_tok = token_counter(prompt, completion, tools)
            if n_tok > max_tokens:
                over += 1
                rejected["over_token_budget"] += item["copies"]
                if item["view"] == "terminal":
                    candidate_rows = []
                    rejected["trajectory_over_budget"] += 1
                    break
                continue
            row = {"prompt": prompt, "completion": completion, "tools": tools,
                   "chat_template_kwargs": {"enable_thinking": False}}
            for _ in range(item["copies"]):
                candidate_rows.append((item["view"], row, n_tok))

        for view, row, n_tok in candidate_rows:
            rows.append(row)
            counts[view] += 1
            meta.append({"task_id": rec["task_id"], "family": family,
                         "horizon": rec["horizon"],
                         "fault_types": list(rec.get("fault_types") or []),
                         "view": view, "tokens": n_tok})

    total = len(rows)
    terminal_weight = counts["terminal"] / total if total else 0.0
    report = {
        "rows": total,
        "view_counts": counts,
        "rejected": rejected,
        "terminal_weight": round(terminal_weight, 4),
        "terminal_weight_min": cfg["views"]["terminal_weight_min"],
        "terminal_weight_ok": terminal_weight >= cfg["views"]["terminal_weight_min"],
        "expected_rows": cfg["views"]["expected_rows"],
    }
    return rows, meta, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", default="data/multiface/accepted.jsonl",
                    help="accepted trajectories from `python -m agentlab.multidistill finalize`")
    ap.add_argument("--out", default="data/multiface/sft_views.jsonl")
    ap.add_argument("--meta-out", default="data/multiface/sft_views.meta.jsonl")
    ap.add_argument("--report-out", default="data/multiface/sft_views.report.json")
    args = ap.parse_args()

    src = pathlib.Path(args.accepted)
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    rows, meta, report = build_views(records, default_token_counter())

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with pathlib.Path(args.meta_out).open("w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    pathlib.Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"[views] {report['rows']} rows -> {out}")
    print(f"[views] counts {report['view_counts']}  rejected {report['rejected']}")
    print(f"[views] terminal weight {report['terminal_weight']:.3f} "
          f"(min {report['terminal_weight_min']})")
    if not report["terminal_weight_ok"]:
        raise SystemExit("terminal weight below the preregistered minimum; "
                         "do not train on this corpus")


if __name__ == "__main__":
    main()
