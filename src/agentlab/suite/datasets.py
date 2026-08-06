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

Views that do not fit `acceptance.max_view_tokens` (5,120, set by the exhaustive
token census in results/agentic/token_census.json) are REJECTED, never truncated:
a silently truncated completion is exactly the termination-free supervision this
pipeline exists to avoid.

PROVENANCE. Every view also carries the chain back to what produced it: the
environment contract it was built under, the content digest of the accepted
trajectory it came from, and that trajectory's producer snapshot (card, driver,
engine fingerprint, effective thinking mode, runtime-manifest digest, session id)
copied verbatim. The chain continues into `agentlab.sft`'s training manifest and
from there into the checkpoint lock, so a locked adapter can name the card behind
every row it was trained on. A trajectory that cannot say what produced it is
dropped and counted, never built from.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

from agentlab.suite import contract, rng
from agentlab.suite.configio import load_config
from agentlab.suite.runtime import tool_schemas_for_family


def _stamp() -> str:
    return contract.environment_contract_sha256()


# --------------------------------------------------------------------------
# the view-side link of the provenance chain
# --------------------------------------------------------------------------
#
# The four TRL columns are frozen (prompt, completion, tools,
# chat_template_kwargs) so the trainer sees no stray columns. Everything that
# says WHERE a view came from therefore lives in the parallel metadata, one meta
# row per training row, in the same order. Before this fix the metadata dropped
# the trajectory's provenance entirely, which broke the chain exactly where it
# mattered most: the corpus a checkpoint is trained on could not name the card,
# engine or session that produced it, so the locked checkpoint could not either.

# Must be present AND non-empty on every view.
VIEW_CHAIN_FIELDS = ("row_id", "task_id", "view", "source_row_sha256",
                     "source_provenance", contract.STAMP_FIELD)

# Must be PRESENT: their value may legitimately be null (a CPU producer has no
# session), but the key going missing is how a chain silently stops being one.
VIEW_CHAIN_OPTIONAL_FIELDS = ("gpu_execution", "runtime_manifest_sha256",
                              "session_id")


def view_row_id(task_id: str, view: str, index: int, copy_index: int) -> str:
    """A stable identity for one training row: task, view kind, turn, copy."""
    from agentlab.suite.schema import digest_text

    return digest_text(f"view|{task_id}|{view}|{index}|{copy_index}")[:24]


def require_view_chain(meta_row: dict, what: str = "an SFT view") -> dict:
    """Refuse a view whose chain back to a producer is missing or partial.

    A view that cannot name the environment contract it was built under, the
    accepted row it came from, or the producer that rolled that row out, is not
    trainable evidence: the trainer manifest and the checkpoint lock inherit
    exactly these fields, so a gap here is a gap in the locked checkpoint.
    """
    from agentlab.multidistill import provenance_gaps

    missing = [k for k in VIEW_CHAIN_FIELDS
               if meta_row.get(k) is None or meta_row.get(k) == ""]
    missing += [k for k in VIEW_CHAIN_OPTIONAL_FIELDS if k not in meta_row]
    if missing:
        raise SystemExit(
            f"REFUSED: {what} is missing {', '.join(missing)}. Every training row "
            f"must name the environment contract it was built under, the accepted "
            f"trajectory it came from, and the producer session that rolled that "
            f"trajectory out. A view without that chain trains a checkpoint no "
            f"lock can attribute.")
    if meta_row[contract.STAMP_FIELD] != _stamp():
        contract.require_current(meta_row, what)
    gaps = provenance_gaps(meta_row["source_provenance"])
    if gaps:
        raise SystemExit(
            f"REFUSED: {what} carries source provenance that is not evidence "
            f"({', '.join(gaps)}). The view inherits the producer snapshot of its "
            f"trajectory verbatim; it never synthesizes one.")
    return meta_row


def require_views_chain(rows: list, meta: list, report: dict | None = None,
                        *, require_gpu_source: bool = True) -> dict:
    """The whole-corpus gate a trainer runs BEFORE it touches an optimizer.

    Checks, and terminates on the first failure:
      the metadata is one-to-one with the training rows and in the same order;
      every meta row carries the complete chain; every row_id is distinct; the
      corpus descends from exactly ONE producer identity; and -- unless a caller
      explicitly asks otherwise -- that producer actually owned a card.

    The GPU requirement is stated separately from the chain requirement on
    purpose: a CPU-scripted corpus is honest evidence about the harness and a
    perfectly good test fixture, but it is not a corpus a reportable checkpoint
    may be trained on, and "the fixture trained fine" is exactly how an
    unattributable adapter would get locked.
    """
    from agentlab.multidistill import require_one_producer

    if len(meta) != len(rows):
        raise SystemExit(
            f"REFUSED: {len(rows)} training rows and {len(meta)} metadata rows. "
            f"The metadata is the only place a view's provenance lives, so a "
            f"one-to-one correspondence is the chain; unequal counts mean some "
            f"row is being trained on with nothing recorded about its origin.")
    for i, meta_row in enumerate(meta):
        require_view_chain(meta_row, f"SFT view {i}")
    ids = [m["row_id"] for m in meta]
    if len(set(ids)) != len(ids):
        dupes = sorted({r for r in ids if ids.count(r) > 1})
        raise SystemExit(
            f"REFUSED: {len(ids) - len(set(ids))} SFT view metadata rows share a "
            f"row_id ({dupes[:4]}). Two rows with one identity cannot both be "
            f"traced, so the mapping from checkpoint to trajectory is ambiguous.")
    ident = require_one_producer(
        [{"provenance": m["source_provenance"]} for m in meta],
        "the SFT view corpus")
    if require_gpu_source and not ident.get("gpu_execution"):
        raise SystemExit(
            "REFUSED: this view corpus was produced without a GPU "
            f"({ident.get('producer')!r}), so no card attested the trajectories "
            f"it trains on. A scripted or CPU-built corpus is a fixture, not "
            f"trainable evidence: roll the trajectories out through the attested "
            f"engine (`python -m agentlab.multidistill run`) first.")
    if report is not None:
        contract.require_current(report, "the SFT view report")
        recorded = report.get("source_provenance")
        if recorded is not None:
            from agentlab.suite.schema import canon

            if canon(recorded) != canon(ident):
                raise SystemExit(
                    f"REFUSED: the view report names producer {canon(recorded)} "
                    f"and the metadata names {canon(ident)}. The report is the "
                    f"summary of these rows or it is a summary of something else.")
        if report.get("rows") is not None and int(report["rows"]) != len(rows):
            raise SystemExit(
                f"REFUSED: the view report counts {report['rows']} rows and the "
                f"corpus holds {len(rows)}; one of the two describes another "
                f"build.")
    return {"rows": len(rows), "source_provenance": ident,
            "row_ids": len(set(ids))}

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
    (prompt, completion, tools, chat_template_kwargs); the provenance chain goes
    to the parallel metadata (one row per training row, same order) and its
    summary to the report, so the trainer sees no stray columns.
    """
    from agentlab.multidistill import (provenance_gaps, require_one_producer,
                                       row_digest)

    cfg = cfg or load_config()
    max_tokens = cfg["acceptance"]["max_view_tokens"]
    schema_cache: dict[str, list] = {}
    rows, meta = [], []
    counts = {"terminal": 0, "pivot": 0, "recovery": 0}
    rejected = {"over_token_budget": 0, "no_terminal": 0, "trajectory_over_budget": 0,
                "stale_environment_contract": 0, "missing_source_provenance": 0}

    records, stale = contract.invalidate(records, "accepted trajectory")
    rejected["stale_environment_contract"] = len(stale)
    # A trajectory that cannot name its producer cannot be trained on: the view
    # inherits the snapshot, so there is nothing honest to inherit. Dropped and
    # COUNTED, never quietly built from.
    attributable, unattributable = [], 0
    for rec in records:
        if provenance_gaps(rec.get("provenance")):
            unattributable += 1
            continue
        attributable.append(rec)
    rejected["missing_source_provenance"] = unattributable
    records = attributable
    source_provenance = require_one_producer(records, "the SFT view corpus")
    schema_sha: dict[str, str] = {}
    for rec in records:
        family = rec["family"]
        source_sha = row_digest(rec)
        rec_prov = dict(rec["provenance"])
        tools = schema_cache.setdefault(family, tool_schemas_for_family(family))
        if family not in schema_sha:
            from agentlab.suite.runtime import tool_schema_bytes
            from agentlab.suite.schema import digest_text

            schema_sha[family] = digest_text(tool_schema_bytes(family))
        plan = select_views(rec, cfg)
        if not plan:
            rejected["no_terminal"] += 1
            continue

        # The terminal view has the longest prompt; if IT does not fit, the
        # whole trajectory is rejected (universal filter: fit the registered
        # view budget or reject -- never truncate).
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
            for copy_index in range(item["copies"]):
                candidate_rows.append((item["view"], row, n_tok, i, copy_index))

        for view, row, n_tok, msg_index, copy_index in candidate_rows:
            rows.append(row)
            counts[view] += 1
            meta.append({"row_id": view_row_id(rec["task_id"], view, msg_index,
                                               copy_index),
                         "task_id": rec["task_id"], "family": family,
                         "horizon": rec["horizon"],
                         "fault_types": list(rec.get("fault_types") or []),
                         "view": view, "msg_index": msg_index,
                         "copy_index": copy_index, "tokens": n_tok,
                         # Which model-visible environment these prompt bytes
                         # came from (D2): an SFT view built from a tokenless
                         # transcript trains on a different environment from the
                         # one the study evaluates in.
                         contract.STAMP_FIELD: _stamp(),
                         "tool_schema_sha256": schema_sha[family],
                         # ... and WHAT PRODUCED them (S19): the accepted row's
                         # content digest plus that row's producer snapshot,
                         # copied verbatim. The trainer manifest and the
                         # checkpoint lock read these three fields, so the
                         # locked checkpoint can name the card, engine and
                         # session behind every row it was trained on.
                         "source_row_sha256": source_sha,
                         "source_provenance": rec_prov,
                         "runtime_manifest_sha256":
                             rec_prov.get("runtime_manifest_sha256"),
                         "session_id": rec_prov.get("session_id"),
                         "gpu_execution": bool(rec_prov.get("gpu_execution"))})

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
        contract.STAMP_FIELD: _stamp(),
        # The chain, summarized: which producer, which sessions, which
        # attestations, and how many trajectories the rows descend from.
        "source_provenance": source_provenance,
        "source_trajectories": len(records),
        "source_row_digests": len({m["source_row_sha256"] for m in meta}),
        "source_sessions": sorted({m["session_id"] for m in meta if m["session_id"]}),
        "source_runtime_manifests": sorted(
            {m["runtime_manifest_sha256"] for m in meta
             if m["runtime_manifest_sha256"]}),
        "gpu_execution": bool(source_provenance.get("gpu_execution")),
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
    print(f"[views] provenance: {report['source_trajectories']} trajectories, "
          f"{len(report['source_sessions'])} producer session(s), "
          f"gpu_execution={report['gpu_execution']}")
    # The chain is checked HERE too, not only in the trainer: a corpus written to
    # disk without it would look finished, and the next stage's refusal would be
    # read as a trainer bug rather than as a missing rollout attestation.
    require_views_chain(rows, meta, report, require_gpu_source=False)
    print(f"[views] terminal weight {report['terminal_weight']:.3f} "
          f"(min {report['terminal_weight_min']})")
    if not report["terminal_weight_ok"]:
        raise SystemExit("terminal weight below the preregistered minimum; "
                         "do not train on this corpus")


if __name__ == "__main__":
    main()
