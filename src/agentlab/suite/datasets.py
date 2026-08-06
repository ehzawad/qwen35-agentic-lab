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

ONE ANSWER GRAMMAR. "Did this trajectory commit a final answer" is asked here
with `schema.extract_committed_answer`, the same reader the strict verifier and
the certification layer use. A local `\\boxed{}`-only regex used to ask it
instead, so a certified-successful trajectory that obeyed the system prompt's
`ANSWER: <value>` form produced ZERO views and was tallied as a trajectory with
no terminal at all (the bucket is now `no_committed_answer`): about 36% of the
now-correct dev episodes commit in exactly that plain form, and the corpus
silently dropped every one of them.

THE ROW RANGE HAS TWO SIDES. `views.expected_rows` is [5,000, 6,000] and it is
enforced, so an OVER-full corpus refuses exactly like a short one -- and the
registered acceptance minima can reach it (>= 1,350 accepted trajectories at
about 4-5 rows each). `plan_view_cap` is the registered answer: a deterministic,
content-blind, seed-keyed per-stratum cap that keeps the first k trajectories of
each stratum in `sha256("view-cap-v1|<stratum>|<task_id>")` order. It never reads
a score, a reward, a length or any other outcome, and it is inert below the
ceiling.

COMPLETION IS A RECEIPT, NOT A PATH. The CLI validates the whole corpus BEFORE
it writes anything -- the chain, the registered row range (views.expected_rows)
naming the short stratum, and the terminal-weight floor -- then writes the three
files atomically with the row file LAST, because both resumers downstream (the
chain script's marker test and `agentlab.sft.load_views_metadata`) trust that
path's existence. The report doubles as the corpus receipt: it carries the row
count, the row-id digest, the per-stratum census and the gate verdicts, and
`require_views_chain` re-checks the count and the digest against the rows in
hand, so a truncated or half-written corpus can never be read as a finished one.
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


# A corpus the CLI wrote for the study, and therefore the one the registered row
# range and terminal-weight floor apply to. An in-process build (the dev
# preflight's deliberately tiny one-optimizer-step canary, and the tests) is
# honest evidence about the builder and is NOT a production corpus, so it carries
# the other label and is never measured against a corpus-level size gate.
PRODUCTION_CORPUS = "production"
IN_PROCESS_CORPUS = "in_process"


def view_row_id(task_id: str, view: str, index: int, copy_index: int) -> str:
    """A stable identity for one training row: task, view kind, turn, copy."""
    from agentlab.suite.schema import digest_text

    return digest_text(f"view|{task_id}|{view}|{index}|{copy_index}")[:24]


def row_ids_digest(meta: list) -> str:
    """The identity of a corpus: its row ids, in order, as one digest.

    The count alone cannot tell a truncated corpus from a differently-built one
    of the same size, and the receipt has to be able to.
    """
    from agentlab.suite.schema import digest_text

    return digest_text("|".join(str(m.get("row_id")) for m in meta))


# --------------------------------------------------------------------------
# THE VIEW CAP: the over-full half of the registered row range
# --------------------------------------------------------------------------
#
# `views.expected_rows` is [5000, 6000] and it is enforced on BOTH sides, so an
# OVER-full corpus is a hard stop exactly like an under-full one. That side is
# reachable from the registered minima alone: `totals.min_accepted` admits >=
# 1,350 accepted trajectories and the view grammar yields about 4-5 rows each
# (terminal x2 + one or two pivots + recovery x2 when a fault fired), so
# acceptance that overshoots its floors -- which nothing forbids, and which good
# model behaviour makes likely -- lands above 6,000 and the corpus refuses.
#
# The cap is written BEFORE any corpus exists (zero accepted trajectories, zero
# rows, zero study GPU-hours), so it is OUTCOME-BLIND by construction: there is
# no result it could be tuned against.
#
# It is DETERMINISTIC, CONTENT-BLIND and SEED-KEYED. Trajectories are ordered
# within their stratum by `sha256("view-cap-v1|<stratum>|<task_id>")` and the
# first k are kept. The key reads the committed task identity and nothing else:
# not the score, not the reward, not the length, not the fault class, not the
# transcript, not the verdict. Keeping "the best" trajectories would make the
# training corpus a function of the outcomes it is supposed to be blind to, and
# keeping "the shortest" would silently re-weight the horizon mixture.

VIEW_CAP_KEY_VERSION = "view-cap-v1"


def view_cap_key(stratum: str, task_id: str) -> str:
    """The seed-keyed sort key for one trajectory inside its stratum."""
    from agentlab.suite.schema import digest_text

    return digest_text(f"{VIEW_CAP_KEY_VERSION}|{stratum}|{task_id}")


def plan_view_cap(trajectories: list, cfg: dict | None = None) -> dict:
    """Which built trajectories survive the registered row ceiling.

    `trajectories` is `[{"stratum", "task_id", "rows"}]` in build order, where
    `rows` is how many training rows that trajectory's view plan produced. The
    plan returned says nothing about which rows: whole trajectories are kept or
    dropped, because splitting one (keeping its terminal view and dropping its
    recovery view, say) would change the view mixture the terminal-weight floor
    is defined over.

    THE RULE, in order:

    1. INERT below the ceiling. If the full build already fits `expected_rows`,
       nothing is dropped -- including a build that is too SMALL, which is the
       existing shortfall refusal and is never padded or rescued from here.
    2. Every non-empty stratum keeps at least its first trajectory. A cell that
       vanished would break stratum balance far worse than a few rows of
       imbalance, and the fragile cells are exactly the measured-only H14/H20
       ones a proportional share can round to nothing.
    3. The remaining budget (ceiling minus those reserved rows) is split over the
       strata in PROPORTION to their remaining rows, by largest remainder with
       ties broken by ascending stratum name. Every stratum's share of the capped
       corpus therefore equals its share of the full corpus to within one row --
       that is what "stratum balance is preserved" means here.
    4. Each stratum keeps the longest PREFIX of its cap-key order whose rows fit
       its share. The prefix stops at the first trajectory that does not fit; the
       scan does NOT continue looking for a smaller one, because that would rank
       trajectories by length.
    5. The result is measured against the UNCHANGED registered range. The cap
       never widens the range, never lowers the floor and never rescues a corpus
       that cannot reach 5,000 rows.
    """
    cfg = cfg or load_config()
    lo, hi = (int(x) for x in cfg["views"]["expected_rows"])
    entries = [{"index": i, "stratum": str(t["stratum"]),
                "task_id": str(t["task_id"]), "rows": int(t["rows"])}
               for i, t in enumerate(trajectories)]
    full_rows = sum(e["rows"] for e in entries)

    by: dict[str, list] = {}
    for e in entries:
        by.setdefault(e["stratum"], []).append(e)
    for name, items in by.items():
        items.sort(key=lambda e: (view_cap_key(name, e["task_id"]), e["task_id"],
                                  e["index"]))
    names = sorted(by)

    def summary(kept: list, capped: list) -> dict:
        from agentlab.suite.schema import digest_text

        kept_by = {}
        for e in kept:
            cell = kept_by.setdefault(e["stratum"], {"rows": 0, "trajectories": 0})
            cell["rows"] += e["rows"]
            cell["trajectories"] += 1
        per = {}
        for name in names:
            cell = kept_by.get(name, {"rows": 0, "trajectories": 0})
            per[name] = {
                "full_rows": sum(e["rows"] for e in by[name]),
                "full_trajectories": len(by[name]),
                "rows": cell["rows"], "trajectories": cell["trajectories"],
                "trajectories_capped": len(by[name]) - cell["trajectories"]}
        return {
            "key_version": VIEW_CAP_KEY_VERSION,
            "expected_rows": [lo, hi],
            "outcome_blind": True,
            "ranked_by": "sha256(view-cap-v1|stratum|task_id) -- never score, "
                         "reward, length or any outcome",
            "full_rows": full_rows, "full_trajectories": len(entries),
            "target_rows": hi,
            "applied": full_rows > hi,
            "rows": sum(e["rows"] for e in kept),
            "trajectories": len(kept),
            "trajectories_capped": len(capped),
            "kept_indexes": sorted(e["index"] for e in kept),
            "kept_task_ids_sha256": digest_text(
                "|".join(f"{e['stratum']}|{e['task_id']}"
                         for e in sorted(kept, key=lambda e: (e["stratum"],
                                                              view_cap_key(e["stratum"],
                                                                           e["task_id"]))))),
            "per_stratum": per}

    if full_rows <= hi:
        return summary(entries, [])

    reserved = {name: by[name][0] for name in names}
    reserved_rows = sum(e["rows"] for e in reserved.values())
    if reserved_rows > hi:
        raise SystemExit(
            f"REFUSED: one trajectory per stratum is already {reserved_rows} rows, "
            f"above the registered ceiling {hi}. This is not a cap decision: the "
            f"view plan itself is too large for the registered range, and the range "
            f"is preregistered. Do not widen it.")
    rest_rows = {name: sum(e["rows"] for e in by[name][1:]) for name in names}
    budget = hi - reserved_rows
    total_rest = sum(rest_rows.values())
    share = {name: 0 for name in names}
    if total_rest > 0 and budget > 0:
        exact = {name: budget * rest_rows[name] / total_rest for name in names}
        share = {name: int(exact[name]) for name in names}
        leftover = budget - sum(share.values())
        for name in sorted(names, key=lambda n: (-(exact[n] - share[n]), n))[:leftover]:
            share[name] += 1

    kept, capped = [], []
    for name in names:
        kept.append(reserved[name])
        used, stop = 0, False
        for e in by[name][1:]:
            if stop or used + e["rows"] > share[name]:
                stop = True
                capped.append(e)
                continue
            used += e["rows"]
            kept.append(e)
    plan = summary(kept, capped)
    if not lo <= plan["rows"] <= hi:
        raise SystemExit(
            f"REFUSED: the seed-keyed view cap landed on {plan['rows']} rows, "
            f"outside the registered range {lo}-{hi} (full build {full_rows} rows "
            f"in {len(names)} strata). Whole trajectories are indivisible, so this "
            f"means the per-trajectory row counts cannot tile the range -- a "
            f"generator or acceptance problem to fix upstream. The registered range "
            f"does not move.")
    return plan


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

    The report is the corpus RECEIPT, so it is checked against the rows in hand,
    not merely read: the recorded row count, the number of distinct row ids and
    the row-id digest must all describe exactly these rows. A corpus whose row
    file was truncated (or half-written, or built by another run) fails here
    instead of training.

    And when the receipt says it describes the PRODUCTION corpus, the registered
    corpus-level gates -- the `views.expected_rows` range and the terminal-weight
    floor -- must have passed. The trainer runs this function, so a production
    corpus that missed its row range cannot be trained on even if its files are
    on disk. An in-process build (the dev canary, tests) is labelled as such and
    is measured against neither: those gates describe the study corpus.

    The GPU requirement is stated separately from the chain requirement on
    purpose: a CPU-scripted corpus is honest evidence about the harness and a
    perfectly good test fixture, but it is not a corpus a reportable checkpoint
    may be trained on, and "the fixture trained fine" is exactly how an
    unattributable adapter would get locked.
    """
    from agentlab.multidistill import require_one_producer

    if not rows:
        raise SystemExit(
            "REFUSED: a zero-row view corpus is not a corpus. An empty file is "
            "indistinguishable from a finished build to every resume check in this "
            "chain, so nothing-was-built stops here: check the accepted corpus and "
            "the view report's `rejected` counts.")
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
        if report.get("row_ids") is not None and int(report["row_ids"]) != len(set(ids)):
            raise SystemExit(
                f"REFUSED: the view report counts {report['row_ids']} distinct row "
                f"ids and the metadata holds {len(set(ids))}. The receipt names the "
                f"rows it certifies; a different set of rows is a different corpus.")
        recorded_digest = report.get("row_ids_sha256")
        if recorded_digest and recorded_digest != row_ids_digest(meta):
            raise SystemExit(
                f"REFUSED: the view report's row-id digest {recorded_digest[:12]}... "
                f"does not match the metadata in hand ({row_ids_digest(meta)[:12]}"
                f"...). Completion means a receipt that matches these bytes, not a "
                f"path that exists: rebuild the corpus with `python -m "
                f"agentlab.suite.datasets`.")
        if report.get("corpus_kind") == PRODUCTION_CORPUS:
            require_expected_rows(report)
    return {"rows": len(rows), "source_provenance": ident,
            "row_ids": len(set(ids)),
            "row_ids_sha256": row_ids_digest(meta),
            "corpus_kind": (report or {}).get("corpus_kind", IN_PROCESS_CORPUS)}


def stratum_shortfall(report: dict, cfg: dict | None = None) -> list[dict]:
    """Where a row shortfall actually happened, worst stratum first.

    A total-only report ("4,912 rows, wanted 5,000") says nothing about what to
    regenerate. Every dropped trajectory is counted in the family/horizon cell it
    was dropped from, so the shortfall can be attributed: a stratum's own
    rows-per-kept-trajectory (or the corpus mean, when it kept nothing) turns its
    dropped trajectories into an estimate of the rows that stratum did not
    contribute. The estimate is DIAGNOSTIC -- the gate is the registered range --
    but it names the stratum that has to be re-rolled.
    """
    cfg = cfg or load_config()
    strata = report.get("strata") or {}
    rows = int(report.get("rows") or 0)
    kept = sum(int(s.get("trajectories") or 0) for s in strata.values())
    # Fallback rate for a stratum that kept nothing: the corpus mean, or -- with
    # nothing kept anywhere -- the minimum a surviving trajectory always yields
    # (the terminal view's copies, which every plan contains).
    mean_rate = (rows / kept) if kept else float(cfg["views"]["terminal_copies"])
    out = []
    for name in sorted(strata):
        cell = strata[name]
        n_rows = int(cell.get("rows") or 0)
        n_kept = int(cell.get("trajectories") or 0)
        n_drop = int(cell.get("trajectories_dropped") or 0)
        if n_drop == 0 and n_rows > 0:
            continue
        rate = (n_rows / n_kept) if n_kept else mean_rate
        out.append({"stratum": name, "rows": n_rows, "trajectories": n_kept,
                    "trajectories_dropped": n_drop,
                    "dropped_reasons": dict(sorted(
                        (cell.get("dropped_reasons") or {}).items())),
                    "rows_lost_estimate": int(round(rate * n_drop))})
    out.sort(key=lambda c: (-c["rows_lost_estimate"], c["rows"], c["stratum"]))
    return out


def require_expected_rows(report: dict, cfg: dict | None = None) -> dict:
    """ENFORCE the registered SFT-view row range and terminal-weight floor.

    `views.expected_rows` (5,000-6,000) and `views.terminal_weight_min` were
    reported and never checked, so a corpus of 300 rows -- or of 4,912 after a
    stratum silently dropped out -- would have been written, resumed over and
    trained on. Both are registered numbers: this raises, it never adjusts them,
    and the failure names the stratum that is short so the fix is a re-roll of
    that cell rather than a lowered gate.
    """
    cfg = cfg or load_config()
    lo, hi = (int(x) for x in cfg["views"]["expected_rows"])
    rows = int(report.get("rows") or 0)
    floor = float(cfg["views"]["terminal_weight_min"])
    weight = float(report.get("terminal_weight") or 0.0)
    problems = []
    if not lo <= rows <= hi:
        problems.append(f"{rows} rows is outside the registered range {lo}-{hi}")
    if weight < floor:
        problems.append(f"terminal weight {weight:.3f} is below the registered "
                        f"floor {floor}")
    if not problems:
        return {"rows": rows, "expected_rows": [lo, hi], "terminal_weight": weight,
                "terminal_weight_min": floor, "ok": True}
    detail = stratum_shortfall(report, cfg)
    lines = [f"REFUSED: this SFT view corpus is not trainable: "
             f"{'; '.join(problems)}."]
    if rows < lo and detail:
        lines.append("  Short strata (worst first), by the trajectories each cell "
                     "lost:")
        for cell in detail[:8]:
            reasons = ", ".join(f"{k} {v}" for k, v in
                                cell["dropped_reasons"].items()) or "none"
            lines.append(
                f"    {cell['stratum']}: {cell['rows']} rows from "
                f"{cell['trajectories']} kept trajectories, "
                f"{cell['trajectories_dropped']} dropped ({reasons}); "
                f"~{cell['rows_lost_estimate']} rows not contributed")
    elif rows < lo:
        lines.append("  No stratum lost a trajectory, so the accepted corpus "
                     "itself is too small: the quotas it passed cannot fill the "
                     "registered row range.")
    if rows > hi:
        lines.append(f"  Rows ABOVE the ceiling means the registered seed-keyed cap "
                     f"({VIEW_CAP_KEY_VERSION}) did not run: `build_views` applies it "
                     f"before it emits a row, so a corpus over {hi} was built by some "
                     f"other path. Rebuild it with `python -m agentlab.suite.datasets`. "
                     f"Do not raise the ceiling and do not hand-trim the rows.")
    lines.append("  The range and the floor are preregistered. Roll out and accept "
                 "more trajectories in the named strata (`python -m "
                 "agentlab.multidistill run` then `finalize`); do not widen the "
                 "range and do not train on a partial corpus.")
    raise SystemExit("\n".join(lines))

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
# There is NO answer regex in this module: `committed_answer` below delegates to
# schema.extract_committed_answer, the one grammar in the repo.


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


def committed_answer(text) -> str | None:
    """What this text COMMITTED, read by the one shared grammar.

    Delegates to `schema.extract_committed_answer`: the preregistered system
    prompt asks for `ANSWER: <value>` and the generated task prompt asks for
    `\\boxed{}`, and the strict verifier, the certification layer and this view
    builder must all read a commitment the same way. When the builder asked with
    its own `\\boxed{}`-only regex, a trajectory the verifier had CERTIFIED --
    terminating with a plain `ANSWER: 55640a29...` -- selected no views at all, so
    trajectories that obeyed the system prompt alone were dropped from the
    training corpus.
    """
    from agentlab.suite.schema import extract_committed_answer

    return extract_committed_answer(str(text or ""))


def select_views(record: dict, cfg: dict | None = None) -> list[dict]:
    """The (msg_index, view, copies) plan for one accepted trajectory."""
    cfg = cfg or load_config()
    v = cfg["views"]
    messages = record["messages"]
    a_idx = assistant_indices(messages)
    if not a_idx:
        return []
    terminal = a_idx[-1]
    if committed_answer(messages[terminal].get("content", "")) is None:
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


def stratum_of(family: str, horizon) -> str:
    """The structural stratum of a view: the family/horizon cell it came from.

    Spelled exactly as the acceptance report spells its cells (`family-hN`), so a
    row shortfall can be attributed to the stratum that is actually short -- and
    re-rolled there -- instead of being reported as one small total.
    """
    return f"{family}-h{int(horizon)}"


def build_views(records: list, token_counter, cfg: dict | None = None):
    """Accepted trajectories -> (SFT rows, build report).

    Rows carry exactly the four keys TRL's completion-only SFT path consumes
    (prompt, completion, tools, chat_template_kwargs); the provenance chain goes
    to the parallel metadata (one row per training row, same order) and its
    summary to the report, so the trainer sees no stray columns.

    The report also carries a per-STRATUM census (rows, trajectories kept, and
    every trajectory dropped with its reason, per family/horizon cell). It is
    what `require_expected_rows` reads to name which stratum is short instead of
    reporting only that the corpus total is small.

    OVER-FULL is also a refusal, so the registered seed-keyed cap
    (`plan_view_cap`) runs over the built trajectories before any row is emitted.
    It is inert below the ceiling, which is every in-process build and every test
    corpus here. Rows are counted first and capped second because the cap is
    defined on rows: the view plan does not produce a fixed number per trajectory.
    """
    from agentlab.multidistill import (provenance_gaps, require_one_producer,
                                       row_digest)

    cfg = cfg or load_config()
    max_tokens = cfg["acceptance"]["max_view_tokens"]
    schema_cache: dict[str, list] = {}
    rows, meta, built = [], [], []
    counts = {"terminal": 0, "pivot": 0, "recovery": 0}
    rejected = {"over_token_budget": 0, "no_committed_answer": 0,
                "trajectory_over_budget": 0, "stale_environment_contract": 0,
                "missing_source_provenance": 0}
    strata: dict[str, dict] = {}

    def stratum(name: str) -> dict:
        # `trajectories_capped` is deliberately NOT `trajectories_dropped`: a
        # capped trajectory was eligible and complete, and counting it as a drop
        # would feed `stratum_shortfall`'s lost-row estimate with rows that were
        # removed on purpose.
        return strata.setdefault(name, {"rows": 0, "trajectories": 0,
                                        "trajectories_dropped": 0,
                                        "trajectories_capped": 0,
                                        "dropped_reasons": {},
                                        "view_counts": {"terminal": 0, "pivot": 0,
                                                        "recovery": 0}})

    def drop(name: str, reason: str) -> None:
        cell = stratum(name)
        cell["trajectories_dropped"] += 1
        cell["dropped_reasons"][reason] = cell["dropped_reasons"].get(reason, 0) + 1

    def cell_name(rec: dict) -> str:
        try:
            return stratum_of(rec["family"], rec["horizon"])
        except (KeyError, TypeError, ValueError):
            return "unattributable-stratum"

    records, stale = contract.invalidate(records, "accepted trajectory")
    rejected["stale_environment_contract"] = len(stale)
    for rec in stale:
        drop(cell_name(rec), "stale_environment_contract")
    # A trajectory that cannot name its producer cannot be trained on: the view
    # inherits the snapshot, so there is nothing honest to inherit. Dropped and
    # COUNTED, never quietly built from.
    attributable, unattributable = [], 0
    for rec in records:
        if provenance_gaps(rec.get("provenance")):
            unattributable += 1
            drop(cell_name(rec), "missing_source_provenance")
            continue
        attributable.append(rec)
    rejected["missing_source_provenance"] = unattributable
    records = attributable
    source_provenance = require_one_producer(records, "the SFT view corpus")
    schema_sha: dict[str, str] = {}
    for rec in records:
        family = rec["family"]
        cell = cell_name(rec)
        source_sha = row_digest(rec)
        rec_prov = dict(rec["provenance"])
        tools = schema_cache.setdefault(family, tool_schemas_for_family(family))
        if family not in schema_sha:
            from agentlab.suite.runtime import tool_schema_bytes
            from agentlab.suite.schema import digest_text

            schema_sha[family] = digest_text(tool_schema_bytes(family))
        plan = select_views(rec, cfg)
        if not plan:
            # No committed final answer under the ONE shared grammar (or no
            # assistant turn at all): nothing here is worth supervising.
            rejected["no_committed_answer"] += 1
            drop(cell, "no_committed_answer")
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

        if not candidate_rows:
            drop(cell, "trajectory_over_budget")
            continue
        built.append({"stratum": cell, "task_id": rec["task_id"], "family": family,
                      "horizon": rec["horizon"],
                      "fault_types": list(rec.get("fault_types") or []),
                      "source_sha": source_sha, "provenance": rec_prov,
                      "rows": candidate_rows})

    # THE REGISTERED CEILING. Every trajectory above is eligible and complete;
    # the cap decides how many of them the registered range admits, using only
    # stratum and task id. Nothing below this line reads a score or a verdict.
    cap = plan_view_cap([{"stratum": b["stratum"], "task_id": b["task_id"],
                          "rows": len(b["rows"])} for b in built], cfg)
    keep = set(cap["kept_indexes"])
    for index, b in enumerate(built):
        cell = b["stratum"]
        if index not in keep:
            stratum(cell)["trajectories_capped"] += 1
            continue
        family = b["family"]
        stratum(cell)["trajectories"] += 1
        for view, row, n_tok, msg_index, copy_index in b["rows"]:
            rows.append(row)
            counts[view] += 1
            stratum(cell)["rows"] += 1
            stratum(cell)["view_counts"][view] += 1
            meta.append({"row_id": view_row_id(b["task_id"], view, msg_index,
                                               copy_index),
                         "task_id": b["task_id"], "family": family,
                         "horizon": b["horizon"],
                         "fault_types": list(b["fault_types"]),
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
                         "source_row_sha256": b["source_sha"],
                         "source_provenance": b["provenance"],
                         "runtime_manifest_sha256":
                             b["provenance"].get("runtime_manifest_sha256"),
                         "session_id": b["provenance"].get("session_id"),
                         "gpu_execution": bool(b["provenance"].get("gpu_execution"))})

    total = len(rows)
    terminal_weight = counts["terminal"] / total if total else 0.0
    lo, hi = (int(x) for x in cfg["views"]["expected_rows"])
    report = {
        "rows": total,
        "view_counts": counts,
        "rejected": rejected,
        # The per-stratum census: which family/horizon cells the rows came from,
        # and every trajectory that was dropped, with its reason, in the cell it
        # was dropped from. A total-only report cannot say which stratum is short.
        "strata": {k: strata[k] for k in sorted(strata)},
        "terminal_weight": round(terminal_weight, 4),
        "terminal_weight_min": cfg["views"]["terminal_weight_min"],
        "terminal_weight_ok": terminal_weight >= cfg["views"]["terminal_weight_min"],
        "expected_rows": cfg["views"]["expected_rows"],
        "rows_in_expected_range": lo <= total <= hi,
        # The over-full side of that range, and its receipt: what the full build
        # was, what the seed-keyed cap kept per stratum, and the digest of the
        # kept identities so the decision replays. `applied: false` is the normal
        # case and says the cap did nothing.
        "view_cap": {k: v for k, v in cap.items() if k != "kept_indexes"},
        # The receipt half of the report: what a reader must be able to check
        # against the rows and metadata it actually loaded (count and identity),
        # so a truncated corpus cannot pass as this build.
        "row_ids": len({m["row_id"] for m in meta}),
        "row_ids_sha256": row_ids_digest(meta),
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
        # An in-process build until a writer says otherwise: the CLI relabels the
        # corpus it validated and wrote as the production one, and the registered
        # size gates apply to exactly that label.
        "corpus_kind": IN_PROCESS_CORPUS,
    }
    return rows, meta, report


# --------------------------------------------------------------------------
# writing the corpus: three files, validated first, row file LAST
# --------------------------------------------------------------------------

def _write_atomic(path: pathlib.Path, text: str) -> pathlib.Path:
    """Write via a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_view_corpus(rows: list, meta: list, report: dict, *, out, meta_out,
                      report_out) -> dict:
    """Publish a validated view corpus, in the order the resumers read it.

    The three files are written atomically and the ROW FILE LAST, because that is
    the path both resumers test for: `scripts/run_multifaceted_chain.sh` skips the
    views stage when `sft_views.jsonl` exists, and `agentlab.sft` refuses only
    when a file is missing. Writing the rows first meant a build killed halfway
    left a row file with no metadata and no receipt that the next invocation
    treated as a finished corpus. Now the rows appear only after the metadata and
    the receipt that describe them are already on disk, and any pre-existing
    receipt is removed BEFORE the corpus changes, so a crash can never leave a
    receipt describing rows it does not match.
    """
    out, meta_out = pathlib.Path(out), pathlib.Path(meta_out)
    report_out = pathlib.Path(report_out)
    # Invalidate the old completion marker first: while the corpus is being
    # replaced there must be no receipt on disk claiming it is finished.
    for stale in (report_out, out):
        if stale.exists():
            stale.unlink()
    _write_atomic(meta_out, "".join(json.dumps(m, ensure_ascii=False) + "\n"
                                    for m in meta))
    _write_atomic(report_out, json.dumps(report, indent=2) + "\n")
    _write_atomic(out, "".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in rows))
    return {"rows_path": str(out), "meta_path": str(meta_out),
            "report_path": str(report_out)}


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

    from agentlab.multidistill import require_accepted_corpus

    src = pathlib.Path(args.accepted)
    # The accepted corpus is trusted because its OWN receipt validates -- expected
    # shards, kept task ids, count, digest and passing quotas -- not because the
    # path exists. A quota-missing or half-finalized corpus stops the chain here.
    accepted_receipt = require_accepted_corpus(src)
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    rows, meta, report = build_views(records, default_token_counter())
    report["corpus_kind"] = PRODUCTION_CORPUS
    report["accepted_corpus"] = {
        "path": str(src),
        "sha256": accepted_receipt["corpus_sha256"],
        "accepted": accepted_receipt["accepted"],
        "receipt_sha256": accepted_receipt["receipt_sha256"]}

    print(f"[views] {report['rows']} rows built from {report['source_trajectories']} "
          f"accepted trajectories")
    print(f"[views] counts {report['view_counts']}  rejected {report['rejected']}")
    capinfo = report["view_cap"]
    if capinfo["applied"]:
        print(f"[views] view cap {capinfo['key_version']} APPLIED: the full build was "
              f"{capinfo['full_rows']} rows from {capinfo['full_trajectories']} "
              f"trajectories, above the registered ceiling "
              f"{capinfo['expected_rows'][1]}; {capinfo['trajectories_capped']} "
              f"trajectories were capped out by seed-keyed order (never by score, "
              f"reward or length), leaving {capinfo['rows']} rows")
        for name in sorted(capinfo["per_stratum"]):
            cell = capinfo["per_stratum"][name]
            if cell["trajectories_capped"]:
                print(f"[views]   {name}: kept {cell['trajectories']}/"
                      f"{cell['full_trajectories']} trajectories, "
                      f"{cell['rows']}/{cell['full_rows']} rows")
    else:
        print(f"[views] view cap {capinfo['key_version']} inert: "
              f"{capinfo['full_rows']} rows is within the registered ceiling "
              f"{capinfo['expected_rows'][1]}")
    print(f"[views] provenance: {len(report['source_sessions'])} producer "
          f"session(s), gpu_execution={report['gpu_execution']}")
    print(f"[views] terminal weight {report['terminal_weight']:.3f} "
          f"(min {report['terminal_weight_min']}), rows "
          f"{report['rows']} in {report['expected_rows']}: "
          f"{report['rows_in_expected_range']}")
    # NOTHING IS WRITTEN UNTIL BOTH GATES PASS. The chain is checked here and not
    # only in the trainer, and the registered row range and terminal-weight floor
    # are ENFORCED here rather than reported: a corpus on disk is a corpus the
    # chain script and the trainer treat as finished work.
    require_views_chain(rows, meta, report, require_gpu_source=False)
    require_expected_rows(report)
    paths = write_view_corpus(rows, meta, report, out=args.out,
                             meta_out=args.meta_out, report_out=args.report_out)
    print(f"[views] {report['rows']} rows -> {paths['rows_path']}")
    # Read the corpus back through the same gate a consumer uses: the receipt on
    # disk must describe the bytes on disk.
    reread_rows = [json.loads(line) for line in
                   pathlib.Path(paths["rows_path"]).read_text(
                       encoding="utf-8").splitlines() if line.strip()]
    reread_meta = [json.loads(line) for line in
                   pathlib.Path(paths["meta_path"]).read_text(
                       encoding="utf-8").splitlines() if line.strip()]
    reread_report = json.loads(pathlib.Path(paths["report_path"]).read_text(
        encoding="utf-8"))
    checked = require_views_chain(reread_rows, reread_meta, reread_report,
                                 require_gpu_source=False)
    print(f"[views] receipt verified on disk: {checked['rows']} rows, "
          f"row-id digest {checked['row_ids_sha256'][:12]}..., "
          f"corpus_kind={checked['corpus_kind']}")


if __name__ == "__main__":
    main()
