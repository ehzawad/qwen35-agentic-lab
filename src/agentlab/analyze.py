"""Verdict on a comparison run, computed instead of eyeballed.

Reads the eval JSONs and JSONL traces that eval.py writes, and answers the
question the run was designed to ask -- did training change behaviour? -- with
confidence intervals and pre-registered gates rather than a glance at a table.

The gates were stated before the RS-SFT chain launched, so they cannot drift to
fit the result:

  G1  rssft accuracy >= 0.800          (recover the base; broken SFT hit 0.050)
  G2  rssft calls/episode <= 6.0       (base 3.3; broken SFT 50.0)
  G3  rssft runaway episodes <= 10%    (>10 calls; base ~6%, broken SFT 95%)
  G4  rssft no-box failures < base's   (the failure mode RS-SFT targets)
  G5  rsgrpo accuracy >= rssft         (directional; may be within noise)

Design decisions that came out of adversarial review of this module itself:

  * A SKIPPED gate is never a FAILED gate. Pronouncing "hypothesis not
    supported" because a trace file was missing is the harness-vs-model
    confusion this module exists to prevent.
  * Every checkpoint is evaluated on the SAME seeded problems, so pairwise
    comparisons use McNemar on the discordant pairs (joined by episode index),
    which resolves smaller differences than the unpaired z-test at equal n.
    The unpaired test and MDE are still printed, labelled as conservative.
  * The scorer-blind check compares each episode's boxed content against its
    recorded ground truth -- an independent scoring path -- rather than keying
    on "accuracy is exactly zero", which both misses partial scorer bugs and
    falsely brands honest zero-scoring models as broken.

Stdlib only -- no scipy on the box.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

Z95 = 1.959964
Z80 = 0.841621  # power term for MDE

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _numeric(s):
    from .chat import numeric_answer

    return numeric_answer(s)



def _boxed_matches_gt(episode: dict) -> bool | None:
    """Independently re-score one episode: last boxed value vs ground truth.

    Returns None when there is nothing to compare (no box, or no ground truth
    recorded), True/False otherwise. This is the third scoring path that lets
    the sanity layer catch a scorer that mis-reads a correct answer.
    """
    gt = _numeric(episode.get("ground_truth"))
    hits = _BOXED_RE.findall(str(episode.get("final", "")))
    if gt is None or not hits:
        return None
    got = _numeric(hits[-1])
    return got is not None and abs(got - gt) < 1e-4


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(point, lo, hi) Wilson score interval; exact at the k=0 and k=n edges."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Analytically lo=0 at k=0 and hi=1 at k=n; float evaluation can land one
    # ulp inside, inverting the documented lo <= p <= hi ordering.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return p, min(lo, p), max(hi, p)


def two_prop_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(z, two_sided_p) pooled two-proportion z-test. Assumes INDEPENDENT samples."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """(z, two_sided_p) McNemar test on discordant pairs.

    b = pairs where only the FIRST condition succeeded, c = only the second.
    The correct instrument when both checkpoints answered the same problems;
    concordant pairs carry no information about the difference. Uses the exact
    binomial for small b+c, the normal approximation otherwise.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n <= 25:  # exact binomial, two-sided
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
        return (c - b) / math.sqrt(n), min(1.0, 2 * tail)
    z = (c - b) / math.sqrt(n)
    return z, math.erfc(abs(z) / math.sqrt(2))


def mde(n1: int, n2: int, p_base: float) -> float:
    """Unpaired minimum detectable difference, alpha=.05 two-sided, 80% power.

    An UPPER BOUND here: the paired design can resolve smaller differences.
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    return (Z95 + Z80) * math.sqrt(2 * p_base * (1 - p_base) * (1 / n1 + 1 / n2) / 2)


def paired_compare(eps_a: list[dict], eps_b: list[dict]) -> dict | None:
    """McNemar over episodes joined by index; None when nothing aligns."""
    def _ok(e):
        return bool(e.get("_ok_rescored", e.get("ok")))

    by_a = {e.get("index"): _ok(e) for e in eps_a if e.get("index") is not None}
    by_b = {e.get("index"): _ok(e) for e in eps_b if e.get("index") is not None}
    common = sorted(set(by_a) & set(by_b))
    if len(common) < 10:
        return None
    b = sum(1 for i in common if by_a[i] and not by_b[i])
    c = sum(1 for i in common if by_b[i] and not by_a[i])
    z, p = mcnemar(b, c)
    return {"n_pairs": len(common), "b": b, "c": c, "z": z, "p": p}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _dedupe(episodes: list[dict]) -> tuple[list[dict], int]:
    """Keep the LAST record per index; append-across-runs leaves stale earlier ones.

    Returns (deduped, n_duplicates). The real base trace on disk carries index 0
    twice -- one episode from a killed run, one from its replacement -- and every
    denominator computed over the raw multiset is silently wrong.
    """
    by_idx: dict = {}
    without_idx = []
    for e in episodes:
        idx = e.get("index")
        if idx is None:
            without_idx.append(e)
        else:
            by_idx[idx] = e
    deduped = sorted(by_idx.values(), key=lambda e: e["index"]) + without_idx
    return deduped, len(episodes) - len(deduped)


def load_tag(tag: str, out_dir: pathlib.Path, trace_dirs: list[pathlib.Path]) -> dict | None:
    """Summary + per-episode behaviour for one checkpoint, or None if not evaluated."""
    summary_path = out_dir / f"eval-{tag}.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())

    episodes = []
    for d in trace_dirs:
        p = d / f"trace-{tag}.jsonl"
        if p.exists():
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
            break  # first dir that has the trace wins

    episodes, n_dupes = _dedupe(episodes)

    n = summary["n"]
    row = {"tag": tag, "summary": summary, "n": n, "episodes": len(episodes),
           "dupes": n_dupes}
    row["acc_k"] = round(summary["accuracy"] * n)
    if episodes:
        calls = [e.get("n_calls", 0) for e in episodes]
        row["calls_mean"] = sum(calls) / len(calls)
        row["calls_max"] = max(calls)
        row["runaway_k"] = sum(1 for c in calls if c > 10)
        row["nobox_k"] = sum(
            1 for e in episodes if not e.get("ok") and "boxed" not in str(e.get("final", ""))
        )
        # Rescore every episode with the corrected normalizer. Review found a
        # real base episode answering \boxed{24\%} against ground truth 24,
        # scored wrong purely on notation. Gates and the paired test use this
        # uniform rescoring; the original ok flags are kept for the sanity layer,
        # which audits the harness that wrote the file.
        for e in episodes:
            m = _boxed_matches_gt(e)
            e["_ok_rescored"] = bool(e.get("ok")) if m is None else m
        row["rescored_k"] = sum(1 for e in episodes if e["_ok_rescored"])
        row["corrections"] = sum(
            1 for e in episodes if e["_ok_rescored"] != bool(e.get("ok"))
        )
        # Referee metric: correct AND actually used a tool. 0.920 with 16
        # no-tool successes is a different claim than 0.920 tool-compliant.
        row["tools_ok_k"] = sum(
            1 for e in episodes if e["_ok_rescored"] and e.get("n_calls", 0) > 0
        )
        row["trace_n"] = len(episodes)
        row["_episodes"] = episodes
    return row


# ---------------------------------------------------------------------------
# harness sanity
# ---------------------------------------------------------------------------

def sanity_checks(row: dict) -> list[tuple[str, str, str]]:
    """Distinguish 'the model is weak' from 'the harness is broken'.

    A weak model is a legitimate result; a broken harness masquerading as one is
    not. Levels: "BUG" = fix the harness before reading any table; "WARN" =
    investigate before concluding. Checks are built to be evidence-based in both
    directions -- an honest zero-scoring model must NOT be branded a bug, and a
    partial scorer bug at nonzero accuracy must not slip through.
    """
    out = []
    eps = row.get("_episodes") or []
    n, summary = row["n"], row["summary"]

    if row.get("dupes"):
        out.append(("WARN", "S0",
                    f"{row['dupes']} duplicate episode indices in the trace (stale appends "
                    f"from an earlier run); deduplicated keeping the last record"))

    # S1: after dedupe the trace must cover the run exactly.
    if eps and len(eps) != n:
        out.append(("BUG", "S1", f"trace has {len(eps)} episodes but eval ran n={n}"))

    if eps:
        # S2: recompute accuracy from the traces.
        ok_k = sum(1 for e in eps if e.get("ok"))
        if abs(ok_k / len(eps) - summary["accuracy"]) > 0.02:
            out.append(("BUG", "S2",
                        f"trace accuracy {ok_k/len(eps):.3f} != summary {summary['accuracy']:.3f}"))

        # S3: independent re-score. An episode whose boxed content EQUALS the
        # recorded ground truth but is marked wrong is a scorer bug at any
        # accuracy level; boxes that are present but wrong are just a weak model.
        blind = sum(1 for e in eps if e.get("ok") is False and _boxed_matches_gt(e) is True)
        if blind >= max(2, 0.05 * len(eps)):
            out.append(("BUG", "S3",
                        f"{blind}/{len(eps)} episodes have a boxed answer equal to their "
                        f"ground truth yet are scored wrong -- scorer-blind signature"))
        elif summary["accuracy"] == 0 and len(eps) >= 10:
            out.append(("WARN", "S3",
                        "accuracy is exactly 0; boxed-vs-ground-truth re-scoring agrees, so "
                        "this is consistent with a genuinely broken policy -- verify one "
                        "episode by hand before concluding"))

        # S4: zero recorded tool use. If the finals visibly CONTAIN tool-call
        # syntax the parser missed, that is a harness bug; a model that simply
        # stopped calling tools is a (bad) result, not a defect.
        if summary.get("tool_use_rate", 1) == 0 and len(eps) >= 10:
            syntactic = sum(1 for e in eps if "<tool_call" in str(e.get("final", "")))
            if syntactic:
                out.append(("BUG", "S4",
                            f"tool_use_rate is 0 but {syntactic}/{len(eps)} finals contain "
                            f"tool-call syntax -- parser-blind signature"))
            else:
                out.append(("WARN", "S4",
                            "tool_use_rate is 0 and no tool-call syntax appears anywhere; "
                            "consistent with policy collapse, worth a manual look"))

        # S5: impossible states, in BOTH directions.
        for e in eps:
            r = e.get("rewards") or {}
            pred, exp = r.get("predicted"), r.get("expected")
            if e.get("ok") and not str(e.get("final", "")).strip():
                out.append(("BUG", "S5", f"episode {e.get('index')} ok=True with empty final"))
                break
            if pred is not None and exp is not None:
                agree = abs(float(pred) - float(exp)) < 1e-4
                if e.get("ok") and not agree:
                    out.append(("BUG", "S5",
                                f"episode {e.get('index')} ok=True but predicted != expected"))
                    break
                if e.get("ok") is False and agree:
                    out.append(("BUG", "S5",
                                f"episode {e.get('index')} ok=False but predicted == expected"))
                    break

        # S6: mass-duplicate finals are a generation/indexing bug signature.
        finals = [str(e.get("final", ""))[:200] for e in eps if str(e.get("final", "")).strip()]
        if len(finals) >= 10:
            from collections import Counter
            top = Counter(finals).most_common(1)[0][1]
            if top / len(finals) > 0.5:
                # WARN, not BUG: from the trace alone this is indistinguishable
                # from a policy that collapsed to one canned answer, which is a
                # (dire) result rather than a defect. A genuine indexing fault
                # normally also trips S0/S1/S2, so those carry the BUG verdict.
                out.append(("WARN", "S6",
                            f"{top}/{len(finals)} finals are identical -- either an "
                            f"indexing/generation fault or a collapsed policy; check S0/S1/S2 "
                            f"and read one episode before concluding"))

        # S7: the summary and the trace must agree on tool use.
        used = sum(1 for e in eps if e.get("n_calls", 0) > 0)
        if abs(used / len(eps) - summary.get("tool_use_rate", 0)) > 0.05:
            out.append(("BUG", "S7",
                        f"trace tool use {used/len(eps):.2f} != summary "
                        f"{summary.get('tool_use_rate'):.2f}"))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def gate(name: str, ok: bool, detail: str, lines: list[str]) -> bool:
    lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def report(tags: list[str], out_dir: str = "out", trace_dirs: list[str] | None = None) -> str:
    out_p = pathlib.Path(out_dir)
    dirs = [pathlib.Path(d) for d in (trace_dirs or ["out/chain", "out/comparison", "out"])]
    rows = {t: r for t in tags if (r := load_tag(t, out_p, dirs))}

    lines = ["# Comparison verdict", ""]
    hdr = (f"{'ckpt':<8}{'accuracy (95% CI)':>26}{'calls/ep':>10}{'runaway':>10}"
           f"{'no-box':>9}{'tool-ok':>9}{'n':>6}")
    lines += [hdr, "-" * len(hdr)]
    for t, r in rows.items():
        if "rescored_k" in r:
            r["acc_k"] = r["rescored_k"]  # gates/CIs on uniformly corrected scoring
        acc = fmt_ci(r["acc_k"], r["n"])
        calls = f"{r.get('calls_mean', float('nan')):.1f}" if "calls_mean" in r else "-"
        run = (f"{r['runaway_k']}/{r['trace_n']}" if "runaway_k" in r else "-")
        nob = (f"{r['nobox_k']}/{r['trace_n']}" if "nobox_k" in r else "-")
        tok = (f"{r['tools_ok_k']/r['trace_n']:.3f}" if "tools_ok_k" in r else "-")
        lines.append(f"{t:<8}{acc:>26}{calls:>10}{run:>10}{nob:>9}{tok:>9}{r['n']:>6}")
    lines.append("")
    corr = {t: r["corrections"] for t, r in rows.items() if r.get("corrections")}
    if corr:
        lines.append("rescoring: " + ", ".join(f"{t}: {k} episode(s) corrected" for t, k in corr.items())
                     + "  (notation-tolerant normalizer; original flags kept for the sanity layer)")
        lines.append("")

    # Harness sanity FIRST. A "weak model" verdict is only meaningful once these
    # are clean; a BUG here means fix the harness, not blame the model.
    lines.append("## Harness sanity (must be clean before reading anything above)")
    any_bug = False
    for t, r in rows.items():
        issues = sanity_checks(r)
        if not issues:
            lines.append(f"  OK    {t}")
        for level, code, msg in issues:
            any_bug = any_bug or level == "BUG"
            lines.append(f"  {level:<5} {t} {code}: {msg}")
    lines.append("")

    base, rssft, rsgrpo = rows.get("base"), rows.get("rssft"), rows.get("rsgrpo")

    # pairwise comparisons: paired McNemar when episodes align, unpaired as backup
    def compare(a, b, label):
        pa = paired_compare(a.get("_episodes") or [], b.get("_episodes") or [])
        d = b["acc_k"] / b["n"] - a["acc_k"] / a["n"]
        if pa:
            lines.append(f"{label}: {d:+.3f}  paired McNemar over {pa['n_pairs']} shared "
                         f"problems: b={pa['b']} c={pa['c']} z={pa['z']:+.2f} p={pa['p']:.3f}")
        z, p = two_prop_test(b["acc_k"], b["n"], a["acc_k"], a["n"])
        m = mde(a["n"], b["n"], a["acc_k"] / a["n"])
        lines.append(f"{'' if not pa else '  '}unpaired (conservative): z={z:+.2f} p={p:.3f}; "
                     f"unpaired MDE ~ {m:.3f} (paired design resolves less)")
        if a["n"] != b["n"]:
            lines.append(f"  note: unequal n ({a['n']} vs {b['n']}); the smaller run is a "
                         f"seeded prefix of the larger, so the paired test covers the overlap.")

    if base and rssft:
        compare(base, rssft, "rssft - base ")
    if rssft and rsgrpo:
        compare(rssft, rsgrpo, "rsgrpo - rssft")
    lines.append("")

    # pre-registered gates: PASS / FAIL / SKIP are three different things
    gates_hdr = "## Gates (registered before launch)"
    if any_bug:
        gates_hdr += "   [SUSPECT: harness bug flagged above -- see final verdict]"
    lines.append(gates_hdr)
    passed = failed = skipped = 0
    failed_names: list[str] = []
    if rssft:
        acc = rssft["acc_k"] / rssft["n"]
        ok = gate("G1 accuracy >= 0.800", acc >= 0.800, f"rssft {acc:.3f}", lines)
        passed, failed = passed + ok, failed + (not ok)
        if not ok:
            failed_names.append("G1")
        if "calls_mean" in rssft:
            base_calls = f"{base['calls_mean']:.1f}" if base and "calls_mean" in base else "n/a"
            ok = gate("G2 calls/ep <= 6.0", rssft["calls_mean"] <= 6.0,
                      f"rssft {rssft['calls_mean']:.1f} (base {base_calls}, broken 50.0)", lines)
            passed, failed = passed + ok, failed + (not ok)
            if not ok:
                failed_names.append("G2")
            ok = gate("G3 runaway <= 10%", rssft["runaway_k"] / rssft["trace_n"] <= 0.10,
                      f"{rssft['runaway_k']}/{rssft['trace_n']}", lines)
            passed, failed = passed + ok, failed + (not ok)
            if not ok:
                failed_names.append("G3")
        else:
            skipped += 2
            lines.append("  SKIP  G2/G3: no trace found for rssft (locate the trace; not a model result)")
        if base and "nobox_k" in rssft and "nobox_k" in base:
            b_r = base["nobox_k"] / base["trace_n"]
            r_r = rssft["nobox_k"] / rssft["trace_n"]
            ok = gate("G4 no-box < base", r_r < b_r, f"rssft {r_r:.1%} vs base {b_r:.1%}", lines)
            passed, failed = passed + ok, failed + (not ok)
            if not ok:
                failed_names.append("G4")
        else:
            skipped += 1
            lines.append("  SKIP  G4: missing no-box data")
    else:
        lines.append("  (rssft not evaluated yet)")
    if "rsgrpo" in tags and rssft:
        # G5 needs corroborating traces on BOTH sides. A summary json alone is a
        # self-reported number with no evidence behind it -- the dress rehearsal
        # produced a full false-success from exactly that state, and the silent
        # variant (rsgrpo missing entirely, gate never mentioned) was reproduced
        # against the real output directory. Requested-but-missing is a SKIP.
        if rsgrpo and rsgrpo.get("_episodes") and rssft.get("_episodes"):
            ok = gate("G5 rsgrpo >= rssft (directional)",
                      rsgrpo["acc_k"] / rsgrpo["n"] >= rssft["acc_k"] / rssft["n"],
                      f"{rsgrpo['acc_k']/rsgrpo['n']:.3f} vs {rssft['acc_k']/rssft['n']:.3f}", lines)
            passed, failed = passed + ok, failed + (not ok)
            if not ok:
                failed_names.append("G5")
        else:
            skipped += 1
            why = "no eval json" if not rsgrpo else "no trace episodes"
            lines.append(f"  SKIP  G5: rsgrpo {why} -- locate the data; not a model result")

    lines.append("")
    lines.append(f"## Verdict: {passed} passed, {failed} failed, {skipped} skipped")
    if any_bug:
        lines.append("HARNESS BUG DETECTED -- the numbers above are NOT a statement about the")
        lines.append("model. Fix the flagged checks and re-run before drawing any conclusion.")
    elif skipped:
        lines.append("INCOMPLETE DATA -- gates were skipped, so no verdict on the hypothesis is")
        lines.append("issued. Locate the missing traces and re-run the analyzer.")
    elif rssft and failed == 0:
        lines.append("Single-turn SFT destroyed termination; outcome-filtered multi-turn SFT restored it.")
    elif rssft and failed_names and set(failed_names) == {"G5"}:
        lines.append("RS-SFT's own gates (G1-G4) all passed -- the restoration result stands.")
        lines.append("The additional GRPO stage did not improve on RS-SFT (G5): a real null for")
        lines.append("the RL add-on on this task, not evidence against RS-SFT.")
    elif rssft:
        lines.append("Harness is clean and data complete, so this is a real model result: the")
        lines.append("RS-SFT hypothesis is NOT supported at these gates. Read the traces first.")
    return "\n".join(lines)


# ===========================================================================
# agentic verdict (multifaceted suite)
#
# Everything below implements the preregistered evaluation contract in
# configs/agentic_preregister.json and docs/AGENTIC_PROTOCOL.md: harness
# vetoes S8-S18 first, then gates ER1-ER8 / MT1-MT6 / HR1-HR3, launch floors,
# and the winner rule. Four outcome states everywhere:
#
#   PASS / FAIL      a real model-level result (positive or negative)
#   INCONCLUSIVE     evidence missing or underpowered; NEVER read favourably
#   BUG              the harness is broken; vetoes every claim and the winner
# ===========================================================================

OUTCOME_STATES = ("PASS", "FAIL", "INCONCLUSIVE", "BUG")


def _g(status: str, detail: str, **numbers) -> dict:
    assert status in OUTCOME_STATES + ("OK", "WARN")
    return {"status": status, "detail": detail, "numbers": numbers}


def load_preregister(path: str | pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_agentic_episodes(traces_dir: str | pathlib.Path, secret: bytes) -> dict:
    """Recompute every episode's scores from raw messages/events (S17 input).

    Returns {(arm, condition, control): {task_id: episode}} with the LAST
    record kept per task (dedupe against stale appends).
    """
    from agentlab import provenance

    by_key: dict = {}
    for path in sorted(pathlib.Path(traces_dir).glob("*.jsonl")):
        for rec in _load_jsonl(path):
            if rec.get("kind") != "episode":
                continue
            key = (rec.get("arm"), rec.get("condition"), rec.get("control", "none"))
            rep = provenance.certify_episode(rec, secret)
            ep = {"task_id": rec.get("task_id"), "family": rec.get("family"),
                  "horizon": rec.get("horizon"), "template": rec.get("template_id"),
                  "pattern_id": rec.get("pattern_id"),
                  "all_tools_required": bool(rec.get("all_tools_required")),
                  "rep": rep, "trace": rec}
            if key[1] in ("faulted", "stress"):
                ep["rec"] = provenance.certify_recovery(rec, secret, rep)
            if ep["all_tools_required"]:
                ep["orch"] = provenance.certify_orchestration(rec, secret, rep)
            by_key.setdefault(key, {})[ep["task_id"]] = ep
    return by_key


def _crashish(ep: dict) -> bool:
    term = (ep["trace"].get("runner") or {}).get("termination_reason")
    return ep["rep"]["runaway"]["runaway"] or term in (
        "parser_budget", "wall_clock", "spec_error")


# ---------------------------------------------------------------------------
# harness vetoes S8-S18
# ---------------------------------------------------------------------------

def veto_s8_pairing(eps: dict, arms=("BP", "TP")) -> dict:
    """Identical task IDs, spec values, budgets, decoding, prompt across arms."""
    problems = []
    for condition in ("clean", "faulted"):
        a = eps.get((arms[0], condition, "none"), {})
        b = eps.get((arms[1], condition, "none"), {})
        if not a and not b:
            continue
        if set(a) != set(b):
            problems.append(f"{condition}: task sets differ "
                            f"({len(set(a) ^ set(b))} mismatched)")
            continue
        for tid in a:
            ta, tb = a[tid]["trace"], b[tid]["trace"]
            for field in ("budgets", "decode"):
                if ta.get(field) != tb.get(field):
                    problems.append(f"{condition}/{tid}: {field} differ")
                    break
            if (ta.get("provenance", {}).get("spec_sha256")
                    != tb.get("provenance", {}).get("spec_sha256")):
                problems.append(f"{condition}/{tid}: spec digests differ")
            if (ta.get("prompt", {}).get("sha256") != tb.get("prompt", {}).get("sha256")):
                problems.append(f"{condition}/{tid}: prompts differ between arms")
            if ta.get("fault") != tb.get("fault"):
                problems.append(f"{condition}/{tid}: fault assignment differs")
            if problems:
                break
    if problems:
        return _g("BUG", "; ".join(problems[:4]))
    return _g("OK", f"arms {arms[0]}/{arms[1]} identically paired")


def veto_s9_oracle(specs: list[dict] | None) -> dict:
    if not specs:
        return _g("INCONCLUSIVE", "no spec manifest supplied; oracle reachability unverified")
    from agentlab import provenance

    bad = []
    for spec in specs:
        res = provenance.verify_oracle(spec)
        if not res["ok"]:
            bad.append(f"{spec.get('task_id')}: {'; '.join(res['problems'][:1])}")
    if bad:
        return _g("BUG", f"{len(bad)} unreachable/mis-declared specs, e.g. {bad[:3]}")
    return _g("OK", f"all {len(specs)} specs replay to their declared horizon and answer")


def veto_s10_splits(split_manifests: dict[str, list[dict]] | None) -> dict:
    if not split_manifests or len(split_manifests) < 2:
        return _g("INCONCLUSIVE", "split manifests not supplied; leakage unverified")
    from agentlab.suite.splits import check_split_leakage

    violations = check_split_leakage(split_manifests)
    if violations:
        return _g("BUG", f"{len(violations)} split overlaps: "
                  + "; ".join(f"{v['kind']} across {v['splits']}" for v in violations[:3]))
    return _g("OK", f"no overlap across {sorted(split_manifests)}")


def veto_s11_absent_info(eps: dict, prereg: dict, arms=("BP", "TP")) -> dict:
    need = int(prereg["controls"]["absent_information"]["n_per_family"])
    per_family: dict = {}
    leaks = []
    for (arm, condition, control), tasks in eps.items():
        if control != "redacted" or arm not in arms:
            continue
        for ep in tasks.values():
            per_family[ep["family"]] = per_family.get(ep["family"], 0) + 1
            if ep["rep"]["raw_success"] or ep["rep"]["certified_success"]:
                leaks.append(f"{arm}/{ep['task_id']}")
    if leaks:
        return _g("BUG", f"redacted-control SUCCESS (harness leakage): {leaks[:3]}",
                  leaks=len(leaks))
    if not per_family:
        return _g("INCONCLUSIVE", "no redacted-control traces found")
    short = {f: n for f, n in per_family.items() if n < need * len(arms)}
    if short:
        return _g("INCONCLUSIVE", f"redacted coverage below {need}/family/arm: {short}",
                  counts=per_family)
    return _g("OK", f"zero raw and certified success on {sum(per_family.values())} "
              f"redacted instances", counts=per_family)


def veto_s12_injection(eps: dict, specs_by_id: dict | None) -> dict:
    from agentlab import provenance

    problems = []
    n_checked = 0
    for (arm, condition, control), tasks in eps.items():
        if condition != "faulted" or control != "none":
            continue
        for ep in tasks.values():
            n_checked += 1
            events = ep["trace"].get("events", [])
            emitted = [e for e in events if e.get("fault_emitted")]
            if len(emitted) > 1:
                problems.append(f"{arm}/{ep['task_id']}: fault fired {len(emitted)}x")
                continue
            if emitted and specs_by_id:
                spec = specs_by_id.get(ep["task_id"])
                fault = ep["trace"].get("fault") or {}
                if spec is not None and fault:
                    replay = provenance.execute_oracle(spec)
                    idx = fault.get("node_index")
                    if replay["ok"] and idx is not None and idx < len(replay["nodes"]):
                        want = replay["nodes"][idx]["args_digest"]
                        if emitted[0].get("args_digest") != want:
                            problems.append(f"{arm}/{ep['task_id']}: fault fired at the "
                                            f"wrong node")
                    budgets = ep["trace"].get("budgets", {})
                    remaining = budgets.get("max_calls", 0) - emitted[0].get("call_id", 0)
                    needed = len(spec.get("oracle", [])) - (idx or 0)
                    if remaining < needed:
                        problems.append(f"{arm}/{ep['task_id']}: unrecoverable injection "
                                        f"(remaining {remaining} < needed {needed})")
    if problems:
        return _g("BUG", "; ".join(problems[:4]), checked=n_checked)
    if n_checked == 0:
        return _g("INCONCLUSIVE", "no faulted traces found")
    return _g("OK", f"{n_checked} faulted episodes: single emission at the registered "
              f"node, recoverable", checked=n_checked)


def veto_s13_receipts(eps: dict) -> dict:
    bad = []
    for (arm, condition, control), tasks in eps.items():
        for ep in tasks.values():
            if not ep["rep"]["receipts_ok"]:
                bad.append(f"{arm}/{condition}/{ep['task_id']}")
    if bad:
        return _g("BUG", f"invalid receipt chains: {bad[:3]} ({len(bad)} total)")
    return _g("OK", "every event receipt validates against the run secret")


def veto_s14_counterfactual(eps: dict, specs: list[dict] | None, prereg: dict,
                            arms=("BP", "TP")) -> dict:
    from agentlab import provenance

    problems = []
    # (a) generation-time sensitivity: mutating the hidden value must change
    # the replayed answer and the scorer decision.
    if specs:
        import copy
        checked = 0
        for spec in specs:
            hk, field = spec.get("hidden_key"), spec.get("answer_field", "code")
            if not hk or hk not in spec.get("kb", {}):
                if not spec.get("counterfactual_sensitive"):
                    problems.append(f"{spec.get('task_id')}: no hidden_key and no "
                                    f"counterfactual_sensitive flag")
                continue
            rec = spec["kb"][hk]
            if not isinstance(rec, dict) or field not in rec:
                continue
            mutant = copy.deepcopy(spec)
            mutant["kb"][hk][field] = "CF" + "0" * 14
            res = provenance.execute_oracle(mutant)
            if res["ok"] and str(res["answer"]) == str(spec.get("answer")):
                problems.append(f"{spec.get('task_id')}: answer insensitive to the "
                                f"hidden value")
            checked += 1
            if checked >= 200:
                break
    # (b) permuted replays must be scored against the permuted answer.
    n_perm = 0
    for (arm, condition, control), tasks in eps.items():
        if control != "permuted" or arm not in arms:
            continue
        for ep in tasks.values():
            n_perm += 1
            trace = ep["trace"]
            got = provenance.extract_final_answer(
                provenance._final_assistant_text(trace))
            if got is None:
                continue
            recorded = str(trace.get("answer", ""))
            if (ep["rep"]["raw_success"]
                    and got.strip().lower() != recorded.strip().lower()):
                problems.append(f"{arm}/{ep['task_id']}: scored success without "
                                f"matching the permuted answer")
    if problems:
        return _g("BUG", "; ".join(problems[:4]))
    n_min = int(prereg["controls"]["counterfactual_permutation"]["n_min"])
    if specs is None and n_perm == 0:
        return _g("INCONCLUSIVE", "no specs and no permuted traces supplied")
    if n_perm < n_min:
        return _g("INCONCLUSIVE", f"only {n_perm} permuted replays (< {n_min})")
    return _g("OK", f"counterfactual sensitivity holds; {n_perm} permuted replays track "
              f"the returned value")


def veto_s15_attrition(eps: dict, specs_by_id: dict | None, arms=("BP", "TP")) -> dict:
    if not specs_by_id:
        return _g("INCONCLUSIVE", "no spec manifest; attrition unverified")
    missing = {}
    for arm in arms:
        for condition in ("clean", "faulted"):
            have = set(eps.get((arm, condition, "none"), {}))
            if not have:
                missing[f"{arm}/{condition}"] = len(specs_by_id)
                continue
            gap = set(specs_by_id) - have
            if gap:
                missing[f"{arm}/{condition}"] = len(gap)
    if missing:
        return _g("INCONCLUSIVE", f"assigned tasks without traces: {missing} -- "
                  f"INCOMPLETE, never favourable", missing=missing)
    return _g("OK", "every assigned task has a trace in every arm/condition")


def veto_s16_control_integrity(eps: dict, prereg: dict, locks: dict | None) -> dict:
    problems = []
    registered = set(prereg["prompt_candidates"]["sha256"].values())
    winner_sha = (locks or {}).get("prompt_winner", {}).get("sha256")
    base_id = prereg["model"]["base_id"]
    for (arm, condition, control), tasks in eps.items():
        for ep in tasks.values():
            prov = ep["trace"].get("provenance", {})
            psha = ep["trace"].get("prompt", {}).get("sha256")
            if prov.get("base_id") != base_id:
                problems.append(f"{arm}: base_id {prov.get('base_id')!r} != registered")
            if psha not in registered:
                problems.append(f"{arm}: prompt hash not among the eight registered")
            if winner_sha and arm in ("BP", "TP", "RP") and psha != winner_sha:
                problems.append(f"{arm}: prompt is not the locked winner")
            if arm in ("B0", "BP") and prov.get("adapter"):
                problems.append(f"{arm}: adapter loaded in a prompt-only arm")
            break  # provenance is constant per file; one episode suffices
    if problems:
        return _g("BUG", "; ".join(sorted(set(problems))[:4]))
    if not eps:
        return _g("INCONCLUSIVE", "no traces loaded")
    return _g("OK", "checkpoints, prompts, and adapters match the preregistration")


def veto_s17_trace_summary(eps: dict) -> dict:
    problems = []
    for (arm, condition, control), tasks in eps.items():
        for ep in tasks.values():
            trace, rep = ep["trace"], ep["rep"]
            score = trace.get("score", {})
            checks = [("raw_success", rep["raw_success"]),
                      ("certified_success", rep["certified_success"]),
                      ("runaway", rep["runaway"]["runaway"]),
                      ("hallucinated", rep["hallucination"]["hallucinated"])]
            for name, want in checks:
                if name in score and bool(score[name]) != bool(want):
                    problems.append(f"{arm}/{condition}/{ep['task_id']}: {name} "
                                    f"recorded {score[name]} recomputed {want}")
            if "recovery" in score and "rec" in ep:
                if bool(score["recovery"].get("certified_recovery")) != bool(
                        ep["rec"]["certified_recovery"]):
                    problems.append(f"{arm}/{condition}/{ep['task_id']}: recovery disagrees")
            n_calls = (trace.get("runner") or {}).get("n_calls")
            if n_calls is not None and n_calls != len(trace.get("events", [])):
                problems.append(f"{arm}/{condition}/{ep['task_id']}: n_calls "
                                f"{n_calls} != {len(trace.get('events', []))} events")
    if problems:
        return _g("BUG", "; ".join(problems[:4]), disagreements=len(problems))
    return _g("OK", "independent recomputation agrees with every recorded score")


def veto_s18_test_blindness(results_dir: str | pathlib.Path) -> dict:
    import hashlib

    d = pathlib.Path(results_dir)
    locks_p, reveal_p = d / "locks.json", d / "seed_reveal.json"
    if not locks_p.exists() or not reveal_p.exists():
        return _g("INCONCLUSIVE", "locks.json / seed_reveal.json not present yet")
    locks = json.loads(locks_p.read_text())
    reveal = json.loads(reveal_p.read_text())
    for key in ("checkpoint", "prompt_winner"):
        if key not in locks or "locked_at" not in locks.get(key, {}):
            return _g("BUG", f"locks.json missing a timestamped {key} lock")
    if "revealed_at" not in reveal or "preregistration_commit" not in reveal:
        return _g("BUG", "seed_reveal.json missing revealed_at/preregistration_commit")
    latest_lock = max(locks[k]["locked_at"] for k in ("checkpoint", "prompt_winner"))
    if str(reveal["revealed_at"]) <= str(latest_lock):
        return _g("BUG", f"seed revealed at {reveal['revealed_at']} before the last "
                  f"lock at {latest_lock}")
    want = int.from_bytes(hashlib.sha256(
        (str(reveal["preregistration_commit"]) + ":agentic-heldout-v1").encode()
    ).digest()[:8], "big")
    if int(reveal.get("heldout_seed", -1)) != want:
        return _g("BUG", "revealed held-out seed does not match the committed derivation")
    return _g("OK", "winners locked before the held-out seed reveal; derivation verified")


# ---------------------------------------------------------------------------
# paired data assembly + gates
# ---------------------------------------------------------------------------

def _pairs(eps: dict, condition: str, arms=("BP", "TP")) -> dict:
    a = eps.get((arms[0], condition, "none"), {})
    b = eps.get((arms[1], condition, "none"), {})
    return {tid: (a[tid], b[tid]) for tid in set(a) & set(b)}


def _recovery_ok(ep: dict) -> bool:
    return bool(ep.get("rec", {}).get("certified_recovery"))


def _bootstrap_gate(pairs: list[tuple], outcome_fn, *, label: str, seed: int,
                    margin: float, replicates: int = 100_000) -> dict:
    from agentlab.suite.stats import cluster_bootstrap_lb, mcnemar_exact

    diffs, clusters, b_cnt, c_cnt = [], [], 0, 0
    for bp, tp in pairs:
        yb, yt = int(outcome_fn(bp)), int(outcome_fn(tp))
        diffs.append(yt - yb)
        clusters.append(str(bp.get("template") or bp["task_id"]))
        if yb and not yt:
            b_cnt += 1
        if yt and not yb:
            c_cnt += 1
    boot = cluster_bootstrap_lb(diffs, clusters, seed, label, replicates=replicates)
    mc = mcnemar_exact(b_cnt, c_cnt)
    return {"point": boot["point"], "lb": boot["lb"], "margin": margin,
            "n_pairs": len(diffs), "n_clusters": boot["n_clusters"],
            "degenerate": boot["degenerate"], "mcnemar": mc,
            "pass": (not boot["degenerate"]) and boot["lb"] > margin}


def evaluate_agentic_gates(eps: dict, prereg: dict) -> dict:
    from agentlab.suite.stats import wilson as _wilson

    seed = int(prereg["statistics"]["clustered_bootstrap"]["seed"])
    replicates = int(prereg["statistics"]["clustered_bootstrap"]["replicates"])
    gates: dict = {}

    clean = _pairs(eps, "clean")
    faulted = _pairs(eps, "faulted")

    # ---- ER: primary claim ------------------------------------------------
    common_clean = {tid for tid, (bp, tp) in clean.items()
                    if bp["rep"]["certified_success"] and tp["rep"]["certified_success"]}
    c_set = sorted(common_clean & set(faulted))
    min_c = 500
    if len(c_set) < min_c:
        gates["ER1"] = _g("INCONCLUSIVE",
                          f"|C|={len(c_set)} < {min_c}: underpowered; the recovery claim "
                          f"receives NO verdict regardless of point estimates", C=len(c_set))
    else:
        gates["ER1"] = _g("PASS", f"|C|={len(c_set)} >= {min_c}", C=len(c_set))

    if c_set:
        er2 = _bootstrap_gate([faulted[t] for t in c_set], _recovery_ok,
                              label="ER2", seed=seed, margin=0.05, replicates=replicates)
        gates["ER2"] = _g("PASS" if er2["pass"] else "FAIL",
                          f"certified recovery diff on C: point {er2['point']:+.3f}, "
                          f"97.5% clustered LB {er2['lb']:+.3f} vs margin +0.05; exact "
                          f"McNemar b={er2['mcnemar']['b']} c={er2['mcnemar']['c']} "
                          f"p={er2['mcnemar']['p_two_sided']:.4g}", **{k: v for k, v in
                          er2.items() if k != "mcnemar"}, mcnemar=er2["mcnemar"])
        k = sum(1 for t in c_set if _recovery_ok(faulted[t][1]))
        p, lo, hi = _wilson(k, len(c_set))
        gates["ER3"] = _g("PASS" if lo >= 0.60 else "FAIL",
                          f"TP certified recovery on C {p:.3f} [{lo:.3f},{hi:.3f}], "
                          f"Wilson LB vs 0.60", k=k, n=len(c_set), lb=lo)
    else:
        gates["ER2"] = _g("INCONCLUSIVE", "no common-clean pairs with faulted replays")
        gates["ER3"] = _g("INCONCLUSIVE", "no common-clean pairs with faulted replays")
    if gates["ER1"]["status"] == "INCONCLUSIVE":
        for name in ("ER2", "ER3"):
            if gates[name]["status"] != "INCONCLUSIVE":
                gates[name] = _g("INCONCLUSIVE",
                                 f"|C| below the preregistered floor; measured anyway for "
                                 f"the record: {gates[name]['detail']}",
                                 **gates[name]["numbers"])

    if clean:
        er4 = _bootstrap_gate(list(clean.values()),
                              lambda ep: ep["rep"]["certified_success"],
                              label="ER4", seed=seed, margin=-0.03, replicates=replicates)
        gates["ER4"] = _g("PASS" if er4["pass"] else "FAIL",
                          f"clean non-inferiority: diff {er4['point']:+.3f}, 97.5% "
                          f"clustered LB {er4['lb']:+.3f} vs margin -0.03",
                          **{k: v for k, v in er4.items() if k != "mcnemar"})
    else:
        gates["ER4"] = _g("INCONCLUSIVE", "no paired clean episodes")

    min_assigned = 900
    if faulted:
        itt_tp = sum(_recovery_ok(tp) for _, tp in faulted.values()) / len(faulted)
        itt_bp = sum(_recovery_ok(bp) for bp, _ in faulted.values()) / len(faulted)
        diff = itt_tp - itt_bp
        if len(faulted) < min_assigned:
            gates["ER5"] = _g("INCONCLUSIVE",
                              f"only {len(faulted)} assigned fault pairs (< {min_assigned}); "
                              f"ITT diff {diff:+.3f} reported, not gated",
                              n=len(faulted), diff=diff)
        else:
            gates["ER5"] = _g("PASS" if diff >= 0 else "FAIL",
                              f"intention-to-treat certified recovery diff {diff:+.3f} "
                              f"over {len(faulted)} assigned pairs (must be >= 0)",
                              n=len(faulted), diff=diff, tp=itt_tp, bp=itt_bp)
    else:
        gates["ER5"] = _g("INCONCLUSIVE", "no assigned fault pairs")

    tp_core = [ep for (arm, cond, ctl), tasks in eps.items() if arm == "TP"
               and ctl == "none" and cond in ("clean", "faulted")
               for ep in tasks.values()]
    if tp_core:
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_core)
        p, lo, hi = _wilson(k_run, len(tp_core))
        gates["ER6"] = _g("PASS" if hi <= 0.03 else "FAIL",
                          f"TP runaway {p:.4f} [{lo:.4f},{hi:.4f}], Wilson UB vs 0.03",
                          k=k_run, n=len(tp_core), ub=hi)
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_core)
        p, lo, hi = _wilson(k_h, len(tp_core))
        gates["ER7"] = _g("PASS" if hi <= 0.01 else "FAIL",
                          f"TP hallucinated-result {p:.4f} [{lo:.4f},{hi:.4f}], Wilson UB "
                          f"vs 0.01", k=k_h, n=len(tp_core), ub=hi)
    else:
        gates["ER6"] = _g("INCONCLUSIVE", "no TP episodes")
        gates["ER7"] = _g("INCONCLUSIVE", "no TP episodes")

    from agentlab.suite.faults import group_of

    groups: dict = {}
    for bp, tp in faulted.values():
        fault = bp["trace"].get("fault") or {}
        cls = fault.get("class")
        if not cls:
            continue
        g = group_of(cls)
        acc = groups.setdefault(g, [0, 0, 0])
        acc[0] += _recovery_ok(tp)
        acc[1] += _recovery_ok(bp)
        acc[2] += 1
    if groups:
        worst = {g: (a[0] - a[1]) / a[2] for g, a in groups.items()}
        bad = {g: d for g, d in worst.items() if d < -0.05}
        gates["ER8"] = _g("FAIL" if bad else "PASS",
                          f"per-group ITT recovery diffs: "
                          + ", ".join(f"{g} {d:+.3f} (n={groups[g][2]})"
                                      for g, d in sorted(worst.items()))
                          + ("; below -0.05: " + ",".join(bad) if bad else ""),
                          diffs=worst)
    else:
        gates["ER8"] = _g("INCONCLUSIVE", "no fault-class metadata on faulted pairs")

    # ---- MT: secondary (a) --------------------------------------------------
    mt_pairs = {tid: pair for tid, pair in clean.items()
                if pair[0]["all_tools_required"] and pair[0]["horizon"] == 4}
    min_mt = 600
    if len(mt_pairs) < min_mt:
        gates["MT1"] = _g("INCONCLUSIVE", f"only {len(mt_pairs)} all-tools H4 pairs "
                          f"(< {min_mt})", n=len(mt_pairs))
    else:
        mt1 = _bootstrap_gate(list(mt_pairs.values()),
                              lambda ep: bool(ep.get("orch", {}).get(
                                  "certified_orchestration")),
                              label="MT1", seed=seed, margin=0.05, replicates=replicates)
        gates["MT1"] = _g("PASS" if mt1["pass"] else "FAIL",
                          f"certified all-tools diff: point {mt1['point']:+.3f}, 97.5% "
                          f"clustered LB {mt1['lb']:+.3f} vs +0.05; exact McNemar "
                          f"p={mt1['mcnemar']['p_two_sided']:.4g}",
                          **{k: v for k, v in mt1.items() if k != "mcnemar"},
                          mcnemar=mt1["mcnemar"])
    if mt_pairs:
        tp_eps = [tp for _, tp in mt_pairs.values()]
        k = sum(bool(ep.get("orch", {}).get("certified_orchestration")) for ep in tp_eps)
        p, lo, hi = _wilson(k, len(tp_eps))
        gates["MT2"] = _g("PASS" if lo >= 0.60 else "FAIL",
                          f"TP certified all-tools {p:.3f} [{lo:.3f},{hi:.3f}] vs 0.60",
                          k=k, n=len(tp_eps), lb=lo)
        calls = sorted(ep["rep"]["n_calls"] for ep in tp_eps)
        med = calls[len(calls) // 2]
        gates["MT3"] = _g("PASS" if med <= 4 + 2 else "FAIL",
                          f"TP median calls {med} vs oracle 4 + 2", median=med)
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_eps)
        _, _, ub_run = _wilson(k_run, len(tp_eps))
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_eps)
        _, _, ub_h = _wilson(k_h, len(tp_eps))
        gates["MT4"] = _g("PASS" if ub_run <= 0.03 and ub_h <= 0.01 else "FAIL",
                          f"TP MT runaway UB {ub_run:.4f} (<=0.03), hallucination UB "
                          f"{ub_h:.4f} (<=0.01)", runaway_ub=ub_run, halluc_ub=ub_h)
        pat: dict = {}
        for bp, tp in mt_pairs.values():
            pid = bp.get("pattern_id")
            if pid is None:
                continue
            acc = pat.setdefault(pid, [0, 0, 0])
            acc[0] += bool(tp.get("orch", {}).get("certified_orchestration"))
            acc[1] += bool(bp.get("orch", {}).get("certified_orchestration"))
            acc[2] += 1
        if len(pat) >= 6 and all(a[2] >= 80 for a in pat.values()):
            diffs = {p_: (a[0] - a[1]) / a[2] for p_, a in pat.items()}
            bad = {p_: d for p_, d in diffs.items() if d < -0.05}
            gates["MT5"] = _g("FAIL" if bad else "PASS",
                              "per-pattern diffs: " + ", ".join(
                                  f"p{p_} {d:+.3f}" for p_, d in sorted(diffs.items()))
                              + ("; below -0.05: " + ",".join(str(x) for x in bad)
                                 if bad else ""), diffs=diffs)
        else:
            gates["MT5"] = _g("INCONCLUSIVE",
                              f"order patterns incomplete: {len(pat)} patterns, "
                              f"min cell {min((a[2] for a in pat.values()), default=0)} "
                              f"(need 6 patterns x >=80 pairs)")
    else:
        for name in ("MT2", "MT3", "MT4", "MT5"):
            gates[name] = _g("INCONCLUSIVE", "no all-tools H4 pairs")
    # MT6 is stitched in from the S11 veto by the caller.

    # ---- HR: secondary (b) --------------------------------------------------
    hr_pairs = {tid: pair for tid, pair in clean.items()
                if pair[0]["horizon"] == 8
                and pair[0]["family"] in ("lookup_chain", "typed_relay")}
    min_hr, min_disc = 400, 20
    if len(hr_pairs) < min_hr:
        gates["HR1"] = _g("INCONCLUSIVE", f"only {len(hr_pairs)} H8 clean pairs "
                          f"(< {min_hr})", n=len(hr_pairs))
    else:
        hr1 = _bootstrap_gate(list(hr_pairs.values()),
                              lambda ep: ep["rep"]["certified_success"],
                              label="HR1", seed=seed, margin=0.05, replicates=replicates)
        if hr1["mcnemar"]["n_discordant"] < min_disc:
            gates["HR1"] = _g("INCONCLUSIVE",
                              f"only {hr1['mcnemar']['n_discordant']} discordant H8 pairs "
                              f"(< {min_disc}); diff {hr1['point']:+.3f} reported, not "
                              f"gated", **{k: v for k, v in hr1.items() if k != "mcnemar"})
        else:
            gates["HR1"] = _g("PASS" if hr1["pass"] else "FAIL",
                              f"H8 certified success diff: point {hr1['point']:+.3f}, "
                              f"97.5% clustered LB {hr1['lb']:+.3f} vs +0.05",
                              **{k: v for k, v in hr1.items() if k != "mcnemar"},
                              mcnemar=hr1["mcnemar"])
    if hr_pairs:
        tp_eps = [tp for _, tp in hr_pairs.values()]
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_eps)
        _, _, ub = _wilson(k_run, len(tp_eps))
        gates["HR2"] = _g("PASS" if ub <= 0.03 else "FAIL",
                          f"TP H8 runaway Wilson UB {ub:.4f} vs 0.03", ub=ub)
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_eps)
        _, _, ub = _wilson(k_h, len(tp_eps))
        gates["HR3"] = _g("PASS" if ub <= 0.01 else "FAIL",
                          f"TP H8 hallucinated-result Wilson UB {ub:.4f} vs 0.01", ub=ub)
    else:
        gates["HR2"] = _g("INCONCLUSIVE", "no H8 clean pairs")
        gates["HR3"] = _g("INCONCLUSIVE", "no H8 clean pairs")

    return gates


def evaluate_floors(eps: dict, arm: str) -> dict:
    """Absolute launch floors F1-F5 for one arm."""
    from agentlab.suite.stats import wilson as _wilson  # noqa: F401

    clean = list(eps.get((arm, "clean", "none"), {}).values())
    faulted = list(eps.get((arm, "faulted", "none"), {}).values())
    floors: dict = {}
    if not clean or not faulted:
        for f in ("F1", "F2", "F3", "F4", "F5"):
            floors[f] = _g("INCONCLUSIVE", f"{arm}: missing clean or faulted traces")
        return floors

    def rate(eps_list, fn):
        return sum(map(fn, eps_list)) / len(eps_list)

    def by_family(eps_list, fn):
        fam: dict = {}
        for ep in eps_list:
            fam.setdefault(ep["family"], []).append(fn(ep))
        return {f: sum(v) / len(v) for f, v in fam.items()}

    ok = rate(clean, lambda e: e["rep"]["certified_success"])
    floors["F1"] = _g("PASS" if ok >= 0.65 else "FAIL",
                      f"{arm} overall clean certified success {ok:.3f} vs 0.65", rate=ok)
    fam = by_family(clean, lambda e: e["rep"]["certified_success"])
    bad = {f: r for f, r in fam.items() if r < 0.50}
    floors["F2"] = _g("FAIL" if bad else "PASS",
                      f"{arm} clean by family: "
                      + ", ".join(f"{f} {r:.3f}" for f, r in sorted(fam.items()))
                      + (f"; below 0.50: {sorted(bad)}" if bad else ""), by_family=fam)
    okf = rate(faulted, lambda e: e["rep"]["certified_success"])
    floors["F3"] = _g("PASS" if okf >= 0.40 else "FAIL",
                      f"{arm} overall faulted strict success {okf:.3f} vs 0.40 "
                      f"(intention-to-treat)", rate=okf)
    famf = by_family(faulted, lambda e: e["rep"]["certified_success"])
    badf = {f: r for f, r in famf.items() if r < 0.25}
    floors["F4"] = _g("FAIL" if badf else "PASS",
                      f"{arm} faulted by family: "
                      + ", ".join(f"{f} {r:.3f}" for f, r in sorted(famf.items()))
                      + (f"; below 0.25: {sorted(badf)}" if badf else ""), by_family=famf)
    allx = clean + faulted
    crash = rate(allx, _crashish)
    floors["F5"] = _g("PASS" if crash < 0.02 else "FAIL",
                      f"{arm} loop/crash rate {crash:.4f} vs < 0.02", rate=crash)
    return floors


def horizon_curves(eps: dict, arms=("BP", "TP")) -> dict:
    """Descriptive success-vs-horizon curves with Wilson bands. No fitting;
    H50 is reported only when the observed curve crosses 50%."""
    from agentlab.suite.stats import wilson as _wilson

    curves: dict = {}
    for arm in arms:
        for condition in ("clean", "faulted", "stress"):
            tasks = eps.get((arm, condition, "none"), {})
            cells: dict = {}
            for ep in tasks.values():
                cells.setdefault((ep["family"], ep["horizon"]), []).append(
                    ep["rep"]["certified_success"])
            for (family, h), vals in sorted(cells.items()):
                k = sum(vals)
                p, lo, hi = _wilson(k, len(vals))
                curves.setdefault(f"{arm}/{condition}/{family}", []).append(
                    {"horizon": h, "k": k, "n": len(vals), "p": round(p, 4),
                     "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
    for key, pts in curves.items():
        pts.sort(key=lambda r: r["horizon"])
        above = [r for r in pts if r["p"] >= 0.5]
        below = [r for r in pts if r["p"] < 0.5]
        if not below:
            h50 = f"right-censored (> H{pts[-1]['horizon']})"
        elif not above:
            h50 = f"left-censored (< H{pts[0]['horizon']})"
        else:
            lo_h = max(r["horizon"] for r in above)
            hi_h = min(r["horizon"] for r in below if r["horizon"] > lo_h) \
                if any(r["horizon"] > lo_h for r in below) else lo_h
            h50 = f"crossed in [H{lo_h}, H{hi_h}]"
        curves[key] = {"points": pts, "H50": h50}
    return curves


def agentic_verdict(traces_dir: str, preregister: str, secret_path: str,
                    specs_path: str | None = None,
                    split_manifests: dict[str, str] | None = None,
                    results_dir: str | None = None) -> dict:
    """The machine verdict: vetoes, gates, floors, winner. Pure given inputs."""
    prereg = load_preregister(preregister)
    secret = bytes.fromhex(pathlib.Path(secret_path).read_text().strip())
    eps = load_agentic_episodes(traces_dir, secret)

    specs = None
    specs_by_id = None
    if specs_path and pathlib.Path(specs_path).exists():
        specs = _load_jsonl(pathlib.Path(specs_path))
        specs_by_id = {s["task_id"]: s for s in specs}
    splits = None
    if split_manifests:
        splits = {name: _load_jsonl(pathlib.Path(p))
                  for name, p in split_manifests.items() if pathlib.Path(p).exists()}
        if len(splits) < 2:
            splits = None
    locks = None
    rdir = results_dir or str(pathlib.Path(traces_dir).parent)
    locks_p = pathlib.Path(rdir) / "locks.json"
    if locks_p.exists():
        locks = json.loads(locks_p.read_text())

    vetoes = {
        "S8": veto_s8_pairing(eps),
        "S9": veto_s9_oracle(specs),
        "S10": veto_s10_splits(splits),
        "S11": veto_s11_absent_info(eps, prereg),
        "S12": veto_s12_injection(eps, specs_by_id),
        "S13": veto_s13_receipts(eps),
        "S14": veto_s14_counterfactual(eps, specs, prereg),
        "S15": veto_s15_attrition(eps, specs_by_id),
        "S16": veto_s16_control_integrity(eps, prereg, locks),
        "S17": veto_s17_trace_summary(eps),
        "S18": veto_s18_test_blindness(rdir),
    }
    any_bug = any(v["status"] == "BUG" for v in vetoes.values())

    gates = evaluate_agentic_gates(eps, prereg)
    # MT6 mirrors the absent-information veto restricted to the MT family.
    s11 = vetoes["S11"]
    gates["MT6"] = {"status": {"OK": "PASS"}.get(s11["status"], s11["status"]),
                    "detail": f"absent-information control (S11): {s11['detail']}",
                    "numbers": s11["numbers"]}

    def claim_status(names: list[str]) -> str:
        st = [gates[n]["status"] for n in names]
        if any_bug:
            return "BUG"
        if any(s == "INCONCLUSIVE" for s in st):
            return "INCONCLUSIVE"
        return "PASS" if all(s == "PASS" for s in st) else "FAIL"

    claims = {
        "primary_certified_error_recovery": claim_status(
            ["ER1", "ER2", "ER3", "ER4", "ER5", "ER6", "ER7", "ER8"]),
        "secondary_all_tools_orchestration_H4": claim_status(
            ["MT1", "MT2", "MT3", "MT4", "MT5", "MT6"]),
        "secondary_H8_execution_reliability": claim_status(["HR1", "HR2", "HR3"]),
    }

    floors_tp = evaluate_floors(eps, "TP")
    floors_bp = evaluate_floors(eps, "BP")

    def floors_pass(fl: dict) -> bool:
        return all(v["status"] == "PASS" for v in fl.values())

    if any_bug:
        winner = "NO VERDICT: harness BUG vetoes every claim and the winner rule"
    elif (floors_pass(floors_tp) and gates["ER4"]["status"] == "PASS"
          and gates["ER2"]["status"] == "PASS"):
        winner = ("TP (trained arm ships: floors clear, clean-non-inferior at -0.03, "
                  "certified recovery LB > +0.05)")
    elif floors_pass(floors_bp):
        winner = "BP (frozen prompted base ships; the training leg is dropped and reported)"
    elif any(v["status"] == "INCONCLUSIVE" for v in {**floors_tp, **floors_bp}.values()):
        winner = "NO VERDICT: floor evidence incomplete (INCONCLUSIVE)"
    else:
        winner = "none: no successful multifaceted pipeline yet"

    return {"vetoes": vetoes, "any_bug": any_bug, "gates": gates, "claims": claims,
            "floors": {"TP": floors_tp, "BP": floors_bp}, "winner": winner,
            "curves": horizon_curves(eps),
            "claims_to_reject": prereg["claims_to_reject"]}


def render_agentic_verdict(v: dict) -> str:
    lines = ["# Agentic machine verdict", ""]
    lines.append("## Harness vetoes S8-S18 (any BUG vetoes everything below)")
    for name, res in v["vetoes"].items():
        lines.append(f"  {res['status']:<13} {name}: {res['detail']}")
    lines.append("")
    lines.append("## Gates")
    for name, res in v["gates"].items():
        lines.append(f"  {res['status']:<13} {name}: {res['detail']}")
    lines.append("")
    lines.append("## Launch floors")
    for arm, floors in v["floors"].items():
        for name, res in floors.items():
            lines.append(f"  {res['status']:<13} {arm} {name}: {res['detail']}")
    lines.append("")
    lines.append("## Preregistered claims")
    for claim, status in v["claims"].items():
        lines.append(f"  {status:<13} {claim}")
    lines.append("")
    lines.append("## Descriptive horizon curves (Wilson 95% bands; no extrapolation)")
    for key, cur in v["curves"].items():
        pts = " ".join(f"H{r['horizon']}:{r['p']:.2f}[{r['wilson_lo']:.2f},"
                       f"{r['wilson_hi']:.2f}]" for r in cur["points"])
        lines.append(f"  {key}: {pts}  H50 {cur['H50']}")
    lines.append("")
    lines.append(f"## Winner: {v['winner']}")
    if v["any_bug"]:
        lines.append("HARNESS BUG DETECTED -- nothing above is a statement about the "
                     "model. Fix the harness, then re-run.")
    lines.append("")
    lines.append("## Claims this run can NEVER support (preregistered rejections)")
    for c in v["claims_to_reject"]:
        lines.append(f"  - {c}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "sft", "rssft", "rsgrpo"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--trace-dirs", nargs="+", default=None)
    ap.add_argument("--save", default=None, help="also write the report to this path")
    ap.add_argument("--agentic", action="store_true",
                    help="run the multifaceted agentic verdict instead of the legacy one")
    ap.add_argument("--traces", default="results/agentic/traces")
    ap.add_argument("--preregister", default="configs/agentic_preregister.json")
    ap.add_argument("--secret", default="out/agentic/run_secret.hex")
    ap.add_argument("--specs", default=None, help="held-out eval spec manifest (S9/S12/S15)")
    ap.add_argument("--split-manifest", action="append", default=None,
                    metavar="NAME=PATH", help="e.g. --split-manifest train=data/...jsonl")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    if args.agentic:
        splits = None
        if args.split_manifest:
            splits = dict(kv.split("=", 1) for kv in args.split_manifest)
        v = agentic_verdict(args.traces, args.preregister, args.secret,
                            specs_path=args.specs, split_manifests=splits,
                            results_dir=args.results_dir)
        text = render_agentic_verdict(v)
        print(text)
        if args.save:
            pathlib.Path(args.save).write_text(text + "\n", encoding="utf-8")
        if args.save_json:
            pathlib.Path(args.save_json).write_text(
                json.dumps(v, indent=2, default=str) + "\n", encoding="utf-8")
        return

    text = report(args.tags, args.out_dir, args.trace_dirs)
    print(text)
    if args.save:
        pathlib.Path(args.save).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
