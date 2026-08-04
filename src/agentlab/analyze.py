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
    lines.append("## Gates (registered before launch)")
    passed = failed = skipped = 0
    if rssft:
        acc = rssft["acc_k"] / rssft["n"]
        ok = gate("G1 accuracy >= 0.800", acc >= 0.800, f"rssft {acc:.3f}", lines)
        passed, failed = passed + ok, failed + (not ok)
        if "calls_mean" in rssft:
            ok = gate("G2 calls/ep <= 6.0", rssft["calls_mean"] <= 6.0,
                      f"rssft {rssft['calls_mean']:.1f} (base 3.3, broken 50.0)", lines)
            passed, failed = passed + ok, failed + (not ok)
            ok = gate("G3 runaway <= 10%", rssft["runaway_k"] / rssft["trace_n"] <= 0.10,
                      f"{rssft['runaway_k']}/{rssft['trace_n']}", lines)
            passed, failed = passed + ok, failed + (not ok)
        else:
            skipped += 2
            lines.append("  SKIP  G2/G3: no trace found for rssft (locate the trace; not a model result)")
        if base and "nobox_k" in rssft and "nobox_k" in base:
            b_r = base["nobox_k"] / base["trace_n"]
            r_r = rssft["nobox_k"] / rssft["trace_n"]
            ok = gate("G4 no-box < base", r_r < b_r, f"rssft {r_r:.1%} vs base {b_r:.1%}", lines)
            passed, failed = passed + ok, failed + (not ok)
        else:
            skipped += 1
            lines.append("  SKIP  G4: missing no-box data")
    else:
        lines.append("  (rssft not evaluated yet)")
    if rssft and rsgrpo:
        ok = gate("G5 rsgrpo >= rssft (directional)",
                  rsgrpo["acc_k"] / rsgrpo["n"] >= rssft["acc_k"] / rssft["n"],
                  f"{rsgrpo['acc_k']/rsgrpo['n']:.3f} vs {rssft['acc_k']/rssft['n']:.3f}", lines)
        passed, failed = passed + ok, failed + (not ok)

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
    elif rssft:
        lines.append("Harness is clean and data complete, so this is a real model result: the")
        lines.append("RS-SFT hypothesis is NOT supported at these gates. Read the traces first.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "sft", "rssft", "rsgrpo"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--trace-dirs", nargs="+", default=None)
    ap.add_argument("--save", default=None, help="also write the report to this path")
    args = ap.parse_args()

    text = report(args.tags, args.out_dir, args.trace_dirs)
    print(text)
    if args.save:
        pathlib.Path(args.save).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
