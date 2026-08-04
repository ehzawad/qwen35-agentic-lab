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

Statistics are deliberately plain: Wilson intervals for a single proportion,
pooled two-proportion z-test between checkpoints, and the minimum detectable
difference at 80% power so "no difference found" is always qualified by what
could have been found. Stdlib only -- no scipy on the box.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

Z95 = 1.959964
Z80 = 0.841621  # power term for MDE


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(point, lo, hi) Wilson score interval; safe at k=0 and k=n."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def two_prop_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(z, two_sided_p) pooled two-proportion z-test."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def mde(n1: int, n2: int, p_base: float) -> float:
    """Approximate minimum detectable difference, alpha=.05 two-sided, 80% power."""
    if n1 == 0 or n2 == 0:
        return 1.0
    return (Z95 + Z80) * math.sqrt(2 * p_base * (1 - p_base) * (1 / n1 + 1 / n2) / 2)


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

    n = summary["n"]
    row = {"tag": tag, "summary": summary, "n": n, "episodes": len(episodes)}
    row["acc_k"] = round(summary["accuracy"] * n)
    if episodes:
        calls = [e.get("n_calls", 0) for e in episodes]
        row["calls_mean"] = sum(calls) / len(calls)
        row["calls_max"] = max(calls)
        row["runaway_k"] = sum(1 for c in calls if c > 10)
        row["nobox_k"] = sum(
            1 for e in episodes if not e.get("ok") and "boxed" not in str(e.get("final", ""))
        )
        row["trace_n"] = len(episodes)
        row["_episodes"] = episodes
    return row


def sanity_checks(row: dict) -> list[tuple[str, str, str]]:
    """Distinguish 'the model is weak' from 'the harness is broken'.

    A weak model is a legitimate result; a broken harness masquerading as one is
    not, and this session produced exactly that: GRPO logged accuracy 0.000 for
    16 straight steps and the causes were two bugs, not the policy. Every check
    here is derived from a bug that actually happened, and each fires on the bug
    signature while staying silent on genuinely weak models (the broken-SFT
    checkpoint scores 0.050 with working parsing -- these checks pass on it).

    Returns (level, code, message); level is "BUG" (fix the harness before
    reading any table) or "WARN" (investigate the traces before concluding).
    """
    out = []
    eps = row.get("_episodes") or []
    n, summary = row["n"], row["summary"]

    # S1: the trace must cover the run. A shortfall means episodes were dropped
    # and every trace-derived metric is computed on a biased subset.
    if eps and abs(len(eps) - n) > 1:
        out.append(("BUG", "S1", f"trace has {len(eps)} episodes but eval ran n={n}"))

    if eps:
        # S2: recompute accuracy from the traces; disagreement with the summary
        # means the two scoring paths have drifted apart (predicted/expected vs
        # ok flag -- the kind of drift that mislabelled rewards earlier).
        ok_k = sum(1 for e in eps if e.get("ok"))
        if abs(ok_k / len(eps) - summary["accuracy"]) > 0.02:
            out.append(("BUG", "S2",
                        f"trace accuracy {ok_k/len(eps):.3f} != summary {summary['accuracy']:.3f}"))

        # S3: exactly-zero accuracy with boxed answers PRESENT in finals is the
        # scorer-blind signature (a box the scorer cannot see). Zero accuracy
        # with no boxes anywhere is consistent with a genuinely broken policy.
        if summary["accuracy"] == 0 and len(eps) >= 10:
            boxed = sum(1 for e in eps if "boxed" in str(e.get("final", "")))
            if boxed:
                out.append(("BUG", "S3",
                            f"accuracy is exactly 0 but {boxed}/{len(eps)} finals contain a box "
                            f"-- scorer-blind signature, verify the scoring path by hand"))
            else:
                out.append(("WARN", "S3",
                            "accuracy exactly 0 and no boxes at all; plausible policy failure "
                            "but rule out a decode/template bug before concluding"))

        # S4: zero tool use when tools were offered is the parser-blind
        # signature (the XML-vs-JSON bug produced exactly this reading).
        if summary.get("tool_use_rate", 1) == 0 and len(eps) >= 10:
            out.append(("BUG", "S4",
                        "tool_use_rate is 0 with tools offered -- parser-blind signature "
                        "(a model that CAN answer directly still calls tools sometimes)"))

        # S5: internal inconsistencies no model behaviour can produce.
        for e in eps:
            if e.get("ok") and not str(e.get("final", "")).strip():
                out.append(("BUG", "S5", f"episode {e.get('index')} ok=True with empty final"))
                break
            r = e.get("rewards") or {}
            if e.get("ok") and r.get("predicted") is not None and r.get("expected") is not None:
                if abs(float(r["predicted"]) - float(r["expected"])) > 1e-4:
                    out.append(("BUG", "S5",
                                f"episode {e.get('index')} ok=True but predicted != expected"))
                    break

        # S6: mass-duplicate finals are a generation/indexing bug signature
        # (zip misalignment or a stuck sampler), not a property of weak models.
        finals = [str(e.get("final", ""))[:200] for e in eps if str(e.get("final", "")).strip()]
        if len(finals) >= 10:
            from collections import Counter
            top = Counter(finals).most_common(1)[0][1]
            if top / len(finals) > 0.5:
                out.append(("BUG", "S6",
                            f"{top}/{len(finals)} finals are identical -- generation or "
                            f"indexing bug signature"))

        # S7: the summary and the trace must agree on tool use.
        used = sum(1 for e in eps if e.get("n_calls", 0) > 0)
        if abs(used / len(eps) - summary.get("tool_use_rate", 0)) > 0.05:
            out.append(("BUG", "S7",
                        f"trace tool use {used/len(eps):.2f} != summary "
                        f"{summary.get('tool_use_rate'):.2f}"))
    return out


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
    hdr = f"{'ckpt':<8}{'accuracy (95% CI)':>26}{'calls/ep':>10}{'runaway':>10}{'no-box':>9}{'n':>6}"
    lines += [hdr, "-" * len(hdr)]
    for t, r in rows.items():
        acc = fmt_ci(r["acc_k"], r["n"])
        calls = f"{r.get('calls_mean', float('nan')):.1f}" if "calls_mean" in r else "-"
        run = (f"{r['runaway_k']}/{r['trace_n']}" if "runaway_k" in r else "-")
        nob = (f"{r['nobox_k']}/{r['trace_n']}" if "nobox_k" in r else "-")
        lines.append(f"{t:<8}{acc:>26}{calls:>10}{run:>10}{nob:>9}{r['n']:>6}")
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

    # pairwise significance, honestly qualified
    if base and rssft:
        z, p = two_prop_test(rssft["acc_k"], rssft["n"], base["acc_k"], base["n"])
        d = rssft["acc_k"] / rssft["n"] - base["acc_k"] / base["n"]
        m = mde(base["n"], rssft["n"], base["acc_k"] / base["n"])
        lines.append(f"rssft - base : {d:+.3f}  (z={z:+.2f}, p={p:.3f}; MDE at 80% power ~ {m:.3f})")
        if abs(d) < m and p >= 0.05:
            lines.append(f"  note: differences smaller than ~{m:.1%} are not resolvable at these n;")
            lines.append("  'no significant difference' here does NOT mean 'equal'.")
        if base["n"] != rssft["n"]:
            lines.append(f"  note: unequal n ({base['n']} vs {rssft['n']}) -- base@{rssft['n']} re-run is queued;")
            lines.append("  the base slice is a seeded prefix of the larger eval, so episodes overlap by design.")
    if rssft and rsgrpo:
        z, p = two_prop_test(rsgrpo["acc_k"], rsgrpo["n"], rssft["acc_k"], rssft["n"])
        d = rsgrpo["acc_k"] / rsgrpo["n"] - rssft["acc_k"] / rssft["n"]
        lines.append(f"rsgrpo - rssft: {d:+.3f}  (z={z:+.2f}, p={p:.3f})")
    lines.append("")

    # pre-registered gates
    lines.append("## Gates (registered before launch)")
    passed = failed = 0
    if rssft:
        acc = rssft["acc_k"] / rssft["n"]
        ok1 = gate("G1 accuracy >= 0.800", acc >= 0.800, f"rssft {acc:.3f}", lines)
        if "calls_mean" in rssft:
            ok2 = gate("G2 calls/ep <= 6.0", rssft["calls_mean"] <= 6.0,
                       f"rssft {rssft['calls_mean']:.1f} (base 3.3, broken 50.0)", lines)
            ok3 = gate("G3 runaway <= 10%", rssft["runaway_k"] / rssft["trace_n"] <= 0.10,
                       f"{rssft['runaway_k']}/{rssft['trace_n']}", lines)
        else:
            ok2 = ok3 = False
            lines.append("  SKIP  G2/G3: no trace found for rssft")
        if base and "nobox_k" in rssft and "nobox_k" in base:
            b = base["nobox_k"] / base["trace_n"]
            r = rssft["nobox_k"] / rssft["trace_n"]
            ok4 = gate("G4 no-box < base", r < b, f"rssft {r:.1%} vs base {b:.1%}", lines)
        else:
            ok4 = False
            lines.append("  SKIP  G4: missing no-box data")
        for ok in (ok1, ok2, ok3, ok4):
            passed, failed = passed + ok, failed + (not ok)
    else:
        lines.append("  (rssft not evaluated yet)")
    if rssft and rsgrpo:
        ok5 = gate("G5 rsgrpo >= rssft", rsgrpo["acc_k"] / rsgrpo["n"] >= rssft["acc_k"] / rssft["n"],
                   f"{rsgrpo['acc_k']/rsgrpo['n']:.3f} vs {rssft['acc_k']/rssft['n']:.3f} (directional)", lines)
        passed, failed = passed + ok5, failed + (not ok5)
    lines.append("")
    lines.append(f"## Verdict: {passed} passed, {failed} failed")
    if any_bug:
        lines.append("HARNESS BUG DETECTED -- the numbers above are NOT a statement about the")
        lines.append("model. Fix the flagged checks and re-run before drawing any conclusion.")
    elif rssft and passed >= 4:
        lines.append("Single-turn SFT destroyed termination; outcome-filtered multi-turn SFT restored it.")
    elif rssft:
        lines.append("Harness is clean, so this is a real model result: the RS-SFT hypothesis")
        lines.append("is NOT supported at these gates. Read the traces before theorising.")
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
