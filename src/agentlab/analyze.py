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

# Official statuses only. `observed_status` / `measured_status` are explicitly
# NOT verdicts: they carry the raw arithmetic of a gate that has been vetoed by a
# harness BUG or downgraded for underpower, so nothing is deleted, but a reader
# or a script pulling a status can never mistake them for a result.
NON_VERDICT_FIELDS = ("observed_status", "measured_status")


def _g(status: str, detail: str, **numbers) -> dict:
    # SKIP is forbidden in the agentic path: the legacy analyzer above uses it,
    # but a preregistered gate that cannot be computed is INCONCLUSIVE and a gate
    # in an unknown state is a BUG. Translating either into SKIP is how a missing
    # measurement stops being visible.
    assert status in OUTCOME_STATES + ("OK", "WARN"), f"forbidden gate state {status!r}"
    return {"status": status, "detail": detail, "numbers": numbers}


def registered(prereg: dict, *path):
    """Read one numeric/structural field from the governing preregistration.

    Every threshold, margin, floor, sample-size minimum and rate ceiling the
    agentic analyzer applies is fetched through here, out of the `machine` block
    of configs/agentic_preregister.json. Nothing is defaulted: a missing field
    raises rather than falling back to a literal, because a silent default is
    exactly the failure this indirection exists to prevent -- the protocol used
    to declare the JSON authoritative while the code carried its own copies of
    500, 900, +0.05, -0.03 and -0.05, so editing the governing file changed no
    verdict at all.
    """
    node = prereg.get("machine")
    if not isinstance(node, dict):
        raise KeyError("configs/agentic_preregister.json has no `machine` block: "
                       "the analyzer has no thresholds to apply")
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError("the preregistration is missing the registered field "
                           f"machine.{'.'.join(map(str, path))} "
                           f"(stopped at machine.{'.'.join(map(str, path[:i + 1]))})")
        node = node[key]
    return node


def _no_verdict(res: dict, why: str) -> dict:
    """Downgrade a measured gate to INCONCLUSIVE without ERASING what it measured.

    Underpowered evidence must never be read as a PASS -- but it must never be
    used to bury a FAIL either. A downgrade therefore keeps the measured status
    under `measured_status` and repeats the original detail, so a FAIL is always
    still visible in the record and in the rendered report.
    """
    if res["status"] not in ("PASS", "FAIL"):
        return res
    out = _g("INCONCLUSIVE", f"{why}; measured anyway for the record "
             f"[{res['status']}]: {res['detail']}", **res["numbers"])
    out["measured_status"] = res["status"]
    return out


def _n_clusters(eps_list) -> dict:
    """Structural-cluster census for a Wilson (non-resampled) sample.

    Wilson intervals are not clustered, but every interval report still has to
    say how many structural templates its denominator actually spans: an upper
    bound computed over 400 value instantiations of three structural templates is
    a different claim from the same bound over 400 instantiations of 80.
    """
    ids = [ep.get("template_cluster_id") for ep in eps_list]
    known = {str(c) for c in ids if c is not None and str(c) != ""}
    return {"n_clusters": len(known),
            "missing_cluster_id": sum(1 for c in ids
                                      if c is None or str(c) == "")}


def _wilson_ub_gate(name: str, k: int, n: int, threshold: float, detail: str,
                    **numbers) -> dict:
    """A Wilson-UPPER-bound gate, with an explicit unmeasurable-at-this-n state.

    If the interval is so wide that even k=0 cannot clear the threshold, the gate
    is not a statement about the model at all: reporting FAIL would blame the
    policy for a sample size, and reporting PASS is impossible. That is
    INCONCLUSIVE. It can never turn a real FAIL into a PASS -- the k=0 bound is
    the smallest upper bound available at this n, so if it clears the threshold
    the gate is genuinely measurable and the observed k decides.
    """
    from agentlab.suite.stats import wilson as _wilson

    p, lo, hi = _wilson(k, n)
    _, _, best = _wilson(0, n)
    if best > threshold:
        return _g("INCONCLUSIVE",
                  f"{detail}: unmeasurable at n={n} (even k=0 gives UB {best:.4f} "
                  f"> {threshold}); observed {k}/{n}, UB {hi:.4f}",
                  k=k, n=n, ub=hi, best_possible_ub=best, **numbers)
    return _g("PASS" if hi <= threshold else "FAIL",
              f"{detail} {p:.4f} [{lo:.4f},{hi:.4f}], Wilson UB vs {threshold}",
              k=k, n=n, ub=hi, **numbers)


def _wilson_lb_gate(name: str, k: int, n: int, threshold: float, detail: str,
                    **numbers) -> dict:
    """A Wilson-LOWER-bound floor gate, with the same unmeasurable-at-this-n state.

    Mirror of `_wilson_ub_gate`: if even a perfect k=n cannot clear the floor,
    the sample is too small for the gate to be a statement about the model.
    """
    from agentlab.suite.stats import wilson as _wilson

    p, lo, hi = _wilson(k, n)
    _, best, _ = _wilson(n, n)
    if best < threshold:
        return _g("INCONCLUSIVE",
                  f"{detail}: unmeasurable at n={n} (even k=n gives LB {best:.4f} "
                  f"< {threshold}); observed {k}/{n}, LB {lo:.4f}",
                  k=k, n=n, lb=lo, best_possible_lb=best, **numbers)
    return _g("PASS" if lo >= threshold else "FAIL",
              f"{detail} {p:.3f} [{lo:.3f},{hi:.3f}], Wilson LB vs {threshold}",
              k=k, n=n, lb=lo, **numbers)


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
                  "horizon": rec.get("horizon"), "split": rec.get("split"),
                  # `template` is the PARAPHRASE/wording id. It is carried for the
                  # record and is deliberately NOT used for clustering.
                  "template": rec.get("template_id"),
                  # the SOLE bootstrap clustering field (frozen contract): a
                  # structural identity -- family + horizon + oracle-DAG shape +
                  # tool-order pattern + operand roles.
                  "template_cluster_id": rec.get("template_cluster_id"),
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
    """Absent-information control: zero raw AND zero certified success.

    Precedence is fixed and BUG-first, because the two failure modes are not
    comparable: a LEAK (any raw success on a redacted instance) means the answers
    are recoverable without the hidden value, which invalidates every capability
    number in the run, so it can never be reported as mere missing coverage. A
    control episode that never ran is the second BUG: "zero success" over
    unattempted episodes is a vacuous pass on the one control that makes the rest
    of the suite mean anything. Only genuinely thin-but-honest coverage is
    INCONCLUSIVE.
    """
    need = int(prereg["controls"]["absent_information"]["n_per_family"])
    # Coverage is counted PER (family, arm), never pooled across arms: 400
    # redacted lookup_chain episodes all in TP is not "200 per family per arm",
    # and a family that appears in the capability numbers but not in the control
    # has no evidence that its answers require the hidden value at all.
    per_cell: dict = {}
    leaks, unexecuted = [], []
    for (arm, condition, control), tasks in eps.items():
        if control != "redacted" or arm not in arms:
            continue
        for ep in tasks.values():
            runner = ep["trace"].get("runner") or {}
            harness_failed = (runner.get("termination_reason") == "spec_error"
                              or not runner.get("n_decisions"))
            if ep["rep"]["raw_success"] or ep["rep"]["certified_success"]:
                leaks.append(f"{arm}/{ep['task_id']}")
            if harness_failed:
                unexecuted.append(f"{arm}/{ep['task_id']}")
                continue  # a control that never ran contributes no coverage
            per_cell[(ep["family"], arm)] = per_cell.get((ep["family"], arm), 0) + 1
    if leaks:
        return _g("BUG", f"redacted-control SUCCESS (harness leakage): {leaks[:3]}",
                  leaks=len(leaks))
    if unexecuted:
        return _g("BUG", f"{len(unexecuted)} redacted-control episodes never ran a "
                  f"single decision, e.g. {unexecuted[:3]} -- their zero success rate "
                  f"is vacuous, not evidence that the hidden value is required",
                  unexecuted=len(unexecuted))
    if not per_cell:
        return _g("INCONCLUSIVE", "no redacted-control traces found")
    counts = {f"{fam}/{arm}": n for (fam, arm), n in sorted(per_cell.items())}
    scored_families = {ep["family"] for (arm, cond, ctl), tasks in eps.items()
                       if ctl == "none" and arm in arms
                       for ep in tasks.values()}
    short = {k: n for k, n in counts.items() if n < need}
    uncovered = sorted({f"{fam}/{arm}" for fam in scored_families for arm in arms
                        if (fam, arm) not in per_cell})
    if short or uncovered:
        why = []
        if short:
            why.append(f"redacted coverage below {need} per family per arm: {short}")
        if uncovered:
            why.append(f"scored families with NO redacted control: {uncovered}")
        return _g("INCONCLUSIVE", "; ".join(why), counts=counts, need=need)
    return _g("OK", f"zero raw and certified success on {sum(per_cell.values())} "
              f"redacted instances, >= {need} per family per arm", counts=counts,
              need=need)


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
        cap = int(registered(prereg, "controls_checks", "s14_generation_check_cap"))
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
            if checked >= cap:
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
    """Prompt/base/adapter integrity, plus the ONE-locked-checkpoint rule.

    "Exactly one locked trained checkpoint maps to TP" is a harness property, not
    a matter of interpretation: if locks.json names no stage, an unregistered
    stage, or a path that the TP traces' adapter disagrees with, then the arm
    labelled TP is not demonstrably the arm that was locked before the reveal,
    and every trained-arm number is unattributable.
    """
    problems = []
    registered_shas = set(prereg["prompt_candidates"]["sha256"].values())
    winner_sha = (locks or {}).get("prompt_winner", {}).get("sha256")
    base_id = prereg["model"]["base_id"]
    ckpt = (locks or {}).get("checkpoint") or {}
    stages = list(prereg["grpo"]["dev_selection"]["stages"])
    trained_arms = ("T0", "TP", "R0", "RP")
    if locks is not None:
        if not ckpt.get("path"):
            problems.append("locks.json names no trained checkpoint path")
        elif not isinstance(ckpt.get("path"), str):
            problems.append("locks.json checkpoint path is not a single path")
        if ckpt.get("stage") not in stages:
            problems.append(f"locked checkpoint stage {ckpt.get('stage')!r} is not one "
                            f"of the registered {stages}")
    for (arm, condition, control), tasks in eps.items():
        for ep in tasks.values():
            prov = ep["trace"].get("provenance", {})
            psha = ep["trace"].get("prompt", {}).get("sha256")
            if prov.get("base_id") != base_id:
                problems.append(f"{arm}: base_id {prov.get('base_id')!r} != registered")
            if psha not in registered_shas:
                problems.append(f"{arm}: prompt hash not among the eight registered")
            if winner_sha and arm in ("BP", "TP", "RP") and psha != winner_sha:
                problems.append(f"{arm}: prompt is not the locked winner")
            if arm in ("B0", "BP") and prov.get("adapter"):
                problems.append(f"{arm}: adapter loaded in a prompt-only arm")
            if ckpt.get("path") and arm in trained_arms:
                if prov.get("adapter") != ckpt["path"]:
                    problems.append(f"{arm}: adapter {prov.get('adapter')!r} is not the "
                                    f"locked checkpoint {ckpt['path']!r}")
            break  # provenance is constant per file; one episode suffices
    if problems:
        return _g("BUG", "; ".join(sorted(set(problems))[:4]))
    if not eps:
        return _g("INCONCLUSIVE", "no traces loaded")
    return _g("OK", "checkpoints, prompts, and adapters match the preregistration; "
              + (f"exactly one locked checkpoint ({ckpt.get('stage')}) maps to TP"
                 if ckpt.get("path") else "no locks supplied to check against"))


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
# S19 HARDWARE-INTEGRITY
# ---------------------------------------------------------------------------

def _canon(value) -> str:
    """Order-insensitive canonical form of a fingerprint value, for equality."""
    return json.dumps(value, sort_keys=True, default=str)


def _engine_contract_problems(fingerprint, contract: dict, where: str) -> list[str]:
    """Every engine setting a trace DECLARES must equal the registered contract.

    The fingerprint is allowed to be richer than the contract (library versions,
    attention backend, ...), but it may not declare a *registered* setting and
    disagree: an engine running at a different `gpu_memory_utilization`,
    `max_model_len`, or with thinking left on, is a different measurement
    apparatus from the one the run registered.
    """
    if not isinstance(fingerprint, dict):
        return []
    problems = []
    for key, want in contract.items():
        if key == "note" or key.endswith("_note") or key not in fingerprint:
            continue
        got = fingerprint[key]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)) \
                and not isinstance(want, bool) and not isinstance(got, bool):
            same = float(got) == float(want)
        else:
            same = _canon(got) == _canon(want)
        if not same:
            problems.append(f"{where}: engine {key}={got!r} contradicts the registered "
                            f"engine contract {want!r}")
    return problems


def veto_s19_hardware_integrity(eps: dict, prereg: dict) -> dict:
    """Same physical card, same engine, full provenance -- or no winner.

    The protocol's central paired claim is that the *weights* are the only
    difference between BP and TP. That is not verifiable unless every
    claim-bearing trace says which physical GPU and which engine produced it, so
    this veto is mandatory rather than conventional:

      * MISSING provenance     -> INCONCLUSIVE and no winner. Missing evidence is
                                  never a favourable default, and an unlabelled
                                  trace is never *assumed* to be the registered
                                  card.
      * a known-wrong card     -> BUG (name or CUDA-visible byte count that is not
                                  the registered one).
      * mixed UUID / run_id /
        engine fingerprint     -> BUG. One run_id must sit on exactly one physical
                                  GPU UUID, and an independent replication on
                                  another A5000 needs a NEW run_id and its own
                                  trace set: appending to an existing one silently
                                  mixes two cards inside one claim.
      * a paired BP/TP or B0/T0
        pair whose two members
        disagree on the
        fingerprint            -> BUG. Then the weight change was not the only
                                  difference.

    Every direction this check can move a verdict is unfavourable: it can veto or
    withhold, never promote. That is what makes adding it before any GPU-hour
    exists an outcome-blind strengthening rather than a moved goalpost.
    """
    required = list(registered(prereg, "hardware_integrity", "required_trace_fields"))
    want_name = registered(prereg, "hardware_integrity", "expected_gpu_name")
    want_bytes = registered(prereg, "hardware_integrity",
                            "expected_cuda_visible_bytes")
    want_thinking = registered(prereg, "hardware_integrity",
                               "enable_thinking_effective_required")
    one_uuid = bool(registered(prereg, "hardware_integrity", "one_gpu_uuid_per_run_id"))
    one_run = bool(registered(prereg, "hardware_integrity", "one_run_id_per_trace_set"))
    paired_sets = [list(p) for p in registered(prereg, "hardware_integrity",
                                               "paired_arm_sets")]
    paired_fields = list(registered(prereg, "hardware_integrity",
                                    "paired_fingerprint_fields"))
    contract = dict(registered(prereg, "engine_contract"))

    if not eps:
        return _g("INCONCLUSIVE", "no traces loaded; hardware provenance unverified")

    # Three buckets, reported run-level first, then per-trace, then paired, so the
    # most structural defect leads the message: a mixed card, a spliced
    # replication or a wrong card EXPLAINS the paired mismatches downstream of it,
    # and truncating the detail must not hide the cause behind its symptoms.
    run_bugs: list[str] = []
    pair_bugs: list[str] = []
    bugs: list[str] = []
    missing: list[str] = []
    prints: dict = {}
    uuids_by_run: dict = {}
    engines: set = set()
    for (arm, condition, control), tasks in eps.items():
        for tid, ep in tasks.items():
            prov = ep["trace"].get("provenance") or {}
            absent = [f for f in required
                      if prov.get(f) is None or prov.get(f) == ""]
            if absent:
                missing.append(f"{arm}/{condition}/{control}/{tid}: "
                               f"{','.join(absent)}")
                continue
            fp = {f: prov.get(f) for f in required}
            prints[(arm, condition, control, tid)] = fp
            where = f"{arm}/{condition}/{control}/{tid}"
            uuids_by_run.setdefault(str(fp["run_id"]), set()).add(str(fp["gpu_uuid"]))
            engines.add(_canon(fp["engine_fingerprint"]))
            if str(fp["gpu_name"]) != str(want_name):
                bugs.append(f"{where}: gpu_name {fp['gpu_name']!r} is not the "
                            f"registered {want_name!r} (known-wrong card)")
            try:
                same_bytes = int(fp["cuda_visible_bytes"]) == int(want_bytes)
            except (TypeError, ValueError):
                same_bytes = False
            if not same_bytes:
                bugs.append(f"{where}: cuda_visible_bytes "
                            f"{fp['cuda_visible_bytes']!r} is not the registered "
                            f"{want_bytes} (known-wrong card)")
            if bool(fp["enable_thinking_effective"]) != bool(want_thinking):
                bugs.append(f"{where}: enable_thinking_effective "
                            f"{fp['enable_thinking_effective']!r} contradicts the "
                            f"registered engine contract {bool(want_thinking)!r}")
            bugs.extend(_engine_contract_problems(fp["engine_fingerprint"],
                                                  contract, where))

    if one_uuid:
        for run_id, uuids in sorted(uuids_by_run.items()):
            if len(uuids) > 1:
                run_bugs.append(f"run_id {run_id}: {len(uuids)} distinct physical GPU "
                                f"UUIDs in one run ({sorted(uuids)}) -- one run_id "
                                f"must sit on exactly one card")
    if one_run and len(uuids_by_run) > 1:
        run_bugs.append(f"{len(uuids_by_run)} distinct run_ids in one trace set "
                        f"({sorted(uuids_by_run)}) -- a replication needs a NEW "
                        f"run_id and its own trace set, it may not append to this one")
    if len(engines) > 1:
        run_bugs.append(f"{len(engines)} distinct engine fingerprints among the "
                        f"claim-bearing traces; a paired claim needs one engine")

    for pair in paired_sets:
        if len(pair) != 2:
            raise KeyError("machine.hardware_integrity.paired_arm_sets must hold "
                           f"two-arm comparisons; got {pair}")
        a, b = pair
        for (arm, condition, control, tid), fp in sorted(
                prints.items(), key=lambda kv: _canon(kv[0])):
            if arm != a:
                continue
            other = prints.get((b, condition, control, tid))
            if other is None:
                continue
            for field in paired_fields:
                if _canon(fp.get(field)) != _canon(other.get(field)):
                    pair_bugs.append(f"{condition}/{control}/{tid}: {a} and {b} "
                                     f"disagree on {field} ({fp.get(field)!r} vs "
                                     f"{other.get(field)!r}) -- the weight change is "
                                     f"not the only difference")
                    break

    all_bugs = run_bugs + sorted(set(bugs)) + sorted(set(pair_bugs))
    if all_bugs:
        return _g("BUG", "; ".join(all_bugs[:4]),
                  n_bugs=len(all_bugs), n_traces=len(prints))
    if missing:
        return _g("INCONCLUSIVE",
                  f"{len(missing)} claim-bearing traces carry incomplete hardware "
                  f"provenance, e.g. {sorted(missing)[:3]}; the registered "
                  f"fingerprint is {required}",
                  n_missing=len(missing), n_traces=len(prints) + len(missing))
    run_id = next(iter(uuids_by_run))
    uuid = next(iter(uuids_by_run[run_id]))
    return _g("OK", f"all {len(prints)} claim-bearing traces share run_id {run_id}, "
              f"one physical {want_name} ({uuid}) and one engine fingerprint; "
              f"every paired comparison matches",
              n_traces=len(prints), run_id=run_id, gpu_uuid=uuid)


# ---------------------------------------------------------------------------
# GRPO stage disposition (NOT a gate state)
# ---------------------------------------------------------------------------

def grpo_stage_disposition(eps: dict, prereg: dict, locks: dict | None,
                           artifact: dict | None) -> dict:
    """Validate the registered GRPO STAGE DISPOSITION when no GRPO checkpoint exists.

    A stage *disposition* says what happened to a stage. It is never one of the
    capability gate states PASS / FAIL / INCONCLUSIVE / BUG, is never written into
    a gate, claim, floor or winner field, and is deliberately absent from
    `outcome_states`: "the trainer could not be instantiated on the registered
    card" is not a statement about the model.

    The rule this implements is the one that keeps a missing stage honest: when
    `locks.json` locks a trained checkpoint whose stage is not `grpo`, *no GRPO
    checkpoint exists*, and the run must say why in a durable artifact carrying an
    allowed disposition label. Without it, "GRPO was skipped for a reason nobody
    wrote down" is indistinguishable from "GRPO ran and lost the dev comparison",
    and only one of those is reportable. So a missing artifact is a BUG and RS-SFT
    is never silently selected.

    The two allowed labels are NOT interchangeable:
      * GRPO_NOT_RUN_HARDWARE_INFEASIBLE  -- the registered trainer cannot even
        instantiate on the registered card (the arithmetic is checked here). It
        says nothing about whether the variance gate would have opened, so the
        probe is recorded NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT, never "closed".
      * GRPO_NOT_RUN_VARIANCE_GATE_CLOSED -- the complete registered probe ran and
        a binding pooled gate failed. That IS evidence about the policy's variance.
    """
    allowed = list(registered(prereg, "grpo_disposition", "allowed_dispositions"))
    required_fields = list(registered(prereg, "grpo_disposition",
                                      "required_artifact_fields"))
    grpo_stage = registered(prereg, "grpo_disposition", "expected_branch")
    infeasible = dict(registered(prereg, "grpo_disposition", "hardware_infeasible"))
    path = registered(prereg, "grpo_disposition", "artifact_path")
    want_name = registered(prereg, "hardware_integrity", "expected_gpu_name")
    want_bytes = registered(prereg, "hardware_integrity",
                            "expected_cuda_visible_bytes")
    stages = list(prereg["grpo"]["dev_selection"]["stages"])

    stage = ((locks or {}).get("checkpoint") or {}).get("stage")
    if stage is None or stage not in stages:
        # No trained checkpoint is locked (or the lock itself is malformed, which
        # S16 reports on its own): there is no selection to keep honest yet.
        return {"check_status": "OK", "disposition": None,
                "disposition_kind": "STAGE DISPOSITION (never a gate state)",
                "detail": "no locked trained checkpoint; the GRPO stage disposition "
                          "is not yet required",
                "required": False, "problems": []}
    if stage == grpo_stage:
        return {"check_status": "OK", "disposition": None,
                "disposition_kind": "STAGE DISPOSITION (never a gate state)",
                "detail": f"a {grpo_stage} checkpoint is locked; no not-run "
                          f"disposition applies",
                "required": False, "problems": []}

    problems: list[str] = []
    if artifact is None:
        return {"check_status": "BUG", "disposition": None,
                "disposition_kind": "STAGE DISPOSITION (never a gate state)",
                "detail": f"locks.json locks the {stage!r} checkpoint, so NO GRPO "
                          f"checkpoint exists, but the registered disposition "
                          f"artifact {path} is absent. A missing checkpoint with no "
                          f"allowed disposition is an ERROR: {stage} may not be "
                          f"silently selected. Allowed dispositions: {allowed}",
                "required": True, "problems": ["disposition artifact absent"]}

    absent = [f for f in required_fields
              if artifact.get(f) is None or artifact.get(f) == ""]
    # `optimizer_steps: 0` is required content, and 0 is falsy, so presence is
    # tested against None/"" rather than truthiness.
    if absent:
        problems.append(f"{path} is missing required fields {absent}")
    outcome = artifact.get("outcome")
    if outcome not in allowed:
        problems.append(f"{path} outcome {outcome!r} is not one of the registered "
                        f"dispositions {allowed}")
    if artifact.get("branch") not in (None, grpo_stage):
        problems.append(f"{path} branch {artifact.get('branch')!r} is not "
                        f"{grpo_stage!r}")
    if artifact.get("substituted_variant"):
        problems.append(f"{path} records a substituted variant "
                        f"{artifact['substituted_variant']!r}; a substitution is a "
                        f"DIFFERENT treatment and may not be reported under this "
                        f"registered branch")
    if outcome == infeasible["outcome"]:
        for field in ("variance_gate", "trainer_feasibility"):
            if str(artifact.get(field)) != str(infeasible[field]):
                problems.append(f"{path} {field}={artifact.get(field)!r} but the "
                                f"registered value for {outcome} is "
                                f"{infeasible[field]!r}")
        try:
            steps_ok = int(artifact.get("optimizer_steps")) == int(
                infeasible["optimizer_steps"])
        except (TypeError, ValueError):
            steps_ok = False
        if not steps_ok:
            problems.append(f"{path} optimizer_steps="
                            f"{artifact.get('optimizer_steps')!r}; {outcome} "
                            f"requires exactly {infeasible['optimizer_steps']}")
        arith = artifact.get("arithmetic") or {}
        keys = list(infeasible["arithmetic_fields"])
        if not isinstance(arith, dict) or any(k not in arith for k in keys):
            problems.append(f"{path} arithmetic must record {keys}; the disposition "
                            f"is only established by the recorded shortfall")
        else:
            try:
                alloc, copy_gib = float(arith[keys[0]]), float(arith[keys[1]])
            except (TypeError, ValueError):
                alloc, copy_gib = float("nan"), float("nan")
            if not alloc < copy_gib:
                problems.append(f"{path} arithmetic does not show the shortfall: "
                                f"{keys[0]}={arith[keys[0]]!r} is not smaller than "
                                f"{keys[1]}={arith[keys[1]]!r}")
    elif outcome == registered(prereg, "grpo_disposition", "variance_gate_closed",
                               "outcome"):
        if str(artifact.get("variance_gate")) == str(infeasible["variance_gate"]):
            problems.append(f"{path} claims {outcome} while recording the "
                            f"hardware short-circuit variance_gate "
                            f"{infeasible['variance_gate']!r}; those two labels are "
                            f"not interchangeable -- a closed gate requires the "
                            f"complete probe to have run")
        if str(artifact.get("trainer_feasibility")) == str(
                infeasible["trainer_feasibility"]):
            problems.append(f"{path} claims {outcome} while recording "
                            f"trainer_feasibility {infeasible['trainer_feasibility']!r}; "
                            f"an infeasible trainer cannot have closed a variance gate")
    if str(artifact.get("gpu_name", want_name)) != str(want_name):
        problems.append(f"{path} gpu_name {artifact.get('gpu_name')!r} is not the "
                        f"registered {want_name!r}")
    try:
        if int(artifact.get("cuda_visible_bytes", want_bytes)) != int(want_bytes):
            problems.append(f"{path} cuda_visible_bytes "
                            f"{artifact.get('cuda_visible_bytes')!r} is not the "
                            f"registered {want_bytes}")
    except (TypeError, ValueError):
        problems.append(f"{path} cuda_visible_bytes "
                        f"{artifact.get('cuda_visible_bytes')!r} is not an integer")
    grpo_arms = sorted({arm for (arm, _c, _ctl) in eps
                        if arm and arm in prereg["arms"] and arm.startswith("R")})
    if grpo_arms and outcome in allowed:
        problems.append(f"traces exist for GRPO arms {grpo_arms} while the "
                        f"registered disposition is {outcome}; R0/RP are absent by "
                        f"design when GRPO did not run")

    if problems:
        return {"check_status": "BUG", "disposition": outcome,
                "disposition_kind": "STAGE DISPOSITION (never a gate state)",
                "detail": "; ".join(problems[:4]), "required": True,
                "problems": problems}
    return {"check_status": "OK", "disposition": outcome,
            "disposition_kind": "STAGE DISPOSITION (never a gate state)",
            "detail": f"{outcome}: no GRPO checkpoint exists, {stage} is the sole "
                      f"trained candidate, R0/RP absent by design, "
                      f"variance probe {artifact.get('variance_gate')}, "
                      f"{artifact.get('optimizer_steps')} optimizer steps",
            "required": True, "problems": []}


# ---------------------------------------------------------------------------
# paired data assembly + gates
# ---------------------------------------------------------------------------

def _pairs(eps: dict, condition: str, arms=("BP", "TP"),
           splits: set | None = None) -> dict:
    """Paired episodes for one condition, restricted to a declared stratum.

    `splits` is the registered stratum membership (machine.strata). Restricting
    it is load-bearing: MT and H8 augmentation traces also run with
    condition="clean", so an unrestricted join let a separately-sized secondary
    sample into core denominators, where oversampling it could move a launch
    floor or a runaway rate and therefore the winner.
    """
    a = eps.get((arms[0], condition, "none"), {})
    b = eps.get((arms[1], condition, "none"), {})
    common = set(a) & set(b)
    if splits is not None:
        common = {tid for tid in common if a[tid].get("split") in splits}
    return {tid: (a[tid], b[tid]) for tid in common}


def _recovery_ok(ep: dict) -> bool:
    return bool(ep.get("rec", {}).get("certified_recovery"))


def _bootstrap_gate(pairs: list[tuple], outcome_fn, *, label: str, seed: int,
                    margin: float, replicates: int, block: int,
                    min_clusters: int, max_per_cluster: int | None = None) -> dict:
    """Clustered paired difference on the registered STRUCTURAL cluster field.

    Clustering is on `template_cluster_id` and nothing else. The previous key,
    `template_id or task_id`, silently swapped between two incomparable
    resampling units: with a two-wording paraphrase pool it collapsed 1,200 core
    tasks into two clusters, and where the field was absent it fell back to
    `task_id`, i.e. one cluster per observation, which is not a clustered
    bootstrap at all. Both mistakes make the interval NARROWER than the design
    supports, so neither may be reached by a silent fallback.

    A sample with too few clusters, with any missing cluster id, or with more
    value instantiations per cluster than registered gets an INCONCLUSIVE reason
    rather than a bound: an interval over a handful of structural templates is a
    statement about those templates, not about the generator distribution.
    """
    from collections import Counter

    from agentlab.suite.stats import cluster_bootstrap_lb, mcnemar_exact

    diffs, clusters, b_cnt, c_cnt = [], [], 0, 0
    missing = 0
    for bp, tp in pairs:
        yb, yt = int(outcome_fn(bp)), int(outcome_fn(tp))
        diffs.append(yt - yb)
        cid = bp.get("template_cluster_id")
        if cid is None or str(cid) == "":
            missing += 1
            # a distinct sentinel per observation, so a missing field can never
            # masquerade as membership of one big well-populated cluster
            clusters.append(f"\x00missing:{bp['task_id']}")
        else:
            clusters.append(str(cid))
        if yb and not yt:
            b_cnt += 1
        if yt and not yb:
            c_cnt += 1
    boot = cluster_bootstrap_lb(diffs, clusters, seed, label,
                               replicates=replicates, block=block)
    mc = mcnemar_exact(b_cnt, c_cnt)
    sizes = Counter(clusters)
    biggest = max(sizes.values()) if sizes else 0

    reasons = []
    if missing:
        reasons.append(f"{missing}/{len(diffs)} pairs carry no "
                       f"`template_cluster_id`; the registered clustering field "
                       f"is unavailable for this sample")
    if boot["n_clusters"] < min_clusters:
        reasons.append(f"only {boot['n_clusters']} structural clusters "
                       f"(< registered minimum {min_clusters})")
    if max_per_cluster is not None and biggest > max_per_cluster:
        reasons.append(f"largest cluster holds {biggest} value instantiations "
                       f"(> registered maximum {max_per_cluster})")
    if boot["degenerate"]:
        reasons.append("the resampling distribution is degenerate (a single "
                       "cluster or an empty sample)")
    return {"point": boot["point"], "lb": boot["lb"], "margin": margin,
            "n_pairs": len(diffs), "n_clusters": boot["n_clusters"],
            "min_clusters": min_clusters, "missing_cluster_id": missing,
            "max_cluster_size": biggest, "block": block,
            "replicates": boot["replicates"],
            "degenerate": boot["degenerate"], "mcnemar": mc,
            "inconclusive": reasons,
            "pass": (not reasons) and boot["lb"] > margin}


def _bootstrap_verdict(res: dict, detail: str) -> dict:
    """PASS/FAIL on a computable bound; INCONCLUSIVE with the arithmetic kept."""
    numbers = {k: v for k, v in res.items() if k != "mcnemar"}
    if res["inconclusive"]:
        return _g("INCONCLUSIVE",
                  f"{detail}; NO VERDICT: " + "; ".join(res["inconclusive"]),
                  **numbers, mcnemar=res["mcnemar"])
    return _g("PASS" if res["pass"] else "FAIL", detail, **numbers,
              mcnemar=res["mcnemar"])


def _median(sorted_vals: list, convention: str):
    """Median under the REGISTERED even-sample convention.

    An even sample has no unique middle observation, and MT3 is an at-most gate,
    so the choice is a real (if small) part of the gate. The registered value is
    `upper_middle` -- sorted[n//2], 0-based -- which is what the analyzer computed
    at registration and the more conservative of the two for an at-most gate.
    `lower_middle` is implemented so the JSON field genuinely governs rather than
    being a comment on a hard-coded choice.
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("median of an empty sample")
    if convention == "upper_middle":
        return sorted_vals[n // 2]
    if convention == "lower_middle":
        return sorted_vals[(n - 1) // 2]
    raise ValueError(f"unregistered median convention {convention!r}")


def evaluate_agentic_gates(eps: dict, prereg: dict) -> dict:
    boot_cfg = prereg["statistics"]["clustered_bootstrap"]
    seed = int(boot_cfg["seed"])
    replicates = int(boot_cfg["replicates"])
    block = int(boot_cfg["chunk_block"])
    min_clusters = int(registered(prereg, "clustering", "min_clusters_default"))
    per_gate_clusters = registered(prereg, "clustering", "min_clusters_per_gate")
    per_gate_max_inst = registered(prereg, "clustering",
                                   "max_instantiations_per_cluster")

    def _boot(pairs, fn, *, label, margin):
        return _bootstrap_gate(
            pairs, fn, label=label, seed=seed, margin=margin,
            replicates=replicates, block=block,
            min_clusters=int(per_gate_clusters.get(label, min_clusters)),
            max_per_cluster=(int(per_gate_max_inst[label])
                             if label in per_gate_max_inst else None))

    core_splits = set(registered(prereg, "strata", "core"))
    mt_splits = set(registered(prereg, "strata", "mt"))
    h8_splits = set(registered(prereg, "strata", "h8"))
    gates: dict = {}

    clean = _pairs(eps, "clean", splits=core_splits)
    faulted = _pairs(eps, "faulted", splits=core_splits)

    # ---- ER: primary claim ------------------------------------------------
    min_c = int(registered(prereg, "er", "min_common_clean"))
    er2_margin = float(registered(prereg, "er", "er2_margin"))
    er3_lb = float(registered(prereg, "er", "er3_wilson_lb"))
    er4_margin = float(registered(prereg, "er", "er4_margin"))

    common_clean = {tid for tid, (bp, tp) in clean.items()
                    if bp["rep"]["certified_success"] and tp["rep"]["certified_success"]}
    c_set = sorted(common_clean & set(faulted))
    if len(c_set) < min_c:
        gates["ER1"] = _g("INCONCLUSIVE",
                          f"|C|={len(c_set)} < {min_c}: underpowered; the recovery claim "
                          f"receives NO verdict regardless of point estimates", C=len(c_set),
                          min_c=min_c)
    else:
        gates["ER1"] = _g("PASS", f"|C|={len(c_set)} >= {min_c}", C=len(c_set),
                          min_c=min_c)

    if c_set:
        er2 = _boot([faulted[t] for t in c_set], _recovery_ok, label="ER2",
                    margin=er2_margin)
        gates["ER2"] = _bootstrap_verdict(
            er2, f"certified recovery diff on C: point {er2['point']:+.3f}, "
                 f"97.5% clustered LB {er2['lb']:+.3f} vs margin {er2_margin:+.2f} "
                 f"over {er2['n_clusters']} structural clusters; exact McNemar "
                 f"b={er2['mcnemar']['b']} c={er2['mcnemar']['c']} "
                 f"p={er2['mcnemar']['p_two_sided']:.4g}")
        k = sum(1 for t in c_set if _recovery_ok(faulted[t][1]))
        gates["ER3"] = _wilson_lb_gate("ER3", k, len(c_set), er3_lb,
                                       "TP certified recovery on C",
                                       **_n_clusters([faulted[t][1] for t in c_set]))
    else:
        gates["ER2"] = _g("INCONCLUSIVE", "no common-clean pairs with faulted replays")
        gates["ER3"] = _g("INCONCLUSIVE", "no common-clean pairs with faulted replays")
    if gates["ER1"]["status"] == "INCONCLUSIVE":
        for name in ("ER2", "ER3"):
            gates[name] = _no_verdict(
                gates[name], f"|C| below the preregistered floor of {min_c}")

    if clean:
        er4 = _boot(list(clean.values()),
                    lambda ep: ep["rep"]["certified_success"],
                    label="ER4", margin=er4_margin)
        gates["ER4"] = _bootstrap_verdict(
            er4, f"clean non-inferiority: diff {er4['point']:+.3f}, 97.5% "
                 f"clustered LB {er4['lb']:+.3f} vs margin {er4_margin:+.2f} over "
                 f"{er4['n_clusters']} structural clusters")
    else:
        gates["ER4"] = _g("INCONCLUSIVE", "no paired clean episodes")

    min_assigned = int(registered(prereg, "er", "min_assigned_faults"))
    er5_min_diff = float(registered(prereg, "er", "er5_min_diff"))
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
            gates["ER5"] = _g("PASS" if diff >= er5_min_diff else "FAIL",
                              f"intention-to-treat certified recovery diff {diff:+.3f} "
                              f"over {len(faulted)} assigned pairs "
                              f"(must be >= {er5_min_diff:g})",
                              n=len(faulted), diff=diff, tp=itt_tp, bp=itt_bp)
    else:
        gates["ER5"] = _g("INCONCLUSIVE", "no assigned fault pairs")

    # ER6/ER7 read "all core TP episodes" literally: the CORE stratum only. MT and
    # H8 augmentation traces also run with condition="clean", and they are a
    # separately sized sample; letting them into this denominator would let the
    # size of a secondary sample move the primary claim's runaway and
    # hallucination bounds in either direction.
    er67_conditions = tuple(registered(prereg, "er", "er67_conditions"))
    er6_ub = float(registered(prereg, "er", "er6_runaway_ub"))
    er7_ub = float(registered(prereg, "er", "er7_hallucination_ub"))
    tp_core = [ep for (arm, cond, ctl), tasks in eps.items() if arm == "TP"
               and ctl == "none" and cond in er67_conditions
               for ep in tasks.values() if ep.get("split") in core_splits]
    if tp_core:
        cl = _n_clusters(tp_core)
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_core)
        gates["ER6"] = _wilson_ub_gate("ER6", k_run, len(tp_core), er6_ub,
                                       "TP runaway on the core stratum", **cl)
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_core)
        gates["ER7"] = _wilson_ub_gate("ER7", k_h, len(tp_core), er7_ub,
                                       "TP hallucinated-result on the core stratum",
                                       **cl)
    else:
        gates["ER6"] = _g("INCONCLUSIVE", "no core-stratum TP episodes")
        gates["ER7"] = _g("INCONCLUSIVE", "no core-stratum TP episodes")

    from agentlab.suite.faults import group_of

    er8_floor = float(registered(prereg, "er", "er8_group_floor"))
    er8_groups = list(registered(prereg, "er", "er8_groups"))
    er8_min = int(registered(prereg, "er", "er8_min_per_group"))
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
    counts = {g: groups.get(g, [0, 0, 0])[2] for g in er8_groups}
    # "no group fell below the floor" computed over two of three registered groups
    # is not the registered gate: a missing or short group is INCONCLUSIVE, never
    # a pass by omission.
    short = {g: n for g, n in counts.items() if n < er8_min}
    extra = sorted(set(groups) - set(er8_groups))
    if short or extra:
        why = []
        if short:
            why.append("fault groups below the registered "
                       f"{er8_min} assigned episodes each: "
                       + ", ".join(f"{g} n={n}" for g, n in sorted(short.items())))
        if extra:
            why.append(f"unregistered fault groups present: {extra}")
        gates["ER8"] = _g("INCONCLUSIVE",
                          "; ".join(why) + "; the per-group floor is only a gate "
                          "when all three registered groups are present at size",
                          counts=counts, min_per_group=er8_min)
    else:
        worst = {g: (a[0] - a[1]) / a[2] for g, a in groups.items()}
        bad = {g: d for g, d in worst.items() if d < er8_floor}
        gates["ER8"] = _g("FAIL" if bad else "PASS",
                          f"per-group ITT recovery diffs: "
                          + ", ".join(f"{g} {d:+.3f} (n={groups[g][2]})"
                                      for g, d in sorted(worst.items()))
                          + (f"; below {er8_floor:+.2f}: " + ",".join(bad)
                             if bad else ""),
                          diffs=worst, counts=counts, min_per_group=er8_min,
                          floor=er8_floor)

    # ---- MT: secondary (a) --------------------------------------------------
    mt_horizon = int(registered(prereg, "mt", "horizon"))
    min_mt = int(registered(prereg, "mt", "min_pairs"))
    mt1_margin = float(registered(prereg, "mt", "mt1_margin"))
    mt2_lb = float(registered(prereg, "mt", "mt2_wilson_lb"))
    mt3_oracle = int(registered(prereg, "mt", "mt3_oracle_min_calls"))
    mt3_slack = int(registered(prereg, "mt", "mt3_slack"))
    mt3_convention = str(registered(prereg, "mt", "mt3_median_convention"))
    mt4_run_ub = float(registered(prereg, "mt", "mt4_runaway_ub"))
    mt4_hal_ub = float(registered(prereg, "mt", "mt4_hallucination_ub"))
    mt5_n_patterns = int(registered(prereg, "mt", "mt5_n_patterns"))
    mt5_min_cell = int(registered(prereg, "mt", "mt5_min_per_pattern"))
    mt5_floor = float(registered(prereg, "mt", "mt5_pattern_floor"))

    mt_clean = _pairs(eps, "clean", splits=mt_splits)
    mt_pairs = {tid: pair for tid, pair in mt_clean.items()
                if pair[0]["all_tools_required"]
                and pair[0]["horizon"] == mt_horizon}
    if len(mt_pairs) < min_mt:
        gates["MT1"] = _g("INCONCLUSIVE", f"only {len(mt_pairs)} all-tools "
                          f"H{mt_horizon} pairs (< {min_mt})", n=len(mt_pairs),
                          min_pairs=min_mt)
    else:
        mt1 = _boot(list(mt_pairs.values()),
                    lambda ep: bool(ep.get("orch", {}).get(
                        "certified_orchestration")),
                    label="MT1", margin=mt1_margin)
        gates["MT1"] = _bootstrap_verdict(
            mt1, f"certified all-tools diff: point {mt1['point']:+.3f}, 97.5% "
                 f"clustered LB {mt1['lb']:+.3f} vs {mt1_margin:+.2f} over "
                 f"{mt1['n_clusters']} structural clusters; exact McNemar "
                 f"p={mt1['mcnemar']['p_two_sided']:.4g}")
    if mt_pairs:
        tp_eps = [tp for _, tp in mt_pairs.values()]
        cl = _n_clusters(tp_eps)
        k = sum(bool(ep.get("orch", {}).get("certified_orchestration")) for ep in tp_eps)
        gates["MT2"] = _wilson_lb_gate("MT2", k, len(tp_eps), mt2_lb,
                                       "TP certified all-tools", **cl)
        calls = sorted(ep["rep"]["n_calls"] for ep in tp_eps)
        med = _median(calls, mt3_convention)
        gates["MT3"] = _g("PASS" if med <= mt3_oracle + mt3_slack else "FAIL",
                          f"TP median calls {med} vs oracle {mt3_oracle} + "
                          f"{mt3_slack} (n={len(calls)}, "
                          f"{mt3_convention} median convention)",
                          median=med, n=len(calls), convention=mt3_convention,
                          threshold=mt3_oracle + mt3_slack)
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_eps)
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_eps)
        mt4_run = _wilson_ub_gate("MT4run", k_run, len(tp_eps), mt4_run_ub,
                                  "TP MT runaway", **cl)
        mt4_h = _wilson_ub_gate("MT4hal", k_h, len(tp_eps), mt4_hal_ub,
                                "TP MT hallucinated-result", **cl)
        # A compound gate is only as measurable as its weakest leg; a FAIL on
        # either leg is still a FAIL (never softened by the other's INCONCLUSIVE).
        legs = [mt4_run["status"], mt4_h["status"]]
        status = ("FAIL" if "FAIL" in legs
                  else "INCONCLUSIVE" if "INCONCLUSIVE" in legs else "PASS")
        gates["MT4"] = _g(status, f"runaway leg [{mt4_run['status']}] "
                          f"{mt4_run['detail']}; hallucination leg [{mt4_h['status']}] "
                          f"{mt4_h['detail']}",
                          runaway_ub=mt4_run["numbers"]["ub"],
                          halluc_ub=mt4_h["numbers"]["ub"])
        pat: dict = {}
        for bp, tp in mt_pairs.values():
            pid = bp.get("pattern_id")
            if pid is None:
                continue
            acc = pat.setdefault(pid, [0, 0, 0])
            acc[0] += bool(tp.get("orch", {}).get("certified_orchestration"))
            acc[1] += bool(bp.get("orch", {}).get("certified_orchestration"))
            acc[2] += 1
        if (len(pat) >= mt5_n_patterns
                and all(a[2] >= mt5_min_cell for a in pat.values())):
            diffs = {p_: (a[0] - a[1]) / a[2] for p_, a in pat.items()}
            bad = {p_: d for p_, d in diffs.items() if d < mt5_floor}
            gates["MT5"] = _g("FAIL" if bad else "PASS",
                              "per-pattern diffs: " + ", ".join(
                                  f"p{p_} {d:+.3f}" for p_, d in sorted(diffs.items()))
                              + (f"; below {mt5_floor:+.2f}: "
                                 + ",".join(str(x) for x in bad) if bad else ""),
                              diffs=diffs, floor=mt5_floor)
        else:
            gates["MT5"] = _g("INCONCLUSIVE",
                              f"order patterns incomplete: {len(pat)} patterns, "
                              f"min cell {min((a[2] for a in pat.values()), default=0)} "
                              f"(need {mt5_n_patterns} patterns x "
                              f">={mt5_min_cell} pairs)")
    else:
        for name in ("MT2", "MT3", "MT4", "MT5"):
            gates[name] = _g("INCONCLUSIVE",
                             f"no all-tools H{mt_horizon} pairs")
    # MT6 is stitched in from the S11 veto by the caller.

    # ---- HR: secondary (b) --------------------------------------------------
    hr_horizon = int(registered(prereg, "hr", "horizon"))
    hr_families = tuple(registered(prereg, "hr", "families"))
    min_hr = int(registered(prereg, "hr", "min_pairs"))
    min_disc = int(registered(prereg, "hr", "min_discordant"))
    hr1_margin = float(registered(prereg, "hr", "hr1_margin"))
    hr2_ub = float(registered(prereg, "hr", "hr2_runaway_ub"))
    hr3_ub = float(registered(prereg, "hr", "hr3_hallucination_ub"))

    hr_clean = _pairs(eps, "clean", splits=h8_splits)
    hr_pairs = {tid: pair for tid, pair in hr_clean.items()
                if pair[0]["horizon"] == hr_horizon
                and pair[0]["family"] in hr_families}
    if len(hr_pairs) < min_hr:
        gates["HR1"] = _g("INCONCLUSIVE", f"only {len(hr_pairs)} H{hr_horizon} clean "
                          f"pairs (< {min_hr})", n=len(hr_pairs), min_pairs=min_hr)
    else:
        hr1 = _boot(list(hr_pairs.values()),
                    lambda ep: ep["rep"]["certified_success"],
                    label="HR1", margin=hr1_margin)
        if hr1["mcnemar"]["n_discordant"] < min_disc:
            gates["HR1"] = _g("INCONCLUSIVE",
                              f"only {hr1['mcnemar']['n_discordant']} discordant "
                              f"H{hr_horizon} pairs (< {min_disc}); diff "
                              f"{hr1['point']:+.3f} reported, not gated",
                              **{k: v for k, v in hr1.items() if k != "mcnemar"},
                              mcnemar=hr1["mcnemar"])
        else:
            gates["HR1"] = _bootstrap_verdict(
                hr1, f"H{hr_horizon} certified success diff: point "
                     f"{hr1['point']:+.3f}, 97.5% clustered LB {hr1['lb']:+.3f} vs "
                     f"{hr1_margin:+.2f} over {hr1['n_clusters']} structural clusters")
    if hr_pairs:
        tp_eps = [tp for _, tp in hr_pairs.values()]
        cl = _n_clusters(tp_eps)
        k_run = sum(ep["rep"]["runaway"]["runaway"] for ep in tp_eps)
        gates["HR2"] = _wilson_ub_gate("HR2", k_run, len(tp_eps), hr2_ub,
                                       f"TP H{hr_horizon} runaway", **cl)
        k_h = sum(ep["rep"]["hallucination"]["hallucinated"] for ep in tp_eps)
        gates["HR3"] = _wilson_ub_gate("HR3", k_h, len(tp_eps), hr3_ub,
                                       f"TP H{hr_horizon} hallucinated-result", **cl)
    else:
        gates["HR2"] = _g("INCONCLUSIVE", f"no H{hr_horizon} clean pairs")
        gates["HR3"] = _g("INCONCLUSIVE", f"no H{hr_horizon} clean pairs")

    return gates


def evaluate_floors(eps: dict, arm: str, prereg: dict) -> dict:
    """Absolute launch floors F1-F5 for one arm, on the registered denominators.

    Denominators are frozen in machine.floors: F1/F2 over the arm's CORE clean
    episodes with control `none`, F3/F4 over its CORE faulted episodes (ITT --
    every assigned fault episode stays in, including timeouts, parser failures,
    crashes and runaways), F5 over the union. That restriction is the point: MT
    and H8 augmentation traces also carry condition="clean", so an unrestricted
    denominator let the SIZE of a secondary sample move a floor, and the floors
    decide the winner.

    Floors are POINT estimates against the threshold, as registered; Wilson
    intervals are reported beside them for the reader but do not decide.
    """
    from agentlab.suite.stats import wilson as _wilson

    core_splits = set(registered(prereg, "strata",
                                 registered(prereg, "floors",
                                            "denominator_stratum")))
    f1_t = float(registered(prereg, "floors", "f1_clean_overall"))
    f2_t = float(registered(prereg, "floors", "f2_clean_per_family"))
    f3_t = float(registered(prereg, "floors", "f3_faulted_overall"))
    f4_t = float(registered(prereg, "floors", "f4_faulted_per_family"))
    f5_t = float(registered(prereg, "floors", "f5_loop_crash_max"))

    def _core(condition):
        return [ep for ep in eps.get((arm, condition, "none"), {}).values()
                if ep.get("split") in core_splits]

    clean, faulted = _core("clean"), _core("faulted")
    floors: dict = {}
    if not clean or not faulted:
        for f in ("F1", "F2", "F3", "F4", "F5"):
            floors[f] = _g("INCONCLUSIVE",
                           f"{arm}: missing clean or faulted core-stratum traces "
                           f"(clean n={len(clean)}, faulted n={len(faulted)}; "
                           f"splits {sorted(core_splits)})")
        return floors

    def rate(eps_list, fn):
        k = sum(1 for e in eps_list if fn(e))
        return k, len(eps_list), k / len(eps_list)

    def by_family(eps_list, fn):
        fam: dict = {}
        for ep in eps_list:
            fam.setdefault(ep["family"], []).append(bool(fn(ep)))
        return {f: sum(v) / len(v) for f, v in fam.items()}

    k, n, ok = rate(clean, lambda e: e["rep"]["certified_success"])
    _, lo, hi = _wilson(k, n)
    floors["F1"] = _g("PASS" if ok >= f1_t else "FAIL",
                      f"{arm} overall clean certified success {ok:.3f} "
                      f"[{lo:.3f},{hi:.3f}] vs {f1_t} (point estimate decides; "
                      f"n={n} core clean)", rate=ok, k=k, n=n, threshold=f1_t)
    fam = by_family(clean, lambda e: e["rep"]["certified_success"])
    bad = {f: r for f, r in fam.items() if r < f2_t}
    floors["F2"] = _g("FAIL" if bad else "PASS",
                      f"{arm} clean by family: "
                      + ", ".join(f"{f} {r:.3f}" for f, r in sorted(fam.items()))
                      + (f"; below {f2_t}: {sorted(bad)}" if bad else ""),
                      by_family=fam, threshold=f2_t)
    kf, nf, okf = rate(faulted, lambda e: e["rep"]["certified_success"])
    _, lof, hif = _wilson(kf, nf)
    floors["F3"] = _g("PASS" if okf >= f3_t else "FAIL",
                      f"{arm} overall faulted strict success {okf:.3f} "
                      f"[{lof:.3f},{hif:.3f}] vs {f3_t} (intention-to-treat over "
                      f"n={nf} assigned core fault episodes)", rate=okf, k=kf, n=nf,
                      threshold=f3_t)
    famf = by_family(faulted, lambda e: e["rep"]["certified_success"])
    badf = {f: r for f, r in famf.items() if r < f4_t}
    floors["F4"] = _g("FAIL" if badf else "PASS",
                      f"{arm} faulted by family: "
                      + ", ".join(f"{f} {r:.3f}" for f, r in sorted(famf.items()))
                      + (f"; below {f4_t}: {sorted(badf)}" if badf else ""),
                      by_family=famf, threshold=f4_t)
    kc, nc, crash = rate(clean + faulted, _crashish)
    floors["F5"] = _g("PASS" if crash < f5_t else "FAIL",
                      f"{arm} loop/crash rate {crash:.4f} vs < {f5_t} "
                      f"({kc}/{nc} core episodes)", rate=crash, k=kc, n=nc,
                      threshold=f5_t)
    return floors


def horizon_curves(eps: dict, arms=("BP", "TP"), *, prereg: dict) -> dict:
    """Descriptive success-vs-horizon curves with Wilson bands. No fitting;
    H50 is reported only when the OBSERVED curve crosses the registered rate.

    `prereg` is required rather than defaulted: the crossing rate is a registered
    reporting rule, and a default here would be one more number living in code.
    """
    from agentlab.suite.stats import wilson as _wilson

    cross = float(registered(prereg, "curves", "h50_crossing_rate"))
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
        above = [r for r in pts if r["p"] >= cross]
        below = [r for r in pts if r["p"] < cross]
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


def _backfill_from_specs(eps: dict, specs_by_id: dict) -> int:
    """Take `template_cluster_id` / `split` from the authoritative spec manifest.

    Same registered field, second authoritative source -- not a fallback to a
    different field. The manifest passed as --specs is the frozen definition of
    the held-out sample, so when a trace writer has not yet propagated
    `template_cluster_id` onto the trace row the manifest still carries it and
    the clustered bounds stay computable. A task absent from the manifest is left
    exactly as the trace had it, so nothing is invented.
    """
    filled = 0
    for tasks in eps.values():
        for tid, ep in tasks.items():
            spec = specs_by_id.get(tid)
            if not spec:
                continue
            if ep.get("template_cluster_id") in (None, "") and \
                    spec.get("template_cluster_id") not in (None, ""):
                ep["template_cluster_id"] = spec["template_cluster_id"]
                filled += 1
            if ep.get("split") in (None, "") and spec.get("split"):
                ep["split"] = spec["split"]
    return filled


def strata_census(eps: dict, prereg: dict) -> dict:
    """Every (arm, condition, control, split) count the verdict was computed on.

    Emitted unconditionally so a reader never has to trust a denominator: if a
    stratum is oversampled, mislabelled, or absent, it shows up here rather than
    silently inside a rate.
    """
    declared = {}
    for name in ("core", "mt", "h8", "stress"):
        for split in registered(prereg, "strata", name):
            declared.setdefault(split, []).append(name)
    census: dict = {}
    unassigned: dict = {}
    for (arm, cond, ctl), tasks in eps.items():
        for ep in tasks.values():
            split = ep.get("split")
            key = f"{arm}/{cond}/{ctl}/{split}"
            census[key] = census.get(key, 0) + 1
            if split not in declared:
                unassigned[key] = unassigned.get(key, 0) + 1
    return {"counts": dict(sorted(census.items())),
            "declared_splits": {k: sorted(v) for k, v in sorted(declared.items())},
            "unassigned_split_traces": dict(sorted(unassigned.items())),
            "note": "traces whose split is in no declared stratum are excluded "
                    "from every gated sample; exclusion can only widen an "
                    "interval or lower a count, never favour an arm"}


def holm_secondary_mcnemar(gates: dict, prereg: dict) -> dict:
    """Holm-adjusted exact McNemar p-values across the two secondary claims.

    The preregistration has always specified this table; nothing produced it, so
    the supporting evidence for the two secondary claims was simply missing from
    every report. It is REPORTED, never gated -- the gate is the clustered lower
    bound -- and an absent or underpowered secondary enters the family with
    p = 1.0 rather than being dropped, because dropping a member shrinks the
    family and weakens the adjustment for the survivor.
    """
    from agentlab.suite.stats import holm

    cfg = prereg["statistics"]["holm"]
    family = list(cfg["family"])
    p_kind = str(cfg["p_kind"])
    raw, labels = [], []
    for name in family:
        g = gates.get(name) or {}
        mc = (g.get("numbers") or {}).get("mcnemar") or {}
        p = mc.get(p_kind)
        if p is None:
            raw.append(1.0)
            labels.append("INCONCLUSIVE (no exact McNemar available)")
        else:
            raw.append(float(p))
            labels.append(g.get("status", "INCONCLUSIVE"))
    adj = holm(raw)
    return {"family": family, "p_kind": p_kind, "alpha": float(cfg["alpha"]),
            "raw": {n: raw[i] for i, n in enumerate(family)},
            "adjusted": {n: adj[i] for i, n in enumerate(family)},
            "member_status": {n: labels[i] for i, n in enumerate(family)},
            "reported_never_gated": bool(cfg["reported_never_gated"])}


def _overlay_bug(block: dict, bug_names: list[str]) -> None:
    """Relabel every official status in `block` as BUG, in place.

    The raw arithmetic moves to `observed_status`, which is declared a NON-VERDICT
    field. Nothing is deleted -- but a script that reads
    gates["ER2"]["status"] out of a vetoed run can no longer see "PASS" for a
    number the run cannot support, and neither can a reader skimming the table.
    """
    for name, res in list(block.items()):
        out = {"status": "BUG", "observed_status": res["status"],
               "detail": f"vetoed by harness BUG {','.join(bug_names)}; "
                         f"observed (NOT a verdict) [{res['status']}]: "
                         f"{res['detail']}",
               "numbers": res.get("numbers", {})}
        if "measured_status" in res:
            out["measured_status"] = res["measured_status"]
        block[name] = out


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
        _backfill_from_specs(eps, specs_by_id)
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

    # The GRPO stage-disposition artifact. Its registered path is repo-relative;
    # it is read from the same results directory as locks.json so a fixture or a
    # relocated run reads its own artifact rather than the repo's.
    disposition_artifact = None
    disp_p = pathlib.Path(rdir) / pathlib.Path(
        registered(prereg, "grpo_disposition", "artifact_path")).name
    if disp_p.exists():
        disposition_artifact = json.loads(disp_p.read_text())
    grpo_disposition = grpo_stage_disposition(eps, prereg, locks,
                                              disposition_artifact)

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
        "S19": veto_s19_hardware_integrity(eps, prereg),
    }
    # The one-locked-checkpoint veto also owns the missing-GRPO-checkpoint rule:
    # a locked non-GRPO checkpoint with no allowed stage disposition means the
    # sole trained candidate was selected by default rather than by the
    # registered route, which makes every trained number unattributable exactly
    # as an ambiguous lock does.
    if grpo_disposition["check_status"] == "BUG":
        prior = vetoes["S16"]
        detail = f"GRPO stage disposition: {grpo_disposition['detail']}"
        if prior["status"] == "BUG":
            detail = f"{prior['detail']}; {detail}"
        vetoes["S16"] = _g("BUG", detail, **prior.get("numbers", {}))
    mandatory = list(registered(prereg, "mandatory_harness_checks"))
    unknown = sorted(set(mandatory) - set(vetoes))
    if unknown:
        raise KeyError(f"the preregistration declares mandatory harness checks the "
                       f"analyzer does not implement: {unknown}")
    bug_names = sorted(n for n, v in vetoes.items() if v["status"] == "BUG")
    any_bug = bool(bug_names)
    mandatory_inconclusive = sorted(
        n for n in mandatory if vetoes[n]["status"] == "INCONCLUSIVE")

    gates = evaluate_agentic_gates(eps, prereg)
    # MT6 mirrors the absent-information veto restricted to the MT family.
    s11 = vetoes["S11"]
    gates["MT6"] = {"status": {"OK": "PASS"}.get(s11["status"], s11["status"]),
                    "detail": f"absent-information control (S11): {s11['detail']}",
                    "numbers": s11["numbers"]}

    # A harness BUG vetoes every DOWNSTREAM official status -- gates, floors,
    # claims and the winner -- not only the claims: a reader (or a script) that
    # pulled gates["ER2"]["status"] out of a vetoed verdict would otherwise see
    # "PASS" for a number the run cannot support. The arithmetic survives under
    # the NON-VERDICT field `observed_status`: relabelled, not deleted.
    if any_bug:
        _overlay_bug(gates, bug_names)

    # After the overlay, so the reported member status is the OFFICIAL one: a
    # supporting p-value beside a gate labelled INCONCLUSIVE in a run whose gate
    # is officially BUG would read as if the gate had been evaluated.
    holm_secondary = holm_secondary_mcnemar(gates, prereg)

    def claim_status(names: list[str]) -> str:
        # Reads the EFFECTIVE status. `measured_status` is deliberately not used:
        # a gate downgraded for underpower is preregistered as no-verdict, and
        # promoting its measured FAIL back into the claim would read an
        # underpowered sample as a refutation.
        st = [gates[n]["status"] for n in names]
        bad = sorted({s for s in st} - set(OUTCOME_STATES))
        if bad:
            # SKIP is forbidden and an unknown state is a defect, not a nuance.
            return "BUG"
        if any_bug:
            return "BUG"
        # A real FAIL outranks missing evidence: one refuted gate refutes the
        # claim whatever else is unmeasured. Only when nothing FAILed does thin
        # evidence make the claim INCONCLUSIVE -- and INCONCLUSIVE never reads
        # as support.
        if any(s == "FAIL" for s in st):
            return "FAIL"
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

    floors_tp = evaluate_floors(eps, "TP", prereg)
    floors_bp = evaluate_floors(eps, "BP", prereg)

    def floors_pass(fl: dict) -> bool:
        return bool(fl) and all(v["status"] == "PASS" for v in fl.values())

    if any_bug:
        # Launch floors are model-level statements too; a vetoed run must not
        # show an arm "clearing the floor". Same relabel-not-delete rule.
        _overlay_bug(floors_tp, bug_names)
        _overlay_bug(floors_bp, bug_names)

    # ---- winner rule: the frozen truth table, evaluated in rank order -------
    # Ranks come from winner_rule.truth_table in the preregistration. No
    # discretion is left anywhere: in particular rank 2 is the fix for a real
    # defect -- the analyzer used to ship a winner while S9, S10, S14 or S18 was
    # INCONCLUSIVE, i.e. while the harness that makes a winner meaningful was
    # unverified.
    er4, er2 = gates["ER4"]["status"], gates["ER2"]["status"]
    floors_incomplete = [f"{arm} {n}" for arm, fl in (("TP", floors_tp),
                                                     ("BP", floors_bp))
                         for n, v in fl.items() if v["status"] == "INCONCLUSIVE"]
    if any_bug:
        winner_rank = 1
        winner = (f"BUG / NO VERDICT: mandatory harness check(s) "
                  f"{','.join(bug_names)} are BUG; no winner, and no official "
                  f"status above is a statement about the model")
    elif mandatory_inconclusive:
        winner_rank = 2
        winner = (f"NO VERDICT: mandatory harness check(s) "
                  f"{','.join(mandatory_inconclusive)} are INCONCLUSIVE; a winner "
                  f"may not ship while the harness that would make it meaningful "
                  f"is unverified")
    elif floors_pass(floors_tp) and er4 == "PASS" and er2 == "PASS":
        winner_rank = 4
        winner = ("TP (trained arm ships: every launch floor clear, "
                  "clean-non-inferior at ER4, certified recovery clustered LB "
                  "above the ER2 margin)")
    elif floors_pass(floors_bp):
        winner_rank = 5
        winner = ("BP (frozen prompted base ships; the trained comparison did not "
                  f"clear ER2/ER4 [ER2 {er2}, ER4 {er4}] and the training leg is "
                  f"dropped and reported)")
    elif floors_incomplete:
        winner_rank = 6
        winner = ("NO VERDICT: floor evidence incomplete (INCONCLUSIVE): "
                  + ", ".join(sorted(floors_incomplete)))
    else:
        winner_rank = 7
        winner = "none: no successful multifaceted pipeline yet"

    return {"vetoes": vetoes, "any_bug": any_bug,
            "mandatory_harness_checks": mandatory,
            "mandatory_bug": bug_names,
            "mandatory_inconclusive": mandatory_inconclusive,
            # A STAGE disposition, kept in its own namespace: it is not a gate, not
            # a claim, not a floor and not a winner, and its labels are absent from
            # `outcome_states` so nothing can read one as a capability result.
            "grpo_stage_disposition": grpo_disposition,
            "gates": gates, "claims": claims,
            "floors": {"TP": floors_tp, "BP": floors_bp},
            "winner": winner, "winner_rank": winner_rank,
            "holm_secondary_mcnemar": holm_secondary,
            "strata_census": strata_census(eps, prereg),
            "curves": horizon_curves(eps, prereg=prereg),
            "claims_to_reject": prereg["claims_to_reject"]}


def render_agentic_verdict(v: dict) -> str:
    lines = ["# Agentic machine verdict", ""]
    lines.append("## Harness vetoes S8-S19 (ALL mandatory: any BUG makes the whole "
                 "verdict BUG, any INCONCLUSIVE means NO VERDICT)")
    for name, res in v["vetoes"].items():
        lines.append(f"  {res['status']:<13} {name}: {res['detail']}")
    lines.append("")
    d = v.get("grpo_stage_disposition")
    if d and d.get("disposition"):
        lines.append("## GRPO stage disposition (a STAGE DISPOSITION, never a gate "
                     "state: not PASS/FAIL/INCONCLUSIVE/BUG about the model)")
        lines.append(f"  {d['disposition']} [{d['check_status']}]: {d['detail']}")
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
    h = v.get("holm_secondary_mcnemar")
    if h:
        lines.append(f"## Holm-adjusted exact McNemar across the secondary claims "
                     f"({h['p_kind']}, alpha {h['alpha']}; REPORTED, never gated -- "
                     f"the gate is the clustered lower bound)")
        for name in h["family"]:
            lines.append(f"  {name}: raw p={h['raw'][name]:.4g}  "
                         f"Holm-adjusted p={h['adjusted'][name]:.4g}  "
                         f"[gate {h['member_status'][name]}]")
        lines.append("")
    cen = v.get("strata_census")
    if cen:
        lines.append("## Stratum census (the denominators every number above used)")
        for key, n in cen["counts"].items():
            lines.append(f"  {n:>6}  {key}")
        if cen["unassigned_split_traces"]:
            lines.append(f"  EXCLUDED (split in no declared stratum): "
                         f"{cen['unassigned_split_traces']}")
        lines.append("")
    lines.append("## Descriptive horizon curves (Wilson 95% bands; no extrapolation)")
    for key, cur in v["curves"].items():
        pts = " ".join(f"H{r['horizon']}:{r['p']:.2f}[{r['wilson_lo']:.2f},"
                       f"{r['wilson_hi']:.2f}]" for r in cur["points"])
        lines.append(f"  {key}: {pts}  H50 {cur['H50']}")
    lines.append("")
    lines.append(f"## Winner (truth-table rank {v.get('winner_rank')}): {v['winner']}")
    if v["any_bug"]:
        lines.append("HARNESS BUG DETECTED -- nothing above is a statement about the "
                     "model; every official status is BUG and the raw arithmetic is "
                     "kept only under the non-verdict field `observed_status`. Fix "
                     "the harness, then re-run.")
    elif v.get("mandatory_inconclusive"):
        lines.append("MANDATORY HARNESS CHECKS UNVERIFIED "
                     f"({','.join(v['mandatory_inconclusive'])}) -- NO VERDICT. This "
                     "is not a negative result about the model; it is missing "
                     "harness evidence.")
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
