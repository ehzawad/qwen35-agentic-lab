"""Elicitation control: the eight preregistered prompt files and their tournament.

Round 1 of this lab measured a one-sentence prompt recovering 81.8% of an
apparent SFT gain, so the control is frozen FIRST and every training claim is
measured against it (the primary comparison is TP vs BP, never T0 vs B0).

THE CANDIDATES ARE THE COMMITTED FILES IN `prompts/agentic/`, hash-pinned in
`configs/agentic_preregister.json` under `prompt_candidates.sha256`. That is the
frozen preregistration and therefore the single source of truth. An earlier
parallel build carried a SECOND elicitation control in this module -- eight
one-sentence fragments appended to a hardcoded system prompt, hashed into a
separate configs/prompt_candidates.json -- which would have produced a BP arm
that no preregistered gate describes. It is gone; only the preregistered set
survives, together with its preregistered neutral default, tie-break rule and
per-axis sample sizes.

Commands:

  verify    the eight files on disk hash back to the preregistration
  axes      what the tournament would draw from the committed dev split, and
            whether the split can supply the preregistered sample sizes
  run       GPU: one candidate over one round, resumable by file presence
  finalize  apply the preregistered winner rule, write configs/frozen_prompt.json

Preregistered tournament (docs/AGENTIC_PROTOCOL.md section 2):

  round 1   100 dev instances per axis, all eight candidates
  round 2   another 200 dev instances per axis, the top two
  winner    highest mean certified strict success on the combined 300, the three
            axes weighted equally; ties to the SHORTER file, then the lower index
  honest    "best of eight preregistered system prompts under a fixed search
            budget" -- never "best possible prompt"

The three axes are the three claim axes, not the three families:

  recovery        fault-assigned dev episodes (the primary claim)
  orchestration   H4 typed_relay, where the answer causally requires all three
                  canonical tools (secondary claim a)
  h8              clean H8 episodes, lookup_chain + typed_relay (secondary b)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from agentlab.suite.configio import ROOT, ledger_append, ledger_guard, load_config
from agentlab.suite.generate import cell_slice, group_by_cell
from agentlab.suite.schema import digest_text, file_sha256

TOURNEY_DIR = ROOT / "out" / "multiface" / "prompt_tournament"
RESULT_PATH = ROOT / "out" / "multiface" / "prompt_tournament.json"
PREREGISTER_PATH = ROOT / "configs" / "agentic_preregister.json"

AXES = ("recovery", "orchestration", "h8")


# --------------------------------------------------------------------------
# the preregistered candidate set
# --------------------------------------------------------------------------

def preregistration(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else PREREGISTER_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def candidates(prereg: dict | None = None) -> list[dict]:
    """The eight preregistered prompt files, in committed (sorted) order."""
    prereg = prereg or preregistration()
    block = prereg["prompt_candidates"]
    directory = ROOT / block["directory"]
    out = []
    for index, name in enumerate(sorted(block["sha256"])):
        path = directory / name
        out.append({"index": index, "id": name, "path": path,
                    "committed_sha256": block["sha256"][name],
                    "neutral": name == block["neutral_default"]})
    return out


def candidate_text(cand: dict) -> str:
    """The exact bytes the arm's system prompt is, stripped of trailing newline."""
    return cand["path"].read_text(encoding="utf-8").strip()


def verify_frozen(prereg: dict | None = None) -> list[dict]:
    """Raise unless every candidate file hashes back to the preregistration."""
    prereg = prereg or preregistration()
    block = prereg["prompt_candidates"]
    directory = ROOT / block["directory"]
    on_disk = sorted(p.name for p in directory.glob("*.txt"))
    if on_disk != sorted(block["sha256"]):
        raise SystemExit(
            f"{directory} holds {on_disk}, the preregistration pins "
            f"{sorted(block['sha256'])}. The candidate set is frozen: adding or "
            f"removing a prompt after preregistration invalidates the control.")
    cands = candidates(prereg)
    bad = [c["id"] for c in cands
           if file_sha256(str(c["path"])) != c["committed_sha256"]]
    if bad:
        raise SystemExit(
            f"prompt candidate files {bad} do not match their preregistered "
            f"SHA-256. Refusing to run: an edited candidate is a different "
            f"control, and the frozen gates describe the committed one.")
    return cands


def neutral_prompt(prereg: dict | None = None) -> str:
    """The neutral default system prompt (the B0/T0 arms), read from its file."""
    prereg = prereg or preregistration()
    return candidate_text(next(c for c in candidates(prereg) if c["neutral"]))


# The canonical neutral system prompt every non-frozen rollout uses. Read from
# the preregistered file rather than hardcoded, so there is exactly one copy of
# the neutral arm's bytes in the project.
CANONICAL_SYSTEM = neutral_prompt()


def frozen_file(cfg: dict | None = None) -> pathlib.Path:
    cfg = cfg or load_config()
    return ROOT / cfg["prompt_control"]["frozen_file"]


def frozen_winner(cfg: dict | None = None) -> dict:
    """The tournament winner: {candidate, sha256, prompt}. Refuses if unfrozen."""
    p = frozen_file(cfg)
    if not p.exists():
        raise SystemExit(
            f"{p} not found. The elicitation control must be frozen BEFORE any "
            f"production rollout: run the prompt tournament (`python -m "
            f"agentlab.prompt_control run/finalize`) first.")
    data = json.loads(p.read_text(encoding="utf-8"))
    cand = next(c for c in candidates() if c["id"] == data["winner"]["candidate"])
    if file_sha256(str(cand["path"])) != data["winner"]["sha256"]:
        raise SystemExit(f"{cand['path']} changed after the freeze; refusing to run.")
    return {"candidate": cand["id"], "sha256": data["winner"]["sha256"],
            "prompt": candidate_text(cand)}


# --------------------------------------------------------------------------
# the three claim axes over the committed dev split
# --------------------------------------------------------------------------

def _all_three_tools(bundle) -> bool:
    return {"kb_lookup", "unit_convert", "calculator"} <= {n.tool for n in bundle.nodes}


def axis_pool(bundles: list, axis: str) -> list:
    """Every dev bundle eligible for one claim axis, in committed order.

    The dev split assigns one fault to every spec (the paired counterfactual
    design), so the clean axes use each spec's paired clean arm via
    `TaskSpec.without_faults()` rather than a differently generated task.
    """
    if axis == "recovery":
        return [b for b in bundles if b.spec.faults]
    if axis == "orchestration":
        return [_clean(b) for b in bundles
                if b.spec.family == "typed_relay" and b.spec.horizon == 4
                and _all_three_tools(b)]
    if axis == "h8":
        return [_clean(b) for b in bundles
                if b.spec.horizon == 8 and b.spec.family in ("lookup_chain",
                                                             "typed_relay")]
    raise ValueError(f"unknown axis {axis!r}")


def _clean(bundle):
    import dataclasses

    return dataclasses.replace(bundle, spec=bundle.spec.without_faults())


def axis_bundles(bundles: list, axis: str, n: int, offset: int = 0) -> list:
    """`n` bundles for one axis, balanced across the axis's cells.

    Cells are drawn round-robin from their `offset` window, so when n is not a
    multiple of the cell count the shortfall is spread evenly instead of dropping
    the last cells entirely. Raises rather than quietly returning fewer: a
    tournament run on a smaller sample than the preregistration specifies is not
    the preregistered control.
    """
    pool = axis_pool(bundles, axis)
    cells = group_by_cell(pool)
    if not cells:
        raise ValueError(f"axis {axis!r} has no eligible dev instances")
    per_cell = -(-n // len(cells))
    windows = [cell_slice([b for b in pool
                           if (b.spec.family, b.spec.horizon) == cell],
                          per_cell, offset=offset)
               for cell in sorted(cells)]
    take = []
    for i in range(per_cell):
        for window in windows:
            if len(take) < n and i < len(window):
                take.append(window[i])
    if len(take) < n:
        raise ValueError(f"axis {axis!r} can supply {len(take)} of {n} instances")
    return take


def axis_capacity(cfg: dict | None = None) -> dict:
    """Can the committed dev split supply the preregistered per-axis samples?

    Reports, per axis, the pool size, the requirement, and the per-cell dev size
    that WOULD satisfy it. This is a preflight, not a warning to be ignored: the
    frozen sizes are 100 per axis for round 1 plus another 200 for round 2.
    """
    from agentlab.multidistill import load_split

    cfg = cfg or load_config()
    pc = cfg["prompt_control"]
    need = pc["round1_per_axis"] + pc["round2_per_axis"]
    bundles = load_split(cfg["suite"]["dev"], cfg)
    out = {}
    for axis in AXES:
        pool = axis_pool(bundles, axis)
        cells = len(group_by_cell(pool)) or 1
        out[axis] = {"pool": len(pool), "required": need,
                     "cells": cells,
                     "required_dev_per_cell": -(-need // cells),
                     "ok": len(pool) >= need}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


# --------------------------------------------------------------------------
# scoring and the preregistered winner rule
# --------------------------------------------------------------------------

def score_rollout(rec: dict) -> bool:
    """Tournament success = the strict verifier's verdict on the episode.

    The tournament scores exactly what the study scores. A looser "answer looks
    right" rule here would tune the control against a different target from the
    one the training claim is measured on.
    """
    return bool(rec["verdict"]["strict_success"])


def axis_rates(rows: list) -> dict:
    """candidate -> {axis: success rate, 'combined': equal-weight mean}."""
    by_candidate: dict = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    out = {}
    for cid, crows in by_candidate.items():
        rates = {}
        for axis in AXES:
            arows = [r for r in crows if r["axis"] == axis]
            if arows:
                rates[axis] = sum(r["success"] for r in arows) / len(arows)
        combined = (sum(rates.values()) / len(rates)) if rates else 0.0
        out[cid] = {**{a: round(rates.get(a, 0.0), 4) for a in AXES},
                    "combined": round(combined, 4),
                    "n": len(crows)}
    return out


def rank(rates: dict, cands: list) -> list[str]:
    """Preregistered ordering: highest combined, then shorter file, then index."""
    meta = {c["id"]: (c["path"].stat().st_size, c["index"]) for c in cands}
    return sorted(rates, key=lambda cid: (-rates[cid]["combined"], meta[cid][0],
                                          meta[cid][1]))


def pick_winner(rows: list, cfg: dict | None = None) -> dict:
    """Apply the frozen winner rule to round-1 + round-2 rows."""
    cfg = cfg or load_config()
    cands = candidates()
    rates = axis_rates(rows)
    order = rank(rates, cands)
    top2 = order[:2]
    # Round 2 exists to separate the top two; the decision is made on the
    # COMBINED 300 rows per axis, which is what `rows` already contains.
    winner = order[0]
    cand = next(c for c in cands if c["id"] == winner)
    h8_best = max((r["h8"] for r in rates.values()), default=0.0)
    h8_min = cfg["prompt_control"]["h8_feasibility_min"]
    return {
        "winner": {"candidate": winner, "sha256": file_sha256(str(cand["path"])),
                   "prompt_sha256": digest_text(candidate_text(cand)),
                   "rates": rates[winner]},
        "ranking": order, "round2_candidates": top2, "per_candidate": rates,
        "h8": {"best_prompt_success": round(h8_best, 4),
               "feasibility_min": h8_min,
               "measured_only": h8_best < h8_min},
        "honest_description": preregistration()["prompt_candidates"]["tournament"][
            "honest_description"],
    }


# --------------------------------------------------------------------------
# GPU tournament (one candidate/round per invocation; resumable by file)
# --------------------------------------------------------------------------

def _rows_path(round_no: int, candidate: str) -> pathlib.Path:
    return TOURNEY_DIR / f"r{round_no}-{candidate}.jsonl"


def tournament_rows(engine, bundles: list, axis: str, candidate: dict,
                    prompt: str) -> list[dict]:
    """One candidate over one axis sample: one attempt per task."""
    convos = engine.rollouts_for(bundles, k_override=1, variants=("frozen",))
    for convo in convos:
        convo["messages"][0]["content"] = prompt
    records = engine.run(convos, verbose=False)
    return [{"candidate": candidate["id"], "axis": axis, "task_id": r["task_id"],
             "family": r["family"], "horizon": r["horizon"],
             "fault_types": r["fault_types"], "success": score_rollout(r),
             "n_calls": r["verdict"]["calls"],
             "milestone_fraction": r["milestone_fraction"],
             "truncated": r["truncated"], "exhausted": r["exhausted"]}
            for r in records]


def cmd_run(args) -> None:
    from agentlab.multidistill import _vllm_engine, _write_jsonl, load_split

    cfg = load_config()
    cands = verify_frozen()
    pc = cfg["prompt_control"]
    cap = axis_capacity(cfg)
    if not cap["ok"]:
        raise SystemExit(
            "the committed dev split cannot supply the preregistered per-axis "
            f"samples: {json.dumps({a: cap[a] for a in AXES})}. Enlarge the dev "
            "split in configs/suite_v1.toml (it is not a frozen file) and "
            "regenerate, or file a dated AMENDMENT. Refusing to run a smaller "
            "tournament under the preregistered name.")
    cand = next((c for c in cands if c["id"] == args.candidate), None)
    if cand is None:
        raise SystemExit(f"unknown candidate {args.candidate!r}; "
                         f"choose from {[c['id'] for c in cands]}")
    out = _rows_path(args.round, cand["id"])
    if out.exists() and not args.force:
        print(f"[tournament] r{args.round} {cand['id']} already done")
        return
    ledger_guard("prompt_tournament", args.budget_minutes, cfg)

    n = pc["round1_per_axis"] if args.round == 1 else pc["round2_per_axis"]
    offset = 0 if args.round == 1 else pc["round1_per_axis"]
    bundles = load_split(cfg["suite"]["dev"], cfg)
    prompt = candidate_text(cand)

    t0 = time.time()
    engine = _vllm_engine(cfg, args, frozen=None)
    rows = []
    for axis in AXES:
        rows += tournament_rows(engine, axis_bundles(bundles, axis, n, offset),
                                axis, cand, prompt)
    _write_jsonl(out, rows)
    minutes = (time.time() - t0) / 60.0
    cumulative = ledger_append("prompt_tournament", minutes, cfg)
    rates = axis_rates(rows)[cand["id"]]
    print(f"[tournament] r{args.round} {cand['id']}: combined {rates['combined']:.3f} "
          f"over {len(rows)} dev tasks in {minutes:.1f} min (ledger {cumulative:.2f}h)")


def cmd_finalize(args) -> None:
    from agentlab.multidistill import _read_jsonl

    cfg = load_config()
    cands = verify_frozen()
    rows = []
    missing = []
    for cand in cands:
        p = _rows_path(1, cand["id"])
        if not p.exists():
            missing.append(f"r1-{cand['id']}")
            continue
        rows += _read_jsonl(p)
    if missing:
        raise SystemExit(f"missing round-1 tournament files: {missing}")

    round1 = pick_winner(rows, cfg)
    for cid in round1["round2_candidates"]:
        p = _rows_path(2, cid)
        if not p.exists():
            raise SystemExit(
                f"round 2 is preregistered for the top two candidates "
                f"({round1['round2_candidates']}); missing r2-{cid}. Run it "
                f"before finalizing.")
        rows += _read_jsonl(p)

    verdict = pick_winner(rows, cfg)
    verdict["round1_ranking"] = round1["ranking"]
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    fp = frozen_file(cfg)
    fp.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"[tournament] frozen selection -> {fp}")
    if verdict["h8"]["measured_only"]:
        print("[tournament] H8 best-prompt success below the feasibility floor: "
              "the H8 stratum is MEASURED-ONLY; self-distillation cannot "
              "bootstrap a stratum with no successful trajectories.")


def cmd_verify(args) -> None:
    cands = verify_frozen()
    for cand in cands:
        mark = "  (neutral default)" if cand["neutral"] else ""
        print(f"  {cand['id']:28s} {cand['committed_sha256'][:16]}{mark}")
    print(f"[verify] {len(cands)} preregistered prompt files match their hashes")


def cmd_axes(args) -> None:
    cfg = load_config()
    cap = axis_capacity(cfg)
    for axis in AXES:
        row = cap[axis]
        status = "OK" if row["ok"] else "SHORT"
        print(f"  {axis:14s} pool {row['pool']:5d}  required {row['required']:5d}  "
              f"[{status}] needs dev per_cell >= {row['required_dev_per_cell']}")
    print(f"[axes] preregistered sizes satisfiable: {cap['ok']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("axes")
    run = sub.add_parser("run")
    run.add_argument("--candidate", required=True,
                     help="prompt file name, e.g. p4_error_repair.txt")
    run.add_argument("--round", type=int, default=1, choices=(1, 2))
    run.add_argument("--force", action="store_true")
    run.add_argument("--model", default=None)
    run.add_argument("--gpu-frac", type=float, default=0.85)
    run.add_argument("--max-model-len", type=int, default=8192)
    run.add_argument("--enforce-eager", action="store_true")
    run.add_argument("--budget-minutes", type=float, default=8.0)
    sub.add_parser("finalize")
    args = ap.parse_args()
    if args.cmd == "run":
        from agentlab import env as labenv

        args.model = args.model or labenv.MODEL
    {"verify": cmd_verify, "axes": cmd_axes, "run": cmd_run,
     "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
