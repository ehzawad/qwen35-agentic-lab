"""The GRPO grip probe: does any within-group signal exist to optimise?

GRPO cannot learn from a group whose members all score the same. The previous
run's null was exactly that -- 56% of training groups had zero reward variance --
so this probe runs BEFORE any GRPO and reports the four preregistered
eligibility quantities per family and per cell:

  nonzero_reward_sd_frac        fraction of groups with nonzero total-reward SD
  terminal_disagreement_frac    fraction of groups where terminal SUCCESS itself
                                disagrees (not merely shaping-reward spread)
  mean_success                  mean terminal success over all generations
  clip_frac                     fraction of generations clipped by the
                                completion budget

The distinction between the first two is the whole point. A group can have
nonzero total-reward SD purely because two failures reached different milestone
counts; that is shaping spread, not a learning signal about the outcome. The
gate therefore requires BOTH, exactly as configs/multifaceted.yaml states, and
`gate_report` computes every threshold from this module's own output alone:

  min_nonzero_reward_sd_frac        >= 0.60
  min_terminal_disagreement_frac    >= 0.40
  mean_success_range                within [0.15, 0.85]
  max_clip_frac                     <= 0.05

Design (council: "Probe 48 disjoint prompt groups per family, eight generations
per group"):

  * 48 groups per family = 12 per cell x 4 cells, taken from the front of the
    committed GRPO pool split; the GRPO training pool starts after them, so the
    probe and the training pool never share a task.
  * one group = one task spec repeated `generations_per_group` times, so every
    member of a group faces the byte-identical task, KB, fault schedule and
    logical clock. That is what makes the within-group comparison valid, and it
    is why the probe rebuilds each generation from the committed spec through
    `EpisodeRuntime` rather than from a procedural reset().
  * the reward is the preregistered lexicographic reward from
    configs/multifaceted.yaml, clamped as a TOTAL; terminal success is the
    strict verifier's `certified_success` and nothing else.

Sharding: ONE engine serves every pending cell (12 groups x 8 generations = 96
rollouts each), each cell writing results/agentic/variance/<split>-<cell>.jsonl,
so a killed run resumes by re-invoking and no cell pays the measured 289.7 s
engine startup of its own. `report` is CPU-only and needs no GPU at all.

THE POLICY THIS PROBES. The GRPO decision is about the RS-SFT checkpoint's group
variance, so the engine must SERVE THE RS-SFT ADAPTER. An earlier version checked
that the adapter existed and then built the engine from the base model alone,
which probes the wrong policy and would have reported base-model variance under
the trained policy's name. The adapter is now required and loaded.

STAGE DISPOSITION (v1). GRPO cannot instantiate on the registered A5000
(configs/multifaceted.yaml grpo.stage_disposition), and the probe exists only to
decide whether to spend GRPO hours. It is therefore short-circuited BEFORE it
runs and recorded as NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT -- never as "closed",
which would claim the complete 144-group probe ran and a binding gate failed.
`run` refuses while that label stands, rather than silently probing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

from agentlab.suite.configio import (ROOT, ledger_append, ledger_guard, load_config,
                                     now_utc)
from agentlab.suite.generate import cell_slice, group_by_cell

OUT_DIR = ROOT / "results" / "agentic" / "variance"
REPORT_PATH = ROOT / "results" / "agentic" / "variance_report.json"


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

def lexicographic_reward(rec: dict, cfg: dict | None = None) -> float:
    """The preregistered GRPO reward for one rollout, clamped as a total.

    Every term is read off the strict verifier's verdict and the environment-side
    trace -- never off model text:

      terminal_success             verdict.certified_success
      verified_milestone_fraction  unique dependency-valid oracle nodes / H
      required_recovery_success    verdict.recovery_success (fault arms only)
      excess_calls                 max(0, calls - H)
      illegal_state_action         verdict.unsafe_mutation
      no_commit                    the episode never produced a committed answer

    No failed trajectory can outscore a valid success. The success term alone is
    1.0 while every positive partial term sums to at most 0.30, and the only term
    that can pull a success DOWN is the excess-call penalty. That penalty is
    bounded because the verifier refuses success past `call_budget(H) = 2H+4`:
    the worst reachable success scores 1.0 - 0.02*(H+4), i.e. 0.88 at H2 and 0.52
    at H20 -- both above the 0.30 ceiling for any failure. The property therefore
    holds for every cell in this suite and is tested at that boundary; it would
    stop holding above H46, so a future deeper cell must re-check it.
    """
    cfg = cfg or load_config()
    w = cfg["grpo"]["reward"]["weights"]
    lo, hi = cfg["grpo"]["reward"]["clamp"]
    v = rec["verdict"]
    horizon = max(int(rec["horizon"]), 1)
    excess = max(0, int(v["calls"]) - horizon)
    total = (w["terminal_success"] * (1.0 if v["certified_success"] else 0.0)
             + w["verified_milestone_fraction"] * float(rec["milestone_fraction"])
             + w["required_recovery_success"] * (1.0 if v["recovery_success"] else 0.0)
             + w["excess_calls"] * excess
             + w["illegal_state_action"] * (1.0 if v["unsafe_mutation"] else 0.0)
             + w["no_commit"] * (0.0 if rec["final"] else 1.0))
    return max(lo, min(hi, total))


def is_clipped(rec: dict) -> bool:
    """Clipped = the completion budget, not the task, ended the rollout.

    Both forms count: a truncated final decision (finish_reason == "length") and
    an episode that spent its whole decision budget without committing an answer.
    TRL does not classify the second as clipped, and the council's guard requires
    it to be counted, so it is counted here.
    """
    return bool(rec["truncated"]) or bool(rec["exhausted"])


# ---------------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------------

def _sd(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def group_stats(rows: list, cfg: dict | None = None) -> dict:
    """One group (all generations of ONE task) -> its variance evidence."""
    cfg = cfg or load_config()
    rewards = [lexicographic_reward(r, cfg) for r in rows]
    successes = [bool(r["verdict"]["certified_success"]) for r in rows]
    sd = _sd(rewards)
    return {
        "task_id": rows[0]["task_id"], "family": rows[0]["family"],
        "horizon": rows[0]["horizon"],
        "cell": f"{rows[0]['family']}-h{rows[0]['horizon']}",
        "fault_types": list(rows[0].get("fault_types") or []),
        "n": len(rows),
        "reward_sd": round(sd, 6),
        "nonzero_reward_sd": sd > 0.0,
        # Terminal DISAGREEMENT: both a success and a failure inside the group.
        # Shaping spread alone never satisfies this.
        "terminal_disagreement": any(successes) and not all(successes),
        "n_success": sum(successes),
        "mean_success": round(sum(successes) / len(rows), 6),
        "mean_reward": round(sum(rewards) / len(rows), 6),
        "milestone_sd": round(_sd([float(r["milestone_fraction"]) for r in rows]), 6),
        "n_clipped": sum(1 for r in rows if is_clipped(r)),
        "unsafe": any(r["verdict"]["unsafe_mutation"] for r in rows),
        "inconsistent": any(not r["verdict"]["consistent"] for r in rows),
        "replay_failures": sum(1 for r in rows if r.get("replay_ok") is False),
    }


def _aggregate(groups: list, generations: int) -> dict:
    n = len(groups)
    total_gen = sum(g["n"] for g in groups)
    return {
        "n_groups": n,
        "n_generations": total_gen,
        "generations_per_group": generations,
        "nonzero_reward_sd_frac": (sum(g["nonzero_reward_sd"] for g in groups) / n
                                   if n else 0.0),
        "terminal_disagreement_frac": (sum(g["terminal_disagreement"] for g in groups) / n
                                       if n else 0.0),
        "mean_success": (sum(g["n_success"] for g in groups) / total_gen
                         if total_gen else 0.0),
        "clip_frac": (sum(g["n_clipped"] for g in groups) / total_gen
                      if total_gen else 0.0),
        "shaping_only_frac": (sum(1 for g in groups if g["nonzero_reward_sd"]
                                  and not g["terminal_disagreement"]) / n
                              if n else 0.0),
        "unsafe_groups": sum(1 for g in groups if g["unsafe"]),
        "inconsistent_groups": sum(1 for g in groups if g["inconsistent"]),
        "replay_failures": sum(g["replay_failures"] for g in groups),
    }


def summarize(rows: list, cfg: dict | None = None) -> dict:
    """All probe rollouts -> per-cell, per-family and overall variance evidence."""
    cfg = cfg or load_config()
    generations = cfg["variance_probe"]["generations_per_group"]
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    groups = [group_stats(g, cfg) for _tid, g in sorted(by_task.items())]

    per_cell: dict = {}
    for cell in sorted({g["cell"] for g in groups}):
        per_cell[cell] = _aggregate([g for g in groups if g["cell"] == cell],
                                    generations)
    per_family: dict = {}
    for fam in sorted({g["family"] for g in groups}):
        per_family[fam] = _aggregate([g for g in groups if g["family"] == fam],
                                      generations)
    return {"overall": _aggregate(groups, generations),
            "per_family": per_family, "per_cell": per_cell,
            "groups": groups}


# ---------------------------------------------------------------------------
# eligibility gates (computed from the summary alone)
# ---------------------------------------------------------------------------

def gate_report(summary: dict, cfg: dict | None = None) -> dict:
    """Apply the configs/multifaceted.yaml eligibility gates to a summary.

    Every threshold is read from the config and every measured quantity from the
    summary: nothing else is consulted, so a summary file is a complete input for
    the GRPO go/no-go decision.
    """
    cfg = cfg or load_config()
    gates = cfg["variance_probe"]["gates"]
    lo, hi = gates["mean_success_range"]

    def one(scope: str, name: str, agg: dict) -> dict:
        checks = {
            "nonzero_reward_sd": {
                "value": round(agg["nonzero_reward_sd_frac"], 4),
                "threshold": gates["min_nonzero_reward_sd_frac"],
                "ok": agg["nonzero_reward_sd_frac"] >= gates["min_nonzero_reward_sd_frac"]},
            "terminal_disagreement": {
                "value": round(agg["terminal_disagreement_frac"], 4),
                "threshold": gates["min_terminal_disagreement_frac"],
                "ok": agg["terminal_disagreement_frac"] >= gates["min_terminal_disagreement_frac"]},
            "mean_success": {
                "value": round(agg["mean_success"], 4), "range": [lo, hi],
                "ok": lo <= agg["mean_success"] <= hi},
            "clip_frac": {
                "value": round(agg["clip_frac"], 4),
                "threshold": gates["max_clip_frac"],
                "ok": agg["clip_frac"] <= gates["max_clip_frac"]},
            "no_reward_leakage": {
                "unsafe_groups": agg["unsafe_groups"],
                "inconsistent_groups": agg["inconsistent_groups"],
                "replay_failures": agg["replay_failures"],
                "ok": (agg["unsafe_groups"] == 0
                       and agg["inconsistent_groups"] == 0
                       and agg["replay_failures"] == 0)},
        }
        return {"scope": scope, "name": name,
                "n_groups": agg["n_groups"], "checks": checks,
                "eligible": all(c["ok"] for c in checks.values())}

    per_family = {fam: one("family", fam, agg)
                  for fam, agg in summary["per_family"].items()}
    per_cell = {cell: one("cell", cell, agg)
                for cell, agg in summary["per_cell"].items()}
    eligible = sorted(c for c, v in per_cell.items() if v["eligible"])
    return {
        "overall": one("overall", "overall", summary["overall"]),
        "per_family": per_family, "per_cell": per_cell,
        "eligible_cells": eligible,
        # A zero-variance cell is skipped, not run ceremonially: GRPO on a cell
        # where every group scores identically cannot move the policy.
        "skipped_cells": sorted(c for c in per_cell if c not in eligible),
        "grpo_enabled": bool(eligible),
    }


# ---------------------------------------------------------------------------
# probe groups
# ---------------------------------------------------------------------------

def probe_bundles(cfg: dict | None = None, cell: tuple | None = None) -> list:
    """The committed probe tasks: `groups_per_cell` from the front of each cell."""
    from agentlab.multidistill import load_split

    cfg = cfg or load_config()
    vp = cfg["variance_probe"]
    bundles = load_split(vp["source_split"], cfg)
    probe = cell_slice(bundles, vp["groups_per_cell"])
    if cell is None:
        return probe
    return [b for b in probe if (b.spec.family, b.spec.horizon) == cell]


def cells_in_probe(cfg: dict | None = None) -> list:
    return sorted(group_by_cell(probe_bundles(cfg)))


def _cell_path(split: str, cell: tuple) -> pathlib.Path:
    return OUT_DIR / f"{split}-{cell[0]}-h{cell[1]}.jsonl"


def probe_records(engine, bundles: list, generations: int) -> list:
    """Roll `generations` disjoint generations of each task through the engine.

    The group members are built from ONE committed spec, so `reset` equivalence
    is structural: all eight generations run the identical KB, fault schedule and
    decision clock. Each record is additionally replay-checked, because a group
    whose members did not run the same task would produce meaningless variance.
    """
    from agentlab.multidistill import replay_record

    convos = engine.rollouts_for(bundles, k_override=generations,
                                 variants=("canonical",))
    records = engine.run(convos)
    by_id = {b.spec.task_id: b for b in bundles}
    for rec in records:
        # THE ENGINE's secret, not a fresh one: the recovery tokens and receipts
        # the model saw were keyed with it, so replaying under another secret
        # would fail every receipt and label a valid group unreplayable.
        ok, why = replay_record(rec, by_id[rec["task_id"]], secret=engine.secret)
        rec["replay_ok"] = ok
        rec["replay_reason"] = why
    return records


def _slim(rec: dict) -> dict:
    """Probe rows keep the scoring surface, not the transcripts.

    They also keep the HARDWARE PROVENANCE. Dropping the transcripts is a size
    decision; dropping which card and which engine produced the row would make
    the probe unusable as evidence, and S19 requires the fingerprint on every
    claim-bearing row rather than only in the report footer.
    """
    slim = {k: rec[k] for k in ("task_id", "family", "horizon", "split",
                                "fault_types", "sample_index", "prompt_variant",
                                "final", "decisions_used", "truncated",
                                "exhausted", "unknown_tool", "arg_error",
                                "call_cap", "milestone_fraction", "verdict",
                                "replay_ok", "replay_reason")}
    slim["provenance"] = rec.get("provenance")
    return slim


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_plan(args) -> None:
    cfg = load_config()
    vp = cfg["variance_probe"]
    cells = cells_in_probe(cfg)
    for cell in cells:
        path = _cell_path(vp["source_split"], cell)
        mark = "done" if path.exists() else "    "
        print(f"  {cell[0]:13s} h{cell[1]:<3d} {vp['groups_per_cell']:3d} groups "
              f"x {vp['generations_per_group']} generations  {mark}")
    total = len(cells) * vp["groups_per_cell"] * vp["generations_per_group"]
    print(f"[probe] {len(cells)} cells, {vp['groups_per_family']} groups per family, "
          f"{total} rollouts total, split={vp['source_split']}")


DISPOSITION_SHORT_CIRCUIT = "NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT"


def registered_disposition(cfg: dict | None = None) -> str | None:
    cfg = cfg or load_config()
    return (cfg.get("variance_probe") or {}).get("stage_disposition")


def refuse_if_short_circuited(cfg: dict | None = None) -> None:
    """Hard-refuse the probe while its registered disposition stands.

    The probe's ONLY purpose is to decide whether GRPO hours are worth spending.
    GRPO cannot instantiate on the registered card, so the probe is not evaluated
    at all -- and running it anyway would produce a number that looks like the
    registered probe result and would invite reporting the GRPO branch as
    "variance gate closed", which is a DIFFERENT and unearned claim.
    """
    disposition = registered_disposition(cfg)
    if disposition and disposition != "RUN":
        raise SystemExit(
            f"REFUSED: the variance probe's registered stage disposition is "
            f"{disposition}.\n"
            f"  GRPO is not run on this card (grpo.stage_disposition = "
            f"GRPO_NOT_RUN_HARDWARE_INFEASIBLE), so the probe that exists to "
            f"decide GRPO is NOT EVALUATED -- not 'closed'.\n"
            f"  Recording a probe result now would make the GRPO branch look "
            f"like it lost a variance gate it never faced. Running the probe is a "
            f"NEW registered decision: set variance_probe.stage_disposition: RUN "
            f"in configs/multifaceted.yaml as a dated amendment.")


def probe_adapter(args, cfg: dict | None = None) -> str:
    """The RS-SFT adapter the probe MUST serve, or a refusal.

    `--adapter` may name it explicitly; otherwise the locked checkpoint is used,
    and failing that the conventional RS-SFT output directory. Falling back to the
    base model is not an option: that measures a policy the GRPO decision is not
    about.
    """
    cfg = cfg or load_config()
    locked_digest = None
    if getattr(args, "adapter", None):
        candidate = pathlib.Path(args.adapter)
    else:
        locks = ROOT / "results" / "agentic" / "locks.json"
        candidate = None
        if locks.exists():
            rec = json.loads(locks.read_text(encoding="utf-8")).get("checkpoint") or {}
            if rec.get("path"):
                candidate = ROOT / rec["path"]
                locked_digest = rec.get("checkpoint_sha256")
        candidate = candidate or ROOT / "out" / "multiface" / "rssft-lora"
    if locked_digest:
        # The lock pins BYTES. Probing a directory whose contents no longer hash
        # to the locked digest would report the variance of a different policy
        # under the locked checkpoint's name.
        from agentlab.suite.configio import checkpoint_tree_sha256

        got = checkpoint_tree_sha256(candidate)
        if got != locked_digest:
            raise SystemExit(
                f"REFUSED: {candidate} hashes {got} and the checkpoint lock pins "
                f"{locked_digest}. The locked adapter's bytes changed, so probing "
                f"it would measure a policy the lock does not describe.")
    if not (candidate / "adapter_model.safetensors").exists():
        raise SystemExit(
            f"REFUSED: no RS-SFT adapter at {candidate}. The probe measures the "
            f"RS-SFT policy's within-group variance; building the engine from the "
            f"BASE model would probe the wrong policy and report it under the "
            f"trained policy's name. Train the adapter first or pass --adapter.")
    return str(candidate)


def cmd_run(args) -> None:
    cfg = load_config()
    refuse_if_short_circuited(cfg)
    vp = cfg["variance_probe"]
    split = vp["source_split"]
    cells = cells_in_probe(cfg)
    if args.family is not None or args.horizon is not None:
        if args.family is None or args.horizon is None:
            raise SystemExit("pass BOTH --family and --horizon, or neither")
        cell = (args.family, args.horizon)
        if cell not in cells:
            raise SystemExit(f"{cell} is not a probe cell; see `plan`")
        if _cell_path(split, cell).exists() and not args.force:
            print(f"[probe] {cell} already done (use --force to redo)")
            return
        units, forced = [cell], ({cell} if args.force else set())
    else:
        units, forced = cells, set()
    pending = [c for c in units if c in forced or not _cell_path(split, c).exists()]
    if not pending:
        print("[probe] all cells done")
        return
    ledger_guard("variance_probe", args.budget_minutes, cfg)

    from agentlab.multidistill import (_vllm_engine, run_units,
                                       write_attested_jsonl)

    adapter = probe_adapter(args, cfg)
    args.stage = "variance_probe"
    started = now_utc()
    t_engine = time.time()
    # The RS-SFT adapter, not the base model: this probes the trained policy.
    engine = _vllm_engine(cfg, args, frozen=None, adapter=adapter)
    startup_min = (time.time() - t_engine) / 60.0
    ledger_append("variance_probe:engine_start", startup_min, cfg,
                  kind="engine_start", started_at=started,
                  manifest=engine.manifest_path,
                  work={"unit": "engine", "count": 1, "adapter": adapter,
                        "cells_pending": len(pending)})
    print(f"[probe] engine up in {startup_min:.1f} min with adapter {adapter}; "
          f"it serves all {len(pending)} pending cells")

    def is_done(cell):
        return cell not in forced and _cell_path(split, cell).exists()

    def run_one(cell):
        bundles = probe_bundles(cfg, cell)
        t0 = time.time()
        records = probe_records(engine, bundles, vp["generations_per_group"])
        rows = [_slim(r) for r in records]
        write_attested_jsonl(_cell_path(split, cell), rows,
                             f"variance probe cell {cell[0]}-h{cell[1]}")
        forced.discard(cell)
        minutes = (time.time() - t0) / 60.0
        cumulative = ledger_append("variance_probe", minutes, cfg, kind="cell",
                                   manifest=engine.manifest_path,
                                   work={"unit": "rollouts", "count": len(records),
                                         "cell": f"{cell[0]}-h{cell[1]}",
                                         "adapter": adapter})
        agg = summarize(rows, cfg)["overall"]
        print(f"[probe] {cell[0]}-h{cell[1]}: {len(records)} rollouts in "
              f"{minutes:.1f} min (ledger {cumulative:.2f}h)")
        print(f"[probe] reward-SD {agg['nonzero_reward_sd_frac']:.2f}  "
              f"terminal-disagreement {agg['terminal_disagreement_frac']:.2f}  "
              f"mean-success {agg['mean_success']:.2f}  clip {agg['clip_frac']:.3f}")

    status = run_units(engine, units, run_one=run_one, is_done=is_done,
                       budget_minutes=args.budget_minutes, label="probe",
                       cfg=cfg, work_unit="cell")
    print(json.dumps(status))


def load_probe_rows(cfg: dict | None = None) -> list:
    cfg = cfg or load_config()
    split = cfg["variance_probe"]["source_split"]
    rows = []
    for path in sorted(OUT_DIR.glob(f"{split}-*.jsonl")):
        rows += [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def write_disposition_report(cfg: dict | None = None) -> dict:
    """Record NOT_EVALUATED rather than leaving the probe artifact missing.

    A missing artifact is indistinguishable from "we forgot"; an artifact that
    says NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT says exactly what happened. `gates`
    and `summary` are explicitly null: there is no probe measurement to report,
    and inventing an empty-but-shaped gate block would let a downstream reader
    treat "not evaluated" as "evaluated and failed".

    Its provenance says `gpu_execution: false` and leaves the card identity NULL.
    Nothing here ran on a card, and reading the run's hardware lock would have
    made a short-circuited stage look like a measured one merely because an
    earlier stage bound a GPU.
    """
    cfg = cfg or load_config()
    from agentlab.multidistill import cpu_provenance

    out = {"stage_disposition": registered_disposition(cfg),
           "probe_run": False, "complete": False,
           "cells_done": 0, "cells_expected": len(cells_in_probe(cfg)),
           "rollouts": 0, "summary": None, "gates": None,
           "reason": "GRPO is not run on the registered card "
                     "(grpo.stage_disposition = GRPO_NOT_RUN_HARDWARE_INFEASIBLE), "
                     "so the probe that exists to decide GRPO was never "
                     "evaluated. This is NOT the same as a closed variance gate.",
           "provenance": cpu_provenance("variance-probe-disposition", cfg)}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("stage_disposition", "probe_run",
                                          "cells_done", "cells_expected")}, indent=2))
    print(f"[probe] disposition recorded -> {REPORT_PATH}")
    return out


def cmd_report(args) -> None:
    cfg = load_config()
    rows = load_probe_rows(cfg)
    if not rows:
        if registered_disposition(cfg) not in (None, "RUN"):
            write_disposition_report(cfg)
            return
        raise SystemExit(f"no probe rows under {OUT_DIR}; run the probe cells first")
    summary = summarize(rows, cfg)
    gates = gate_report(summary, cfg)
    done = len(summary["per_cell"])
    expected = len(cells_in_probe(cfg))
    from agentlab.multidistill import (provenance_gaps, require_one_producer,
                                       require_row_provenance)

    # The hardware provenance of the probe ROWS, not of this reporting process: a
    # report that cannot say which card measured it is not evidence, and a report
    # that substitutes the reporting process's own fingerprint for the missing one
    # is worse -- it looks measured. Missing or mixed provenance stops the report.
    row_fps = [r.get("provenance") for r in rows if not provenance_gaps(
        r.get("provenance"))]
    if len(row_fps) != len(rows):
        require_row_provenance(
            next(r.get("provenance") for r in rows if provenance_gaps(
                r.get("provenance"))),
            f"{len(rows) - len(row_fps)} of {len(rows)} probe rows under {OUT_DIR}")
    producer = require_one_producer(rows, f"the probe rows under {OUT_DIR}")
    out = {"stage_disposition": "RUN", "probe_run": True,
           "complete": done == expected, "cells_done": done,
           "cells_expected": expected, "rollouts": len(rows),
           "summary": summary, "gates": gates,
           "provenance": row_fps[0] if row_fps else None,
           "source_provenance": producer,
           "rows_missing_provenance": len(rows) - len(row_fps)}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("complete", "cells_done",
                                          "cells_expected")}, indent=2))
    print(json.dumps(gates["overall"], indent=2))
    print(f"[probe] eligible cells: {gates['eligible_cells'] or 'NONE'}")
    print(f"[probe] skipped cells:  {gates['skipped_cells'] or 'none'}")
    if not out["complete"]:
        print("[probe] PARTIAL: the GRPO decision needs every probe cell.")
    elif not gates["grpo_enabled"]:
        print("[probe] NO CELL IS ELIGIBLE: GRPO must be skipped, not run "
              "ceremonially. That is a reportable result, not a failure.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")

    run = sub.add_parser("run")
    run.add_argument("--family", default=None)
    run.add_argument("--horizon", type=int, default=None)
    run.add_argument("--auto", action="store_true",
                     help="accepted for compatibility; serving every pending "
                          "cell from one engine is now the default")
    run.add_argument("--force", action="store_true")
    run.add_argument("--model", default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--adapter", default=None,
                     help="RS-SFT adapter to probe; defaults to the locked "
                          "checkpoint. The probe never runs on the base model.")
    # No engine knobs: configs/multifaceted.yaml `engine:` is the contract.
    run.add_argument("--budget-minutes", type=float, default=55.0)

    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run":
        from agentlab import env as labenv

        args.model = args.model or labenv.MODEL
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
