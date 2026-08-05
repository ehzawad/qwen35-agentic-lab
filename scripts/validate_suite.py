#!/usr/bin/env python
"""Validate suite v1 against the eleven binding failure conditions.

The validator FAILS (exit 1) on any of:

   1. non-byte-identical regeneration;
   2. any key, terminal token, task ID, template, or whole-task CONTENT HASH
      crossing splits (and no duplicate task content inside a split group);
   3. incorrect declared horizon;
   4. an oracle trajectory that does not execute successfully;
   5. a fault that fires more than once;
   6. a malformed result leaking its canonical value;
   7. non-idempotent replay after an ambiguous mutation;
   8. a successful verifier result with a missing oracle node;
   9. a same-decision dependency being credited;
  10. unknown-key errors exposing the KB key list;
  11. evaluation data appearing in SFT or GRPO inputs.

CPU-only. Usage:

    PYTHONPATH=src .venv/bin/python scripts/validate_suite.py \
        [--config configs/suite_v1.toml] [--data data/suite/v1]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

EXPECTED_TOTALS = {"oracle_sft": 4800, "distill": 2400, "grpo_train": 2400,
                   "dev": 240, "eval": 1200, "eval_stress": 280}
TRAIN_SPLITS = ("oracle_sft", "distill", "grpo_train")
EVAL_SPLITS = ("eval", "eval_stress")


def _fresh_runtime(bundle):
    from agentlab.suite.runtime import EpisodeRuntime

    return EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes)


def check_regeneration(cfg, data_dir) -> list[str]:
    from agentlab.suite.generate import generate_all

    sums_path = os.path.join(data_dir, "SHA256SUMS")
    if not os.path.exists(sums_path):
        return [f"missing {sums_path}; run scripts/generate_suite.py first"]
    with open(sums_path, encoding="utf-8") as fh:
        committed = fh.read()
    with tempfile.TemporaryDirectory(prefix="suite-regen-") as tmp:
        generate_all(cfg, tmp)
        with open(os.path.join(tmp, "SHA256SUMS"), encoding="utf-8") as fh:
            regenerated = fh.read()
    if committed != regenerated:
        return ["regeneration is not byte-identical to the committed artifacts"]
    return []


def check_sizes(splits_data, cfg) -> list[str]:
    problems = []
    for split, bundles in splits_data.items():
        cells = {}
        for b in bundles:
            cells.setdefault((b.spec.family, b.spec.horizon), 0)
            cells[(b.spec.family, b.spec.horizon)] += 1
        expected_cell = cfg["sizes"][split]
        for cell, n in cells.items():
            if n != expected_cell:
                problems.append(f"{split} cell {cell}: {n} != {expected_cell}")
        default_total = EXPECTED_TOTALS.get(split)
        if default_total is not None and cfg["sizes"][split] * len(cells) != default_total:
            problems.append(f"{split}: total {cfg['sizes'][split] * len(cells)} "
                            f"!= committed {default_total}")
    return problems


def check_isolation(splits_data) -> list[str]:
    """Checks 2 and 11: nothing crosses the train/dev/eval namespace groups."""
    from agentlab.suite.splits import check_split_leakage

    groups: dict[str, list[dict]] = {"train": [], "dev": [], "eval": []}
    # The binding condition is "no KEY, TERMINAL TOKEN, task ID or template
    # crosses splits". Terminal tokens are the 128-bit codes and the fulfillment
    # completion tokens: a single crossing would mean a memorisable secret, so
    # zero is required, and a collision inside one group is a generator bug.
    #
    # Numeric answers are deliberately NOT part of this condition. A typed_relay
    # answer is a bounded integer, so identical values recur across splits by the
    # birthday paradox (measured: 53 of 2,554 distinct integers between train and
    # eval, with zero token crossings). Treating those as leakage would be a false
    # alarm that hides real ones: knowing that some other task also answered 4712
    # is worthless without that task's keys and template, both of which ARE
    # checked as disjoint, and strict success additionally requires the full
    # verified trace.
    tokens: dict[str, list] = {"train": [], "dev": [], "eval": []}
    task_ids: dict[str, set] = {"train": set(), "dev": set(), "eval": set()}
    for split, bundles in splits_data.items():
        group = ("train" if split in TRAIN_SPLITS
                 else "dev" if split == "dev" else "eval")
        for b in bundles:
            groups[group].append({"task_id": b.spec.task_id,
                                  "template_id": b.spec.template_id,
                                  "kb": b.kb})
            if b.spec.answer_kind == "token":
                tokens[group].append(b.spec.answer)
            task_ids[group].add(b.spec.task_id)

    problems = [f"leakage[{v['kind']}] between {v['splits']}: {v['count']} "
                f"(e.g. {v['examples'][:2]})"
                for v in check_split_leakage(groups)]
    names = sorted(groups)
    for group in names:
        if len(set(tokens[group])) != len(tokens[group]):
            problems.append(f"{group}: duplicate terminal tokens within the group")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = set(tokens[a]) & set(tokens[b])
            if inter:
                problems.append(f"terminal tokens cross {a}/{b}: {len(inter)}")
    total_ids = sum(len(s) for s in task_ids.values())
    if len(set().union(*task_ids.values())) != total_ids:
        problems.append("task IDs are not globally unique")

    from agentlab.suite.schema import TEMPLATE_RANGES
    for group, rows in groups.items():
        allowed = set(TEMPLATE_RANGES[group])
        bad = {r["template_id"] for r in rows} - allowed
        if bad:
            problems.append(f"{group} uses out-of-range templates {sorted(bad)}")
    return problems


def check_horizons(splits_data) -> list[str]:
    return [f"{b.spec.task_id}: declared horizon {b.spec.horizon} != "
            f"{len(b.nodes)} oracle nodes"
            for bundles in splits_data.values() for b in bundles
            if len(b.nodes) != b.spec.horizon]


def check_oracle_execution(splits_data) -> list[str]:
    """Checks 4, 5, 6, 7 over every spec (both arms for dev/eval/stress)."""
    from agentlab.suite.faults import MALFORMED_LITERAL
    from agentlab.suite.runtime import run_oracle
    from agentlab.suite.schema import digest_text

    literal_digest = digest_text(MALFORMED_LITERAL)
    problems: list[str] = []

    def run_arm(bundle, spec, label: str) -> None:
        # A malformed spec used to raise out of the whole validator, so ONE bad
        # spec aborted every remaining check and you learned about one defect
        # instead of all of them. An exception is a reported problem.
        try:
            rt, verdict = run_oracle(spec, bundle.kb, bundle.nodes)
        except Exception as exc:  # noqa: BLE001 -- any failure is a spec defect
            problems.append(f"{spec.task_id}[{label}]: oracle replay raised "
                            f"{type(exc).__name__}: {exc}")
            return
        if not verdict.strict_success:
            problems.append(f"{spec.task_id}[{label}]: oracle trajectory failed "
                            f"strict verification: {verdict.reasons[:2]}")
            # deliberately NOT returning: checks 5-7 read the event log and the
            # fire counts, which exist either way. Returning here made check 4
            # SHADOW checks 5/6/7 -- a fault that fires twice reported only as
            # "trajectory failed", so the diagnosis pointed at the wrong defect.
        # 5: every scheduled fault fired exactly once
        for key, count in verdict.fault_fire_counts.items():
            if count != 1:
                problems.append(f"{spec.task_id}[{label}]: fault {key} fired "
                                f"{count} times")
        # 6: malformed exposure is exactly the truncated literal
        for f in spec.faults:
            if f.fault_type != "malformed":
                continue
            fire = [e for e in rt.events
                    if e.fault_triggered and e.oracle_node == f.target_node]
            if fire and fire[0].exposed_result_digest != literal_digest:
                problems.append(f"{spec.task_id}[{label}]: malformed exposure "
                                "is not the committed truncated literal")
        # 7: idempotent replay after the ambiguous mutation
        for f in spec.faults:
            if not (f.fault_type == "malformed"
                    and f.params.get("ambiguous_mutation")):
                continue
            line = f.params["line"]
            reserves = [m for m in rt.env.mutations
                        if m["kind"] == "reserve" and m["line"] == line]
            if len(reserves) != 1:
                problems.append(f"{spec.task_id}[{label}]: ambiguous mutation "
                                f"produced {len(reserves)} reservations")
            replays = [e for e in rt.events
                       if e.oracle_node == f.target_node and e.replay
                       and e.exposed_canonical]
            if not replays:
                problems.append(f"{spec.task_id}[{label}]: no idempotent "
                                "replay observation after ambiguous mutation")

    for split, bundles in splits_data.items():
        paired = split in ("dev",) + EVAL_SPLITS
        for b in bundles:
            if b.spec.faults:
                run_arm(b, b.spec, "fault")
                if paired:
                    run_arm(b, b.spec.without_faults(), "clean")
            else:
                run_arm(b, b.spec, "clean")
    return problems


def check_content_hashes(splits_data) -> list[str]:
    """Check 2b: no episode CONTENT crosses split groups, and none repeats.

    The id/key/template/token checks cover every label attached to a task; this
    covers the task itself. The hash is over exactly what determines the answer --
    the rendered prompt, the KB records, the oracle's tools and resolved args, and
    the committed answer -- so two tasks hashing alike ARE the same task whatever
    their ids say. A crossing would be memorisable held-out content; a repeat
    inside one group inflates that group's denominator with a duplicate.
    """
    import hashlib
    import json as _json

    def content_hash(b) -> str:
        payload = {
            "prompt": b.spec.prompt,
            "kb": {k: b.kb[k] for k in sorted(b.kb)},
            "env": b.spec.env,
            "oracle": [[n.tool, {k: n.args[k] for k in sorted(n.args)}]
                       for n in b.nodes],
            "answer": b.spec.answer,
        }
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False).encode("utf-8")).hexdigest()

    groups: dict[str, dict[str, str]] = {"train": {}, "dev": {}, "eval": {}}
    problems = []
    for split, bundles in splits_data.items():
        group = ("train" if split in TRAIN_SPLITS
                 else "dev" if split == "dev" else "eval")
        for b in bundles:
            h = content_hash(b)
            prior = groups[group].get(h)
            if prior is not None:
                problems.append(f"{group}: duplicate task content "
                                f"{b.spec.task_id} == {prior}")
            else:
                groups[group][h] = b.spec.task_id
    names = sorted(groups)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            inter = set(groups[a]) & set(groups[c])
            if inter:
                ex = sorted(groups[a][h] for h in inter)[:2]
                problems.append(f"task content crosses {a}/{c}: {len(inter)} "
                                f"(e.g. {ex})")
    return problems


def check_missing_node_rejected(splits_data) -> list[str]:
    """Check 8: a trace that skips one oracle node must never verify."""
    problems = []
    # eval specs all carry an assigned fault; test on their clean arms
    sample = [b for b in splits_data["eval"] if b.spec.family == "lookup_chain"][:12]
    for b in sample:
        spec = b.spec.without_faults()
        rt = _fresh_runtime(type(b)(spec=spec, kb=b.kb, nodes=b.nodes))
        skip = b.nodes[1].node_id
        for node in b.nodes:
            if node.node_id == skip:
                continue
            rt.begin_decision()
            rt.dispatch(node.tool, dict(node.args))
        rt.begin_decision()
        verdict = rt.verify(f"\\boxed{{{spec.answer}}}")
        if verdict.strict_success:
            problems.append(f"{spec.task_id}: verifier passed a trace missing "
                            f"node {skip}")
    return problems


def check_same_decision_rejected(splits_data) -> list[str]:
    """Check 9: a dependency inside one assistant decision earns no credit."""
    problems = []
    sample = [b for b in splits_data["eval"] if b.spec.family == "lookup_chain"][:12]
    for b in sample:
        spec = b.spec.without_faults()
        rt = _fresh_runtime(type(b)(spec=spec, kb=b.kb, nodes=b.nodes))
        rt.begin_decision()
        rt.dispatch(b.nodes[0].tool, dict(b.nodes[0].args))
        rt.dispatch(b.nodes[1].tool, dict(b.nodes[1].args))  # same decision
        batched = rt.events[-1]
        if batched.credited:
            problems.append(f"{spec.task_id}: same-decision dependency was "
                            "credited by the runtime")
        for node in b.nodes[2:]:
            rt.begin_decision()
            rt.dispatch(node.tool, dict(node.args))
        rt.begin_decision()
        verdict = rt.verify(f"\\boxed{{{spec.answer}}}")
        if verdict.strict_success:
            problems.append(f"{spec.task_id}: verifier passed a same-decision "
                            "dependency trace")
    return problems


def check_kb_miss_no_leak(splits_data) -> list[str]:
    """Check 10: unknown-key errors must never expose the KB key list."""
    import json

    problems = []
    for split in ("eval",):
        for family in ("lookup_chain", "typed_relay", "fulfillment"):
            b = next(x for x in splits_data[split] if x.spec.family == family)
            rt = _fresh_runtime(b)
            rt.begin_decision()
            exposed = rt.dispatch("kb_lookup", {"key": "KNOSUCHKEY404"})
            obj = json.loads(exposed)
            if obj.get("ok") is not False or obj.get("error") != "no_entry":
                problems.append(f"{family}: kb miss payload is {exposed!r}")
            extras = set(obj) - {"ok", "error", "event_id"}
            if extras:
                problems.append(f"{family}: kb miss leaks fields {sorted(extras)}")
            for key in b.kb:
                if key in exposed:
                    problems.append(f"{family}: kb miss leaked key {key}")
    return problems


CHECKS = (
    ("1 regeneration", "regen"),
    ("2+11 split isolation / eval leakage", "isolation"),
    ("2b task-content hashes (no crossing, no duplicates)", "content"),
    ("3 declared horizons", "horizons"),
    ("4-7 oracle execution, single fire, malformed leak, idempotent replay",
     "oracle"),
    ("8 missing-node rejection", "missing_node"),
    ("9 same-decision rejection", "same_decision"),
    ("10 kb miss no key leak", "kb_miss"),
    ("0 committed sizes", "sizes"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/suite_v1.toml")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    from agentlab.suite.generate import (SPLITS, SPLIT_SEED_KEY, build_split,
                                         load_suite_config)

    cfg = load_suite_config(args.config)
    data_dir = args.data or cfg["out_dir"]

    print("rebuilding all splits in memory (deterministic) ...")
    splits_data = {}
    for split in SPLITS:
        result = build_split(cfg["suite"], split,
                             cfg["seeds"][SPLIT_SEED_KEY[split]],
                             cfg["sizes"][split])
        splits_data[split] = result["bundles"]
        print(f"  {split:<12} {len(result['bundles']):>6} specs")

    runners = {
        "regen": lambda: check_regeneration(cfg, data_dir),
        "isolation": lambda: check_isolation(splits_data),
        "content": lambda: check_content_hashes(splits_data),
        "horizons": lambda: check_horizons(splits_data),
        "oracle": lambda: check_oracle_execution(splits_data),
        "missing_node": lambda: check_missing_node_rejected(splits_data),
        "same_decision": lambda: check_same_decision_rejected(splits_data),
        "kb_miss": lambda: check_kb_miss_no_leak(splits_data),
        "sizes": lambda: check_sizes(splits_data, cfg),
    }

    failed = False
    for label, key in CHECKS:
        problems = runners[key]()
        status = "PASS" if not problems else "FAIL"
        print(f"[{status}] {label}")
        for p in problems[:10]:
            print(f"        {p}")
        if len(problems) > 10:
            print(f"        ... and {len(problems) - 10} more")
        failed = failed or bool(problems)

    print("VALIDATION " + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
