#!/usr/bin/env python
"""Validate suite v1 against the binding failure conditions.

The validator FAILS (exit 1) on any of:

   1. non-byte-identical regeneration, or a committed file whose bytes do not
      hash to its listed SHA256SUMS value;
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
  11. evaluation data appearing in SFT or GRPO inputs;
  12. a SHORTFALL in any registered claim or control cardinality -- exact split
      totals, the three core-eval fault groups, the six MT order patterns and
      their structural-cluster floor, the 400 H8 pairs, 200 absent-information
      tasks per family per arm, exactly 100 deranged permutation tasks, and the
      per-axis dev pool the prompt tournament needs;
  13. a missing or drifted `template_cluster_id` -- the bootstrap resampling
      unit -- on any scored task;
  14. an absent-information control whose hidden value is in fact reachable, or
      whose redaction is a no-op against its own unredacted twin;
  15. a permutation control that is not a fixed-point-free derangement, or whose
      terminal value did not actually move.

Condition 12 exists because the alternative failure mode is silent: a split that
is short does not look broken, it looks like a preregistered gate that turned out
to be "INCONCLUSIVE" after the GPU time was already spent.

CPU-only. Usage:

    PYTHONPATH=src .venv/bin/python scripts/validate_suite.py \
        [--config configs/suite_v1.toml] [--data data/suite/v1]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

TRAIN_SPLITS = ("oracle_sft", "distill", "grpo_train")
EVAL_SPLITS = ("eval", "eval_stress", "eval_mt", "eval_h8", "eval_absent",
               "eval_perm")
# The absent-information control is deliberately unreachable, so it is exempt
# from "the oracle trajectory must verify" -- and gets its own, stricter check
# (14) that the trajectory must NOT reach the hidden value.
UNREACHABLE_SPLITS = ("eval_absent",)


def _fresh_runtime(bundle):
    from agentlab.suite.runtime import EpisodeRuntime

    return EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes)


def check_regeneration(cfg, data_dir) -> list[str]:
    """Check 1: the committed bytes hash to their listed sums AND regenerate.

    Two separate claims, and the second used to be mistaken for the first: the
    old check compared the committed SHA256SUMS TEXT to a regenerated one and
    never hashed a single data file, so a corrupted payload passed as long as the
    sums file was untouched. `sha256sum -c` is done here, in Python, over every
    listed path -- and the paths themselves are checked for the tricks a hostile
    or buggy writer produces (absolute, escaping, duplicated).
    """
    from agentlab.suite.generate import generate_all
    from agentlab.suite.schema import file_sha256

    sums_path = os.path.join(data_dir, "SHA256SUMS")
    if not os.path.exists(sums_path):
        return [f"missing {sums_path}; run scripts/generate_suite.py first"]
    with open(sums_path, encoding="utf-8") as fh:
        committed = fh.read()

    problems: list[str] = []
    listed: dict[str, str] = {}
    for line in committed.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel:
            problems.append(f"SHA256SUMS: malformed line {line!r}")
            continue
        if rel in listed:
            problems.append(f"SHA256SUMS: duplicate entry {rel}")
        if os.path.isabs(rel) or "\\" in rel or ".." in rel.split("/"):
            problems.append(f"SHA256SUMS: unsafe path {rel!r}")
            continue
        listed[rel] = digest
    for rel, digest in sorted(listed.items()):
        path = os.path.join(data_dir, rel)
        if not os.path.exists(path):
            problems.append(f"SHA256SUMS lists {rel}, which is missing on disk")
        elif file_sha256(path) != digest:
            problems.append(f"{rel}: on-disk bytes do not match SHA256SUMS")

    with tempfile.TemporaryDirectory(prefix="suite-regen-") as tmp:
        regen_dir = os.path.join(tmp, "v1")
        generate_all(cfg, regen_dir)
        with open(os.path.join(regen_dir, "SHA256SUMS"), encoding="utf-8") as fh:
            regenerated = fh.read()
    if committed != regenerated:
        problems.append("regeneration is not byte-identical to the committed "
                        "artifacts")
    return problems


def check_sizes(splits_data, cfg) -> list[str]:
    """Check 12: every registered cardinality, per split and across splits.

    A shortfall is a hard failure here rather than an "INCONCLUSIVE" verdict
    later. Nothing in this function is allowed to compare against a number
    derived from the data itself -- the expectations come from the registered
    tables in agentlab.suite.generate.
    """
    from agentlab.suite.generate import (EVAL_FAULT_GROUPS, MT_MAX_PER_CLUSTER,
                                         MT_MIN_CLUSTERS, MT_PATTERNS,
                                         REGISTERED_DEV_PER_AXIS,
                                         REGISTERED_H8_PER_FAMILY,
                                         REGISTERED_H8_TOTAL,
                                         REGISTERED_TOTALS, SPLITS,
                                         assert_suite_cardinalities,
                                         cells_of, cluster_census,
                                         fault_group_census)

    problems = []
    for split in SPLITS:
        bundles = splits_data.get(split)
        if not bundles:
            problems.append(f"{split}: split is absent from the suite")
            continue
        cells = {}
        for b in bundles:
            key = (b.spec.family, b.spec.horizon, b.spec.pattern_id)
            cells[key] = cells.get(key, 0) + 1
        expected_cell = cfg["sizes"][split]
        n_cells = len(cells_of(split))
        for cell, n in sorted(cells.items(), key=str):
            if n != expected_cell:
                problems.append(f"{split} cell {cell}: {n} != {expected_cell}")
        if len(cells) != n_cells:
            problems.append(f"{split}: {len(cells)} realized cells != "
                            f"{n_cells} registered")
        if len(bundles) != REGISTERED_TOTALS[split]:
            problems.append(f"{split}: {len(bundles)} specs != registered "
                            f"{REGISTERED_TOTALS[split]}")

    # core eval fault GROUPS: transient+rate_limit / malformed / wrong_unit
    got_groups = fault_group_census(splits_data.get("eval", []))
    if got_groups != EVAL_FAULT_GROUPS:
        problems.append(f"eval fault groups {got_groups} != registered "
                        f"{EVAL_FAULT_GROUPS}")

    # the six registered MT order patterns and their structural clusters
    mt = splits_data.get("eval_mt", [])
    per_pattern: dict = {}
    for b in mt:
        per_pattern[b.spec.pattern_id] = per_pattern.get(b.spec.pattern_id, 0) + 1
    want_each = REGISTERED_TOTALS["eval_mt"] // len(MT_PATTERNS)
    if sorted(k for k in per_pattern if k is not None) != list(range(len(MT_PATTERNS))):
        problems.append(f"MT patterns realized: {sorted(map(str, per_pattern))} "
                        f"!= 0..{len(MT_PATTERNS) - 1}")
    for pid, n in sorted(per_pattern.items(), key=str):
        if n != want_each:
            problems.append(f"MT pattern {pid}: {n} specs != registered {want_each}")
    census = cluster_census(mt)
    if census["clusters"] < MT_MIN_CLUSTERS:
        problems.append(f"MT structural clusters {census['clusters']} < registered "
                        f"{MT_MIN_CLUSTERS}")
    if census["max_per_cluster"] > MT_MAX_PER_CLUSTER:
        problems.append(f"MT cluster holds {census['max_per_cluster']} "
                        f"instantiations > registered {MT_MAX_PER_CLUSTER}")
    # The registered MT1 stratum is ["eval", "eval_mt"], so the cluster floor and
    # ceiling bind over the combined all-tools H4 sample, not over eval_mt alone.
    mt_stratum = [b for split in ("eval", "eval_mt")
                  for b in splits_data.get(split, [])
                  if b.spec.horizon == 4
                  and {"kb_lookup", "unit_convert", "calculator"}
                  <= {n.tool for n in b.nodes}]
    stratum = cluster_census(mt_stratum)
    if stratum["clusters"] < MT_MIN_CLUSTERS:
        problems.append(f"MT gated stratum: {stratum['clusters']} clusters < "
                        f"registered {MT_MIN_CLUSTERS}")
    if stratum["max_per_cluster"] > MT_MAX_PER_CLUSTER:
        problems.append(f"MT gated stratum: a cluster holds "
                        f"{stratum['max_per_cluster']} instantiations > "
                        f"registered {MT_MAX_PER_CLUSTER}")

    # H8 pairs, absent-information coverage, permutation size
    h8: dict = {}
    for split in ("eval", "eval_h8"):
        for b in splits_data.get(split, []):
            if b.spec.horizon == 8 and b.spec.family in ("lookup_chain",
                                                         "typed_relay"):
                h8[b.spec.family] = h8.get(b.spec.family, 0) + 1
    if sum(h8.values()) != REGISTERED_H8_TOTAL:
        problems.append(f"H8 pairs {sum(h8.values())} != registered "
                        f"{REGISTERED_H8_TOTAL} ({h8})")
    for family, n in sorted(h8.items()):
        if n != REGISTERED_H8_PER_FAMILY:
            problems.append(f"H8 {family}: {n} != registered "
                            f"{REGISTERED_H8_PER_FAMILY}")

    absent: dict = {}
    for b in splits_data.get("eval_absent", []):
        absent[b.spec.family] = absent.get(b.spec.family, 0) + 1
    want_absent = REGISTERED_TOTALS["eval_absent"] // 3
    if sorted(absent) != ["fulfillment", "lookup_chain", "typed_relay"]:
        problems.append(f"absent-information families {sorted(absent)}: all three "
                        f"are registered, per arm")
    for family, n in sorted(absent.items()):
        if n != want_absent:
            problems.append(f"absent-information {family}: {n} != registered "
                            f"{want_absent} per arm")

    try:
        realized = assert_suite_cardinalities(splits_data)
    except AssertionError as exc:
        problems.append(f"cross-split cardinality: {exc}")
        realized = {}
    for axis in ("recovery", "orchestration", "h8"):
        size = realized.get(f"dev_{axis}_axis")
        if size is not None and size < REGISTERED_DEV_PER_AXIS:
            problems.append(f"dev {axis} axis pool {size} < registered "
                            f"{REGISTERED_DEV_PER_AXIS}: the two-round prompt "
                            f"tournament cannot execute")
    return problems


def check_template_clusters(splits_data) -> list[str]:
    """Check 13: the bootstrap unit exists, is structural, and has not drifted.

    Also reports the realized cluster count per split. A split whose scored tasks
    all collapse into one cluster is not a validity failure by itself -- a family
    can genuinely have one structure -- but the CORE eval split collapsing is,
    because the primary claim's clustered interval is computed over it.
    """
    from agentlab.suite.schema import template_cluster_id

    problems = []
    for split, bundles in sorted(splits_data.items()):
        for b in bundles:
            want = template_cluster_id(b.spec.family, b.spec.horizon, b.nodes)
            if b.spec.template_cluster_id != want:
                problems.append(f"{b.spec.task_id}: stored template_cluster_id "
                                f"{b.spec.template_cluster_id} != structural {want}")
                break
    core = splits_data.get("eval", [])
    n_clusters = len({b.spec.template_cluster_id for b in core})
    if core and n_clusters < 12:
        problems.append(f"core eval has {n_clusters} structural clusters; fewer "
                        f"than one per family/horizon cell means the clustered "
                        f"bootstrap is not clustering on structure")
    return problems


def check_absent_information(splits_data, cfg) -> list[str]:
    """Check 14: the hidden value really is unavailable, per family."""
    from agentlab.suite.generate import (SPLIT_SEED_KEY, absent_information_problems,
                                         build_task, cells_of)

    problems: list[str] = []
    cells = {c.family: c for c in cells_of("eval_absent")}
    seed = cfg["seeds"][SPLIT_SEED_KEY["eval_absent"]]
    for b in splits_data.get("eval_absent", []):
        cell = cells[b.spec.family]
        index = int(b.spec.task_id.rsplit("-", 1)[1])
        twin = build_task(cfg["suite"], seed, "eval_absent", cell.family,
                          cell.horizon, index, None, variant=cell.variant,
                          apply_control=False)
        problems += absent_information_problems(b, twin)
    return problems


def check_permutation(splits_data) -> list[str]:
    """Check 15: a fixed-point-free derangement in which every value moved."""
    bundles = splits_data.get("eval_perm", [])
    problems = []
    donors = {}
    for b in bundles:
        meta = b.spec.control_meta or {}
        if b.spec.control != "permuted" or not meta:
            problems.append(f"{b.spec.task_id}: no committed permutation")
            continue
        if meta["donor_task_id"] == b.spec.task_id:
            problems.append(f"{b.spec.task_id}: derangement fixed point")
        if meta["original_answer"] == meta["permuted_answer"]:
            problems.append(f"{b.spec.task_id}: terminal value did not move")
        if b.spec.answer != meta["permuted_answer"]:
            problems.append(f"{b.spec.task_id}: committed answer is not the "
                            f"permuted value")
        record = b.kb.get(meta["target"], {})
        if record.get(meta["field"]) != meta["permuted_answer"]:
            problems.append(f"{b.spec.task_id}: terminal record still holds the "
                            f"original value")
        if meta["original_answer"] in b.spec.prompt:
            problems.append(f"{b.spec.task_id}: the original value is in the prompt")
        donors[b.spec.task_id] = meta["donor_task_id"]
    if bundles and len(set(donors.values())) != len(bundles):
        problems.append("the derangement is not a bijection over the 100 tasks")
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
        if split in UNREACHABLE_SPLITS:
            continue  # check 14 owns these, with the opposite expectation
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
    ("1 regeneration + on-disk sha256sum -c", "regen"),
    ("2+11 split isolation / eval leakage", "isolation"),
    ("2b task-content hashes (no crossing, no duplicates)", "content"),
    ("3 declared horizons", "horizons"),
    ("4-7 oracle execution, single fire, malformed leak, idempotent replay",
     "oracle"),
    ("8 missing-node rejection", "missing_node"),
    ("9 same-decision rejection", "same_decision"),
    ("10 kb miss no key leak", "kb_miss"),
    ("12 registered claim/control cardinalities", "sizes"),
    ("13 structural template clusters (the bootstrap unit)", "clusters"),
    ("14 absent-information control is genuinely unreachable", "absent"),
    ("15 permutation control is a derangement that moved every value", "perm"),
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

    from agentlab.suite.generate import cluster_census

    print("rebuilding all splits in memory (deterministic) ...")
    splits_data = {}
    for split in SPLITS:
        result = build_split(cfg["suite"], split,
                             cfg["seeds"][SPLIT_SEED_KEY[split]],
                             cfg["sizes"][split])
        splits_data[split] = result["bundles"]
        census = cluster_census(result["bundles"])
        print(f"  {split:<12} {len(result['bundles']):>6} specs   "
              f"{census['clusters']:>5} structural clusters "
              f"(max {census['max_per_cluster']} instantiations)")

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
        "clusters": lambda: check_template_clusters(splits_data),
        "absent": lambda: check_absent_information(splits_data, cfg),
        "perm": lambda: check_permutation(splits_data),
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
