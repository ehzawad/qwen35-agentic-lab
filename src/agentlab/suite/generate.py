"""Deterministic suite v1 generation: splits, fault mixture, serialization.

Splits and exact committed sizes (12 primary family/horizon cells):

  oracle_sft   4,800 specs, 400 per cell     (train templates 0-7)
  distill      2,400 specs, 200 per cell     (train templates 0-7)
  grpo_train   2,400 specs, 200 per cell     (train templates 0-7)
  dev            240 specs,  20 per cell     (dev templates 8-9)
  eval         1,200 specs, 100 per cell     (eval templates 10-11)
  eval_stress    280 specs,  40 per cell at or above H8 (eval templates 10-11)

Fault mixture (binding):

  * training-source splits are exactly 50% clean / 50% single-fault; in cells
    containing unit_convert the faulted half splits 25/25/25/25 across
    transient/malformed/wrong-unit/rate-limit; elsewhere 34/33/33 across
    transient/malformed/rate-limit (wrong-unit would be artificial there);
  * dev/eval specs each carry ONE assigned fault -- held-out evaluation is
    counterfactual and paired (each base task runs once clean, once faulted),
    so the fault-arm injected-call rate is exactly 1/H;
  * eval_stress episodes carry TWO distinct faults on distinct nodes, pairs
    balanced within eligible types (rate 2/H); never trained on in v1;
  * 25% of malformed fulfillment cases target a mutation (the ambiguous
    truncated-reserve case), assigned deterministically by malformed ordinal.

Everything is derived from configs/suite_v1.toml committed seeds through the
SHA-256 counter RNG; regeneration is byte-identical.
"""

from __future__ import annotations

import dataclasses
import os

from .envs import family_module
from .faults import pick_wrong_unit
from .rng import CounterRNG
from .schema import (CELLS, SUITE_NAME, SUITE_VERSION, TEMPLATE_RANGES,
                     FaultSpec, OracleNode, TaskSpec, call_budget,
                     decision_budget, digest_text, file_sha256, read_json,
                     read_jsonl, write_json, write_jsonl)

SPLITS = ("oracle_sft", "distill", "grpo_train", "dev", "eval", "eval_stress")
SPLIT_KIND = {"oracle_sft": "train", "distill": "train", "grpo_train": "train",
              "dev": "dev", "eval": "eval", "eval_stress": "eval"}
SPLIT_SEED_KEY = {"oracle_sft": "oracle_sft", "distill": "distill",
                  "grpo_train": "grpo_train", "dev": "dev", "eval": "eval",
                  "eval_stress": "stress"}
DEFAULT_SIZES = {"oracle_sft": 400, "distill": 200, "grpo_train": 200,
                 "dev": 20, "eval": 100, "eval_stress": 40}
STRESS_CELLS = tuple((f, h) for f, h in CELLS if h >= 8)

_UC_TYPES = ("transient", "malformed", "wrong_unit", "rate_limit")
_NO_UC_TYPES = ("transient", "malformed", "rate_limit")
_NO_UC_WEIGHTS = (34, 33, 33)
_AMBIGUOUS_STRIDE = 4  # every 4th malformed fulfillment case targets a mutation


def load_suite_config(path: str) -> dict:
    import tomllib

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    seeds = {key: int(raw[key]) for key in
             ("oracle_sft", "distill", "grpo_train", "dev", "eval", "stress")}
    sizes = dict(DEFAULT_SIZES)
    for split in SPLITS:
        key = SPLIT_SEED_KEY[split]
        if key in raw.get("sizes", {}):
            sizes[split] = int(raw["sizes"][key])
    return {"suite": raw.get("suite", SUITE_NAME), "seeds": seeds,
            "sizes": sizes,
            "out_dir": raw.get("layout", {}).get("out_dir", "data/suite/v1")}


def apportion(n: int, weights) -> list[int]:
    """Largest-remainder apportionment; ties break toward earlier entries."""
    total = sum(weights)
    exact = [n * w / total for w in weights]
    counts = [int(x) for x in exact]
    order = sorted(range(len(weights)),
                   key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def fault_types_for(family: str, horizon: int):
    if family_module(family).has_unit_convert(horizon):
        return _UC_TYPES, (1, 1, 1, 1)
    return _NO_UC_TYPES, _NO_UC_WEIGHTS


def _pairs(types) -> list[tuple[str, str]]:
    return [(types[i], types[j]) for i in range(len(types))
            for j in range(i + 1, len(types))]


def fault_plan(split: str, family: str, horizon: int, n: int) -> list:
    """Per-spec fault assignment: None | [(type, ambiguous)] | [(t1,a1),(t2,a2)].

    Deterministic and exact: no draws involved, only counts.
    """
    types, weights = fault_types_for(family, horizon)
    is_fulfillment = family == "fulfillment"
    plan: list = []

    def entries_for(seq_types: list) -> list:
        """Expand a flat type sequence, marking ambiguous malformed ordinals."""
        out = []
        malformed_ordinal = 0
        for t in seq_types:
            if t == "malformed" and is_fulfillment:
                ambiguous = malformed_ordinal % _AMBIGUOUS_STRIDE == 0
                malformed_ordinal += 1
            else:
                ambiguous = False
            out.append((t, ambiguous))
        return out

    if split == "eval_stress":
        pairs = _pairs(types)
        counts = apportion(n, [1] * len(pairs))
        seq_pairs: list = []
        for pair, c in zip(pairs, counts):
            seq_pairs.extend([pair] * c)
        # ambiguous ordinals count across ALL malformed occurrences in the cell
        malformed_ordinal = 0
        for t1, t2 in seq_pairs:
            entry = []
            for t in (t1, t2):
                if t == "malformed" and is_fulfillment:
                    ambiguous = malformed_ordinal % _AMBIGUOUS_STRIDE == 0
                    malformed_ordinal += 1
                else:
                    ambiguous = False
                entry.append((t, ambiguous))
            plan.append(entry)
        return plan

    if SPLIT_KIND[split] == "train":
        n_clean = n // 2
        counts = apportion(n - n_clean, weights)
        seq = [t for t, c in zip(types, counts) for _ in range(c)]
        plan = [None] * n_clean + [[e] for e in entries_for(seq)]
        return plan

    # dev / eval: every base task carries one assigned fault (paired design)
    counts = apportion(n, weights)
    seq = [t for t, c in zip(types, counts) for _ in range(c)]
    return [[e] for e in entries_for(seq)]


def _eligible_nodes(nodes, fault_type: str, ambiguous: bool) -> list:
    if fault_type == "wrong_unit":
        return [n for n in nodes if n.tool == "unit_convert"]
    if fault_type == "malformed":
        if ambiguous:
            return [n for n in nodes
                    if n.mutating and n.match.get("action") == "reserve"]
        return [n for n in nodes if not n.mutating]
    return list(nodes)  # transient and rate_limit: any node (pre-mutation)


def _assignment_order(draft, fault_entries) -> list[int]:
    """Indices of `fault_entries` in most-constrained-first order.

    Two-fault stress episodes must land on DISTINCT nodes, and the eligible
    sets are wildly different sizes: `wrong_unit` can only sit on a
    unit_convert node (exactly one in fulfillment H8, and only two in
    typed_relay H8), while transient/rate_limit accept any node. Assigning in
    plan order therefore lets a permissive fault steal the single node a
    scarce one needs, and the pair becomes unschedulable. Ordering by eligible
    count (ties by plan position) is deterministic and always succeeds when a
    valid assignment exists for these grammars.
    """
    sizes = [len(_eligible_nodes(draft.nodes, t, a)) for t, a in fault_entries]
    return sorted(range(len(fault_entries)), key=lambda i: (sizes[i], i))


def _build_fault(draft, fault_type: str, ambiguous: bool, rng,
                 taken: set) -> FaultSpec:
    eligible = [n for n in _eligible_nodes(draft.nodes, fault_type, ambiguous)
                if n.node_id not in taken]
    if not eligible:
        raise RuntimeError(f"no eligible node for {fault_type} "
                           f"(ambiguous={ambiguous}) outside {sorted(taken)}")
    node = rng.choice(eligible)
    taken.add(node.node_id)
    params: dict = {}
    if fault_type == "rate_limit":
        params["retry_after_turns"] = 1
    elif fault_type == "wrong_unit":
        params["wrong_unit"] = pick_wrong_unit(node.args["to_unit"], rng)
    elif fault_type == "malformed" and ambiguous:
        params["ambiguous_mutation"] = True
        quote_token = node.match["tokens"][0]
        line = next(ln["line"] for ln in draft.env["lines"]
                    if ln["quote_token"] == quote_token)
        params["line"] = line
    return FaultSpec(fault_type=fault_type, target_node=node.node_id,
                     params=params)


@dataclasses.dataclass
class TaskBundle:
    spec: TaskSpec
    kb: dict
    nodes: list

    def rows(self) -> tuple[dict, dict, dict]:
        return (self.spec.to_row(), self.kb,
                {"task_id": self.spec.task_id,
                 "nodes": [n.to_row() for n in self.nodes]})


def build_task(suite: str, seed_value: int, split: str, family: str,
               horizon: int, index: int, fault_entries) -> TaskBundle:
    """One fully deterministic task; identical labels -> identical bytes."""
    trng = CounterRNG(suite, f"{seed_value:#x}", split, family,
                      f"h{horizon}", index)
    mod = family_module(family)
    draft = mod.generate_task(trng.derive("task"), horizon)
    template_id = trng.derive("template").choice(
        TEMPLATE_RANGES[SPLIT_KIND[split]])
    prompt = mod.render_prompt(template_id, draft.prompt_fields)

    faults: list[FaultSpec] = []
    if fault_entries:
        frng = trng.derive("fault")
        taken: set = set()
        built = {}
        for i in _assignment_order(draft, fault_entries):
            t, a = fault_entries[i]
            built[i] = _build_fault(draft, t, a, frng, taken)
        node_pos = {n.node_id: i for i, n in enumerate(draft.nodes)}
        faults = sorted(built.values(), key=lambda f: node_pos[f.target_node])

    spec = TaskSpec(
        task_id=f"{split}-{family}-h{horizon}-{index:04d}",
        suite=suite, split=split, family=family, horizon=horizon,
        template_id=template_id, prompt=prompt,
        answer=draft.answer, answer_kind=draft.answer_kind,
        start=draft.start, env=draft.env, faults=faults,
        max_decisions=decision_budget(horizon, len(faults)),
        max_calls=call_budget(horizon),
        secret_tokens=draft.secret_tokens,
    )
    return TaskBundle(spec=spec, kb=draft.kb, nodes=draft.nodes)


def build_split(suite: str, split: str, seed_value: int,
                per_cell: int) -> dict:
    cells = STRESS_CELLS if split == "eval_stress" else CELLS
    specs: list[dict] = []
    kb: dict = {}
    oracles: list[dict] = []
    bundles: list[TaskBundle] = []
    fault_mix: dict = {}
    for family, horizon in cells:
        plan = fault_plan(split, family, horizon, per_cell)
        mix: dict = {"clean": 0}
        for index, entries in enumerate(plan):
            bundle = build_task(suite, seed_value, split, family, horizon,
                                index, entries)
            bundles.append(bundle)
            srow, krow, orow = bundle.rows()
            specs.append(srow)
            kb[bundle.spec.task_id] = krow
            oracles.append(orow)
            if not entries:
                mix["clean"] += 1
            else:
                label = "+".join(t for t, _ in entries)
                if any(a for _, a in entries):
                    label += "(ambiguous)"
                mix[label] = mix.get(label, 0) + 1
        fault_mix[f"{family}-h{horizon}"] = dict(sorted(mix.items()))
    return {"specs": specs, "kb": kb, "oracles": oracles,
            "fault_mix": fault_mix, "bundles": bundles}


# ---------------------------------------------------------------------------
# reading a generated split back (the ONE loader for committed suite data)
# ---------------------------------------------------------------------------

def split_paths(out_dir: str, split: str) -> dict:
    return {"specs": os.path.join(out_dir, "specs", f"{split}.jsonl"),
            "kb": os.path.join(out_dir, "kb", f"{split}.json"),
            "oracles": os.path.join(out_dir, "oracles", f"{split}.jsonl")}


def load_bundles(out_dir: str, split: str, task_ids=None) -> list:
    """Rebuild TaskBundles from the committed specs/kb/oracles of one split.

    This is the only way any consumer obtains suite tasks: there is no second
    manifest format and no second seed-derivation path. `task_ids`, when given,
    selects a subset in the order it lists.
    """
    paths = split_paths(out_dir, split)
    kb_all = read_json(paths["kb"])
    oracles = {row["task_id"]: row["nodes"] for row in read_jsonl(paths["oracles"])}
    bundles = {}
    order = []
    for row in read_jsonl(paths["specs"]):
        spec = TaskSpec.from_row(row)
        nodes = [OracleNode.from_row(n) for n in oracles[spec.task_id]]
        kb = kb_all[spec.task_id]
        bundles[spec.task_id] = TaskBundle(spec=spec, kb=kb, nodes=nodes)
        order.append(spec.task_id)
    if task_ids is None:
        return [bundles[t] for t in order]
    return [bundles[t] for t in task_ids]


def cells_of(split: str) -> tuple:
    return STRESS_CELLS if split == "eval_stress" else CELLS


def group_by_cell(bundles) -> dict:
    """{(family, horizon): [bundles]} preserving committed spec-file order."""
    out: dict = {}
    for b in bundles:
        out.setdefault((b.spec.family, b.spec.horizon), []).append(b)
    return out


def cell_slice(bundles, per_cell: int, offset: int = 0) -> list:
    """A balanced, deterministic, reproducible subsample: per_cell from each cell.

    Spec files are written in a fixed order, so `offset`/`per_cell` windows over
    the same split are disjoint by construction. This is how the prompt
    tournament, the post-SFT gate, the variance probe and the GRPO training pool
    carve non-overlapping task sets out of one committed split instead of each
    inventing its own seeded sampler.
    """
    out = []
    for _cell, block in sorted(group_by_cell(bundles).items()):
        window = block[offset:offset + per_cell]
        if len(window) < per_cell:
            raise ValueError(
                f"cell {_cell} has {len(block)} specs; cannot take {per_cell} "
                f"at offset {offset}. Regenerate the split with larger sizes "
                f"rather than silently shrinking a sample.")
        out.extend(window)
    return out


# ---------------------------------------------------------------------------
# certification-layer spec adapter
# ---------------------------------------------------------------------------

def _tool_pattern(nodes) -> str:
    short = {"kb_lookup": "kb", "unit_convert": "uc", "calculator": "calc",
             "warehouse_query": "wq", "warehouse_update": "wu"}
    return ">".join(short.get(n.tool, n.tool) for n in nodes)


def certification_spec(bundle: TaskBundle) -> dict:
    """One canonical task rendered into the certification-layer spec contract.

    `agentlab.suite.evaluate` and `agentlab.provenance` implement the frozen
    receipt/recovery-token protocol of docs/AGENTIC_PROTOCOL.md over a flat spec
    dict that carries its KB and oracle inline. This adapter is the ONLY bridge:
    the suite generator stays the single source of tasks, and the certification
    layer stops needing a task format of its own.

    Faithfulness is asserted, not assumed: `tests/test_suite_reconciliation.py`
    replays every adapted spec through `provenance.execute_oracle` and requires
    the node-by-node envelopes to equal the canonical `OracleNode.expect`
    payloads and the derived answer to equal the committed answer.
    """
    spec, nodes = bundle.spec, bundle.nodes
    node_pos = {n.node_id: i for i, n in enumerate(nodes)}
    template = family_module(spec.family).template_text(
        spec.template_id, {"express": bool((spec.env or {}).get("express"))})
    kb_lookups = [n for n in nodes if n.tool == "kb_lookup"]
    tools_used = {n.tool for n in nodes}
    faults = [{"class": f.fault_type, "node_index": node_pos[f.target_node],
               "node": f.target_node, "params": f.params} for f in spec.faults]
    return {
        "task_id": spec.task_id, "suite": spec.suite, "family": spec.family,
        "split": spec.split, "horizon": spec.horizon,
        "template_id": spec.template_id,
        "template_hash": digest_text(template),
        "kb_namespace": f"{spec.suite}/{spec.split}",
        "pattern_id": _tool_pattern(nodes),
        "all_tools_required": {"kb_lookup", "unit_convert",
                               "calculator"} <= tools_used,
        "prompt": spec.prompt,
        "kb": dict(bundle.kb),
        "env": spec.env,
        "oracle": [{"node": n.node_id, "tool": n.tool, "args": dict(n.args)}
                   for n in nodes],
        "answer": spec.answer, "answer_kind": spec.answer_kind,
        "answer_field": "code",
        # Redaction target: the last KB record on the oracle path. Removing it
        # makes the required lookup unable to return the hidden value, which is
        # what the absent-information control needs. Express fulfillment (H4) has
        # no KB lookup at all -- its hidden value is the finalize completion
        # token, which a KB deletion cannot withhold -- so it is explicitly NOT
        # redactable rather than silently redacted into an unchanged task.
        "hidden_key": (kb_lookups[-1].args["key"] if kb_lookups else None),
        "redactable": bool(kb_lookups),
        # Every scored value in this suite is drawn per task from the committed
        # seed (keys, codes, operands, tokens), so the committed answer changes
        # whenever the hidden values change.
        "counterfactual_sensitive": True,
        "fault": (faults[0] if len(faults) == 1 else None),
        "faults": faults or None,
        "max_decisions": spec.max_decisions, "max_calls": spec.max_calls,
        "secret_tokens": list(spec.secret_tokens),
    }


def generate_all(cfg: dict, out_dir: str) -> dict:
    """Generate every split into out_dir; returns the manifest dict."""
    suite = cfg["suite"]
    manifest_splits: dict = {}
    files: dict = {}

    for split in SPLITS:
        seed_value = cfg["seeds"][SPLIT_SEED_KEY[split]]
        per_cell = cfg["sizes"][split]
        result = build_split(suite, split, seed_value, per_cell)

        spec_path = os.path.join(out_dir, "specs", f"{split}.jsonl")
        kb_path = os.path.join(out_dir, "kb", f"{split}.json")
        oracle_path = os.path.join(out_dir, "oracles", f"{split}.jsonl")
        write_jsonl(spec_path, result["specs"])
        write_json(kb_path, result["kb"])
        write_jsonl(oracle_path, result["oracles"])

        for path, rows in ((spec_path, len(result["specs"])),
                           (kb_path, len(result["kb"])),
                           (oracle_path, len(result["oracles"]))):
            rel = os.path.relpath(path, out_dir)
            files[rel] = {"sha256": file_sha256(path),
                          "bytes": os.path.getsize(path), "rows": rows}

        manifest_splits[split] = {
            "count": len(result["specs"]),
            "per_cell": per_cell,
            "seed": f"{seed_value:#010x}",
            "template_ids": list(TEMPLATE_RANGES[SPLIT_KIND[split]]),
            "fault_mix": result["fault_mix"],
        }

    manifest = {
        "suite": suite,
        "version": SUITE_VERSION,
        "cells": [f"{f}-h{h}" for f, h in CELLS],
        "stress_cells": [f"{f}-h{h}" for f, h in STRESS_CELLS],
        "splits": manifest_splits,
        "files": files,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    files["manifest.json"] = {
        "sha256": file_sha256(os.path.join(out_dir, "manifest.json")),
        "bytes": os.path.getsize(os.path.join(out_dir, "manifest.json")),
        "rows": 1}
    sums = "".join(f"{meta['sha256']}  {rel}\n"
                   for rel, meta in sorted(files.items()))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write(sums)
    return manifest
