"""Deterministic suite v1 generation: splits, fault mixture, serialization.

Splits and EXACT committed sizes -- these are cardinalities, not minimums, and
scripts/validate_suite.py fails on any shortfall:

  oracle_sft   4,800 specs, 400 per cell   (12 cells, train templates 0-7)
  distill      2,400 specs, 200 per cell   (12 cells, train templates 0-7)
  grpo_train   2,400 specs, 200 per cell   (12 cells, train templates 0-7)
  dev          3,600 specs, 300 per cell   (12 cells, dev templates 8-9)
  eval         1,200 specs, 100 per cell   (12 cells, eval templates 10-11)
  eval_stress    280 specs,  40 per cell   (7 cells at or above H8)
  eval_mt        600 specs, 100 per registered order pattern (6 patterns)
  eval_h8        200 specs, 100 per family (lookup_chain + typed_relay at H8)
  eval_absent    600 specs, 200 per family (absent-information control)
  eval_perm      100 specs,  50 per family (counterfactual permutation control)

Why the claim splits exist as splits. The registered secondary claims are gated
at sample sizes the 12 primary cells cannot supply: MT needs 600 all-tools H4
pairs where core eval holds 100, and HR needs 400 H8 pairs where core eval holds
200. Generating the shortfall is the only honest fix -- the alternative would be
lowering a preregistered gate, and no GPU outcome exists to justify that. The
same applies to the two controls: the absent-information arm needs 200 tasks per
family PER ARM whose hidden value is genuinely unavailable, and the
counterfactual arm needs exactly 100 tasks whose hidden values have provably
moved. Both are generated in their controlled form and committed, so no consumer
reimplements the transform and none can silently skip it.

  eval_mt      typed_relay H4 in each of the six registered tool-order patterns
               (`pattern_id` 0..5, 100 each). The orders genuinely differ and are
               causally forced -- see envs/typed_relay.MT_PATTERNS.
  eval_h8      the H8 augmentation that lifts H8 to 400 pairs, 200 per family.
  eval_absent  family-specific redaction: the terminal KB record (lookup_chain),
               the NUMERIC terminal's source record with a 2**48-wide coefficient
               (typed_relay), the terminal warehouse completion token
               (fulfillment). >= 48 bits of hidden entropy in every case.
  eval_perm    a deterministic single-cycle derangement over exactly 100 tasks:
               every terminal value moves to a different task, so no task keeps
               its own answer.

Fault mixture (binding):

  * training-source splits are exactly 50% clean / 50% single-fault; in cells
    containing unit_convert the faulted half splits 25/25/25/25 across
    transient/malformed/wrong-unit/rate-limit; elsewhere 34/33/33 across
    transient/malformed/rate-limit (wrong-unit would be artificial there);
  * dev/eval specs each carry ONE assigned fault -- held-out evaluation is
    counterfactual and paired (each base task runs once clean, once faulted),
    so the fault-arm injected-call rate is exactly 1/H;
  * core eval additionally hits the registered FAULT-GROUP cardinalities exactly
    -- 400 transient/rate-limit, 400 malformed, 400 wrong-unit -- through the
    committed per-cell allocation in EVAL_FAULT_WEIGHTS. The previous
    proportional mixture produced 685/340/175, so the wrong-unit group could
    never reach its registered 400;
  * eval_stress episodes carry TWO distinct faults on distinct nodes, pairs
    balanced within eligible types (rate 2/H); never trained on in v1;
  * the four claim/control splits are CLEAN: MT measures tool-order
    orchestration, H8 measures depth, and a control must isolate one variable;
  * 25% of malformed fulfillment cases target a mutation (the ambiguous
    truncated-reserve case), assigned deterministically by malformed ordinal.

Everything is derived from configs/suite_v1.toml committed seeds through the
SHA-256 counter RNG; regeneration is byte-identical.
"""

from __future__ import annotations

import dataclasses
import os
import shutil

from .envs import family_module
from .envs.typed_relay import MT_COMBOS, MT_PATTERNS
from .faults import pick_wrong_unit
from .rng import CounterRNG
from .schema import (CELLS, SUITE_NAME, SUITE_VERSION, TEMPLATE_RANGES,
                     FaultSpec, OracleNode, TaskSpec, call_budget,
                     decision_budget, digest_text, file_sha256, read_json,
                     read_jsonl, template_cluster_id, tool_pattern, write_json,
                     write_jsonl, write_text)

SPLITS = ("oracle_sft", "distill", "grpo_train", "dev", "eval", "eval_stress",
          "eval_mt", "eval_h8", "eval_absent", "eval_perm")
SPLIT_KIND = {"oracle_sft": "train", "distill": "train", "grpo_train": "train",
              "dev": "dev", "eval": "eval", "eval_stress": "eval",
              "eval_mt": "eval", "eval_h8": "eval", "eval_absent": "eval",
              "eval_perm": "eval"}
SPLIT_SEED_KEY = {"oracle_sft": "oracle_sft", "distill": "distill",
                  "grpo_train": "grpo_train", "dev": "dev", "eval": "eval",
                  "eval_stress": "stress", "eval_mt": "mt", "eval_h8": "h8",
                  "eval_absent": "absent", "eval_perm": "perm"}
DEFAULT_SIZES = {"oracle_sft": 400, "distill": 200, "grpo_train": 200,
                 "dev": 300, "eval": 100, "eval_stress": 40, "eval_mt": 100,
                 "eval_h8": 100, "eval_absent": 200, "eval_perm": 50}
# Splits whose specs carry no scheduled fault at all.
CLEAN_SPLITS = ("eval_mt", "eval_h8", "eval_absent", "eval_perm")
# Splits that are one of the registered controls rather than a measurement.
CONTROL_SPLITS = {"eval_absent": "redacted", "eval_perm": "permuted"}

_UC_TYPES = ("transient", "malformed", "wrong_unit", "rate_limit")
_NO_UC_TYPES = ("transient", "malformed", "rate_limit")
_NO_UC_WEIGHTS = (34, 33, 33)
_AMBIGUOUS_STRIDE = 4  # every 4th malformed fulfillment case targets a mutation

# The committed core-eval fault allocation, per cell, in FAULT-GROUP terms:
# (transient, malformed, wrong_unit, rate_limit). At the committed 100 specs per
# cell these weights ARE the counts, and they sum to exactly 400 per registered
# group: transient+rate_limit 400, malformed 400, wrong_unit 400. Cells without a
# unit_convert node carry no wrong-unit weight, so the 400 wrong-unit cases are
# concentrated where that fault is meaningful.
EVAL_FAULT_WEIGHTS = {
    ("lookup_chain", 2): (25, 50, 0, 25),
    ("lookup_chain", 4): (25, 50, 0, 25),
    ("lookup_chain", 8): (25, 50, 0, 25),
    ("lookup_chain", 12): (25, 50, 0, 25),
    ("typed_relay", 2): (8, 15, 70, 7),
    ("typed_relay", 4): (8, 15, 70, 7),
    ("typed_relay", 8): (8, 15, 70, 7),
    ("typed_relay", 12): (8, 15, 70, 7),
    ("fulfillment", 4): (25, 50, 0, 25),
    ("fulfillment", 8): (15, 30, 40, 15),
    ("fulfillment", 14): (15, 30, 40, 15),
    ("fulfillment", 20): (15, 30, 40, 15),
}
EVAL_FAULT_ORDER = ("transient", "malformed", "wrong_unit", "rate_limit")
# The registered core-eval fault-group cardinalities (exact).
EVAL_FAULT_GROUPS = {"transient_rate_limit": 400, "malformed": 400,
                     "wrong_unit": 400}

# The registered MT structural-cluster floor and per-cluster ceiling.
MT_MIN_CLUSTERS = 240
MT_MAX_PER_CLUSTER = 5
# Registered exact totals per split, asserted at generation AND validation time.
REGISTERED_TOTALS = {"oracle_sft": 4800, "distill": 2400, "grpo_train": 2400,
                     "dev": 3600, "eval": 1200, "eval_stress": 280,
                     "eval_mt": 600, "eval_h8": 200, "eval_absent": 600,
                     "eval_perm": 100}
# H8 pairs are the HR gate's denominator and come from two splits.
REGISTERED_H8_TOTAL = 400
REGISTERED_H8_PER_FAMILY = 200
# The prompt tournament draws 100 + 200 instances per claim axis from dev; the
# narrowest axis is a single cell (typed_relay H4), so dev needs >= 300 per cell.
REGISTERED_DEV_PER_AXIS = 300


@dataclasses.dataclass
class Cell:
    """One generation cell: a family/horizon plus any registered variant.

    The 12 primary cells carry no variant. A claim or control split adds one:
    `pattern_id` selects a registered typed_relay tool-order pattern, `absent`
    selects the family's absent-information grammar.
    """

    family: str
    horizon: int
    variant: dict = dataclasses.field(default_factory=dict)

    @property
    def pattern_id(self):
        return self.variant.get("pattern_id")

    @property
    def absent(self) -> bool:
        return bool(self.variant.get("absent"))

    @property
    def label(self) -> str:
        """The cell's stable label in task ids, manifests and fault mixtures."""
        suffix = "" if self.pattern_id is None else f"-p{self.pattern_id}"
        return f"{self.family}-h{self.horizon}{suffix}"


PRIMARY_CELLS = tuple(Cell(f, h) for f, h in CELLS)
STRESS_CELLS = tuple(Cell(f, h) for f, h in CELLS if h >= 8)
MT_CELLS = tuple(Cell("typed_relay", 4, {"pattern_id": p})
                 for p in range(len(MT_PATTERNS)))
H8_CELLS = (Cell("lookup_chain", 8), Cell("typed_relay", 8))
# The absent-information control uses each family's cheapest horizon that still
# exercises the family, so 1,200 control episodes do not consume the GPU ceiling
# that the mandatory core/MT/H8 measurements need.
ABSENT_CELLS = (Cell("lookup_chain", 4), Cell("typed_relay", 4, {"absent": True}),
                Cell("fulfillment", 4))
# The permutation control needs a terminal KB record whose field can move.
PERM_CELLS = (Cell("lookup_chain", 4), Cell("typed_relay", 4))

SPLIT_CELLS = {"eval_stress": STRESS_CELLS, "eval_mt": MT_CELLS,
               "eval_h8": H8_CELLS, "eval_absent": ABSENT_CELLS,
               "eval_perm": PERM_CELLS}


def load_suite_config(path: str) -> dict:
    import tomllib

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    seeds = {key: int(raw[key]) for key in
             ("oracle_sft", "distill", "grpo_train", "dev", "eval", "stress",
              "mt", "h8", "absent", "perm")}
    sizes = dict(DEFAULT_SIZES)
    for split in SPLITS:
        key = SPLIT_SEED_KEY[split]
        if key in raw.get("sizes", {}):
            sizes[split] = int(raw["sizes"][key])
    return {"suite": raw.get("suite", SUITE_NAME), "seeds": seeds,
            "sizes": sizes,
            "out_dir": raw.get("layout", {}).get("out_dir", "data/suite/v1"),
            "config_path": path, "config_sha256": file_sha256(path)}


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


def eval_fault_types_for(family: str, horizon: int):
    """Core eval's committed per-cell allocation, zero-weight types dropped."""
    weights = EVAL_FAULT_WEIGHTS[(family, horizon)]
    pairs = [(t, w) for t, w in zip(EVAL_FAULT_ORDER, weights) if w]
    return tuple(t for t, _ in pairs), tuple(w for _, w in pairs)


def _pairs(types) -> list[tuple[str, str]]:
    return [(types[i], types[j]) for i in range(len(types))
            for j in range(i + 1, len(types))]


def fault_plan(split: str, family: str, horizon: int, n: int) -> list:
    """Per-spec fault assignment: None | [(type, ambiguous)] | [(t1,a1),(t2,a2)].

    Deterministic and exact: no draws involved, only counts.
    """
    if split in CLEAN_SPLITS:
        # A claim or control split isolates one variable. Mixing an injected
        # fault into the MT or H8 augmentation would put recovery back into the
        # secondary claims' denominators, and into a control it would add a
        # second reason an episode can fail.
        return [None] * n
    types, weights = (eval_fault_types_for(family, horizon) if split == "eval"
                      else fault_types_for(family, horizon))
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
               horizon: int, index: int, fault_entries, *,
               variant: dict | None = None,
               apply_control: bool = True) -> TaskBundle:
    """One fully deterministic task; identical labels -> identical bytes.

    `variant` carries the registered per-cell variation (`pattern_id` for an MT
    order pattern, `absent` for the absent-information grammar). It also enters
    the RNG label, so pattern 0 and pattern 3 at the same index never draw the
    same keys, and a primary cell's stream is unchanged when no variant is given.
    """
    variant = dict(variant or {})
    pattern_id = variant.get("pattern_id")
    labels = [suite, f"{seed_value:#x}", split, family, f"h{horizon}"]
    if pattern_id is not None:
        labels.append(f"p{pattern_id}")
    if pattern_id is not None or (family == "typed_relay" and horizon == 4):
        # The registered structural-template allocation: index i realises
        # template i mod 50, so 100 tasks cover all 50 (conversion, form)
        # templates exactly twice each. It covers the whole typed_relay H4 cell,
        # not just the MT split, because MT1's registered stratum is
        # ["eval", "eval_mt"] and its <= 5 instantiations-per-cluster ceiling
        # binds over that COMBINED sample.
        variant["combo"] = index % len(MT_COMBOS)
    if variant.get("absent"):
        labels.append("absent")
    labels.append(index)
    trng = CounterRNG(*labels)
    mod = family_module(family)
    draft = mod.generate_task(trng.derive("task"), horizon, **variant)

    control = CONTROL_SPLITS.get(split)
    control_meta: dict | None = None
    if control == "redacted" and apply_control:
        # The committed control task IS the redacted task: family-specific,
        # applied here once, verifiable on disk.
        control_meta = mod.redact_absent(draft)

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

    suffix = "" if pattern_id is None else f"-p{pattern_id}"
    spec = TaskSpec(
        task_id=f"{split}-{family}-h{horizon}{suffix}-{index:04d}",
        suite=suite, split=split, family=family, horizon=horizon,
        template_id=template_id, prompt=prompt,
        answer=draft.answer, answer_kind=draft.answer_kind,
        start=draft.start, env=draft.env, faults=faults,
        max_decisions=decision_budget(horizon, len(faults)),
        max_calls=call_budget(horizon),
        secret_tokens=draft.secret_tokens,
        template_cluster_id=template_cluster_id(family, horizon, draft.nodes),
        pattern_id=pattern_id,
        control=control if control_meta is not None else None,
        control_meta=control_meta,
    )
    return TaskBundle(spec=spec, kb=draft.kb, nodes=draft.nodes)


def _terminal_kb_key(bundle: TaskBundle) -> str:
    kb_nodes = [n for n in bundle.nodes if n.tool == "kb_lookup"]
    if not kb_nodes:
        raise RuntimeError(f"{bundle.spec.task_id}: no kb_lookup node to permute")
    return kb_nodes[-1].args["key"]


def apply_derangement(suite: str, seed_value: int, split: str,
                      bundles: list) -> dict:
    """Move every terminal value to a DIFFERENT task; return the mapping.

    A single deterministic cycle over the shuffled task order is a derangement
    by construction: sigma(i) = the next task in the cycle, so no task can keep
    its own value, and the assertion below proves it rather than trusting it. The
    permuted answers are a bijection of the originals, so the split's terminal
    tokens stay globally unique.

    `provenance.permute_hidden_values` re-derives a permutation from a runtime
    seed and can leave fixed points; committing the derangement here makes
    "exactly 100 tasks, every value moved" a property of the artifact.
    """
    n = len(bundles)
    if n < 2:
        raise RuntimeError("a derangement needs at least two tasks")
    order = CounterRNG(suite, f"{seed_value:#x}", split,
                       "derangement-v1").shuffle(range(n))
    donor_of = {order[i]: order[(i + 1) % n] for i in range(n)}
    originals = [b.spec.answer for b in bundles]
    if len(set(originals)) != n:
        raise RuntimeError("permutation control has duplicate terminal values")

    mapping = {}
    for i, bundle in enumerate(bundles):
        j = donor_of[i]
        if j == i:
            raise RuntimeError(f"derangement fixed point at {i}")
        donor = bundles[j]
        key = _terminal_kb_key(bundle)
        field = "code"
        record = bundle.kb.get(key)
        if not isinstance(record, dict) or field not in record:
            raise RuntimeError(f"{bundle.spec.task_id}: terminal record has no "
                               f"{field!r} field to permute")
        original = originals[i]
        moved = originals[j]
        if moved == original:
            raise RuntimeError(f"{bundle.spec.task_id}: permuted value did not move")
        record[field] = moved
        # The oracle node's canonical envelope holds the same record object, but
        # rebind it explicitly rather than relying on aliasing.
        terminal = bundle.nodes[-1]
        terminal.expect = {"ok": True, "record": record}
        bundle.spec.answer = moved
        bundle.spec.control = "permuted"
        bundle.spec.control_meta = {
            "kind": "terminal_kb_field", "target": key, "field": field,
            "original_answer": original, "permuted_answer": moved,
            "donor_task_id": donor.spec.task_id, "index": i, "donor_index": j,
            "why": "the returned value is another task's; an output that tracks "
                   "the prompt instead of the observation is detectable"}
        mapping[bundle.spec.task_id] = donor.spec.task_id
    return mapping


def build_split(suite: str, split: str, seed_value: int,
                per_cell: int) -> dict:
    cells = SPLIT_CELLS.get(split, PRIMARY_CELLS)
    specs: list[dict] = []
    kb: dict = {}
    oracles: list[dict] = []
    bundles: list[TaskBundle] = []
    fault_mix: dict = {}
    for cell in cells:
        plan = fault_plan(split, cell.family, cell.horizon, per_cell)
        mix: dict = {"clean": 0}
        for index, entries in enumerate(plan):
            bundle = build_task(suite, seed_value, split, cell.family,
                                cell.horizon, index, entries,
                                variant=cell.variant)
            bundles.append(bundle)
            if not entries:
                mix["clean"] += 1
            else:
                label = "+".join(t for t, _ in entries)
                if any(a for _, a in entries):
                    label += "(ambiguous)"
                mix[label] = mix.get(label, 0) + 1
        fault_mix[cell.label] = dict(sorted(mix.items()))

    permutation: dict = {}
    if split == "eval_perm":
        permutation = apply_derangement(suite, seed_value, split, bundles)

    # Rows are serialized only after every in-place control transform, so what
    # lands on disk is exactly what the bundles say.
    for bundle in bundles:
        srow, krow, orow = bundle.rows()
        specs.append(srow)
        kb[bundle.spec.task_id] = krow
        oracles.append(orow)

    result = {"specs": specs, "kb": kb, "oracles": oracles,
              "fault_mix": fault_mix, "bundles": bundles,
              "permutation": permutation}
    assert_split_cardinalities(split, result, per_cell)
    return result


# ---------------------------------------------------------------------------
# what a model can actually reach (the absent-information proof)
# ---------------------------------------------------------------------------

def observation_frontier(bundle: TaskBundle) -> dict:
    """The exposed payloads the oracle path can canonically reach, and where it
    stops.

    Dispatches the canonical oracle calls in order through a fresh runtime and
    STOPS at the first observation that is not that node's canonical envelope:
    beyond that point the next canonical call's arguments are no longer derivable
    from anything the model has seen, so replaying them would credit the model
    with knowledge only the oracle has. That distinction is the whole
    absent-information argument -- replaying the full oracle path on a redacted
    typed_relay task "reaches" the terminal number only because the oracle's
    stored expression already contains the coefficient the deleted record held.

    -> {"exposed": [payload strings], "broke_at": index|None, "reached": n}
    """
    from .runtime import EpisodeRuntime

    rt = EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes)
    exposed: list[str] = []
    broke_at = None
    for i, node in enumerate(bundle.nodes):
        rt.begin_decision()
        exposed.append(rt.dispatch(node.tool, dict(node.args)))
        if not rt.events[-1].exposed_canonical:
            broke_at = i
            break
    return {"exposed": exposed, "broke_at": broke_at, "reached": len(exposed)}


def absent_information_problems(bundle: TaskBundle, twin: TaskBundle) -> list:
    """Is this control task's hidden value genuinely unavailable? -> problems.

    Three independent conditions, none of them a flag to be trusted:

      1. the committed answer appears nowhere in the model-visible material
         (prompt, KB records, environment state);
      2. no observation on the reachable frontier contains it, and a KB-record
         redaction actually breaks the path before the terminal node;
      3. the UNREDACTED twin -- same seed, same labels, redaction skipped -- does
         expose it, which is what proves the redaction is not a no-op.
    """
    from .schema import canon

    spec, meta = bundle.spec, (bundle.spec.control_meta or {})
    answer = spec.answer
    problems = []
    visible = "\n".join([spec.prompt, canon(bundle.kb), canon(spec.env or {})])
    if answer in visible:
        problems.append(f"{spec.task_id}: answer is present in the model-visible "
                        f"task material")
    front = observation_frontier(bundle)
    if any(answer in payload for payload in front["exposed"]):
        problems.append(f"{spec.task_id}: answer is exposed on the reachable "
                        f"frontier")
    kind = meta.get("kind")
    if kind == "kb_record":
        if front["broke_at"] is None:
            problems.append(f"{spec.task_id}: the redacted record did not break "
                            f"the oracle path at all")
    elif kind == "env_completion_token":
        field = meta.get("field")
        if field and field in (front["exposed"][-1] if front["exposed"] else ""):
            problems.append(f"{spec.task_id}: terminal envelope still carries "
                            f"{field}")
    else:
        problems.append(f"{spec.task_id}: unknown redaction kind {kind!r}")
    twin_front = observation_frontier(twin)
    if not any(answer in payload for payload in twin_front["exposed"]):
        problems.append(f"{spec.task_id}: the UNREDACTED twin does not expose the "
                        f"answer either -- the redaction proves nothing")
    if int(meta.get("hidden_entropy_bits", 0)) < 48:
        problems.append(f"{spec.task_id}: hidden entropy "
                        f"{meta.get('hidden_entropy_bits')} bits < 48")
    return problems


# ---------------------------------------------------------------------------
# generation-time cardinality and structure assertions
#
# These fire during generation, not during a later analysis, because that is the
# difference between "the suite is short" and "a preregistered gate turned out to
# be unsatisfiable after the GPU time was spent".
# ---------------------------------------------------------------------------

def cluster_census(bundles) -> dict:
    """{"clusters": n distinct, "max_per_cluster": k, "counts": {...}}."""
    counts: dict = {}
    for b in bundles:
        counts[b.spec.template_cluster_id] = counts.get(
            b.spec.template_cluster_id, 0) + 1
    return {"clusters": len(counts),
            "max_per_cluster": max(counts.values()) if counts else 0,
            "counts": counts}


def fault_group_census(bundles) -> dict:
    """Registered fault GROUPS (transient+rate_limit are one group) -> counts."""
    groups = {k: 0 for k in EVAL_FAULT_GROUPS}
    for b in bundles:
        for f in b.spec.faults:
            key = ("transient_rate_limit"
                   if f.fault_type in ("transient", "rate_limit")
                   else f.fault_type)
            if key in groups:
                groups[key] += 1
    return groups


def assert_split_cardinalities(split: str, result: dict, per_cell: int) -> None:
    """Every registered property of one split, checked before it is written.

    Scaled-down sizes (the unit tests build 2 specs per cell) skip the exact
    totals but keep every structural invariant.
    """
    bundles = result["bundles"]
    n = len(bundles)
    exact = per_cell == DEFAULT_SIZES[split]
    if exact and REGISTERED_TOTALS[split] != n:
        raise AssertionError(f"{split}: generated {n} specs, registered "
                             f"{REGISTERED_TOTALS[split]}")
    for b in bundles:
        if not b.spec.template_cluster_id.startswith("tc-"):
            raise AssertionError(f"{b.spec.task_id}: no template_cluster_id")

    if split == "eval" and exact:
        got = fault_group_census(bundles)
        if got != EVAL_FAULT_GROUPS:
            raise AssertionError(f"eval fault groups {got} != registered "
                                 f"{EVAL_FAULT_GROUPS}")

    if split == "eval_mt":
        per_pattern: dict = {}
        for b in bundles:
            if b.spec.pattern_id is None:
                raise AssertionError(f"{b.spec.task_id}: MT spec has no pattern_id")
            if (b.spec.family, b.spec.horizon) != ("typed_relay", 4):
                raise AssertionError(f"{b.spec.task_id}: MT spec is not "
                                     "typed_relay H4")
            tools = {node.tool for node in b.nodes}
            if not {"kb_lookup", "unit_convert", "calculator"} <= tools:
                raise AssertionError(f"{b.spec.task_id}: MT spec does not "
                                     "require all three tools")
            got = tool_pattern(b.nodes)
            if got != MT_PATTERNS[b.spec.pattern_id]:
                raise AssertionError(f"{b.spec.task_id}: order {got!r} != "
                                     f"registered {MT_PATTERNS[b.spec.pattern_id]!r}")
            per_pattern[b.spec.pattern_id] = per_pattern.get(b.spec.pattern_id, 0) + 1
        if sorted(per_pattern) != list(range(len(MT_PATTERNS))):
            raise AssertionError(f"MT patterns present: {sorted(per_pattern)}")
        if exact and set(per_pattern.values()) != {REGISTERED_TOTALS[split]
                                                   // len(MT_PATTERNS)}:
            raise AssertionError(f"MT per-pattern counts {per_pattern} are not "
                                 "balanced at the registered size")
        census = cluster_census(bundles)
        if exact and census["clusters"] < MT_MIN_CLUSTERS:
            raise AssertionError(f"MT has {census['clusters']} structural "
                                 f"clusters, registered >= {MT_MIN_CLUSTERS}")
        if census["max_per_cluster"] > MT_MAX_PER_CLUSTER:
            raise AssertionError(f"MT cluster holds {census['max_per_cluster']} "
                                 f"instantiations, registered "
                                 f"<= {MT_MAX_PER_CLUSTER}")

    if split == "eval_h8":
        per_family: dict = {}
        for b in bundles:
            if b.spec.horizon != 8:
                raise AssertionError(f"{b.spec.task_id}: H8 augmentation at "
                                     f"horizon {b.spec.horizon}")
            per_family[b.spec.family] = per_family.get(b.spec.family, 0) + 1
        if set(per_family) != {"lookup_chain", "typed_relay"}:
            raise AssertionError(f"H8 augmentation families {sorted(per_family)}")

    if split == "eval_absent":
        per_family = {}
        for b in bundles:
            meta = b.spec.control_meta or {}
            if b.spec.control != "redacted" or not meta:
                raise AssertionError(f"{b.spec.task_id}: no redaction descriptor")
            bits = int(meta.get("hidden_entropy_bits", 0))
            if bits < 48:
                raise AssertionError(f"{b.spec.task_id}: redaction leaves "
                                     f"{bits} bits of hidden entropy (< 48)")
            per_family[b.spec.family] = per_family.get(b.spec.family, 0) + 1
        if set(per_family) != {"lookup_chain", "typed_relay", "fulfillment"}:
            raise AssertionError(f"absent-information families "
                                 f"{sorted(per_family)}: all three required")
        if exact and set(per_family.values()) != {per_cell}:
            raise AssertionError(f"absent-information per family {per_family}")

    if split == "eval_perm":
        donors = result["permutation"]
        if len(donors) != n:
            raise AssertionError(f"permutation covers {len(donors)} of {n} tasks")
        for b in bundles:
            meta = b.spec.control_meta or {}
            if b.spec.control != "permuted" or not meta:
                raise AssertionError(f"{b.spec.task_id}: no permutation descriptor")
            if meta["original_answer"] == meta["permuted_answer"]:
                raise AssertionError(f"{b.spec.task_id}: value did not move")
            if meta["donor_task_id"] == b.spec.task_id:
                raise AssertionError(f"{b.spec.task_id}: derangement fixed point")
        if len({b.spec.answer for b in bundles}) != n:
            raise AssertionError("permuted answers are not a bijection")


def assert_suite_cardinalities(splits: dict) -> dict:
    """Cross-split registered properties; returns the realized census.

    Like the per-split assertions, the exact counts bind only when the splits
    involved are at their registered sizes -- the unit tests build 2 specs per
    cell. scripts/validate_suite.py enforces the exact numbers unconditionally
    against the registered tables, so a scaled-down build cannot hide a
    shortfall in the committed artifact.
    """
    h8_family: dict = {}
    for split in ("eval", "eval_h8"):
        for b in splits.get(split, []):
            if b.spec.horizon == 8 and b.spec.family in ("lookup_chain",
                                                         "typed_relay"):
                h8_family[b.spec.family] = h8_family.get(b.spec.family, 0) + 1
    h8_total = sum(h8_family.values())
    census = {"h8_pairs": h8_total, "h8_per_family": h8_family}
    h8_exact = all(len(splits.get(s, [])) == REGISTERED_TOTALS[s]
                   for s in ("eval", "eval_h8"))
    if h8_exact and h8_total != REGISTERED_H8_TOTAL:
        raise AssertionError(f"H8 pairs {h8_total} != registered "
                             f"{REGISTERED_H8_TOTAL} ({h8_family})")
    if h8_exact and set(h8_family.values()) != {REGISTERED_H8_PER_FAMILY}:
        raise AssertionError(f"H8 per family {h8_family} != registered "
                             f"{REGISTERED_H8_PER_FAMILY} each")

    # MT1's registered stratum is ["eval", "eval_mt"], and its cluster floor and
    # per-cluster ceiling bind over that COMBINED sample -- not over eval_mt
    # alone. Core eval's all-tools H4 tasks share the MT cluster space, so a
    # randomly drawn structural template there put 8 instantiations in one
    # cluster against a registered ceiling of 5.
    mt_sample = [b for split in ("eval", "eval_mt") for b in splits.get(split, [])
                 if b.spec.horizon == 4
                 and {"kb_lookup", "unit_convert", "calculator"}
                 <= {n.tool for n in b.nodes}]
    if mt_sample:
        mt_census = cluster_census(mt_sample)
        census["mt_stratum_pairs"] = len(mt_sample)
        census["mt_stratum_clusters"] = mt_census["clusters"]
        census["mt_stratum_max_per_cluster"] = mt_census["max_per_cluster"]
        if mt_census["max_per_cluster"] > MT_MAX_PER_CLUSTER:
            raise AssertionError(
                f"MT gated stratum has a cluster with "
                f"{mt_census['max_per_cluster']} instantiations > registered "
                f"{MT_MAX_PER_CLUSTER}")
        if (len(mt_sample) >= REGISTERED_TOTALS["eval_mt"]
                and mt_census["clusters"] < MT_MIN_CLUSTERS):
            raise AssertionError(
                f"MT gated stratum has {mt_census['clusters']} clusters < "
                f"registered {MT_MIN_CLUSTERS}")

    # The prompt tournament's narrowest claim axis is the single typed_relay H4
    # dev cell; 100 round-one plus 200 round-two instances need 300 there.
    dev = splits.get("dev", [])
    if dev:
        cells: dict = {}
        for b in dev:
            cells[(b.spec.family, b.spec.horizon)] = cells.get(
                (b.spec.family, b.spec.horizon), 0) + 1
        orch = cells.get(("typed_relay", 4), 0)
        census["dev_orchestration_axis"] = orch
        census["dev_h8_axis"] = (cells.get(("lookup_chain", 8), 0)
                                 + cells.get(("typed_relay", 8), 0))
        census["dev_recovery_axis"] = sum(1 for b in dev if b.spec.faults)
        if sum(cells.values()) == REGISTERED_TOTALS["dev"]:
            for axis, size in (("orchestration", orch),
                               ("h8", census["dev_h8_axis"]),
                               ("recovery", census["dev_recovery_axis"])):
                if size < REGISTERED_DEV_PER_AXIS:
                    raise AssertionError(
                        f"dev {axis} axis pool {size} < registered "
                        f"{REGISTERED_DEV_PER_AXIS}: the prompt tournament "
                        f"cannot execute")
    return census


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
    return SPLIT_CELLS.get(split, PRIMARY_CELLS)


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

def _template_fields(spec: TaskSpec, mod) -> dict:
    """The fields that select a family's paraphrase template SET.

    Recovered from the committed spec alone: `start` says which bootstrap values
    the task exposed (a start key, or the operands of a conversion-first or
    calculation-first order pattern), and fulfillment's express core is decided by
    the horizon. Previously this passed `env["express"]`, a key fulfillment never
    writes, so every express task hashed the WRONG template.
    """
    fields = dict(spec.start or {})
    if hasattr(mod, "is_express"):
        fields["express"] = mod.is_express(spec.horizon)
    return fields


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
    mod = family_module(spec.family)
    node_pos = {n.node_id: i for i, n in enumerate(nodes)}
    template = mod.template_text(spec.template_id, _template_fields(spec, mod))
    kb_lookups = [n for n in nodes if n.tool == "kb_lookup"]
    tools_used = {n.tool for n in nodes}
    faults = [{"class": f.fault_type, "node_index": node_pos[f.target_node],
               "node": f.target_node, "params": f.params} for f in spec.faults]
    row = {
        "task_id": spec.task_id, "suite": spec.suite, "family": spec.family,
        "split": spec.split, "horizon": spec.horizon,
        "template_id": spec.template_id,
        "template_hash": digest_text(template),
        # The bootstrap resampling unit. `template_id` is the PARAPHRASE id and
        # held-out evaluation uses only ids 10-11, so clustering on it collapses
        # 1,200 eval tasks into two clusters; this is the structural identity.
        "template_cluster_id": spec.template_cluster_id,
        "kb_namespace": f"{spec.suite}/{spec.split}",
        # `pattern_id` is now the registered ORDER PATTERN (0..5) or None -- it
        # used to be the tool-sequence string, which made the per-pattern gate
        # group by an unregistered label. The string kept its own field.
        "pattern_id": spec.pattern_id,
        "tool_pattern": tool_pattern(nodes),
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
        # A committed control row is redactABLE because it is already redacted.
        # Saying otherwise would let the arm's "drop unredactable specs" filter
        # silently delete the 200 fulfillment control tasks -- the exact family
        # whose hidden value is not in the KB at all -- and the (arm, family)
        # coverage check would report a shortfall instead of a leak.
        "redactable": spec.control == "redacted" or bool(kb_lookups),
        # Every scored value in this suite is drawn per task from the committed
        # seed (keys, codes, operands, tokens), so the committed answer changes
        # whenever the hidden values change.
        "counterfactual_sensitive": True,
        "fault": (faults[0] if len(faults) == 1 else None),
        "faults": faults or None,
        "max_decisions": spec.max_decisions, "max_calls": spec.max_calls,
        "secret_tokens": list(spec.secret_tokens),
    }
    if spec.control is not None:
        meta = dict(spec.control_meta or {})
        # The control is already APPLIED in the committed row -- the KB record is
        # gone, the completion token is withheld, the terminal value has moved.
        # `answer` stays the hidden/true value, which is exactly what a leakage
        # check scores against; nothing here is model-visible.
        row["control"] = spec.control
        row[("redaction" if spec.control == "redacted" else "permutation")] = meta
        if spec.control == "permuted":
            row["original_answer"] = meta.get("original_answer")
            row["permuted_from"] = meta.get("donor_task_id")
    return row


def _generate_into(cfg: dict, out_dir: str) -> dict:
    suite = cfg["suite"]
    manifest_splits: dict = {}
    files: dict = {}
    all_bundles: dict = {}

    for split in SPLITS:
        seed_value = cfg["seeds"][SPLIT_SEED_KEY[split]]
        per_cell = cfg["sizes"][split]
        result = build_split(suite, split, seed_value, per_cell)
        all_bundles[split] = result["bundles"]

        spec_path = os.path.join(out_dir, "specs", f"{split}.jsonl")
        kb_path = os.path.join(out_dir, "kb", f"{split}.json")
        oracle_path = os.path.join(out_dir, "oracles", f"{split}.jsonl")
        write_jsonl(spec_path, result["specs"])
        write_json(kb_path, result["kb"])
        write_jsonl(oracle_path, result["oracles"])

        for path, rows in ((spec_path, len(result["specs"])),
                           (kb_path, len(result["kb"])),
                           (oracle_path, len(result["oracles"]))):
            # POSIX separators: SHA256SUMS must read the same on every platform.
            rel = os.path.relpath(path, out_dir).replace(os.sep, "/")
            files[rel] = {"sha256": file_sha256(path),
                          "bytes": os.path.getsize(path), "rows": rows}

        census = cluster_census(result["bundles"])
        meta = {
            "count": len(result["specs"]),
            "per_cell": per_cell,
            "registered_count": REGISTERED_TOTALS[split],
            "cells": [c.label for c in cells_of(split)],
            "seed": f"{seed_value:#010x}",
            "template_ids": list(TEMPLATE_RANGES[SPLIT_KIND[split]]),
            "fault_mix": result["fault_mix"],
            "template_clusters": census["clusters"],
            "max_per_template_cluster": census["max_per_cluster"],
        }
        if split == "eval":
            meta["fault_groups"] = fault_group_census(result["bundles"])
        if split == "eval_mt":
            meta["patterns"] = {str(p): MT_PATTERNS[p]
                               for p in range(len(MT_PATTERNS))}
            meta["per_pattern"] = {
                str(p): sum(1 for b in result["bundles"] if b.spec.pattern_id == p)
                for p in range(len(MT_PATTERNS))}
        if split in CONTROL_SPLITS:
            meta["control"] = CONTROL_SPLITS[split]
            meta["control_kinds"] = sorted({
                (b.spec.control_meta or {}).get("kind", "?")
                for b in result["bundles"]})
            if CONTROL_SPLITS[split] == "redacted":
                # Only the absent-information control has a hidden-entropy
                # budget; reporting 0 for a permutation would read as a defect.
                meta["min_hidden_entropy_bits"] = min(
                    int((b.spec.control_meta or {}).get("hidden_entropy_bits", 0))
                    for b in result["bundles"])
        if split == "eval_perm":
            meta["derangement"] = result["permutation"]
        manifest_splits[split] = meta

    suite_census = assert_suite_cardinalities(all_bundles)

    manifest = {
        "suite": suite,
        "version": SUITE_VERSION,
        "generator": "agentlab.suite.generate",
        "config": {"path": os.path.basename(cfg.get("config_path", "")),
                   "sha256": cfg.get("config_sha256", "")},
        "cells": [f"{f}-h{h}" for f, h in CELLS],
        "stress_cells": [c.label for c in STRESS_CELLS],
        "registered_totals": dict(REGISTERED_TOTALS),
        "registered_fault_groups": dict(EVAL_FAULT_GROUPS),
        "registered_h8_pairs": REGISTERED_H8_TOTAL,
        "registered_mt_clusters": {"min_clusters": MT_MIN_CLUSTERS,
                                   "max_per_cluster": MT_MAX_PER_CLUSTER},
        "registered_dev_per_axis": REGISTERED_DEV_PER_AXIS,
        "realized": suite_census,
        "splits": manifest_splits,
        "files": files,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    files["manifest.json"] = {
        "sha256": file_sha256(os.path.join(out_dir, "manifest.json")),
        "bytes": os.path.getsize(os.path.join(out_dir, "manifest.json")),
        "rows": 1}
    # Forced LF, POSIX paths: `sha256sum -c SHA256SUMS` must pass as written.
    write_text(os.path.join(out_dir, "SHA256SUMS"),
               "".join(f"{meta['sha256']}  {rel}\n"
                       for rel, meta in sorted(files.items())))
    return manifest


def generate_all(cfg: dict, out_dir: str, *, atomic: bool = True) -> dict:
    """Generate every split into out_dir; returns the manifest dict.

    Atomic by default: the tree is built in a staging directory beside the
    destination and swapped in only after every split, every cardinality
    assertion, the manifest and SHA256SUMS have succeeded. A crashed run
    previously left five complete-looking splits with no manifest, which is
    exactly the shape a half-generated suite must never have. Derived outputs
    under the destination (certspecs/) are replaced, not merged -- they are
    regenerated from the new bytes by scripts/export_eval_specs.py.
    """
    if not atomic:
        return _generate_into(cfg, out_dir)
    dest = os.path.abspath(out_dir)
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)
    staging = f"{dest}.staging-{os.getpid()}"
    previous = f"{dest}.previous-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        manifest = _generate_into(cfg, staging)
        if os.path.exists(dest):
            os.rename(dest, previous)
        try:
            os.rename(staging, dest)
        except OSError:
            if os.path.exists(previous):
                os.rename(previous, dest)
            raise
        shutil.rmtree(previous, ignore_errors=True)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
