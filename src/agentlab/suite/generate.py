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

TWO PHASES, TWO COMMITMENTS (D3, the post-lock held-out state machine).

Generation is no longer one pass over ten splits. It is two phases with two
different seed sources, and the second one cannot run early:

  train-dev  oracle_sft, distill, grpo_train, dev -- seeded from the PUBLIC
             committed seeds in configs/suite_v1.toml, because prompt selection
             and training must be able to generate them before anything is
             locked. Pinned at P (the commit adding configs/preregistration_final.json)
             by manifest.train-dev.json + SHA256SUMS.train-dev.
  heldout    eval, eval_stress, eval_mt, eval_h8, eval_absent, eval_perm --
             seeded from L, the dedicated commit that adds the COMPLETE
             results/agentic/locks.json. There is no held-out seed in the config,
             no --seed flag, no environment seed and no fallback: without a
             verified reveal receipt these six splits cannot be generated at all.
             Pinned at R (the reveal commit) by manifest.heldout.json +
             SHA256SUMS.heldout.

`generate_all` no longer exists as a working entry point -- it REFUSES, because
emitting every split in one pass is exactly the behaviour that made the held-out
answers derivable from the preregistration commit alone.

Everything is still derived through the SHA-256 counter RNG and regeneration is
byte-identical WITHIN a phase: train/dev from the committed seeds, held-out from
the seed the locks commit determines.
"""

from __future__ import annotations

import dataclasses
import hashlib
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
DEFAULT_SIZES = {"oracle_sft": 400, "distill": 200, "grpo_train": 200,
                 "dev": 300, "eval": 100, "eval_stress": 40, "eval_mt": 100,
                 "eval_h8": 100, "eval_absent": 200, "eval_perm": 50}

# ---------------------------------------------------------------------------
# generation phases (D3)
# ---------------------------------------------------------------------------
# The two phases are disjoint, exhaustive over SPLITS, and they have DIFFERENT
# seed sources. `PHASE_OF` is the single authority every consumer asks -- the
# validator, the loader, the certification adapter and the exporter -- so a
# newly registered split cannot end up in neither phase and therefore in neither
# commitment.
TRAIN_DEV_PHASE = "train-dev"
HELDOUT_PHASE = "heldout"
TRAIN_DEV_SPLITS = ("oracle_sft", "distill", "grpo_train", "dev")
HELDOUT_SPLITS = ("eval", "eval_stress", "eval_mt", "eval_h8", "eval_absent",
                  "eval_perm")
PHASES = {TRAIN_DEV_PHASE: TRAIN_DEV_SPLITS, HELDOUT_PHASE: HELDOUT_SPLITS}
PHASE_OF = {s: phase for phase, splits in PHASES.items() for s in splits}
assert tuple(sorted(PHASE_OF)) == tuple(sorted(SPLITS))

# Per-phase commitments. The old GLOBAL manifest.json / SHA256SUMS pair is gone:
# it disclosed held-out hashes and claimed the whole suite was pinned at a commit
# where the held-out bytes did not (and must not) exist yet.
PHASE_MANIFEST = {TRAIN_DEV_PHASE: "manifest.train-dev.json",
                  HELDOUT_PHASE: "manifest.heldout.json"}
PHASE_SUMS = {TRAIN_DEV_PHASE: "SHA256SUMS.train-dev",
              HELDOUT_PHASE: "SHA256SUMS.heldout"}
# Refused on sight: a tree carrying these is a tree pinned by the retired
# whole-suite commitment.
LEGACY_COMMITMENTS = ("manifest.json", "SHA256SUMS")

# The held-out derivation, frozen. `heldout-master-v2` is the version label: the
# retired v1 derivation hung off the PREREGISTRATION commit, which exists before
# any lock, so every held-out answer was derivable before the prompt winner and
# the checkpoint were frozen. v2 hangs off L.
HELDOUT_DERIVATION = "heldout-master-v2"
HELDOUT_MASTER_LABEL_PARTS = ("qwen35-agentic-lab", "agentlab-suite-v1",
                              "heldout-master-v2")
HELDOUT_SPLIT_LABEL = "agentlab-heldout-split-v2"
HELDOUT_RELEASE_LABEL = "agentlab-heldout-release-v2"
REVEAL_SCHEMA = "agentic-heldout-reveal-v2"
GENERATION_PROTOCOL = 2
STUDY_ID = "agentic-v1"
_HEX = set("0123456789abcdef")


def _commit_bytes(commit: str, what: str) -> bytes:
    """A full 40-hex commit id, or a refusal. No prefixes, ever.

    A shortened prefix would let two different commits derive the same suite, and
    "the seed came from some commit starting with a1b2c3" is not a commitment.
    """
    raw = str(commit or "").strip().lower()
    if len(raw) != 40 or not set(raw) <= _HEX:
        raise ValueError(
            f"REFUSED: {what} must be a full 40-hex git commit id, got "
            f"{commit!r}. A shortened prefix or a symbolic name is not a "
            f"commitment.")
    return bytes.fromhex(raw)


def heldout_master_seed(locks_commit: str) -> bytes:
    """The 256-bit master seed, a pure function of L.

    L is the dedicated commit that adds the complete results/agentic/locks.json.
    Because the seed is a function of L, the held-out realization cannot exist
    before the prompt winner and the checkpoint digest are published -- which is
    the entire mechanism. It is NOT a randomness beacon: a commit id is
    author-influenceable, and docs/AGENTIC_PROTOCOL.md says so out loud.
    """
    label = "\0".join(HELDOUT_MASTER_LABEL_PARTS) + "\0"
    return hashlib.sha256(label.encode("ascii")
                          + _commit_bytes(locks_commit, "the locks commit L")).digest()


def heldout_split_seed(master: bytes, split: str) -> int:
    """One independent 256-bit seed per held-out split, derived from the master."""
    if split not in HELDOUT_SPLITS:
        raise ValueError(f"{split!r} is not a held-out split: {HELDOUT_SPLITS}")
    if not isinstance(master, bytes) or len(master) != 32:
        raise ValueError("the master seed must be exactly 32 bytes")
    return int.from_bytes(
        hashlib.sha256((HELDOUT_SPLIT_LABEL + "\0").encode("ascii") + master
                       + b"\0" + split.encode("ascii")).digest(), "big")


def heldout_release_id(master: bytes) -> str:
    """The full 256-bit release id every held-out spec and certspec carries.

    Task ids and file names do not encode their seed, so a stale old-seed file
    looks superficially valid. The release id is what makes it fail.
    """
    if not isinstance(master, bytes) or len(master) != 32:
        raise ValueError("the master seed must be exactly 32 bytes")
    return hashlib.sha256((HELDOUT_RELEASE_LABEL + "\0").encode("ascii")
                          + master).hexdigest()
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


# The legacy key names the retired whole-suite config used for the six held-out
# seeds. Their PRESENCE is refused, not ignored: a config that still carries them
# is a config from which the held-out set is derivable before any lock.
_RETIRED_HELDOUT_SEED_KEYS = ("eval", "stress", "mt", "h8", "absent", "perm",
                              "eval_stress", "eval_mt", "eval_h8",
                              "eval_absent", "eval_perm")


def load_suite_config(path: str) -> dict:
    """The committed train/dev seeds plus the frozen held-out PLAN.

    The loader no longer requires -- and now actively refuses -- a held-out seed
    value anywhere in the file. There is no fallback: `cfg["seeds"]` holds four
    train/dev seeds and nothing else, so a consumer that wants a held-out split
    has to come with a verified reveal receipt.
    """
    import tomllib

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    seeds_block = raw.get("seeds")
    if isinstance(seeds_block, dict) and "train_dev" in seeds_block:
        train_dev = seeds_block["train_dev"]
    else:                                    # a flat file, i.e. the retired shape
        train_dev = raw
    offenders = sorted(
        k for k in _RETIRED_HELDOUT_SEED_KEYS
        if k in (train_dev if isinstance(train_dev, dict) else {})
        or k in (seeds_block if isinstance(seeds_block, dict) else {})
        or (k in raw and not isinstance(raw[k], dict) and k != "suite"))
    if offenders:
        raise SystemExit(
            f"REFUSED: {path} carries held-out seed value(s) {offenders}. The six "
            f"held-out seeds derive from L (the commit that adds the complete "
            f"results/agentic/locks.json) through {HELDOUT_DERIVATION}; a value in "
            f"the config would make the held-out set derivable before the prompt "
            f"winner and the checkpoint are frozen, which is the defect this "
            f"phase split exists to close.")
    missing = [s for s in TRAIN_DEV_SPLITS if s not in train_dev]
    if missing:
        raise SystemExit(f"REFUSED: {path} is missing train/dev seed(s) {missing}")
    seeds = {s: int(train_dev[s]) for s in TRAIN_DEV_SPLITS}

    sizes = dict(DEFAULT_SIZES)
    for split, value in (raw.get("sizes") or {}).items():
        if split not in DEFAULT_SIZES:
            raise SystemExit(f"REFUSED: {path} sizes an unregistered split "
                             f"{split!r}; the registered splits are "
                             f"{sorted(DEFAULT_SIZES)}")
        sizes[split] = int(value)

    heldout = dict(raw.get("heldout") or {})
    # The derivation is FROZEN: the file and the code must agree, in both
    # directions, or the commitment means whichever one you happened to read.
    frozen = {
        "derivation": HELDOUT_DERIVATION,
        "master_label_parts": list(HELDOUT_MASTER_LABEL_PARTS),
        "split_label": HELDOUT_SPLIT_LABEL,
        "release_label": HELDOUT_RELEASE_LABEL,
        "splits": list(HELDOUT_SPLITS),
        "generation_protocol": GENERATION_PROTOCOL,
    }
    for key, want in frozen.items():
        got = heldout.get(key)
        if got is None:
            raise SystemExit(f"REFUSED: {path} does not register the frozen "
                             f"held-out derivation field {key!r}")
        if (list(got) if isinstance(want, list) else got) != want:
            raise SystemExit(f"REFUSED: {path} registers {key}={got!r}; the "
                             f"generator implements {want!r}. The derivation is "
                             f"frozen -- change both together in a dated "
                             f"AMENDMENT or neither.")
    phases = {k.replace("_", "-"): list(v)
              for k, v in (raw.get("phases") or {}).items()}
    for phase, splits in PHASES.items():
        if phases.get(phase) != list(splits):
            raise SystemExit(f"REFUSED: {path} registers phase {phase} as "
                             f"{phases.get(phase)!r}; the generator implements "
                             f"{list(splits)!r}")
    return {"suite": raw.get("suite", SUITE_NAME), "seeds": seeds,
            "sizes": sizes, "heldout": heldout, "phases": phases,
            "out_dir": raw.get("layout", {}).get("out_dir", "data/suite/v1"),
            "config_path": path, "config_sha256": file_sha256(path)}


def split_seed(cfg: dict, split: str, reveal: dict | None = None) -> int:
    """The ONE seed lookup. Train/dev from the config, held-out from the reveal."""
    if PHASE_OF[split] == TRAIN_DEV_PHASE:
        return int(cfg["seeds"][split])
    if not reveal:
        raise RuntimeError(
            f"REFUSED: {split} is a held-out split and there is no reveal "
            f"receipt. Its seed derives from L (the commit adding the complete "
            f"results/agentic/locks.json); it is not in the config, there is no "
            f"--seed flag and there is no fallback.")
    return int(reveal["split_seeds"][split])


def load_reveal(path: str, *, study_id: str = STUDY_ID) -> dict:
    """Read a reveal receipt and REDERIVE everything it claims.

    Nothing in the receipt is taken on trust: the master seed is recomputed from
    the locks commit it names, the release id is recomputed from the master seed,
    and the six split seeds are derived here rather than read. A receipt that
    carries a seed value of its own -- a supplied held-out seed -- is refused
    outright.
    """
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise SystemExit(f"REFUSED: {path} is not a reveal receipt object")

    def need(key: str):
        if key not in receipt:
            raise SystemExit(f"REFUSED: {path} has no {key!r}; an incomplete "
                             f"receipt reveals nothing")
        return receipt[key]

    if need("schema") != REVEAL_SCHEMA:
        raise SystemExit(f"REFUSED: {path} schema {receipt['schema']!r} != "
                         f"{REVEAL_SCHEMA!r}")
    if need("study_id") != study_id:
        raise SystemExit(f"REFUSED: {path} belongs to study "
                         f"{receipt['study_id']!r}, not {study_id!r}")
    if int(need("generation_protocol")) != GENERATION_PROTOCOL:
        raise SystemExit(f"REFUSED: {path} generation_protocol "
                         f"{receipt['generation_protocol']!r} != "
                         f"{GENERATION_PROTOCOL}")
    if need("derivation_label") != HELDOUT_DERIVATION:
        raise SystemExit(f"REFUSED: {path} derivation_label "
                         f"{receipt['derivation_label']!r} != "
                         f"{HELDOUT_DERIVATION!r}")
    stray = sorted(k for k in receipt
                   if "seed" in k.lower() and k != "master_seed_hex")
    if stray:
        raise SystemExit(
            f"REFUSED: {path} carries seed field(s) {stray}. The only seed in a "
            f"receipt is the master seed REDERIVED from L; a supplied seed is "
            f"exactly what this mechanism forbids.")
    locks_commit = str(need("locks_commit")).strip().lower()
    prereg_commit = str(need("preregistration_commit")).strip().lower()
    _commit_bytes(prereg_commit, "the preregistration commit P")
    master = heldout_master_seed(locks_commit)
    claimed = str(need("master_seed_hex")).strip()
    if claimed != master.hex():
        raise SystemExit(
            f"REFUSED: {path} claims master seed {claimed[:16]}..., but "
            f"{HELDOUT_DERIVATION} over locks commit {locks_commit[:12]} gives "
            f"{master.hex()[:16]}.... The seed is derived, never supplied.")
    rid = heldout_release_id(master)
    if str(need("heldout_release_id")) != rid:
        raise SystemExit(f"REFUSED: {path} claims release id "
                         f"{receipt['heldout_release_id']!r}, derivation gives "
                         f"{rid!r}")
    out = dict(receipt)
    out["master_seed"] = master
    out["heldout_release_id"] = rid
    out["split_seeds"] = {s: heldout_split_seed(master, s) for s in HELDOUT_SPLITS}
    out["reveal_path"] = path
    out["reveal_sha256"] = file_sha256(path)
    return out


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
    # Which held-out RELEASE these bytes belong to, or None for train/dev. Set by
    # the generator from the reveal receipt and by `load_bundles` from the
    # verified release on disk -- never guessed. The certification adapter refuses
    # a held-out bundle without it, so a stale old-seed file cannot be exported.
    release_id: str | None = None

    def rows(self) -> tuple[dict, dict, dict]:
        srow = self.spec.to_row()
        if self.release_id is not None:
            srow["heldout_release_id"] = self.release_id
        return (srow, self.kb,
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


def build_split(suite: str, split: str, seed_value: int, per_cell: int, *,
                release_id: str | None = None) -> dict:
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
            bundle.release_id = release_id
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
    from .contract import load_or_create_secret
    from .runtime import EpisodeRuntime

    rt = EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes,
                        secret=load_or_create_secret())
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


def split_rels(split: str) -> tuple[str, str, str]:
    """The three POSIX-relative payload paths of one split, in commitment order."""
    return (f"kb/{split}.json", f"oracles/{split}.jsonl", f"specs/{split}.jsonl")


def phase_manifest_path(out_dir: str, phase: str) -> str:
    return os.path.join(out_dir, PHASE_MANIFEST[phase])


def phase_sums_path(out_dir: str, phase: str) -> str:
    return os.path.join(out_dir, PHASE_SUMS[phase])


def read_phase_manifest(out_dir: str, phase: str) -> dict | None:
    p = phase_manifest_path(out_dir, phase)
    return read_json(p) if os.path.exists(p) else None


def read_sums(path: str) -> dict:
    """`SHA256SUMS`-format text -> {relpath: digest}, refusing hostile paths."""
    listed: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or set(digest) - _HEX or not rel:
            raise ValueError(f"{path}: malformed line {line!r}")
        if rel in listed:
            raise ValueError(f"{path}: duplicate entry {rel}")
        if os.path.isabs(rel) or "\\" in rel or ".." in rel.split("/"):
            raise ValueError(f"{path}: unsafe path {rel!r}")
        listed[rel] = digest
    return listed


def heldout_release(out_dir: str) -> dict | None:
    """The VERIFIED held-out release on disk, or None if there is no reveal yet.

    Verified means: a sealed manifest.heldout.json, a SHA256SUMS.heldout that
    lists it, and a release id that rederives from the master seed the manifest
    records. Anything less is not a release and this returns None rather than a
    half-trusted dict.
    """
    manifest = read_phase_manifest(out_dir, HELDOUT_PHASE)
    if not manifest:
        return None
    rel = manifest.get("heldout_release_id")
    master_hex = str(manifest.get("master_seed_hex", ""))
    if len(master_hex) != 64 or set(master_hex) - _HEX:
        raise RuntimeError(f"{PHASE_MANIFEST[HELDOUT_PHASE]} carries no full "
                           f"master seed; it is not a release manifest")
    if rel != heldout_release_id(bytes.fromhex(master_hex)):
        raise RuntimeError(f"{PHASE_MANIFEST[HELDOUT_PHASE]}: release id does not "
                           f"rederive from its own master seed")
    sums_p = phase_sums_path(out_dir, HELDOUT_PHASE)
    if not os.path.exists(sums_p):
        raise RuntimeError(f"REFUSED: {PHASE_MANIFEST[HELDOUT_PHASE]} exists but "
                           f"{PHASE_SUMS[HELDOUT_PHASE]} does not; the reveal "
                           f"commit R must add both")
    return manifest


def _verify_listed(out_dir: str, listed: dict, rels) -> None:
    for rel in rels:
        if rel not in listed:
            raise RuntimeError(
                f"REFUSED: {rel} is not covered by "
                f"{PHASE_SUMS[HELDOUT_PHASE]}; a held-out file outside the "
                f"release commitment is a cached early value, not evidence.")
        path = os.path.join(out_dir, rel)
        if not os.path.exists(path):
            raise RuntimeError(f"REFUSED: {rel} is listed in the release but "
                               f"missing on disk")
        if file_sha256(path) != listed[rel]:
            raise RuntimeError(
                f"REFUSED: {rel} does not hash to its committed release value. "
                f"These bytes are not the revealed held-out set.")


def load_bundles(out_dir: str, split: str, task_ids=None) -> list:
    """Rebuild TaskBundles from the committed specs/kb/oracles of one split.

    This is the only way any consumer obtains suite tasks: there is no second
    manifest format and no second seed-derivation path. `task_ids`, when given,
    selects a subset in the order it lists.

    A HELD-OUT split additionally has to prove it belongs to the revealed
    release: the three payload files must hash to their committed values in
    SHA256SUMS.heldout, and every spec row must carry that release's id. The
    old-seed bundles that used to sit in this tree carry no release id and hash
    to nothing, so they cannot be loaded at all -- which is what "invalidated"
    has to mean for a file whose name and task ids look exactly right.
    """
    paths = split_paths(out_dir, split)
    release_id = None
    if PHASE_OF[split] == HELDOUT_PHASE:
        manifest = heldout_release(out_dir)
        if manifest is None:
            raise RuntimeError(
                f"REFUSED: {split} is a held-out split and this tree carries no "
                f"revealed release ({PHASE_MANIFEST[HELDOUT_PHASE]} is absent). "
                f"Held-out data exists only after L is published and "
                f"`agentic_locks.py reveal` has written the receipt.")
        release_id = manifest["heldout_release_id"]
        listed = read_sums(phase_sums_path(out_dir, HELDOUT_PHASE))
        _verify_listed(out_dir, listed, split_rels(split))
    kb_all = read_json(paths["kb"])
    oracles = {row["task_id"]: row["nodes"] for row in read_jsonl(paths["oracles"])}
    bundles = {}
    order = []
    for row in read_jsonl(paths["specs"]):
        if release_id is not None and row.get("heldout_release_id") != release_id:
            raise RuntimeError(
                f"REFUSED: {paths['specs']} row {row.get('task_id')!r} carries "
                f"heldout_release_id {row.get('heldout_release_id')!r}, not the "
                f"revealed {release_id!r}")
        spec = TaskSpec.from_row(row)
        nodes = [OracleNode.from_row(n) for n in oracles[spec.task_id]]
        kb = kb_all[spec.task_id]
        bundles[spec.task_id] = TaskBundle(spec=spec, kb=kb, nodes=nodes,
                                          release_id=release_id)
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

    THE CANONICAL RUNTIME INPUTS ARE EXPORTED, not reconstructed. `spec_row` is
    the serialized `TaskSpec` and `oracle_nodes` is every `OracleNode.to_row()`,
    including `expect` and `match`. The evaluator builds a real
    `suite.runtime.EpisodeRuntime` from them: the flat `oracle` list below carries
    neither the canonical payloads nor the semantic matchers, so an evaluator that
    only had it would have to invent both -- which is how the tree came to hold two
    environments. `environment_contract_sha256` stamps which model-visible
    environment these bytes describe, so a resume can never mix contracts.
    """
    spec, nodes = bundle.spec, bundle.nodes
    if PHASE_OF[spec.split] == HELDOUT_PHASE and not bundle.release_id:
        raise RuntimeError(
            f"REFUSED: {spec.task_id} is a held-out task with no "
            f"heldout_release_id. Held-out certspecs may only be exported from a "
            f"verified release (load_bundles supplies the id); a row without one "
            f"is an old-seed cached value and the evaluator must reject it.")
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
        # The canonical runtime inputs (see the docstring).
        "spec_row": spec.to_row(),
        "oracle_nodes": [n.to_row() for n in nodes],
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
    if bundle.release_id is not None:
        # Which reveal these evaluator-facing bytes belong to. Task ids and file
        # names do not encode their seed, so this is the only field that
        # distinguishes the revealed set from a stale one.
        row["heldout_release_id"] = bundle.release_id
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
    from .contract import stamp

    return stamp(row)

# ---------------------------------------------------------------------------
# phase generation, the two commitments, and the release seal
# ---------------------------------------------------------------------------
# The old whole-suite writer emitted ten splits, one manifest and one SHA256SUMS
# in a single pass. That is the mechanism D3 removes: the held-out bytes cannot
# be written in the same pass as the train/dev bytes, because at that moment the
# seed they need does not exist yet.

def _split_meta(split: str, result: dict, per_cell: int, seed_value: int) -> dict:
    census = cluster_census(result["bundles"])
    meta = {
        "count": len(result["specs"]),
        "per_cell": per_cell,
        "registered_count": REGISTERED_TOTALS[split],
        "cells": [c.label for c in cells_of(split)],
        "seed": f"{seed_value:#x}",
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
            # Only the absent-information control has a hidden-entropy budget;
            # reporting 0 for a permutation would read as a defect.
            meta["min_hidden_entropy_bits"] = min(
                int((b.spec.control_meta or {}).get("hidden_entropy_bits", 0))
                for b in result["bundles"])
    if split == "eval_perm":
        meta["derangement"] = result["permutation"]
    return meta


def heldout_plan(cfg: dict) -> dict:
    """The registered held-out PLAN: names, counts, cells, allocation, derivation.

    This is what the train/dev commitment at P is allowed to contain. It fixes
    what the held-out sample WILL be without disclosing a single realized value:
    no seeds, no hashes, no cluster census, no derangement, no file metadata.
    """
    return {
        "splits": list(HELDOUT_SPLITS),
        "per_cell": {s: cfg["sizes"][s] for s in HELDOUT_SPLITS},
        "registered_totals": {s: REGISTERED_TOTALS[s] for s in HELDOUT_SPLITS},
        "cells": {s: [c.label for c in cells_of(s)] for s in HELDOUT_SPLITS},
        "template_ids": {s: list(TEMPLATE_RANGES[SPLIT_KIND[s]])
                         for s in HELDOUT_SPLITS},
        "core_eval_fault_allocation": {f"{f}-h{h}": list(w)
                                       for (f, h), w in
                                       sorted(EVAL_FAULT_WEIGHTS.items())},
        "core_eval_fault_order": list(EVAL_FAULT_ORDER),
        "registered_fault_groups": dict(EVAL_FAULT_GROUPS),
        "clean_splits": list(CLEAN_SPLITS),
        "control_splits": dict(CONTROL_SPLITS),
        "derivation": HELDOUT_DERIVATION,
        "master_label_parts": list(HELDOUT_MASTER_LABEL_PARTS),
        "split_label": HELDOUT_SPLIT_LABEL,
        "release_label": HELDOUT_RELEASE_LABEL,
        "seed_source": "L, the dedicated commit that adds the complete "
                       "results/agentic/locks.json",
        "realized_values_disclosed_here": "none, by construction: this block is "
                                          "committed at P, before L exists",
    }


def certspec_rels(phase: str) -> tuple[str, ...]:
    """The derived certspec files a phase's commitment must cover.

    Six evaluation spec manifests plus the merged eval group manifest S10 reads;
    four train/dev manifests plus the train and dev group manifests. The exporter
    (scripts/export_eval_specs.py) writes them; this says which ones belong to
    which commitment, so neither phase can seal the other's bytes.
    """
    splits = PHASES[phase]
    groups = sorted({SPLIT_KIND[s] for s in splits})
    return tuple([f"certspecs/{s}.jsonl" for s in splits]
                 + [f"certspecs/groups/{g}.jsonl" for g in groups])


def _hash_files(out_dir: str, rels) -> dict:
    files: dict = {}
    for rel in rels:
        path = os.path.join(out_dir, rel)
        files[rel] = {"sha256": file_sha256(path),
                      "bytes": os.path.getsize(path)}
    return files


def _write_commitment(out_dir: str, phase: str, body: dict, source_rels,
                      certspecs: dict | None) -> dict:
    """Write manifest.<phase>.json and SHA256SUMS.<phase> for one phase.

    `sealed` is False until the derived certspecs exist and have been hashed in.
    The validator refuses an unsealed phase, because "the suite is pinned" must
    not be claimed for the bytes the evaluator actually reads.
    """
    files = _hash_files(out_dir, source_rels)
    manifest = dict(body)
    manifest["phase"] = phase
    manifest["files"] = files
    manifest["sealed"] = certspecs is not None
    manifest["certspecs"] = certspecs if certspecs is not None else "PENDING_EXPORT"
    mpath = phase_manifest_path(out_dir, phase)
    write_json(mpath, manifest)
    listed = {rel: meta["sha256"] for rel, meta in files.items()}
    for rel, meta in (certspecs or {}).items():
        listed[rel] = meta["sha256"]
    listed[PHASE_MANIFEST[phase]] = file_sha256(mpath)
    write_text(phase_sums_path(out_dir, phase),
               "".join(f"{digest}  {rel}\n" for rel, digest in sorted(listed.items())))
    return manifest


def _generate_phase_into(cfg: dict, out_dir: str, phase: str,
                         reveal: dict | None) -> dict:
    suite = cfg["suite"]
    splits = PHASES[phase]
    release = reveal["heldout_release_id"] if phase == HELDOUT_PHASE else None
    manifest_splits: dict = {}
    all_bundles: dict = {}
    source_rels: list[str] = []

    for split in splits:
        seed_value = split_seed(cfg, split, reveal)
        per_cell = cfg["sizes"][split]
        result = build_split(suite, split, seed_value, per_cell,
                             release_id=release)
        all_bundles[split] = result["bundles"]
        paths = split_paths(out_dir, split)
        write_jsonl(paths["specs"], result["specs"])
        write_json(paths["kb"], result["kb"])
        write_jsonl(paths["oracles"], result["oracles"])
        source_rels.extend(split_rels(split))
        manifest_splits[split] = _split_meta(split, result, per_cell, seed_value)

    suite_census = assert_suite_cardinalities(all_bundles)

    body = {
        "suite": suite,
        "version": SUITE_VERSION,
        "generator": "agentlab.suite.generate",
        "generation_protocol": GENERATION_PROTOCOL,
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
    }
    if phase == TRAIN_DEV_PHASE:
        body["heldout_plan"] = heldout_plan(cfg)
        body["heldout_acceptance"] = "DEFERRED_UNTIL_POST_LOCK_REVEAL"
    else:
        body["heldout_release_id"] = release
        body["master_seed_hex"] = reveal["master_seed"].hex()
        body["derivation_label"] = HELDOUT_DERIVATION
        body["locks_commit"] = reveal["locks_commit"]
        body["preregistration_commit"] = reveal["preregistration_commit"]
        if reveal.get("reveal_sha256"):
            body["reveal_sha256"] = reveal["reveal_sha256"]
    return _write_commitment(out_dir, phase, body, source_rels, None)


def _commit_phase(staging: str, dest: str, phase: str) -> None:
    """Move one phase's freshly generated files into place, and nothing else.

    SELECTIVE replacement, not a whole-directory swap: the two phases live in one
    tree, so swapping the directory would destroy the other phase's payload and
    the derived certspecs. Every byte is generated and every cardinality assertion
    has already passed before the first destination path is touched, so a failed
    run leaves the destination exactly as it was.
    """
    rels = [PHASE_MANIFEST[phase], PHASE_SUMS[phase]]
    for split in PHASES[phase]:
        rels.extend(split_rels(split))
    for rel in rels:
        src = os.path.join(staging, rel)
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        os.replace(src, dst)


def generate_phase(cfg: dict, out_dir: str, phase: str, *,
                   reveal: dict | None = None, atomic: bool = True) -> dict:
    """Generate exactly one phase. The held-out phase needs a verified reveal.

    The refusal happens BEFORE a staging directory exists, so a failed receipt
    check emits no partial held-out files at all.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; registered: {sorted(PHASES)}")
    if phase == HELDOUT_PHASE:
        if not reveal:
            raise RuntimeError(
                "REFUSED: the held-out phase needs a verified reveal receipt "
                "(agentic_locks.py reveal). Its six seeds derive from L, the "
                "commit that adds the complete results/agentic/locks.json -- they "
                "are not in the config, there is no --seed flag, no environment "
                "seed and no fallback.")
        for key in ("heldout_release_id", "master_seed", "split_seeds",
                    "locks_commit", "preregistration_commit"):
            if not reveal.get(key):
                raise RuntimeError(f"REFUSED: the reveal receipt has no {key!r}; "
                                   f"load it through load_reveal()")
    else:
        if reveal:
            raise RuntimeError("REFUSED: the train/dev phase takes no reveal; its "
                               "seeds are the public committed ones")
        assert_no_stale_heldout(out_dir)
    dest = os.path.abspath(out_dir)
    if not atomic:
        os.makedirs(dest, exist_ok=True)
        return _generate_phase_into(cfg, dest, phase, reveal)
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)
    staging = f"{dest}.staging-{phase}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        manifest = _generate_phase_into(cfg, staging, phase, reveal)
        _commit_phase(staging, dest, phase)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def seal_phase(cfg: dict, out_dir: str, phase: str, *,
               reveal: dict | None = None) -> dict:
    """Extend a phase commitment to the derived certspecs the evaluator reads.

    Two writers, one commitment: this generator writes the source bundles, and
    scripts/export_eval_specs.py writes the certification-layer manifests from
    them. The phase is not pinned until the certspecs are hashed in, so the seal
    is a separate, idempotent step that runs after the export and refuses if a
    certspec belongs to another phase or another release.
    """
    manifest = read_phase_manifest(out_dir, phase)
    if not manifest:
        raise RuntimeError(f"REFUSED: {PHASE_MANIFEST[phase]} does not exist; "
                           f"generate the phase before sealing it")
    rels = certspec_rels(phase)
    missing = [r for r in rels if not os.path.exists(os.path.join(out_dir, r))]
    if missing:
        raise RuntimeError(
            f"REFUSED: cannot seal {phase}: the derived certspecs {missing} are "
            f"absent. Run scripts/export_eval_specs.py for exactly this phase's "
            f"splits ({' '.join(PHASES[phase])}), then seal.")
    want_splits = set(PHASES[phase])
    release = manifest.get("heldout_release_id") if phase == HELDOUT_PHASE else None
    if phase == HELDOUT_PHASE and not release:
        raise RuntimeError(f"REFUSED: {PHASE_MANIFEST[phase]} carries no release id")
    for rel in rels:
        for row in read_jsonl(os.path.join(out_dir, rel)):
            split = row.get("split")
            if split not in want_splits:
                raise RuntimeError(
                    f"REFUSED: {rel} carries split {split!r}, which is not in "
                    f"phase {phase}. A group manifest that mixes phases would "
                    f"pin held-out bytes into the train/dev commitment.")
            if release and row.get("heldout_release_id") != release:
                raise RuntimeError(
                    f"REFUSED: {rel} row {row.get('task_id')!r} carries release "
                    f"{row.get('heldout_release_id')!r}, not {release!r}; it was "
                    f"exported from other bytes than the revealed release.")
    body = {k: v for k, v in manifest.items()
            if k not in ("files", "sealed", "certspecs", "phase")}
    source_rels = list(manifest["files"])
    return _write_commitment(out_dir, phase, body, source_rels,
                             _hash_files(out_dir, rels))


def generate_all(cfg: dict, out_dir: str, *, atomic: bool = True) -> dict:
    """REFUSED. Held-out generation is post-lock; there is no one-pass suite.

    This used to emit every split from the committed config seeds, which made all
    1,200 held-out answers derivable from the preregistration commit alone --
    before any prompt winner or checkpoint existed. Keeping the name as a loud
    refusal is deliberate: a caller that still expects one pass gets an error that
    names the two phases, not a suite with a quietly re-derived held-out set.
    """
    raise RuntimeError(
        "REFUSED: generate_all() no longer exists as a behaviour. Generation is "
        "two phases with two seed sources:\n"
        "  generate_phase(cfg, out_dir, 'train-dev')\n"
        "  generate_phase(cfg, out_dir, 'heldout', reveal=load_reveal(path))\n"
        "The held-out seeds derive from L (the commit that adds the complete "
        "results/agentic/locks.json), so the held-out splits cannot be emitted in "
        "the same pass as train/dev.")


# ---------------------------------------------------------------------------
# invalidating the old-seed held-out bytes
# ---------------------------------------------------------------------------
# The tree really did contain held-out payloads and certspecs generated from the
# retired public seeds. They were never evaluated, but they are cached early
# values, and a migration that leaves them where a consumer can read them has not
# invalidated anything.

QUARANTINE_DEFAULT = os.path.join("out", "quarantine", "stale-heldout-v1")


def stale_heldout_paths(out_dir: str) -> list[str]:
    """Every canonical held-out payload/certspec path present but not released.

    "Not released" is decided by the release commitment, not by a file name: with
    a verified release the files must hash to their committed values and carry the
    release id, and with no release ANY held-out file present is stale.
    """
    present: list[str] = []
    rels: list[str] = []
    for split in HELDOUT_SPLITS:
        rels.extend(split_rels(split))
    rels.extend(certspec_rels(HELDOUT_PHASE))
    for rel in rels:
        if os.path.exists(os.path.join(out_dir, rel)):
            present.append(rel)
    if not present:
        return []
    try:
        manifest = heldout_release(out_dir)
    except RuntimeError:
        return present
    if manifest is None:
        return present
    listed = read_sums(phase_sums_path(out_dir, HELDOUT_PHASE))
    stale = []
    for rel in present:
        path = os.path.join(out_dir, rel)
        if listed.get(rel) != file_sha256(path):
            stale.append(rel)
    return stale


def assert_no_stale_heldout(out_dir: str) -> None:
    stale = stale_heldout_paths(out_dir)
    if not stale:
        return
    raise RuntimeError(
        f"REFUSED: {out_dir} still holds {len(stale)} held-out file(s) that do "
        f"not belong to a verified release, e.g. {stale[:4]}. These are cached "
        f"values from the retired public-seed derivation and no consumer may read "
        f"them.\n  Quarantine them explicitly:\n"
        f"    scripts/generate_suite.py --quarantine-stale-heldout")


def quarantine_stale_heldout(out_dir: str, dest: str = QUARANTINE_DEFAULT) -> dict:
    """Move the stale held-out bytes out of the consumed tree, with a receipt.

    Moved rather than silently deleted so the receipt can say what existed, how
    big it was and what it hashed to -- an honest record that obsolete values were
    once visible, which is a fact the protocol states rather than hides.
    """
    stale = stale_heldout_paths(out_dir)
    receipt = {
        "kind": "stale-heldout-quarantine",
        "why": "generated from the retired public held-out seeds (derivation v1, "
               "anchored on the preregistration commit). The registered "
               "derivation is now heldout-master-v2 over L, so these bytes are "
               "not the designated held-out set and cannot become it.",
        "retired_derivation": "heldout-v1 (preregistration-commit anchored)",
        "current_derivation": HELDOUT_DERIVATION,
        "source_dir": os.path.abspath(out_dir),
        "quarantined_to": os.path.abspath(dest),
        "files": {},
    }
    os.makedirs(dest, exist_ok=True)
    for rel in stale:
        src = os.path.join(out_dir, rel)
        receipt["files"][rel] = {"sha256": file_sha256(src),
                                 "bytes": os.path.getsize(src)}
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        os.replace(src, dst)
    # The retired whole-suite commitment goes with them: it listed held-out hashes
    # and claimed the entire suite was pinned at a commit where the held-out bytes
    # must not exist.
    for rel in LEGACY_COMMITMENTS:
        src = os.path.join(out_dir, rel)
        if os.path.exists(src):
            receipt["files"][rel] = {"sha256": file_sha256(src),
                                     "bytes": os.path.getsize(src)}
            dst = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            os.replace(src, dst)
    write_json(os.path.join(dest, "QUARANTINE.json"), receipt)
    return receipt
