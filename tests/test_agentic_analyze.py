"""End-to-end machine verdict: S8-S18 vetoes, ER/MT/HR gates, floors, winner.

The heavy fixture drives the REAL episode loop (scripted policies, real fault
injection, real receipts) at PREREGISTERED scale: the registered 1,200 core
specs, the three registered fault groups at 400 assigned episodes each, redacted
controls at 200 per family per arm, and 100k-replicate deterministic bootstraps
read from the frozen preregistration file. Small fixtures then break the harness
on purpose and assert that a BUG vetoes everything.

The governance tests are the other half of this file. `configs/agentic_preregister.json`
has always claimed to be authoritative while the analyzer carried its own copies
of 500, 900, +0.05, -0.03 and -0.05, so editing the governing file changed no
verdict at all. Three tests close that: no gate literal survives in the analyzer
source, every registered value is exactly the value the code used to carry, and
editing a value here really does move the verdict.
"""

import ast
import copy
import hashlib
import json
import pathlib

import pytest
from agentic_helpers import (SECRET, Guesser, ScriptedOracle, chain_spec,
                             make_arm_policy, relay_spec, run)

from agentlab import provenance
from agentlab.analyze import (agentic_verdict, load_preregister,
                             render_agentic_verdict)

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "agentic_preregister.json"
ANALYZE_PY = REPO / "src" / "agentlab" / "analyze.py"
P8 = REPO / "prompts" / "agentic" / "p8_combined.txt"
P8_SHA = hashlib.sha256(P8.read_bytes()).hexdigest()

# Registered cardinalities, read from the governing file rather than retyped.
_M = load_preregister(PREREG)["machine"]
N_PER_GROUP = int(_M["er"]["er8_min_per_group"])          # 400
N_TASKS = int(_M["registered_cardinalities"]["eval_core"])  # 1200
N_REDACTED = int(_M["registered_cardinalities"]
                 ["absent_information_per_family_per_arm"])  # 200 per family per arm
N_PERMUTED = 100   # per arm; 200 total >= the unchanged n_min of 100
LOCKED_ADAPTER = "out/adapter"


def _write(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _locks_and_reveal(results_dir: pathlib.Path):
    locks = {"checkpoint": {"path": LOCKED_ADAPTER, "stage": "rs_sft",
                            "locked_at": "2026-08-05T00:00:00Z", "commit": "aaa111"},
             "prompt_winner": {"file": "prompts/agentic/p8_combined.txt",
                               "sha256": P8_SHA,
                               "locked_at": "2026-08-05T00:00:00Z", "commit": "aaa111"}}
    seed = int.from_bytes(hashlib.sha256(
        b"deadbeef:agentic-heldout-v1").digest()[:8], "big")
    reveal = {"revealed_at": "2026-08-05T01:00:00Z",
              "preregistration_commit": "deadbeef", "heldout_seed": seed}
    (results_dir / "locks.json").write_text(json.dumps(locks))
    (results_dir / "seed_reveal.json").write_text(json.dumps(reveal))


def core_specs(n_per_group=N_PER_GROUP):
    """The registered core sample: all three fault groups at their registered size.

    ER8 gates every registered fault group at its registered cardinality, and
    wrong-unit can only be scheduled on a `unit_convert` node, so the sample has
    to carry both families by construction. Leaving the class to the
    deterministic draw would need thousands of tasks to reach 400 wrong-unit
    episodes -- the fix for an unsatisfiable gate is to generate the registered
    sample, never to lower the gate.
    """
    specs = []
    for i in range(n_per_group):
        specs.append(chain_spec(i, horizon=4, ns="eval-a",
                                fault_class="transient" if i % 2 else "rate_limit"))
    for i in range(n_per_group):
        specs.append(chain_spec(i, horizon=4, ns="eval-m", fault_class="malformed"))
    for i in range(n_per_group):
        specs.append(relay_spec(i, ns="eval-b", fault_class="wrong_unit"))
    return specs


def _run_arm_traces(specs, arm, out_dir, *, recover_pct=None):
    clean_rows, fault_rows = [], []
    for spec in specs:
        clean_rows.append(run(spec, ScriptedOracle(spec), arm=arm,
                              condition="clean", prompt_sha=P8_SHA))
        fault_rows.append(run(spec, make_arm_policy(arm, spec,
                                                    recover_pct=recover_pct),
                              arm=arm, condition="faulted", prompt_sha=P8_SHA))
    _write(out_dir / f"{arm}.clean.none.jsonl", clean_rows)
    _write(out_dir / f"{arm}.faulted.none.jsonl", fault_rows)
    return clean_rows


def _control_rows(specs, arm):
    """Redacted + permuted controls. Independent of the recovery share, so the
    two winner scenarios below share one execution of them."""
    red = []
    for family in ("lookup_chain", "typed_relay"):
        pool = [s for s in specs if s["family"] == family][:N_REDACTED]
        assert len(pool) == N_REDACTED, (family, len(pool))
        for s in pool:
            r = provenance.redact_spec(s)
            red.append(run(r, ScriptedOracle(r), arm=arm, condition="clean",
                           control="redacted", prompt_sha=P8_SHA))
    permuted = provenance.permute_hidden_values(specs, seed=2786983944)
    perm = [run(p, ScriptedOracle(p), arm=arm, condition="clean",
                control="permuted", prompt_sha=P8_SHA)
            for p in permuted[:N_PERMUTED]]
    return red, perm


@pytest.fixture(scope="module")
def registered_run(tmp_path_factory):
    """Two verdicts over one registered-scale sample.

    `trained` gives TP a real recovery edge (85 vs 30); `matched` gives both arms
    the identical recovery share, so the trained comparison genuinely fails and
    the prompted-base branch of the winner rule has to carry the run. Clean and
    control episodes do not depend on the recovery share, so they are executed
    once and written into both trace directories.
    """
    root = tmp_path_factory.mktemp("agentic-verdict")
    specs = core_specs()
    specs_path = root / "eval.jsonl"
    _write(specs_path, specs)
    train_path, dev_path = root / "train.jsonl", root / "dev.jsonl"
    _write(train_path, [chain_spec(i, split="train", ns="train-a") for i in range(20)])
    _write(dev_path, [chain_spec(i, split="dev", ns="dev-a") for i in range(20)])

    shared: dict = {}
    for arm in ("BP", "TP"):
        clean = [run(s, ScriptedOracle(s), arm=arm, condition="clean",
                     prompt_sha=P8_SHA) for s in specs]
        red, perm = _control_rows(specs, arm)
        shared[arm] = {f"{arm}.clean.none.jsonl": clean,
                       f"{arm}.clean.redacted.jsonl": red,
                       f"{arm}.clean.permuted.jsonl": perm}

    verdicts = {}
    for name, pct in (("trained", {"BP": 30, "TP": 85}),
                      ("matched", {"BP": 50, "TP": 50})):
        d = root / name
        traces = d / "traces"
        for arm in ("BP", "TP"):
            for fname, rows in shared[arm].items():
                _write(traces / fname, rows)
            _write(traces / f"{arm}.faulted.none.jsonl",
                   [run(s, make_arm_policy(arm, s, recover_pct=pct), arm=arm,
                        condition="faulted", prompt_sha=P8_SHA) for s in specs])
        secret_path = d / "secret.hex"
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.write_text(SECRET.hex())
        _locks_and_reveal(d)
        verdicts[name] = agentic_verdict(
            str(traces), str(PREREG), str(secret_path), specs_path=str(specs_path),
            split_manifests={"train": str(train_path), "dev": str(dev_path),
                             "eval": str(specs_path)},
            results_dir=str(d))
    verdicts["_specs"] = specs
    return verdicts


@pytest.fixture(scope="module")
def full_results(registered_run):
    return registered_run["trained"]


# ---------------------------------------------------------------------------
# happy path at registered scale
# ---------------------------------------------------------------------------

def test_happy_path_vetoes_all_clean(full_results):
    v = full_results
    for name, res in v["vetoes"].items():
        assert res["status"] == "OK", (name, res)
    assert not v["any_bug"]
    assert v["mandatory_inconclusive"] == []


def test_happy_path_primary_claim_passes(full_results):
    v = full_results
    for g in ("ER1", "ER2", "ER3", "ER4", "ER5", "ER6", "ER7", "ER8"):
        assert v["gates"][g]["status"] == "PASS", (g, v["gates"][g])
    assert v["claims"]["primary_certified_error_recovery"] == "PASS"
    # the bound really is the preregistered clustered LB above the margin
    assert v["gates"]["ER2"]["numbers"]["lb"] > _M["er"]["er2_margin"]
    assert v["gates"]["ER2"]["numbers"]["n_pairs"] >= _M["er"]["min_common_clean"]
    assert v["gates"]["ER5"]["numbers"]["n"] >= _M["er"]["min_assigned_faults"]


def test_er8_covers_all_three_registered_fault_groups_at_size(full_results):
    """ER8 is only a gate when every registered group is present at its size."""
    counts = full_results["gates"]["ER8"]["numbers"]["counts"]
    assert set(counts) == set(_M["er"]["er8_groups"])
    assert all(n >= N_PER_GROUP for n in counts.values()), counts


def test_happy_path_secondaries_are_inconclusive_without_their_samples(full_results):
    v = full_results
    # 400 all-tools H4 pairs is below the registered 600, and there are no H8
    # pairs at all: the analyzer must say INCONCLUSIVE, never PASS/FAIL on
    # absent or underpowered evidence.
    assert v["gates"]["MT1"]["status"] == "INCONCLUSIVE"
    assert v["gates"]["HR1"]["status"] == "INCONCLUSIVE"
    assert v["claims"]["secondary_all_tools_orchestration_H4"] == "INCONCLUSIVE"
    assert v["claims"]["secondary_H8_execution_reliability"] == "INCONCLUSIVE"


def test_happy_path_winner_is_trained_arm(full_results):
    v = full_results
    for name, res in v["floors"]["TP"].items():
        assert res["status"] == "PASS", (name, res)
    assert v["winner"].startswith("TP")
    assert v["winner_rank"] == 4


def test_prompted_base_ships_when_the_trained_comparison_fails(registered_run):
    """Rank 5 of the frozen truth table, at registered scale.

    Both arms recover the identical share, so ER2 genuinely fails rather than
    being underpowered, and every mandatory harness check is OK. BP must ship --
    that branch is a real preregistered outcome and may not quietly become
    "no successful pipeline".
    """
    v = registered_run["matched"]
    assert not v["any_bug"] and v["mandatory_inconclusive"] == []
    assert v["gates"]["ER2"]["status"] == "FAIL", v["gates"]["ER2"]
    assert all(f["status"] == "PASS" for f in v["floors"]["BP"].values())
    assert v["winner"].startswith("BP")
    assert v["winner_rank"] == 5


def test_happy_path_render_mentions_rejected_claims(full_results):
    text = render_agentic_verdict(full_results)
    assert "Winner (truth-table rank 4): TP" in text
    assert "general agentic competence" in text
    assert "H50" in text
    assert "Holm-adjusted exact McNemar" in text
    assert "Stratum census" in text


def test_curves_are_present_and_censored_not_fitted(full_results):
    cur = full_results["curves"]["TP/clean/lookup_chain"]
    assert cur["points"][0]["n"] == 2 * N_PER_GROUP  # two chain groups
    assert cur["H50"].startswith("right-censored")


# ---------------------------------------------------------------------------
# item 2: the bootstrap clusters on STRUCTURE
# ---------------------------------------------------------------------------

def test_bootstrap_clusters_on_the_structural_field_and_emits_n_clusters(full_results):
    """Every interval report names the number of structural clusters behind it."""
    assert _M["clustering"]["field"] == "template_cluster_id"
    for name in ("ER2", "ER4"):
        nums = full_results["gates"][name]["numbers"]
        assert nums["n_clusters"] >= _M["clustering"]["min_clusters_default"]
        assert nums["missing_cluster_id"] == 0
        assert nums["min_clusters"] == _M["clustering"]["min_clusters_default"]
    # Wilson gates carry the census too: an upper bound over 400 instantiations
    # of three templates is a different claim from the same bound over 80.
    for name in ("ER3", "ER6", "ER7"):
        assert "n_clusters" in full_results["gates"][name]["numbers"]


def test_too_few_clusters_is_inconclusive_not_a_narrow_interval(tmp_path):
    """A sample spanning fewer clusters than registered gets NO bound at all."""
    traces = tmp_path / "traces"
    # 600 tasks, but only three structural clusters between them
    specs = [chain_spec(i, horizon=2, n_clusters=3) for i in range(600)]
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces, recover_pct={"BP": 10, "TP": 95})
    (tmp_path / "secret.hex").write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(tmp_path / "secret.hex"),
                        results_dir=str(tmp_path))
    er2 = v["gates"]["ER2"]
    assert er2["status"] == "INCONCLUSIVE", er2
    assert "structural clusters" in er2["detail"]
    assert er2["numbers"]["n_clusters"] == 3
    # the point estimate is still on the record -- downgraded, not deleted
    assert er2["numbers"]["point"] > 0


def test_a_missing_cluster_id_is_inconclusive_never_clustered_on_task_id(tmp_path):
    """The old fallback (`template_id or task_id`) made the interval narrower."""
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(600)]
    for spec in specs:
        del spec["template_cluster_id"]
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces, recover_pct={"BP": 10, "TP": 95})
    (tmp_path / "secret.hex").write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(tmp_path / "secret.hex"),
                        results_dir=str(tmp_path))
    er2 = v["gates"]["ER2"]
    assert er2["status"] == "INCONCLUSIVE", er2
    assert "template_cluster_id" in er2["detail"]
    assert er2["numbers"]["missing_cluster_id"] == er2["numbers"]["n_pairs"]


# ---------------------------------------------------------------------------
# item 4: stratum membership
# ---------------------------------------------------------------------------

def _augmented(tmp_path, aug_split):
    """Core sample plus an oversampled augmentation stratum full of runaways."""
    traces = tmp_path / "traces"
    core = [chain_spec(i, horizon=2, ns="eval-a") for i in range(40)]
    aug = [chain_spec(i, horizon=2, ns="aug-a", split=aug_split) for i in range(400)]
    for arm in ("BP", "TP"):
        _run_arm_traces(core, arm, traces)
        rows = [run(s, Guesser("nope"), arm=arm, condition="clean",
                    prompt_sha=P8_SHA) for s in aug]
        _write(traces / f"{arm}.clean.none.jsonl", rows)
    (tmp_path / "secret.hex").write_text(SECRET.hex())
    return agentic_verdict(str(traces), str(PREREG), str(tmp_path / "secret.hex"),
                           results_dir=str(tmp_path))


def test_augmentation_traces_never_enter_core_denominators(tmp_path):
    """ER6/ER7 and every floor read the CORE stratum only.

    400 failing H8-augmentation episodes against 40 core ones would otherwise
    swamp the core clean floor and the core hallucination rate -- i.e. the SIZE
    of a secondary sample could move the primary claim and the winner.
    """
    v = _augmented(tmp_path, "eval_h8")
    # 40 core clean + 40 core faulted TP episodes, and not one of the 400 others
    assert v["gates"]["ER6"]["numbers"]["n"] == 80, v["gates"]["ER6"]
    assert v["floors"]["TP"]["F1"]["numbers"]["n"] == 40
    assert v["floors"]["TP"]["F1"]["numbers"]["rate"] == 1.0
    census = v["strata_census"]["counts"]
    assert census["TP/clean/none/eval"] == 40
    assert census["TP/clean/none/eval_h8"] == 400


def test_an_undeclared_split_is_excluded_and_reported(tmp_path):
    v = _augmented(tmp_path, "eval_mystery")
    assert v["gates"]["ER6"]["numbers"]["n"] == 80
    assert v["strata_census"]["unassigned_split_traces"][
        "TP/clean/none/eval_mystery"] == 400


# ---------------------------------------------------------------------------
# item 6: Holm, median convention, frozen bootstrap block
# ---------------------------------------------------------------------------

def test_holm_adjusted_secondary_mcnemar_values_are_emitted(full_results):
    """Specified since v1 and never produced before."""
    h = full_results["holm_secondary_mcnemar"]
    assert h["family"] == ["MT1", "HR1"]
    assert h["p_kind"] == "p_one_sided"
    assert h["reported_never_gated"] is True
    assert set(h["adjusted"]) == {"MT1", "HR1"}
    # Holm is monotone in the raw p-values and never reduces one
    for name in h["family"]:
        assert h["adjusted"][name] >= h["raw"][name] - 1e-12
        assert h["adjusted"][name] <= 1.0
    # an absent secondary enters the family at p=1.0 rather than being dropped
    assert h["raw"]["HR1"] == 1.0


def test_median_convention_is_registered_and_governs():
    from agentlab.analyze import _median

    assert _M["mt"]["mt3_median_convention"] == "upper_middle"
    assert _median([1, 2, 3, 4], "upper_middle") == 3
    assert _median([1, 2, 3, 4], "lower_middle") == 2
    assert _median([1, 2, 3], "upper_middle") == 2
    with pytest.raises(ValueError):
        _median([1, 2], "midhinge")


def test_bootstrap_block_size_is_frozen_in_json_because_it_moves_the_stream():
    """The chunk size is not an implementation detail.

    Each chunk draws from its own SHAKE label (`{label}:chunk{i}`), so the
    REALISED replicate stream depends on the block size; the bound is an order
    statistic of that stream. It was an unregistered internal constant, which
    meant a bound could change without any governing value changing. It is now a
    numeric field in the preregistration, read by the analyzer.
    """
    from agentlab.suite.rng import choice_indices
    from agentlab.suite.stats import cluster_bootstrap_lb

    cfg = load_preregister(PREREG)["statistics"]["clustered_bootstrap"]
    assert cfg["chunk_block"] == 2000

    # A different block size partitions the replicates into different chunks, so
    # replicate 2000 onward is drawn from a different labelled stream.
    k, block = 40, cfg["chunk_block"]
    at_registered_block = list(choice_indices(7, f"x:chunk1", block * k, k))
    at_other_block = list(choice_indices(7, f"x:chunk2", 997 * k, k))
    assert at_registered_block[:100] != at_other_block[:100]

    diffs = [1.0 if (i % 7) else 0.0 for i in range(400)]
    clusters = [f"c{i % k:02d}" for i in range(400)]
    a = cluster_bootstrap_lb(diffs, clusters, seed=7, label="x", replicates=4000,
                             block=block)
    again = cluster_bootstrap_lb(diffs, clusters, seed=7, label="x",
                                 replicates=4000, block=block)
    assert a["lb"] == again["lb"], "the registered block must be reproducible"
    with pytest.raises(ValueError):
        cluster_bootstrap_lb(diffs, clusters, seed=7, label="x", replicates=10,
                             block=0)


# ---------------------------------------------------------------------------
# item 1: the JSON really governs
# ---------------------------------------------------------------------------

# Every threshold the council found hard-coded in the analyzer, with the exact
# value it carried. This table is the proof that the governance fix moved values
# without changing any of them.
FROZEN_VALUES = [
    (("er", "min_common_clean"), 500),
    (("er", "er2_margin"), 0.05),
    (("er", "er3_wilson_lb"), 0.60),
    (("er", "er4_margin"), -0.03),
    (("er", "min_assigned_faults"), 900),
    (("er", "er5_min_diff"), 0.0),
    (("er", "er6_runaway_ub"), 0.03),
    (("er", "er7_hallucination_ub"), 0.01),
    (("er", "er8_group_floor"), -0.05),
    (("mt", "horizon"), 4),
    (("mt", "min_pairs"), 600),
    (("mt", "mt1_margin"), 0.05),
    (("mt", "mt2_wilson_lb"), 0.60),
    (("mt", "mt3_oracle_min_calls"), 4),
    (("mt", "mt3_slack"), 2),
    (("mt", "mt4_runaway_ub"), 0.03),
    (("mt", "mt4_hallucination_ub"), 0.01),
    (("mt", "mt5_n_patterns"), 6),
    (("mt", "mt5_min_per_pattern"), 80),
    (("mt", "mt5_pattern_floor"), -0.05),
    (("hr", "horizon"), 8),
    (("hr", "min_pairs"), 400),
    (("hr", "min_discordant"), 20),
    (("hr", "hr1_margin"), 0.05),
    (("hr", "hr2_runaway_ub"), 0.03),
    (("hr", "hr3_hallucination_ub"), 0.01),
    (("floors", "f1_clean_overall"), 0.65),
    (("floors", "f2_clean_per_family"), 0.50),
    (("floors", "f3_faulted_overall"), 0.40),
    (("floors", "f4_faulted_per_family"), 0.25),
    (("floors", "f5_loop_crash_max"), 0.02),
    (("controls_checks", "s14_generation_check_cap"), 200),
    (("curves", "h50_crossing_rate"), 0.5),
]


def test_every_registered_threshold_holds_its_unchanged_value():
    """The governance fix moved thresholds out of code without editing one."""
    m = load_preregister(PREREG)["machine"]
    for path, want in FROZEN_VALUES:
        node = m
        for key in path:
            assert key in node, f"machine.{'.'.join(path)} is missing"
            node = node[key]
        assert node == want, f"machine.{'.'.join(path)} is {node}, registered {want}"
    assert m["hr"]["families"] == ["lookup_chain", "typed_relay"]
    assert m["er"]["er8_groups"] == ["transient_or_rate_limit", "malformed",
                                     "wrong_unit"]


# Values that must never appear as a literal in the agentic analyzer again: the
# thresholds, margins, floors, sample sizes and rate ceilings the council found
# expressed in code (analyze.py:978 min_c=500, :1017 min_assigned=900, and the
# literal margins at :988, :1012, :1064), plus the bootstrap block size.
FORBIDDEN_LITERALS = {500, 900, 600, 400, 300, 240, 80, 20, 2000, 1152,
                      0.05, 0.03, 0.01, 0.02, 0.6, 0.60, 0.65, 0.40, 0.25,
                      -0.03, -0.05, 36.0}
# Structurally innocent numbers: list/byte slices, `n // 2`, string padding,
# truth-table rank ids, and the 0.0/1.0 identities of a probability.
ALLOWED_LITERALS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 0.0, 1.0}


def _agentic_numeric_literals():
    src = ANALYZE_PY.read_text(encoding="utf-8")
    first_line = src[:src.index("# agentic verdict (multifaceted suite)")].count("\n") + 1
    out = {}
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Constant) and not isinstance(node.value, bool)
                and isinstance(node.value, (int, float))
                and node.lineno >= first_line):
            out.setdefault(node.value, []).append(node.lineno)
    return out


def test_no_gate_threshold_is_expressed_in_the_analyzer_source():
    """JSON/code parity: the analyzer may not carry its own copy of a threshold.

    This is the test that would have caught the original defect. The protocol
    declared configs/agentic_preregister.json authoritative while analyze.py
    carried min_c = 500, min_assigned = 900 and the literal margins +0.05, -0.03
    and -0.05, so changing the governing JSON changed not one verdict.
    """
    literals = _agentic_numeric_literals()
    leaked = {v: lines for v, lines in literals.items() if v in FORBIDDEN_LITERALS}
    assert not leaked, (f"gate thresholds expressed in analyze.py: {leaked}; "
                        f"move them into machine.* and read them via registered()")
    stray = {v: lines for v, lines in literals.items() if v not in ALLOWED_LITERALS}
    assert not stray, (f"unexplained numeric literals in the agentic analyzer: "
                       f"{stray}; a gate number belongs in the preregistration, "
                       f"and a structural constant belongs in ALLOWED_LITERALS "
                       f"with a reason")


def test_stats_module_carries_no_unregistered_gate_number():
    """The bootstrap's alpha, replicates and block all come from the caller."""
    src = (REPO / "src" / "agentlab" / "suite" / "stats.py").read_text()
    tree = ast.parse(src)
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cluster_bootstrap_lb":
            for arg, default in zip(node.args.kwonlyargs + node.args.args[-3:],
                                    node.args.defaults):
                defaults[arg.arg] = getattr(default, "value", None)
    # the defaults mirror the registered values; the preregistered path always
    # passes them explicitly, so the mirror can never silently diverge unnoticed
    cfg = load_preregister(PREREG)["statistics"]["clustered_bootstrap"]
    assert defaults["replicates"] == cfg["replicates"]
    assert defaults["alpha"] == cfg["alpha_one_sided"]
    assert defaults["block"] == cfg["chunk_block"]


@pytest.fixture(scope="module")
def governance_traces(tmp_path_factory):
    """A small but complete run reused by every JSON-governance mutation."""
    root = tmp_path_factory.mktemp("governance")
    traces = root / "traces"
    specs = core_specs(n_per_group=40)   # all three fault groups, 120 tasks
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces, recover_pct={"BP": 20, "TP": 90})
    specs_path = root / "eval.jsonl"
    _write(specs_path, specs)
    (root / "secret.hex").write_text(SECRET.hex())
    return {"traces": str(traces), "secret": str(root / "secret.hex"),
            "specs": str(specs_path), "root": str(root)}


def _scale_down(prereg: dict) -> None:
    """Lower only the SAMPLE-SIZE minimums, in a throwaway copy of the JSON.

    The mutation tests below prove that each decision threshold is really read
    from the file. To do that on a 120-task fixture the registered sample-size
    minimums have to be satisfiable, so they are lowered HERE, in a temp copy
    that is never committed and never used by any other test. The committed
    values are pinned separately by
    `test_every_registered_threshold_holds_its_unchanged_value`, and the
    sample-size fields are themselves mutation-tested below.
    """
    m = prereg["machine"]
    m["er"]["min_common_clean"] = 10
    m["er"]["min_assigned_faults"] = 10
    m["er"]["er8_min_per_group"] = 1


def _verdict_with(tmp_path, governance_traces, mutate):
    """Run the verdict against a MUTATED copy of the governing preregistration."""
    prereg = load_preregister(PREREG)
    prereg = copy.deepcopy(prereg)
    # replicates is itself a registered field; lowering it in a throwaway copy
    # keeps these mutations fast and touches nothing committed
    prereg["statistics"]["clustered_bootstrap"]["replicates"] = 400
    mutate(prereg)
    p = tmp_path / "mutated_preregister.json"
    p.write_text(json.dumps(prereg))
    return agentic_verdict(governance_traces["traces"], str(p),
                           governance_traces["secret"],
                           specs_path=governance_traces["specs"],
                           results_dir=governance_traces["root"])


@pytest.mark.parametrize("path,value,gate,before,after", [
    (("er", "min_common_clean"), 5000, "ER1", "PASS", "INCONCLUSIVE"),
    (("er", "er2_margin"), 0.99, "ER2", "PASS", "FAIL"),
    (("er", "er3_wilson_lb"), 0.95, "ER3", "PASS", "FAIL"),
    # a floor that even k=n cannot reach at this n is a sample-size statement,
    # never a FAIL blamed on the policy -- and that rule is threshold-driven too
    (("er", "er3_wilson_lb"), 0.999, "ER3", "PASS", "INCONCLUSIVE"),
    (("er", "er4_margin"), 0.99, "ER4", "PASS", "FAIL"),
    (("er", "min_assigned_faults"), 5000, "ER5", "PASS", "INCONCLUSIVE"),
    (("er", "er8_group_floor"), 0.99, "ER8", "PASS", "FAIL"),
    (("floors", "f1_clean_overall"), 1.01, "F1", "PASS", "FAIL"),
    (("floors", "f3_faulted_overall"), 1.01, "F3", "PASS", "FAIL"),
    (("floors", "f5_loop_crash_max"), 0.0, "F5", "PASS", "FAIL"),
])
def test_editing_a_registered_threshold_changes_the_verdict(
        tmp_path, governance_traces, path, value, gate, before, after):
    """The whole point of the governance fix: the JSON is not decoration.

    Before this, editing any of these fields changed nothing at all, because the
    analyzer carried its own copy of the number.
    """
    def base(p):
        _scale_down(p)

    def mutated(p):
        _scale_down(p)
        node = p["machine"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    def status(v):
        return (v["gates"].get(gate) or v["floors"]["TP"][gate])["status"]

    assert status(_verdict_with(tmp_path, governance_traces, base)) == before
    assert status(_verdict_with(tmp_path, governance_traces, mutated)) == after


def test_a_missing_registered_field_is_loud_not_defaulted(tmp_path,
                                                          governance_traces):
    """No silent fallback: a missing threshold must never become a literal."""
    def drop(p):
        del p["machine"]["er"]["er2_margin"]

    with pytest.raises(KeyError, match="machine.er.er2_margin"):
        _verdict_with(tmp_path, governance_traces, drop)


# ---------------------------------------------------------------------------
# small adversarial fixtures: BUG vetoes everything
# ---------------------------------------------------------------------------

def _mini(tmp_path, mutate=None, arms=("BP", "TP"), n=12):
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(n)]
    rows_by_file = {}
    for arm in arms:
        rows_by_file[f"{arm}.clean.none.jsonl"] = [
            run(s, ScriptedOracle(s), arm=arm, condition="clean",
                prompt_sha=P8_SHA) for s in specs]
        rows_by_file[f"{arm}.faulted.none.jsonl"] = [
            run(s, ScriptedOracle(s), arm=arm, condition="faulted",
                prompt_sha=P8_SHA) for s in specs]
    if mutate:
        mutate(rows_by_file, specs)
    for name, rows in rows_by_file.items():
        _write(traces / name, rows)
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    return agentic_verdict(str(traces), str(PREREG), str(secret_path),
                           results_dir=str(tmp_path))


def test_underpowered_common_clean_is_inconclusive_not_pass(tmp_path):
    v = _mini(tmp_path)
    assert v["gates"]["ER1"]["status"] == "INCONCLUSIVE"
    assert v["gates"]["ER2"]["status"] == "INCONCLUSIVE"
    assert v["claims"]["primary_certified_error_recovery"] == "INCONCLUSIVE"


def test_s8_pairing_mismatch_is_a_bug(tmp_path):
    def mutate(rows, specs):
        rows["TP.clean.none.jsonl"] = rows["TP.clean.none.jsonl"][:-1]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S8"]["status"] == "BUG"
    assert v["any_bug"]
    assert all(s == "BUG" for s in v["claims"].values())
    assert "NO VERDICT" in v["winner"]


def test_s11_redacted_success_is_a_bug(tmp_path):
    def mutate(rows, specs):
        red = provenance.redact_spec(specs[0])
        leak = run(red, Guesser(specs[0]["answer"]), arm="TP", condition="clean",
                   control="redacted", prompt_sha=P8_SHA)
        rows["TP.clean.redacted.jsonl"] = [leak]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S11"]["status"] == "BUG"
    assert v["any_bug"] and "NO VERDICT" in v["winner"]


def test_s11_unexecuted_redacted_control_is_a_bug_not_a_pass(tmp_path):
    """A control that never ran must never read as a clean control.

    Before the fix every redacted episode looked like this row: the runtime
    rejected the deliberately-broken oracle, so the policy made zero decisions and
    S11 reported "zero raw and certified success" over episodes that had never
    been attempted.
    """
    def mutate(rows, specs):
        red = provenance.redact_spec(specs[0])
        stub = run(red, Guesser("whatever"), arm="TP", condition="clean",
                   control="redacted", prompt_sha=P8_SHA)
        stub["messages"], stub["events"] = [], []
        stub["runner"] = {"n_decisions": 0, "n_calls": 0,
                          "termination_reason": "spec_error", "wall_s": 0.0}
        stub["score"] = {"raw_success": False, "certified_success": False,
                         "runaway": False, "hallucinated": False}
        rows["TP.clean.redacted.jsonl"] = [stub]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S11"]["status"] == "BUG", v["vetoes"]["S11"]
    assert "never ran" in v["vetoes"]["S11"]["detail"]
    assert v["any_bug"] and "NO VERDICT" in v["winner"]


def test_s11_coverage_is_counted_per_family_per_arm(registered_run):
    """Pooling arms would let 400 redacted TP episodes stand in for 200 each."""
    counts = registered_run["trained"]["vetoes"]["S11"]["numbers"]["counts"]
    assert set(counts) == {"lookup_chain/BP", "lookup_chain/TP",
                           "typed_relay/BP", "typed_relay/TP"}
    assert all(n >= N_REDACTED for n in counts.values()), counts


def test_a_single_s_check_bug_leaves_no_official_pass_anywhere(tmp_path):
    """One harness BUG => no official PASS survives in the whole verdict.

    The council observed the opposite: in an S8-BUG fixture individual gates
    still printed PASS/FAIL/INCONCLUSIVE and every launch floor printed PASS, so
    "a BUG vetoes everything below" was not true at the gate/floor output level.
    """
    def mutate(rows, specs):
        row = rows["TP.clean.none.jsonl"][0]
        row["events"][0]["receipt"] = "r-" + "0" * 32   # forge one receipt: S13
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S13"]["status"] == "BUG"

    official = []
    for name, res in v["vetoes"].items():
        official.append((f"veto {name}", res["status"]))
    for name, res in v["gates"].items():
        official.append((f"gate {name}", res["status"]))
    for arm, floors in v["floors"].items():
        for name, res in floors.items():
            official.append((f"floor {arm}/{name}", res["status"]))
    for claim, status in v["claims"].items():
        official.append((f"claim {claim}", status))
    assert not [x for x in official if x[1] == "PASS"], \
        f"official PASS survived a harness BUG: {[x for x in official if x[1] == 'PASS']}"
    assert {s for _, s in official} <= {"BUG", "OK", "INCONCLUSIVE"}
    assert all(g["status"] == "BUG" for g in v["gates"].values())
    assert "PASS" not in v["winner"] and "NO VERDICT" in v["winner"]
    # and the rendered report a human reads shows no PASS gate either
    text = render_agentic_verdict(v)
    assert "\n  PASS " not in text


def test_a_harness_bug_vetoes_every_gate_and_floor_without_erasing_them(tmp_path):
    """Relabel, never delete: the arithmetic survives as a NON-VERDICT field."""
    def mutate(rows, specs):
        rows["TP.clean.none.jsonl"] = rows["TP.clean.none.jsonl"][:-1]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S8"]["status"] == "BUG"
    assert {g["status"] for g in v["gates"].values()} == {"BUG"}
    assert {f["status"] for arm in v["floors"].values() for f in arm.values()} == {"BUG"}
    # nothing is hidden: what each gate observed survives for the record, under a
    # field name that can never be mistaken for a verdict
    assert all("observed_status" in g for g in v["gates"].values())
    assert all("observed_status" in f
               for arm in v["floors"].values() for f in arm.values())
    assert all("S8" in g["detail"] for g in v["gates"].values())
    assert all("NOT a verdict" in g["detail"] for g in v["gates"].values())
    assert "NO VERDICT" in v["winner"] and v["winner_rank"] == 1


def test_no_gate_silently_degrades_a_fail_into_inconclusive(tmp_path):
    """Underpowered gates say INCONCLUSIVE, but a measured FAIL stays on the record.

    Also pins the companion rule: an interval gate whose threshold is unreachable
    at the observed n is INCONCLUSIVE (a sample-size statement), never a FAIL
    blamed on the policy.
    """
    v = _mini(tmp_path)  # n=12: far below every preregistered sample size
    for name in ("ER2", "ER3", "ER6", "ER7"):
        assert v["gates"][name]["status"] == "INCONCLUSIVE", (name, v["gates"][name])
    # ER6/ER7 are unmeasurable at n=24, and they say so with the arithmetic
    assert "unmeasurable at n=" in v["gates"]["ER7"]["detail"]
    assert v["gates"]["ER7"]["numbers"]["best_possible_ub"] > _M["er"][
        "er7_hallucination_ub"]
    # a downgrade never deletes the measurement it downgraded
    for name in ("ER2", "ER3"):
        g = v["gates"][name]
        if "measured_status" in g:
            assert g["measured_status"] in ("PASS", "FAIL")
            assert "measured anyway for the record" in g["detail"]
    # and INCONCLUSIVE never reads as support
    assert v["claims"]["primary_certified_error_recovery"] == "INCONCLUSIVE"
    assert not v["winner"].startswith("TP")


def test_a_real_fail_outranks_missing_evidence_in_a_claim(tmp_path):
    """One refuted gate refutes the claim even when other gates are unmeasured.

    ER4 (clean non-inferiority) is the refuted gate here: it has no sample-size
    precondition, only the registered cluster minimum, so it can FAIL honestly at
    a fixture size where every ER sample-size gate is INCONCLUSIVE. Aggregate
    precedence must then be FAIL, not INCONCLUSIVE -- the council observed the
    opposite, an aggregate reading INCONCLUSIVE while two of its gates FAILed.
    """
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2, n_clusters=40) for i in range(120)]
    for arm in ("BP", "TP"):
        for condition in ("clean", "faulted"):
            rows = [run(s,
                        ScriptedOracle(s) if arm == "BP" else Guesser("wrong"),
                        arm=arm, condition=condition, prompt_sha=P8_SHA)
                    for s in specs]
            _write(traces / f"{arm}.{condition}.none.jsonl", rows)
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(secret_path),
                        results_dir=str(tmp_path))
    assert not v["any_bug"]
    assert v["gates"]["ER4"]["status"] == "FAIL", v["gates"]["ER4"]
    assert v["gates"]["ER4"]["numbers"]["n_clusters"] >= _M["clustering"][
        "min_clusters_default"]
    assert any(g["status"] == "INCONCLUSIVE" for g in v["gates"].values())
    assert v["claims"]["primary_certified_error_recovery"] == "FAIL"
    assert not v["winner"].startswith("TP")


def test_er8_short_group_is_inconclusive_not_a_pass_by_omission(tmp_path):
    """Two of three registered groups is not the registered gate."""
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2, ns="eval-a", fault_class="transient")
             for i in range(40)]
    for arm in ("BP", "TP"):
        _run_arm_traces(specs, arm, traces)
    (tmp_path / "secret.hex").write_text(SECRET.hex())
    v = agentic_verdict(str(traces), str(PREREG), str(tmp_path / "secret.hex"),
                        results_dir=str(tmp_path))
    er8 = v["gates"]["ER8"]
    assert er8["status"] == "INCONCLUSIVE", er8
    assert er8["numbers"]["counts"]["wrong_unit"] == 0
    assert er8["numbers"]["min_per_group"] == N_PER_GROUP


def test_s13_forged_receipt_is_a_bug(tmp_path):
    def mutate(rows, specs):
        row = rows["TP.clean.none.jsonl"][0]
        row["events"][0]["receipt"] = "r-" + "0" * 32
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S13"]["status"] == "BUG"
    assert v["any_bug"]


def test_s17_score_tampering_is_a_bug(tmp_path):
    def mutate(rows, specs):
        row = rows["TP.faulted.none.jsonl"][0]
        row["score"]["certified_success"] = not row["score"]["certified_success"]
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S17"]["status"] == "BUG"
    assert v["any_bug"]


def test_s16_adapter_in_prompt_only_arm_is_a_bug(tmp_path):
    def mutate(rows, specs):
        for row in rows["BP.clean.none.jsonl"]:
            row["provenance"]["adapter"] = "out/some-adapter"
    v = _mini(tmp_path, mutate)
    assert v["vetoes"]["S16"]["status"] == "BUG"


def test_s16_requires_exactly_one_locked_checkpoint_mapping_to_tp(tmp_path):
    """A TP arm that is not demonstrably the locked checkpoint is unattributable."""
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(4)]
    for arm in ("BP", "TP"):
        _write(traces / f"{arm}.clean.none.jsonl",
               [run(s, ScriptedOracle(s), arm=arm, condition="clean",
                    prompt_sha=P8_SHA) for s in specs])
    (tmp_path / "secret.hex").write_text(SECRET.hex())
    _locks_and_reveal(tmp_path)

    def verdict():
        return agentic_verdict(str(traces), str(PREREG),
                               str(tmp_path / "secret.hex"),
                               results_dir=str(tmp_path))

    assert verdict()["vetoes"]["S16"]["status"] == "OK"
    for bad in ({"path": LOCKED_ADAPTER, "stage": "handpicked"},
                {"path": "out/a-different-adapter", "stage": "rs_sft"},
                {"stage": "rs_sft"}):
        locks = json.loads((tmp_path / "locks.json").read_text())
        locks["checkpoint"] = dict(bad, locked_at="2026-08-05T00:00:00Z")
        (tmp_path / "locks.json").write_text(json.dumps(locks))
        res = verdict()["vetoes"]["S16"]
        assert res["status"] == "BUG", (bad, res)


def test_s18_reveal_before_lock_is_a_bug(tmp_path):
    traces = tmp_path / "traces"
    specs = [chain_spec(i, horizon=2) for i in range(4)]
    for arm in ("BP", "TP"):
        _write(traces / f"{arm}.clean.none.jsonl",
               [run(s, ScriptedOracle(s), arm=arm, condition="clean",
                    prompt_sha=P8_SHA) for s in specs])
    secret_path = tmp_path / "secret.hex"
    secret_path.write_text(SECRET.hex())
    _locks_and_reveal(tmp_path)
    reveal = json.loads((tmp_path / "seed_reveal.json").read_text())
    reveal["revealed_at"] = "2025-01-01T00:00:00Z"  # before the locks
    (tmp_path / "seed_reveal.json").write_text(json.dumps(reveal))
    v = agentic_verdict(str(traces), str(PREREG), str(secret_path),
                        results_dir=str(tmp_path))
    assert v["vetoes"]["S18"]["status"] == "BUG"


# ---------------------------------------------------------------------------
# item 4: mandatory vetoes block the winner
# ---------------------------------------------------------------------------

def test_a_mandatory_inconclusive_veto_blocks_the_winner(tmp_path):
    """Rank 2 of the frozen truth table.

    The analyzer used to ship a winner while S9, S10, S14 or S18 was
    INCONCLUSIVE -- i.e. while oracle reachability, split isolation, the
    counterfactual control or test-blindness was simply unverified. Only BUG
    blocked the winner. Now an unverified mandatory check means NO VERDICT.
    """
    v = _mini(tmp_path, n=40)          # no specs, no splits, no locks supplied
    assert not v["any_bug"]
    assert set(v["mandatory_inconclusive"]) >= {"S9", "S10", "S15", "S18"}
    assert v["winner_rank"] == 2
    assert "NO VERDICT" in v["winner"]
    assert "S9" in v["winner"]
    assert "MANDATORY HARNESS CHECKS UNVERIFIED" in render_agentic_verdict(v)


def test_every_declared_mandatory_check_is_implemented(full_results):
    declared = set(_M["mandatory_harness_checks"])
    assert declared == set(full_results["vetoes"])
    assert declared == set(full_results["mandatory_harness_checks"])
