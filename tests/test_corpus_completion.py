"""The training-corpus seams: one answer grammar, and completion means a receipt.

Four defects, all of them silent:

  1. THE ANSWER GRAMMAR. `suite.datasets.select_views` asked "is there a committed
     final answer" with a local `\\boxed{}`-only regex, and
     `multidistill.accept_record` asked the same question with `boxed_answer`.
     A trajectory the strict verifier CERTIFIED that terminated with the
     preregistered system prompt's plain `ANSWER: <value>` form was therefore
     rejected as `no_box` one layer up, and would have produced ZERO SFT views one
     layer down. Both now delegate to `schema.extract_committed_answer` -- the one
     grammar the verifier and the certification layer already use.

  2. A QUOTA MISS EXITED ZERO. `finalize` printed a warning AFTER writing
     accepted.jsonl, so the next invocation of the chain saw the file, skipped the
     stage, and trained on a corpus that had failed its preregistered minimum.

  3. THE ROW RANGE WAS REPORTED, NOT ENFORCED. `views.expected_rows` (5,000-6,000)
     and the terminal-weight floor were written into the report and never checked.

  4. COMPLETION MEANT A PATH EXISTS. A shard holding three of the 384 rollouts it
     owed, a zero-row artifact, a corpus mutated after it was summarized, and a
     view corpus whose row file was published before its metadata all passed as
     finished work.

Nothing here touches a GPU. The rollouts are real -- the canonical runtime, real
receipts, the real strict verifier -- with a CPU producer attestation, which is
exactly the shape the real seam hands these consumers.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from rollout_helpers import TEST_SECRET, run_engine, token_counter_stub

from agentlab import multidistill as md
from agentlab.suite import configio
from agentlab.suite import contract as contract_mod
from agentlab.suite import datasets as ds
from agentlab.suite.generate import build_task
from agentlab.suite.schema import canon, digest_text, extract_committed_answer

CFG = configio.load_config()
SUITE = "agentlab-suite-v1"
SEED = 0xA61E0001          # the committed distill seed


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _bundle(family="lookup_chain", horizon=4, index=11, faults=None):
    return build_task(SUITE, SEED, "distill", family, horizon, index, faults)


def _rollout(*, terminal_text=None, terminal_fmt=None, family="lookup_chain",
             horizon=4, index=11, faults=None):
    """One real rollout, optionally with a scripted terminal turn.

    `terminal_fmt` is formatted with the committed answer, so a test can script
    the exact commitment form under discussion.
    """
    from rollout_helpers import OraclePolicy

    bundle = _bundle(family, horizon, index, faults)
    if terminal_fmt is not None:
        terminal_text = terminal_fmt.format(answer=bundle.spec.answer)
    policy = OraclePolicy([bundle], terminal_text=terminal_text)
    rec = run_engine([bundle], policy=policy, cfg=CFG, secret=TEST_SECRET)[0]
    return bundle, rec


def _cpu_attested(records: list) -> list:
    """The explicit non-GPU producer attestation every claim-bearing row needs."""
    prov = md.cpu_provenance("test-corpus-completion", CFG)
    for rec in records:
        rec["provenance"] = dict(prov)
    return records


def _paths(tmp_path, monkeypatch) -> dict:
    """Point every artifact this module writes at tmp_path."""
    raw = tmp_path / "raw"
    accepted = tmp_path / "accepted.jsonl"
    monkeypatch.setattr(md, "MULTIFACE_DIR", tmp_path)
    monkeypatch.setattr(md, "RAW_DIR", raw)
    monkeypatch.setattr(md, "ACCEPTED_PATH", accepted)
    monkeypatch.setattr(md, "SUMMARY_PATH", tmp_path / "rs_summary.json")
    monkeypatch.setattr(md, "FAILURE_PATH", tmp_path / "rs_finalize_failure.json")
    return {"raw": raw, "accepted": accepted,
            "receipt": md.accepted_receipt_path(accepted),
            "summary": tmp_path / "rs_summary.json",
            "failure": tmp_path / "rs_finalize_failure.json"}


# ---------------------------------------------------------------------------
# 1. one answer grammar: a plain `ANSWER: <code>` trajectory is trainable
# ---------------------------------------------------------------------------

def test_a_plain_answer_line_trajectory_yields_views():
    """THE defect: certified success, and the view builder returned nothing.

    The rollout below obeys the preregistered system prompt and nothing else: it
    terminates with `ANSWER: <access code>`, no `\\boxed{}` anywhere. The strict
    verifier certifies it. Before the fix `select_views` returned `[]` and
    `build_views` tallied it as a trajectory with no terminal -- so every
    trajectory that followed the system prompt alone left the corpus silently.
    """
    _bundle_, rec = _rollout(terminal_fmt="ANSWER: {answer}")
    assert rec["verdict"]["certified_success"], "the premise: this IS success"
    final = rec["messages"][-1]["content"]
    assert "\\boxed" not in final and final.startswith("ANSWER: ")

    plan = ds.select_views(rec, CFG)
    assert plan, "a certified trajectory committing `ANSWER: x` must yield views"
    assert [i for i in plan if i["view"] == "terminal"], "the terminal view is owed"

    rows, meta, report = ds.build_views(_cpu_attested([rec]), token_counter_stub(),
                                        CFG)
    assert rows and len(meta) == len(rows)
    assert report["rejected"]["no_committed_answer"] == 0
    assert report["view_counts"]["terminal"] == CFG["views"]["terminal_copies"]
    # the completion trained on is the committed answer itself
    assert any(r["completion"][0]["content"] == final for r in rows)


def test_the_acceptance_filter_admits_the_same_commitment():
    """The layer ABOVE the views asked the same question with its own regex.

    `accept_record` rejected the identical trajectory as `no_box`, so the view fix
    alone would never have seen a plain-`ANSWER:` trajectory in production.
    """
    bundle, rec = _rollout(terminal_fmt="ANSWER: {answer}")
    ok, why = md.accept_record(rec, CFG, {bundle.spec.task_id: bundle},
                               secret=TEST_SECRET)
    assert ok, why


@pytest.mark.parametrize("final,commits", [
    ("ANSWER: 55640a29c0a34d0f", True),          # the system prompt's form
    ("Done. The answer is \\boxed{55640a29c0a34d0f}", True),   # the task prompt's
    ("ANSWER: \\boxed{55640a29c0a34d0f}", True),               # both at once
    ("answer : 55640a29c0a34d0f", True),                       # the shared reader
    ("the code is 55640a29c0a34d0f", False),                   # never decided
    ("", False),
])
def test_the_view_builder_has_no_answer_grammar_of_its_own(final, commits):
    """`datasets.committed_answer` IS the shared extractor, case for case."""
    assert (ds.committed_answer(final) is not None) is commits
    assert ds.committed_answer(final) == extract_committed_answer(final)
    assert md.committed_answer(final) == extract_committed_answer(final)
    assert not hasattr(ds, "_BOXED_RE"), \
        "a second answer regex in the view builder is the defect itself"


def test_a_trajectory_that_never_commits_still_yields_nothing():
    """The fix widens the GRAMMAR, not the rule: no commitment, no supervision."""
    _bundle_, rec = _rollout(terminal_text="the code is right here somewhere")
    assert ds.select_views(rec, CFG) == []
    rows, meta, report = ds.build_views(_cpu_attested([rec]), token_counter_stub(),
                                        CFG)
    assert rows == [] and meta == []
    assert report["rejected"]["no_committed_answer"] == 1
    # ... and the drop is attributed to the stratum it happened in
    cell = ds.stratum_of(rec["family"], rec["horizon"])
    assert report["strata"][cell]["dropped_reasons"] == {"no_committed_answer": 1}


# ---------------------------------------------------------------------------
# 3. the registered row range is ENFORCED, and names the short stratum
# ---------------------------------------------------------------------------

def _report(rows: int, *, weight=0.55, strata=None) -> dict:
    lo, hi = CFG["views"]["expected_rows"]
    return {"rows": rows, "terminal_weight": weight,
            "expected_rows": [lo, hi],
            "terminal_weight_min": CFG["views"]["terminal_weight_min"],
            "rows_in_expected_range": lo <= rows <= hi,
            "strata": strata or {}}


def test_the_registered_row_range_is_enforced_not_merely_reported():
    lo, hi = (int(x) for x in CFG["views"]["expected_rows"])
    assert ds.require_expected_rows(_report(lo))["ok"] is True
    assert ds.require_expected_rows(_report(hi))["ok"] is True
    for rows in (0, lo - 1, hi + 1):
        with pytest.raises(SystemExit) as exc:
            ds.require_expected_rows(_report(rows))
        assert f"{lo}-{hi}" in str(exc.value) and str(rows) in str(exc.value)


def test_the_terminal_weight_floor_is_enforced_by_the_same_gate():
    lo, _hi = (int(x) for x in CFG["views"]["expected_rows"])
    floor = float(CFG["views"]["terminal_weight_min"])
    with pytest.raises(SystemExit, match="terminal weight"):
        ds.require_expected_rows(_report(lo, weight=floor - 0.01))
    assert ds.require_expected_rows(_report(lo, weight=floor))["ok"] is True


def test_a_short_corpus_names_which_stratum_is_short():
    """The failure has to say what to re-roll, not only that the total is small."""
    strata = {
        "lookup_chain-h4": {"rows": 400, "trajectories": 100,
                            "trajectories_dropped": 0, "dropped_reasons": {}},
        "fulfillment-h20": {"rows": 0, "trajectories": 0,
                            "trajectories_dropped": 96,
                            "dropped_reasons": {"no_committed_answer": 94,
                                                "trajectory_over_budget": 2}},
        "typed_relay-h8": {"rows": 40, "trajectories": 10,
                           "trajectories_dropped": 3,
                           "dropped_reasons": {"no_committed_answer": 3}},
    }
    ranked = ds.stratum_shortfall(_report(440, strata=strata), CFG)
    assert [c["stratum"] for c in ranked] == ["fulfillment-h20", "typed_relay-h8"]
    assert ranked[0]["rows_lost_estimate"] > ranked[1]["rows_lost_estimate"]

    with pytest.raises(SystemExit) as exc:
        ds.require_expected_rows(_report(440, strata=strata))
    text = str(exc.value)
    assert "fulfillment-h20" in text and "no_committed_answer 94" in text
    assert "lookup_chain-h4" not in text, "a stratum that lost nothing is not short"
    assert "do not widen the range" in text


def test_the_per_stratum_census_counts_kept_and_dropped_trajectories():
    kept_b, kept = _rollout(index=12)
    _drop_b, dropped = _rollout(index=13, terminal_text="no commitment here")
    rows, meta, report = ds.build_views(
        _cpu_attested([kept, dropped]), token_counter_stub(), CFG)
    cell = ds.stratum_of(kept["family"], kept["horizon"])
    census = report["strata"][cell]
    assert census["trajectories"] == 1 and census["trajectories_dropped"] == 1
    assert census["rows"] == len(rows) == sum(
        c["rows"] for c in report["strata"].values())
    assert sum(census["view_counts"].values()) == census["rows"]
    assert {m["task_id"] for m in meta} == {kept["task_id"]}


# ---------------------------------------------------------------------------
# 4a. the view corpus: a receipt that matches these rows, and the rows LAST
# ---------------------------------------------------------------------------

def _built(index=14, n=1):
    records = [_rollout(index=index + i)[1] for i in range(n)]
    return ds.build_views(_cpu_attested(records), token_counter_stub(), CFG)


def test_the_view_report_is_a_receipt_checked_against_the_rows_in_hand():
    rows, meta, report = _built()
    checked = ds.require_views_chain(rows, meta, report, require_gpu_source=False)
    assert checked["rows"] == len(rows)
    assert checked["row_ids_sha256"] == report["row_ids_sha256"]

    # a truncated corpus: the count no longer matches the receipt
    with pytest.raises(SystemExit, match="another build"):
        ds.require_views_chain(rows[:-1], meta[:-1], report, require_gpu_source=False)
    # the same rows in another order: the count matches and the identity does not
    reordered = list(reversed(meta))
    assert len(reordered) == len(meta)
    with pytest.raises(SystemExit, match="row-id digest"):
        ds.require_views_chain(rows, reordered, report, require_gpu_source=False)
    # a receipt that counts a different number of distinct rows
    with pytest.raises(SystemExit, match="distinct row"):
        ds.require_views_chain(rows, meta, dict(report, row_ids=len(rows) + 5),
                              require_gpu_source=False)


def test_a_zero_row_view_corpus_is_never_a_finished_one():
    _rows, _meta, report = ds.build_views([], token_counter_stub(), CFG)
    assert report["rows"] == 0
    with pytest.raises(SystemExit, match="zero-row"):
        ds.require_views_chain([], [], report, require_gpu_source=False)


def test_a_production_view_corpus_that_missed_its_row_range_fails_the_trainer_gate():
    """The gate reaches the trainer, which this task may not edit.

    `agentlab.sft` calls `require_views_chain`. A corpus the CLI labelled
    production must therefore satisfy the registered row range there too, however
    it got onto disk. An in-process build -- the dev preflight's deliberate
    one-optimizer-step canary -- is labelled otherwise and is measured against
    neither corpus-level gate.
    """
    rows, meta, report = _built()
    assert report["corpus_kind"] == ds.IN_PROCESS_CORPUS
    ds.require_views_chain(rows, meta, report, require_gpu_source=False)

    claims_production = dict(report, corpus_kind=ds.PRODUCTION_CORPUS)
    with pytest.raises(SystemExit, match="outside the registered range"):
        ds.require_views_chain(rows, meta, claims_production,
                              require_gpu_source=False)


def test_the_view_corpus_publishes_the_row_file_last(tmp_path, monkeypatch):
    """Both resumers downstream test the ROW path, so it appears last.

    `scripts/run_multifaceted_chain.sh` skips the views stage when
    `sft_views.jsonl` exists and `agentlab.sft` refuses only on a missing file, so
    a build killed after writing the rows used to leave a corpus with no metadata
    and no receipt that the next invocation treated as finished.
    """
    rows, meta, report = _built()
    out = tmp_path / "sft_views.jsonl"
    meta_out = tmp_path / "sft_views.meta.jsonl"
    report_out = tmp_path / "sft_views.report.json"

    order = []
    real = ds._write_atomic

    def spy(path, text):
        order.append(pathlib.Path(path).name)
        return real(path, text)

    monkeypatch.setattr(ds, "_write_atomic", spy)
    ds.write_view_corpus(rows, meta, report, out=out, meta_out=meta_out,
                         report_out=report_out)
    assert order[-1] == out.name, order
    assert order.index(report_out.name) < order.index(out.name)

    # and the receipt on disk describes the bytes on disk
    disk_rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    disk_meta = [json.loads(x) for x in meta_out.read_text().splitlines() if x.strip()]
    disk_report = json.loads(report_out.read_text())
    ds.require_views_chain(disk_rows, disk_meta, disk_report,
                          require_gpu_source=False)


def test_replacing_a_view_corpus_removes_the_old_marker_first(tmp_path, monkeypatch):
    rows, meta, report = _built()
    out = tmp_path / "sft_views.jsonl"
    meta_out = tmp_path / "sft_views.meta.jsonl"
    report_out = tmp_path / "sft_views.report.json"
    out.write_text("stale\n", encoding="utf-8")
    report_out.write_text("{}\n", encoding="utf-8")

    seen = []
    real = ds._write_atomic

    def spy(path, text):
        # what still exists at the moment the first new file is written
        seen.append({"writing": pathlib.Path(path).name,
                     "rows_present": out.exists(),
                     "report_present": report_out.exists()})
        return real(path, text)

    monkeypatch.setattr(ds, "_write_atomic", spy)
    ds.write_view_corpus(rows, meta, report, out=out, meta_out=meta_out,
                         report_out=report_out)
    assert seen[0] == {"writing": meta_out.name, "rows_present": False,
                       "report_present": False}


def test_the_view_cli_refuses_an_accepted_corpus_with_no_receipt(tmp_path,
                                                                monkeypatch):
    """Views may not be built from a corpus whose completion nobody validated."""
    paths = _paths(tmp_path, monkeypatch)
    records = _cpu_attested([_rollout(index=15)[1]])
    md.write_attested_jsonl(paths["accepted"], records, "a corpus")
    assert paths["accepted"].exists() and not paths["receipt"].exists()
    with pytest.raises(SystemExit, match="not a completed accepted corpus"):
        md.require_accepted_corpus(paths["accepted"])

    out = tmp_path / "sft_views.jsonl"
    monkeypatch.setattr(ds, "default_token_counter", token_counter_stub)
    monkeypatch.setattr("sys.argv",
                        ["datasets", "--accepted", str(paths["accepted"]),
                         "--out", str(out),
                         "--meta-out", str(tmp_path / "v.meta.jsonl"),
                         "--report-out", str(tmp_path / "v.report.json")])
    with pytest.raises(SystemExit, match="receipt_absent"):
        ds.main()
    assert not out.exists(), "a refused build must leave nothing a resume trusts"


def test_the_view_cli_writes_nothing_when_the_row_range_fails(tmp_path, monkeypatch):
    """A validated accepted corpus, and still too few rows: no files at all."""
    paths = _paths(tmp_path, monkeypatch)
    records = _cpu_attested([_rollout(index=16)[1]])
    _seal_accepted(paths, records)

    out = tmp_path / "sft_views.jsonl"
    monkeypatch.setattr(ds, "default_token_counter", token_counter_stub)
    monkeypatch.setattr("sys.argv",
                        ["datasets", "--accepted", str(paths["accepted"]),
                         "--out", str(out),
                         "--meta-out", str(tmp_path / "v.meta.jsonl"),
                         "--report-out", str(tmp_path / "v.report.json")])
    with pytest.raises(SystemExit, match="outside the registered range"):
        ds.main()
    assert not out.exists()
    assert not (tmp_path / "v.meta.jsonl").exists()
    assert not (tmp_path / "v.report.json").exists()


# ---------------------------------------------------------------------------
# 4b. shards: done means a receipt over the planned ids, count and digest
# ---------------------------------------------------------------------------

def _shard_rows(shard: dict) -> list:
    """One synthetic-but-attested row per planned rollout id."""
    prov = md.cpu_provenance("test-corpus-completion", CFG)
    rows = []
    for task_id in shard["task_ids"]:
        for j in range(int(shard["k"])):
            rows.append(contract_mod.stamp(
                {"kind": "rollout", "task_id": task_id, "sample_index": j,
                 "family": shard["family"], "horizon": shard["horizon"],
                 "provenance": dict(prov)}))
    return rows


def _shard(index=0, family="lookup_chain", horizon=2, n_tasks=3, k=2) -> dict:
    return {"index": index, "family": family, "horizon": horizon, "split": "distill",
            "task_ids": [f"task-{index}-{i}" for i in range(n_tasks)], "k": k,
            "expected_rollouts": n_tasks * k}


def test_a_shard_is_done_only_when_its_receipt_validates(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    shard = _shard()
    rows = _shard_rows(shard)
    assert md.shard_gaps(shard, CFG) == ["rows_file_absent"]

    # a rows file with no receipt is NOT done -- the old "the path exists" rule
    md._write_jsonl(md._shard_path(0), rows)
    assert md.shard_gaps(shard, CFG) == ["receipt_absent"]
    assert md.shard_is_current(0) is True, "current, and still not complete"

    md.write_shard(shard, rows, CFG)
    assert md.shard_is_complete(shard, CFG)


def test_a_short_shard_is_never_published(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    shard = _shard(index=1)
    rows = _shard_rows(shard)
    gaps = md.shard_rows_gaps(shard, rows[:2])
    assert any(g.startswith("2_rollouts_of_6") for g in gaps)
    assert any(g.startswith("missing_rollouts:") for g in gaps)
    with pytest.raises(SystemExit, match="not the work it was planned to do"):
        md.write_shard(shard, rows[:2], CFG)
    assert not md._shard_path(1).exists()
    assert not md.shard_receipt_path(1).exists()


def test_a_truncated_shard_file_stops_being_done(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    shard = _shard(index=2)
    rows = _shard_rows(shard)
    md.write_shard(shard, rows, CFG)
    assert md.shard_is_complete(shard, CFG)

    md._write_jsonl(md._shard_path(2), rows[:3])       # a kill mid-append
    assert md.shard_gaps(shard, CFG) == ["rows_file_changed_since_the_receipt"]


def test_a_shard_receipt_from_another_plan_is_not_this_shard(tmp_path, monkeypatch):
    """A different --shard-size re-cuts the shards; the old receipt must not count."""
    _paths(tmp_path, monkeypatch)
    shard = _shard(index=3, n_tasks=3)
    md.write_shard(shard, _shard_rows(shard), CFG)
    replanned = dict(shard, task_ids=shard["task_ids"] + ["task-3-3"],
                     expected_rollouts=8)
    gaps = md.shard_gaps(replanned, CFG)
    assert "receipt_covers_another_task_set" in gaps
    assert not md.shard_is_complete(replanned, CFG)


def test_a_tampered_shard_receipt_is_not_evidence(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    shard = _shard(index=4)
    md.write_shard(shard, _shard_rows(shard), CFG)
    path = md.shard_receipt_path(4)
    rec = json.loads(path.read_text())
    rec["rollouts"] = 999
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert md.shard_gaps(shard, CFG) == ["receipt_self_hash_mismatch"]


def test_a_stale_contract_shard_is_re_rolled_not_resumed(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    shard = _shard(index=5)
    rows = [dict(r, **{contract_mod.STAMP_FIELD: "0" * 64})
            for r in _shard_rows(shard)]
    gaps = md.shard_rows_gaps(shard, rows)
    assert any("another_environment_contract" in g for g in gaps)
    with pytest.raises(SystemExit, match="not the work it was planned to do"):
        md.write_shard(shard, rows, CFG)


def test_a_zero_row_attested_artifact_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    with pytest.raises(SystemExit, match="zero rows"):
        md.write_attested_jsonl(path, [], "an empty batch")
    assert not path.exists()


def test_the_real_plan_states_the_work_each_shard_owes():
    shards = md.plan_shards("distill", 48, CFG)
    assert shards, "the committed distill split must plan shards"
    total = sum(s["expected_rollouts"] for s in shards)
    assert total == int(CFG["totals"]["rollouts"]), total
    for s in shards:
        assert s["expected_rollouts"] == s["k"] * len(s["task_ids"])
        assert len(md.expected_rollout_ids(s)) == s["expected_rollouts"]


# ---------------------------------------------------------------------------
# 4c. the accepted corpus: a quota miss stops the chain nonzero
# ---------------------------------------------------------------------------

def _seal_accepted(paths: dict, kept: list, *, quotas=None) -> dict:
    """A validly-receipted accepted corpus, the way `finalize` writes one."""
    md.write_attested_jsonl(paths["accepted"], kept, "the accepted RS corpus")
    summary = {"split": "distill", "rollouts": len(kept),
               "quotas": quotas or {"lookup_chain": {"accepted": len(kept),
                                                     "min_accepted": 1, "ok": True}},
               "per_cell": {}, "faulted_accepted": 0,
               "source_provenance": md.provenance_identity(kept[0]["provenance"]),
               "source_sessions": []}
    receipt = md.build_accepted_receipt(kept, summary, [], paths["accepted"], CFG)
    md._write_json(paths["receipt"], receipt)
    return receipt


def test_a_validated_accepted_corpus_is_what_the_next_stage_trusts(tmp_path,
                                                                   monkeypatch):
    paths = _paths(tmp_path, monkeypatch)
    kept = _cpu_attested([_rollout(index=17)[1]])
    receipt = _seal_accepted(paths, kept)
    assert md.accepted_corpus_gaps(paths["accepted"]) == []
    got = md.require_accepted_corpus(paths["accepted"])
    assert got[md.RECEIPT_HASH_FIELD] == receipt[md.RECEIPT_HASH_FIELD]

    # every way the corpus can stop describing the receipt
    paths["accepted"].write_text(
        paths["accepted"].read_text() + json.dumps(kept[0]) + "\n", encoding="utf-8")
    assert md.accepted_corpus_gaps(paths["accepted"]) == \
        ["corpus_changed_since_the_receipt"]


def test_a_quota_miss_receipt_is_never_accepted(tmp_path, monkeypatch):
    paths = _paths(tmp_path, monkeypatch)
    kept = _cpu_attested([_rollout(index=18)[1]])
    receipt = _seal_accepted(paths, kept)
    broken = md.seal_receipt(dict(receipt, quota_ok=False))
    md._write_json(paths["receipt"], broken)
    assert md.accepted_corpus_gaps(paths["accepted"]) == ["receipt_says_quota_miss"]
    with pytest.raises(SystemExit, match="receipt_says_quota_miss"):
        md.require_accepted_corpus(paths["accepted"])


def test_quota_misses_are_ranked_by_deficit():
    summary = {"quotas": {
        "lookup_chain": {"accepted": 270, "min_accepted": 270, "ok": True},
        "typed_relay": {"accepted": 300, "min_accepted": 360, "ok": False},
        "_faulted": {"accepted": 0, "min_accepted": 480, "ok": False}}}
    misses = md.quota_misses(summary)
    assert [m["quota"] for m in misses] == ["_faulted", "typed_relay"]
    assert misses[0]["short_by"] == 480 and misses[1]["short_by"] == 60


def test_a_quota_miss_leaves_no_corpus_a_resume_would_trust(tmp_path, monkeypatch):
    """THE stop. A miss used to print a warning and exit 0 with accepted.jsonl.

    The shard here holds real rollouts that the strict verifier refuses (the
    terminal turn commits nothing), so nothing is accepted and every family quota
    misses. A stale corpus and receipt from an earlier pass are on disk, exactly as
    they would be on a rerun: both must be gone, the reason must be written down,
    and the exit must be nonzero.
    """
    paths = _paths(tmp_path, monkeypatch)
    bundle, rec = _rollout(index=19, terminal_text="I could not work it out")
    _cpu_attested([rec])
    shard = {"index": 0, "family": rec["family"], "horizon": rec["horizon"],
             "split": "distill", "task_ids": [rec["task_id"]], "k": 1,
             "expected_rollouts": 1}
    monkeypatch.setattr(md, "plan_shards", lambda *a, **k: [shard])
    monkeypatch.setattr(md, "load_split", lambda *a, **k: [bundle])
    md.write_shard(shard, [rec], CFG)

    # what a previous pass left behind
    stale_kept = _cpu_attested([_rollout(index=20)[1]])
    _seal_accepted(paths, stale_kept)
    assert paths["accepted"].exists() and paths["receipt"].exists()

    args = type("A", (), {"split": "distill", "shard_size": 48, "partial": False})()
    with pytest.raises(SystemExit) as exc:
        md.cmd_finalize(args)
    text = str(exc.value)
    assert "not complete" in text and "quota" in text
    assert exc.value.code != 0

    assert not paths["accepted"].exists(), "a resume must not find a corpus"
    assert not paths["receipt"].exists(), "nor a receipt for one"
    assert paths["failure"].exists(), "the reason belongs on disk"
    failure = json.loads(paths["failure"].read_text())
    assert failure["reason"] == "quota_miss"
    assert failure["accepted"] == 0 and failure["quota_misses"]
    assert str(paths["accepted"]) in failure["removed_untrustworthy_artifacts"]
    summary = json.loads(paths["summary"].read_text())
    assert summary["complete"] is False and summary["quota_ok"] is False


def test_a_partial_finalize_produces_no_corpus_and_exits_nonzero(tmp_path,
                                                                monkeypatch):
    paths = _paths(tmp_path, monkeypatch)
    bundle, rec = _rollout(index=21)
    shard = {"index": 0, "family": rec["family"], "horizon": rec["horizon"],
             "split": "distill", "task_ids": [rec["task_id"]], "k": 1,
             "expected_rollouts": 1}
    missing = dict(shard, index=1, task_ids=["never-rolled"])
    monkeypatch.setattr(md, "plan_shards", lambda *a, **k: [shard, missing])
    monkeypatch.setattr(md, "load_split", lambda *a, **k: [bundle])
    md.write_shard(shard, _cpu_attested([rec]), CFG)

    strict = type("A", (), {"split": "distill", "shard_size": 48, "partial": False})()
    with pytest.raises(SystemExit, match="no valid receipt"):
        md.cmd_finalize(strict)
    assert not paths["accepted"].exists()

    partial = type("A", (), {"split": "distill", "shard_size": 48, "partial": True})()
    with pytest.raises(SystemExit) as exc:
        md.cmd_finalize(partial)
    assert exc.value.code != 0
    assert not paths["accepted"].exists() and not paths["receipt"].exists()
    failure = json.loads(paths["failure"].read_text())
    assert failure["shards_incomplete"] == [1]


def test_a_finalize_that_passes_writes_a_receipt_the_next_stage_validates(
        tmp_path, monkeypatch):
    """The success path: every quota passes, so a corpus AND its receipt appear.

    No registered number is touched. The acceptance/quota arithmetic is exercised
    against the real config elsewhere (`test_quota_misses_are_ranked_by_deficit`,
    and the miss test above runs the whole real `finalize`); what is under test
    here is the WRITE path -- receipt written, stale failure report cleared, and the
    corpus re-read through the consumer's own gate -- so `finalize` is stubbed to
    hand it an all-quotas-pass result.
    """
    paths = _paths(tmp_path, monkeypatch)
    bundle, rec = _rollout(index=22)
    kept = _cpu_attested([rec])
    shard = {"index": 0, "family": rec["family"], "horizon": rec["horizon"],
             "split": "distill", "task_ids": [rec["task_id"]], "k": 1,
             "expected_rollouts": 1}
    summary = {"rollouts": 1, "accepted": 1, "per_cell": {"lookup_chain-h4": 1},
               "faulted_accepted": 0,
               "source_provenance": md.provenance_identity(rec["provenance"]),
               "source_sessions": [],
               "quotas": {fam: {"accepted": 1, "min_accepted": 1, "ok": True}
                          for fam in CFG["mixture"]}}
    summary["quotas"]["_faulted"] = {"accepted": 0, "min_accepted": 0, "ok": True}
    monkeypatch.setattr(md, "plan_shards", lambda *a, **k: [shard])
    monkeypatch.setattr(md, "load_split", lambda *a, **k: [bundle])
    monkeypatch.setattr(md, "finalize", lambda *a, **k: (kept, dict(summary)))
    md.write_shard(shard, kept, CFG)
    paths["failure"].write_text('{"stale": true}\n', encoding="utf-8")

    args = type("A", (), {"split": "distill", "shard_size": 48, "partial": False})()
    md.cmd_finalize(args)

    assert paths["accepted"].exists() and paths["receipt"].exists()
    assert not paths["failure"].exists(), "a passing finalize voids the old reason"
    receipt = md.require_accepted_corpus(paths["accepted"])
    assert receipt["accepted"] == 1 and receipt["quota_ok"] is True
    assert receipt["partial"] is False and receipt["complete"] is True
    assert receipt["corpus_sha256"] == md.file_sha256(paths["accepted"])
    assert receipt["task_ids_sha256"] == digest_text(canon([rec["task_id"]]))
    assert receipt["shard_receipts"] == [
        {"index": 0, md.RECEIPT_HASH_FIELD:
            json.loads(md.shard_receipt_path(0).read_text())[md.RECEIPT_HASH_FIELD]}]
    assert md.receipt_seal_ok(receipt)
    written = json.loads(paths["summary"].read_text())
    assert written["complete"] is True and written["accepted"] == 1
