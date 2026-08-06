"""The driver/transport integrity seams: a dead engine, the ledger, the census.

Three properties, each of which used to be violated silently:

  * INFRASTRUCTURE IS NOT BEHAVIOUR. Every exception out of the chat backend was
    caught inside the episode loop and committed as a scored `parser_budget`
    row -- so a vLLM server that died mid-shard would have produced hundreds of
    model failures, in the denominators, and then been SKIPPED on resume because
    those task ids already had rows.
  * A LAUNCH WINDOW IS NOT A KILL SWITCH. --time-budget-s must stop launching new
    episodes and drain the ones already running; killing them discards GPU
    seconds the ledger has already charged.
  * THE LEDGER MUST SEE INTERRUPTED TIME, AND THE GUARD MUST SEE THE WHOLE
    REMAINING STAGE. A 360-second launch window fits under a 120-hour ceiling
    every single time, right up to the invocation that crosses it.

Plus the census reconciliation: the driver must schedule EXACTLY the registered
episode census, which is the thing a work table and a preregistration drift apart
on when nothing compares them.

Every test here is CPU-only, starts no server and touches no card.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import time

import pytest

from agentlab.suite import configio, evaluate

# The fixtures for a fail-closed evaluator invocation live with the hardware
# contract they encode; duplicating them here would be a second definition of
# "what a complete producer manifest looks like".
from test_hardware_contract import (CERTSPECS, PRODUCER_UUID, _cfg,  # noqa: E402
                                    _producer_manifest, _shard_args)

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "run_multifaceted_chain.sh"
PREREG = json.loads((REPO / "configs" / "agentic_preregister.json").read_text())
PY = REPO / ".venv" / "bin" / "python"

needs_specs = pytest.mark.skipif(
    not (CERTSPECS / "dev.jsonl").exists(),
    reason="run scripts/export_eval_specs.py first")


# ---------------------------------------------------------------------------
# 1. a dead server is never a scored episode
# ---------------------------------------------------------------------------

def test_the_http_backend_separates_transport_death_from_model_behaviour():
    """Each infrastructure signal is raised as TransportFailure, by name."""
    import requests

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status=200, body=None, text=""):
            self.status_code = status
            self._body = body
            self.text = text or json.dumps(body)

        def json(self):
            if self._body is _BAD:
                raise ValueError("not json")
            return self._body

    _BAD = object()
    scripted = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls["n"] += 1
        item = scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    decode = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 8,
              "enable_thinking": False}
    chat = evaluate.make_http_chat("http://127.0.0.1:1", "m", decode)
    original = requests.post
    requests.post = fake_post
    try:
        for item, kind in (
                (requests.exceptions.ConnectionError("refused"), "unreachable"),
                (requests.exceptions.Timeout("timed out"), "unreachable"),
                (FakeResp(status=500, text="internal"), "http_status"),
                (FakeResp(status=404, text="no such model"), "http_status"),
                (FakeResp(status=200, body=_BAD, text="<html>"),
                 "malformed_response"),
                (FakeResp(status=200, body={"object": "error"}),
                 "malformed_response")):
            scripted.append(item)
            with pytest.raises(evaluate.TransportFailure) as exc:
                chat([{"role": "user", "content": "x"}], [])
            assert exc.value.kind == kind
        # and a WELL-FORMED answer is not a transport failure, however odd
        scripted.append(FakeResp(status=200, body={
            "choices": [{"message": {"content": "\\boxed{1}"}}]}))
        out = chat([{"role": "user", "content": "x"}], [])
        assert out["content"] == "\\boxed{1}" and out["tool_calls"] == []
    finally:
        requests.post = original


@needs_specs
def test_a_dead_server_aborts_the_shard_and_writes_no_scored_row(tmp_path):
    """THE DANGEROUS ONE. A dead engine must leave no episode row at all.

    Before: `except Exception` inside the decision loop turned every HTTP failure
    into `termination_reason: "parser_budget"`, a fully scored row with a verdict,
    a certification receipt and a place in the arm's denominators -- and resume
    then treated the task as done for ever.
    """
    cfg = _cfg(tmp_path)
    path, _ = _producer_manifest(tmp_path, cfg)
    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl",
                       runtime_manifest=str(path), limit=3, concurrency=1)

    def dead(messages, tools):
        raise evaluate.TransportFailure(
            "connection refused", kind="unreachable")

    with pytest.raises(SystemExit) as exc:
        evaluate.run_shard(args, chat_fn=dead, cfg=cfg)
    msg = str(exc.value)
    assert "ABORTED (infrastructure)" in msg
    assert "unreachable" in msg
    assert "parser_budget" in msg          # says WHY it is not one
    out = pathlib.Path(args.out) / "BP.clean.none.jsonl"
    rows = [] if not out.exists() else [
        l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows == [], "an infrastructure failure may not leave a scored row"
    # it IS recorded as evidence -- outside the trace corpus, because every
    # consumer of that directory globs *.jsonl
    log = pathlib.Path(args.out) / "BP.clean.none.transport.log"
    assert log.exists()
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert recs and all(r["kind_of_record"] == "transport_failure" for r in recs)
    assert all(r["kind"] == "unreachable" for r in recs)
    # and the trace file this invocation created is REMOVED rather than left as an
    # empty file that reads as "this arm ran and produced nothing"
    assert not list(pathlib.Path(args.out).glob("*.jsonl"))
    assert not out.exists()


@needs_specs
def test_a_genuine_parse_failure_still_scores_as_parser_budget(tmp_path):
    """The other half: a PARSE failure over a healthy server stays registered.

    The registered F5 denominator includes parser failures. Closing the transport
    seam must not quietly remove a real one from the sample.
    """
    cfg = _cfg(tmp_path)
    path, _ = _producer_manifest(tmp_path, cfg)
    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl",
                       runtime_manifest=str(path), limit=1, concurrency=1)

    def broken_parse(messages, tools):
        raise ValueError("the client could not read a well-formed answer")

    status = evaluate.run_shard(args, chat_fn=broken_parse, cfg=cfg)
    rows = [json.loads(l) for l in
            pathlib.Path(status["out"]).read_text().splitlines() if l.strip()]
    assert rows and all(r["runner"]["termination_reason"] == "parser_budget"
                        for r in rows)


def test_an_unreachable_server_refuses_before_the_trace_file_is_opened(tmp_path):
    probe = evaluate.probe_server("http://127.0.0.1:1", "m", timeout_s=1.0)
    assert probe["ok"] is False and probe["kind"] == "unreachable"
    with pytest.raises(SystemExit, match="no live engine"):
        evaluate.require_live_server("http://127.0.0.1:1", "m", "BP/clean/none")


def test_the_health_probe_refuses_a_server_serving_another_model(monkeypatch):
    import requests

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "Qwen/Qwen3.5-4B"}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    assert evaluate.probe_server("http://x", "Qwen/Qwen3.5-4B")["ok"] is True
    bad = evaluate.probe_server("http://x", "trained")
    assert bad["ok"] is False and bad["kind"] == "model_absent"


# ---------------------------------------------------------------------------
# 2. the launch window drains, it never kills
# ---------------------------------------------------------------------------

@needs_specs
def test_the_time_budget_stops_launching_and_never_kills_a_running_episode(tmp_path):
    cfg = _cfg(tmp_path)
    path, _ = _producer_manifest(tmp_path, cfg)
    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl",
                       runtime_manifest=str(path), limit=2, concurrency=2,
                       time_budget_s=0.25)

    def slow(messages, tools):
        time.sleep(1.0)              # still running when the window closes
        return {"content": "\\boxed{1}", "tool_calls": []}

    status = evaluate.run_shard(args, chat_fn=slow, cfg=cfg)
    # both episodes were launched inside the window and BOTH were drained and
    # written after it closed -- the window closed at 0.25 s and each episode took
    # a full second, so nothing here survives if the budget kills instead of gating
    assert status["written"] == 2
    assert status["elapsed_s"] >= 1.0
    rows = [json.loads(l) for l in
            pathlib.Path(status["out"]).read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert all(r["runner"]["termination_reason"] != "wall_clock" for r in rows)


# ---------------------------------------------------------------------------
# 3. the ledger: interrupted GPU time is charged, the guard sees the stage
# ---------------------------------------------------------------------------

def test_an_interrupted_session_is_charged_from_its_last_heartbeat(tmp_path):
    """A killed stage's occupancy is real GPU time and must reach the ledger."""
    cfg = _cfg(tmp_path)
    rec = configio.journal_open("multidistill", cfg)
    jid = rec["journal_id"]
    assert configio.journal_heartbeat(jid, cfg) is True
    # rewrite the journal so the session looks like a DEAD process that ran for
    # 30 minutes: this is the SIGKILL case, where no close ever runs
    rows = configio.journal_rows(cfg)
    now = time.time()
    rows[0].update(opened_epoch=now - 1800, pid=999999999, process_start=1)
    rows[1].update(at_epoch=now - 60)
    configio.journal_path(cfg).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert configio.ledger_hours(cfg) == 0.0
    closed = configio.reconcile_open_sessions(cfg)
    assert len(closed) == 1 and closed[0]["reason"] == "interrupted"
    charged_h = configio.ledger_hours(cfg)
    assert charged_h == pytest.approx((1800 - 60) / 3600.0, abs=0.01)
    row = configio.ledger_rows(cfg)[-1]
    assert row["kind"] == "interrupted_session"
    assert row["stage"].startswith("multidistill")
    # idempotent: a second reconcile charges nothing, because the session closed
    assert configio.reconcile_open_sessions(cfg) == []
    assert configio.ledger_hours(cfg) == charged_h


def test_a_close_charges_only_the_uncharged_remainder(tmp_path):
    """Layering the journal over modules that write their own receipts.

    The stage's own per-unit rows are the primary accounting. The journal exists
    for the part an interruption prevents, so a session whose module already
    charged its whole interval must add nothing.
    """
    cfg = _cfg(tmp_path)
    rec = configio.journal_open("prompt_tournament", cfg)
    rows = configio.journal_rows(cfg)
    rows[0]["opened_epoch"] = time.time() - 600      # ten minutes of occupancy
    configio.journal_path(cfg).write_text(json.dumps(rows[0]) + "\n",
                                          encoding="utf-8")
    # the module charged nine of those ten minutes itself
    configio.ledger_append("prompt_tournament", 9.0, cfg, kind="unit")
    before = configio.ledger_hours(cfg)
    configio.journal_close(rec["journal_id"], cfg)
    extra_s = (configio.ledger_hours(cfg) - before) * 3600.0
    assert 30.0 <= extra_s <= 90.0, "only the uncharged ~60 s may be added"
    # and closing again is a no-op rather than a second charge
    again = configio.ledger_hours(cfg)
    configio.journal_close(rec["journal_id"], cfg)
    assert configio.ledger_hours(cfg) == again


def test_a_live_session_is_never_charged_by_another_stages_guard(tmp_path):
    """The session's OWNER is the driver, not the helper that wrote the record.

    If the pid in the journal were the short-lived python helper's, the session
    would look dead the instant it was created and the very next ledger guard
    would close and charge a stage that is still running -- and the stage's own
    close would then find nothing to charge.
    """
    cfg = _cfg(tmp_path)
    rec = configio.journal_open("sft", cfg, pid=os.getpid())
    assert rec["pid"] == os.getpid()
    assert configio.reconcile_open_sessions(cfg) == []
    assert configio.ledger_hours(cfg) == 0.0
    assert len(configio.open_sessions(cfg)) == 1
    # the driver passes its own pid, not the helper's
    assert 'pid=int(sys.argv[3])' in CHAIN.read_text(encoding="utf-8")


def test_a_session_with_no_heartbeat_undercharges_visibly_never_silently(tmp_path):
    """The bound is the last heartbeat, and the gap is RECORDED.

    Charging up to "whenever someone noticed" would invent GPU time; charging up
    to the last heartbeat can undercount, which is the permissive direction for a
    hard ceiling. So the unattestable gap is written into the close record rather
    than dropped, and the heartbeat interval (20 s) bounds it in the normal case.
    """
    cfg = _cfg(tmp_path)
    rec = configio.journal_open("multidistill", cfg, pid=999999999)
    rows = configio.journal_rows(cfg)
    rows[0].update(opened_epoch=time.time() - 600, process_start=1)
    configio.journal_path(cfg).write_text(json.dumps(rows[0]) + "\n",
                                          encoding="utf-8")
    closed = configio.reconcile_open_sessions(cfg)
    assert len(closed) == 1
    assert closed[0]["charged_s"] == 0.0
    assert closed[0]["unobserved_s"] >= 590.0
    assert configio.ledger_hours(cfg) == 0.0


def test_the_ledger_total_is_the_sum_of_measured_seconds(tmp_path):
    cfg = _cfg(tmp_path)
    for _ in range(10):
        configio.ledger_append("s", 0.005, cfg)   # 0.3 s each; rounds to 0.01 min
    assert configio.ledger_elapsed_s(cfg) == pytest.approx(3.0, abs=0.1)
    assert configio.ledger_hours(cfg) == pytest.approx(3.0 / 3600.0, abs=1e-5)


def test_the_guard_projects_the_whole_remaining_stage_not_one_invocation(tmp_path):
    """The defect: `--time-budget-s 360` is six minutes and always fits."""
    cfg = _cfg(tmp_path)
    configio.ledger_append("earlier", 119.0 * 60.0, cfg)     # 119 h used
    # the weak check passes, over and over, right up to the crossing
    weak = configio.ledger_guard("eval:BP.clean.none", 6.0, cfg)
    assert weak["projection_source"] == "invocation_nominal"
    assert weak["calibrated"] is False
    # a caller that declares the remaining stage is refused immediately
    with pytest.raises(SystemExit, match="BUDGET"):
        configio.ledger_guard("eval:BP.clean.none", 6.0, cfg,
                              remaining_minutes=600.0)
    # and so is one whose CALIBRATED remaining-stage projection does not fit
    configio.budget_commitment_path(cfg).write_text(json.dumps({
        "schema": configio.BUDGET_COMMITMENT_SCHEMA, "run_id": "agentic-v1",
        "remaining_minutes_by_stage": {"eval": 600.0}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="budget_commitment"):
        configio.ledger_guard("eval:BP.clean.none", 6.0, cfg,
                              run_id="agentic-v1")


def test_a_committed_projection_governs_every_stage_invocation(tmp_path):
    cfg = _cfg(tmp_path)
    configio.budget_commitment_path(cfg).write_text(json.dumps({
        "schema": configio.BUDGET_COMMITMENT_SCHEMA, "run_id": "agentic-v1",
        "remaining_minutes_by_stage": {"eval": 90.0, "multidistill": 30.0}}),
        encoding="utf-8")
    d = configio.ledger_guard("eval:TP.faulted.none", 6.0, cfg,
                              run_id="agentic-v1")
    assert d["projection_source"] == "budget_commitment[eval]"
    assert d["projection_minutes"] == 90.0 and d["calibrated"] is True
    # a stage the commitment does not mention falls back, and says so
    d2 = configio.ledger_guard("ship_smoke", 5.0, cfg, run_id="agentic-v1")
    assert d2["calibrated"] is False


def test_a_malformed_or_foreign_budget_commitment_is_fatal(tmp_path):
    cfg = _cfg(tmp_path)
    p = configio.budget_commitment_path(cfg)
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable"):
        configio.ledger_guard("eval", 6.0, cfg)
    p.write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="budget_commitment/v1"):
        configio.ledger_guard("eval", 6.0, cfg)
    p.write_text(json.dumps({"schema": configio.BUDGET_COMMITMENT_SCHEMA,
                             "remaining_minutes_by_stage": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="commits nothing"):
        configio.ledger_guard("eval", 6.0, cfg)
    p.write_text(json.dumps({"schema": configio.BUDGET_COMMITMENT_SCHEMA,
                             "run_id": "some-other-run",
                             "remaining_minutes_by_stage": {"eval": 1.0}}),
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="another run"):
        configio.ledger_guard("eval", 6.0, cfg, run_id="agentic-v1")


def test_the_guard_charges_interrupted_time_before_it_projects(tmp_path):
    """The two halves meet: a killed predecessor cannot hide under the ceiling."""
    cfg = _cfg(tmp_path)
    cfg["budget"]["gpu_hours_ceiling"] = 1.0
    rec = configio.journal_open("multidistill", cfg)
    configio.journal_heartbeat(rec["journal_id"], cfg)
    rows = configio.journal_rows(cfg)
    now = time.time()
    rows[0].update(opened_epoch=now - 3300, pid=999999999,
                   process_start=1)          # 55 min of occupancy, process gone
    rows[1]["at_epoch"] = now - 60           # heartbeating until a minute ago
    configio.journal_path(cfg).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="BUDGET"):
        configio.ledger_guard("multidistill", 30.0, cfg)
    assert configio.ledger_hours(cfg) == pytest.approx(54.0 / 60.0, abs=0.02)
    assert rec["journal_id"] in {r.get("journal_id")
                                 for r in configio.journal_rows(cfg)}


# ---------------------------------------------------------------------------
# 4. the census: the driver runs EXACTLY the registered episode counts
# ---------------------------------------------------------------------------

def chain(*args, block_tasks="100"):
    """The driver's own read-only views of its schedule. Touches no card."""
    return subprocess.run(
        ["bash", str(CHAIN), *args], cwd=REPO, capture_output=True, text=True,
        timeout=180, env=dict(os.environ, PYTHONPATH="src",
                              CUDA_VISIBLE_DEVICES="",
                              EVAL_BLOCK_TASKS=block_tasks))


def eval_units(**kw):
    r = chain("--print-units", **kw)
    assert r.returncode == 0, r.stderr
    return [ln.split() for ln in r.stdout.splitlines() if ln.strip()]


def eval_census(**kw):
    r = chain("--print-census", **kw)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_driver_census_is_the_registered_census():
    """1,400 episodes of drift: B0/T0 were scheduled over the CONTROLS too.

    The controls are registered per MANDATORY arm (600 x 2 redacted, 100 x 2
    permuted). Running the descriptive arms over them added 1,400 episodes that
    the registered 15,320 total never contained -- a disagreement between the
    driver and the preregistration that nothing compared.
    """
    census = eval_census()
    projection = PREREG["budget"]["episode_projection"]
    assert census["mandatory"] == 7800 == projection["bp_tp_mandatory"]
    assert census["mandatory"] + census["stress_rank4"] == 8360 == \
        projection["bp_tp_with_optional_stress"]
    assert census["descriptive_rank3"] == 6960
    assert census["total"] == 15320 == projection["with_b0_t0"]
    # and the preregistration records the corrected descriptive census itself
    reg = PREREG["budget"]["mandatory_episode_census"]
    optional = reg["optional_not_in_the_mandatory_total"]
    assert optional["b0_t0_descriptive_arms_episodes"] == 6960
    assert optional["stress_280_x_2_arms"] == 560


def test_the_descriptive_arms_never_run_against_a_registered_control():
    units = eval_units()
    for arm, name, specs, condition, control, mandatory, _b, _n in units:
        if arm in ("B0", "T0"):
            assert control == "none", (arm, name, control)
            assert mandatory == "False"
    # they ARE still scheduled -- cut rank 3, not dropped
    assert {u[0] for u in units} == {"BP", "TP", "B0", "T0"}
    assert {(u[0], u[1]) for u in units if u[0] in ("B0", "T0")} == {
        ("B0", "core"), ("T0", "core"), ("B0", "mt"), ("T0", "mt"),
        ("B0", "h8"), ("T0", "h8"), ("B0", "stress"), ("T0", "stress")}


def test_every_mandatory_unit_is_scheduled_before_any_optional_one():
    units = eval_units()
    last_mandatory = max(i for i, u in enumerate(units) if u[5] == "True")
    assert all(u[5] == "True" for u in units[:last_mandatory + 1])
    # cut rank 3 (B0/T0) is cut FIRST, so it is scheduled LAST
    first_descriptive = min(i for i, u in enumerate(units) if u[0] in ("B0", "T0"))
    first_stress = min(i for i, u in enumerate(units) if u[1] == "stress")
    assert last_mandatory < first_stress < first_descriptive


def test_evaluation_is_interleaved_by_immutable_task_block():
    """BP and TP run the SAME block back to back, order flipped every block.

    Running every BP episode days before every TP episode confounds the paired
    comparison with thermal and host-load drift. Interleaving changes no estimand:
    the blocks are a deterministic function of the manifest order.
    """
    units = eval_units()
    core = [u for u in units if u[1] == "core" and u[3] == "clean"
            and u[0] in ("BP", "TP")]
    nblocks = int(core[0][7])
    assert nblocks == 12                      # 1,200 core tasks / 100 per block
    pairs = [core[i:i + 2] for i in range(0, len(core), 2)]
    assert len(pairs) == nblocks
    for i, (first, second) in enumerate(pairs):
        assert first[6] == second[6] == str(i)          # the SAME block
        assert {first[0], second[0]} == {"BP", "TP"}    # both arms
        assert first[0] == ("BP" if i % 2 == 0 else "TP")   # order alternates
    # every manifest/condition is fully covered by its blocks, and the block
    # count is ceil(tasks / block size) so no task is left unscheduled
    cfg_tasks = {m["name"]: m["tasks"] for m in
                 configio.load_config()["eval"]["manifests"]}
    for u in units:
        assert int(u[7]) == max(1, math.ceil(cfg_tasks[u[1]] / 100))
        assert 0 <= int(u[6]) < int(u[7])


def test_the_block_size_only_changes_scheduling_never_the_census():
    small = eval_census(block_tasks="50")
    assert small["total"] == 15320 and small["block_tasks"] == 50
    assert small["units"] > eval_census()["units"]


# ---------------------------------------------------------------------------
# 5. the driver's stage budgets and its completion receipts
# ---------------------------------------------------------------------------

@needs_specs
def test_a_limited_invocation_can_never_report_the_shard_complete(tmp_path):
    """--limit is a smoke knob. It used to be able to say `complete: true`.

    `remaining` was computed against the LIMITED slice, so an eight-episode smoke
    over a 1,200-task block reported the block finished and the driver's loop --
    which breaks on exactly that field -- would have believed it.
    """
    cfg = _cfg(tmp_path)
    path, _ = _producer_manifest(tmp_path, cfg)
    args = _shard_args(tmp_path, CERTSPECS / "dev.jsonl",
                       runtime_manifest=str(path), limit=2, concurrency=2)
    status = evaluate.run_shard(
        args, chat_fn=lambda m, t: {"content": "\\boxed{1}", "tool_calls": []},
        cfg=cfg)
    assert status["written"] == 2 and status["limit"] == 2
    assert status["complete"] is False
    assert status["remaining_in_shard"] > 2


@needs_specs
def test_completion_is_a_census_not_a_shared_trace_file(tmp_path):
    """Core, MT and H8 all write to `{arm}.clean.none.jsonl`.

    So the file's existence says nothing about whether a registered manifest ran,
    and MT and H8 were preregistered and never invoked once already. Attribution is
    by TASK ID.
    """
    specs = evaluate.load_specs(CERTSPECS / "dev.jsonl")
    assert len(specs) > 20
    certspecs = tmp_path / "certspecs"
    certspecs.mkdir()
    first = specs[:10]
    second = specs[10:20]
    for name, rows in (("man_a.jsonl", first), ("man_b.jsonl", second)):
        (certspecs / name).write_text(
            "".join(json.dumps(s) + "\n" for s in rows), encoding="utf-8")
    traces = tmp_path / "traces"
    traces.mkdir()
    # one shared file that contains manifest A in full and NOTHING of manifest B
    from agentlab.suite import contract as contract_mod
    (traces / "BP.clean.none.jsonl").write_text(
        "".join(json.dumps(contract_mod.stamp(
            {"kind": "episode", "task_id": s["task_id"]})) + "\n" for s in first),
        encoding="utf-8")

    units = [("BP", "a", "man_a.jsonl", "clean", "none", True),
             ("BP", "b", "man_b.jsonl", "clean", "none", True)]
    rows = evaluate.unit_census(units, certspecs, traces)
    assert rows[0]["status"] == "complete" and rows[0]["written"] == 10
    assert rows[1]["status"] == "short" and rows[1]["missing"] == 10
    with pytest.raises(SystemExit, match="MANDATORY census is incomplete"):
        evaluate.require_mandatory_census(rows)
    # an OPTIONAL short unit is budget-conditional, not a refusal
    optional = evaluate.unit_census(
        [("B0", "b", "man_b.jsonl", "clean", "none", False)], certspecs, traces)
    assert evaluate.require_mandatory_census(optional) == optional
    # a missing manifest is never silently "complete"
    gone = evaluate.unit_census(
        [("BP", "c", "nope.jsonl", "clean", "none", True)], certspecs, traces)
    assert gone[0]["status"] == "no_manifest"
    with pytest.raises(SystemExit, match="no_manifest"):
        evaluate.require_mandatory_census(gone)


def test_the_driver_refuses_a_census_it_cannot_verify():
    """Before the reveal the held-out manifests do not exist, so this refuses."""
    r = chain("--verify-census")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "MANDATORY census is incomplete" in combined
    assert "INCOMPLETE / INCONCLUSIVE" in combined


def test_an_empty_checkpoint_tree_has_no_digest(tmp_path):
    """sha256("") is a perfectly stable digest for a checkpoint that isn't there.

    Same defect class as "the path exists, so the stage is done": a lock recording
    it would claim to have pinned a checkpoint tree with no files in it.
    """
    empty = tmp_path / "adapter"
    empty.mkdir()
    assert configio.checkpoint_tree_sha256(empty) is None
    (empty / "nested").mkdir()
    assert configio.checkpoint_tree_sha256(empty) is None
    (empty / "adapter_model.safetensors").write_bytes(b"weights")
    real = configio.checkpoint_tree_sha256(empty)
    assert real and real != configio.checkpoint_tree_sha256(tmp_path / "gone")


def test_the_stage_invocation_budgets_are_the_raised_ones():
    """Cold starts are a controlled cost, not a per-55-minute tax."""
    text = CHAIN.read_text(encoding="utf-8")
    assert "--budget-minutes 55" not in text
    assert "--budget-minutes 120" in text        # prompt tournament
    assert "--budget-minutes 240" in text        # rejection sampling
    assert 'SERVER_SESSION_CAP_MIN="${SERVER_SESSION_CAP_MIN:-360}"' in text
    assert "--time-budget-s 360" in text         # the client launch window
    # the prose no longer promises one engine for a whole stage
    assert "ONE ENGINE PER STAGE" not in text
    assert "85 cold starts become 6" not in text


def test_baselock_is_prompt_lock_only():
    """10,800 dev episodes that were neither the tournament nor the calibration.

    The registered dev measurement belongs to the prompt tournament (3,600
    episodes) and to the calibration stage. Running B0 clean, BP clean and BP
    faulted over all 3,600 dev tasks was a third, unregistered dev census.
    """
    text = CHAIN.read_text(encoding="utf-8")
    start = text.index("stage_baselock()")
    body = text[start:text.index("\nstage_distill()")]
    assert "eval_arm" not in body
    assert "start_server" not in body
    assert "require_gpu" not in body
    assert "lock-prompt" in body
    assert "# CPU" in body.splitlines()[0]


def test_no_stage_treats_mere_file_existence_as_completion():
    text = CHAIN.read_text(encoding="utf-8")
    body = text[text.index("done_already()"):text.index("# The registered hardware")]
    assert "-e " not in body, "existence is not completion; -s is the minimum"
    assert "-s " in body
    # the receipt validators are wired into the completion decision
    assert "stage_receipt " in text or "receipt_ok" in text


def test_the_chain_opens_a_gpu_session_journal_for_every_gpu_stage():
    text = CHAIN.read_text(encoding="utf-8")
    assert "gpu_session_open" in text and "gpu_session_close" in text
    for stage in ("stage_prompt", "stage_distill", "stage_sft", "stage_eval",
                  "stage_ship"):
        body = text[text.index(f"{stage}()"):]
        body = body[:body.index("\nstage_") if "\nstage_" in body[1:] else len(body)]
        assert "gpu_session_open" in body, stage
    # and the exit trap closes it, so a kill still charges the card
    assert "gpu_session_close interrupted" in text


def test_the_chain_is_still_valid_bash():
    r = subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
