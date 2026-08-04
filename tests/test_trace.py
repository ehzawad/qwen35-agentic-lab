"""The trace layer must be inert when off, and must never break a run when on."""

from __future__ import annotations

import json

import pytest

from agentlab import trace


@pytest.fixture
def traced(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    monkeypatch.setenv("AGENTLAB_TRACE", str(path))
    return path


class TestGating:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AGENTLAB_TRACE", raising=False)
        assert not trace.enabled()
        trace.emit("episode", task="x")  # must be a no-op, not an error

    def test_enabled_when_env_set(self, traced):
        assert trace.enabled() and trace.path() == traced


class TestEpisode:
    def test_records_a_full_episode(self, traced):
        ep = trace.Episode("q", index=0, ground_truth="5")
        ep.turn(thinking="think", calls=[{"name": "calculator", "arguments": {"expression": "1"}}],
                results=["1"])
        ep.turn(text=r"\boxed{5}")
        ep.finish(r"\boxed{5}", ok=True, correctness=1.0)

        rec = json.loads(traced.read_text().strip())
        assert rec["kind"] == "episode"
        assert rec["ok"] is True
        assert rec["n_turns"] == 2
        assert rec["n_calls"] == 1
        assert rec["n_tool_errors"] == 0
        assert rec["rewards"]["correctness"] == 1.0

    def test_counts_tool_errors(self, traced):
        ep = trace.Episode("q")
        ep.turn(calls=[{"name": "calculator", "arguments": {}}], results=["error: bad"])
        ep.finish("x", ok=False)
        assert json.loads(traced.read_text().strip())["n_tool_errors"] == 1

    def test_every_line_is_valid_json(self, traced):
        for i in range(5):
            ep = trace.Episode("q", index=i)
            ep.turn(text="t")
            ep.finish("f", ok=bool(i % 2))
        lines = [l for l in traced.read_text().splitlines() if l.strip()]
        assert len(lines) == 5
        for line in lines:
            json.loads(line)


class TestRobustness:
    def test_unserialisable_values_do_not_raise(self, traced):
        ep = trace.Episode("q")
        ep.turn(calls=[{"name": "x", "arguments": {"o": object()}}], results=["r"])
        ep.finish("f", ok=True)
        json.loads(traced.read_text().strip())  # default=str kept it parseable

    def test_unwritable_path_is_swallowed(self, monkeypatch):
        # Tracing is diagnostics; it must never take down a training run.
        monkeypatch.setenv("AGENTLAB_TRACE", "/proc/cannot/write/here.jsonl")
        trace.Episode("q").finish("f", ok=True)


class TestRender:
    def test_missing_file_reports_cleanly(self, tmp_path):
        assert "no trace" in trace.render(tmp_path / "nope.jsonl")

    def test_renders_summary_metrics(self, traced):
        for i in range(4):
            ep = trace.Episode("q", index=i, ground_truth="1")
            ep.turn(calls=[{"name": "calculator", "arguments": {}}], results=["1"])
            ep.finish("f", ok=(i < 3))
        out = trace.render(traced, limit=2)
        assert "accuracy        0.750" in out
        assert "tool_use_rate   1.000" in out
        assert "episodes        4" in out

    def test_ignores_malformed_lines(self, traced):
        ep = trace.Episode("q", index=0)
        ep.turn(text="t")
        ep.finish("f", ok=True)
        with traced.open("a") as fh:
            fh.write("not json at all\n")
        assert "episodes        1" in trace.render(traced)


class TestRolloutRendering:
    """GRPO emits `rollout` records; rendering only `episode` made it write-only."""

    def test_grpo_rollouts_render_instead_of_reporting_nothing(self, traced, monkeypatch):
        from agentlab import grpo

        comps = [
            [{"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "calculator", "arguments": {}}}]},
             {"role": "tool", "name": "calculator", "content": "250"},
             {"role": "assistant", "content": r"\boxed{250}"}],
            [{"role": "assistant", "content": "no answer"}],
        ]
        gt = ["250", "250"]
        # All three must fire before anything is written.
        grpo.correctness_reward(comps, gt)
        assert not traced.exists() or traced.read_text() == ""
        grpo.tool_use_reward(comps)
        grpo.format_reward(comps)

        from agentlab import trace as tr

        out = tr.render(traced)
        assert "no episodes" not in out
        assert "rollouts        2" in out
        assert "accuracy        0.500" in out

    def test_every_reward_component_is_recorded(self, traced):
        import json

        from agentlab import grpo

        comps = [[{"role": "assistant", "content": r"\boxed{7}"}]]
        grpo.correctness_reward(comps, ["7"])
        grpo.tool_use_reward(comps)
        grpo.format_reward(comps)
        rec = json.loads(traced.read_text().strip())
        assert set(rec["rewards"]) == set(grpo.REWARD_NAMES)
        assert rec["rewards"]["correctness"] == 1.0
        assert "total" in rec

    def test_weights_are_a_single_source_of_truth(self):
        from agentlab import grpo

        assert len(grpo.REWARD_WEIGHTS) == len(grpo.REWARD_NAMES)


class TestRewardBufferKeying:
    """_record keys on id(completions); callers must pass ONE list object.

    Three separate `[comp]` literals never complete a triple, and sporadically
    emit a record with scores mixed across rollouts when CPython recycles a
    freed id. That produced 23 bogus records in a real eval trace.
    """

    COMP = [{"role": "assistant", "content": r"\boxed{7}"}]

    def test_shared_list_emits_exactly_one_complete_record(self, traced):
        import json

        from agentlab import grpo

        batch = [self.COMP]
        grpo.correctness_reward(batch, ["7"])
        grpo.tool_use_reward(batch)
        grpo.format_reward(batch)
        recs = [json.loads(l) for l in traced.read_text().splitlines() if l.strip()]
        assert len(recs) == 1
        assert set(recs[0]["rewards"]) == set(grpo.REWARD_NAMES)
        assert recs[0]["rewards"]["correctness"] == 1.0

    def test_buffer_is_capped_against_orphaned_partials(self, traced):
        from agentlab import grpo

        grpo._REWARD_BUF.clear()
        for i in range(200):
            grpo._record("correctness", [[{"role": "assistant", "content": str(i)}]], [0.0])
        assert len(grpo._REWARD_BUF) <= grpo._REWARD_BUF_MAX + 1

    def test_eval_scores_through_a_single_batch(self):
        # Pin the calling convention itself: eval must not rebuild the list.
        import inspect

        from agentlab import eval as ev

        src = inspect.getsource(ev.evaluate)
        assert "batch = [comp]" in src
        assert "correctness_reward([comp]" not in src
