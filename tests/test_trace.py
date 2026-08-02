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
