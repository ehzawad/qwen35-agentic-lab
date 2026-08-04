"""The rejection-sampling stage: acceptance filter and training-row encoding.

The audit found this module load-bearing but untested -- it produced the corpus
behind the headline result.
"""

from __future__ import annotations

from agentlab.distill import accept, to_sft_rows


def _traj(final=r"The answer is \boxed{42}.", gt="42", n_calls=2,
          tool_error=False, tools_used=None, **flags):
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "calculator", "arguments": {"expression": "6*7"}}}]},
        {"role": "tool", "name": "calculator", "content": "42"},
        {"role": "assistant", "content": final},
    ]
    return {"final": final, "gt": gt, "n_calls": n_calls, "tool_error": tool_error,
            "tools_used": tools_used or ["calculator"], "messages": msgs, **flags}


class TestAccept:
    def test_good_trajectory_passes(self):
        assert accept(_traj(), 6) == (True, "")

    def test_rejections_each_have_their_own_bucket(self):
        cases = [
            (_traj(final=""), "no_final"),
            (_traj(gt="99"), "wrong"),
            (_traj(n_calls=0), "no_tool"),
            (_traj(n_calls=99), "too_many_calls"),
            (_traj(tool_error=True), "tool_error"),
            (_traj(exhausted=True), "max_turns_exhausted"),
            (_traj(truncated=True), "truncated_final"),
            (_traj(tools_used=["kb_lookup"]), "irrelevant_tool"),
            (_traj(final="w " * 200 + r"\boxed{42}"), "final_too_long"),
            (_traj(final=r"x</think> \boxed{42}"), "stray_think"),
        ]
        for traj, want in cases:
            ok, why = accept(traj, 6)
            assert not ok and why == want, (want, why)

    def test_notation_tolerant_ground_truth(self):
        # the normalizer fix applies here too: 24\% must match gt 24
        assert accept(_traj(final=r"\boxed{24\%}", gt="24"), 6)[0]

    def test_error_recovery_admitted_only_under_quota(self):
        t = _traj(tool_error=True)
        assert not accept(t, 6)[0]
        assert accept(t, 6, allow_tool_error=True)[0]


class TestToSftRows:
    def test_exactly_one_row_terminal_turn_only(self):
        rows = to_sft_rows(_traj())
        assert len(rows) == 1
        assert rows[0]["completion"][0]["content"].endswith(r"\boxed{42}.")
        assert rows[0]["prompt"][-1]["role"] == "tool"

    def test_thinking_disabled_on_the_row(self):
        assert to_sft_rows(_traj())[0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_loss_never_lands_on_tool_output(self):
        row = to_sft_rows(_traj())[0]
        assert all(m["role"] != "tool" for m in row["completion"])
