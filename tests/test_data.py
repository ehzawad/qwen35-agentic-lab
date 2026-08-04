"""Each builder must emit exactly the dataset type its trainer expects.

A dataset that is merely well-formed is not enough: TRL infers the task from
the column names, and getting that wrong does not raise -- it silently trains
the wrong objective.

These read from the local HF cache and need no network and no GPU.
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")


@pytest.fixture(scope="module")
def grpo():
    from agentlab.data import build_grpo

    return build_grpo(n=32)


class TestGRPO:
    """GRPOTrainer prompt-only type, plus the ground truth the reward needs."""

    def test_columns(self, grpo):
        assert set(grpo.column_names) == {"prompt", "ground_truth", "chat_template_kwargs"}

    def test_thinking_disabled_on_the_CONFIG_not_just_the_rows(self):
        # The row-level key is NOT enough and asserting it was the bug: TRL keeps
        # only `self.chat_template_kwargs = args.chat_template_kwargs or {}` and
        # renders rollout prompts from that global dict (grpo_trainer.py:742,
        # :1758). Row kwargs reach the reward functions and never the renderer,
        # so GRPO sampled with thinking ON while SFT trained with it OFF.
        import inspect

        from agentlab import grpo

        src = inspect.getsource(grpo.main)
        assert 'chat_template_kwargs={"enable_thinking": False}' in src, \
            "GRPOConfig must disable thinking; the dataset column cannot do it"

    def test_prompt_is_conversational_and_ends_on_the_user(self, grpo):
        p = grpo[0]["prompt"]
        assert p[0]["role"] == "system" and p[-1]["role"] == "user"

    def test_no_assistant_turn_leaks_into_the_prompt(self, grpo):
        # A prompt-only dataset containing the answer would make the reward
        # trivially satisfiable and the whole run meaningless.
        for row in grpo.select(range(min(16, len(grpo)))):
            assert all(m["role"] != "assistant" for m in row["prompt"])

    def test_ground_truth_parses_as_a_number(self, grpo):
        # Deliberately not asserting the answer is absent from the question:
        # GSM8K answers are derived from numbers that legitimately recur in the
        # text ("5" is a substring of "50"), so any such check is a false alarm.
        # Leakage is covered by test_no_assistant_turn_leaks_into_the_prompt.
        for row in grpo.select(range(min(16, len(grpo)))):
            float(row["ground_truth"])

    def test_eval_split_is_disjoint_from_train(self):
        from agentlab.data import build_eval, build_grpo

        train = {r["prompt"][-1]["content"] for r in build_grpo(n=64)}
        held = {r["prompt"][-1]["content"] for r in build_eval(n=64)}
        assert not (train & held), "eval questions appear in the GRPO training set"


class TestToolSuite:
    def test_schemas_generate_for_every_tool(self):
        from agentlab.tools import TOOLS, tool_schemas

        schemas = tool_schemas()
        assert len(schemas) == len(TOOLS)
        for s in schemas:
            fn = s["function"]
            assert fn["description"], f"{fn['name']} has no description for the model to read"
            for name, spec in fn["parameters"]["properties"].items():
                assert spec.get("description"), f"{fn['name']}.{name} undocumented"


