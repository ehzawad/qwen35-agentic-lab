"""Each builder must emit exactly the dataset type its trainer expects.

A dataset that is merely well-formed is not enough: TRL infers the task from the
column names, so `chosen`/`rejected` with an explicit `prompt` trains DPO while
the same rows without it are what RewardTrainer wants. Getting that wrong does
not raise -- it silently trains the wrong objective.

These read from the local HF cache and need no network and no GPU.
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")


@pytest.fixture(scope="module")
def sft():
    from agentlab.data import build_sft

    return build_sft(n=32)


@pytest.fixture(scope="module")
def grpo():
    from agentlab.data import build_grpo

    return build_grpo(n=32)


class TestSFT:
    """SFTTrainer language-modeling type, plus the `tools` column."""

    def test_columns(self, sft):
        assert set(sft.column_names) == {"messages", "tools"}

    def test_rows_survived_conversion(self, sft):
        assert len(sft) > 0, "every xlam row was dropped by the converter"

    def test_assistant_turn_is_a_tool_call(self, sft):
        msgs = sft[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["tool_calls"], "the SFT target must be a tool call, not prose"

    def test_tool_schemas_are_openai_shaped(self, sft):
        for tool in sft[0]["tools"]:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert fn["name"]
            assert fn["parameters"]["type"] == "object"
            assert isinstance(fn["parameters"]["properties"], dict)
            assert isinstance(fn["parameters"]["required"], list)

    def test_called_function_is_among_the_offered_tools(self, sft):
        # If the target calls a function that was never offered, SFT teaches the
        # model to hallucinate tools rather than to use the ones it was given.
        for row in sft.select(range(min(16, len(sft)))):
            offered = {t["function"]["name"] for t in row["tools"]}
            called = {tc["function"]["name"] for tc in row["messages"][-1]["tool_calls"]}
            assert called <= offered, f"{called - offered} called but not offered"

    def test_renders_through_the_real_chat_template(self, sft):
        from agentlab import env

        proc = env.load_processor()
        tok = env.get_tokenizer(proc)
        row = sft[0]
        text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        assert row["messages"][-1]["tool_calls"][0]["function"]["name"] in text


class TestPreference:
    def test_explicit_prompt_form_for_dpo(self):
        from agentlab.data import build_preference

        ds = build_preference(n=16, explicit_prompt=True)
        assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
        row = ds[0]
        assert row["prompt"][0]["role"] == "user"
        # chosen/rejected must be completions only, else the prompt is duplicated
        assert all(m["role"] == "assistant" for m in row["chosen"])
        assert all(m["role"] == "assistant" for m in row["rejected"])

    def test_implicit_prompt_form_for_reward_model(self):
        from agentlab.data import build_preference

        ds = build_preference(n=16, explicit_prompt=False)
        assert set(ds.column_names) == {"chosen", "rejected"}
        # RewardTrainer scores whole conversations, so the user turn must be present
        assert ds[0]["chosen"][0]["role"] == "user"

    def test_chosen_and_rejected_differ(self):
        from agentlab.data import build_preference

        ds = build_preference(n=16, explicit_prompt=True)
        differing = sum(1 for r in ds if r["chosen"] != r["rejected"])
        assert differing == len(ds), "identical pairs carry no preference signal"


class TestGRPO:
    """GRPOTrainer prompt-only type, plus the ground truth the reward needs."""

    def test_columns(self, grpo):
        assert set(grpo.column_names) == {"prompt", "ground_truth"}

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
