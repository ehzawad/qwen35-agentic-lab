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
def grpo():
    from agentlab.data import build_grpo

    return build_grpo(n=32)


class TestPreference:
    def test_explicit_prompt_form_for_dpo(self):
        from agentlab.data import build_preference

        ds = build_preference(n=16, explicit_prompt=True)
        assert set(ds.column_names) == {"prompt", "chosen", "rejected", "chat_template_kwargs"}
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


class TestDPOAlignment:
    """The prompt render must be a token prefix of the prompt+completion render.

    DPOTrainer slices chosen_ids = prompt_chosen_ids[len(prompt_ids):]. When the
    prefix property fails it only logs a warning and trains on the misaligned
    completion, so this invariant has to be asserted here instead.
    """

    def test_prompt_is_a_token_prefix_of_prompt_plus_chosen(self):
        from agentlab import env
        from agentlab.data import build_preference

        tok = env.get_tokenizer(env.load_processor())
        ds = build_preference(n=8, explicit_prompt=True)
        for row in ds:
            kw = row.get("chat_template_kwargs") or {}
            p = tok.apply_chat_template(row["prompt"], tokenize=False,
                                        add_generation_prompt=True, **kw)
            for branch in ("chosen", "rejected"):
                f = tok.apply_chat_template(list(row["prompt"]) + list(row[branch]),
                                            tokenize=False, **kw)
                pi, fi = tok(p).input_ids, tok(f).input_ids
                assert fi[:len(pi)] == pi, f"{branch} misaligned; DPO would slice wrongly"

    def test_thinking_is_disabled_for_preference_rows(self):
        from agentlab.data import build_preference

        ds = build_preference(n=4, explicit_prompt=True)
        assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}
