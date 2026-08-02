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

    def test_prompt_completion_shape_not_messages(self, sft):
        # Decides whether SFT loss lands on the tool call or on the schema
        # preamble: TRL enables completion_only_loss ONLY for prompt/completion.
        assert set(sft.column_names) == {"prompt", "completion", "tools"}

    def test_rows_survived_conversion(self, sft):
        assert len(sft) > 0, "every xlam row was dropped by the converter"

    def test_completion_is_a_tool_call(self, sft):
        assert sft[0]["prompt"][0]["role"] == "user"
        comp = sft[0]["completion"]
        assert comp[-1]["role"] == "assistant"
        assert comp[-1]["tool_calls"], "the SFT target must be a tool call, not prose"

    def test_completion_only_loss_would_be_enabled(self, sft):
        # Mirrors TRL: completion_only_loss = "prompt" in s and "completion" in s
        s = sft[0]
        assert "prompt" in s and "completion" in s

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
            called = {tc["function"]["name"] for tc in row["completion"][-1]["tool_calls"]}
            assert called <= offered, f"{called - offered} called but not offered"

    def test_renders_through_the_real_chat_template(self, sft):
        from agentlab import env

        proc = env.load_processor()
        tok = env.get_tokenizer(proc)
        row = sft[0]
        msgs = list(row["prompt"]) + list(row["completion"])
        text = tok.apply_chat_template(msgs, tools=row["tools"], tokenize=False)
        assert row["completion"][-1]["tool_calls"][0]["function"]["name"] in text


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


class TestXlamSchemaConversion:
    """Optionality and types must survive the xlam -> JSON Schema conversion."""

    def test_optional_suffix_in_type_is_honoured(self):
        from agentlab.data import _xlam_tool_to_schema

        s = _xlam_tool_to_schema({
            "name": "f", "description": "d",
            "parameters": {
                "must": {"type": "str", "description": "x"},
                "maybe": {"type": "str, optional", "description": "y"},
            },
        })
        req = s["function"]["parameters"]["required"]
        assert "must" in req and "maybe" not in req

    def test_supplied_default_marks_optional_even_when_falsy(self):
        from agentlab.data import _xlam_tool_to_schema

        s = _xlam_tool_to_schema({
            "name": "f", "description": "d",
            "parameters": {
                "a": {"type": "int", "description": "x", "default": 0},
                "b": {"type": "str", "description": "y", "default": ""},
                "c": {"type": "str", "description": "z"},
            },
        })
        req = s["function"]["parameters"]["required"]
        assert req == ["c"], f"empty/zero defaults are still defaults, got {req}"

    def test_container_types_map_to_array(self):
        from agentlab.data import _json_type

        for t in ("list", "List[int]", "tuple", "Tuple[str, int]", "set"):
            assert _json_type(t) == "array", t

    def test_scalar_types(self):
        from agentlab.data import _json_type

        assert _json_type("int") == "integer"
        assert _json_type("float") == "number"
        assert _json_type("bool") == "boolean"
        assert _json_type("str") == "string"


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
