"""Tool-call parsing, argument coercion, and calculator safety.

The parser is the single highest-leverage surface in the repo: if it silently
returns zero calls, GRPO reads that as "the policy never uses tools" and trains
against a reward signal that is measuring the harness rather than the model.
Every case below is one that actually occurs in generated output.
"""

from __future__ import annotations

import pytest

from agentlab.chat import boxed_answer, extract_thinking, parse_tool_calls, strip_thinking
from agentlab.tools import calculator, call_tool, coerce_args, kb_lookup, unit_convert

XML = """<tool_call>
<function=calculator>
<parameter=expression>
4871 * 209 - 1337
</parameter>
</function>
</tool_call>"""

# Thinking eats the completion budget, so a rollout truncated at
# max_completion_length routinely ends mid-call. Dropping these would bias the
# tool-use reward against exactly the rollouts that were trying hardest.
TRUNCATED = """<tool_call>
<function=unit_convert>
<parameter=value>
26.2
</parameter>
<parameter=from_unit>
mile
</parameter>
<parameter=to_unit>
km"""

JSON_FORM = '<tool_call>{"name": "kb_lookup", "arguments": {"key": "a6000"}}</tool_call>'


class TestParseToolCalls:
    def test_xml_form(self):
        calls = parse_tool_calls(XML)
        assert calls == [{"name": "calculator", "arguments": {"expression": "4871 * 209 - 1337"}}]

    def test_truncated_call_still_parses(self):
        calls = parse_tool_calls(TRUNCATED)
        assert len(calls) == 1
        assert calls[0]["name"] == "unit_convert"
        assert calls[0]["arguments"]["to_unit"] == "km"

    def test_json_form_still_supported(self):
        assert parse_tool_calls(JSON_FORM) == [
            {"name": "kb_lookup", "arguments": {"key": "a6000"}}
        ]

    def test_two_calls_in_one_block(self):
        text = (
            "<tool_call>\n"
            "<function=calculator>\n<parameter=expression>\n1+1\n</parameter>\n</function>\n"
            "<function=kb_lookup>\n<parameter=key>\ngrpo\n</parameter>\n</function>\n"
            "</tool_call>"
        )
        assert [c["name"] for c in parse_tool_calls(text)] == ["calculator", "kb_lookup"]

    def test_two_separate_blocks(self):
        text = XML + "\n" + JSON_FORM
        assert [c["name"] for c in parse_tool_calls(text)] == ["calculator", "kb_lookup"]

    def test_thinking_before_call_is_ignored(self):
        assert parse_tool_calls("<think>I should multiply</think>\n" + XML)[0]["name"] == "calculator"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "no tool call here at all",
            "<tool_call>not a call</tool_call>",
            "<tool_call></tool_call>",
            '<tool_call>{"broken": json</tool_call>',
            "<tool_call>\n<function=>\n</function>\n</tool_call>",
        ],
    )
    def test_garbage_yields_no_calls_and_does_not_raise(self, text):
        # Early in RL the policy emits nonsense. That must score zero, not crash
        # the rollout worker and take the run down with it.
        assert parse_tool_calls(text) == []

    def test_expression_containing_angle_brackets(self):
        text = (
            "<tool_call>\n<function=calculator>\n"
            "<parameter=expression>\n3 < 4\n</parameter>\n</function>\n</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert calls and calls[0]["arguments"]["expression"] == "3 < 4"

    def test_multiline_parameter_body_is_preserved(self):
        text = (
            "<tool_call>\n<function=kb_lookup>\n"
            "<parameter=key>\nline one\nline two\n</parameter>\n</function>\n</tool_call>"
        )
        assert parse_tool_calls(text)[0]["arguments"]["key"] == "line one\nline two"


class TestCoerceArgs:
    def test_numeric_strings_become_numbers(self):
        # The wire format carries every parameter as text. Without coercion each
        # numeric call the model got RIGHT would raise TypeError and be punished.
        args = coerce_args("unit_convert", {"value": "26.2", "from_unit": "mile", "to_unit": "km"})
        assert args["value"] == pytest.approx(26.2)
        assert isinstance(args["value"], float)

    def test_uncastable_value_is_left_for_the_tool_to_reject(self):
        args = coerce_args("unit_convert", {"value": "not-a-number"})
        assert args["value"] == "not-a-number"

    def test_unknown_tool_passes_through(self):
        assert coerce_args("nope", {"a": "1"}) == {"a": "1"}

    def test_postponed_annotations_do_not_defeat_the_cast(self):
        # tools.py uses `from __future__ import annotations`, so __annotations__
        # holds the *string* "float". Regression guard for using get_type_hints.
        assert isinstance(coerce_args("unit_convert", {"value": "1"})["value"], float)


class TestCalculator:
    @pytest.mark.parametrize(
        "expr,want",
        [("2+2", "4"), ("4871 * 209 - 1337", "1016702"), ("10/4", "2.5"), ("-3 * -3", "9"), ("7 % 3", "1")],
    )
    def test_arithmetic(self, expr, want):
        assert calculator(expr) == want

    def test_integral_float_renders_without_decimal(self):
        # GSM8K ground truth is compared as text in places; 12.0 != "12".
        assert calculator("6*2.0") == "12"

    @pytest.mark.parametrize(
        "expr",
        ["import os", "__import__('os').system('ls')", "open('/etc/passwd')", "1/0", "9**9**9", "[].__class__"],
    )
    def test_hostile_input_is_refused_not_executed(self, expr):
        assert calculator(expr).startswith("error:")


class TestOtherTools:
    def test_unit_convert_round_trip(self):
        assert unit_convert(26.2, "mile", "km") == "42.1648"

    def test_cross_family_conversion_refused(self):
        assert unit_convert(1, "kg", "m").startswith("error:")

    def test_kb_lookup_is_case_insensitive(self):
        assert "48 GB" in kb_lookup("A6000")

    def test_unknown_key_lists_valid_keys(self):
        out = kb_lookup("nonexistent")
        assert out.startswith("error:") and "grpo" in out

    def test_call_tool_rejects_unknown_tool(self):
        assert call_tool("no_such_tool", {}).startswith("error:")

    def test_call_tool_reports_bad_arguments(self):
        assert call_tool("unit_convert", {"wrong": 1}).startswith("error:")


class TestThinkingAndAnswer:
    def test_dangling_close_tag_is_handled(self):
        # The generation prompt already opens <think>, so completions usually
        # carry only the closing tag. Mishandling this leaks the whole chain of
        # thought into what we score as the answer.
        assert strip_thinking("reasoning here</think>\n\nThe answer is 4.") == "The answer is 4."

    def test_balanced_tags_are_handled(self):
        assert strip_thinking("<think>r</think>answer") == "answer"

    def test_extract_thinking_both_forms(self):
        assert extract_thinking("<think>abc</think>x") == "abc"
        assert extract_thinking("abc</think>x") == "abc"

    def test_no_thinking_returns_text_unchanged(self):
        assert strip_thinking("just an answer") == "just an answer"

    @pytest.mark.parametrize(
        "text,want",
        [
            (r"so \boxed{42}", "42"),
            (r"\boxed{1,234}", "1234"),
            (r"\boxed{$5}", "5"),
            (r"\boxed{1} then \boxed{2}", "2"),
        ],
    )
    def test_boxed_answer(self, text, want):
        assert boxed_answer(text) == want

    def test_missing_box_is_none(self):
        assert boxed_answer("no box here") is None
