"""Chat-template plumbing: thinking-mode control, tool-call parsing, agent loop.

Kept separate from the stage scripts because SFT, GRPO and eval all need the same
answers to "how do I turn a message list into tokens" and "what did the model
just try to call".
"""

from __future__ import annotations

import json
import re

from .tools import call_tool, coerce_args

# Qwen3.5 does NOT emit JSON tool calls. Its template uses an XML-ish form
# (the one vLLM parses with --tool-call-parser qwen3_coder):
#
#   <tool_call>
#   <function=calculator>
#   <parameter=expression>
#   2+2
#   </parameter>
#   </function>
#   </tool_call>
#
# Verified by round-tripping a tool_calls message through apply_chat_template.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?(?:</parameter>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def strip_thinking(text: str) -> str:
    """Drop the reasoning trace, leaving the user-facing answer.

    The generation prompt already ends with an open `<think>`, so a completion
    typically carries only the *closing* tag. Handle the dangling case, else the
    whole chain of thought leaks into what we treat as the answer.
    """
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_thinking(text: str) -> str:
    """Return just the reasoning trace, empty string when the model did not think."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "</think>" in text:  # opening tag came from the prompt, not the completion
        return text.split("</think>")[0].strip()
    return ""


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from a completion.

    Returns a list of {"name": str, "arguments": dict}. Handles the Qwen XML form
    and falls back to the JSON form other checkpoints use. Malformed calls are
    skipped rather than raised: during RL the policy emits garbage early on, and
    that must score zero rather than kill the run.

    The regexes tolerate a missing closing tag so a completion truncated at
    max_completion_length still yields the call it was midway through -- that is
    a real and frequent case once thinking eats into the token budget.
    """
    calls = []
    for blob in _TOOL_CALL_RE.findall(text):
        blob = blob.strip()

        matched = False
        for name, body in _FUNCTION_RE.findall(blob):
            args = {k: v for k, v in _PARAM_RE.findall(body)}
            calls.append({"name": name.strip(), "arguments": coerce_args(name.strip(), args)})
            matched = True
        if matched:
            continue

        # JSON form: {"name": ..., "arguments": {...}}
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):  # some templates double-encode the arguments
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)


def assistant_tool_message(text: str, calls: list[dict]) -> dict:
    """The ONE assistant-turn shape for a decision that called tools.

    `{"role": "assistant", "tool_calls": [{"type": "function", "function":
    {"name", "arguments"}}], "content": prose}` -- prose only when there is any,
    because an empty `content` renders differently from an absent one.

    Both the training rollout engine and the claim-bearing evaluator build the
    assistant turn here. The evaluator used to append the raw content and DROP the
    tool-call object, so the transcript the model conditioned on during evaluation
    contained an empty assistant message where the trained-on transcript contained
    a structured call. That is invisible to observation digests and visible in the
    rendered token ids, which is why the parity test compares those.
    """
    prose = _TOOL_CALL_BLOCK_RE.sub("", text or "").strip()
    msg = {"role": "assistant",
           "tool_calls": [{"type": "function",
                           "function": {"name": c["name"],
                                        "arguments": c.get("arguments", {})}}
                          for c in calls]}
    if prose:
        msg["content"] = prose
    return msg


def boxed_answer(text: str) -> str | None:
    """Return the last \\boxed{...} payload, normalised for numeric comparison."""
    hits = _BOXED_RE.findall(text)
    if not hits:
        return None
    return hits[-1].strip().replace(",", "").replace("$", "").replace(" ", "")


def render(proc, messages: list[dict], tools: list[dict] | None = None,
           enable_thinking: bool = True) -> str:
    """Apply the chat template for generation.

    enable_thinking is passed through the template kwargs; Qwen3.5 thinks by
    default, so this is how a stage opts out.
    """
    return proc.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def run_agent_loop(model, proc, messages: list[dict], tools_schemas: list[dict],
                   max_turns: int = 4, max_new_tokens: int = 768,
                   enable_thinking: bool = True, verbose: bool = True,
                   episode=None, dispatcher=call_tool, **gen_kw) -> tuple[list[dict], str]:
    """Generate, execute any tool calls, feed results back, repeat.

    Returns the full transcript and the final assistant text. Stops as soon as a
    turn produces no tool calls, or after max_turns.

    Pass `episode` (a trace.Episode) to record each turn's thinking, calls and
    tool results. It is optional and costs nothing when absent.

    `dispatcher(name, arguments) -> str` routes tool execution. The default is
    the global call_tool; suite evaluation passes an EpisodeRuntime so each
    rollout gets its own KB view, state and fault schedule. If the dispatcher
    exposes begin_decision(), it is called once per assistant turn -- that is
    the logical clock the rate-limit fault contract counts in.
    """
    import torch

    convo = list(messages)
    final = ""
    device = next(model.parameters()).device
    begin_decision = getattr(dispatcher, "begin_decision", None)

    for turn in range(max_turns):
        if begin_decision is not None:
            begin_decision()
        text = render(proc, convo, tools=tools_schemas, enable_thinking=enable_thinking)
        inputs = proc(text=[text], return_tensors="pt").to(device) if _is_processor(proc) \
            else proc([text], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, **gen_kw)
        gen = out[0][inputs["input_ids"].shape[1]:]
        completion = _decode(proc, gen)

        calls = parse_tool_calls(completion)
        if verbose:
            think = extract_thinking(completion)
            if think:
                print(f"  [turn {turn}] thought {len(think)} chars")
            print(f"  [turn {turn}] {len(calls)} tool call(s)")

        if not calls:
            final = strip_thinking(completion)
            convo.append({"role": "assistant", "content": final})
            if episode is not None:
                episode.turn(thinking=extract_thinking(completion), text=final)
            record_final = getattr(dispatcher, "finalize_text", None)
            if record_final is not None:
                record_final(final)
            break

        convo.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            }
        )
        results = []
        for c in calls:
            result = dispatcher(c["name"], c["arguments"])
            results.append(result)
            if verbose:
                print(f"    -> {c['name']}({c['arguments']}) = {result}")
            convo.append({"role": "tool", "name": c["name"], "content": result})

        if episode is not None:
            episode.turn(thinking=extract_thinking(completion), calls=calls, results=results)

    return convo, final


def _is_processor(proc) -> bool:
    return hasattr(proc, "tokenizer")


def _decode(proc, ids) -> str:
    # skip_special_tokens=True is safe here: <tool_call>/<function=...> are plain
    # text in this template, so the call survives while <|im_end|>/<|endoftext|> go.
    tok = getattr(proc, "tokenizer", proc)
    return tok.decode(ids, skip_special_tokens=True).strip()

_LATEX_STRIP = re.compile(r"\\(?:%|\$|,|;|!|\s)|\\text\{([^{}]*)\}")


def numeric_answer(s) -> float | None:
    """Parse a model answer to a float, tolerating the notation models emit.

    The scorer previously rejected `\\boxed{24\\%}` against ground truth 24 --
    a real base-eval episode scored wrong purely on notation (found by review,
    verified in the trace). Strips LaTeX percent/dollar/space escapes and
    \\text{} wrappers, then plain %, $, commas and spaces. Returns None when no
    number remains.
    """
    if s is None:
        return None
    t = str(s)
    t = _LATEX_STRIP.sub(lambda m: m.group(1) or "", t)
    t = t.replace("%", "").replace("$", "").replace(",", "").replace(" ", "").strip()
    t = t.rstrip("\\.")
    try:
        return float(t)
    except ValueError:
        return None

