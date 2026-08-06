"""THE SERVED REQUEST SHAPE: what the evaluator posts must be what the API takes.

Probe 3 of the dev preflight stopped the chain here. D2 made the evaluator
PRESERVE the assistant tool-call object (dropping it was the transcript drift),
and the preserved object is the TEMPLATE form:

    {"type": "function", "function": {"name": ..., "arguments": {dict}}}

The OpenAI-compatible request model in front of vLLM requires `id` and a JSON
*string* for `function.arguments`, so every clean episode died on its SECOND
decision -- the first one that echoes a tool call back -- with

    ('body', 0, 'ChatCompletionMessageFunctionToolCallParam', 'id')
        -> Field required
    ('body', 0, 'ChatCompletionMessageFunctionToolCallParam',
     'function', 'arguments')                   -> Input should be a string

against a perfectly healthy server: six transport failures, zero scored rows.

Why the existing transport tests missed it: every one of them monkeypatches
`requests.post` and posts a SINGLE USER TURN, so no assistant tool-call echo was
ever validated against the real request model. This module fixes that class of
blind spot three ways:

  1. the exact failing shape is validated against vLLM's OWN request model, and
     must still be rejected -- so the defect stays pinned;
  2. the shape the evaluator posts now is validated against that same model, and
     must be accepted;
  3. a full multi-decision episode runs through `evaluate.make_http_chat` with a
     fake transport that validates EVERY payload it is handed. That is probe 3's
     path with no GPU and no server.

And the parity half: the canonical form and the served form must render to
IDENTICAL token ids. vLLM's `_postprocess_messages` parses `arguments` back into
the mapping the Qwen3.5 template needs (`tool_call.arguments|items`), which is
exactly why the conversion can live at the transport boundary without becoming a
second transcript. If that ever stopped holding, the served model would condition
on different bytes from the offline/canonical path and the eight-surface parity
assertion would be measuring the wrong thing.

CPU-only. Starts no server and touches no card.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from agentlab import chat
from agentlab.suite import evaluate
from agentlab.suite import runtime as rt_mod
from agentlab.suite.generate import build_task, certification_spec

SUITE = "agentlab-suite-v1"
SEED = 0xA61E0002
SECRET = bytes.fromhex("7e" * 32)
BASE_MODEL = "Qwen/Qwen3.5-4B"
DECODE = {"temperature": 0.0, "top_p": 1.0, "seed": 2786983945,
          "max_tokens": 256, "enable_thinking": False}

# The exact call probe 3 died on (dev-lookup_chain-h2-0000, kb_lookup on a start
# key), reproduced as a literal so the pinned defect does not depend on a
# generator.
FAILING_CALL = {"name": "kb_lookup", "arguments": {"key": "KD6FJTKOG2CBCSFIV"}}


def _request_model():
    """vLLM's real request model -- the thing that answered HTTP 400."""
    pytest.importorskip("vllm", reason="the served request contract needs vLLM")
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest)

    return ChatCompletionRequest


def _episode_prefix(tools):
    """[system, user, assistant-with-tool-call, tool-result] -- the shape that

    a second decision posts, built through the ONE canonical assistant builder.
    """
    return [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "Resolve the chain and answer."},
        evaluate.assistant_message("calling kb_lookup", [FAILING_CALL]),
        {"role": "tool", "name": "kb_lookup",
         "content": '{"ok":true,"value":41.5}\nreceipt: r-abc123'},
    ]


# ---------------------------------------------------------------------------
# 1. the defect stays pinned
# ---------------------------------------------------------------------------

def test_the_canonical_assistant_turn_is_still_rejected_by_the_real_request_model():
    """The canonical (template) form must NOT validate: both errors, by loc.

    This is the assertion that makes the rest of the module meaningful. If a
    future engine started accepting the mapping form, this test fails and the
    transport conversion becomes optional -- a fact we would want to learn from a
    red test rather than from a silent behaviour change.
    """
    ChatCompletionRequest = _request_model()
    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    with pytest.raises(Exception) as exc:
        ChatCompletionRequest(model="m", messages=_episode_prefix(schemas),
                              tools=schemas)
    errors = getattr(exc.value, "errors", None)
    assert errors is not None, f"expected a pydantic validation error, got {exc.value!r}"
    locs = [".".join(str(p) for p in e["loc"]) for e in exc.value.errors()]
    types = {e["type"] for e in exc.value.errors()}
    assert any(loc.endswith("ChatCompletionMessageFunctionToolCallParam.id")
               for loc in locs), locs
    assert any(loc.endswith("function.arguments") for loc in locs), locs
    assert {"missing", "string_type"} <= types, types


# ---------------------------------------------------------------------------
# 2. the shape the evaluator posts now
# ---------------------------------------------------------------------------

def test_the_served_shape_carries_an_id_and_a_string_arguments_field():
    served = chat.served_messages(_episode_prefix(None))
    calls = served[2]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert isinstance(calls[0]["id"], str) and calls[0]["id"]
    assert isinstance(calls[0]["function"]["arguments"], str), \
        "function.arguments must be a JSON-ENCODED STRING on the wire"
    # ... and it decodes back to the canonical mapping, unchanged.
    assert json.loads(calls[0]["function"]["arguments"]) == FAILING_CALL["arguments"]
    assert calls[0]["function"]["name"] == "kb_lookup"


def test_the_served_shape_is_accepted_by_the_real_request_model():
    ChatCompletionRequest = _request_model()
    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    served = chat.served_messages(_episode_prefix(schemas))
    req = ChatCompletionRequest(model="m", messages=served, tools=schemas,
                                temperature=0.0, seed=1, max_tokens=8)
    assert req.messages[2]["tool_calls"][0]["id"] == served[2]["tool_calls"][0]["id"]


def test_a_tool_result_without_a_tool_call_id_is_still_accepted():
    """No second wall behind the first: the tool turn needs no `tool_call_id`.

    `served_messages` deliberately does not invent one (the chat template never
    reads it). This is the assertion that would catch an engine that starts
    requiring it, and it names the one function that would have to change.
    """
    ChatCompletionRequest = _request_model()
    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    served = chat.served_messages(_episode_prefix(schemas))
    assert "tool_call_id" not in served[3]
    ChatCompletionRequest(model="m", messages=served, tools=schemas)


def test_served_messages_does_not_mutate_the_canonical_transcript():
    """The recorded transcript is the canonical one; the wire form is a copy.

    Mutating in place would put a JSON string into the transcript the trace, the
    digests and the parity assertion are all defined over -- and the local chat
    template would then raise on `arguments|items`.
    """
    messages = _episode_prefix(None)
    before = copy.deepcopy(messages)
    chat.served_messages(messages)
    assert messages == before
    assert isinstance(messages[2]["tool_calls"][0]["function"]["arguments"], dict)


# ---------------------------------------------------------------------------
# 3. the id is DETERMINISTIC: no uuid, no clock
# ---------------------------------------------------------------------------

def test_the_tool_call_id_is_a_pure_digest_of_ordinal_name_and_arguments():
    """Recomputed here from the documented inputs, so the id cannot drift into a

    uuid or a timestamp without this failing.
    """
    import hashlib

    payload = json.dumps({"key": "KD6FJTKOG2CBCSFIV"}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    expect = "call-" + hashlib.sha256(
        f"served-tool-call-v1|0|kb_lookup|{payload}".encode()).hexdigest()[:24]
    assert chat.tool_call_id(0, "kb_lookup", FAILING_CALL["arguments"]) == expect


def test_two_conversions_of_the_same_transcript_post_byte_identical_bytes():
    """A replay must reproduce the request byte for byte.

    Byte equality is what lets the rendered-prefix parity assertion -- and any
    offline reconstruction of a served run -- mean anything at all.
    """
    messages = _episode_prefix(None)
    first = json.dumps(chat.served_messages(messages), sort_keys=True)
    second = json.dumps(chat.served_messages(copy.deepcopy(messages)),
                        sort_keys=True)
    assert first == second


def test_the_id_separates_calls_by_ordinal_and_by_arguments():
    """Distinct within one request, stable across the decisions of an episode."""
    a = {"name": "kb_lookup", "arguments": {"key": "AAAA"}}
    b = {"name": "kb_lookup", "arguments": {"key": "BBBB"}}
    two = [{"role": "assistant", "content": "u"},
           evaluate.assistant_message("x", [a]),
           {"role": "tool", "name": "kb_lookup", "content": "{}"},
           evaluate.assistant_message("y", [a]),      # the SAME call again
           {"role": "tool", "name": "kb_lookup", "content": "{}"},
           evaluate.assistant_message("z", [b])]
    ids = [c["id"] for m in chat.served_messages(two)
           for c in (m.get("tool_calls") or [])]
    assert len(set(ids)) == 3, ids
    # PREFIX STABILITY: a later decision re-posts the earlier calls with the
    # same ids it already gave them, exactly as a real client would.
    prefix_ids = [c["id"] for m in chat.served_messages(two[:4])
                  for c in (m.get("tool_calls") or [])]
    assert prefix_ids == ids[:2]


def test_an_argument_string_is_not_double_encoded():
    already = {"role": "assistant",
               "tool_calls": [{"type": "function",
                               "function": {"name": "kb_lookup",
                                            "arguments": '{"key":"K"}'}}]}
    out = chat.served_messages([already])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"key":"K"}'


# ---------------------------------------------------------------------------
# 4. probe 3's own path: a real multi-decision episode over the HTTP backend
# ---------------------------------------------------------------------------

class _ValidatingTransport:
    """A fake `requests.post` that validates every payload it is handed.

    Payload validation uses vLLM's request model -- the same code that returned
    HTTP 400 -- and the reply is the OpenAI response shape a served engine really
    answers with (`arguments` as a JSON string), so the evaluator's own response
    parser is exercised too.
    """

    def __init__(self, script, request_model):
        self.script = list(script)
        self.request_model = request_model
        self.payloads: list[dict] = []
        self.errors: list[str] = []

    def __call__(self, url, json=None, timeout=None):  # noqa: A002
        import json as jsonlib
        payload = jsonlib.loads(jsonlib.dumps(json))    # what really goes on the wire
        self.payloads.append(payload)
        try:
            self.request_model(**payload)
        except Exception as exc:                        # the HTTP 400 the server sent
            self.errors.append(str(exc))
            return _Resp(400, text=str(exc)[:400])
        step = self.script.pop(0) if self.script else {"content": "", "tool_calls": []}
        message = {"role": "assistant", "content": step["content"]}
        if step["tool_calls"]:
            message["tool_calls"] = [
                {"id": f"chatcmpl-tool-{i}", "type": "function",
                 "function": {"name": c["name"],
                              "arguments": jsonlib.dumps(c["arguments"])}}
                for i, c in enumerate(step["tool_calls"])]
        return _Resp(200, body={"choices": [{"message": message}]})


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text or json.dumps(body)

    def json(self):
        return self._body


def _oracle_script(bundle):
    """One call per decision along the oracle path, then the committed answer."""
    script = [{"content": f"calling {n.tool}",
               "tool_calls": [{"name": n.tool, "arguments": dict(n.args)}]}
              for n in bundle.nodes]
    script.append({"content": f"Done.\nANSWER: \\boxed{{{bundle.spec.answer}}}",
                   "tool_calls": []})
    return script


def test_a_full_episode_over_the_http_backend_posts_only_valid_requests():
    """PROBE 3, with no GPU. The second decision is where it used to die."""
    import requests

    request_model = _request_model()
    bundle = build_task(SUITE, SEED, "distill", "lookup_chain", 4, 3, None)
    spec_row = certification_spec(bundle)
    transport = _ValidatingTransport(_oracle_script(bundle), request_model)

    original = requests.post
    requests.post = transport
    try:
        chat_fn = evaluate.make_http_chat("http://127.0.0.1:1", "m", DECODE)
        trace = evaluate.run_episode(
            spec_row, arm="B0", condition="clean", control="none", secret=SECRET,
            fault_seed=1, system_prompt="You can call tools.",
            prompt_meta={"path": "-", "sha256": "-"}, chat_fn=chat_fn,
            decode=DECODE, run_meta={"run_id": "served-shape"})
    finally:
        requests.post = original

    assert transport.errors == [], transport.errors[:1]
    # It really did echo tool calls back: more than one decision, and at least
    # one posted payload carries an assistant tool_calls message.
    assert len(transport.payloads) >= 2, transport.payloads
    echoed = [p for p in transport.payloads
              if any(m.get("role") == "assistant" and m.get("tool_calls")
                     for m in p["messages"])]
    assert echoed, "no payload echoed a tool call: the regression is not covered"
    for payload in echoed:
        for msg in payload["messages"]:
            for call in msg.get("tool_calls") or []:
                assert call.get("id")
                assert isinstance(call["function"]["arguments"], str)
    assert trace["runner"]["termination_reason"] == "answered", trace["runner"]
    assert trace["score"]["certified_success"] is True, trace["score"]
    # The RECORDED transcript stayed canonical: mappings, no ids.
    for msg in trace["messages"]:
        for call in msg.get("tool_calls") or []:
            assert isinstance(call["function"]["arguments"], dict)
            assert "id" not in call


def test_an_unencodable_argument_object_is_a_transport_refusal_not_a_scored_row():
    """Nothing is silently coerced, and the guard is not weakened.

    A stringified argument object would be posted as different bytes from the ones
    the offline path renders -- a new silent drift. So the encoder raises, and the
    backend converts that into the SAME fail-closed refusal a dead engine gets:
    no request, no row, resume re-runs the id. Scoring it as `parser_budget` would
    charge a harness bug to the model.
    """
    import requests

    request_model = _request_model()
    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    transport = _ValidatingTransport([], request_model)
    original = requests.post
    requests.post = transport
    try:
        chat_fn = evaluate.make_http_chat("http://127.0.0.1:1", "m", DECODE)
        bad = _episode_prefix(schemas)
        bad[2]["tool_calls"][0]["function"]["arguments"] = {"key": {1, 2}}
        with pytest.raises(evaluate.TransportFailure) as exc:
            chat_fn(bad, schemas)
    finally:
        requests.post = original
    assert exc.value.kind == "unservable_request"
    assert transport.payloads == [], "nothing may be posted for a shape we cannot encode"


# ---------------------------------------------------------------------------
# 5. parity: the served bytes and the canonical bytes render identically
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    if os.environ.get("AGENTIC_SKIP_TOKENIZER"):
        pytest.skip("AGENTIC_SKIP_TOKENIZER is set")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL)


def _render(tok, messages, schemas) -> list[int]:
    text = tok.apply_chat_template(messages, tools=schemas, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def test_the_server_renders_the_served_form_to_the_canonical_token_ids(tokenizer):
    """The decisive parity assertion for the transport repair.

    The served payload is put through vLLM's OWN `_postprocess_messages` -- the
    function the server calls before applying the chat template -- and the result
    must tokenize to exactly what the offline/canonical path renders. If the two
    differed, the arguments serialisation would be a NEW model-visible drift
    between training and served evaluation, which is the defect D2 closed.
    """
    pytest.importorskip("vllm", reason="the render check needs vLLM's postprocess")
    from vllm.entrypoints.chat_utils import _postprocess_messages

    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    canonical = _episode_prefix(schemas)
    served = copy.deepcopy(chat.served_messages(canonical))
    _postprocess_messages(served)
    assert _render(tokenizer, served, schemas) == \
        _render(tokenizer, canonical, schemas)


def test_the_wire_form_must_not_be_rendered_locally(tokenizer):
    """TEETH for the test above: the string form is NOT template-renderable.

    `tool_call.arguments|items` raises on a string, which is precisely why the
    canonical transcript keeps the mapping and the conversion happens at the HTTP
    boundary only. If this ever stopped raising, "the canonical shape must not
    change to satisfy the wire" would need re-deciding, not silently relaxing.
    """
    schemas = rt_mod.tool_schemas_for_family("lookup_chain")
    served = chat.served_messages(_episode_prefix(schemas))
    with pytest.raises(Exception):
        _render(tokenizer, served, schemas)


def test_every_family_and_a_faulted_episode_serve_and_render_identically(tokenizer):
    """Not one hand-built prefix: real transcripts from all three families,

    clean and faulted (the faulted one carries the recovery-token argument that
    only appears in remediation calls).
    """
    pytest.importorskip("vllm", reason="the render check needs vLLM's postprocess")
    from vllm.entrypoints.chat_utils import _postprocess_messages

    cases = [("lookup_chain", 2, None), ("typed_relay", 4, None),
             ("fulfillment", 4, None),
             ("lookup_chain", 4, [("transient", False)])]
    for family, horizon, entries in cases:
        bundle = build_task(SUITE, SEED, "distill", family, horizon, 3, entries)
        schemas = rt_mod.tool_schemas_for_family(family)
        messages = [{"role": "system", "content": "You can call tools."},
                    {"role": "user", "content": bundle.spec.prompt}]
        for node in bundle.nodes:
            args = dict(node.args)
            if entries:
                args["recovery_token"] = "f" * 32
            messages.append(evaluate.assistant_message(
                f"calling {node.tool}", [{"name": node.tool, "arguments": args}]))
            messages.append({"role": "tool", "name": node.tool,
                             "content": '{"ok":true,"value":1}\nreceipt: r-1'})
        served = copy.deepcopy(chat.served_messages(messages))
        _postprocess_messages(served)
        assert _render(tokenizer, served, schemas) == \
            _render(tokenizer, messages, schemas), f"{family}-h{horizon}"
