"""THE ANTI-FALSE-RECONCILIATION TEST.

The failure mode this file exists for is **wire-format covariate drift under
semantic parity**: both consumers import the same runtime, both pass their own
unit tests, and the model still sees different schemas, tool-message bytes,
receipts, message roles, tool-call objects or rendered tokens. "Everything
imports and the tests pass" is precisely how the tree came to hold two
environments, so semantic equality is not what is asserted here.

For the SAME task bundle, the SAME run secret, the SAME system prompt, the SAME
scripted decisions and the SAME call sequence, the training path and the
claim-bearing evaluation path must agree on:

    environment_contract_sha256
    tool_schema_bytes
    budgets
    model_visible_tool_bytes
    rendered_prefix_token_ids_by_decision      <- the decisive assertion
    progress
    final_state_digest
    verdict

The rendered-prefix token ids are decisive because observation digests alone
would catch a receipt versus an `event_id`, but NOT a dropped assistant
tool-call object, a missing tool `name`, or any other role-level difference. The
real A5000 trace contained an empty assistant message where the offline rollout
contained a structured tool call; digests were blind to it and the tokenizer is
not.

Run over all 12 family/horizon cells, clean plus every eligible fault class plus
the ambiguous malformed mutation.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import pytest

from agentlab.suite import contract as contract_mod
from agentlab.suite import evaluate
from agentlab.suite import runtime as rt_mod
from agentlab.suite.envs import family_module
from agentlab.suite.generate import build_task, certification_spec
from agentlab.suite.schema import CELLS, digest, digest_text

SUITE = "agentlab-suite-v1"
SEED = 0xA61E0002              # the committed distill seed
SECRET = bytes.fromhex("7e" * 32)
SYSTEM_PROMPT = ("You can call tools. Echo a recovery_token when a tool error "
                 "supplies one. Finish with ANSWER: <value>.")
BASE_MODEL = "Qwen/Qwen3.5-4B"


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

def _cases_for(family: str, horizon: int) -> list:
    """Clean, plus every fault class this cell can carry, plus the ambiguous
    malformed mutation where the family supports it."""
    cases = [None, [("transient", False)], [("rate_limit", False)],
             [("malformed", False)]]
    if family_module(family).has_unit_convert(horizon):
        cases.append([("wrong_unit", False)])
    if family == "fulfillment":
        cases.append([("malformed", True)])
    return cases


def _label(entries) -> str:
    if not entries:
        return "clean"
    kind, ambiguous = entries[0]
    return f"{kind}_ambiguous" if ambiguous else kind


CASES = [(family, horizon, entries)
         for i, (family, horizon) in enumerate(CELLS)
         for entries in _cases_for(family, horizon)]


# ---------------------------------------------------------------------------
# one scripted policy, replayed identically on both paths
# ---------------------------------------------------------------------------

class ScriptedDecisions:
    """A fixed decision script: the oracle path with registered remediation.

    The script is computed ONCE against a throwaway runtime and then replayed
    verbatim on both paths, so neither path can influence what the "model" did.
    Both paths therefore face identical decisions and identical calls, and any
    difference that remains is the environment's.
    """

    def __init__(self, bundle, secret: bytes) -> None:
        self.decisions = self._script(bundle, secret)

    @staticmethod
    def _script(bundle, secret: bytes) -> list[dict]:
        from agentlab.suite.faults import TOKEN_ARG

        rt = rt_mod.EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes,
                                   secret=secret)
        decisions: list[dict] = []
        for node in bundle.nodes:
            token = None
            for _ in range(4):
                args = dict(node.args)
                if token is not None:
                    args[TOKEN_ARG] = token
                rt.begin_decision()
                decisions.append({"content": f"calling {node.tool}",
                                  "tool_calls": [{"name": node.tool,
                                                  "arguments": args}]})
                text = rt.dispatch(node.tool, args)
                token = rt_mod.recovery_token_in(text)
                if token is not None:
                    continue
                body = next((o for o in rt_mod.parse_observation(text)["objects"]
                             if isinstance(o, dict)), None)
                if body is None or not body.get("ok"):
                    break
                if (node.tool == "unit_convert"
                        and str(body.get("unit", "")).lower()
                        != str(node.args["to_unit"]).lower()):
                    continue
                break
        decisions.append({"content": f"Done.\nANSWER: \\boxed{{{bundle.spec.answer}}}",
                          "tool_calls": []})
        return decisions

    def as_chat_fn(self):
        """The evaluator's `chat_fn(messages, tools) -> {content, tool_calls}`."""
        state = {"i": 0}

        def chat_fn(messages, tools):
            i = state["i"]
            state["i"] += 1
            if i >= len(self.decisions):
                return {"content": "", "tool_calls": []}
            step = self.decisions[i]
            return {"content": step["content"],
                    "tool_calls": [dict(c) for c in step["tool_calls"]]}

        return chat_fn

    def as_generate_fn(self):
        """The rollout engine's `generate(prompts) -> [(text, finish_reason)]`."""
        state = {"i": 0}

        def generate(prompts):
            out = []
            for _ in prompts:
                i = state["i"]
                state["i"] += 1
                step = (self.decisions[i] if i < len(self.decisions)
                        else {"content": "", "tool_calls": []})
                text = step["content"]
                for call in step["tool_calls"]:
                    # `sort_keys` is deliberately OFF. The evaluator receives the
                    # decision as a dict and the rollout engine receives it as text
                    # it parses, so re-ordering the arguments here would inject a
                    # difference the HARNESS made rather than the environment --
                    # and the rendered template prints arguments in the order the
                    # model gave them. In production both paths parse the same
                    # model output and preserve the same order.
                    payload = json.dumps({"name": call["name"],
                                          "arguments": call["arguments"]})
                    text += f"\n<tool_call>\n{payload}\n</tool_call>"
                out.append((text, "stop"))
            return out

        return generate


# ---------------------------------------------------------------------------
# the two paths, each reporting the SAME parity surface
# ---------------------------------------------------------------------------

def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL)


def _rendered_prefix_token_ids(tok, messages, schemas) -> list[list[int]]:
    """Token ids of the rendered prefix BEFORE each assistant decision.

    This is what the model was actually conditioned on, turn by turn, including
    the chat template's own structure -- the assistant tool-call object, the tool
    message's `name`, the receipt line. A digest over observations cannot see any
    of that.
    """
    out = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        text = tok.apply_chat_template(messages[:i], tools=schemas,
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def _surface(*, messages, schemas, budgets, observations, progress, state_digest,
             verdict, tok) -> dict:
    return {
        "environment_contract_sha256": contract_mod.environment_contract_sha256(),
        "tool_schema_bytes": json.dumps(schemas, sort_keys=True,
                                        separators=(",", ":")),
        "budgets": dict(budgets),
        "model_visible_tool_bytes": list(observations),
        "rendered_prefix_token_ids_by_decision":
            _rendered_prefix_token_ids(tok, messages, schemas),
        "progress": dict(progress),
        "final_state_digest": state_digest,
        "verdict": dict(verdict),
    }


def training_run(bundle, script, tok, *, condition: str) -> dict:
    """The training path: `multidistill.RolloutEngine` on the canonical runtime."""
    from agentlab.multidistill import RolloutEngine
    from agentlab.suite.configio import load_config

    spec = contract_mod.spec_for_condition(bundle.spec, condition)
    bundle = dataclasses.replace(bundle, spec=spec)
    engine = RolloutEngine(load_config(), lambda m, s: m, script.as_generate_fn(),
                           secret=SECRET)
    convos = engine.rollouts_for([bundle], k_override=1, variants=("canonical",))
    for convo in convos:
        convo["messages"][0]["content"] = SYSTEM_PROMPT
    rec = engine.run(convos, verbose=False)[0]
    observations = [c["exposed"] for c in rec["calls"]]
    return _surface(
        messages=rec["messages"],
        schemas=rt_mod.tool_schemas_for_family(spec.family),
        budgets={"max_decisions": spec.max_decisions, "max_calls": spec.max_calls},
        observations=observations, progress=rec["parity"]["progress"],
        state_digest=_state_digest_of(rec["parity"]["episode"], rec),
        verdict=rec["verdict"], tok=tok)


def _state_digest_of(episode_digest: str, rec: dict) -> str:
    """The terminal environment state, isolated from the observation history.

    `episode_digest` mixes observations, progress and state, so it cannot answer
    "did both paths end in the same world". The last event's `state_after` can,
    and it is empty outside fulfillment, so it is digested rather than compared
    raw.
    """
    events = rec.get("events") or []
    return digest_text(events[-1]["state_after"] if events else "")


def evaluation_run(bundle, script, tok, *, condition: str) -> dict:
    """The claim-bearing path: `evaluate.run_episode` on the SAME runtime."""
    spec_row = certification_spec(bundle)
    trace = evaluate.run_episode(
        spec_row, arm="B0", condition=condition, control="none", secret=SECRET,
        fault_seed=1, system_prompt=SYSTEM_PROMPT,
        prompt_meta={"path": "-", "sha256": "-"}, chat_fn=script.as_chat_fn(),
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 256,
                "enable_thinking": False},
        run_meta={"run_id": "parity"})
    assert trace["runner"]["termination_reason"] != "spec_error", trace["runner"]
    observations = [c["exposed"] for c in trace["calls"]]
    events = trace["events"]
    return _surface(
        messages=trace["messages"],
        schemas=rt_mod.tool_schemas_for_family(trace["family"]),
        budgets=trace["budgets"], observations=observations,
        progress=trace["parity"]["progress"],
        state_digest=digest_text(events[-1]["state_after"] if events else ""),
        verdict=trace["verdict"], tok=tok)


# ---------------------------------------------------------------------------
# the assertion
# ---------------------------------------------------------------------------

PARITY_FIELDS = ("environment_contract_sha256", "tool_schema_bytes", "budgets",
                 "model_visible_tool_bytes",
                 "rendered_prefix_token_ids_by_decision", "progress",
                 "final_state_digest", "verdict")


@pytest.fixture(scope="module")
def tokenizer():
    if os.environ.get("AGENTIC_SKIP_TOKENIZER"):
        pytest.skip("AGENTIC_SKIP_TOKENIZER is set")
    try:
        return _tokenizer()
    except Exception as exc:  # no local tokenizer: the decisive assertion cannot run
        pytest.fail(
            f"the Qwen3.5-4B tokenizer is required for the parity test and could "
            f"not be loaded ({exc}). Rendered-prefix token ids are the only "
            f"assertion that catches a dropped assistant tool-call object or a "
            f"missing tool name, so this test does not fall back to digests.")


@pytest.mark.parametrize("family,horizon,entries", CASES,
                         ids=[f"{f}-h{h}-{_label(e)}" for f, h, e in CASES])
def test_training_and_evaluation_face_the_identical_environment(
        family, horizon, entries, tokenizer):
    condition = "clean" if not entries else "faulted"
    bundle = build_task(SUITE, SEED, "distill", family, horizon, 3, entries)
    script = ScriptedDecisions(bundle, SECRET)

    training = training_run(bundle, script, tokenizer, condition=condition)
    evaluation = evaluation_run(bundle, script, tokenizer, condition=condition)

    assert training["environment_contract_sha256"] == \
        evaluation["environment_contract_sha256"]
    assert training["tool_schema_bytes"] == evaluation["tool_schema_bytes"]
    assert training["budgets"] == evaluation["budgets"]
    assert training["model_visible_tool_bytes"] == \
        evaluation["model_visible_tool_bytes"]
    assert training["rendered_prefix_token_ids_by_decision"] == \
        evaluation["rendered_prefix_token_ids_by_decision"]
    assert training["progress"] == evaluation["progress"]
    assert training["final_state_digest"] == evaluation["final_state_digest"]
    assert training["verdict"] == evaluation["verdict"]

    # the case really exercised what it says it does
    if entries:
        assert training["verdict"]["fault_assigned"] == 1
        assert training["verdict"]["faults_triggered"] == 1
        assert training["verdict"]["recovery_reason"] == "ok"
    assert training["verdict"]["certified_success"], training["verdict"]["reasons"]
    assert training["model_visible_tool_bytes"]


def test_the_parity_surface_is_not_vacuous(tokenizer):
    """Each parity field must be able to FAIL, or the test proves nothing.

    Every mutation below is one of the drifts the D2 reconciliation removed, and
    each is applied to a copy of the training surface. If any of them compared
    equal, the corresponding assertion above would be decoration.
    """
    bundle = build_task(SUITE, SEED, "distill", "typed_relay", 4,
                        3, [("transient", False)])
    script = ScriptedDecisions(bundle, SECRET)
    base = training_run(bundle, script, tokenizer, condition="faulted")
    schemas = rt_mod.tool_schemas_for_family("typed_relay")

    # (1) the tokenless tool surface: recovery_token removed from every schema
    tokenless = json.loads(json.dumps(schemas))
    for schema in tokenless:
        schema["function"]["parameters"]["properties"].pop("recovery_token")
    assert json.dumps(tokenless, sort_keys=True, separators=(",", ":")) != \
        base["tool_schema_bytes"]

    # (2) the tokenless observation form: event_id instead of a receipt line
    tokenless_obs = []
    for i, text in enumerate(base["model_visible_tool_bytes"], start=1):
        body = rt_mod.parse_observation(text)["objects"][0]
        tokenless_obs.append(json.dumps({**body, "event_id": f"e{i}"},
                                        sort_keys=True, separators=(",", ":")))
    assert tokenless_obs != base["model_visible_tool_bytes"]

    # (3) THE decisive one: the dropped assistant tool-call object and the
    #     nameless tool result, exactly as the real A5000 trace carried them
    engine_messages = _messages_of(bundle, script, condition="faulted")
    flattened = []
    for msg in engine_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            flattened.append({"role": "assistant",
                              "content": msg.get("content", "")})
        elif msg.get("role") == "tool":
            flattened.append({"role": "tool", "content": msg["content"]})
        else:
            flattened.append(msg)
    drifted = _rendered_prefix_token_ids(tokenizer, flattened, schemas)
    assert drifted != base["rendered_prefix_token_ids_by_decision"], (
        "flattening the assistant tool-call object must change the rendered "
        "tokens, or the decisive assertion is blind to the drift it exists for")

    # (4) the budgets, the progress map, the state digest and the verdict
    assert {"max_decisions": 1, "max_calls": 1} != base["budgets"]
    assert {k: 1 for k in base["progress"]} != base["progress"]
    assert digest_text("tampered") != base["final_state_digest"]
    assert dict(base["verdict"], certified_success=False) != base["verdict"]


def _messages_of(bundle, script, *, condition: str) -> list:
    from agentlab.multidistill import RolloutEngine
    from agentlab.suite.configio import load_config

    spec = contract_mod.spec_for_condition(bundle.spec, condition)
    engine = RolloutEngine(load_config(), lambda m, s: m, script.as_generate_fn(),
                           secret=SECRET)
    convos = engine.rollouts_for([dataclasses.replace(bundle, spec=spec)],
                                 k_override=1, variants=("canonical",))
    for convo in convos:
        convo["messages"][0]["content"] = SYSTEM_PROMPT
    return engine.run(convos, verbose=False)[0]["messages"]


def test_a_stale_environment_contract_is_refused_everywhere():
    """Resume logic can never reuse a tokenless artifact after this fix."""
    stale = {"task_id": "t", contract_mod.STAMP_FIELD: "0" * 64}
    absent = {"task_id": "t"}
    for row in (stale, absent):
        assert not contract_mod.is_current(row)
        with pytest.raises(SystemExit):
            contract_mod.require_current(row, "an artifact")
    current, invalid = contract_mod.invalidate(
        [contract_mod.stamp({"task_id": "ok"}), stale, absent])
    assert len(current) == 1 and len(invalid) == 2


def test_the_certification_spec_carries_the_canonical_runtime_inputs():
    """Reconstructing matchers from the flat oracle is what forked the layer."""
    for family, horizon in CELLS:
        bundle = build_task(SUITE, SEED, "distill", family, horizon, 1, None)
        row = certification_spec(bundle)
        assert contract_mod.is_current(row)
        assert row["spec_row"] == bundle.spec.to_row()
        assert row["oracle_nodes"] == [n.to_row() for n in bundle.nodes]
        for node in row["oracle_nodes"]:
            assert node["expect"] and node["match"]
        stripped = {k: v for k, v in row.items()
                    if k not in ("spec_row", "oracle_nodes")}
        with pytest.raises(ValueError):
            evaluate.episode_runtime(stripped, SECRET, "clean")


def test_the_evaluation_trace_is_self_describing_about_its_environment():
    bundle = build_task(SUITE, SEED, "distill", "lookup_chain", 4, 2,
                        [("malformed", False)])
    script = ScriptedDecisions(bundle, SECRET)
    row = certification_spec(bundle)
    trace = evaluate.run_episode(
        row, arm="TP", condition="faulted", control="none", secret=SECRET,
        fault_seed=1, system_prompt=SYSTEM_PROMPT,
        prompt_meta={"path": "-", "sha256": "-"}, chat_fn=script.as_chat_fn(),
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 256,
                "enable_thinking": False},
        run_meta={"run_id": "parity"})
    assert contract_mod.is_current(trace)
    assert trace["tool_schema_sha256"] == digest_text(
        rt_mod.tool_schema_bytes("lookup_chain"))
    assert trace["verdict"] and trace["parity"] and trace["calls"]
    assert trace["score"]["verdict_agrees"]
    assert digest(trace["parity"]["progress"]) == digest(
        trace["verdict"]["node_decisions"])


def test_resume_can_never_reuse_a_tokenless_artifact(tmp_path, monkeypatch):
    """The invalidation rule reaches the two places resume actually happens.

    Both consumers resume by presence: evaluation deduplicates by task id in an
    existing trace file, and rejection sampling treats a shard file as done. A
    tokenless artifact would therefore be resumed into for ever, and the repair
    would never reach the tasks it already "finished".
    """
    from agentlab.multidistill import shard_is_current

    # (1) evaluation: a stale row makes the file un-appendable, and it is never
    #     counted as done
    out = tmp_path / "BP.clean.none.jsonl"
    fresh = contract_mod.stamp({"kind": "episode", "task_id": "fresh"})
    stale = {"kind": "episode", "task_id": "stale",
             contract_mod.STAMP_FIELD: "0" * 64}
    tokenless = {"kind": "episode", "task_id": "tokenless"}
    out.write_text("".join(json.dumps(r) + "\n" for r in (fresh, stale, tokenless)),
                   encoding="utf-8")
    assert evaluate.done_task_ids(out) == {"fresh"}
    with pytest.raises(SystemExit) as exc:
        evaluate.refuse_stale_environment_rows(out)
    assert contract_mod.STAMP_FIELD in str(exc.value)
    # a file whose every row is current is appendable and fully counted
    out.write_text(json.dumps(fresh) + "\n", encoding="utf-8")
    evaluate.refuse_stale_environment_rows(out)
    assert evaluate.done_task_ids(out) == {"fresh"}

    # (2) rejection sampling: a stale shard is NOT done, so it is re-rolled
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr("agentlab.multidistill.RAW_DIR", raw)
    (raw / "shard-0000.jsonl").write_text(json.dumps(tokenless) + "\n",
                                          encoding="utf-8")
    (raw / "shard-0001.jsonl").write_text(json.dumps(fresh) + "\n", encoding="utf-8")
    assert shard_is_current(0) is False
    assert shard_is_current(1) is True
    assert shard_is_current(2) is False          # absent


def test_stale_rows_are_dropped_from_the_accepted_corpus_and_the_views(tmp_path):
    """Acceptance and view construction count the invalidated rows out loud."""
    from agentlab.multidistill import finalize
    from agentlab.suite.datasets import build_views

    tokenless = {"kind": "rollout", "task_id": "t", "family": "lookup_chain",
                 "horizon": 2, "fault_types": [], "messages": [], "verdict": {},
                 "events": [], "calls": [], "sample_index": 0}
    kept, summary = finalize([tokenless], {}, __import__(
        "agentlab.suite.configio", fromlist=["x"]).load_config())
    assert kept == []
    assert summary["stale_environment_contract"] == 1
    assert summary[contract_mod.STAMP_FIELD] == \
        contract_mod.environment_contract_sha256()

    rows, meta, report = build_views([tokenless], lambda p, c, t: 1)
    assert rows == [] and meta == []
    assert report["rejected"]["stale_environment_contract"] == 1
    assert report[contract_mod.STAMP_FIELD] == \
        contract_mod.environment_contract_sha256()
