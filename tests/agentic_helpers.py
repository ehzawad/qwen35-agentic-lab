"""Shared fixtures for the agentic evaluation tests.

Builds synthetic CERTIFICATION specs and scripted policies that drive
`evaluate.run_episode` without any model or server, so the whole measurement
stack (the one canonical runtime, faults, receipts, certification, analyzer) is
exercised end to end on CPU.

A certification spec now has to carry the CANONICAL runtime inputs -- the
serialized `TaskSpec` (`spec_row`) and every `OracleNode.to_row()` with its
`expect` payload and its semantic `match` -- because `evaluate.run_episode` builds
a real `EpisodeRuntime` from them instead of a private `SpecRuntime`. These
fixtures therefore build real `OracleNode`s with real matchers. That is fixture
construction, not a parallel environment: the matcher KINDS are the canonical
ones `suite.runtime._match_node` interprets, the payloads come from
`suite.runtime.canonical_payload`, and nothing here decides what a call means or
whether a fault recovered.
"""

from __future__ import annotations

import json
import pathlib

from agentlab import provenance
from agentlab.suite import contract, evaluate, faults as faults_mod, rng
from agentlab.suite.schema import (FaultSpec, OracleNode, TaskSpec, call_budget,
                                   decision_budget)

SECRET = bytes.fromhex("aa" * 32)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_PREREG = json.loads((_REPO / "configs" / "agentic_preregister.json")
                     .read_text(encoding="utf-8"))
_HW = _PREREG["machine"]["hardware_integrity"]
_ENGINE = _PREREG["machine"]["engine_contract"]

# One synthetic physical card, standing in for the registered single RTX A5000.
A5000_UUID = "GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567"
DRIVER_VERSION = "610.43.02"


def engine_fingerprint(**overrides) -> dict:
    """The engine half of the fingerprint: library versions + the REGISTERED
    engine contract, read from the preregistration rather than retyped.

    S19 compares every engine setting a trace declares against
    `machine.engine_contract`, so a fixture that hardcoded 0.85 or
    `enable_thinking: true` here would be declaring a different apparatus.
    """
    fp = {"vllm": "0.25.1", "torch": "2.10.0", "transformers": "5.1.0"}
    for key, value in _ENGINE.items():
        if key == "note" or key.endswith("_note"):
            continue
        fp[key] = value
    fp.update(overrides)
    return fp


def fingerprint(**overrides) -> dict:
    """The FROZEN cross-agent hardware fingerprint every claim-bearing row carries.

    ---- SEAM (frozen contract, owed by the runtime/evaluator layer) ------------
    `agentlab.suite.evaluate._trace_row` copies the evaluator's `run_meta`
    verbatim into every trace row's `provenance` block, so these fields reach the
    analyzer with no further plumbing: whoever launches an episode owns putting
    them into `run_meta`, and the analyzer only READS them (S19). This fixture
    supplies them the way the production evaluator must, and invents nothing the
    contract does not name -- the field list itself comes from
    `machine.hardware_integrity.required_trace_fields`.
    """
    fp = {
        "run_id": "test",
        "git_sha": "test",
        "config_hash": "cfg-0123456789ab",
        "gpu_name": _HW["expected_gpu_name"],
        "gpu_uuid": A5000_UUID,
        "cuda_visible_bytes": _HW["expected_cuda_visible_bytes"],
        "driver_version": DRIVER_VERSION,
        "engine_fingerprint": engine_fingerprint(),
        "enable_thinking_effective": _HW["enable_thinking_effective_required"],
        "timestamp_utc": "2026-08-05T00:00:00Z",
    }
    fp.update(overrides)
    return fp

# The frozen cross-agent field contract: every scored spec carries
# `template_cluster_id`, a STRUCTURAL cluster identity (family + horizon +
# oracle-DAG shape + tool-order pattern + operand roles), distinct from the
# paraphrase/wording `template_id`. It is the SOLE bootstrap clustering field.
# `N_CLUSTERS_*` below keep <= 5 value instantiations per cluster at the sample
# sizes these fixtures build, matching the registered MT cluster contract.
N_CLUSTERS_CHAIN = 96
N_CLUSTERS_RELAY = 96


# ---------------------------------------------------------------------------
# spec builders
# ---------------------------------------------------------------------------

def chain_spec(i: int, *, horizon: int = 4, split: str = "eval", ns: str = "eval-a",
               n_templates: int = 48, n_clusters: int = N_CLUSTERS_CHAIN,
               fault_class: str | None = None) -> dict:
    """A lookup_chain task: follow `next` keys, report the terminal code."""
    keys = [f"{ns}-K{i:04d}-{j}" for j in range(horizon)]
    token = rng.stream_bytes(0xE0E0, f"tok:{ns}:{i}", 16).hex()
    kb, oracle = {}, []
    for j, k in enumerate(keys):
        if j < horizon - 1:
            kb[k] = {"next": keys[j + 1], "fact": f"distractor-{i}-{j}"}
        else:
            kb[k] = {"code": token}
        args = {"key": k} if j == 0 else {"key": {"$from": f"n{j}", "field": "next"}}
        oracle.append({"node": f"n{j + 1}", "tool": "kb_lookup", "args": args})
    spec = {
        "task_id": f"{ns}-t{i:05d}", "family": "lookup_chain", "split": split,
        "horizon": horizon, "template_id": f"{ns}-tpl{i % n_templates:03d}",
        "template_hash": f"{ns}-tplhash{i % n_templates:03d}",
        "template_cluster_id": f"lookup_chain:H{horizon}:chain:"
                               f"c{i % n_clusters:03d}",
        "kb_namespace": ns,
        "prompt": f"Start key: {keys[0]}. Follow the chain via kb_lookup and report "
                  f"the terminal code.",
        "kb": kb, "oracle": oracle, "answer": token, "answer_kind": "token",
        "hidden_key": keys[-1], "answer_field": "code",
        "counterfactual_sensitive": True,
    }
    _assign_fault(spec, fault_class)
    return _canonicalize(spec)


def _matcher(tool: str, args: dict) -> dict:
    """The canonical matcher for one fixture node.

    Matcher KINDS are exactly the ones `suite.runtime._match_node` interprets
    (`kb`, `convert`, `calc`); nothing here decides semantics, it only states the
    node's own arguments in the canonical matcher shape the generator would have
    committed.
    """
    if tool == "kb_lookup":
        return {"kind": "kb", "key": str(args["key"])}
    if tool == "unit_convert":
        return {"kind": "convert", "value": float(args["value"]),
                "from": str(args["from_unit"]).lower(),
                "to": str(args["to_unit"]).lower()}
    if tool == "calculator":
        from agentlab.suite.runtime import _calc_terms

        ok, result, consts = _calc_terms(args["expression"])
        assert ok, f"fixture calculator expression is invalid: {args['expression']!r}"
        return {"kind": "calc", "result": result, "required": sorted(consts)}
    raise AssertionError(f"fixture has no matcher for tool {tool!r}")


def _canonicalize(spec: dict) -> dict:
    """Add the canonical runtime inputs and the environment-contract stamp.

    The oracle path is replayed once through `provenance.execute_oracle` (the
    shared canonical semantics), so `expect` is never hand-written and the
    fixtures cannot drift from the runtime's idea of a payload. Args are the
    RESOLVED ones, exactly as the production generator commits them.
    """
    replay = provenance.execute_oracle(spec)
    assert replay["ok"], (spec["task_id"], replay.get("error"))
    nodes, faults = [], []
    positions = {}
    for i, got in enumerate(replay["nodes"]):
        node_id = got["node"]
        positions[node_id] = i
        nodes.append(OracleNode(node_id=node_id, tool=got["tool"],
                                args=dict(got["args"]), expect=got["envelope"],
                                match=_matcher(got["tool"], got["args"])))
    assigned = spec.get("faults") or ([spec["fault"]] if spec.get("fault") else [])
    for sched in assigned:
        node = nodes[sched["node_index"]]
        params = dict(sched.get("params") or {})
        if sched["class"] == "wrong_unit" and "wrong_unit" not in params:
            # The trap unit is COMMITTED at generation time, never re-derived at
            # dispatch: that re-derivation was half of the wrong-unit drift.
            params["wrong_unit"] = faults_mod.wrong_unit_candidates(
                str(node.args["to_unit"]))[0]
        if sched["class"] == "rate_limit":
            params.setdefault("retry_after_turns", 1)
        faults.append(FaultSpec(fault_type=sched["class"],
                                target_node=node.node_id, params=params))
        sched["node"] = node.node_id
        sched["params"] = params
    horizon = int(spec["horizon"])
    task = TaskSpec(
        task_id=spec["task_id"], suite="agentlab-suite-v1", split=spec["split"],
        family=spec["family"], horizon=horizon, template_id=0,
        prompt=spec["prompt"], answer=str(spec["answer"]),
        answer_kind=spec["answer_kind"], start={}, env=spec.get("env"),
        faults=faults, max_decisions=decision_budget(horizon, len(faults)),
        max_calls=call_budget(horizon), secret_tokens=[],
        template_cluster_id=str(spec.get("template_cluster_id") or ""))
    spec["spec_row"] = task.to_row()
    spec["oracle_nodes"] = [n.to_row() for n in nodes]
    return contract.stamp(spec)


def _assign_fault(spec: dict, fault_class: str | None) -> None:
    """Pin a spec's fault CLASS so a fixture can hit the registered group sizes.

    ER8 gates all three registered fault groups at their registered cardinality,
    so a fixture that leaves the class to the deterministic draw cannot reach 400
    wrong-unit episodes without thousands of tasks. Pinning the class per spec is
    exactly what the generator does in production (`spec["fault"]`); the node
    index still comes from the frozen scheduler, so the injection point is not
    hand-picked.
    """
    if fault_class is None:
        return
    sched = faults_mod.schedule_fault(spec["task_id"], spec["oracle"],
                                      0xA61E0007, fault_class=fault_class)
    if sched is None:
        raise AssertionError(f"{spec['task_id']}: no eligible node for "
                             f"fault class {fault_class!r}")
    spec["fault"] = sched


def faulted_variant(spec: dict, fault_class: str | None = None) -> dict:
    """The paired FAULTED arm of a spec: exactly one COMMITTED fault.

    A `faulted` condition is satisfied only from committed faults. The evaluator
    no longer invents one when the spec carries none -- `SpecRuntime` used to fall
    back to `schedule_fault` at dispatch time, which meant the evaluated episode
    could carry a fault the generator never committed and the training path could
    therefore never see. Production dev/eval specs each carry one; a fixture that
    wants the faulted arm asks for it here.
    """
    if spec.get("fault") or spec.get("faults"):
        return spec
    new = json.loads(json.dumps(spec))
    new.pop("spec_row", None)
    new.pop("oracle_nodes", None)
    sched = faults_mod.schedule_fault(new["task_id"], new["oracle"], 0xA61E0007,
                                      fault_class=fault_class)
    assert sched is not None, (new["task_id"], fault_class)
    new["fault"] = sched
    return _canonicalize(new)


def relay_spec(i: int, *, split: str = "eval", ns: str = "eval-b",
               n_templates: int = 40, n_clusters: int = N_CLUSTERS_RELAY,
               fault_class: str | None = None) -> dict:
    """A typed_relay task at H4 whose ANSWER causally requires all three tools:
    kb_lookup(K1) -> kb_lookup(K2 from K1) -> unit_convert(grams->kg) ->
    calculator(kg * coeff) = the exact integer answer."""
    grams = 1000 * (2 + i % 7)
    coeff = 3 + i % 5
    k1 = f"{ns}-K{i:04d}"
    k2 = f"{ns}-K{i:04d}-spec"
    answer = (grams // 1000) * coeff
    kb = {k1: {"next": k2, "fact": f"distractor-{i}"},
          k2: {"grams": str(grams), "coeff": str(coeff)}}
    oracle = [
        {"node": "n1", "tool": "kb_lookup", "args": {"key": k1}},
        {"node": "n2", "tool": "kb_lookup",
         "args": {"key": {"$from": "n1", "field": "next"}}},
        {"node": "n3", "tool": "unit_convert",
         "args": {"value": {"$from": "n2", "field": "grams"},
                  "from_unit": "g", "to_unit": "kg"}},
        {"node": "n4", "tool": "calculator",
         "args": {"expression": {"$from": "n3", "format": "{}*" + str(coeff)}}},
    ]
    spec = {
        "task_id": f"{ns}-t{i:05d}", "family": "typed_relay", "split": split,
        "horizon": 4, "template_id": f"{ns}-tpl{i % n_templates:03d}",
        "template_hash": f"{ns}-tplhash{i % n_templates:03d}",
        # structural identity: family + horizon + DAG shape + tool-order pattern
        "template_cluster_id": f"typed_relay:H4:kb-kb-unit-calc:p{i % 6}:"
                               f"c{i % n_clusters:03d}",
        "kb_namespace": ns, "pattern_id": i % 6, "all_tools_required": True,
        "prompt": f"Look up {k1}, follow its next key, convert that record's grams "
                  f"to kg, multiply by its coeff, and report the exact result.",
        "kb": kb, "oracle": oracle, "answer": str(answer), "answer_kind": "integer",
        "hidden_key": k2,
        "counterfactual_sensitive": True,
    }
    _assign_fault(spec, fault_class)
    return _canonicalize(spec)


# ---------------------------------------------------------------------------
# scripted policies (chat_fn implementations)
# ---------------------------------------------------------------------------

def _last_tool_payload(messages):
    for m in reversed(messages):
        if m.get("role") == "tool":
            return m.get("content", "")
        if m.get("role") == "assistant":
            return None
    return None


def _parse_lines(payload):
    out = []
    for line in (payload or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# An agent that gives up commits NOTHING. The frozen protocol defines a
# hallucinated result as "citing an unminted receipt, tool-role content without
# an environment event, or a committed answer value absent from every validated
# observation" (docs/AGENTIC_PROTOCOL.md s3), and the committed-answer reader is
# `ANSWER: <value>` / \boxed{} (schema.extract_committed_answer). So a scripted
# policy that abandons an episode with "ANSWER: unknown" is not modelling an
# honest failure -- it is asserting a value it never observed, which is exactly a
# hallucinated result under the preregistration. Abandonment must therefore end
# the episode with no commitment at all: that is a plain unrecovered failure
# (`no_remediation`), which is what the fault-recovery fixtures mean to exercise.
ABANDON_TEXT = ("giving up: the tool error blocks the required lookup and I have "
                "no validated value to report")


class ScriptedOracle:
    """Follows the oracle path one call per decision; repairs faults properly."""

    def __init__(self, spec: dict, *, blind_retry: bool = False,
                 abandon_on_error: bool = False, final_answer: str | None = None):
        replay = provenance.execute_oracle(spec)
        self.calls = [{"name": n["tool"], "arguments": dict(n["args"])}
                      for n in replay["nodes"]]
        self.answer = (final_answer if final_answer is not None
                       else (replay["answer"] if replay["ok"] else "unknown"))
        self.blind_retry = blind_retry
        self.abandon_on_error = abandon_on_error
        self.idx = 0

    def __call__(self, messages, tools):
        payload = _last_tool_payload(messages)
        if payload is not None and self.idx > 0:
            objs = _parse_lines(payload)
            err = next((o for o in objs if o.get("ok") is False), None)
            prev = self.calls[self.idx - 1]
            if err is not None and err.get("error") != "no_entry":
                if self.abandon_on_error:
                    return {"content": ABANDON_TEXT, "tool_calls": []}
                args = dict(prev["arguments"])
                if not self.blind_retry and err.get("recovery_token"):
                    args["recovery_token"] = err["recovery_token"]
                return {"content": "retrying after error",
                        "tool_calls": [{"name": prev["name"], "arguments": args}]}
            if (prev["name"] == "unit_convert" and objs
                    and objs[0].get("ok") is True
                    and objs[0].get("unit")
                    and objs[0]["unit"] != str(prev["arguments"].get("to_unit", ""))
                    .strip().lower()):
                if self.abandon_on_error:
                    return {"content": ABANDON_TEXT, "tool_calls": []}
                return {"content": "wrong unit returned; correcting",
                        "tool_calls": [{"name": prev["name"],
                                        "arguments": dict(prev["arguments"])}]}
        if self.idx < len(self.calls):
            call = self.calls[self.idx]
            self.idx += 1
            # replay nodes carry RESOLVED args, so the policy can just reuse them
            return {"content": f"calling {call['name']}",
                    "tool_calls": [{"name": call["name"],
                                    "arguments": dict(call["arguments"])}]}
        return {"content": f"done\nANSWER: {self.answer}", "tool_calls": []}


class Guesser:
    """Answers immediately with the given value; makes zero tool calls."""

    def __init__(self, answer: str):
        self.answer = answer

    def __call__(self, messages, tools):
        return {"content": f"I know this one.\nANSWER: {self.answer}", "tool_calls": []}


def make_arm_policy(arm: str, spec: dict, *, recover_pct: dict | None = None):
    """Deterministic per-task behaviour: both arms solve clean episodes; on a
    fault, the arm recovers only for its share of task IDs (hash-assigned)."""
    recover_pct = recover_pct or {"BP": 30, "TP": 85}
    share = recover_pct.get(arm, 0)
    bucket = rng.stream_u64(0xBEEF, f"assign:{spec['task_id']}", 1)[0] % 100
    if bucket < share:
        return ScriptedOracle(spec)
    return ScriptedOracle(spec, abandon_on_error=True)


# ---------------------------------------------------------------------------
# episode runners
# ---------------------------------------------------------------------------

def run(spec: dict, policy, *, arm: str = "TP", condition: str = "clean",
        control: str = "none", prompt_sha: str = "0" * 64,
        adapter: str | None = "out/adapter", secret: bytes = SECRET,
        fault_seed: int = 0xA61E0007, base_id: str = "Qwen/Qwen3.5-4B",
        hardware: dict | None = None) -> dict:
    """Run one scripted episode. `hardware` overrides the S19 fingerprint fields
    (pass e.g. `hardware={"gpu_name": "NVIDIA RTX A6000"}` to build a wrong-card
    trace, or `{"gpu_uuid": None}` to build one with missing provenance)."""
    run_meta = {"git_sha": "test", "server_model": "m", "base_id": base_id,
                "adapter": (None if arm in ("B0", "BP") else adapter)}
    run_meta.update(fingerprint(**(hardware or {})))
    trace = evaluate.run_episode(
        spec, arm=arm, condition=condition, control=control, secret=secret,
        fault_seed=fault_seed, system_prompt="test prompt",
        prompt_meta={"path": "prompts/agentic/p8_combined.txt", "sha256": prompt_sha},
        chat_fn=policy,
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 2786983945,
                "max_tokens": 1024, "enable_thinking": False},
        run_meta=run_meta,
    )
    # SEAM CLOSED: `evaluate._trace_row` now carries `template_cluster_id`
    # itself, so this helper no longer stamps it. The assertion below is the
    # regression guard: if the trace writer ever stops carrying the registered
    # clustering field, every scored fixture fails here rather than quietly
    # falling back to the analyzer's --specs manifest back-fill.
    if spec.get("template_cluster_id") is not None:
        assert trace.get("template_cluster_id") == spec["template_cluster_id"], \
            "evaluate._trace_row must carry the registered clustering field"
    return trace
