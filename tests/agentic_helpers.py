"""Shared fixtures for the agentic evaluation tests.

Builds synthetic task specs matching the frozen spec contract, and scripted
policies that drive `evaluate.run_episode` without any model or server, so
the whole measurement stack (runtime, faults, receipts, certification,
analyzer) is exercised end to end on CPU.
"""

from __future__ import annotations

import json

from agentlab import provenance
from agentlab.suite import evaluate, rng

SECRET = bytes.fromhex("aa" * 32)


# ---------------------------------------------------------------------------
# spec builders
# ---------------------------------------------------------------------------

def chain_spec(i: int, *, horizon: int = 4, split: str = "eval", ns: str = "eval-a",
               n_templates: int = 48) -> dict:
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
    return {
        "task_id": f"{ns}-t{i:05d}", "family": "lookup_chain", "split": split,
        "horizon": horizon, "template_id": f"{ns}-tpl{i % n_templates:03d}",
        "template_hash": f"{ns}-tplhash{i % n_templates:03d}",
        "kb_namespace": ns,
        "prompt": f"Start key: {keys[0]}. Follow the chain via kb_lookup and report "
                  f"the terminal code.",
        "kb": kb, "oracle": oracle, "answer": token, "answer_kind": "token",
        "hidden_key": keys[-1], "answer_field": "code",
        "counterfactual_sensitive": True,
    }


def relay_spec(i: int, *, split: str = "eval", ns: str = "eval-b",
               n_templates: int = 40) -> dict:
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
    return {
        "task_id": f"{ns}-t{i:05d}", "family": "typed_relay", "split": split,
        "horizon": 4, "template_id": f"{ns}-tpl{i % n_templates:03d}",
        "template_hash": f"{ns}-tplhash{i % n_templates:03d}",
        "kb_namespace": ns, "pattern_id": i % 6, "all_tools_required": True,
        "prompt": f"Look up {k1}, follow its next key, convert that record's grams "
                  f"to kg, multiply by its coeff, and report the exact result.",
        "kb": kb, "oracle": oracle, "answer": str(answer), "answer_kind": "integer",
        "hidden_key": k2,
        "counterfactual_sensitive": True,
    }


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
                    return {"content": "giving up\nANSWER: unknown", "tool_calls": []}
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
                    return {"content": "giving up\nANSWER: unknown", "tool_calls": []}
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
        fault_seed: int = 0xA61E0007, base_id: str = "Qwen/Qwen3.5-4B") -> dict:
    return evaluate.run_episode(
        spec, arm=arm, condition=condition, control=control, secret=secret,
        fault_seed=fault_seed, system_prompt="test prompt",
        prompt_meta={"path": "prompts/agentic/p8_combined.txt", "sha256": prompt_sha},
        chat_fn=policy,
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 2786983945,
                "max_tokens": 1024},
        run_meta={"git_sha": "test", "server_model": "m", "base_id": base_id,
                  "adapter": (None if arm in ("B0", "BP") else adapter),
                  "run_id": "test"},
    )
