"""Paired agentic evaluation against a vLLM OpenAI server.

One invocation runs ONE arm over a shard of the committed spec manifest and
appends full JSONL traces; it is resumable (already-traced task IDs are
skipped) and self-limiting (--time-budget-s, default 420 s, keeps every
invocation under the 8-minute ceiling). Loop invocations until the shard
reports complete.

Arms (identical specs, budgets, schemas, decoding, seeds across arms):

  B0  untouched base model, neutral default prompt (descriptive)
  BP  untouched base model, frozen winning prompt   (the elicitation control)
  T0  locked trained checkpoint, neutral prompt     (descriptive)
  TP  locked trained checkpoint, winning prompt     (primary comparison vs BP)
  R0/RP  the GRPO checkpoint arms, only when GRPO ran

Conditions: clean | faulted (one scheduled fault) | stress (two faults,
measured-only). Controls: none | redacted (absent-information; certified
success is zero by construction, any raw success is a harness BUG) | permuted
(counterfactual value permutation; outputs must track the returned value).

Spec contract consumed here. Specs are NOT written by hand or by a second
generator: they are `agentlab.suite.generate.certification_spec(bundle)` over
the committed suite v1 bundles, frozen by hash before any held-out result.

  {"task_id", "family", "split", "horizon", "template_id", "template_hash",
   "prompt", "kb": {key: record}, "env": {...}|None (fulfillment state),
   "oracle": [{"node","tool","args"}], "answer",
   "answer_kind": "token"|"integer", "hidden_key", "fault": {"class",
   "node_index"}?, "faults": [...]?, "pattern_id"?, "all_tools_required"?}

Oracle args may reference earlier results: {"$from": "n2", "field": "next"}.
The suite generator emits fully resolved args instead, which replay identically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import threading
import time

from agentlab import provenance
from agentlab.suite import faults as faults_mod
from agentlab.suite import rng

ARMS = ("B0", "BP", "T0", "TP", "R0", "RP")
CONDITIONS = ("clean", "faulted", "stress")
CONTROLS = ("none", "redacted", "permuted")

# Frozen episode budgets (council spec): decisions H+3 clean / H+5 single-fault
# / H+8 stress; hard tool-call cap 2H+4 everywhere. The arithmetic lives in
# suite.schema so the generator's committed budgets and the evaluator's runtime
# budgets cannot drift apart.
_CONDITION_FAULTS = {"clean": 0, "faulted": 1, "stress": 2}


def budgets_for(horizon: int, condition: str) -> dict:
    from agentlab.suite.schema import call_budget, decision_budget

    return {"max_decisions": decision_budget(horizon, _CONDITION_FAULTS[condition]),
            "max_calls": call_budget(horizon)}


_TOKEN_PROPERTY = {
    "type": "string",
    "description": "Only after a tool error that supplied a recovery_token: "
                   "echo that token here when re-issuing the call."}


def suite_tool_schemas(family: str = "typed_relay") -> list[dict]:
    """The family's canonical tool schemas plus the recovery_token argument.

    The tool surface comes from suite.runtime.tool_schemas_for_family -- the one
    definition the training path and the generator also use -- so an evaluated
    model can never be shown a different tool set from the one the suite was
    built against. The certification layer adds exactly one optional argument,
    `recovery_token`, which the frozen remediation contract requires.
    """
    from agentlab.suite.runtime import tool_schemas_for_family

    out = []
    for schema in tool_schemas_for_family(family):
        schema = json.loads(json.dumps(schema))  # never mutate the shared dicts
        params = schema["function"].setdefault("parameters", {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})["recovery_token"] = dict(_TOKEN_PROPERTY)
        out.append(schema)
    return out


# ---------------------------------------------------------------------------
# episode runtime
# ---------------------------------------------------------------------------

class SpecRuntime:
    """Episode-scoped dispatch: isolated KB view, fault schedule, receipts."""

    def __init__(self, spec: dict, secret: bytes, condition: str, fault_seed: int):
        self.spec = spec
        self.secret = secret
        self.condition = condition
        self.fault_seed = fault_seed
        self.task_id = spec["task_id"]
        self.env = None
        if spec.get("env"):
            from agentlab.suite.envs.fulfillment import WarehouseState

            # One mutable warehouse per episode, isolated from every other
            # episode and from the oracle replay's own throwaway copy.
            self.env = WarehouseState(spec["env"])
        replay = provenance.execute_oracle(spec)
        if not replay["ok"]:
            raise ValueError(f"spec {self.task_id}: oracle replay failed: "
                             f"{replay.get('error')}")
        self.oracle_nodes = replay["nodes"]
        self.canonical_answer = replay["answer"]
        self.injectors: list[faults_mod.FaultInjector] = []
        if condition in ("faulted", "stress"):
            assigned = spec.get("faults") if condition == "stress" else None
            if assigned is None:
                one = spec.get("fault") or faults_mod.schedule_fault(
                    self.task_id, spec.get("oracle", []), fault_seed)
                assigned = [one] if one else []
            for sched in assigned:
                idx = sched.get("node_index")
                if idx is None or idx >= len(self.oracle_nodes):
                    continue
                self.injectors.append(faults_mod.FaultInjector(
                    self.task_id, {"class": sched["class"], "node_index": idx},
                    self.oracle_nodes[idx]["args_digest"], secret))
        self.events: list[dict] = []
        self._call_id = 0

    @property
    def assigned_faults(self) -> list[dict]:
        return [dict(i.schedule, node=self.oracle_nodes[i.schedule["node_index"]]["node"])
                for i in self.injectors]

    def dispatch(self, tool: str, raw_args: dict, decision: int) -> str:
        """Execute one model call; returns the model-visible tool message text."""
        self._call_id += 1
        call_id = self._call_id
        token_provided = faults_mod.parse_token_from_args(raw_args or {})
        args = faults_mod.strip_token(raw_args or {})
        envelope = provenance.canonical_dispatch(self.spec.get("kb", {}), tool, args,
                                                 env=self.env)
        canonical_text = rng.canonical_json(envelope)
        args_digest = provenance.call_digest(tool, args)
        requested_unit = (str(args.get("to_unit", "")).strip().lower()
                          if tool == "unit_convert" else None)

        wrong_text = None
        if tool == "unit_convert" and any(
                i.schedule and i.schedule["class"] == "wrong_unit" for i in self.injectors):
            wrong = faults_mod.wrong_unit_target(requested_unit or "", self.task_id,
                                                 self.fault_seed)
            wrong_env = provenance.canonical_dispatch(
                self.spec.get("kb", {}), tool, dict(args, to_unit=wrong),
                env=self.env)
            wrong_text = rng.canonical_json(wrong_env)

        exposed_text, ev_fields = canonical_text, {"fault_class": None,
                                                   "fault_emitted": False}
        for inj in self.injectors:
            text, ev = inj.filter(call_id=call_id, decision=decision, tool=tool,
                                  args_digest=args_digest,
                                  canonical_text=canonical_text,
                                  requested_unit=requested_unit,
                                  wrong_unit_text=wrong_text)
            if ev.get("fault_emitted") or ev.get("rate_limit_active"):
                exposed_text, ev_fields = text, ev
                break

        exposed_digest = provenance.observation_digest(exposed_text)
        receipt = provenance.mint_receipt(self.secret, self.task_id, call_id,
                                          exposed_digest)
        event = {
            "call_id": call_id, "decision": decision, "tool": tool,
            "args": args, "token_provided": token_provided,
            "args_digest": args_digest,
            "requested_unit": requested_unit,
            "canonical_digest": provenance.observation_digest(canonical_text),
            "exposed_text": exposed_text, "exposed_digest": exposed_digest,
            "receipt": receipt,
        }
        event.update(ev_fields)
        self.events.append(event)
        return f"{exposed_text}\nreceipt: {receipt}"


# ---------------------------------------------------------------------------
# chat backends
# ---------------------------------------------------------------------------

def make_http_chat(server: str, model: str, decode: dict, timeout_s: float = 300.0):
    """OpenAI-compatible chat backend against a vLLM server."""
    import requests

    url = server.rstrip("/") + "/v1/chat/completions"

    def chat_fn(messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": model, "messages": messages, "tools": tools,
                   "temperature": decode["temperature"], "top_p": decode["top_p"],
                   "seed": decode["seed"], "max_tokens": decode["max_tokens"]}
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        content = msg.get("content") or ""
        if not calls and "<tool_call" in content:
            from agentlab.chat import parse_tool_calls
            calls = parse_tool_calls(content)
        return {"content": content, "tool_calls": calls}

    return chat_fn


# ---------------------------------------------------------------------------
# episode driver
# ---------------------------------------------------------------------------

def run_episode(spec: dict, *, arm: str, condition: str, control: str,
                secret: bytes, fault_seed: int, system_prompt: str,
                prompt_meta: dict, chat_fn, decode: dict, run_meta: dict,
                wall_limit_s: float = 240.0) -> dict:
    horizon = int(spec.get("horizon") or 0)
    budgets = budgets_for(horizon, condition)
    t0 = time.monotonic()
    try:
        runtime = SpecRuntime(spec, secret, condition, fault_seed)
    except (ValueError, NotImplementedError) as exc:
        return _trace_row(spec, arm=arm, condition=condition, control=control,
                          budgets=budgets, messages=[], events=[],
                          runner={"n_decisions": 0, "n_calls": 0,
                                  "termination_reason": "spec_error",
                                  "error": str(exc), "wall_s": 0.0},
                          prompt_meta=prompt_meta, decode=decode,
                          run_meta=run_meta, secret=secret, faults=[])

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": spec.get("prompt", "")}]
    schemas = suite_tool_schemas(spec.get("family") or "typed_relay")
    termination = "decision_budget"
    n_calls = 0
    decision = 0
    for decision in range(1, budgets["max_decisions"] + 1):
        if time.monotonic() - t0 > wall_limit_s:
            termination = "wall_clock"
            break
        try:
            out = chat_fn(messages, schemas)
        except Exception as exc:  # server/parse failure stays in denominators
            termination = "parser_budget"
            messages.append({"role": "assistant", "content": f"[harness error: {exc}]"})
            break
        messages.append({"role": "assistant", "content": out.get("content", "")})
        calls = out.get("tool_calls") or []
        if not calls:
            termination = "answered"
            break
        capped = False
        for call in calls:
            if n_calls >= budgets["max_calls"]:
                capped = True
                break
            text = runtime.dispatch(call.get("name", ""), call.get("arguments") or {},
                                    decision)
            n_calls += 1
            messages.append({"role": "tool", "content": text})
        if capped:
            termination = "call_cap"
            break

    runner = {"n_decisions": decision, "n_calls": n_calls,
              "termination_reason": termination,
              "wall_s": round(time.monotonic() - t0, 3)}
    return _trace_row(spec, arm=arm, condition=condition, control=control,
                      budgets=budgets, messages=messages, events=runtime.events,
                      runner=runner, prompt_meta=prompt_meta, decode=decode,
                      run_meta=run_meta, secret=secret,
                      faults=runtime.assigned_faults)


def _trace_row(spec: dict, *, arm: str, condition: str, control: str, budgets: dict,
               messages: list, events: list, runner: dict, prompt_meta: dict,
               decode: dict, run_meta: dict, secret: bytes, faults: list) -> dict:
    trace = {
        "kind": "episode", "schema_version": 1,
        "task_id": spec["task_id"], "family": spec.get("family"),
        "split": spec.get("split"), "horizon": spec.get("horizon"),
        "template_id": spec.get("template_id"),
        "template_hash": spec.get("template_hash"),
        "pattern_id": spec.get("pattern_id"),
        "all_tools_required": bool(spec.get("all_tools_required")),
        "arm": arm, "condition": condition, "control": control,
        "fault": (faults[0] if len(faults) == 1 else None),
        "faults": faults or None,
        "answer": spec.get("answer"), "answer_kind": spec.get("answer_kind", "token"),
        "budgets": budgets, "messages": messages, "events": events,
        "runner": runner,
        "prompt": prompt_meta, "decode": decode,
        "provenance": dict(run_meta,
                           secret_sha256=provenance.observation_digest(secret.hex()),
                           spec_sha256=rng.digest(spec)),
    }
    rep = provenance.certify_episode(trace, secret)
    score = {"raw_success": rep["raw_success"],
             "certified_success": rep["certified_success"],
             "runaway": rep["runaway"]["runaway"],
             "hallucinated": rep["hallucination"]["hallucinated"]}
    if condition in ("faulted", "stress"):
        score["recovery"] = provenance.certify_recovery(trace, secret, rep)
    if trace["all_tools_required"]:
        score["orchestration"] = provenance.certify_orchestration(trace, secret, rep)
    trace["score"] = score
    return trace


# ---------------------------------------------------------------------------
# manifest handling, sharding, resume
# ---------------------------------------------------------------------------

def load_specs(path: str | pathlib.Path) -> list[dict]:
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def apply_control(specs: list[dict], control: str, permutation_seed: int) -> list[dict]:
    if control == "none":
        return specs
    if control == "redacted":
        # A spec whose hidden value cannot be withheld by deleting a KB record
        # (express fulfillment: the value is the finalize completion token) is
        # DROPPED, never passed through unredacted -- an unredacted task in the
        # absent-information arm would score real successes and look like the
        # harness bug the control exists to detect.
        return [provenance.redact_spec(s) for s in specs
                if s.get("redactable", True)]
    if control == "permuted":
        return provenance.permute_hidden_values(specs, permutation_seed)
    raise ValueError(f"unknown control {control!r}")


def done_task_ids(out_path: pathlib.Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "episode":
                done.add(rec.get("task_id"))
    return done


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10,
                              cwd=pathlib.Path(__file__).resolve().parents[3],
                              ).stdout.strip() or None
    except Exception:
        return None


def load_or_create_secret(path: pathlib.Path) -> bytes:
    """Run secret for receipts/tokens. Lives OUTSIDE the committed tree (out/)."""
    if path.exists():
        return bytes.fromhex(path.read_text().strip())
    import os

    secret = os.urandom(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret.hex() + "\n", encoding="utf-8")
    return secret


def run_shard(args, chat_fn=None) -> dict:
    specs = load_specs(args.specs)
    specs = apply_control(specs, args.control, args.permutation_seed)
    specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.{args.condition}.{args.control}.jsonl"
    done = done_task_ids(out_path)
    todo = [s for s in specs if s["task_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    secret = load_or_create_secret(pathlib.Path(args.secret_file))
    prompt_raw = pathlib.Path(args.prompt).read_text(encoding="utf-8")
    system_prompt = prompt_raw.strip()
    # hash the raw committed file, so S16 can compare against the preregistration
    prompt_meta = {"path": str(args.prompt),
                   "sha256": provenance.observation_digest(prompt_raw)}
    decode = {"temperature": 0.0, "top_p": 1.0, "seed": args.decode_seed,
              "max_tokens": args.max_tokens}
    run_meta = {"git_sha": git_sha(), "server_model": args.model,
                "base_id": args.base_id, "adapter": args.adapter,
                "run_id": args.run_id}
    if chat_fn is None:
        chat_fn = make_http_chat(args.server, args.model, decode)

    t0 = time.monotonic()
    lock = threading.Lock()
    written = 0

    def work(spec):
        return run_episode(spec, arm=args.arm, condition=args.condition,
                           control=args.control, secret=secret,
                           fault_seed=args.fault_seed, system_prompt=system_prompt,
                           prompt_meta=prompt_meta, chat_fn=chat_fn, decode=decode,
                           run_meta=run_meta, wall_limit_s=args.episode_wall_s)

    with out_path.open("a", encoding="utf-8") as fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {}
        it = iter(todo)
        exhausted = False
        while pending or not exhausted:
            while (not exhausted and len(pending) < args.concurrency
                   and time.monotonic() - t0 < args.time_budget_s):
                spec = next(it, None)
                if spec is None:
                    exhausted = True
                    break
                fut = pool.submit(work, spec)
                pending[fut] = spec["task_id"]
            if time.monotonic() - t0 >= args.time_budget_s:
                exhausted = True
            if not pending:
                break
            done_futs, _ = concurrent.futures.wait(
                list(pending), timeout=5.0,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done_futs:
                pending.pop(fut)
                trace = fut.result()
                with lock:
                    fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1

    remaining = len(todo) - written
    status = {"arm": args.arm, "condition": args.condition, "control": args.control,
              "shard": args.shard, "num_shards": args.num_shards,
              "written": written, "remaining_in_shard": max(0, remaining),
              "complete": remaining <= 0, "out": str(out_path),
              "elapsed_s": round(time.monotonic() - t0, 1)}
    print(json.dumps(status))
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="served model name on the vLLM server")
    ap.add_argument("--base-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None, help="adapter path (trained arms) or None")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--control", default="none", choices=CONTROLS)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--prompt", required=True, help="system prompt file (frozen, hashed)")
    ap.add_argument("--out", default="results/agentic/traces")
    ap.add_argument("--secret-file", default="out/agentic/run_secret.hex")
    ap.add_argument("--run-id", default="agentic-v1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--time-budget-s", type=float, default=420.0,
                    help="stop launching new episodes after this; resumable")
    ap.add_argument("--episode-wall-s", type=float, default=240.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--decode-seed", type=int, default=0xA61E0009)
    ap.add_argument("--fault-seed", type=int, default=0xA61E0007)
    ap.add_argument("--permutation-seed", type=int, default=0xA61E0008)
    args = ap.parse_args()
    run_shard(args)


if __name__ == "__main__":
    main()
