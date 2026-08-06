"""Paired agentic evaluation against a vLLM OpenAI server.

One invocation runs ONE arm over a shard (an immutable task BLOCK) of the
committed spec manifest and appends full JSONL traces; it is resumable
(already-traced task IDs are skipped) and self-limiting. Loop invocations until
the shard reports complete.

--time-budget-s IS A LAUNCH WINDOW, NOT A KILL SWITCH. It stops launching new
episodes; every episode already in flight is drained and written. A budget that
killed an active episode would discard GPU seconds the ledger has already
charged, and would leave the shard's remaining count lying about what ran.

TRANSPORT DEATH IS NOT AN EPISODE OUTCOME. A dead, unreachable or wrong-model
server aborts the shard nonzero and writes NO row for the affected episodes; it
is never committed as a scored `parser_budget` failure. See `TransportFailure`
-- that conversion was the single most dangerous defect in this file, because it
would have entered infrastructure failures into the arm's denominators and then
marked them done for resume.

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

This is a pure HTTP client: it opens no CUDA context, reads no nvidia-smi and
never assumes the registered card. The process that produced the tokens is the
authority on the hardware that produced them, so `--runtime-manifest` is
MANDATORY and is verified -- whole, current, and against this run's physical
binding -- before the trace file is opened and before the first request. A run
that cannot attest its producer writes nothing at all.

ONE RUNTIME. This module used to carry `SpecRuntime`, a second episode runtime
with its own tool schemas, its own error envelopes, its own receipt suffix, its
own transcript shape, its own recomputed wrong unit and its own recovery
predicate. It is gone. Every episode here dispatches through
`agentlab.suite.runtime.EpisodeRuntime` -- the same class rejection sampling, the
prompt tournament, the variance probe and generation validation use -- and the
canonical verdict it produces is written straight into the trace.

Spec contract consumed here. Specs are NOT written by hand or by a second
generator: they are `agentlab.suite.generate.certification_spec(bundle)` over
the committed suite v1 bundles, frozen by hash before any held-out result. The
fields this runner needs are the CANONICAL runtime inputs:

  {"task_id", ..., "prompt", "kb": {key: record},
   "spec_row": serialized TaskSpec, "oracle_nodes": [OracleNode.to_row(), ...],
   "environment_contract_sha256": which model-visible environment this describes}

The flat `oracle` list is retained for the S9 reachability replay and the
controls; it carries neither the canonical payloads nor the semantic matchers, so
it is never the source a runtime is built from.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import threading
import time

from agentlab import provenance
from agentlab.suite import configio, contract
from agentlab.suite import rng

ARMS = ("B0", "BP", "T0", "TP", "R0", "RP")
CONDITIONS = ("clean", "faulted", "stress")
CONTROLS = ("none", "redacted", "permuted")

# The registered control seeds, named once. The permutation seed decides WHICH
# specs survive `apply_control` (permutation is only defined for eligible terminal
# lookups), so a census validator that used a different one would compute a
# different expected cardinality from the run it is checking.
DEFAULT_PERMUTATION_SEED = 0xA61E0008
DEFAULT_FAULT_SEED = 0xA61E0007


def budgets_for(horizon: int, condition: str) -> dict:
    """The registered budgets. ONE definition, in `suite.contract`."""
    return contract.budgets_for(horizon, condition)


# ---------------------------------------------------------------------------
# transport integrity: the engine dying is not something the MODEL did
# ---------------------------------------------------------------------------

class TransportFailure(RuntimeError):
    """The server/transport failed. This is NEVER an episode outcome.

    THE DANGEROUS SEAM THIS CLOSES. Every exception out of the chat backend used
    to be caught inside the episode loop and committed as
    `termination_reason: "parser_budget"` -- a SCORED row, in the denominator,
    counted against the arm, and (worse) marked done for resume. So if the vLLM
    server died mid-shard, the shard did not stop: it kept going, wrote hundreds
    of `parser_budget` rows attributed to the policy, exited 0, and a later resume
    SKIPPED every one of those task ids because their ids were already present.
    An unreachable engine would have been reported as a model that stopped
    committing answers -- and F5 (loop/crash < 0.02) would have failed the arm for
    the harness's own infrastructure.

    The two classes are now distinguished at the source:

      TransportFailure   the server is unreachable, dead, returned a non-2xx
                         status, or answered something that is not an
                         OpenAI-shaped completion. INFRASTRUCTURE. It aborts the
                         shard loudly and writes NO row for the affected episode,
                         so a resumed shard re-runs exactly those task ids.
      anything else      a genuine client-side PARSE failure over a well-formed
                         HTTP response. That is model-visible behaviour, it keeps
                         its `parser_budget` termination, and it stays in the
                         denominators exactly as registered.

    `kind` records which signal fired, so the abort message can tell an operator
    "the server is gone" apart from "the server answered 500".
    """

    def __init__(self, message: str, *, kind: str, status: int | None = None,
                 task_id: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.task_id = task_id

    def to_row(self) -> dict:
        return {"kind": self.kind, "status": self.status,
                "task_id": self.task_id, "detail": str(self)}


# ---------------------------------------------------------------------------
# building the canonical runtime from a committed certification spec
# ---------------------------------------------------------------------------

def episode_runtime(spec: dict, secret: bytes, condition: str):
    """-> `suite.runtime.EpisodeRuntime` for one certification spec.

    The spec must carry the canonical runtime inputs (`spec_row`,
    `oracle_nodes`). Reconstructing matchers or expected payloads from the flat
    `oracle` list is forbidden: that reconstruction is what forked the
    environment layer, so a spec that lacks them is refused rather than
    approximated.

    The absent-information control needs no special case any more. Redaction makes
    the required lookup return `no_entry`, the canonical runtime exposes exactly
    that, the node is never credited, and the policy really runs -- which is what
    makes S11 a real leak detector. `SpecRuntime` used to replay the oracle first
    and abort the episode as a `spec_error` unless the control was `redacted`; the
    exception existed only because the abort did.
    """
    from agentlab.suite.runtime import EpisodeRuntime
    from agentlab.suite.schema import OracleNode, TaskSpec

    if not spec.get("spec_row") or not spec.get("oracle_nodes"):
        raise ValueError(
            f"spec {spec.get('task_id')!r} carries no canonical runtime inputs "
            f"(spec_row / oracle_nodes). Regenerate the certification specs: an "
            f"evaluator that rebuilt matchers and expected payloads from the flat "
            f"oracle would be a second environment implementation.")
    contract.require_current(spec, f"certification spec {spec.get('task_id')!r}")
    task = contract.spec_for_condition(TaskSpec.from_row(spec["spec_row"]), condition)
    nodes = [OracleNode.from_row(n) for n in spec["oracle_nodes"]]
    return EpisodeRuntime(task, spec.get("kb", {}), nodes, secret=secret)


def assigned_faults(runtime) -> list[dict]:
    """The faults this episode really scheduled, in the trace's frozen shape."""
    positions = {n.node_id: i for i, n in enumerate(runtime.nodes)}
    return [{"class": f.fault_type, "node_index": positions.get(f.target_node),
             "node": f.target_node, "params": dict(f.params)}
            for f in runtime.spec.faults]


# ---------------------------------------------------------------------------
# chat backends
# ---------------------------------------------------------------------------

def make_http_chat(server: str, model: str, decode: dict, timeout_s: float = 300.0):
    """OpenAI-compatible chat backend against a vLLM server.

    `chat_template_kwargs` is not decoration. This checkpoint defaults thinking
    ON, and the server renders the chat template, so an evaluation request that
    does not say `enable_thinking: false` runs a DIFFERENT policy from the offline
    rejection sampler (which renders with thinking disabled), spends the
    completion budget on reasoning tokens, and reads as "the model never
    committed an answer" -- a failure mode this repo has already been burned by.
    The server is started with the same default; the request states it as well,
    because a per-request field cannot be forgotten by a restarted server.
    """
    import requests

    from agentlab.suite import configio

    url = server.rstrip("/") + "/v1/chat/completions"

    def chat_fn(messages: list[dict], tools: list[dict]) -> dict:
        configio.reject_multimodal(messages)
        payload = {"model": model, "messages": messages, "tools": tools,
                   "temperature": decode["temperature"], "top_p": decode["top_p"],
                   "seed": decode["seed"], "max_tokens": decode["max_tokens"],
                   "chat_template_kwargs": {
                       "enable_thinking": bool(decode["enable_thinking"])}}
        # Every failure below is the ENGINE's, not the policy's, so each one is
        # raised as TransportFailure and none of them can become a scored row.
        # `raise_for_status` used to be the only check and its HTTPError was
        # indistinguishable, one frame up, from a parse failure.
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
        except Exception as exc:  # ConnectionError, Timeout, ChunkedEncoding...
            raise TransportFailure(
                f"the vLLM server at {server} did not answer "
                f"({type(exc).__name__}: {exc})",
                kind="unreachable") from exc
        if resp.status_code != 200:
            raise TransportFailure(
                f"the vLLM server at {server} answered HTTP "
                f"{resp.status_code}: {resp.text[:400]!r}",
                kind="http_status", status=resp.status_code)
        try:
            body = resp.json()
        except Exception as exc:
            raise TransportFailure(
                f"the vLLM server at {server} answered 200 with a body that is "
                f"not JSON ({type(exc).__name__}): {resp.text[:400]!r}",
                kind="malformed_response") from exc
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportFailure(
                f"the vLLM server at {server} answered 200 with no "
                f"choices[0].message ({type(exc).__name__}): "
                f"{json.dumps(body)[:400]}",
                kind="malformed_response") from exc
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


def probe_server(server: str, model: str | None = None,
                 timeout_s: float = 30.0) -> dict:
    """Is an engine actually serving `model` right now? Never raises.

    A pure read of /v1/models. It is the cheapest possible way to tell "the
    server is gone" from "the model stopped answering", and it runs BEFORE the
    first episode and again at an abort, so the operator is told which one
    happened rather than left to infer it from a wall of `parser_budget` rows.
    """
    import requests

    url = server.rstrip("/") + "/v1/models"
    try:
        resp = requests.get(url, timeout=timeout_s)
    except Exception as exc:
        return {"ok": False, "kind": "unreachable", "served": [],
                "reason": f"{url} did not answer ({type(exc).__name__}: {exc})"}
    if resp.status_code != 200:
        return {"ok": False, "kind": "http_status", "served": [],
                "reason": f"{url} answered HTTP {resp.status_code}"}
    try:
        served = [str(m.get("id")) for m in (resp.json().get("data") or [])]
    except Exception as exc:
        return {"ok": False, "kind": "malformed_response", "served": [],
                "reason": f"{url} answered 200 with an unreadable body ({exc})"}
    if model is not None and served and model not in served:
        return {"ok": False, "kind": "model_absent", "served": served,
                "reason": (f"{url} is up but serves {served!r}; this shard "
                           f"requests {model!r}")}
    return {"ok": True, "kind": "live", "served": served, "reason": "live"}


def require_live_server(server: str, model: str, what: str) -> dict:
    """No live engine, no episodes. Refuses BEFORE the trace file is opened.

    A dead server used to be discovered one episode at a time, and each discovery
    wrote a scored failure. Discovering it once, before anything is written, is
    the whole difference between an aborted shard and a corrupted arm.
    """
    probe = probe_server(server, model)
    if not probe["ok"]:
        raise SystemExit(
            f"REFUSED: no live engine for {what}.\n"
            f"  {probe['reason']}\n"
            f"  This is INFRASTRUCTURE, not model behaviour: an unreachable or "
            f"wrongly-loaded server may not produce a single scored episode. "
            f"Start the server through scripts/serve.sh (the chain does this) and "
            f"re-run the shard -- resume re-runs exactly the task ids that have "
            f"no row.")
    return probe


# ---------------------------------------------------------------------------
# episode driver
# ---------------------------------------------------------------------------

def assistant_message(content: str, calls: list) -> dict:
    """The ONE assistant-message shape: prose plus the structured tool calls.

    `RolloutEngine._step` and this evaluator now build the assistant turn the same
    way, through `agentlab.chat.assistant_tool_message`. The evaluator used to
    append `{"role": "assistant", "content": out["content"]}` and DROP the
    tool-call object entirely, and its tool results carried no `name`. The model
    conditions on both: a transcript with an empty assistant turn followed by a
    nameless tool result renders to different tokens from one carrying a tool-call
    object and a named result, so training and evaluation were showing the policy
    two different conversations. The parity test asserts the rendered token ids,
    which is the only assertion that catches this.
    """
    from agentlab.chat import assistant_tool_message

    return assistant_tool_message(content, calls)


def run_episode(spec: dict, *, arm: str, condition: str, control: str,
                secret: bytes, fault_seed: int, system_prompt: str,
                prompt_meta: dict, chat_fn, decode: dict, run_meta: dict,
                wall_limit_s: float = 240.0) -> dict:
    from agentlab.suite.runtime import tool_schemas_for_family

    horizon = int(spec.get("horizon") or 0)
    budgets = budgets_for(horizon, condition)
    t0 = time.monotonic()
    try:
        runtime = episode_runtime(spec, secret, condition)
    except (ValueError, NotImplementedError, SystemExit) as exc:
        return _trace_row(spec, arm=arm, condition=condition, control=control,
                          budgets=budgets, messages=[], events=[], calls=[],
                          runner={"n_decisions": 0, "n_calls": 0,
                                  "termination_reason": "spec_error",
                                  "error": str(exc), "wall_s": 0.0},
                          prompt_meta=prompt_meta, decode=decode,
                          run_meta=run_meta, secret=secret, faults=[],
                          verdict=None)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": spec.get("prompt", "")}]
    schemas = tool_schemas_for_family(spec.get("family") or "typed_relay")
    recorded: list[dict] = []
    termination = "decision_budget"
    n_calls = 0
    decision = 0
    final_text = ""
    for decision in range(1, budgets["max_decisions"] + 1):
        if time.monotonic() - t0 > wall_limit_s:
            termination = "wall_clock"
            break
        try:
            out = chat_fn(messages, schemas)
        except TransportFailure as exc:
            # THE ENGINE, NOT THE POLICY. No row is written for this episode at
            # all: it is not a `parser_budget` failure, it is not in any
            # denominator, and resume must re-run this exact task id. run_shard
            # turns this into a loud shard abort.
            exc.task_id = spec.get("task_id")
            raise
        except Exception as exc:  # a real PARSE failure stays in denominators
            termination = "parser_budget"
            messages.append({"role": "assistant", "content": f"[harness error: {exc}]"})
            break
        calls = out.get("tool_calls") or []
        content = out.get("content", "") or ""
        # The logical clock advances once per assistant decision, in the runtime,
        # exactly as it does in the training path -- the rate-limit contract and
        # the "dependency edges need a LATER decision" rule both key off it.
        runtime.begin_decision()
        if not calls:
            termination = "answered"
            final_text = content
            messages.append({"role": "assistant", "content": content})
            break
        messages.append(assistant_message(content, calls))
        capped = False
        for call in calls:
            if n_calls >= budgets["max_calls"]:
                capped = True
                break
            name = call.get("name", "")
            args = dict(call.get("arguments") or {})
            text = runtime.dispatch(name, args)
            n_calls += 1
            recorded.append({"call_id": runtime.events[-1].call_id,
                             "decision_id": runtime.decision_id,
                             "tool": name, "args": args, "exposed": text})
            messages.append({"role": "tool", "name": name, "content": text})
        if capped:
            termination = "call_cap"
            break

    runner = {"n_decisions": decision, "n_calls": n_calls,
              "termination_reason": termination,
              "wall_s": round(time.monotonic() - t0, 3)}
    verdict = runtime.verify(final_text, transcript=messages,
                            termination_reason=termination)
    return _trace_row(spec, arm=arm, condition=condition, control=control,
                      budgets=budgets, messages=messages,
                      events=[e.to_row() for e in runtime.events],
                      calls=recorded, runner=runner, prompt_meta=prompt_meta,
                      decode=decode, run_meta=run_meta, secret=secret,
                      faults=assigned_faults(runtime), verdict=verdict.to_row(),
                      parity={"observations": runtime.observation_digests(),
                              "progress": runtime.progress(),
                              "episode": runtime.episode_digest()})


def _trace_row(spec: dict, *, arm: str, condition: str, control: str, budgets: dict,
               messages: list, events: list, calls: list, runner: dict,
               prompt_meta: dict, decode: dict, run_meta: dict, secret: bytes,
               faults: list, verdict: dict | None,
               parity: dict | None = None) -> dict:
    from agentlab.suite.runtime import tool_schema_bytes

    family = spec.get("family")
    trace = {
        "kind": "episode", "schema_version": 1,
        "task_id": spec["task_id"], "family": spec.get("family"),
        "split": spec.get("split"), "horizon": spec.get("horizon"),
        "template_id": spec.get("template_id"),
        "template_hash": spec.get("template_hash"),
        # FROZEN SEAM (clustering): `template_cluster_id` is the STRUCTURAL
        # cluster the preregistered bootstrap resamples, and the paraphrase
        # `template_id` is explicitly NOT it. It is carried natively here rather
        # than back-filled by the analyzer from the --specs manifest, so a trace
        # set is self-describing: an episode whose cluster is unknown makes its
        # gate INCONCLUSIVE instead of silently clustering on something else.
        "template_cluster_id": spec.get("template_cluster_id"),
        "pattern_id": spec.get("pattern_id"),
        "all_tools_required": bool(spec.get("all_tools_required")),
        "arm": arm, "condition": condition, "control": control,
        "fault": (faults[0] if len(faults) == 1 else None),
        "faults": faults or None,
        "answer": spec.get("answer"), "answer_kind": spec.get("answer_kind", "token"),
        "budgets": budgets, "messages": messages, "events": events,
        # The recorded call sequence, so the analyzer can replay this episode
        # through the canonical runtime (S17) and the orchestration dataflow can
        # read real argument values rather than digests.
        "calls": calls,
        "runner": runner,
        # The canonical verifier's verdict, written straight into the trace. There
        # is one certified-success predicate and this is its output; the
        # certification layer below cross-checks the ledger-side conditions
        # against it rather than defining a second, weaker one.
        "verdict": verdict,
        "parity": parity,
        # Which model-visible environment produced these bytes (D2). A trace
        # without the current stamp is never resumed into or pooled with one that
        # has it.
        contract.STAMP_FIELD: contract.environment_contract_sha256(),
        "tool_schema_sha256": (
            None if not family
            else provenance.observation_digest(tool_schema_bytes(family))),
        "prompt": prompt_meta, "decode": decode,
        # FROZEN SEAM (S19): run_meta is copied VERBATIM into every trace's
        # `provenance`, so the hardware/engine fingerprint is owned here in the
        # runtime layer and merely read by the analyzer. `timestamp_utc` is
        # stamped per row rather than per shard. In a claim run run_meta is COPIED
        # from the producer's verified runtime manifest; the refusal to SERIALIZE
        # an incomplete one lives at the write in run_shard, so this stays a pure
        # function that an offline single-episode replay can still call.
        "provenance": dict(run_meta,
                           timestamp_utc=configio.now_utc(),
                           secret_sha256=provenance.observation_digest(secret.hex()),
                           spec_sha256=rng.digest(spec)),
    }
    rep = provenance.certify_episode(trace, secret, verdict)
    score = {"raw_success": rep["raw_success"],
             "certified_success": rep["certified_success"],
             "verdict_agrees": rep["verdict_agrees"],
             "runaway": rep["runaway"]["runaway"],
             "hallucinated": rep["hallucination"]["hallucinated"]}
    if condition in ("faulted", "stress"):
        score["recovery"] = provenance.certify_recovery(trace, secret, rep, verdict)
    if trace["all_tools_required"]:
        score["orchestration"] = provenance.certify_orchestration(
            trace, secret, rep, verdict=verdict)
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


def existing_rows(out_path: pathlib.Path) -> list[dict]:
    rows = []
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
                rows.append(rec)
    return rows


def done_task_ids(out_path: pathlib.Path) -> set:
    """Task ids already traced UNDER THIS ENVIRONMENT CONTRACT.

    Resume used to deduplicate by task id alone, so a trace produced under the
    retired tokenless contract would be treated as done for ever and the task
    would never be re-evaluated under the contract the gates describe. A row
    without the current stamp is not "done" -- `refuse_stale_environment_rows`
    below refuses to APPEND to such a file at all, so this is the second line of
    the same rule rather than a silent re-run.
    """
    return {r.get("task_id") for r in existing_rows(out_path)
            if contract.is_current(r)}


def unit_census(units, certspecs_dir, traces_dir) -> list[dict]:
    """Expected versus WRITTEN episodes for each scheduled evaluation unit.

    COMPLETION IS A CENSUS, NOT A FILE. Several registered manifests share one
    `{arm}.{condition}.{control}.jsonl` trace file -- core, MT and H8 are all
    clean/none -- so the existence of that file says nothing about whether a
    manifest ran at all. MT and H8 were preregistered and never invoked once
    already, and a shared file is exactly what hid it.

    Attribution is by TASK ID, because the manifests are disjoint sets of task ids
    and a row names its own task. The expected set is the CONTROLLED set: the
    redacted control drops specs whose hidden value cannot be withheld, so the raw
    manifest size would be the wrong denominator.

    `units` is (arm, manifest_name, specs_file, condition, control, mandatory).
    """
    certspecs_dir = pathlib.Path(certspecs_dir)
    traces_dir = pathlib.Path(traces_dir)
    rows = []
    for arm, name, specs, condition, control, mandatory in units:
        path = traces_dir / f"{arm}.{condition}.{control}.jsonl"
        row = {"arm": arm, "manifest": name, "specs": specs,
               "condition": condition, "control": control,
               "mandatory": bool(mandatory), "trace": str(path),
               "expected": 0, "written": 0, "missing": 0,
               "dropped_by_control": 0, "status": "complete"}
        manifest = certspecs_dir / specs
        if not manifest.exists():
            row.update(status="no_manifest")
            rows.append(row)
            continue
        raw = load_specs(manifest)
        wanted = apply_control(raw, control, DEFAULT_PERMUTATION_SEED)
        want_ids = {s["task_id"] for s in wanted}
        have = {r.get("task_id") for r in existing_rows(path)
                if contract.is_current(r)}
        missing = want_ids - have
        row.update(expected=len(want_ids), written=len(want_ids) - len(missing),
                   missing=len(missing),
                   # How many committed specs the CONTROL removed. Zero for the
                   # ordinary manifests and for the absent-information control
                   # (a committed redacted row is redactable by construction). A
                   # non-zero count on a registered control means the control's
                   # registered cardinality is not reachable from the committed
                   # specs -- reported, because it is the suite generator's
                   # contract and not something an evaluator may quietly absorb.
                   dropped_by_control=max(0, len(raw) - len(want_ids)),
                   status="complete" if not missing else "short")
        rows.append(row)
    return rows


def format_census(rows: list[dict]) -> str:
    out = []
    for r in rows:
        drop = (f"  ({r['dropped_by_control']} dropped by the control)"
                if r["dropped_by_control"] else "")
        out.append(f"    {r['arm']:<3} {r['manifest']:<7} {r['condition']:<8} "
                   f"{r['control']:<9} {r['written']:>5}/{r['expected']:<5} "
                   f"{'MANDATORY' if r['mandatory'] else 'optional':<9} "
                   f"{r['status']}{drop}")
    return "\n".join(out)


def require_mandatory_census(rows: list[dict]) -> list[dict]:
    """No partial MANDATORY census may pass as the registered evaluation.

    Optional units (cut ranks 3 and 4) may legitimately be absent or incomplete --
    they are budget-conditional by registration. A short mandatory unit is a
    different thing entirely: mandatory samples may never shrink, so the run
    reports INCOMPLETE / INCONCLUSIVE rather than presenting what it has.
    """
    bad = [r for r in rows if r["mandatory"] and r["status"] != "complete"]
    if bad:
        lines = "\n".join(
            f"    {r['arm']} {r['manifest']} {r['condition']}/{r['control']}: "
            f"{r['status']} ({r['written']}/{r['expected']} written)" for r in bad)
        raise SystemExit(
            "REFUSED: the MANDATORY census is incomplete, so this is not the "
            "registered evaluation:\n" + lines + "\n"
            "  Mandatory samples may never shrink and a partial mandatory census "
            "may never be reported as the registered one. Re-invoke the eval stage "
            "(it resumes exactly the missing task ids); if it cannot finish inside "
            "the ceiling, report INCOMPLETE / INCONCLUSIVE.")
    return [r for r in rows if r["status"] != "complete"]


def refuse_stale_environment_rows(out_path: pathlib.Path) -> None:
    """A run may not APPEND to a trace file produced under another environment.

    Mixing an episode the model faced with recovery tokens, remediation text and
    receipts together with one it faced without them puts TWO environments inside
    one claim -- the D2 defect, reintroduced through resume. The old trace set is
    evidence of the defect and keeps its own run id; a repaired run starts a new
    directory.
    """
    for i, row in enumerate(existing_rows(out_path)):
        if not contract.is_current(row):
            raise SystemExit(
                f"REFUSED: {out_path} row {i} carries "
                f"{'no' if not row.get(contract.STAMP_FIELD) else 'a stale'} "
                f"{contract.STAMP_FIELD}, so it was produced under a different "
                f"model-visible environment (recovery tokens, remediation text, "
                f"receipts, transcript shape).\n"
                f"  This build's contract is "
                f"{contract.environment_contract_sha256()}.\n"
                f"  Move that trace set aside under its own run id -- it is "
                f"evidence of the retired contract -- and start a NEW directory. "
                f"Appending would put two environments inside one claim, which is "
                f"exactly the defect the unified contract closed.")


def require_same_fingerprint(out_path: pathlib.Path, fingerprint: dict) -> int:
    """Refuse to APPEND to a trace file that another card or engine produced.

    Resume used to deduplicate by task ID alone, which means a shard restarted
    under a different card, driver, engine setting or effective thinking mode
    would happily append to the same file -- and the resulting trace set would
    carry two hardware fingerprints inside one claim. S19 calls that a BUG; the
    cheapest place to stop it is before the first append, not at analysis time.

    Returns the number of existing rows checked.
    """
    rows = existing_rows(out_path)
    for i, row in enumerate(rows):
        prior = row.get("provenance") or {}
        if not any(prior.get(k) is not None for k in
                   configio.FINGERPRINT_IDENTITY_FIELDS):
            continue  # a pre-fingerprint row: nothing to compare against
        conflict = configio.fingerprint_conflict(prior, fingerprint)
        if conflict:
            raise SystemExit(
                f"REFUSED: {out_path} row {i} was produced under a different "
                f"runtime fingerprint ({', '.join(conflict)}).\n"
                f"  existing: "
                f"{json.dumps(configio.fingerprint_identity(prior), sort_keys=True)}\n"
                f"  current:  "
                f"{json.dumps(configio.fingerprint_identity(fingerprint), sort_keys=True)}\n"
                f"  Appending would put two hardware/engine fingerprints inside "
                f"one claim, which is exactly what S19 exists to catch. A "
                f"replication on another card is legitimate science but needs a "
                f"NEW run_id and its own trace set -- never an append.")
    return len(rows)


def refuse_null_hardware_rows(out_path: pathlib.Path) -> None:
    """A claim run may not APPEND to a trace file that carries null hardware.

    The D1 traces exist: 12 rows with `gpu_uuid: null`, written because the
    server was started outside the attested path. Those rows are evidence of the
    defect and must be QUARANTINED under their own run id, never backfilled from
    a later manifest and never extended -- a file with one attributed and one
    unattributed row supports no same-card claim at all.
    """
    for i, row in enumerate(existing_rows(out_path)):
        gaps = configio.fingerprint_gaps(row.get("provenance") or {})
        if gaps:
            raise SystemExit(
                f"REFUSED: {out_path} row {i} carries incomplete hardware "
                f"provenance ({', '.join(gaps)}), so it was produced by an "
                f"unattested engine.\n"
                f"  Move that trace set aside under its own run id (it is "
                f"evidence of the defect) and start a NEW directory. Appending "
                f"attested rows to it would neither repair the old rows nor "
                f"produce a trace set S19 can pass.")


def require_producer_attestation(args, cfg: dict) -> dict:
    """The fail-closed D1 gate: no producer manifest, no episode. No exceptions.

    Runs BEFORE the trace file is opened, before the run secret is created and
    before a single HTTP request, because the point of failing closed is that a
    refused run leaves nothing behind that looks like a result.

    The evaluator never probes a card and never synthesizes hardware from the
    registered A5000 expectation: the server process that produced the tokens is
    the authority, and this reads its attestation.
    """
    path = getattr(args, "runtime_manifest", None)
    if not path:
        raise SystemExit(
            "REFUSED: --runtime-manifest is required. This evaluator is a pure "
            "HTTP client: it opens no CUDA context, so it cannot know which card "
            "answered it. The vLLM launcher (scripts/serve.sh) captures a "
            "producer manifest and the driver countersigns it once /v1/models "
            "answers; without it a trace row could only claim `gpu_uuid: null`, "
            "which S19 reads as INCONCLUSIVE. Start the server through the "
            "supported chain, or pass the manifest that server wrote.")
    return configio.require_runtime_manifest(
        path, run_id=args.run_id, cfg=cfg, stage="serve", server=args.server,
        model=args.model, adapter=args.adapter,
        served_adapter_name=(args.served_adapter_name if args.adapter else None))


def git_sha() -> str | None:
    """One definition, in configio, so the ledger and the traces cannot disagree."""
    return configio.git_sha()


def load_or_create_secret(path: pathlib.Path) -> bytes:
    """The run secret, now owned by `suite.contract` and shared by every consumer.

    It used to live here, which meant only the evaluator had one -- and therefore
    only the evaluator could mint recovery tokens at all. The prompt tournament,
    rejection sampling and view construction now read the same file.
    """
    return contract.load_or_create_secret(path)


def run_shard(args, chat_fn=None, cfg: dict | None = None) -> dict:
    cfg = cfg or configio.load_config()
    contract = configio.engine_contract(cfg)
    dec_cfg = cfg.get("eval_decoding") or {}

    # A trained arm must be served the ADAPTER, not the base weights. The server
    # registers the LoRA under an alias (`trained` by default) and the request has
    # to ask for that alias by name: sending the base model id while passing
    # --adapter for provenance only would evaluate the base model in the T0/TP
    # arms and report it as the trained policy.
    served_model = args.model
    if args.adapter:
        served_model = args.served_adapter_name
        if not served_model:
            raise SystemExit(
                "REFUSED: --adapter was given but no served adapter alias. A "
                "trained arm that requests the base model id evaluates the BASE "
                "policy and labels it trained.")

    # ---- FAIL CLOSED, before any file is opened and before any HTTP ----------
    manifest = require_producer_attestation(args, cfg)
    fingerprint = configio.fingerprint_from_manifest(manifest, cfg)
    out_dir = pathlib.Path(args.out)
    out_path = out_dir / f"{args.arm}.{args.condition}.{args.control}.jsonl"
    refuse_null_hardware_rows(out_path)
    checked = require_same_fingerprint(out_path, fingerprint)
    # After the hardware refusals, because an unattributed or foreign-card file is
    # the more upstream problem and its message is the one that tells the operator
    # which directory to quarantine.
    refuse_stale_environment_rows(out_path)
    # The evaluator READS the ledger and refuses to start work that would cross the
    # ceiling. It does not charge ITSELF: the driver charges the whole
    # server-resident interval -- startup, every client shard and the idle gaps --
    # exactly once, and a per-shard row would double-charge the same seconds. (The
    # guard may still charge a DEAD predecessor's abandoned session journal; that is
    # time already spent by another process, and the alternative is losing it.)
    #
    # The stage name is `eval:<unit>` on purpose: `ledger_guard` resolves the
    # COMPLETE REMAINING STAGE from the committed budget projection under the
    # `eval` key, so this call is no longer a check on one 360-second launch
    # window pretending to be a budget check.
    configio.ledger_guard(f"eval:{args.arm}.{args.condition}.{args.control}",
                          float(args.time_budget_s) / 60.0, cfg,
                          run_id=args.run_id)
    # -------------------------------------------------------------------------

    specs = load_specs(args.specs)
    specs = apply_control(specs, args.control, args.permutation_seed)
    specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]

    out_dir.mkdir(parents=True, exist_ok=True)
    secret = load_or_create_secret(pathlib.Path(args.secret_file))
    prompt_raw = pathlib.Path(args.prompt).read_text(encoding="utf-8")
    system_prompt = prompt_raw.strip()
    # hash the raw committed file, so S16 can compare against the preregistration
    prompt_meta = {"path": str(args.prompt),
                   "sha256": provenance.observation_digest(prompt_raw)}
    decode = {"temperature": float(dec_cfg.get("temperature", 0.0)),
              "top_p": float(dec_cfg.get("top_p", 1.0)),
              "seed": args.decode_seed,
              "max_tokens": args.max_tokens,
              # The contract is the authority; the request states it explicitly
              # and the trace records what was actually sent.
              "enable_thinking": bool(contract["enable_thinking"])}

    done = done_task_ids(out_path)
    todo = [s for s in specs if s["task_id"] not in done]
    # --limit is a SMOKE-TEST knob (the shipping smoke runs eight dev episodes), and
    # it must not be able to say "complete". `remaining` was computed against the
    # LIMITED list, so a limited invocation reported complete:true with the rest of
    # the shard untouched -- and the driver's loop would have believed it.
    pending_in_shard = len(todo)
    if args.limit:
        todo = todo[: args.limit]

    # Every extra field here is the PRODUCER's, copied from its manifest: the
    # session that produced the tokens, the digest of the attestation that says
    # so, its pid, and the exact adapter tree it had loaded. `client_git_sha` is
    # this process's own commit and is deliberately separate from the producer's
    # `git_sha` -- a commit between server start and evaluation is legitimate and
    # S19 does not pair on it, but the two must remain distinguishable.
    run_meta = dict(fingerprint,
                    started_at_utc=fingerprint["timestamp_utc"],
                    server_model=served_model, requested_model=args.model,
                    base_id=args.base_id, adapter=args.adapter,
                    resumed_rows=checked,
                    runtime_manifest_sha256=manifest[configio.MANIFEST_HASH_FIELD],
                    session_id=manifest["session_id"],
                    producer_pid=manifest["pid"],
                    producer_stage=manifest["stage"],
                    adapter_sha256=manifest.get("adapter_sha256"),
                    client_git_sha=configio.git_sha())
    unit = f"{args.arm}/{args.condition}/{args.control} shard {args.shard}"
    if chat_fn is None:
        # A dead engine is discovered ONCE, here, before the trace file is opened
        # -- not one scored `parser_budget` row at a time.
        require_live_server(args.server, served_model, unit)
        chat_fn = make_http_chat(args.server, served_model, decode)

    t0 = time.monotonic()
    lock = threading.Lock()
    written = 0
    # Infrastructure failures collected during this shard. The FIRST one stops
    # new launches; the shard then drains what is already in flight and aborts
    # nonzero without writing a row for any affected episode.
    transport: list[TransportFailure] = []

    def work(spec):
        return run_episode(spec, arm=args.arm, condition=args.condition,
                           control=args.control, secret=secret,
                           fault_seed=args.fault_seed, system_prompt=system_prompt,
                           prompt_meta=prompt_meta, chat_fn=chat_fn, decode=decode,
                           run_meta=run_meta, wall_limit_s=args.episode_wall_s)

    # Whether this invocation is the one that CREATES the trace file decides
    # whether it may remove it again: an aborted or empty invocation must leave
    # nothing behind that looks like "this arm ran and produced no successes".
    created_trace_file = not out_path.exists()
    with out_path.open("a", encoding="utf-8") as fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {}
        it = iter(todo)
        exhausted = False
        while pending or not exhausted:
            # --time-budget-s is a LAUNCH WINDOW. It closes the gate on new
            # episodes and never touches one that is already running: everything
            # in flight is drained below and written. A budget that killed an
            # active shard would throw away GPU seconds the ledger has charged.
            while (not exhausted and not transport
                   and len(pending) < args.concurrency
                   and time.monotonic() - t0 < args.time_budget_s):
                spec = next(it, None)
                if spec is None:
                    exhausted = True
                    break
                fut = pool.submit(work, spec)
                pending[fut] = spec["task_id"]
            if transport or time.monotonic() - t0 >= args.time_budget_s:
                exhausted = True
            if not pending:
                break
            done_futs, _ = concurrent.futures.wait(
                list(pending), timeout=5.0,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done_futs:
                task_id = pending.pop(fut)
                try:
                    trace = fut.result()
                except TransportFailure as exc:
                    # NO ROW. Not a scored failure, not in a denominator, not
                    # marked done for resume. The shard stops launching and
                    # aborts below.
                    exc.task_id = exc.task_id or task_id
                    transport.append(exc)
                    continue
                # The last line of defence, at the write itself: a row whose
                # provenance is not whole is not serialized at all. The gate above
                # already refused an unattested run, so reaching this is a bug --
                # but "refuse to write" is the property, and it belongs where the
                # bytes are produced.
                configio.require_complete_fingerprint(
                    trace["provenance"], f"the trace row for {trace['task_id']}")
                with lock:
                    fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1

    if created_trace_file and out_path.exists() and not out_path.stat().st_size:
        out_path.unlink()          # an empty trace file is not evidence of an arm

    if transport:
        health = probe_server(args.server, served_model)
        # The infrastructure failure is EVIDENCE and is recorded -- but never as an
        # episode. It goes to a `.transport.log` (not `.jsonl`), because every
        # consumer of this directory globs `*.jsonl`, and an infrastructure record
        # inside the trace corpus is exactly the confusion this whole change
        # removes.
        log_path = out_dir / f"{args.arm}.{args.condition}.{args.control}.transport.log"
        with log_path.open("a", encoding="utf-8") as lf:
            for exc in transport:
                lf.write(json.dumps(
                    dict(exc.to_row(), kind_of_record="transport_failure",
                         at_utc=configio.now_utc(), shard=args.shard,
                         num_shards=args.num_shards, server=args.server,
                         served_model=served_model,
                         session_id=manifest["session_id"],
                         server_health_at_abort=health["kind"]),
                    ensure_ascii=False) + "\n")
        raise SystemExit(
            f"ABORTED (infrastructure): {unit} hit "
            f"{len(transport)} transport failure(s) against "
            f"{args.server}.\n"
            f"  first: [{transport[0].kind}] {transport[0]}\n"
            f"  server now: {health['kind']} -- {health['reason']}\n"
            f"  affected task ids (NO row written for any of them): "
            f"{', '.join(sorted(t.task_id or '?' for t in transport)[:12])}"
            f"{' ...' if len(transport) > 12 else ''}\n"
            f"  {written} episode(s) completed against a healthy engine and were "
            f"written to {out_path}; they keep their rows.\n"
            f"  the infrastructure failures are recorded as evidence in "
            f"{log_path} -- NOT as episodes\n"
            f"  A dead or unreachable engine is NOT model behaviour. It is never "
            f"committed as a `parser_budget` episode, so nothing here is in any "
            f"denominator and resume will re-run exactly the affected ids. Fix "
            f"the server (see the serve log), confirm the card is still bound to "
            f"this run, and re-invoke the shard.")

    # The remainder of the SHARD, not of the limited slice: `complete` means the
    # block this invocation was given has no untraced task left.
    remaining = pending_in_shard - written
    status = {"arm": args.arm, "condition": args.condition, "control": args.control,
              "shard": args.shard, "num_shards": args.num_shards,
              "written": written, "remaining_in_shard": max(0, remaining),
              "limit": int(args.limit or 0),
              "complete": remaining <= 0, "out": str(out_path),
              "served_model": served_model,
              "enable_thinking_effective": decode["enable_thinking"],
              "gpu_uuid": fingerprint["gpu_uuid"],
              "session_id": manifest["session_id"],
              "runtime_manifest_sha256": manifest[configio.MANIFEST_HASH_FIELD],
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
    ap.add_argument("--served-adapter-name", default="trained",
                    help="LoRA alias the server registered for --adapter; the "
                         "trained arms REQUEST this name, so they cannot silently "
                         "evaluate the base weights")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--control", default="none", choices=CONTROLS)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--prompt", required=True, help="system prompt file (frozen, hashed)")
    ap.add_argument("--out", default="results/agentic/traces")
    ap.add_argument("--secret-file", default="out/agentic/run_secret.hex")
    ap.add_argument("--run-id", default="agentic-v1")
    # REQUIRED, with no off switch. The manifest is written by the process that
    # owns the card (scripts/serve.sh captures it and then execs vLLM) and
    # countersigned by the driver once /v1/models answers. Without it this client
    # cannot say which card produced its episodes, and a trace row that says
    # `gpu_uuid: null` is not a claim -- so the run refuses instead.
    ap.add_argument("--runtime-manifest", required=True,
                    help="the producer session manifest the serving process "
                         "wrote (results/agentic/manifests/serve.*.json)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    # Concurrency, the decode seed and the per-decision token cap all default to
    # the registered values in configs/multifaceted.yaml `eval_decoding:`, so the
    # CLI cannot quietly disagree with the preregistration. Concurrency is the
    # engine contract's max_num_seqs: pushing more concurrent episodes than the
    # server will schedule buys nothing and costs DeltaNet recurrent state
    # (~49.1 MiB per active sequence).
    _dec = configio.load_config().get("eval_decoding") or {}
    ap.add_argument("--concurrency", type=int,
                    default=int(_dec.get("concurrency", 8)))
    ap.add_argument("--time-budget-s", type=float, default=420.0,
                    help="stop launching new episodes after this; resumable")
    ap.add_argument("--episode-wall-s", type=float, default=240.0)
    ap.add_argument("--max-tokens", type=int,
                    default=int(_dec.get("max_tokens_per_decision", 1024)))
    ap.add_argument("--decode-seed", type=int,
                    default=int(_dec.get("seed", 0xA61E0009)))
    ap.add_argument("--fault-seed", type=int, default=DEFAULT_FAULT_SEED)
    ap.add_argument("--permutation-seed", type=int,
                    default=DEFAULT_PERMUTATION_SEED)
    args = ap.parse_args()
    run_shard(args)


if __name__ == "__main__":
    main()
