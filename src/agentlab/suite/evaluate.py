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


def budgets_for(horizon: int, condition: str) -> dict:
    """The registered budgets. ONE definition, in `suite.contract`."""
    return contract.budgets_for(horizon, condition)


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
        except Exception as exc:  # server/parse failure stays in denominators
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
    return {r.get("task_id") for r in existing_rows(out_path)}


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
    # The evaluator READS the ledger and refuses to start work that would cross the
    # ceiling. It does not APPEND: the driver charges the whole server-resident
    # interval -- startup, every client shard and the idle gaps -- exactly once, and
    # a per-shard row would double-charge the same seconds.
    configio.ledger_guard(f"eval:{args.arm}.{args.condition}.{args.control}",
                          float(args.time_budget_s) / 60.0, cfg)
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
    if chat_fn is None:
        chat_fn = make_http_chat(args.server, served_model, decode)

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

    remaining = len(todo) - written
    status = {"arm": args.arm, "condition": args.condition, "control": args.control,
              "shard": args.shard, "num_shards": args.num_shards,
              "written": written, "remaining_in_shard": max(0, remaining),
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
    ap.add_argument("--fault-seed", type=int, default=0xA61E0007)
    ap.add_argument("--permutation-seed", type=int, default=0xA61E0008)
    args = ap.parse_args()
    run_shard(args)


if __name__ == "__main__":
    main()
