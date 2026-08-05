"""Unforgeable environment receipts and certification of agentic outcomes.

Every tool observation an episode exposes to the model is minted an opaque
receipt: HMAC-SHA256(run_secret, task_id | call_id | sha256(observation)).
The model never sees the secret, so it cannot fabricate a receipt that
validates; the secret is deterministic per run, so resumed shards reproduce
identical episodes. Receipts make three certifications mechanical:

  certify_episode        strict certified success: exact answer + valid
                         receipt chain + no hallucinated result + no runaway
  certify_recovery       the six non-recovery cases are excluded by
                         construction (not_exposed, unvalidated_answer,
                         blind_retry, pre_fault_answer, hallucinated, runaway)
  certify_orchestration  kb_lookup, unit_convert and calculator all lie on the
                         causal dataflow into the final answer, with
                         dependency edges crossing later assistant decisions

`verify_oracle` independently replays a task's oracle path (S9): every scored
task must be reachable and its registered horizon true, without any model.

This module is CPU-only, stdlib + the repo's own tools; it is imported by both
the evaluation runner (to write scores) and the analyzer (to recompute them
from raw messages/events -- S17 compares the two).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re

from agentlab.suite import faults as faults_mod
from agentlab.suite import rng

RECEIPT_PREFIX = "r-"
_RECEIPT_RE = re.compile(r"\br-[0-9a-f]{32}\b")
_ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\s`*]+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

RUNAWAY_TERMINATIONS = {"token_budget", "wall_clock", "parser_budget"}


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

def observation_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mint_receipt(secret: bytes, task_id: str, call_id: int, exposed_digest: str) -> str:
    mac = hmac.new(secret, f"{task_id}|{call_id}|{exposed_digest}".encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return RECEIPT_PREFIX + mac[:32]


def receipt_valid(secret: bytes, task_id: str, event: dict) -> bool:
    text = event.get("exposed_text", "")
    if observation_digest(text) != event.get("exposed_digest"):
        return False
    want = mint_receipt(secret, task_id, int(event.get("call_id", -1)),
                        event["exposed_digest"])
    got = event.get("receipt", "")
    return hmac.compare_digest(want, got)


# ---------------------------------------------------------------------------
# canonical tool semantics (shared by oracle replay and the episode runtime)
# ---------------------------------------------------------------------------

def canonical_dispatch(kb: dict, tool: str, args: dict, env=None) -> dict:
    """The true result envelope for one call against an episode KB.

    Delegates to `suite.runtime.canonical_payload`, the single implementation of
    canonical tool semantics: the certification layer must never compute an
    observation differently from the episode runtime the training path uses, or
    "certified" and "verified" would mean different things.

    kb_lookup misses return only no_entry -- never a key list (the global
    kb_lookup's key-leaking miss message must not reach suite episodes).
    `env` is a fulfillment WarehouseState when the spec carries one.
    """
    from agentlab.suite import runtime as runtime_mod

    known = runtime_mod.CANONICAL_TOOLS + (
        runtime_mod.WAREHOUSE_TOOLS if env is not None else ())
    if tool not in known:
        return {"ok": False, "error": f"unknown tool {tool!r}"}
    payload, _meta = runtime_mod.canonical_payload(tool, args, kb=kb, env=env)
    return payload


def _resolve_args(args: dict, prior: dict) -> dict:
    """Resolve {"$from": "nK", "field": ..., "format": ...} against prior results.

    "field" defaults to the envelope's "value"; any other field reads the KB
    record. "format" (e.g. "{}*4") splices the referenced value into a string,
    which is how typed-relay calculator expressions consume earlier results.
    """
    out = {}
    for k, v in args.items():
        if isinstance(v, dict) and "$from" in v:
            src = prior.get(v["$from"])
            if src is None:
                raise KeyError(f"oracle reference to unexecuted node {v['$from']!r}")
            field = v.get("field", "value")
            if field == "value":
                val = src.get("value")
            else:
                rec = src.get("record", {})
                val = rec.get(field) if isinstance(rec, dict) else None
            fmt = v.get("format")
            out[k] = fmt.replace("{}", str(val)) if fmt else val
        else:
            out[k] = v
    return out


def execute_oracle(spec: dict) -> dict:
    """Replay the spec's oracle path with resolved dataflow; no model involved.

    Returns {"ok", "nodes": [{node, tool, args, args_digest, envelope}], "answer"}.
    """
    kb = spec.get("kb", {})
    state = None
    if spec.get("env"):
        from agentlab.suite.envs.fulfillment import WarehouseState

        state = WarehouseState(spec["env"])
    prior: dict = {}
    nodes = []
    for node in spec.get("oracle", []):
        try:
            args = _resolve_args(node.get("args", {}), prior)
        except KeyError as exc:
            return {"ok": False, "error": str(exc), "nodes": nodes, "answer": None}
        env = canonical_dispatch(kb, node.get("tool", ""), args, env=state)
        nodes.append({"node": node.get("node"), "tool": node.get("tool"), "args": args,
                      "args_digest": call_digest(node.get("tool", ""), args),
                      "envelope": env})
        if not env.get("ok"):
            return {"ok": False, "error": f"oracle node {node.get('node')} failed: "
                                          f"{env.get('error')}", "nodes": nodes, "answer": None}
        prior[node.get("node")] = env
    answer = _derive_answer(spec, nodes)
    return {"ok": True, "nodes": nodes, "answer": answer}


def _derive_answer(spec: dict, nodes: list[dict]) -> str | None:
    """The oracle answer implied by the replayed path (terminal node's value)."""
    if not nodes:
        return None
    env = nodes[-1]["envelope"]
    if "completion_token" in env:
        # fulfillment: the terminal finalize returns the token the assistant
        # must commit; there is no numeric value or KB record to read.
        return str(env["completion_token"])
    if "value" in env:
        return str(env["value"])
    rec = env.get("record")
    if isinstance(rec, dict):
        field = spec.get("answer_field", "code")
        if field in rec:
            return str(rec[field])
        scalars = [v for v in rec.values() if isinstance(v, (str, int, float))]
        return str(scalars[-1]) if scalars else None
    return None


def verify_oracle(spec: dict) -> dict:
    """S9: the task is reachable and its registered minimum horizon is true."""
    res = execute_oracle(spec)
    problems = []
    if not res["ok"]:
        problems.append(res.get("error", "oracle replay failed"))
    declared = spec.get("horizon")
    if declared != len(spec.get("oracle", [])):
        problems.append(f"declared horizon {declared} != oracle length "
                        f"{len(spec.get('oracle', []))}")
    if res["ok"] and spec.get("answer") is not None:
        if _answers_equal(str(spec["answer"]), str(res["answer"]),
                          spec.get("answer_kind", "token")):
            pass
        else:
            problems.append(f"spec answer {spec['answer']!r} != replayed {res['answer']!r}")
    return {"ok": not problems, "problems": problems,
            "replayed_answer": res.get("answer"),
            "nodes": res.get("nodes", [])}


def permute_hidden_values(specs: list[dict], seed: int) -> list[dict]:
    """Counterfactual control: permute terminal hidden values BETWEEN task IDs.

    Applies to specs whose terminal oracle node is a kb_lookup on a record
    holding the answer (answer_kind == "token"). Correct model outputs must
    track the returned (permuted) value, never the prompt/task identity (S14).
    """
    eligible = [s for s in specs if s.get("answer_kind", "token") == "token"
                and s.get("oracle") and s["oracle"][-1].get("tool") == "kb_lookup"]
    if len(eligible) < 2:
        return []
    perm = rng.permutation(seed, "counterfactual-permutation-v1", len(eligible))
    out = []
    for i, spec in enumerate(eligible):
        donor = eligible[perm[i]]
        new = json.loads(json.dumps(spec))  # deep copy
        new["answer"] = donor["answer"]
        term_replay = execute_oracle(spec)
        if not term_replay["ok"]:
            continue
        # rewrite the terminal record's answer field in the copied KB
        term_node = new["oracle"][-1]
        prior = {n["node"]: n["envelope"] for n in term_replay["nodes"][:-1]}
        term_args = _resolve_args(term_node.get("args", {}), prior)
        term_key = str(term_args.get("key", "")).strip()
        field = new.get("answer_field", "code")
        if term_key in new.get("kb", {}) and isinstance(new["kb"][term_key], dict):
            new["kb"][term_key][field] = donor["answer"]
            new["permuted_from"] = donor["task_id"]
            new["control"] = "permuted"
            out.append(new)
    return out


def redact_spec(spec: dict) -> dict:
    """Absent-information control: the required lookup never returns the hidden value."""
    new = json.loads(json.dumps(spec))
    hidden_key = new.get("hidden_key")
    if hidden_key and hidden_key in new.get("kb", {}):
        del new["kb"][hidden_key]
    else:
        # fall back: drop the terminal oracle node's record
        replay = execute_oracle(spec)
        if replay["nodes"]:
            last = replay["nodes"][-1]
            key = str(last["args"].get("key", "")).strip()
            new.get("kb", {}).pop(key, None)
    new["control"] = "redacted"
    return new


# ---------------------------------------------------------------------------
# trace primitives
# ---------------------------------------------------------------------------

def call_digest(tool: str, args: dict) -> str:
    """Normalized call identity; the recovery token never changes identity."""
    return rng.digest({"tool": tool, "args": faults_mod.strip_token(args)})


def _numeric(s):
    try:
        return float(str(s).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _answers_equal(got: str, want: str, kind: str) -> bool:
    if kind == "integer":
        a, b = _numeric(got), _numeric(want)
        return a is not None and b is not None and abs(a - b) < 1e-9
    return got.strip().lower() == want.strip().lower()


def extract_final_answer(final_text: str) -> str | None:
    """Committed answer format: the last `ANSWER: <value>`; \\boxed{} as fallback.

    Delegates to suite.schema so the strict verifier and this certification layer
    read a commitment identically.
    """
    from agentlab.suite.schema import extract_committed_answer

    return extract_committed_answer(final_text)


def _exposed_values(event: dict) -> list:
    """Scalar values the model could have legitimately read from this event."""
    text = event.get("exposed_text", "")
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("ok") is not True:
            continue
        if "value" in obj:
            vals.append(obj["value"])
        rec = obj.get("record")
        if isinstance(rec, dict):
            vals.extend(v for v in rec.values() if isinstance(v, (str, int, float)))
    return vals


def _value_matches(value, candidate) -> bool:
    a, b = _numeric(value), _numeric(candidate)
    if a is not None and b is not None:
        return abs(a - b) < 1e-9
    sv, sc = str(value).strip().lower(), str(candidate).strip().lower()
    return sv == sc or (len(sv) >= 8 and sv in sc)


def _feeds_match(produced, consumed) -> bool:
    """Does a produced value appear inside a consumed argument value?

    Beyond exact/numeric equality this allows word-boundary containment, so a
    numeric result spliced into a calculator expression ("3" in "3*4") counts,
    while "3" inside "13" does not. Opaque tokens need >= 8 chars to count as
    containment, which keeps short strings from matching accidentally.
    """
    if _value_matches(consumed, produced):
        return True
    s_p, s_c = str(produced).strip(), str(consumed)
    if not s_p:
        return False
    if _numeric(produced) is not None:
        return re.search(rf"(?<![\w.]){re.escape(s_p)}(?![\w.])", s_c) is not None
    return len(s_p) >= 8 and s_p.lower() in s_c.lower()


# ---------------------------------------------------------------------------
# runaway + hallucination (recomputed from raw trace material)
# ---------------------------------------------------------------------------

def detect_runaway(trace: dict) -> dict:
    events = trace.get("events", [])
    budgets = trace.get("budgets", {})
    max_calls = budgets.get("max_calls")
    reasons = []
    if max_calls is not None and len(events) >= max_calls:
        reasons.append("call_cap")
    # three identical normalized calls each returning the identical error
    for i in range(len(events) - 2):
        e0, e1, e2 = events[i : i + 3]
        if (e0.get("args_digest") == e1.get("args_digest") == e2.get("args_digest")
                and e0.get("exposed_digest") == e1.get("exposed_digest") == e2.get("exposed_digest")
                and faults_mod.fault_payload_is_error(e0.get("exposed_text", ""))):
            reasons.append("identical_error_loop")
            break
    # four consecutive calls without ledger advance
    seen: set = set()
    stale_run = 0
    for e in events:
        d = e.get("exposed_digest")
        if d in seen:
            stale_run += 1
            if stale_run >= 4:
                reasons.append("no_ledger_advance")
                break
        else:
            stale_run = 0
            seen.add(d)
    term = (trace.get("runner") or {}).get("termination_reason")
    if term in RUNAWAY_TERMINATIONS:
        reasons.append(f"terminated:{term}")
    return {"runaway": bool(reasons), "reasons": reasons}


def detect_hallucination(trace: dict, secret: bytes, valid_events: list[dict]) -> dict:
    """Receipt fabrication or answer values with no validated source."""
    task_id = trace.get("task_id", "")
    reasons = []
    final = _final_assistant_text(trace)
    ledger_receipts = {e.get("receipt") for e in valid_events}
    for cited in _RECEIPT_RE.findall(final or ""):
        if cited not in ledger_receipts:
            reasons.append("cited_unminted_receipt")
            break
    # tool-role messages that have no environment event behind them
    tool_msgs = [m for m in trace.get("messages", []) if m.get("role") == "tool"]
    if len(tool_msgs) > len(trace.get("events", [])):
        reasons.append("tool_message_without_event")
    got = extract_final_answer(final or "")
    if got is not None:
        sourced = any(_value_matches(got, v) for e in valid_events for v in _exposed_values(e))
        if not sourced:
            reasons.append("answer_value_unvalidated")
    _ = task_id, secret
    return {"hallucinated": bool(reasons), "reasons": reasons}


def _final_assistant_text(trace: dict) -> str:
    for m in reversed(trace.get("messages", [])):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if content.strip():
                return content
    return trace.get("final_answer") or ""


# ---------------------------------------------------------------------------
# certifications
# ---------------------------------------------------------------------------

def certify_episode(trace: dict, secret: bytes) -> dict:
    """Strict certified scoring of one episode trace; pure function of the trace."""
    spec_answer = str(trace.get("answer", ""))
    kind = trace.get("answer_kind", "token")
    task_id = trace.get("task_id", "")
    events = trace.get("events", [])
    valid_events = [e for e in events if receipt_valid(secret, task_id, e)]
    receipts_ok = len(valid_events) == len(events)

    final = _final_assistant_text(trace)
    got = extract_final_answer(final)
    raw_success = got is not None and _answers_equal(got, spec_answer, kind)

    run = detect_runaway(trace)
    hall = detect_hallucination(trace, secret, valid_events)

    # answer-bearing validated event: where the committed value was observed
    answer_event = None
    if raw_success:
        for e in valid_events:
            if any(_value_matches(got, v) for v in _exposed_values(e)):
                answer_event = e  # keep the LAST match: latest validated source
    certified = (raw_success and receipts_ok and answer_event is not None
                 and not run["runaway"] and not hall["hallucinated"])

    n_calls = len(events)
    decisions = {e.get("decision") for e in events}
    report = {
        "raw_success": raw_success,
        "certified_success": bool(certified),
        "receipts_ok": receipts_ok,
        "n_events": n_calls,
        "n_valid_events": len(valid_events),
        "n_calls": n_calls,
        "n_decisions": len([m for m in trace.get("messages", []) if m.get("role") == "assistant"]),
        "answer_extracted": got,
        "answer_event_call_id": None if answer_event is None else answer_event.get("call_id"),
        "runaway": run,
        "hallucination": hall,
        "excess_calls": n_calls - int(trace.get("horizon") or 0),
        "distinct_decisions_with_calls": len(decisions),
    }
    return report



# Frozen precedence over the non-recovery cases. `certified_recovery` is the
# CONJUNCTION of every requirement (computed independently below), so no
# ordering can inflate or deflate it; this list only decides which single
# diagnostic label a multiply-failing episode reports, and it is ordered from
# the most upstream/most severe cause to the most downstream:
#
#   not_assigned     no fault was scheduled -- the question does not apply
#   not_exposed      the fault never fired; nothing to recover from (ITT)
#   hallucinated     the trace fabricated evidence, so it is not evidence about
#                    remediation at all -- this must outrank every behavioural
#                    reason below it
#   runaway          the trace crossed a preregistered budget/loop criterion;
#                    nothing after that point can earn credit
#   remediation      the contract was never satisfied. Two mutually exclusive
#                    labels: `blind_retry` when a post-fault result arrived
#                    anyway (the tool recovered, not the agent) and
#                    `no_remediation` when it did not (the agent gave up)
#   no_post_fault_result  remediated, but the faulted node never produced a
#                    validated canonical observation
#   wrong_final      the committed answer is wrong
#   unvalidated_answer  right answer, no validated in-episode source
#   pre_fault_answer the answer's validated source precedes the fault
#   uncertified_episode  residual episode-level certification failure
#                    (e.g. a broken receipt chain)
NON_RECOVERY_PRECEDENCE = (
    "not_assigned", "not_exposed", "hallucinated", "runaway", "remediation",
    "no_post_fault_result", "wrong_final", "unvalidated_answer",
    "pre_fault_answer", "uncertified_episode",
)


def certify_recovery(trace: dict, secret: bytes, episode_report: dict | None = None) -> dict:
    """Certified recovery with the six non-recovery cases excluded explicitly.

      not_exposed          fault never emitted (stays in the ITT denominator)
      no_remediation       no fault-appropriate remedial action
      blind_retry          post-fault result obtained, but the remediation
                           contract (token / later decision / corrected target)
                           was never satisfied
      no_post_fault_result the faulted node never produced a validated
                           canonical observation after the fault
      pre_fault_answer     the answer's validated source precedes the fault
      unvalidated_answer   correct answer with no validated in-episode source
                           (guessing, clean-run value reuse, invention)
      hallucinated         fabricated receipt/tool result anywhere
      runaway              any runaway criterion crossed
      wrong_final          the final answer is simply wrong

    Every requirement is evaluated, then NON_RECOVERY_PRECEDENCE picks the
    reported label; `violations` carries the full set for the record.
    """
    rep = episode_report or certify_episode(trace, secret)
    fault = trace.get("fault") or {}
    events = trace.get("events", [])
    task_id = trace.get("task_id", "")
    valid_events = [e for e in events if receipt_valid(secret, task_id, e)]

    fault_events = [e for e in valid_events if e.get("fault_emitted")]
    out = {"assigned": bool(fault), "exposed": bool(fault_events),
           "certified_recovery": False, "reason": None}
    if not fault:
        out["reason"] = "not_assigned"
        return out
    if not fault_events:
        out["reason"] = "not_exposed"
        return out
    fev = fault_events[0]
    out["fault_class"] = fev.get("fault_class")
    req = faults_mod.remediation_requirement(fev.get("fault_class", ""))
    token = fev.get("recovery_token")
    node_digest = fev.get("args_digest")

    later = [e for e in valid_events if e.get("call_id", -1) > fev.get("call_id", -1)]
    # canonical post-fault observation at the faulted node
    recovered_events = [e for e in later
                        if e.get("args_digest") == node_digest
                        and e.get("exposed_digest") == e.get("canonical_digest")]
    # remedial action per contract
    if req["corrected_target_required"]:
        want_unit = fev.get("requested_unit")
        remedial = [e for e in later
                    if e.get("tool") == "unit_convert"
                    and e.get("requested_unit") == want_unit
                    and e.get("decision", -1) > fev.get("decision", -1)]
    else:
        remedial = [e for e in later
                    if e.get("args_digest") == node_digest
                    and e.get("token_provided") == token]
        if req["later_decision_required"]:
            remedial = [e for e in remedial
                        if e.get("decision", -1) > fev.get("decision", -1)]

    # Evaluate every requirement independently: certified_recovery is their
    # conjunction, so classification order cannot change the boolean.
    answer_id = rep["answer_event_call_id"]
    fault_call = fev.get("call_id", -1)
    labels = {
        "hallucinated": rep["hallucination"]["hallucinated"],
        "runaway": rep["runaway"]["runaway"],
        "remediation": (("blind_retry" if recovered_events else "no_remediation")
                        if not remedial else None),
        "no_post_fault_result": bool(remedial) and not recovered_events,
        "wrong_final": not rep["raw_success"],
        "unvalidated_answer": rep["raw_success"] and answer_id is None,
        "pre_fault_answer": (rep["raw_success"] and answer_id is not None
                             and answer_id <= fault_call),
        "uncertified_episode": not rep["certified_success"],
    }
    violations = [name for name in NON_RECOVERY_PRECEDENCE
                  if name in labels and labels[name]]
    out["violations"] = [labels[n] if isinstance(labels[n], str) else n
                         for n in violations]
    if violations:
        first = violations[0]
        out["reason"] = labels[first] if isinstance(labels[first], str) else first
        return out
    out["certified_recovery"] = True
    out["reason"] = "ok"
    out["violations"] = []
    return out


def certify_orchestration(trace: dict, secret: bytes, episode_report: dict | None = None,
                          required_tools: tuple = ("kb_lookup", "unit_convert", "calculator"),
                          ) -> dict:
    """All required tools must lie on the causal dataflow into the final answer.

    Edges cross later assistant decisions: event A feeds event B only when
    B.decision > A.decision and one of A's exposed values appears among B's
    argument values. Decorative calls contribute nothing.
    """
    rep = episode_report or certify_episode(trace, secret)
    task_id = trace.get("task_id", "")
    valid_events = [e for e in trace.get("events", [])
                    if receipt_valid(secret, task_id, e)]
    out = {"certified_orchestration": False, "contributing_tools": [], "reason": None}
    if not rep["certified_success"]:
        out["reason"] = "episode_not_certified"
        return out
    answer_id = rep["answer_event_call_id"]
    by_id = {e.get("call_id"): e for e in valid_events}
    if answer_id not in by_id:
        out["reason"] = "no_answer_event"
        return out

    values = {e.get("call_id"): _exposed_values(e) for e in valid_events}

    def feeds(a: dict, b: dict) -> bool:
        if a.get("decision", -1) >= b.get("decision", 1 << 30):
            return False
        args = faults_mod.strip_token(b.get("args", {}) or {})
        return any(_feeds_match(av, bv)
                   for av in values.get(a.get("call_id"), [])
                   for bv in args.values() if isinstance(bv, (str, int, float)))

    contributing = {answer_id}
    frontier = [by_id[answer_id]]
    while frontier:
        nxt = []
        for b in frontier:
            for a in valid_events:
                cid = a.get("call_id")
                if cid not in contributing and feeds(a, b):
                    contributing.add(cid)
                    nxt.append(a)
        frontier = nxt
    tools_used = sorted({by_id[c].get("tool") for c in contributing if c in by_id})
    out["contributing_tools"] = tools_used
    missing = [t for t in required_tools if t not in tools_used]
    if missing:
        out["reason"] = f"tools_not_on_causal_chain:{','.join(missing)}"
        return out
    out["certified_orchestration"] = True
    out["reason"] = "ok"
    return out


def certify_redacted(trace: dict, secret: bytes) -> dict:
    """Absent-information control scoring. ANY success here is a harness BUG."""
    rep = certify_episode(trace, secret)
    return {"raw_success": rep["raw_success"],
            "certified_success": rep["certified_success"]}
