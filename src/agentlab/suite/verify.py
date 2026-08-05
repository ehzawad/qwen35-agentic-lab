"""The ONE strict verifier for suite v1 episodes, and the ONE success predicate.

The verifier recomputes oracle progress independently from the trace events
(and cross-checks the runtime's own crediting -- any disagreement is a
consistency failure that vetoes success). Binding rules:

  * every oracle node must complete, in order, with each dependency edge
    crossing a LATER assistant decision -- several guessed calls in one batch
    never count, and a same-decision dependency is never credited;
  * harmless extra read-only calls and idempotent retries are accepted but
    never counted as oracle progress;
  * a correct final answer without a CERTIFIED recovery trace is task-correct
    but not certified success (mental arithmetic, lucky guessing and blind
    retries must not masquerade as recovery).

THE REGISTERED REMEDIATION PREDICATE (binding, one derivation for every consumer)

`_fault_report` used to credit recovery for any later canonical observation at
the faulted node -- a bare retry -- while `provenance.certify_recovery` demanded
the registered contract. Two predicates meant the SFT acceptance filter admitted
trajectories the claim-bearing certifier labels `blind_retry`. There is now one,
and it is the registered one. A certifying recovery EVENT must itself satisfy
both the remediation and the result requirement:

  transient / malformed   a later call with the same stripped call identity,
                          supplying the exact emitted token, exposing the
                          canonical observation.
  rate_limit              the same, and on a strictly later assistant decision.
  wrong_unit              a later unit_convert explicitly requesting the
                          originally requested target unit, exposing the
                          canonical result.
  ambiguous malformed     a token-bearing idempotent replay of the original
  mutation                mutation (`replay` true) with the canonical response.

The qualifying event must BE the event that establishes recovery. Computing
"remedial action" and "canonical result" as two independent event sets is a
laundering hole: one blind retry obtains the value and a later decorative token
echo certifies it. That is closed here. A status query after an ambiguous
malformed mutation may establish operational state, but under the registered
wording it is not certified remediation, because it neither reissues the same
call nor echoes the token.

Classification (frozen):

  ok                    a qualifying recovery event exists.
  blind_retry           a canonical post-fault result exists, but no qualifying
                        remediation event.
  no_remediation        neither remediation nor result.
  no_post_fault_result  a qualifying action exists but no canonical result.
  not_exposed           the fault never fired (stays in the ITT denominator).

THE ONE SUCCESS PREDICATE

`certified_success` is the single claim-bearing boolean. `answer_ok`/`raw_success`
and `task_success` remain diagnostics; the old, weaker `strict_success` is gone
rather than left as a second headline candidate. `provenance.certify_episode`
delegates here instead of recreating task success from the transcript.

Three denominators are reported by `aggregate_recovery`:

  1. fault-arm certified success over ALL assigned fault episodes  (headline)
  2. fault trigger rate over assigned episodes
  3. recovery success over triggered faults                        (diagnostic)
"""

from __future__ import annotations

import dataclasses

from .faults import remediation_requirement
from .schema import extract_committed_answer

# Terminations that are, by themselves, runaway evidence: the episode stopped
# because it ran out of wall clock, out of parser patience, or hit the call cap.
RUNAWAY_TERMINATIONS = frozenset({"token_budget", "wall_clock", "parser_budget",
                                  "call_cap"})

# The worst report names the episode: most upstream cause first.
RECOVERY_REASON_ORDER = ("not_exposed", "no_remediation", "blind_retry",
                         "no_post_fault_result", "ok")


@dataclasses.dataclass
class Verdict:
    task_id: str
    family: str
    horizon: int
    split: str
    certified_success: bool
    task_success: bool
    answer_ok: bool
    raw_success: bool
    nodes_total: int
    unique_valid_nodes: int
    node_decisions: dict
    calls: int
    decisions: int
    excess_calls: int
    within_budget: bool
    unsafe_mutation: bool
    state_ok: bool
    tokens_ok: bool
    consistent: bool
    receipts_ok: bool
    answer_event_call_id: int | None
    runaway: bool
    runaway_reasons: list
    hallucinated: bool
    hallucination_reasons: list
    fault_assigned: int
    faults_triggered: int
    fault_fire_counts: dict
    fault_reports: list
    recovery_attempted: bool
    recovered: bool
    recovery_success: bool
    recovery_reason: str | None
    reasons: list

    def to_row(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def milestone_fraction(self) -> float:
        """Verified oracle progress in [0, 1].

        The only milestone measure in the suite: unique dependency-valid nodes
        over the declared horizon. Repeats and extra read-only calls never
        raise it, because `_recompute_completion` credits a node once and only
        in order. GRPO shaping and the variance probe both read this.
        """
        if self.nodes_total <= 0:
            return 0.0
        return min(self.unique_valid_nodes, self.nodes_total) / self.nodes_total


def _answer_ok(spec, final_text: str) -> bool:
    """Did the assistant COMMIT the right answer, in the committed format?

    Both answer kinds require a real commitment (`ANSWER: <value>`, or `\\boxed{}`
    as the preregistered fallback) read by the one shared extractor. A token
    answer merely appearing somewhere in the prose is NOT a commitment: the model
    could have echoed a KB record without deciding anything.
    """
    committed = extract_committed_answer(final_text)
    if committed is None:
        return False
    if spec.answer_kind == "token":
        return committed.strip().lower() == str(spec.answer).strip().lower()
    try:
        return abs(float(committed.replace(",", "").replace(" ", ""))
                   - float(spec.answer)) < 1e-9
    except ValueError:
        return False


def _recompute_completion(nodes, events) -> dict:
    """Independent replay of the crediting rules -> {node_id: decision_id}."""
    index = {n.node_id: i for i, n in enumerate(nodes)}
    completed: dict[str, int] = {}
    for e in events:
        nid = e.oracle_node
        if nid is None or nid in completed:
            continue
        idx = index[nid]
        if idx != len(completed):
            continue  # strictly ordered progress only
        if idx > 0 and completed[nodes[idx - 1].node_id] >= e.decision_id:
            continue  # same-decision dependency: never credited
        if e.fault_triggered and e.fault_type == "malformed" and e.state_mutated:
            completed[nid] = e.decision_id  # ambiguous mutation: state moved
            continue
        if e.exposed_canonical and not e.fault_triggered:
            completed[nid] = e.decision_id
    return completed


# ---------------------------------------------------------------------------
# the one remediation predicate
# ---------------------------------------------------------------------------

def _qualifying_recovery(fault, fire, events) -> tuple[object | None, bool, bool]:
    """-> (certifying event or None, canonical result exists, action attempted).

    `canonical result exists` is what separates `blind_retry` from
    `no_remediation`: the tool eventually produced the right value, whether or not
    the agent earned it. `action attempted` is what separates
    `no_post_fault_result` from `no_remediation`: the agent DID perform the
    registered remediation action, but no canonical observation followed.
    """
    req = remediation_requirement(fault.fault_type)
    token = fire.recovery_token
    node_identity = fire.canonical_args_digest
    later = [e for e in events if e.call_id > fire.call_id]
    ambiguous = (fault.fault_type == "malformed"
                 and bool(fault.params.get("ambiguous_mutation")))

    if req["corrected_target_required"]:
        want_unit = fire.requested_unit
        attempted = [e for e in later
                     if e.tool == "unit_convert"
                     and e.requested_unit == want_unit
                     and e.decision_id > fire.decision_id]
        canonical_later = [e for e in later
                           if e.oracle_node == fault.target_node
                           and e.exposed_canonical]
        qualifying = [e for e in attempted if e.exposed_canonical
                      and e.oracle_node == fault.target_node]
        return ((qualifying[0] if qualifying else None), bool(canonical_later),
                bool(attempted))

    # Token-bearing contract. The SAME event must carry the exact token AND
    # expose the canonical observation, so a blind retry cannot be laundered by a
    # decorative token echo afterwards.
    same_call = [e for e in later if e.canonical_args_digest == node_identity]
    canonical_later = [e for e in same_call if e.exposed_canonical]
    attempted = [e for e in same_call
                 if token is not None and e.token_provided == token]
    if req["later_decision_required"]:
        attempted = [e for e in attempted if e.decision_id > fire.decision_id]
    if ambiguous:
        # The mutation already happened and was credited; the registered repair is
        # a token-bearing IDEMPOTENT replay of the same call. A status query may
        # establish operational state but is not certified remediation: it neither
        # reissues the same call nor echoes the token.
        canonical_later = [e for e in canonical_later if e.replay]
    qualifying = [e for e in attempted if e.exposed_canonical
                  and (not ambiguous or e.replay)]
    return ((qualifying[0] if qualifying else None), bool(canonical_later),
            bool(attempted))


def _fault_report(fault, events) -> dict:
    """One fault's registered remediation report."""
    fires = [e for e in events
             if e.fault_triggered and e.fault_type == fault.fault_type
             and e.oracle_node == fault.target_node]
    rep = {"fault_type": fault.fault_type, "target_node": fault.target_node,
           "fire_count": len(fires), "triggered": bool(fires),
           "attempted": False, "recovered": False, "reason": "not_exposed",
           "recovery_call_id": None, "recovery_decision": None,
           "fault_decision": None}
    if not fires:
        return rep
    fire = fires[0]
    rep["fault_decision"] = fire.decision_id
    event, canonical_result, attempted = _qualifying_recovery(fault, fire, events)
    rep["attempted"] = bool(attempted)
    rep["recovered"] = event is not None
    if event is not None:
        rep["reason"] = "ok"
        rep["recovery_call_id"] = event.call_id
        rep["recovery_decision"] = event.decision_id
    elif attempted:
        rep["reason"] = "no_post_fault_result"
    elif canonical_result:
        rep["reason"] = "blind_retry"
    else:
        rep["reason"] = "no_remediation"
    return rep


def _state_ok(spec, env) -> tuple[bool, list]:
    if spec.family != "fulfillment":
        return True, []
    reasons = []
    if env is None:
        return False, ["fulfillment episode has no environment state"]
    snap = env.snapshot()
    oracle_final = (spec.env or {}).get("oracle_final")
    if snap != oracle_final:
        reasons.append("final state differs from oracle deltas")
    if snap["ledger"] < 0:
        reasons.append("budget violated")
    reserved_mass = sum(ln["mass_kg"] for ln in spec.env["lines"]
                        if str(ln["line"]) in snap["reserved"])
    if reserved_mass > spec.env["mass_limit_kg"]:
        reasons.append("shipping-mass ceiling violated")
    for ln in spec.env["lines"]:
        got = snap["reserved"].get(str(ln["line"]))
        if got != ln["quantity"]:
            reasons.append(f"line {ln['line']} quantity {got} != {ln['quantity']}")
    if snap["status"] != "complete":
        reasons.append("order status is not complete")
    return not reasons, reasons


# ---------------------------------------------------------------------------
# runaway (recomputed from the canonical events)
# ---------------------------------------------------------------------------

def detect_runaway(events, spec, termination_reason: str | None = None) -> dict:
    """Preregistered runaway criteria over canonical events.

    THE EQUALITY-CAP RULING (binding): an episode that succeeds using exactly
    `max_calls` is WITHIN budget and is not runaway. `call_cap` means the runner
    attempted one more call than the cap allowed, which the runner reports as a
    termination reason -- it is never inferred from `len(events) == max_calls`.
    The verifier and the reward commentary both treat `calls <= max_calls` as
    reachable success, so inferring failure at equality would silently contradict
    them.
    """
    from .faults import fault_payload_is_error

    reasons = []
    if len(events) > spec.max_calls:
        reasons.append("over_call_cap")
    for i in range(len(events) - 2):
        e0, e1, e2 = events[i:i + 3]
        if (e0.canonical_args_digest == e1.canonical_args_digest
                == e2.canonical_args_digest
                and e0.exposed_result_digest == e1.exposed_result_digest
                == e2.exposed_result_digest
                and fault_payload_is_error(e0.exposed_text)):
            reasons.append("identical_error_loop")
            break
    seen: set = set()
    stale_run = 0
    for e in events:
        d = e.exposed_result_digest
        if d in seen:
            stale_run += 1
            if stale_run >= 4:
                reasons.append("no_ledger_advance")
                break
        else:
            stale_run = 0
            seen.add(d)
    if termination_reason in RUNAWAY_TERMINATIONS:
        reasons.append(f"terminated:{termination_reason}")
    return {"runaway": bool(reasons), "reasons": reasons}


# ---------------------------------------------------------------------------
# the one verifier
# ---------------------------------------------------------------------------

def verify_episode(spec, nodes, events, final_text: str, env=None, *,
                   secret: bytes | None = None, transcript: list | None = None,
                   termination_reason: str | None = None) -> Verdict:
    """The single certified-success predicate.

    `secret` enables the receipt-chain check (every event's receipt must be the
    HMAC of its own exposed bytes). `transcript` enables the transcript-level
    hallucination checks (a cited receipt the ledger never minted, a tool-role
    message with no environment event behind it, a committed value absent from
    every validated observation). `termination_reason` carries the runner's own
    account of why the episode stopped.

    All three are optional so an offline single-episode replay can still verify
    the trace-level conditions, but the claim-bearing paths supply all three, and
    `certified_success` is the conjunction of everything that was checkable.
    """
    from agentlab import provenance

    reasons: list[str] = []

    completed = _recompute_completion(nodes, events)
    runtime_credits = {e.oracle_node: e.decision_id for e in events if e.credited}
    consistent = completed == runtime_credits
    if not consistent:
        reasons.append("verifier/runtime crediting disagreement")

    answer_ok = _answer_ok(spec, final_text)
    if not answer_ok:
        reasons.append("final answer missing or wrong")

    all_nodes = len(completed) == len(nodes)
    if not all_nodes:
        missing = [n.node_id for n in nodes if n.node_id not in completed]
        reasons.append(f"oracle nodes missing: {','.join(missing)}")

    unsafe = any(e.unsafe for e in events)
    if unsafe:
        reasons.append("unsafe mutation outside the oracle plan")

    tokens_ok = not any((e.ok or e.state_mutated) and not e.token_known
                        for e in events)
    if not tokens_ok:
        reasons.append("capability token used before any observation revealed it")

    state_ok, state_reasons = _state_ok(spec, env)
    reasons.extend(state_reasons)

    calls = len(events)
    decisions = max((e.decision_id for e in events), default=0)
    within_budget = calls <= spec.max_calls and decisions <= spec.max_decisions
    if not within_budget:
        reasons.append("episode exceeded its call/decision budget")

    # -- receipt chain ------------------------------------------------------
    receipts_ok = True
    if secret is not None:
        bad = [e.call_id for e in events
               if not provenance.receipt_valid(secret, spec.task_id, e.to_row())]
        receipts_ok = not bad
        if bad:
            reasons.append(f"invalid environment receipts on calls {bad[:4]}")

    # -- runaway ------------------------------------------------------------
    run = detect_runaway(events, spec, termination_reason)
    if run["runaway"]:
        reasons.append("runaway: " + ",".join(run["reasons"]))

    # -- answer provenance + hallucination ----------------------------------
    valid_events = [e.to_row() for e in events
                    if secret is None
                    or provenance.receipt_valid(secret, spec.task_id, e.to_row())]
    answer_event_call_id = provenance.answer_event_call_id(
        final_text, str(spec.answer), spec.answer_kind, valid_events)
    if answer_ok and answer_event_call_id is None:
        reasons.append("committed answer has no validated in-episode source")
    hall = provenance.detect_hallucination_events(
        final_text, valid_events, transcript=transcript,
        answer_event_call_id=answer_event_call_id)
    if hall["hallucinated"]:
        reasons.append("hallucinated result: " + ",".join(hall["reasons"]))

    # -- faults -------------------------------------------------------------
    fault_reports = [_fault_report(f, events) for f in spec.faults]
    faults_triggered = sum(1 for r in fault_reports if r["triggered"])
    fire_counts = {f"{r['fault_type']}@{r['target_node']}": r["fire_count"]
                   for r in fault_reports}
    attempted = all(r["attempted"] for r in fault_reports) if fault_reports else False
    recovered = (all(r["triggered"] and r["recovered"] for r in fault_reports)
                 if fault_reports else False)
    recovery_reason = None
    if fault_reports:
        # The worst report names the episode: the registered precedence is
        # not_exposed > no_remediation > blind_retry > no_post_fault_result > ok.
        recovery_reason = min((r["reason"] for r in fault_reports),
                             key=lambda name: RECOVERY_REASON_ORDER.index(name))

    task_success = (answer_ok and all_nodes and not unsafe and state_ok
                    and tokens_ok and within_budget and consistent)
    certified_success = (task_success and receipts_ok
                         and answer_event_call_id is not None
                         and not run["runaway"] and not hall["hallucinated"])
    if spec.faults:
        certified_success = certified_success and recovered
        if task_success and not recovered:
            reasons.append("task-correct but the injected fault was not "
                           f"certifiably recovered ({recovery_reason})")
    recovery_success = bool(spec.faults) and certified_success

    return Verdict(
        task_id=spec.task_id, family=spec.family, horizon=spec.horizon,
        split=spec.split, certified_success=certified_success,
        task_success=task_success, answer_ok=answer_ok, raw_success=answer_ok,
        nodes_total=len(nodes), unique_valid_nodes=len(completed),
        node_decisions=dict(completed), calls=calls, decisions=decisions,
        excess_calls=calls - spec.horizon, within_budget=within_budget,
        unsafe_mutation=unsafe, state_ok=state_ok, tokens_ok=tokens_ok,
        consistent=consistent, receipts_ok=receipts_ok,
        answer_event_call_id=answer_event_call_id,
        runaway=run["runaway"], runaway_reasons=run["reasons"],
        hallucinated=hall["hallucinated"], hallucination_reasons=hall["reasons"],
        fault_assigned=len(spec.faults),
        faults_triggered=faults_triggered, fault_fire_counts=fire_counts,
        fault_reports=fault_reports,
        recovery_attempted=attempted, recovered=recovered,
        recovery_success=recovery_success, recovery_reason=recovery_reason,
        reasons=reasons,
    )


def aggregate_recovery(verdicts: list) -> dict:
    """The three preregistered recovery denominators over fault-arm verdicts."""
    assigned = [v for v in verdicts if v.fault_assigned > 0]
    triggered = [v for v in assigned if v.faults_triggered == v.fault_assigned]
    return {
        "n_assigned": len(assigned),
        "n_triggered": len(triggered),
        "fault_arm_certified_success": (
            sum(v.certified_success for v in assigned) / len(assigned)
            if assigned else 0.0),
        "fault_trigger_rate": (len(triggered) / len(assigned)
                               if assigned else 0.0),
        "recovery_success_over_triggered": (
            sum(v.recovery_success for v in triggered) / len(triggered)
            if triggered else 0.0),
    }
