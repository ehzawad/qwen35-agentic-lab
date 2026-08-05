"""Strict trace/state verifier for suite v1 episodes.

The verifier recomputes oracle progress independently from the trace events
(and cross-checks the runtime's own crediting -- any disagreement is a
consistency failure that vetoes success). Binding rules:

  * every oracle node must complete, in order, with each dependency edge
    crossing a LATER assistant decision -- several guessed calls in one batch
    never count, and a same-decision dependency is never credited;
  * harmless extra read-only calls and idempotent retries are accepted but
    never counted as oracle progress;
  * a correct final answer without a verified recovery trace is task-correct
    but NOT strict fault success (mental arithmetic and lucky guessing must
    not masquerade as recovery).

Recovery definitions (binding):

  fault_triggered     the model reached the selected oracle node with
                      semantically valid arguments and the injector emitted
                      the scheduled faulty observation.
  recovery_attempted  after that event, the model performed a
                      fault-type-specific accepted recovery action.
  recovered           the runtime later produced the canonical valid
                      observation for that node, or, for an ambiguous
                      mutation, established the correct post-mutation state
                      through an idempotent replay / status query.
  recovery_success    fault_triggered AND recovered AND strict task success.

Three denominators are reported by `aggregate_recovery`:

  1. fault-arm strict success over ALL assigned fault episodes  (headline)
  2. fault trigger rate over assigned episodes
  3. recovery success over triggered faults                     (diagnostic)
"""

from __future__ import annotations

import dataclasses

from .schema import extract_committed_answer


@dataclasses.dataclass
class Verdict:
    task_id: str
    family: str
    horizon: int
    split: str
    strict_success: bool
    task_success: bool
    answer_ok: bool
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
    fault_assigned: int
    faults_triggered: int
    fault_fire_counts: dict
    recovery_attempted: bool
    recovered: bool
    recovery_success: bool
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


def _fault_report(fault, events) -> dict:
    fires = [e for e in events
             if e.fault_triggered and e.fault_type == fault.fault_type
             and e.oracle_node == fault.target_node]
    rep = {"fault_type": fault.fault_type, "target_node": fault.target_node,
           "fire_count": len(fires), "triggered": bool(fires),
           "attempted": False, "recovered": False}
    if not fires:
        return rep
    first = fires[0]
    after = [e for e in events if e.call_id > first.call_id]
    attempted = any(e.oracle_node == fault.target_node for e in after)
    recovered = any(e.oracle_node == fault.target_node and e.exposed_canonical
                    for e in after)
    if fault.fault_type == "malformed" and fault.params.get("ambiguous_mutation"):
        line = fault.params.get("line")
        status = any(e.tool == "warehouse_query" and e.ok
                     and e.aux.get("resource") == "reservation"
                     and e.aux.get("line") == line
                     for e in after)
        attempted = attempted or status
        recovered = recovered or status
    rep["attempted"] = attempted
    rep["recovered"] = recovered
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


def verify_episode(spec, nodes, events, final_text: str, env=None) -> Verdict:
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

    fault_reports = [_fault_report(f, events) for f in spec.faults]
    faults_triggered = sum(1 for r in fault_reports if r["triggered"])
    fire_counts = {f"{r['fault_type']}@{r['target_node']}": r["fire_count"]
                   for r in fault_reports}
    attempted = all(r["attempted"] for r in fault_reports) if fault_reports else False
    recovered = (all(r["triggered"] and r["recovered"] for r in fault_reports)
                 if fault_reports else False)

    task_success = (answer_ok and all_nodes and not unsafe and state_ok
                    and tokens_ok and within_budget and consistent)
    if spec.faults:
        strict_success = task_success and recovered
        if task_success and not recovered:
            reasons.append("task-correct but the injected fault was not "
                           "mechanically recovered")
    else:
        strict_success = task_success
    recovery_success = bool(spec.faults) and strict_success

    return Verdict(
        task_id=spec.task_id, family=spec.family, horizon=spec.horizon,
        split=spec.split, strict_success=strict_success,
        task_success=task_success, answer_ok=answer_ok,
        nodes_total=len(nodes), unique_valid_nodes=len(completed),
        node_decisions=dict(completed), calls=calls, decisions=decisions,
        excess_calls=calls - spec.horizon, within_budget=within_budget,
        unsafe_mutation=unsafe, state_ok=state_ok, tokens_ok=tokens_ok,
        consistent=consistent, fault_assigned=len(spec.faults),
        faults_triggered=faults_triggered, fault_fire_counts=fire_counts,
        recovery_attempted=attempted, recovered=recovered,
        recovery_success=recovery_success, reasons=reasons,
    )


def aggregate_recovery(verdicts: list) -> dict:
    """The three preregistered recovery denominators over fault-arm verdicts."""
    assigned = [v for v in verdicts if v.fault_assigned > 0]
    triggered = [v for v in assigned if v.faults_triggered == v.fault_assigned]
    return {
        "n_assigned": len(assigned),
        "n_triggered": len(triggered),
        "fault_arm_strict_success": (
            sum(v.strict_success for v in assigned) / len(assigned)
            if assigned else 0.0),
        "fault_trigger_rate": (len(triggered) / len(assigned)
                               if assigned else 0.0),
        "recovery_success_over_triggered": (
            sum(v.recovery_success for v in triggered) / len(triggered)
            if triggered else 0.0),
    }
