"""The four preregistered fault injectors and the remediation contract.

A fault is scheduled deterministically per task, emitted exactly once, and is
always recoverable within the remaining registered call budget. Faults are
injected at the dispatch layer of one episode -- the global tools in
agentlab.tools stay deterministic.

Fault classes (frozen in configs/agentic_preregister.json):

  transient    error envelope; the next semantically correct attempt succeeds.
  rate_limit   error envelope; retrying within the same assistant decision
               stays limited, retrying on a later decision succeeds.
  malformed    the canonical payload is replaced once by a truncated fragment
               plus a *marked* error envelope (scope ruling: marked, never a
               silently wrong value).
  wrong_unit   unit_convert only: one valid-looking result in a deterministic
               different unit of the same family. Detection is the skill.

Remediation contract (what certified recovery requires, validated post hoc by
agentlab.provenance -- the injector only records ground truth):

  transient / rate_limit / malformed
      re-issue the call INCLUDING the emitted 128-bit recovery_token argument
      (rate_limit additionally: on a later assistant decision). Blind retries
      without the token may still succeed operationally, but they are never
      certified -- that is non-recovery case "blind_retry".
  wrong_unit
      no token is emitted (the trap is not an error envelope); certification
      requires a corrected conversion explicitly targeting the originally
      requested unit, and the trapped value must never enter the answer chain.

Recovery tokens are minted with the run secret (HMAC-SHA256), so the model
cannot fabricate them; they are still deterministic per run, so shard resumes
reproduce the same episode byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import json

from agentlab.suite import rng

FAULT_CLASSES = ("transient", "rate_limit", "malformed", "wrong_unit")

# ER-gate reporting groups: 300 assigned episodes per group, per preregistration.
FAULT_GROUPS = {
    "transient_or_rate_limit": ("transient", "rate_limit"),
    "malformed": ("malformed",),
    "wrong_unit": ("wrong_unit",),
}

TOKEN_ARG = "recovery_token"


def group_of(fault_class: str) -> str:
    for g, members in FAULT_GROUPS.items():
        if fault_class in members:
            return g
    raise ValueError(f"unknown fault class {fault_class!r}")


def eligible_nodes(oracle: list[dict], fault_class: str) -> list[int]:
    """Indices of oracle nodes where this fault class may be scheduled."""
    if fault_class == "wrong_unit":
        return [i for i, n in enumerate(oracle) if n.get("tool") == "unit_convert"]
    return list(range(len(oracle)))


def schedule_fault(task_id: str, oracle: list[dict], fault_seed: int,
                   fault_class: str | None = None) -> dict | None:
    """Deterministic (task_id, fault_seed) -> {"class", "node_index"}.

    When the generator already assigned a class, pass it in; otherwise the
    class is drawn from the eligible classes. Returns None when the oracle has
    no eligible node for the requested class (the task then cannot carry that
    fault and the caller must rebalance at generation time, not silently skip).
    """
    if not oracle:
        return None
    draws = rng.stream_u64(fault_seed, f"fault-schedule:{task_id}", 2)
    if fault_class is None:
        classes = [c for c in FAULT_CLASSES if eligible_nodes(oracle, c)]
        fault_class = classes[int(draws[0] % len(classes))]
    nodes = eligible_nodes(oracle, fault_class)
    if not nodes:
        return None
    return {"class": fault_class, "node_index": nodes[int(draws[1] % len(nodes))]}


_WRONG_UNIT_FALLBACK = {"kg": "g", "g": "mg", "mg": "g", "m": "cm", "cm": "mm",
                        "mm": "cm", "km": "m", "s": "min", "min": "s",
                        "hour": "min", "day": "hour", "lb": "oz", "oz": "lb",
                        "inch": "ft", "ft": "inch", "mile": "km"}


def wrong_unit_candidates(requested: str) -> list[str]:
    """Every other unit in `requested`'s family, sorted.

    The single source of truth for "which units could a wrong-unit trap
    plausibly return": both the generation-time picker (`pick_wrong_unit`,
    CounterRNG) and the certification-layer picker (`wrong_unit_target`, label
    stream) draw from this list, so the two layers can never disagree about
    what counts as the same family.
    """
    from agentlab.tools import _UNITS

    req = str(requested).strip().lower()
    fam = _UNITS.get(req, (None,))[0]
    return sorted(u for u, (f, _) in _UNITS.items() if f == fam and u != req)


def wrong_unit_target(requested: str, task_id: str, fault_seed: int) -> str:
    """A deterministic different unit in the same family as `requested`."""
    req = str(requested).strip().lower()
    candidates = wrong_unit_candidates(req)
    if not candidates:
        return _WRONG_UNIT_FALLBACK.get(req, req)
    pick = rng.stream_u64(fault_seed, f"wrong-unit:{task_id}", 1)[0]
    return candidates[int(pick % len(candidates))]


@dataclasses.dataclass
class FaultState:
    emitted: bool = False
    emitted_call_id: int | None = None
    emitted_decision: int | None = None
    recovery_token: str | None = None


class FaultInjector:
    """Episode-scoped injector. One scheduled fault; exactly one emission.

    The runtime calls `filter(...)` after computing the canonical result for a
    dispatch. The injector decides what the model actually observes and
    returns (exposed_text, event_fields). `event_fields` land in the
    environment-side ledger that the model never sees.
    """

    def __init__(self, task_id: str, schedule: dict | None, node_args_digest: str | None,
                 secret: bytes):
        self.task_id = task_id
        self.schedule = schedule  # {"class", "node_index"} or None (clean episode)
        self.node_args_digest = node_args_digest  # canonical args digest of target node
        self.secret = secret
        self.state = FaultState()
        if schedule is not None:
            self.state.recovery_token = rng.hmac_token(
                secret, f"recovery:{task_id}:{schedule['class']}:{schedule['node_index']}")

    # -- helpers -----------------------------------------------------------
    def _envelope(self, error: str, extra: dict) -> str:
        body = {"ok": False, "error": error, TOKEN_ARG: self.state.recovery_token,
                "remediation": extra}
        return rng.canonical_json(body)

    def _at_target(self, args_digest: str) -> bool:
        return (self.schedule is not None
                and self.node_args_digest is not None
                and args_digest == self.node_args_digest)

    # -- main entry ---------------------------------------------------------
    def filter(self, *, call_id: int, decision: int, tool: str, args_digest: str,
               canonical_text: str, requested_unit: str | None = None,
               wrong_unit_text: str | None = None) -> tuple[str, dict]:
        """Decide the exposed observation for one semantically valid dispatch.

        `canonical_text` is the true result envelope. For unit_convert calls the
        runtime precomputes `wrong_unit_text` (the valid envelope for the
        deterministic wrong unit) so this module stays tool-agnostic.
        """
        ev: dict = {"fault_class": None, "fault_emitted": False}
        if self.schedule is None:
            return canonical_text, ev

        cls = self.schedule["class"]
        st = self.state

        if not st.emitted:
            if self._at_target(args_digest):
                st.emitted = True
                st.emitted_call_id = call_id
                st.emitted_decision = decision
                ev.update({"fault_class": cls, "fault_emitted": True,
                           "recovery_token": st.recovery_token})
                if cls == "transient":
                    return self._envelope("transient_backend", {
                        "action": "reissue_with_token", "retryable": True,
                        "note": f"transient backend failure; re-issue the identical call "
                                f"with argument {TOKEN_ARG} set to the token above"}), ev
                if cls == "rate_limit":
                    return self._envelope("rate_limit", {
                        "action": "reissue_with_token_after_next_decision",
                        "retry_after_turns": 1,
                        "note": f"rate limited; on your NEXT reply re-issue the call with "
                                f"argument {TOKEN_ARG} set to the token above"}), ev
                if cls == "malformed":
                    # Truncated fragment (never contains the canonical value) plus a
                    # MARKED error envelope carrying the token, per the scope ruling.
                    frag = '{"ok":true,"value":'
                    ev["malformed_fragment"] = frag
                    return frag + "\n" + self._envelope("truncated_result", {
                        "action": "reissue_with_token",
                        "note": f"the previous payload was truncated; re-issue the identical "
                                f"call with argument {TOKEN_ARG} set to the token above"}), ev
                if cls == "wrong_unit":
                    ev["wrong_unit"] = True
                    return wrong_unit_text if wrong_unit_text is not None else canonical_text, ev
                raise ValueError(f"unknown fault class {cls!r}")
            return canonical_text, ev

        # Fault already emitted: post-fault behaviour.
        if cls == "rate_limit" and self._at_target(args_digest) and decision == st.emitted_decision:
            # Same-decision repeats stay limited; this is the SAME scheduled fault
            # still in force, not a second emission.
            ev.update({"fault_class": cls, "fault_emitted": False, "rate_limit_active": True})
            return self._envelope("rate_limit_active", {
                "action": "reissue_with_token_after_next_decision",
                "retry_after_turns": 1}), ev
        return canonical_text, ev


def parse_token_from_args(args: dict) -> str | None:
    tok = args.get(TOKEN_ARG)
    return str(tok) if tok is not None else None


def strip_token(args: dict) -> dict:
    return {k: v for k, v in args.items() if k != TOKEN_ARG}


def remediation_requirement(fault_class: str) -> dict:
    """Machine-readable statement of what certification demands (frozen)."""
    if fault_class in ("transient", "malformed"):
        return {"token_required": True, "later_decision_required": False,
                "corrected_target_required": False}
    if fault_class == "rate_limit":
        return {"token_required": True, "later_decision_required": True,
                "corrected_target_required": False}
    if fault_class == "wrong_unit":
        return {"token_required": False, "later_decision_required": False,
                "corrected_target_required": True}
    raise ValueError(f"unknown fault class {fault_class!r}")


def fault_payload_is_error(text: str) -> bool:
    """True when an exposed observation is one of the marked error envelopes."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("ok") is False:
            return True
    return False


# ===========================================================================
# OPEN PRE-PUSH BLOCKER (2026-08-05): TWO FAULT CONTRACTS COEXIST HERE.
#
# `FaultInjector` above is the contract the CLAIM-BEARING evaluation harness
# uses (`suite.evaluate.SpecRuntime`): every fault error carries a 128-bit
# `recovery_token`, and `provenance.certify_recovery` credits recovery only when
# the remediation contract is satisfied -- the token echoed back for
# transient/malformed, the token echoed back on a LATER decision for rate_limit,
# the corrected target unit for wrong_unit.
#
# `FaultEngine` below is the contract every OTHER consumer uses via
# `suite.runtime.EpisodeRuntime`: rejection sampling (`multidistill`), the prompt
# tournament (`prompt_control`), the variance probe (`variance`) and generation.
# Its payloads carry NO token, `tool_schemas_for_family` does not offer a
# `recovery_token` argument, `EpisodeRuntime.dispatch` neither parses nor strips
# one, and `verify._fault_report` credits recovery for any later canonical
# observation at the faulted node -- a bare retry.
#
# MEASURED CONSEQUENCES, all of them outcome-blind (no GPU stage has run):
#   * the environment the policy is TRAINED in is not the environment it is
#     EVALUATED in, for all four fault classes;
#   * two definitions of strict success survive, and the weaker one is what the
#     SFT acceptance filter enforces, so the corpus may contain trajectories the
#     claim-bearing certifier would label `blind_retry`;
#   * prompt candidates p4_error_repair and p8_combined instruct the model about
#     `recovery_token`, but the tournament that SELECTS the winning prompt runs in
#     the tokenless environment, where that instruction is inert.
#
# The paired TP-BP contrast is NOT invalidated -- both arms face the identical
# evaluation environment -- but "one fault engine with recovery tokens" is false
# as the tree stands, and a reader would wrongly conclude the trained arm was
# trained on the contract it is scored against.
#
# THE FIX IS NOT A THRESHOLD CHANGE. It is to port the token-bearing envelope,
# the `recovery_token` tool argument and the remediation predicate into the
# canonical runtime and verifier so ONE engine serves both paths, then
# re-measure `scenario.tool_output_max_tokens` (208, measured against the
# tokenless payloads below). That changes the training treatment, so it belongs
# to the protocol owner as a recorded decision, not to an integrator at a push
# gate. Until then this file is deliberately NOT presented as reconciled.
#
# Suite v1 runtime injectors (env-architect spec; used by suite.runtime), whose
# payloads are the exact committed forms from the council env-architect section:
#
#   transient    {"ok": false, "error": "transient_backend", "retryable": true,
#                 "request_id": ...}; fires once, before any mutation; the next
#                 semantically correct attempt succeeds.
#   malformed    the exposed observation is exactly the truncated fragment
#                 {"ok":true,"value":  -- it contains neither the numeric
#                 result nor the next key. For the scheduled 25% of malformed
#                 fulfillment cases the target is a mutation (reserve): the
#                 mutation DOES occur, the response is truncated, and recovery
#                 is an idempotent replay of the same quote token or a
#                 reservation-status query. A second reservation must never be
#                 created.
#   wrong_unit   unit_convert only: one valid JSON result for a
#                 deterministically chosen different unit of the same family,
#                 with the unit EXPLICIT in the payload. Silent plausible
#                 wrong numbers are out of scope by council ruling.
#   rate_limit   {"ok": false, "error": "rate_limit", "retry_after_turns": 1,
#                 "request_id": ...}. Logical time advances once per assistant
#                 decision: repeats within the same decision stay limited (the
#                 SAME scheduled fault still in force, never a second firing);
#                 a retry on the next decision succeeds.
#
# Wrong calls never consume a scheduled fault: the injector is consulted only
# for credit-eligible calls (semantically valid arguments at the next
# incomplete oracle node whose dependency edge is satisfied).
# ===========================================================================

MALFORMED_LITERAL = '{"ok":true,"value":'


def pick_wrong_unit(requested: str, rng_stream) -> str:
    """A deterministic different unit in the same family, drawn from `rng_stream`
    (a suite.rng.CounterRNG). Used at generation time to freeze the trap unit."""
    req = str(requested).strip().lower()
    candidates = wrong_unit_candidates(req)
    if not candidates:
        return _WRONG_UNIT_FALLBACK.get(req, req)
    return rng_stream.choice(candidates)


class FaultEngine:
    """Episode-scoped state for every scheduled fault of one episode.

    The runtime consults `directive(node_id, decision_id)` only when a call is
    credit-eligible at that node. Each scheduled fault fires exactly once;
    rate-limit same-decision repeats re-expose the error without a second
    firing (`fault_triggered` stays False on those events).
    """

    def __init__(self, faults: list) -> None:
        # faults: list[schema.FaultSpec]
        self._state = {}
        for f in faults:
            if f.target_node in self._state:
                raise ValueError(f"two faults target node {f.target_node!r}")
            self._state[f.target_node] = {
                "spec": f, "fired": False, "blocked_until": None,
            }

    def spec_for(self, node_id: str):
        st = self._state.get(node_id)
        return st["spec"] if st else None

    def triggered(self) -> list:
        """Fault specs that have fired, in no particular order."""
        return [st["spec"] for st in self._state.values() if st["fired"]]

    def directive(self, node_id: str, decision_id: int) -> dict | None:
        """What the injector does to THIS credit-eligible call, or None."""
        st = self._state.get(node_id)
        if st is None:
            return None
        spec = st["spec"]
        if not st["fired"]:
            st["fired"] = True
            if spec.fault_type == "transient":
                return {"kind": "transient", "fault_triggered": True}
            if spec.fault_type == "rate_limit":
                retry = int(spec.params.get("retry_after_turns", 1))
                st["blocked_until"] = decision_id + retry
                return {"kind": "rate_limit", "fault_triggered": True,
                        "retry_after_turns": retry}
            if spec.fault_type == "wrong_unit":
                return {"kind": "wrong_unit", "fault_triggered": True,
                        "unit": spec.params["wrong_unit"]}
            if spec.fault_type == "malformed":
                return {"kind": "malformed", "fault_triggered": True,
                        "ambiguous": bool(spec.params.get("ambiguous_mutation"))}
            raise ValueError(f"unknown fault type {spec.fault_type!r}")
        # Already fired: only the rate-limit logical clock has residual force.
        if (spec.fault_type == "rate_limit" and st["blocked_until"] is not None
                and decision_id < st["blocked_until"]):
            return {"kind": "rate_limit", "fault_triggered": False,
                    "retry_after_turns": int(spec.params.get("retry_after_turns", 1))}
        return None
