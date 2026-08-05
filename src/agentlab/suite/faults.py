"""The four preregistered fault injectors and the ONE remediation contract.

A fault is scheduled deterministically per task, emitted exactly once, and is
always recoverable within the remaining registered call budget. Faults are
injected at the dispatch layer of one episode -- the global tools in
agentlab.tools stay deterministic.

Fault classes (frozen in configs/agentic_preregister.json):

  transient    error envelope carrying a recovery_token; re-issuing the
               identical call WITH the token succeeds.
  rate_limit   error envelope carrying a recovery_token; re-issuing within the
               same assistant decision stays limited, re-issuing with the token
               on a LATER decision succeeds.
  malformed    the canonical payload is replaced once by a truncated fragment
               plus a *marked* error envelope carrying the token (scope ruling:
               marked, never a silently wrong value).
  wrong_unit   unit_convert only: one valid-looking result in the COMMITTED
               different unit of the same family. No token is emitted (the trap
               is not an error envelope). Detection is the skill.

Remediation contract (what certified recovery requires; validated post hoc by
agentlab.suite.verify -- the injector only records ground truth):

  transient / malformed
      re-issue the same call INCLUDING the emitted 128-bit recovery_token
      argument.
  rate_limit
      the same, additionally on a LATER assistant decision.
  wrong_unit
      a corrected conversion explicitly targeting the originally requested
      unit; the trapped value must never enter the answer chain.

Blind retries without the token may still succeed operationally, but they are
never certified -- that is the registered non-recovery case "blind_retry".

Recovery tokens are minted with the run secret (HMAC-SHA256), so the model
cannot fabricate them; they are still deterministic per run, so shard resumes
and replays reproduce the same episode byte-for-byte.

ONE ENGINE. `FaultEngine` below is the only fault implementation in the project.
The parallel `FaultInjector` that `suite.evaluate.SpecRuntime` used is gone: it
was a second contract (token-bearing) racing a first (tokenless), which is the
D2 defect this module now closes. Everything the evaluator needed -- the token,
the remediation text, the same-decision rate-limit behaviour, the truncated
malformed prefix -- is here, and `suite.runtime.EpisodeRuntime` is the single
consumer.
"""

from __future__ import annotations

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

# The ONE declaration of the remediation argument on the model-visible tool
# surface. It used to live in `suite.evaluate` as `_TOKEN_PROPERTY`, where only
# the evaluation path augmented its schemas with it -- so the trained policy was
# never shown the argument the certifier requires it to use.
# `suite.runtime.tool_schemas_for_family` now adds this to EVERY tool of EVERY
# family, because the model cannot know in advance which tool will fault.
TOKEN_PROPERTY = {
    "type": "string",
    "description": "Only after a tool error supplied a recovery_token: echo it "
                   "when reissuing that call.",
}

# The truncated fragment a malformed fault exposes. It contains neither the
# numeric result nor the next key.
MALFORMED_LITERAL = '{"ok":true,"value":'


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
    plausibly return". Only the generation-time picker (`pick_wrong_unit`) draws
    from it now: the runtime uses the COMMITTED `FaultSpec.params["wrong_unit"]`
    and never re-derives a unit, which is the second half of the D2 wrong-unit
    drift (`SpecRuntime` used to recompute one through `wrong_unit_target`).
    """
    from agentlab.tools import _UNITS

    req = str(requested).strip().lower()
    fam = _UNITS.get(req, (None,))[0]
    return sorted(u for u, (f, _) in _UNITS.items() if f == fam and u != req)


def pick_wrong_unit(requested: str, rng_stream) -> str:
    """A deterministic different unit in the same family, drawn from `rng_stream`
    (a suite.rng.CounterRNG). Used at GENERATION time to freeze the trap unit,
    which is then committed in `FaultSpec.params["wrong_unit"]` and is the only
    value the runtime may expose."""
    req = str(requested).strip().lower()
    candidates = wrong_unit_candidates(req)
    if not candidates:
        return _WRONG_UNIT_FALLBACK.get(req, req)
    return rng_stream.choice(candidates)


# ---------------------------------------------------------------------------
# the registered model-visible envelopes
# ---------------------------------------------------------------------------

def recovery_token(secret: bytes, task_id: str, fault_type: str,
                   target_node: str) -> str:
    """The 128-bit (32 hex character) HMAC recovery token for one scheduled fault.

    Keyed by the run secret, so the model cannot fabricate it; a pure function of
    (task_id, fault_type, target_node), so a replay of the same episode under the
    same run secret reproduces the same bytes.
    """
    return rng.hmac_token(secret, f"recovery:{task_id}:{fault_type}:{target_node}")


def _envelope(error: str, token: str, remediation: dict) -> str:
    return rng.canonical_json({"ok": False, "error": error, TOKEN_ARG: token,
                               "remediation": remediation})


def transient_envelope(token: str) -> str:
    return _envelope("transient_backend", token, {
        "action": "reissue_with_token", "retryable": True,
        "note": f"transient backend failure; re-issue the identical call with "
                f"argument {TOKEN_ARG} set to the token above"})


def rate_limit_envelope(token: str, retry_after_turns: int = 1) -> str:
    return _envelope("rate_limit", token, {
        "action": "reissue_with_token_after_next_decision",
        "retry_after_turns": int(retry_after_turns),
        "note": f"rate limited; on your NEXT reply re-issue the call with "
                f"argument {TOKEN_ARG} set to the token above"})


def rate_limit_active_envelope(token: str, retry_after_turns: int = 1) -> str:
    """Same-decision repeat: the SAME scheduled fault still in force."""
    return _envelope("rate_limit_active", token, {
        "action": "reissue_with_token_after_next_decision",
        "retry_after_turns": int(retry_after_turns)})


def malformed_envelope(token: str) -> str:
    """The truncated prefix, a newline, then a MARKED error envelope."""
    return MALFORMED_LITERAL + "\n" + _envelope("truncated_result", token, {
        "action": "reissue_with_token",
        "note": f"the previous payload was truncated; re-issue the identical "
                f"call with argument {TOKEN_ARG} set to the token above"})


def parse_token_from_args(args: dict) -> str | None:
    tok = (args or {}).get(TOKEN_ARG)
    return str(tok) if tok is not None else None


def strip_token(args: dict) -> dict:
    return {k: v for k, v in (args or {}).items() if k != TOKEN_ARG}


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
    for line in (text or "").splitlines():
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


# ---------------------------------------------------------------------------
# the one fault engine
# ---------------------------------------------------------------------------

class FaultEngine:
    """Episode-scoped state for every scheduled fault of one episode.

    `suite.runtime.EpisodeRuntime` consults `directive(node_id, decision_id)`
    only when a call is credit-eligible at that node ("wrong calls do not
    consume the scheduled failure"). Each scheduled fault fires exactly once;
    rate-limit same-decision repeats re-expose the error without a second firing
    (`fault_triggered` stays False on those events).

    The engine is constructed with the episode's `task_id` and the RUN SECRET,
    because the registered envelopes carry a keyed recovery token. A runtime
    built with a different secret produces different model-visible bytes, which
    is exactly why the secret is a run-scoped parameter threaded through every
    consumer rather than a per-module default.
    """

    def __init__(self, faults: list, *, task_id: str, secret: bytes) -> None:
        # faults: list[schema.FaultSpec]
        if not isinstance(secret, (bytes, bytearray)) or not secret:
            raise ValueError("FaultEngine requires the run secret (bytes)")
        self.task_id = task_id
        self.secret = bytes(secret)
        self._state = {}
        for f in faults:
            if f.target_node in self._state:
                raise ValueError(f"two faults target node {f.target_node!r}")
            token = None
            if f.fault_type != "wrong_unit":
                token = recovery_token(self.secret, task_id, f.fault_type,
                                       f.target_node)
            self._state[f.target_node] = {
                "spec": f, "fired": False, "blocked_until": None, "token": token,
            }

    def spec_for(self, node_id: str):
        st = self._state.get(node_id)
        return st["spec"] if st else None

    def token_for(self, node_id: str) -> str | None:
        st = self._state.get(node_id)
        return st["token"] if st else None

    def triggered(self) -> list:
        """Fault specs that have fired, in no particular order."""
        return [st["spec"] for st in self._state.values() if st["fired"]]

    def directive(self, node_id: str, decision_id: int) -> dict | None:
        """What the injector does to THIS credit-eligible call, or None."""
        st = self._state.get(node_id)
        if st is None:
            return None
        spec = st["spec"]
        token = st["token"]
        if not st["fired"]:
            st["fired"] = True
            if spec.fault_type == "transient":
                return {"kind": "transient", "fault_triggered": True, "token": token}
            if spec.fault_type == "rate_limit":
                retry = int(spec.params.get("retry_after_turns", 1))
                st["blocked_until"] = decision_id + retry
                return {"kind": "rate_limit", "fault_triggered": True, "token": token,
                        "retry_after_turns": retry}
            if spec.fault_type == "wrong_unit":
                # The COMMITTED trap unit, never a re-derived one.
                return {"kind": "wrong_unit", "fault_triggered": True, "token": None,
                        "unit": spec.params["wrong_unit"]}
            if spec.fault_type == "malformed":
                return {"kind": "malformed", "fault_triggered": True, "token": token,
                        "ambiguous": bool(spec.params.get("ambiguous_mutation"))}
            raise ValueError(f"unknown fault type {spec.fault_type!r}")
        # Already fired: only the rate-limit logical clock has residual force.
        if (spec.fault_type == "rate_limit" and st["blocked_until"] is not None
                and decision_id < st["blocked_until"]):
            return {"kind": "rate_limit_active", "fault_triggered": False,
                    "token": token,
                    "retry_after_turns": int(spec.params.get("retry_after_turns", 1))}
        return None
