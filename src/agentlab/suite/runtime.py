"""EpisodeRuntime: per-episode KB view, state, fault schedule and trace.

Faults are never implemented by making the three global tools
nondeterministic: agentlab.tools stays pure, and every episode binds its own
KB view, stateful environment (fulfillment), fault schedule, logical decision
clock, mutation log, oracle progress and trace events.

Dispatch contract: `runtime(name, arguments) -> str` (a JSON envelope such as
{"event_id": "e17", "ok": true, "unit": "g", "value": "1200"}), suitable for
`chat.run_agent_loop(..., dispatcher=runtime)`. kb_lookup misses return only
no_entry -- never a key list.

Oracle progress (binding):

  * a call semantically REACHES a node when the node's matcher accepts its
    arguments;
  * it is CREDITED only when that node is the next incomplete node AND its
    predecessor completed at a strictly earlier assistant decision -- several
    guessed calls in one batch never count;
  * repeats of completed nodes and extra read-only calls are harmless but
    earn nothing;
  * the scheduled fault fires only on a credit-eligible call ("wrong calls do
    not consume the scheduled failure");
  * an ambiguous malformed mutation credits the node (the mutation happened)
    while withholding the canonical observation -- recovery is verified
    separately.
"""

from __future__ import annotations

import ast
import json

from agentlab import tools as _tools

from .faults import MALFORMED_LITERAL, FaultEngine
from .kb import KBView
from .schema import (TaskSpec, TraceEvent, call_args_digest, canon, digest,
                     digest_text, normalize_number)

READ_ONLY_TOOLS = ("calculator", "unit_convert", "kb_lookup", "warehouse_query")

_ARG_TYPES = {
    "calculator": {"expression": str},
    "unit_convert": {"value": float, "from_unit": str, "to_unit": str},
    "kb_lookup": {"key": str},
    "warehouse_query": {"resource": str, "token": str, "quantity": int,
                        "mass_kg": float},
    "warehouse_update": {"action": str, "token": str, "quantity": int},
}


def tool_schemas_for_family(family: str) -> list[dict]:
    """The canonical three schemas, plus the two warehouse tools for fulfillment."""
    schemas = _tools.tool_schemas()
    if family == "fulfillment":
        from .envs.fulfillment import warehouse_tool_schemas

        schemas = schemas + warehouse_tool_schemas()
    return schemas


def tool_names_for_family(family: str) -> list[str]:
    """The legal tool surface of a family, derived FROM the schemas.

    Names and schemas cannot drift apart because there is only one source: a
    rollout engine that rejects a call as `unknown_tool` uses exactly the names
    the model was shown.
    """
    return [s["function"]["name"] for s in tool_schemas_for_family(family)]


def _coerce(name: str, raw: dict) -> dict:
    """Cast string arguments to declared types; uncastable values pass through."""
    types = _ARG_TYPES.get(name, {})
    out = {}
    for k, v in (raw or {}).items():
        want = types.get(k)
        if want in (int, float) and isinstance(v, str):
            try:
                out[k] = want(v.strip())
                continue
            except ValueError:
                pass
        if want is int and isinstance(v, float) and v.is_integer():
            out[k] = int(v)
            continue
        out[k] = v.strip() if isinstance(v, str) else v
    return out


def _num_eq(a, b, tol: float = 1e-9) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def _calc_terms(expression: str):
    """-> (ok, result, constants) via the shared calculator's safe AST."""
    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
        result = _tools._eval_node(tree.body)
    except Exception:
        return False, None, set()
    consts = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            consts.add(node.value)
    return True, result, consts


CANONICAL_TOOLS = ("calculator", "unit_convert", "kb_lookup")
WAREHOUSE_TOOLS = ("warehouse_query", "warehouse_update")
NO_META = {"mutated": False, "replay": False, "line": None}


def canonical_payload(tool: str, args: dict, *, kb=None, env=None):
    """The true, fault-free result envelope for one call -> (payload, meta).

    This is the ONE implementation of canonical tool semantics in the project.
    The episode runtime calls it for every dispatch, and the certification
    layer (agentlab.provenance) calls it for oracle replay, so a model can
    never see an observation the oracle replay would compute differently.

    `kb` is a KBView or a plain {key: record} mapping; `env` is a
    fulfillment WarehouseState (or None outside that family).
    """
    meta = dict(NO_META)
    if tool == "kb_lookup":
        view = kb if isinstance(kb, KBView) else KBView(kb or {})
        return view.lookup(args.get("key", "")), meta
    if tool == "calculator":
        out = _tools.calculator(str(args.get("expression", "")))
        if out.startswith("error"):
            return {"ok": False, "error": out}, meta
        return {"ok": True, "value": out}, meta
    if tool == "unit_convert":
        try:
            value = float(args.get("value"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "error: value must be numeric"}, meta
        to_u = str(args.get("to_unit", "")).strip().lower()
        out = _tools.unit_convert(value, str(args.get("from_unit", "")), to_u)
        if out.startswith("error"):
            return {"ok": False, "error": out}, meta
        return {"ok": True, "value": out, "unit": to_u}, meta
    if tool == "warehouse_query":
        if env is None:
            return {"ok": False, "error": "unknown_tool"}, meta
        return env.query(str(args.get("resource", "")), str(args.get("token", "")),
                         int(args.get("quantity", 0) or 0),
                         float(args.get("mass_kg", 0.0) or 0.0))
    if tool == "warehouse_update":
        if env is None:
            return {"ok": False, "error": "unknown_tool"}, meta
        return env.update(str(args.get("action", "")), str(args.get("token", "")),
                          int(args.get("quantity", 0) or 0))
    return {"ok": False, "error": "unknown_tool"}, meta


class EpisodeRuntime:
    """Owns one episode: spec, KB view, env state, faults, clock, trace."""

    def __init__(self, spec: TaskSpec, kb_entries: dict, nodes: list) -> None:
        self.spec = spec
        self.nodes = list(nodes)
        self._node_index = {n.node_id: i for i, n in enumerate(self.nodes)}
        self.kb = KBView(kb_entries)
        self.engine = FaultEngine(spec.faults)
        self.env = None
        if spec.family == "fulfillment":
            from .envs.fulfillment import WarehouseState

            if spec.env is None:
                raise ValueError("fulfillment spec has no env state")
            self.env = WarehouseState(spec.env)
        self.decision_id = 0
        self.call_id = 0
        self.events: list[TraceEvent] = []
        self.completed: dict[str, int] = {}   # node_id -> completing decision
        self.final_text: str | None = None
        self._secrets = set(spec.secret_tokens)
        self._revealed: set[str] = set()
        self._reveal_from(spec.prompt)
        self._reveal_from(canon(spec.start))

    # -- plumbing --------------------------------------------------------------

    def _reveal_from(self, text: str) -> None:
        if not self._secrets:
            return
        for tok in self._secrets:
            if tok in text:
                self._revealed.add(tok)

    def begin_decision(self) -> int:
        """Advance the logical clock: once per assistant decision."""
        self.decision_id += 1
        return self.decision_id

    def _state_digest(self) -> str:
        if self.env is None:
            return ""
        return digest(self.env.snapshot())

    # -- semantic matching -------------------------------------------------------

    def _match_node(self, node, name: str, args: dict) -> bool:
        if node.tool != name:
            return False
        m = node.match
        kind = m["kind"]
        if kind == "kb":
            return str(args.get("key", "")).strip().upper() == m["key"].upper()
        if kind == "convert":
            return (_num_eq(args.get("value"), m["value"])
                    and str(args.get("from_unit", "")).strip().lower() == m["from"]
                    and str(args.get("to_unit", "")).strip().lower() == m["to"])
        if kind == "calc":
            ok, result, consts = _calc_terms(args.get("expression", ""))
            if not ok or not _num_eq(result, m["result"]):
                return False
            return all(any(_num_eq(c, req) for c in consts) for req in m["required"])
        if kind == "wq":
            if str(args.get("resource", "")).strip().lower() != m["resource"]:
                return False
            if str(args.get("token", "")).strip() not in m["tokens"]:
                return False
            if "quantity" in m and not _num_eq(args.get("quantity", 0), m["quantity"]):
                return False
            if "mass_kg" in m and not _num_eq(args.get("mass_kg", 0), m["mass_kg"]):
                return False
            return True
        if kind == "wu":
            if str(args.get("action", "")).strip().lower() != m["action"]:
                return False
            if str(args.get("token", "")).strip() not in m["tokens"]:
                return False
            if "quantity" in m and not _num_eq(args.get("quantity", 0), m["quantity"]):
                return False
            return True
        raise ValueError(f"unknown matcher kind {kind!r}")

    def _semantic_match(self, name: str, args: dict):
        """-> (node, index) preferring the next incomplete node, else any."""
        next_idx = len(self.completed)
        order = list(range(len(self.nodes)))
        if next_idx < len(self.nodes):
            order.remove(next_idx)
            order.insert(0, next_idx)
        for idx in order:
            if self._match_node(self.nodes[idx], name, args):
                return self.nodes[idx], idx
        return None, None

    def _credit_eligible(self, idx: int) -> bool:
        if idx != len(self.completed):
            return False
        if idx == 0:
            return True
        prev = self.nodes[idx - 1].node_id
        return self.completed[prev] < self.decision_id

    # -- execution ----------------------------------------------------------------

    def _execute(self, name: str, args: dict):
        """-> (payload, meta) against the episode's own KB/env."""
        return canonical_payload(name, args, kb=self.kb, env=self.env)

    def _request_id(self) -> str:
        return "req-" + digest_text(f"{self.spec.task_id}:{self.call_id}")[:12]

    # -- the dispatcher --------------------------------------------------------------

    def __call__(self, name: str, arguments: dict) -> str:
        return self.dispatch(name, arguments)

    def dispatch(self, name: str, arguments: dict) -> str:
        if self.decision_id == 0:
            self.begin_decision()
        self.call_id += 1
        event_id = f"e{self.call_id}"
        args = _coerce(name, arguments or {})
        args_digest = call_args_digest(name, args)
        state_before = self._state_digest()

        # capability-token provenance (checked before this call reveals anything)
        token_known = True
        tok = args.get("token")
        if isinstance(tok, str) and tok.strip() in self._secrets:
            token_known = tok.strip() in self._revealed

        node, idx = self._semantic_match(name, args)
        credit_ok = node is not None and self._credit_eligible(idx)
        repeat = node is not None and node.node_id in self.completed
        directive = (self.engine.directive(node.node_id, self.decision_id)
                     if credit_ok else None)

        payload = None
        meta = {"mutated": False, "replay": False, "line": None}
        exposed: str
        exposed_canonical = False
        credited = False
        fault_type = None
        fault_triggered = False
        rate_limited = False
        ok = False

        if directive is None:
            payload, meta = self._execute(name, args)
            ok = bool(payload.get("ok"))
            if node is not None:
                exposed_canonical = payload == node.expect
            if credit_ok and exposed_canonical:
                self.completed[node.node_id] = self.decision_id
                credited = True
            exposed = canon({**payload, "event_id": event_id})
        else:
            kind = directive["kind"]
            fault_triggered = bool(directive.get("fault_triggered"))
            if kind == "transient":
                fault_type = "transient"
                payload = {"ok": False, "error": "transient_backend",
                           "retryable": True, "request_id": self._request_id()}
                exposed = canon({**payload, "event_id": event_id})
            elif kind == "rate_limit":
                fault_type = "rate_limit"
                rate_limited = True
                payload = {"ok": False, "error": "rate_limit",
                           "retry_after_turns": directive["retry_after_turns"],
                           "request_id": self._request_id()}
                exposed = canon({**payload, "event_id": event_id})
            elif kind == "wrong_unit":
                fault_type = "wrong_unit"
                wrong = directive["unit"]
                out = _tools.unit_convert(float(args.get("value")),
                                          str(args.get("from_unit", "")), wrong)
                payload = {"ok": True, "value": out, "unit": wrong}
                ok = True
                exposed = canon({**payload, "event_id": event_id})
            elif kind == "malformed":
                fault_type = "malformed"
                if directive.get("ambiguous") and node.mutating:
                    # The mutation happens; only its response is truncated.
                    payload, meta = self._execute(name, args)
                    if not payload.get("ok"):
                        raise RuntimeError(
                            "ambiguous malformed fault fired on a failing "
                            f"mutation: {payload}")
                    self.completed[node.node_id] = self.decision_id
                    credited = True
                else:
                    # Read-only: canonical result computed internally, withheld.
                    payload, meta = self._execute(name, args)
                exposed = MALFORMED_LITERAL
            else:
                raise ValueError(f"unknown directive {kind!r}")

        state_after = self._state_digest()
        state_mutated = bool(meta.get("mutated"))
        unsafe = state_mutated and not credited

        event = TraceEvent(
            decision_id=self.decision_id, call_id=self.call_id,
            event_id=event_id, tool=name,
            oracle_node=node.node_id if node is not None else None,
            credited=credited, repeat=repeat,
            canonical_args_digest=args_digest,
            canonical_result_digest=(digest(node.expect) if node is not None else None),
            exposed_result_digest=digest_text(exposed),
            exposed_canonical=exposed_canonical, ok=ok,
            fault_type=fault_type, fault_triggered=fault_triggered,
            rate_limited=rate_limited,
            mutating=bool(node.mutating) if node is not None
                     else name == "warehouse_update",
            state_mutated=state_mutated, replay=bool(meta.get("replay")),
            unsafe=unsafe, token_known=token_known,
            aux={k: v for k, v in (
                ("line", meta.get("line")),
                ("resource", str(args.get("resource", "")).strip().lower()
                 if name == "warehouse_query" else None),
                ("action", str(args.get("action", "")).strip().lower()
                 if name == "warehouse_update" else None),
            ) if v is not None},
            state_before=state_before, state_after=state_after,
        )
        self.events.append(event)
        self._reveal_from(exposed)
        return exposed

    # -- terminal ------------------------------------------------------------------

    def finalize_text(self, text: str) -> None:
        self.final_text = text

    # -- episode digests (the parity surface every consumer must reproduce) ----

    def progress(self) -> dict:
        """{node_id: completing decision_id} -- credited oracle progress."""
        return dict(self.completed)

    def observation_digests(self) -> list[dict]:
        """Per-call canonical/exposed digest pairs.

        `canonical` is the digest of the oracle node's canonical payload (None
        for calls that reach no node) and `exposed` is the digest of the exact
        bytes the model saw. A consumer that rebuilds this episode must
        reproduce BOTH sequences: equal canonical digests prove it ran the same
        task, equal exposed digests prove the model saw the same observations
        (including the faulted ones).
        """
        return [{"call_id": e.call_id, "decision_id": e.decision_id,
                 "tool": e.tool, "oracle_node": e.oracle_node,
                 "args": e.canonical_args_digest,
                 "canonical": e.canonical_result_digest,
                 "exposed": e.exposed_result_digest,
                 "credited": e.credited, "fault_triggered": e.fault_triggered}
                for e in self.events]

    def episode_digest(self) -> str:
        """One digest over the whole dispatch history plus oracle progress."""
        return digest({"observations": self.observation_digests(),
                       "progress": self.progress(),
                       "state": self._state_digest()})

    def verify(self, final_text: str | None = None):
        from .verify import verify_episode

        if final_text is not None:
            self.final_text = final_text
        return verify_episode(self.spec, self.nodes, self.events,
                              self.final_text or "", env=self.env)


# ---------------------------------------------------------------------------
# scripted oracle agent (validation + tests; no model involved)
# ---------------------------------------------------------------------------

def run_oracle(spec: TaskSpec, kb_entries: dict, nodes: list,
               max_attempts: int = 4):
    """Execute the oracle path through a fresh runtime, recovering from every
    scheduled fault by the accepted mechanical action; returns (runtime, verdict).
    """
    rt = EpisodeRuntime(spec, kb_entries, nodes)
    for node in nodes:
        done = False
        for _ in range(max_attempts):
            rt.begin_decision()
            exposed = rt.dispatch(node.tool, dict(node.args))
            try:
                obj = json.loads(exposed)
            except json.JSONDecodeError:
                continue  # malformed observation -> replay on the next decision
            if not obj.get("ok"):
                continue  # transient / rate_limit -> retry on the next decision
            if (node.tool == "unit_convert"
                    and str(obj.get("unit", "")).strip().lower()
                    != str(node.args["to_unit"]).strip().lower()):
                continue  # wrong-unit trap -> redo the conversion
            done = True
            break
        if not done:
            break
    rt.begin_decision()
    rt.finalize_text(f"Done. The answer is \\boxed{{{spec.answer}}}")
    return rt, rt.verify()


# ---------------------------------------------------------------------------
# replay parity: a consumer's trajectory must re-run to identical digests
# ---------------------------------------------------------------------------

def replay_trace(spec: TaskSpec, kb_entries: dict, nodes: list, calls: list):
    """Re-execute a recorded call sequence through a FRESH canonical runtime.

    `calls` is an ordered list of {"decision_id", "tool", "args"} (extra keys
    ignored) -- exactly what a rollout writes down. The replay advances the
    logical decision clock to each recorded decision before dispatching, so
    rate-limit timing and the "dependency edges need a later decision" rule are
    reproduced, not bypassed.

    Returns (runtime, report). `report` is the parity evidence:

      observations   the per-call canonical/exposed digest pairs
      progress       {node_id: decision_id} credited oracle progress
      episode        one digest over both plus terminal state

    A consumer that merely imports the canonical modules is NOT reconciled: it
    is reconciled when `replay_trace` over the calls it recorded reproduces the
    digests its own live runtime produced. `verify_replay` asserts exactly that.
    """
    rt = EpisodeRuntime(spec, kb_entries, nodes)
    for call in calls:
        want = int(call.get("decision_id") or call.get("decision") or 1)
        if want < rt.decision_id:
            raise ValueError(f"decision ids must not go backwards: {want} < "
                             f"{rt.decision_id}")
        while rt.decision_id < want:
            rt.begin_decision()
        rt.dispatch(call["tool"], dict(call.get("args") or {}))
    return rt, {"observations": rt.observation_digests(),
                "progress": rt.progress(), "episode": rt.episode_digest()}


def verify_replay(spec: TaskSpec, kb_entries: dict, nodes: list, calls: list,
                  expected: dict) -> tuple[bool, str]:
    """(ok, reason): does replaying `calls` reproduce `expected` exactly?

    `expected` is a report from `replay_trace` or a live runtime
    ({"observations", "progress", "episode"}). The first divergence is named
    precisely -- a mismatch is a faithfulness failure, never a warning.
    """
    _rt, got = replay_trace(spec, kb_entries, nodes, calls)
    # Every component is compared explicitly. Short-circuiting on the composite
    # `episode` digest would trust a number the consumer reported about itself:
    # a record whose progress map disagrees with its own digest must still fail.
    want_obs = expected.get("observations") or []
    if len(got["observations"]) != len(want_obs):
        return False, (f"replay_call_count:{len(got['observations'])}!="
                       f"{len(want_obs)}")
    for a, b in zip(got["observations"], want_obs):
        for field in ("tool", "oracle_node", "args", "canonical", "exposed",
                      "credited", "decision_id"):
            if a.get(field) != b.get(field):
                return False, (f"replay_{field}_mismatch@call{a['call_id']}:"
                               f"{a.get(field)!r}!={b.get(field)!r}")
    if got["progress"] != expected.get("progress"):
        return False, (f"replay_progress_mismatch:{got['progress']}!="
                       f"{expected.get('progress')}")
    if got["episode"] != expected.get("episode"):
        return False, "replay_state_mismatch"
    return True, ""


ENV_FACTORIES = None  # populated lazily via environment_factories()


def environment_factories() -> dict:
    """TRL-style {family: EnvClass} map; reset(**row) rebuilds an episode."""
    from .envs.fulfillment import FulfillmentEnv
    from .envs.lookup_chain import LookupChainEnv
    from .envs.typed_relay import TypedRelayEnv

    return {"lookup_chain": LookupChainEnv, "typed_relay": TypedRelayEnv,
            "fulfillment": FulfillmentEnv}
