"""EpisodeRuntime: per-episode KB view, state, fault schedule and trace.

THE ONE RUNTIME. Every consumer -- rejection sampling, the prompt tournament,
the variance probe, SFT view construction, generation validation and the
claim-bearing held-out evaluation -- dispatches through this class. The parallel
`suite.evaluate.SpecRuntime` is gone; that fork is the D2 defect this module
closes, and `tests/test_environment_parity.py` is the test that keeps it closed.

Faults are never implemented by making the three global tools
nondeterministic: agentlab.tools stays pure, and every episode binds its own
KB view, stateful environment (fulfillment), fault schedule, logical decision
clock, mutation log, oracle progress and trace events.

Dispatch contract: `runtime(name, arguments) -> str`, suitable for
`chat.run_agent_loop(..., dispatcher=runtime)`. kb_lookup misses return only
no_entry -- never a key list.

THE ONE MODEL-VISIBLE OBSERVATION FORM (registered):

    <canonical or faulted envelope, canonical JSON>
    receipt: r-<32 hex>

  * the envelope carries NO `event_id` and no `request_id`. Call and event ids
    stay in the hidden ledger (`TraceEvent`); exposing them was an unregistered
    training-path field.
  * EVERY observation, including a clean one, carries a receipt line. Receipts
    are registered, prompt candidate `p5_provenance` instructs the model about
    them, and the tournament that selects a prompt now really shows them.
  * a faulted envelope for transient / rate_limit / malformed additionally
    carries the 128-bit `recovery_token` and a `remediation` block. The
    `recovery_token` tool argument is declared on EVERY tool schema
    (`tool_schemas_for_family`), parsed and stripped in exactly one place
    (`EpisodeRuntime.dispatch`), and never reaches canonical tool semantics,
    oracle matching or the semantic call digest -- it is remediation evidence,
    not a tool-domain parameter.

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

from . import faults as _faults
from .faults import (MALFORMED_LITERAL, TOKEN_ARG, TOKEN_PROPERTY, FaultEngine,
                     parse_token_from_args, strip_token)
from .kb import KBView
from .schema import (ARG_TYPES, TaskSpec, TraceEvent, call_args_digest, canon,
                     coerce_tool_args, digest, digest_text, normalize_number)

READ_ONLY_TOOLS = ("calculator", "unit_convert", "kb_lookup", "warehouse_query")

RECEIPT_LINE_PREFIX = "receipt: "

# The declared argument types live in `suite.schema` (ARG_TYPES), with the data
# contract, because call IDENTITY depends on them and `call_args_digest` has to
# apply the same coercion this dispatcher does.
_ARG_TYPES = ARG_TYPES


def tool_schemas_for_family(family: str) -> list[dict]:
    """The ONE model-visible tool surface: the canonical three schemas, plus the
    two warehouse tools for fulfillment, plus the optional `recovery_token`
    argument on every one of them.

    The token argument is declared on EVERY tool because the model does not know
    in advance which tool will fault, and it is declared HERE -- in the one
    function the training path, the generator and the evaluator all call --
    because the previous arrangement (evaluation locally augmenting its own copy)
    is precisely how the trained policy came to be trained against a tool surface
    the certifier does not score.

    Each schema is deep-copied before augmentation: `agentlab.tools.tool_schemas`
    hands out shared dicts and mutating them would leak the argument into every
    unrelated caller.
    """
    schemas = _tools.tool_schemas()
    if family == "fulfillment":
        from .envs.fulfillment import warehouse_tool_schemas

        schemas = schemas + warehouse_tool_schemas()
    out = []
    for schema in schemas:
        schema = json.loads(json.dumps(schema))
        params = schema["function"].setdefault("parameters", {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})[TOKEN_ARG] = dict(TOKEN_PROPERTY)
        out.append(schema)
    return out


def tool_schema_bytes(family: str) -> str:
    """Canonical JSON of a family's whole tool surface: the parity comparand."""
    return canon(tool_schemas_for_family(family))


def tool_names_for_family(family: str) -> list[str]:
    """The legal tool surface of a family, derived FROM the schemas.

    Names and schemas cannot drift apart because there is only one source: a
    rollout engine that rejects a call as `unknown_tool` uses exactly the names
    the model was shown.
    """
    return [s["function"]["name"] for s in tool_schemas_for_family(family)]


def _coerce(name: str, raw: dict) -> dict:
    """ONE coercion, shared with `schema.call_args_digest`."""
    return coerce_tool_args(name, raw)


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
    """Owns one episode: spec, KB view, env state, faults, receipts, clock, trace.

    `secret` is the RUN SECRET and is required, not optional. Recovery tokens and
    receipts are HMACs keyed with it, so it is part of the model-visible bytes: a
    consumer that supplied its own would produce a different episode from the one
    the certifier scores. `suite.contract.load_or_create_secret` creates it once
    per run and every consumer threads the same value through.
    """

    def __init__(self, spec: TaskSpec, kb_entries: dict, nodes: list, *,
                 secret: bytes) -> None:
        if not isinstance(secret, (bytes, bytearray)) or not secret:
            raise ValueError(
                "EpisodeRuntime requires the run secret (bytes): recovery tokens "
                "and receipts are keyed with it, so it is part of the "
                "model-visible observation bytes")
        self.spec = spec
        self.secret = bytes(secret)
        self.nodes = list(nodes)
        self._node_index = {n.node_id: i for i, n in enumerate(self.nodes)}
        self.kb = KBView(kb_entries)
        self.engine = FaultEngine(spec.faults, task_id=spec.task_id,
                                  secret=self.secret)
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

    def _receipt(self, call_id: int, exposed_digest: str) -> str:
        from agentlab.provenance import mint_receipt

        return mint_receipt(self.secret, self.spec.task_id, call_id, exposed_digest)

    def model_visible(self, exposed_text: str, receipt: str) -> str:
        """The exact bytes a tool message carries: envelope, newline, receipt."""
        return f"{exposed_text}\n{RECEIPT_LINE_PREFIX}{receipt}"

    # -- the dispatcher --------------------------------------------------------------

    def __call__(self, name: str, arguments: dict) -> str:
        return self.dispatch(name, arguments)

    def dispatch(self, name: str, arguments: dict) -> str:
        if self.decision_id == 0:
            self.begin_decision()
        self.call_id += 1
        event_id = f"e{self.call_id}"
        # ONE place parses and strips the remediation argument. Everything below
        # -- coercion, semantic matching, canonical semantics, the call digest --
        # sees only the stripped tool-domain arguments, so echoing a token can
        # never change what a call MEANS, only whether remediation is certified.
        raw_args = dict(arguments or {})
        token_provided = parse_token_from_args(raw_args)
        args = _coerce(name, strip_token(raw_args))
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
        emitted_token = None
        ok = False

        # The true, fault-free semantic payload for these exact arguments,
        # recorded as a digest on every event whether or not the model saw it, so
        # "the canonical observation was exposed" stays decidable for calls that
        # reach no oracle node. It is computed by EXECUTING only when the call
        # really executes (no directive, or a malformed fault, whose whole point
        # is that the effect happens and the response is truncated) or when the
        # tool is read-only. A transient / rate-limit / wrong-unit fault on a
        # MUTATING call must not perform the mutation, so its canonical digest is
        # deliberately absent rather than obtained by a side effect.
        executes = directive is None or directive["kind"] == "malformed"
        canonical_payload_here = canonical_meta = None
        if executes:
            canonical_payload_here, canonical_meta = self._execute(name, args)
        elif name in READ_ONLY_TOOLS:
            canonical_payload_here, _ro_meta = self._execute(name, args)
        canonical_semantic_digest = (digest(canonical_payload_here)
                                     if canonical_payload_here is not None else None)

        if directive is None:
            payload, meta = canonical_payload_here, canonical_meta
            ok = bool(payload.get("ok"))
            if node is not None:
                exposed_canonical = payload == node.expect
            if credit_ok and exposed_canonical:
                self.completed[node.node_id] = self.decision_id
                credited = True
            exposed = canon(payload)
        else:
            kind = directive["kind"]
            fault_triggered = bool(directive.get("fault_triggered"))
            emitted_token = directive.get("token")
            if kind == "transient":
                fault_type = "transient"
                exposed = _faults.transient_envelope(emitted_token)
            elif kind == "rate_limit":
                fault_type = "rate_limit"
                rate_limited = True
                exposed = _faults.rate_limit_envelope(
                    emitted_token, directive["retry_after_turns"])
            elif kind == "rate_limit_active":
                # The SAME scheduled fault still in force within one decision --
                # never a second firing, so `fault_triggered` stays False.
                fault_type = "rate_limit"
                rate_limited = True
                exposed = _faults.rate_limit_active_envelope(
                    emitted_token, directive["retry_after_turns"])
            elif kind == "wrong_unit":
                fault_type = "wrong_unit"
                wrong = directive["unit"]
                out = _tools.unit_convert(float(args.get("value")),
                                          str(args.get("from_unit", "")), wrong)
                payload = {"ok": True, "value": out, "unit": wrong}
                ok = True
                exposed = canon(payload)
            elif kind == "malformed":
                fault_type = "malformed"
                if directive.get("ambiguous") and node.mutating:
                    # The mutation happens; only its response is truncated.
                    payload, meta = canonical_payload_here, canonical_meta
                    if not payload.get("ok"):
                        raise RuntimeError(
                            "ambiguous malformed fault fired on a failing "
                            f"mutation: {payload}")
                    self.completed[node.node_id] = self.decision_id
                    credited = True
                else:
                    # Read-only: canonical result computed internally, withheld.
                    payload, meta = canonical_payload_here, canonical_meta
                exposed = _faults.malformed_envelope(emitted_token)
            else:
                raise ValueError(f"unknown directive {kind!r}")

        state_after = self._state_digest()
        state_mutated = bool(meta.get("mutated"))
        unsafe = state_mutated and not credited
        exposed_digest = digest_text(exposed)
        receipt = self._receipt(self.call_id, exposed_digest)
        visible = self.model_visible(exposed, receipt)

        event = TraceEvent(
            decision_id=self.decision_id, call_id=self.call_id,
            event_id=event_id, tool=name,
            oracle_node=node.node_id if node is not None else None,
            credited=credited, repeat=repeat,
            canonical_args_digest=args_digest,
            canonical_result_digest=(digest(node.expect) if node is not None else None),
            canonical_semantic_digest=canonical_semantic_digest,
            exposed_text=exposed, exposed_result_digest=exposed_digest,
            receipt=receipt, model_visible_digest=digest_text(visible),
            token_provided=token_provided, recovery_token=emitted_token,
            requested_unit=(str(args.get("to_unit", "")).strip().lower()
                            if name == "unit_convert" else None),
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
        return visible

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
                 # The receipt is part of the bytes the model read, so wire drift
                 # in the receipt line is a replay failure and not a warning.
                 "visible": e.model_visible_digest,
                 "token_provided": e.token_provided,
                 "credited": e.credited, "fault_triggered": e.fault_triggered}
                for e in self.events]

    def episode_digest(self) -> str:
        """One digest over the whole dispatch history plus oracle progress."""
        return digest({"observations": self.observation_digests(),
                       "progress": self.progress(),
                       "state": self._state_digest()})

    def verify(self, final_text: str | None = None, *, transcript: list | None = None,
               termination_reason: str | None = None):
        from .verify import verify_episode

        if final_text is not None:
            self.final_text = final_text
        return verify_episode(self.spec, self.nodes, self.events,
                              self.final_text or "", env=self.env,
                              secret=self.secret, transcript=transcript,
                              termination_reason=termination_reason)


# ---------------------------------------------------------------------------
# reading one model-visible observation back
# ---------------------------------------------------------------------------

def parse_observation(text: str) -> dict:
    """-> {"objects": [...], "receipt": str|None, "truncated_prefix": bool}.

    The one reader of the registered wire format. A policy (scripted or trained)
    sees envelope lines plus a `receipt:` line; this splits them without any
    module re-deriving the layout.
    """
    objects, receipt, truncated = [], None, False
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(RECEIPT_LINE_PREFIX):
            receipt = line[len(RECEIPT_LINE_PREFIX):].strip()
            continue
        if not line.startswith("{"):
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            if line == MALFORMED_LITERAL:
                truncated = True
    return {"objects": objects, "receipt": receipt, "truncated_prefix": truncated}


def recovery_token_in(text: str) -> str | None:
    """The recovery token a faulted observation offered, or None."""
    for obj in parse_observation(text)["objects"]:
        if isinstance(obj, dict) and obj.get("ok") is False and obj.get(TOKEN_ARG):
            return str(obj[TOKEN_ARG])
    return None


# ---------------------------------------------------------------------------
# scripted oracle agent (validation + tests; no model involved)
# ---------------------------------------------------------------------------

def run_oracle(spec: TaskSpec, kb_entries: dict, nodes: list,
               max_attempts: int = 4, *, secret: bytes):
    """Execute the oracle path through a fresh runtime, recovering from every
    scheduled fault by the REGISTERED remediation action; returns (runtime, verdict).

    Registered remediation, not a bare retry: an error envelope's
    `recovery_token` is echoed back on the re-issued call (transient, malformed,
    and rate_limit -- the latter necessarily on a later decision, because this
    agent takes one decision per attempt), and a wrong-unit trap is repaired by
    re-issuing the conversion for the originally requested unit. A blind retry
    would still often succeed operationally and would NOT be certified, which is
    the whole point of the token.
    """
    rt = EpisodeRuntime(spec, kb_entries, nodes, secret=secret)
    for node in nodes:
        done = False
        pending_token = None
        for _ in range(max_attempts):
            rt.begin_decision()
            args = dict(node.args)
            if pending_token is not None:
                args[TOKEN_ARG] = pending_token
            visible = rt.dispatch(node.tool, args)
            parsed = parse_observation(visible)
            pending_token = recovery_token_in(visible)
            if pending_token is not None:
                continue  # transient / rate_limit / malformed -> reissue with token
            obj = next((o for o in parsed["objects"] if isinstance(o, dict)), None)
            if obj is None or not obj.get("ok"):
                continue
            if (node.tool == "unit_convert"
                    and str(obj.get("unit", "")).strip().lower()
                    != str(node.args["to_unit"]).strip().lower()):
                continue  # wrong-unit trap -> redo the conversion, same target
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

def replay_trace(spec: TaskSpec, kb_entries: dict, nodes: list, calls: list, *,
                 secret: bytes):
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
    rt = EpisodeRuntime(spec, kb_entries, nodes, secret=secret)
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
                  expected: dict, *, secret: bytes) -> tuple[bool, str]:
    """(ok, reason): does replaying `calls` reproduce `expected` exactly?

    `expected` is a report from `replay_trace` or a live runtime
    ({"observations", "progress", "episode"}). The first divergence is named
    precisely -- a mismatch is a faithfulness failure, never a warning.
    """
    _rt, got = replay_trace(spec, kb_entries, nodes, calls, secret=secret)
    # Every component is compared explicitly. Short-circuiting on the composite
    # `episode` digest would trust a number the consumer reported about itself:
    # a record whose progress map disagrees with its own digest must still fail.
    want_obs = expected.get("observations") or []
    if len(got["observations"]) != len(want_obs):
        return False, (f"replay_call_count:{len(got['observations'])}!="
                       f"{len(want_obs)}")
    for a, b in zip(got["observations"], want_obs):
        for field in ("tool", "oracle_node", "args", "canonical", "exposed",
                      "visible", "token_provided", "credited", "decision_id"):
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
