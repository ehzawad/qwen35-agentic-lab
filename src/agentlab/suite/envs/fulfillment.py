"""Family C: fulfillment -- a non-trivial stateful warehouse environment.

Exactly two new tools, and they are environment-scoped: the global
agentlab.tools registry keeps only the canonical three. `warehouse_query` is
read-only (orders, quotes, reservation status, capability tokens);
`warehouse_update` performs idempotent reservation, cancellation and
finalization. The episode also exposes the shared calculator, unit_convert
and kb_lookup implementations.

State per episode: one order with 1-3 line items; inventory by lot; per-line
quantities and masses; budget, shipping-mass ceiling and ledger; random
quote/reservation/idempotency/status/finalize tokens; an append-only mutation
log; exactly one feasible generated allocation (distractor lots violate
inventory).

Horizons (exact required sequences):

  H4  express          query order -> query quote -> reserve -> finalize
  H8  one line         query order -> KB spec -> calculate mass -> convert to
                       kg -> query quote -> reserve -> query reservation
                       status -> finalize
  H14 two lines        query order -> six dependent operations per line -> finalize
  H20 three lines      query order -> six dependent operations per line -> finalize

A quote is issued only when the supplied quantity and converted mass match
the generated specification. A reservation decrements inventory and ledger
atomically. Replaying the same quote token is idempotent: it returns the same
reservation instead of double-decrementing.

Capability-token provenance: the finalize token is only revealed by the
reserve response (express) or by a reservation-status query once every line
is reserved (H8+), so each oracle node is causally necessary.
"""

from __future__ import annotations

import copy

from agentlab.tools import unit_convert as _unit_convert

from ..schema import OracleNode, TaskDraft

FAMILY = "fulfillment"
HORIZONS = (4, 8, 14, 20)

_LINES_FOR = {4: 1, 8: 1, 14: 2, 20: 3}
_QTY_CHOICES = (8, 16, 24, 32, 40)


def has_unit_convert(horizon: int) -> bool:
    return horizon >= 8


def is_express(horizon: int) -> bool:
    return horizon == 4


# ---------------------------------------------------------------------------
# environment-scoped tool stubs (schema source of truth; never dispatched
# globally -- EpisodeRuntime routes these names into WarehouseState)
# ---------------------------------------------------------------------------

def warehouse_query(resource: str, token: str, quantity: int = 0,
                    mass_kg: float = 0.0) -> str:
    """
    Reads warehouse state: orders, quotes, and reservation status.

    Args:
        resource: What to read: "order" (the order and its lines), "quote"
            (request a quote for one line), or "reservation" (status of one
            line's reservation).
        token: The capability token for the resource: your order token, a
            line token (for quotes), or a quote/reservation token (for
            reservation status).
        quantity: For quotes: the exact line quantity you are quoting.
        mass_kg: For quotes: the line's total shipping mass in kilograms.

    Returns:
        A JSON envelope describing the resource, or an error envelope.
    """
    raise RuntimeError("warehouse_query is environment-scoped: dispatch it "
                       "through an EpisodeRuntime, never globally")


def warehouse_update(action: str, token: str, quantity: int = 0) -> str:
    """
    Mutates warehouse state: idempotent reservation, cancellation, finalization.

    Args:
        action: One of "reserve" (turn a quote into a reservation), "cancel"
            (release a reservation), or "finalize" (complete the fully
            reserved order).
        token: The capability token: a quote token for reserve, a reservation
            token for cancel, the finalize token for finalize.
        quantity: For reserve: the exact quantity being reserved.

    Returns:
        A JSON envelope describing the new state, or an error envelope.
    """
    raise RuntimeError("warehouse_update is environment-scoped: dispatch it "
                       "through an EpisodeRuntime, never globally")


WAREHOUSE_TOOLS = (warehouse_query, warehouse_update)


def warehouse_tool_schemas() -> list[dict]:
    from transformers.utils import get_json_schema

    return [get_json_schema(fn) for fn in WAREHOUSE_TOOLS]


# ---------------------------------------------------------------------------
# state machine (shared by the generator's oracle simulation and the runtime)
# ---------------------------------------------------------------------------

class WarehouseState:
    """Mutable episode state built from a pristine spec env dict."""

    def __init__(self, env: dict) -> None:
        self.order_id = env["order_id"]
        self.order_token = env["order_token"]
        self.budget = int(env["budget"])
        self.mass_limit_kg = int(env["mass_limit_kg"])
        self.finalize_token = env["finalize_token"]
        # None under the absent-information control: the order still finalizes,
        # but the terminal completion token is withheld, so the committed answer
        # is unobtainable from any observation. Deleting a KB record cannot do
        # this -- express fulfillment reads no KB at all.
        self.completion_token = env.get("completion_token")
        self.lines = copy.deepcopy(env["lines"])
        self.inventory = {lot: dict(v) for lot, v in env["inventory"].items()}
        self.status = "open"
        self.ledger = self.budget
        self.reservations: dict[int, dict] = {}
        self.mutations: list[dict] = []

    # -- helpers -------------------------------------------------------------

    def _line_by(self, field: str, token: str):
        for ln in self.lines:
            if ln.get(field) == token:
                return ln
        return None

    def all_reserved(self) -> bool:
        return len(self.reservations) == len(self.lines)

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "ledger": self.ledger,
            "inventory": {lot: v["quantity"] for lot, v in sorted(self.inventory.items())},
            "reserved": {str(n): r["quantity"] for n, r in sorted(self.reservations.items())},
        }

    def _reserve_payload(self, ln: dict) -> dict:
        payload = {"ok": True, "line": ln["line"],
                   "reservation_token": ln["reservation_token"],
                   "lot": ln["lot"], "quantity": ln["quantity"]}
        if ln.get("express"):
            payload["finalize_token"] = self.finalize_token
        return payload

    # -- read-only -------------------------------------------------------------

    def query(self, resource: str, token: str, quantity: int = 0,
              mass_kg: float = 0.0) -> tuple[dict, dict]:
        meta = {"mutated": False, "replay": False, "line": None}
        resource = str(resource).strip().lower()
        token = str(token).strip()

        if resource == "order":
            if token != self.order_token:
                return {"ok": False, "error": "unknown_token"}, meta
            lines = []
            for ln in self.lines:
                row = {"line": ln["line"], "item": ln["item"],
                       "quantity": ln["quantity"], "line_token": ln["line_token"]}
                if ln.get("express"):
                    row["express"] = True
                    row["mass_kg"] = ln["mass_kg"]
                else:
                    row["spec_key"] = ln["spec_key"]
                lines.append(row)
            return {"ok": True, "order": self.order_id, "status": self.status,
                    "budget": self.budget, "mass_limit_kg": self.mass_limit_kg,
                    "lines": lines}, meta

        if resource == "quote":
            ln = self._line_by("line_token", token)
            if ln is None:
                return {"ok": False, "error": "unknown_token"}, meta
            meta["line"] = ln["line"]
            qty_ok = int(quantity) == ln["quantity"]
            mass_ok = (bool(ln.get("express"))
                       or abs(float(mass_kg) - float(ln["mass_kg"])) < 1e-9)
            if not (qty_ok and mass_ok):
                return {"ok": False, "error": "quote_mismatch"}, meta
            return {"ok": True, "line": ln["line"], "quote_token": ln["quote_token"],
                    "unit_price": ln["unit_price"], "total": ln["total"]}, meta

        if resource == "reservation":
            ln = (self._line_by("quote_token", token)
                  or self._line_by("reservation_token", token))
            if ln is None:
                return {"ok": False, "error": "unknown_token"}, meta
            meta["line"] = ln["line"]
            if ln["line"] not in self.reservations:
                return {"ok": False, "error": "not_reserved"}, meta
            payload = {"ok": True, "line": ln["line"], "status": "reserved",
                       "reservation_token": ln["reservation_token"],
                       "quantity": ln["quantity"]}
            if self.all_reserved():
                payload["finalize_token"] = self.finalize_token
            return payload, meta

        return {"ok": False, "error": "unknown_resource"}, meta

    # -- mutating ---------------------------------------------------------------

    def update(self, action: str, token: str, quantity: int = 0) -> tuple[dict, dict]:
        meta = {"mutated": False, "replay": False, "line": None}
        action = str(action).strip().lower()
        token = str(token).strip()

        if action == "reserve":
            ln = self._line_by("quote_token", token)
            if ln is None:
                return {"ok": False, "error": "unknown_token"}, meta
            meta["line"] = ln["line"]
            if ln["line"] in self.reservations:
                # Idempotent replay: the same reservation, no second decrement.
                meta["replay"] = True
                return self._reserve_payload(ln), meta
            lot = self.inventory.get(ln["lot"])
            if lot is None or lot["quantity"] < ln["quantity"]:
                return {"ok": False, "error": "insufficient_inventory"}, meta
            if self.ledger < ln["total"]:
                return {"ok": False, "error": "insufficient_funds"}, meta
            lot["quantity"] -= ln["quantity"]
            self.ledger -= ln["total"]
            self.reservations[ln["line"]] = {
                "quantity": ln["quantity"], "lot": ln["lot"],
                "reservation_token": ln["reservation_token"]}
            self.mutations.append({"kind": "reserve", "line": ln["line"],
                                   "lot": ln["lot"], "qty": ln["quantity"],
                                   "ledger_delta": -ln["total"]})
            meta["mutated"] = True
            return self._reserve_payload(ln), meta

        if action == "cancel":
            ln = self._line_by("reservation_token", token)
            if ln is None or ln["line"] not in self.reservations:
                return {"ok": False, "error": "unknown_token"}, meta
            meta["line"] = ln["line"]
            if self.status == "complete":
                return {"ok": False, "error": "already_complete"}, meta
            self.inventory[ln["lot"]]["quantity"] += ln["quantity"]
            self.ledger += ln["total"]
            del self.reservations[ln["line"]]
            self.mutations.append({"kind": "cancel", "line": ln["line"],
                                   "lot": ln["lot"], "qty": ln["quantity"],
                                   "ledger_delta": ln["total"]})
            meta["mutated"] = True
            return {"ok": True, "line": ln["line"], "status": "cancelled"}, meta

        if action == "finalize":
            if token != self.finalize_token:
                return {"ok": False, "error": "unknown_token"}, meta
            payload = {"ok": True, "status": "complete"}
            if self.completion_token is not None:
                payload["completion_token"] = self.completion_token
            if self.status == "complete":
                meta["replay"] = True
                return payload, meta
            if not self.all_reserved():
                return {"ok": False, "error": "incomplete_order"}, meta
            self.status = "complete"
            self.mutations.append({"kind": "finalize", "line": None, "lot": None,
                                   "qty": 0, "ledger_delta": 0})
            meta["mutated"] = True
            return payload, meta

        return {"ok": False, "error": "unknown_action"}, meta


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _convert_str(value, from_unit: str, to_unit: str) -> str:
    out = _unit_convert(float(value), from_unit, to_unit)
    if out.startswith("error"):
        raise RuntimeError(f"generator produced invalid conversion: {out}")
    return out


def generate_task(rng, horizon: int) -> TaskDraft:
    if horizon not in HORIZONS:
        raise ValueError(f"fulfillment horizon must be one of {HORIZONS}")
    n_lines = _LINES_FOR[horizon]
    express = is_express(horizon)

    order_id = "ORD-" + rng.base32(8)
    order_token = "OTK-" + rng.base32(16)
    finalize_token = "FIN-" + rng.base32(16)
    completion_token = "CMP-" + rng.base32(16)

    lines: list[dict] = []
    inventory: dict[str, dict] = {}
    kb: dict = {}
    used: set = set()

    def fresh(prefix: str, chars: int) -> str:
        while True:
            t = prefix + rng.base32(chars)
            if t not in used:
                used.add(t)
                return t

    used_uq: set = set()
    for i in range(1, n_lines + 1):
        while True:
            u = rng.randint(1, 16)
            q = rng.choice(_QTY_CHOICES)
            if (u, q) not in used_uq and (u * q) not in {a * b for a, b in used_uq}:
                used_uq.add((u, q))
                break
        unit_mass_g = 125 * u
        total_g = unit_mass_g * q
        mass_kg = total_g // 1000  # q is a multiple of 8, so this is exact
        assert mass_kg * 1000 == total_g
        price = rng.randint(3, 60)
        item = fresh("ITM-", 6)
        lot = fresh("LOT-", 8)
        distractor_lot = fresh("LOT-", 8)
        line = {
            "line": i, "item": item, "quantity": q,
            "unit_mass_g": unit_mass_g, "mass_kg": mass_kg,
            "unit_price": price, "total": q * price,
            "line_token": fresh("LIN-", 16),
            "quote_token": fresh("QTE-", 16),
            "reservation_token": fresh("RSV-", 16),
            "lot": lot, "express": express,
        }
        if not express:
            line["spec_key"] = fresh("SPEC-", 12)
            kb[line["spec_key"]] = {"item": item, "unit_mass_g": unit_mass_g}
        lines.append(line)
        inventory[lot] = {"item": item, "quantity": q + rng.randint(0, 6)}
        # Distractor lot: violates the inventory constraint by construction.
        inventory[distractor_lot] = {"item": item, "quantity": rng.randint(0, q - 1)}

    budget = sum(ln["total"] for ln in lines) + rng.randint(0, 30)
    mass_limit = sum(ln["mass_kg"] for ln in lines) + rng.randint(1, 25)

    env = {
        "order_id": order_id, "order_token": order_token,
        "budget": budget, "mass_limit_kg": mass_limit,
        "finalize_token": finalize_token, "completion_token": completion_token,
        "lines": lines, "inventory": inventory,
    }

    # -- oracle nodes, with expects captured by simulating the real state ----
    sim = WarehouseState(env)
    nodes: list[OracleNode] = []
    n = 0

    def add(tool: str, args: dict, match: dict, mutating: bool = False) -> None:
        nonlocal n
        n += 1
        if tool == "warehouse_query":
            payload, _ = sim.query(**args)
        elif tool == "warehouse_update":
            payload, _ = sim.update(**args)
        elif tool == "kb_lookup":
            payload = {"ok": True, "record": kb[args["key"]]}
        elif tool == "calculator":
            payload = {"ok": True, "value": str(match["result"])}
        elif tool == "unit_convert":
            payload = {"ok": True,
                       "value": _convert_str(args["value"], args["from_unit"],
                                             args["to_unit"]),
                       "unit": args["to_unit"]}
        else:
            raise ValueError(tool)
        if not payload.get("ok"):
            raise RuntimeError(f"oracle simulation failed at {tool}: {payload}")
        nodes.append(OracleNode(node_id=f"n{n}", tool=tool, args=args,
                                expect=payload, match=match, mutating=mutating))

    add("warehouse_query", {"resource": "order", "token": order_token},
        {"kind": "wq", "resource": "order", "tokens": [order_token]})

    for ln in lines:
        if not express:
            m, q = ln["unit_mass_g"], ln["quantity"]
            add("kb_lookup", {"key": ln["spec_key"]},
                {"kind": "kb", "key": ln["spec_key"]})
            add("calculator", {"expression": f"{m} * {q}"},
                {"kind": "calc", "required": [m, q], "result": m * q})
            conv = _convert_str(m * q, "g", "kg")
            if conv != str(ln["mass_kg"]):
                raise RuntimeError(f"mass conversion mismatch: {conv} != {ln['mass_kg']}")
            add("unit_convert",
                {"value": m * q, "from_unit": "g", "to_unit": "kg"},
                {"kind": "convert", "value": m * q, "from": "g", "to": "kg"})
        quote_args = {"resource": "quote", "token": ln["line_token"],
                      "quantity": ln["quantity"]}
        quote_match = {"kind": "wq", "resource": "quote",
                       "tokens": [ln["line_token"]], "quantity": ln["quantity"]}
        if not express:
            quote_args["mass_kg"] = ln["mass_kg"]
            quote_match["mass_kg"] = ln["mass_kg"]
        add("warehouse_query", quote_args, quote_match)
        add("warehouse_update",
            {"action": "reserve", "token": ln["quote_token"],
             "quantity": ln["quantity"]},
            {"kind": "wu", "action": "reserve", "tokens": [ln["quote_token"]],
             "quantity": ln["quantity"]},
            mutating=True)
        if not express:
            add("warehouse_query",
                {"resource": "reservation", "token": ln["reservation_token"]},
                {"kind": "wq", "resource": "reservation",
                 "tokens": [ln["quote_token"], ln["reservation_token"]]})

    add("warehouse_update", {"action": "finalize", "token": finalize_token},
        {"kind": "wu", "action": "finalize", "tokens": [finalize_token]},
        mutating=True)

    if len(nodes) != horizon:
        raise RuntimeError(f"fulfillment built {len(nodes)} nodes for H{horizon}")
    if sim.status != "complete":
        raise RuntimeError("oracle simulation did not complete the order")

    env["oracle_final"] = sim.snapshot()

    secret_tokens = [order_token, finalize_token, completion_token]
    for ln in lines:
        secret_tokens += [ln["line_token"], ln["quote_token"], ln["reservation_token"]]

    return TaskDraft(
        prompt_fields={"order_id": order_id, "order_token": order_token,
                       "express": express},
        kb=kb, nodes=nodes, answer=completion_token, answer_kind="token",
        start={"order_id": order_id, "order_token": order_token},
        env=env, secret_tokens=secret_tokens,
    )


def redact_absent(draft: TaskDraft) -> dict:
    """Withhold the terminal warehouse completion token; return the descriptor.

    The hidden value of a fulfillment episode is not in the KB -- express orders
    read no KB record at all -- it is the token the finalize call hands back. So
    the redaction removes it from the environment and from the finalize node's
    canonical envelope: the order still completes, and the answer is nowhere in
    the episode. `draft.answer` deliberately keeps the withheld token, which is
    what the leakage check scores against.
    """
    terminal = draft.nodes[-1]
    if terminal.tool != "warehouse_update" or terminal.args.get("action") != "finalize":
        raise RuntimeError("fulfillment absent draft does not end in finalize")
    token = draft.env.get("completion_token")
    if not token:
        raise RuntimeError("fulfillment absent draft has no completion token")
    draft.env["completion_token"] = None
    terminal.expect = {k: v for k, v in terminal.expect.items()
                       if k != "completion_token"}
    draft.secret_tokens = [t for t in draft.secret_tokens if t != token]
    return {"kind": "env_completion_token", "target": "env.completion_token",
            "field": "completion_token",
            "hidden_entropy_bits": 16 * 5,   # 16 base32 characters
            "why": "the finalize envelope no longer carries the completion "
                   "token, and no KB record ever held it"}


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

_FULL_CORE = (
    "Fetch the order with warehouse_query(resource='order', token=<order "
    "token>). For each line in order: kb_lookup its spec_key to read "
    "unit_mass_g; use calculator to compute unit_mass_g * quantity (total "
    "grams); unit_convert that total to kg; request a quote with "
    "warehouse_query(resource='quote', token=<line_token>, quantity=<exact "
    "quantity>, mass_kg=<total kg>); reserve with warehouse_update("
    "action='reserve', token=<quote_token>, quantity=<quantity>); then "
    "confirm with warehouse_query(resource='reservation', "
    "token=<reservation_token>). Once every line is reserved the status "
    "check reveals finalize_token; call warehouse_update(action='finalize', "
    "token=<finalize_token>) and commit the returned completion token as "
    "\\boxed{{token}}. Calls can occasionally fail or come back garbled; "
    "recover and complete the order."
)

_EXPRESS_CORE = (
    "Fetch the order with warehouse_query(resource='order', token=<order "
    "token>); its single line is express, so the shipping mass is already "
    "computed. Request a quote with warehouse_query(resource='quote', "
    "token=<line_token>, quantity=<exact quantity>); reserve with "
    "warehouse_update(action='reserve', token=<quote_token>, "
    "quantity=<quantity>) -- the reserve response carries finalize_token; "
    "then warehouse_update(action='finalize', token=<finalize_token>) and "
    "commit the returned completion token as \\boxed{{token}}. Calls can "
    "occasionally fail or come back garbled; recover and complete the order."
)


def _mk_templates(core: str) -> tuple:
    return (
        # 0-7: training
        f"You are the fulfillment operator for order {{order_id}}. Your order "
        f"token is {{order_token}}. {core}",
        f"Complete warehouse order {{order_id}}. Order access token: "
        f"{{order_token}}. {core}",
        f"Order {{order_id}} is yours to fulfill; authenticate with token "
        f"{{order_token}}. {core}",
        f"Process order {{order_id}} end to end. Use order token "
        f"{{order_token}}. {core}",
        f"You must fulfill order {{order_id}} (token {{order_token}}). {core}",
        f"Handle the open order {{order_id}} using access token "
        f"{{order_token}}. {core}",
        f"Your assignment: close out order {{order_id}}. The order token is "
        f"{{order_token}}. {core}",
        f"Take order {{order_id}} from open to complete; token: "
        f"{{order_token}}. {core}",
        # 8-9: development
        f"As warehouse agent, drive order {{order_id}} to completion. Order "
        f"token: {{order_token}}. {core}",
        f"Order {{order_id}} awaits fulfillment; present token "
        f"{{order_token}}. {core}",
        # 10-11: evaluation
        f"Fulfill warehouse order {{order_id}}. Authenticate every order read "
        f"with token {{order_token}}. {core}",
        f"You are dispatching order {{order_id}}; its access token is "
        f"{{order_token}}. {core}",
    )


TEMPLATES_FULL = _mk_templates(_FULL_CORE)
TEMPLATES_EXPRESS = _mk_templates(_EXPRESS_CORE)


def template_text(template_id: int, fields: dict) -> str:
    """The raw paraphrase template behind a rendered prompt.

    Express (H4) and full (H8+) orders carry different instruction cores, so
    their structural template identity differs at the same template_id -- which
    is what the split-leakage checker should see.
    """
    templates = TEMPLATES_EXPRESS if fields.get("express") else TEMPLATES_FULL
    return templates[template_id]


def render_prompt(template_id: int, fields: dict) -> str:
    return template_text(template_id, fields).format(
        order_id=fields["order_id"], order_token=fields["order_token"])


class FulfillmentEnv:
    """TRL-facing environment: reset(**row) rebuilds the episode exactly."""

    FAMILY = FAMILY

    def __init__(self) -> None:
        self.runtime = None

    def reset(self, *, spec: dict, kb: dict, nodes: list, secret: bytes | None = None,
              **_ignored):
        from ..contract import load_or_create_secret
        from ..runtime import EpisodeRuntime
        from ..schema import OracleNode, TaskSpec

        # The run secret keys the recovery tokens and receipts the model sees, so
        # a TRL-side reset takes it too; it defaults to the one run secret rather
        # than to a per-environment value.
        self.runtime = EpisodeRuntime(
            TaskSpec.from_row(spec), kb, [OracleNode.from_row(n) for n in nodes],
            secret=secret if secret is not None else load_or_create_secret())
        return self.runtime
