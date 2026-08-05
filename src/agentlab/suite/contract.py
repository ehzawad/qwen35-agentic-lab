"""The ONE environment contract, and its digest.

Every artifact this study produces -- prompt-tournament rows, raw
rejection-sampling shards, accepted records, SFT views, evaluation traces --
carries `environment_contract_sha256`. That digest is the answer to a question
this repository previously could not answer: *which environment did the model
face when these tokens were produced?*

It exists because the tree used to hold TWO environments. The training path ran a
tokenless `FaultEngine` through `suite.runtime.EpisodeRuntime`; the claim-bearing
evaluator ran a token-bearing `FaultInjector` through `suite.evaluate.SpecRuntime`
with its own tool schemas, its own error envelopes, its own receipt suffix, its
own transcript shape and its own recovery predicate. Both imported "the canonical
modules" and both passed their own tests. The digest below is what makes that
failure mode mechanically visible: it is computed from the model-visible surface
itself -- the exact tool-schema bytes, the observation form, every fault
envelope, the remediation predicate and the budget formulas -- so a run under the
old contract can never be mistaken for, resumed into or pooled with a run under
this one.

`invalidate` therefore is not a nicety. Any artifact whose stamp is absent or
stale was produced under a different environment and is discarded rather than
resumed.
"""

from __future__ import annotations

import functools
import os
import pathlib

from . import faults as _faults
from .schema import CELLS, FAMILIES, SUITE_NAME, SUITE_VERSION, canon, digest_text

STAMP_FIELD = "environment_contract_sha256"

# Bumped whenever the model-visible environment changes. `token-bearing-v1` is
# the D2 reconciliation: one runtime, one tool schema, one event format, one
# certified-success predicate.
CONTRACT_VERSION = "token-bearing-v1"

CONDITION_FAULTS = {"clean": 0, "faulted": 1, "stress": 2}


def budgets_for(horizon: int, condition: str) -> dict:
    """The registered episode budgets: H+3 clean / H+5 one fault / H+8 stress,
    tool calls capped at 2H+4 everywhere. ONE definition, read by the generator,
    the training path and the evaluator."""
    from .schema import call_budget, decision_budget

    if condition not in CONDITION_FAULTS:
        raise ValueError(f"unknown condition {condition!r}")
    return {"max_decisions": decision_budget(horizon, CONDITION_FAULTS[condition]),
            "max_calls": call_budget(horizon)}


def spec_for_condition(spec, condition: str):
    """The committed TaskSpec re-budgeted for an evaluation condition.

    `clean` drops the committed faults; `faulted` keeps one; `stress` keeps both.
    Budgets follow `budgets_for`, so an evaluated episode and a trained episode of
    the same task and condition are budgeted identically -- which the parity test
    asserts rather than assumes.
    """
    import dataclasses

    want = CONDITION_FAULTS[condition]
    kept = list(spec.faults)[:want] if want else []
    if len(kept) < want:
        raise ValueError(
            f"{spec.task_id}: condition {condition!r} needs {want} committed "
            f"faults, the spec carries {len(spec.faults)}. A condition is never "
            f"satisfied by inventing a fault the generator did not commit.")
    budgets = budgets_for(spec.horizon, condition)
    return dataclasses.replace(spec, faults=kept,
                               max_decisions=budgets["max_decisions"],
                               max_calls=budgets["max_calls"])


# ---------------------------------------------------------------------------
# the contract itself
# ---------------------------------------------------------------------------

def environment_contract() -> dict:
    """The full machine-readable description of the model-visible environment."""
    from .runtime import RECEIPT_LINE_PREFIX, tool_schema_bytes

    return {
        "contract_version": CONTRACT_VERSION,
        "suite": {"name": SUITE_NAME, "version": SUITE_VERSION,
                  "families": list(FAMILIES),
                  "cells": [[f, h] for f, h in CELLS]},
        "tool_surface": {
            "token_argument": _faults.TOKEN_ARG,
            "token_property": dict(_faults.TOKEN_PROPERTY),
            "declared_on_every_tool": True,
            "schemas": {family: tool_schema_bytes(family) for family in FAMILIES},
        },
        "observation_form": {
            "envelope": "canonical JSON, sorted keys, compact separators",
            "receipt_line_prefix": RECEIPT_LINE_PREFIX,
            "receipt_on_every_observation": True,
            "event_id_exposed": False,
            "request_id_exposed": False,
            "receipt": "HMAC-SHA256(run_secret, task_id|call_id|sha256(envelope))"
                       "[:32 hex], prefixed r-",
        },
        "fault_envelopes": {
            "transient": _faults.transient_envelope("<token>"),
            "rate_limit": _faults.rate_limit_envelope("<token>", 1),
            "rate_limit_active": _faults.rate_limit_active_envelope("<token>", 1),
            "malformed": _faults.malformed_envelope("<token>"),
            "wrong_unit": "valid success envelope in the COMMITTED "
                          "FaultSpec.params['wrong_unit']; no recovery token",
        },
        "remediation": {cls: _faults.remediation_requirement(cls)
                        for cls in _faults.FAULT_CLASSES},
        "recovery_predicate": (
            "the certifying event must ITSELF satisfy remediation and result: "
            "same stripped call identity with the exact emitted token (transient, "
            "malformed); additionally a strictly later decision (rate_limit); a "
            "later unit_convert explicitly requesting the original target unit "
            "(wrong_unit); a token-bearing idempotent replay for the ambiguous "
            "malformed mutation. A canonical result reached without a qualifying "
            "event is blind_retry and is never certified."),
        "success_predicate": (
            "certified_success = committed exact answer AND every oracle node "
            "completed in order AND every dependency crossed a later assistant "
            "decision AND runtime/verifier credit maps agree AND the terminal "
            "answer came from a validated observation AND the receipt chain is "
            "valid AND no hallucinated result AND no unsafe mutation AND "
            "capability tokens were observed before use AND the fulfillment final "
            "state equals oracle_final AND calls <= max_calls AND decisions <= "
            "max_decisions AND no wall/parser/call-cap termination AND no runaway "
            "AND every assigned fault has a certifying recovery event"),
        "budgets": {"decisions": "H+3 clean, H+5 one fault, H+8 stress",
                    "calls": "2H+4",
                    "per_condition": {c: {"formula": budgets_for(8, c)}
                                      for c in CONDITION_FAULTS}},
        "event_schema": _event_field_names(),
    }


def _event_field_names() -> list[str]:
    import dataclasses

    from .schema import TraceEvent

    return [f.name for f in dataclasses.fields(TraceEvent)]


@functools.lru_cache(maxsize=1)
def environment_contract_sha256() -> str:
    """The stamp. Pure function of the contract above; no configuration input."""
    return digest_text(canon(environment_contract()))


def stamp(row: dict | None = None) -> dict:
    """`{environment_contract_sha256: ...}`, or `row` with it added in place."""
    if row is None:
        return {STAMP_FIELD: environment_contract_sha256()}
    row[STAMP_FIELD] = environment_contract_sha256()
    return row


def is_current(row: dict) -> bool:
    return (row or {}).get(STAMP_FIELD) == environment_contract_sha256()


def require_current(row: dict, what: str) -> None:
    """Refuse to reuse an artifact produced under a different environment."""
    got = (row or {}).get(STAMP_FIELD)
    if got == environment_contract_sha256():
        return
    raise SystemExit(
        f"REFUSED: {what} carries "
        f"{'no' if not got else 'a stale'} {STAMP_FIELD}"
        f"{'' if not got else f' ({got})'}; this build's environment contract is "
        f"{environment_contract_sha256()}.\n"
        f"  An artifact produced under a different model-visible environment "
        f"cannot be resumed into, pooled with or compared against this one. "
        f"Regenerate it. (This is the D2 invalidation rule: the tokenless "
        f"environment is gone, and resume-by-task-id must not smuggle its "
        f"artifacts back in.)")


def invalidate(rows: list[dict], what: str = "artifact") -> tuple[list, list]:
    """-> (current, invalid). Rows without the current stamp are NOT usable."""
    current = [r for r in rows if is_current(r)]
    invalid = [r for r in rows if not is_current(r)]
    return current, invalid


# ---------------------------------------------------------------------------
# the run secret
# ---------------------------------------------------------------------------

def default_secret_path(cfg: dict | None = None) -> pathlib.Path:
    from .configio import ROOT

    return ROOT / "out" / "agentic" / "run_secret.hex"


def load_or_create_secret(path: str | pathlib.Path | None = None) -> bytes:
    """The run secret for receipts and recovery tokens.

    Lives OUTSIDE the committed tree (out/), is created ONCE per run, and is
    shared by the prompt tournament, rejection sampling, view construction and
    evaluation. It used to live in `suite.evaluate`, which meant only the
    evaluator had one -- and therefore only the evaluator could mint tokens at
    all. Every consumer now threads the same value into every `EpisodeRuntime`.
    """
    p = pathlib.Path(path) if path is not None else default_secret_path()
    if p.exists():
        return bytes.fromhex(p.read_text().strip())
    secret = os.urandom(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(secret.hex() + "\n", encoding="utf-8")
    return secret
