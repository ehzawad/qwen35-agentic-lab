"""Multifaceted rejection sampling: batched offline vLLM, resumable shards.

Rolls the base model against the committed suite v1 task specs through the
CANONICAL episode runtime (`agentlab.suite.runtime.EpisodeRuntime`), keeps only
trajectories the strict verifier accepts, and re-verifies EVERY accepted
trajectory by exact replay against the committed spec before it may enter the
SFT corpus.

One environment stack, three consumers:

  * this module               offline rejection sampling
  * agentlab.variance         the GRPO grip probe
  * agentlab.suite.evaluate   held-out certification (through the spec adapter
                              `suite.generate.certification_spec`)

Nothing here builds tasks: `suite.generate.load_bundles` reads the committed
specs/kb/oracles, and `EpisodeRuntime` is the only thing that executes a call.
Acceptance is the strict verifier's verdict plus the preregistered budget and
recovery-timing filters -- never a per-family re-implementation of "did it
work".

Faithfulness (the reason this module exists in this shape): every rollout
records the exact call sequence it made, the exposed observation bytes it saw,
the canonical/exposed digest pairs, and the credited oracle progress. `finalize`
replays that call sequence through a FRESH runtime built from the committed spec
and refuses the trajectory unless the digests, the progress map, the exposed
bytes and the whole verdict row come back identical. A trajectory that merely
"ran" is not accepted.

Sharding contract: ONE engine, MANY shards. Engine startup measured 289.7 s on
the registered A5000, so `run` builds the engine once and feeds it every pending
shard: 50 shards each paying their own startup was 4.02 GPU-hours of pure model
loading. A shard is still an atomic, resumable client-side work unit at 48
variants (<=384 rollouts at k<=8, ~150 rollouts/min measured) so a kill costs at
most one shard -- but it no longer pays for an engine. The engine start is charged
to the ledger as its own row and each shard is charged its own decode minutes, so
the two never overlap. `finalize` is CPU-only.

COMPLETION IS A RECEIPT, NOT A PATH. A shard is done when `shard-NNNN.receipt.
json` validates against the PLAN: the exact rollout ids it owed (task id x sample
index), their count, the current environment contract, one producer identity, and
the digest of the row bytes on disk. A file holding three of 384 rollouts, a file
written under the retired contract and a file planned at another --shard-size are
all re-rolled instead of resumed. The same rule ends the stage: a quota miss or a
partial finalize writes `rs_finalize_failure.json`, REMOVES anything a resume
could trust, and exits nonzero -- it never leaves an `accepted.jsonl` behind for
the next invocation to skip past. `accepted.receipt.json` is the completion marker
the view builder demands before it reads a single trajectory.

Elicitation-control ordering: production sampling REQUIRES the frozen winner
written by `agentlab.prompt_control finalize` (half of every variant's attempts
use the preregistered neutral prompt, half the frozen winning prompt file).
Running production RS before the prompt control is frozen is the exact mistake
the previous headline made; it is refused here.

Subcommands:
  plan      print the shard table for a split
  run       GPU: roll out one shard through vLLM
  finalize  CPU: verifier + replay parity + quotas -> accepted.jsonl
  status    shards done/pending, acceptance so far
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from agentlab.chat import (assistant_tool_message, numeric_answer,
                          parse_tool_calls, strip_thinking)
from agentlab.suite import contract as contract_mod
from agentlab.suite import runtime as rt_mod
from agentlab.suite.configio import (ROOT, ledger_append, ledger_guard, load_config,
                                     now_utc)
from agentlab.suite.generate import load_bundles
from agentlab.suite.schema import canon, digest_text

MULTIFACE_DIR = ROOT / "data" / "multiface"
RAW_DIR = MULTIFACE_DIR / "raw"
ACCEPTED_PATH = MULTIFACE_DIR / "accepted.jsonl"
SUMMARY_PATH = MULTIFACE_DIR / "rs_summary.json"
# Why a finalize refused, written whenever one does: the operator needs the
# reason on disk, and the ABSENCE of accepted.jsonl is what stops the chain.
FAILURE_PATH = MULTIFACE_DIR / "rs_finalize_failure.json"

SHARD_RECEIPT_KIND = "agentlab_rs_shard_receipt"
ACCEPTED_RECEIPT_KIND = "agentlab_accepted_corpus_receipt"
RECEIPT_VERSION = 1
RECEIPT_HASH_FIELD = "receipt_sha256"

_BOXED_ANY = re.compile(r"\\boxed")


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def committed_answer(final_text) -> str | None:
    """What this final turn COMMITTED, read by the one grammar in the repo.

    Delegates to `suite.schema.extract_committed_answer`, exactly as the strict
    verifier and the SFT view builder do. A local `\\boxed{}`-only test here
    rejected certified-successful trajectories that obeyed the preregistered
    system prompt's `ANSWER: <value>` form, which is how they left the corpus.
    """
    from agentlab.suite.schema import extract_committed_answer

    return extract_committed_answer(str(final_text or ""))


def answers_match(got, want) -> bool:
    """Numeric answers match within the preregistered hybrid tolerance;
    non-numeric answers (lookup-chain access codes) must match exactly."""
    if got is None or want is None:
        return False
    g, w = numeric_answer(got), numeric_answer(want)
    if g is not None and w is not None:
        return abs(g - w) <= max(1e-4, 1e-6 * abs(w))
    return str(got).strip() == str(want).strip()


def suite_dir(cfg: dict | None = None) -> pathlib.Path:
    cfg = cfg or load_config()
    return ROOT / cfg["suite"]["data_dir"]


def load_split(split: str, cfg: dict | None = None) -> list:
    """The committed bundles of one suite split (the only task source)."""
    return load_bundles(str(suite_dir(cfg)), split)


def rs_split(cfg: dict | None = None) -> str:
    return (cfg or load_config())["suite"]["rs_source"]


def load_frozen_prompt(cfg: dict | None = None) -> str:
    """The tournament-winning system prompt, verified against its committed hash.

    There is ONE winner (the BP/TP arms' prompt), not one per family: the frozen
    preregistration selects a single prompt file across the three claim axes.
    """
    from agentlab.prompt_control import frozen_winner

    return frozen_winner(cfg)["prompt"]


# --------------------------------------------------------------------------
# the provenance chain: rollout -> accepted -> SFT view -> trainer -> lock
# --------------------------------------------------------------------------
#
# Every row of every claim-bearing training artifact answers ONE question with a
# single grammar: *what produced these bytes?* The grammar is the frozen S19
# fingerprint (configio.FINGERPRINT_FIELDS) plus four fields:
#
#   gpu_execution            True only when a process that OWNED the card wrote
#                            this block. Never inferred, never defaulted.
#   producer                 who wrote it (a stage name, or the scripted policy)
#   runtime_manifest_sha256  the producer session's attestation digest
#   session_id               which producer session
#
# The two legal shapes are exhaustive, and `None` is not one of them:
#
#   a GPU producer   gpu_execution=True, a COMPLETE fingerprint (no nulls), and a
#                    runtime manifest digest + session id to point at.
#   a CPU producer   gpu_execution=False, an explicit `producer`, and the four
#                    card-identity fields EXPLICITLY null. A CPU transformation
#                    never inherits a card from an old hardware lock, because
#                    "there is a lock on disk" is not evidence that this row was
#                    computed on that card (council: stages with no visible
#                    device must not appear to have run on a GPU).
#
# `provenance: None` -- what rejection sampling used to write on the scripted
# path, and what the SFT views dropped entirely -- is refused at the WRITE, so an
# unattributable row never reaches a corpus a trainer could consume.

GPU_EXECUTION = "gpu_execution"

# The card-identity fields a CPU attestation must leave null.
_CARD_FIELDS = ("gpu_name", "gpu_uuid", "cuda_visible_bytes", "driver_version")


def cpu_provenance(producer: str, cfg: dict | None = None,
                   run_id: str | None = None) -> dict:
    """An EXPLICIT non-GPU attestation: same field set, card identity nulled.

    The scripted CPU engines the tests inject, and every CPU-only transformation,
    use this instead of `None`. It keeps the frozen field set (so one reader
    handles every block) while stating out loud that no card produced the row.
    """
    from agentlab.suite.configio import fingerprint

    fp = fingerprint(run_id, cfg)
    for key in _CARD_FIELDS:
        fp[key] = None
    fp[GPU_EXECUTION] = False
    fp["producer"] = str(producer)
    fp["runtime_manifest_sha256"] = None
    fp["session_id"] = None
    return fp


def provenance_gaps(block) -> list[str]:
    """Why this provenance block is not evidence. Empty means the chain holds."""
    from agentlab.suite.configio import fingerprint_gaps

    if not isinstance(block, dict) or not block:
        return ["provenance_absent"]
    if GPU_EXECUTION not in block:
        return [GPU_EXECUTION]
    if block[GPU_EXECUTION]:
        gaps = list(fingerprint_gaps(block))
        gaps += [k for k in ("runtime_manifest_sha256", "session_id")
                 if not block.get(k)]
        return gaps
    gaps = [] if block.get("producer") else ["producer"]
    gaps += [f"{k}_on_a_cpu_attestation" for k in _CARD_FIELDS if block.get(k)]
    return gaps


def require_row_provenance(block, what: str = "a claim-bearing row") -> dict:
    """Refuse to WRITE a row whose producer is unknown or half-recorded."""
    gaps = provenance_gaps(block)
    if gaps:
        raise SystemExit(
            f"REFUSED: {what} would carry provenance that is not evidence "
            f"({', '.join(gaps)}).\n"
            f"  A GPU producer must write a COMPLETE S19 fingerprint plus the "
            f"digest and session id of its runtime manifest; a CPU producer must "
            f"say `{GPU_EXECUTION}: false` and name itself, with the card "
            f"identity explicitly null. `provenance: null` is neither: it is an "
            f"artifact nobody can attribute, and it may not enter a corpus a "
            f"trainer consumes or a checkpoint lock pins.")
    return block


def provenance_identity(block: dict) -> dict:
    """What must be IDENTICAL across one corpus (session and clock excluded).

    Two rejection-sampling sessions of one run on one card are the resumable
    design and must pool; two cards, two engine fingerprints or two effective
    thinking modes inside one corpus are the S19 failure itself.
    """
    from agentlab.suite.configio import fingerprint_identity

    block = block or {}
    ident = {GPU_EXECUTION: bool(block.get(GPU_EXECUTION))}
    if ident[GPU_EXECUTION]:
        ident.update(fingerprint_identity(block))
    else:
        ident.update({"producer": block.get("producer"),
                      "run_id": block.get("run_id"),
                      "config_hash": block.get("config_hash"),
                      "engine_fingerprint": block.get("engine_fingerprint")})
    return ident


def distinct_producers(rows: list) -> list[dict]:
    """The distinct producer identities across rows, in first-seen order."""
    seen, out = set(), []
    for row in rows:
        ident = provenance_identity((row or {}).get("provenance"))
        key = canon(ident)
        if key not in seen:
            seen.add(key)
            out.append(ident)
    return out


def require_one_producer(rows: list, what: str) -> dict:
    """Refuse a corpus assembled from more than one producer identity."""
    idents = distinct_producers(rows)
    if len(idents) > 1:
        raise SystemExit(
            f"FATAL: {what} mixes {len(idents)} producer identities:\n"
            + "\n".join(f"  {canon(i)}" for i in idents)
            + "\n  One claim may not span two cards, two engine fingerprints or "
              "two effective thinking modes (S19). A second producer is a NEW "
              "run with its own run_id, locks, seeds, ledger and declaration.")
    return idents[0] if idents else {}


def row_digest(rec: dict) -> str:
    """Content digest of one artifact row: what the next layer points back at."""
    return digest_text(canon(rec))


# --------------------------------------------------------------------------
# rollout engine (generation backend injected; CPU tests script it)
# --------------------------------------------------------------------------

class RolloutEngine:
    """Multi-turn rollouts over committed task bundles on the canonical runtime.

    `render_fn(messages, tool_schemas) -> str` and
    `generate_fn(list[str]) -> list[(text, finish_reason)]` are injected so the
    same engine serves vLLM production runs, the prompt tournament, the
    variance probe, and scripted CPU tests.
    """

    def __init__(self, cfg: dict, render_fn, generate_fn,
                 frozen_prompt: str | None = None, provenance: dict | None = None,
                 secret: bytes | None = None):
        self.cfg = cfg
        self.render = render_fn
        self.generate = generate_fn
        self.frozen = frozen_prompt
        # THE RUN SECRET. Recovery tokens and receipts are keyed with it, so it is
        # part of the model-visible observation bytes. It defaults to the one run
        # secret in out/ rather than to nothing, because a rollout engine without a
        # secret could not mint the tokens the certifier requires -- which is
        # exactly the state the tokenless training path was in.
        self.secret = (bytes(secret) if secret is not None
                       else contract_mod.load_or_create_secret())
        # The S19 fingerprint of the engine that produced every row this engine
        # emits: which card, which driver, which engine settings, which effective
        # thinking mode. A scripted CPU engine gets an EXPLICIT non-GPU
        # attestation rather than the `None` it used to carry: a row whose
        # producer is unrecorded is indistinguishable from a row produced by an
        # engine nobody attested, and the SFT corpus, the trainer manifest and
        # the checkpoint lock all point back at this block.
        self.provenance = require_row_provenance(
            provenance if provenance is not None
            else cpu_provenance("scripted-cpu-policy", cfg),
            "every rollout this engine records")
        # Set by `_vllm_engine` for a GPU producer: the session attestation this
        # engine's provenance was COPIED from, so the ledger row that charges the
        # session's minutes carries the same identity as its rollouts.
        self.manifest_path = None
        self.manifest = None
        self._schemas: dict = {}
        self._names: dict = {}

    def _schema(self, family: str):
        if family not in self._schemas:
            self._schemas[family] = rt_mod.tool_schemas_for_family(family)
        return self._schemas[family]

    def _legal(self, family: str) -> set:
        if family not in self._names:
            self._names[family] = set(rt_mod.tool_names_for_family(family))
        return self._names[family]

    def _k(self, spec) -> int:
        return self.cfg["mixture"][spec.family]["k"][f"h{spec.horizon}"]

    def system_prompt(self, prompt_variant: str) -> str:
        """The neutral preregistered prompt, or the frozen tournament winner.

        Half of every variant's attempts use each, so the RS corpus is not
        implicitly conditioned on one elicitation.
        """
        from agentlab.prompt_control import CANONICAL_SYSTEM

        if prompt_variant != "frozen" or not self.frozen:
            return CANONICAL_SYSTEM
        return self.frozen

    def new_rollout(self, bundle, sample_index: int, prompt_variant: str) -> dict:
        spec = bundle.spec
        runtime = rt_mod.EpisodeRuntime(spec, bundle.kb, bundle.nodes,
                                        secret=self.secret)
        return {
            "bundle": bundle, "runtime": runtime,
            "sample_index": sample_index, "prompt_variant": prompt_variant,
            "messages": [{"role": "system",
                          "content": self.system_prompt(prompt_variant)},
                         {"role": "user", "content": spec.prompt}],
            # The episode budget is the one committed in the spec (H+3 clean,
            # H+5 single-fault, H+8 stress); the verifier enforces the same
            # number, so a rollout can never be accepted for exceeding it.
            "max_decisions": spec.max_decisions,
            "decisions_used": 0, "done": False, "truncated": False,
            "exhausted": False, "unknown_tool": False, "arg_error": False,
            "call_cap": False, "final": "", "calls": [], "call_map": [],
        }

    def rollouts_for(self, bundles: list, k_override: int | None = None,
                     variants: tuple = ("canonical", "frozen")) -> list:
        convos = []
        for bundle in bundles:
            k = k_override if k_override is not None else self._k(bundle.spec)
            for j in range(k):
                if len(variants) == 1:
                    variant = variants[0]
                else:
                    variant = "canonical" if j < (k + 1) // 2 else "frozen"
                convos.append(self.new_rollout(bundle, j, variant))
        return convos

    def run(self, convos: list, verbose: bool = True) -> list:
        turn = 0
        while True:
            active = [c for c in convos
                      if not c["done"] and c["decisions_used"] < c["max_decisions"]]
            if not active:
                break
            prompts = [self.render(c["messages"],
                                  self._schema(c["bundle"].spec.family))
                       for c in active]
            if verbose:
                print(f"[rs] decision {turn}: {len(active)} active rollouts", flush=True)
            outs = self.generate(prompts)
            for c, (text, finish_reason) in zip(active, outs):
                self._step(c, text, finish_reason)
            turn += 1
        for c in convos:
            if not c["done"]:
                c["exhausted"] = True
                c["final"] = ""
        return [self._record(c) for c in convos]

    def _step(self, c: dict, text: str, finish_reason: str) -> None:
        runtime = c["runtime"]
        # One logical decision per assistant turn: the fault clock, the
        # rate-limit contract and the "dependency edge needs a LATER decision"
        # rule all key off this counter, so it advances here and nowhere else.
        decision = runtime.begin_decision()
        c["decisions_used"] += 1
        calls = parse_tool_calls(text)
        if not calls:
            c["done"] = True
            c["truncated"] = finish_reason == "length"
            clean = strip_thinking(text).strip()
            c["final"] = "" if c["truncated"] else clean
            c["messages"].append({"role": "assistant", "content": clean})
            return

        # ONE assistant-message shape, shared with the evaluator.
        msg = assistant_tool_message(text, calls)
        assistant_idx = len(c["messages"])
        c["messages"].append(msg)
        spec = c["bundle"].spec
        legal = self._legal(spec.family)
        for x in calls:
            name, args = x["name"], dict(x["arguments"])
            if len(runtime.events) >= spec.max_calls:
                c["call_cap"] = True
                break
            if name not in legal:
                # Never dispatched: an unknown tool must not enter the ledger,
                # or replay would have to reproduce a call the runtime rejects.
                c["unknown_tool"] = True
                result = canon({"ok": False, "error": "unknown_tool", "tool": name})
                call_id = None
            else:
                result = runtime.dispatch(name, args)
                call_id = runtime.events[-1].call_id
                c["calls"].append({"call_id": call_id, "decision_id": decision,
                                   "tool": name, "args": args, "exposed": result})
            tool_idx = len(c["messages"])
            c["messages"].append({"role": "tool", "name": name, "content": result})
            c["call_map"].append({"call_id": call_id,
                                  "assistant_msg_index": assistant_idx,
                                  "tool_msg_index": tool_idx})

    def _record(self, c: dict) -> dict:
        from agentlab.suite.schema import digest_text

        bundle, runtime = c["bundle"], c["runtime"]
        spec = bundle.spec
        verdict = runtime.verify(c["final"], transcript=c["messages"],
                                 termination_reason=_termination_reason(c))
        fault = _fault_summary(runtime, spec, c["call_map"], verdict)
        return {
            "task_id": spec.task_id, "family": spec.family, "split": spec.split,
            "horizon": spec.horizon, "template_id": spec.template_id,
            "fault_types": [f.fault_type for f in spec.faults],
            "sample_index": c["sample_index"],
            "prompt_variant": c["prompt_variant"],
            "answer": spec.answer, "answer_kind": spec.answer_kind,
            "messages": c["messages"], "final": c["final"],
            "decisions_used": c["decisions_used"], "truncated": c["truncated"],
            "exhausted": c["exhausted"], "unknown_tool": c["unknown_tool"],
            "arg_error": c["arg_error"], "call_cap": c["call_cap"],
            "calls": c["calls"], "call_map": c["call_map"],
            "events": [e.to_row() for e in runtime.events],
            "fault": fault,
            "verdict": verdict.to_row(),
            "milestone_fraction": round(verdict.milestone_fraction, 6),
            # The parity contract: `finalize` must reproduce all three.
            "parity": {"observations": runtime.observation_digests(),
                       "progress": runtime.progress(),
                       "episode": runtime.episode_digest()},
            # Which model-visible environment produced these bytes (D2). Resume
            # logic can never reuse a tokenless shard after this fix. The tool
            # surface is already inside the contract digest; it is carried
            # separately as well so a reader can answer "which schemas was this
            # rolled against" without recomputing the whole contract.
            contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
            "tool_schema_sha256": digest_text(
                rt_mod.tool_schema_bytes(spec.family)),
            # Which card and which engine produced this row (S19). Carried on the
            # ROW, not only in a nearby ledger: a row that cannot say what
            # produced it cannot support a same-card claim. Verified at
            # construction, so an incomplete block cannot be serialized here.
            "provenance": dict(self.provenance),
        }


def _termination_reason(c: dict) -> str | None:
    """The runner's own account of why this rollout stopped, for the verifier.

    `call_cap`, a length-truncated final turn and an exhausted decision budget are
    runaway criteria the verifier must see; it cannot infer them from the event
    ledger, and inferring `call_cap` from `len(events) == max_calls` is exactly the
    equality-cap mistake the ruling forbids.
    """
    if c.get("call_cap"):
        return "call_cap"
    if c.get("truncated"):
        return "token_budget"
    if c.get("exhausted"):
        return "decision_budget"
    return "answered"


def _fault_summary(runtime, spec, call_map: list, verdict) -> dict | None:
    """Where the scheduled fault fired and where CERTIFIED recovery happened.

    Read off the canonical verifier's registered remediation report, never off
    model text and never re-derived here. This function used to decide recovery
    itself -- any later `exposed_canonical` event at the faulted node, plus a
    reservation-status query for the ambiguous mutation -- which is a third
    definition of recovery alongside the verifier's and the certifier's. The SFT
    view builder reads `recovery_msg_index`, so a looser definition here would
    have trained the model on decisions the certifier calls `blind_retry`.
    """
    if not spec.faults:
        return None
    events = runtime.events
    fired = next((e for e in events if e.fault_triggered), None)
    by_call = {m["call_id"]: m for m in call_map}
    out = {"fired": fired is not None, "result_msg_index": None,
           "recovery_msg_index": None, "fault_decision": None,
           "recovery_decision": None, "post_fault_retries": 0,
           "recovery_reason": None}
    if fired is None:
        return out
    report = next((r for r in verdict.fault_reports
                   if r["target_node"] == fired.oracle_node), None) or {}
    out["fault_decision"] = fired.decision_id
    out["recovery_reason"] = report.get("reason")
    out["result_msg_index"] = (by_call.get(fired.call_id) or {}).get("tool_msg_index")
    out["post_fault_retries"] = len([e for e in events
                                     if e.call_id > fired.call_id
                                     and e.oracle_node == fired.oracle_node])
    call_id = report.get("recovery_call_id")
    if call_id is not None:
        out["recovery_decision"] = report.get("recovery_decision")
        out["recovery_msg_index"] = (by_call.get(call_id) or {}).get(
            "assistant_msg_index")
    return out


# --------------------------------------------------------------------------
# exact replay verification (the faithfulness gate)
# --------------------------------------------------------------------------

def replay_record(rec: dict, bundle, *, secret: bytes) -> tuple[bool, str]:
    """Re-execute a rollout's calls against a FRESH canonical runtime.

    Three things must come back identical, or the record does not describe the
    committed task and must never be trained on:

      1. the exposed observation BYTES, call by call (what the model saw);
      2. the canonical/exposed digest pairs and the credited oracle progress
         (`suite.runtime.verify_replay`);
      3. the entire verifier verdict row (strict success, node decisions,
         recovery flags, budgets, state).
    """
    from agentlab.suite.schema import digest_text

    if bundle.spec.task_id != rec["task_id"]:
        return False, f"replay_wrong_bundle:{bundle.spec.task_id}"
    calls = rec["calls"]
    contract_mod.require_current(rec, f"rollout record for {rec['task_id']}")
    ok, why = rt_mod.verify_replay(bundle.spec, bundle.kb, bundle.nodes, calls,
                                   rec["parity"], secret=secret)
    if not ok:
        return False, why
    runtime, _report = rt_mod.replay_trace(bundle.spec, bundle.kb, bundle.nodes,
                                           calls, secret=secret)
    if len(runtime.events) != len(calls):
        return False, f"replay_call_count:{len(runtime.events)}!={len(calls)}"
    for i, (event, call) in enumerate(zip(runtime.events, calls)):
        # `model_visible_digest` is SHA-256 of the WHOLE tool message the model
        # read -- envelope plus receipt line -- so this IS a byte comparison
        # against what the model was shown, receipt included. Comparing only the
        # envelope would let receipt drift through unnoticed.
        if digest_text(call.get("exposed", "")) != event.model_visible_digest:
            return False, f"replay_observation_bytes@{i}"
    verdict = runtime.verify(rec["final"], transcript=rec["messages"],
                              termination_reason=_termination_reason(rec)).to_row()
    if verdict != rec["verdict"]:
        diff = sorted(k for k in verdict if verdict[k] != rec["verdict"].get(k))
        return False, f"replay_verdict_mismatch:{diff}"
    return True, ""


# --------------------------------------------------------------------------
# acceptance filters
# --------------------------------------------------------------------------

def _duplicate_calls(events: list) -> int:
    seen = set()
    dups = 0
    for e in events:
        key = (e["tool"], e["canonical_args_digest"])
        if key in seen:
            dups += 1
        seen.add(key)
    return dups


def _call_slack(rec: dict, cfg: dict) -> int:
    slack = cfg["acceptance"]["call_upper_slack"]
    extra = slack["faulted_extra"] if rec["fault_types"] else 0
    return slack[rec["family"]] + extra


def _accept_budget(rec: dict, cfg: dict) -> str:
    v = rec["verdict"]
    h = rec["horizon"]
    if not (h <= v["calls"] <= h + _call_slack(rec, cfg)):
        return "call_count"
    if _duplicate_calls(rec["events"]) > cfg["acceptance"]["max_redundant_calls"][rec["family"]]:
        return "too_many_redundant_calls"
    return ""


def _accept_recovery(rec: dict, cfg: dict) -> str:
    """Recovery must be CERTIFIED, prompt, and must not launder the trap value.

    "Certified" is the verifier's registered remediation predicate and nothing
    else: token echoed on the same call identity, on a later decision for
    rate_limit, corrected target unit for wrong_unit, token-bearing idempotent
    replay for the ambiguous mutation. This filter used to accept
    `verdict["recovered"]` under the OLD, weaker definition (any later canonical
    observation), so the SFT corpus could contain trajectories the claim-bearing
    certifier labels `blind_retry`.
    """
    fault = rec.get("fault") or {}
    if not rec["fault_types"]:
        return ""
    if not fault.get("fired"):
        return "fault_not_fired"
    if not rec["verdict"]["recovered"]:
        return "recovery_not_verified"
    if fault.get("recovery_reason") != "ok":
        return f"recovery_{fault.get('recovery_reason')}"
    rcfg = cfg["acceptance"]["recovery"]
    fd, rd = fault.get("fault_decision"), fault.get("recovery_decision")
    if fd is None or rd is None or rd - fd > rcfg["max_decisions_after_fault"]:
        return "recovery_too_late"
    if fault.get("post_fault_retries", 0) > rcfg["max_identical_retries"]:
        return "too_many_retries"
    if "wrong_unit" in rec["fault_types"]:
        # The remediation contract for a wrong-unit trap ("corrected target
        # required") is enforced by the ONE verifier, whose `recovered` flag this
        # function already required above. The only thing left to check here is
        # the acceptance-specific hazard: a trap value that happens to BE the
        # committed answer, which would let an unrecovered episode look correct.
        trapped = _trapped_value(rec)
        if trapped is not None and answers_match(trapped, rec["answer"]):
            return "trapped_value_used"
    return ""


def _trapped_value(rec: dict):
    """The number the wrong-unit trap returned, read off the exposed bytes.

    Downstream misuse of that number cannot make a trajectory succeed: the
    canonical matcher only credits the next node when the correct value feeds it,
    so a trajectory that consumed the trap fails the verifier. The one residual
    risk is a trap value that happens to BE the committed answer, which would let
    an unrecovered episode look correct; that is what is checked.
    """
    idx = (rec.get("fault") or {}).get("result_msg_index")
    if idx is None:
        return None
    try:
        content = rec["messages"][idx]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    for obj in rt_mod.parse_observation(content)["objects"]:
        if isinstance(obj, dict) and "value" in obj:
            return obj["value"]
    return None


def accept_record(rec: dict, cfg: dict | None = None, bundles: dict | None = None,
                  skip_replay: bool = False, *, secret: bytes | None = None
                  ) -> tuple[bool, str]:
    """(accepted, rejection_reason).

    Order: cheap universal filters, then the STRICT VERIFIER (the single
    definition of task success), then the preregistered budget/recovery filters,
    then replay parity. The verifier is not re-implemented here.
    """
    cfg = cfg or load_config()
    gaps = provenance_gaps(rec.get("provenance"))
    if gaps:
        # An accepted row must retain the EXACT producer snapshot of the rollout
        # it came from, because the SFT view, the trainer manifest and the
        # checkpoint lock all inherit it. A row that cannot say what produced it
        # is dropped here rather than laundered into the corpus.
        return False, f"missing_provenance:{gaps[0]}"
    if rec["exhausted"]:
        return False, "max_decisions_exhausted"
    if rec["truncated"]:
        return False, "truncated_final"
    if rec["unknown_tool"]:
        return False, "unknown_tool"
    if rec["arg_error"]:
        return False, "bad_call_arguments"
    if rec["call_cap"]:
        return False, "call_cap_hit"
    if not rec["final"]:
        return False, "no_final"
    if "</think>" in rec["final"]:
        return False, "stray_think"
    if committed_answer(rec["final"]) is None:
        # Commitment discipline, asked with the ONE shared grammar
        # (`schema.extract_committed_answer`): the preregistered system prompt asks
        # for `ANSWER: <value>` and the generated task prompt asks for `\boxed{}`,
        # and either is a commitment. This filter used to demand `\boxed{}`
        # specifically, so a trajectory the strict verifier had CERTIFIED --
        # terminating `ANSWER: 55640a29...`, which is what obeying the system
        # prompt alone produces -- was rejected here as `no_box` and never reached
        # the view builder at all. Answer CORRECTNESS is still decided by the
        # verifier below and nowhere else.
        return False, "no_committed_answer"
    max_chars = cfg["scenario"]["tool_output_max_chars"]
    for m in rec["messages"]:
        if m.get("role") != "tool":
            continue
        content = str(m.get("content", ""))
        if _BOXED_ANY.search(content):
            return False, "answer_leakage"
        if len(content) > max_chars:
            return False, "tool_output_too_long"

    verdict = rec["verdict"]
    if not verdict["consistent"]:
        return False, "verifier_runtime_disagreement"
    if not verdict["certified_success"]:
        return False, "verifier_rejected"

    why = _accept_budget(rec, cfg) or _accept_recovery(rec, cfg)
    if why:
        return False, why

    if not skip_replay:
        bundle = (bundles or {}).get(rec["task_id"])
        if bundle is None:
            return False, "replay_bundle_missing"
        if secret is None:
            secret = contract_mod.load_or_create_secret()
        ok, why = replay_record(rec, bundle, secret=secret)
        if not ok:
            return False, why
    return True, ""


# --------------------------------------------------------------------------
# shards
# --------------------------------------------------------------------------

def plan_shards(split: str | None = None, shard_size: int = 48,
                cfg: dict | None = None) -> list[dict]:
    """The shard table, with the EXACT work each shard owes.

    `k` and `expected_rollouts` are part of the plan because "this shard is done"
    is answered against them: a shard file holding three rollouts of the 384 it
    owes exists exactly like a complete one.
    """
    cfg = cfg or load_config()
    split = split or rs_split(cfg)
    bundles = load_split(split, cfg)
    groups: dict[tuple, list] = {}
    for b in bundles:
        groups.setdefault((b.spec.family, b.spec.horizon), []).append(b)
    shards = []
    for (family, horizon) in sorted(groups):
        block = groups[(family, horizon)]
        k = int(cfg["mixture"][family]["k"][f"h{horizon}"])
        for i in range(0, len(block), shard_size):
            chunk = block[i:i + shard_size]
            task_ids = [b.spec.task_id for b in chunk]
            shards.append({"index": len(shards), "family": family, "horizon": horizon,
                           "split": split, "task_ids": task_ids, "k": k,
                           "expected_rollouts": k * len(task_ids)})
    return shards


def _shard_path(index: int) -> pathlib.Path:
    return RAW_DIR / f"shard-{index:04d}.jsonl"


def shard_receipt_path(index: int) -> pathlib.Path:
    return RAW_DIR / f"shard-{index:04d}.receipt.json"


def _write_jsonl(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _read_jsonl(path: pathlib.Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_json(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def write_attested_jsonl(path: pathlib.Path, rows: list, what: str) -> pathlib.Path:
    """Write claim-bearing rows only after the whole batch is attributable.

    The gate is at the WRITE, not in the row builder alone: an unattributable row
    on disk is indistinguishable from one produced by an engine nobody checked,
    and every later layer (acceptance, views, trainer manifest, checkpoint lock)
    points back at exactly these bytes.

    A ZERO-ROW batch is refused for the same reason: an empty file is a path that
    exists, and every resumer in this pipeline reads a path that exists as work
    that was done. "Nothing was produced" is a stop, not an artifact.
    """
    if not rows:
        raise SystemExit(
            f"REFUSED: {what} would be written with zero rows. An empty artifact "
            f"is indistinguishable from a finished one to every resume check in "
            f"this pipeline, so nothing-produced must stop the stage instead of "
            f"leaving a file behind.")
    for i, row in enumerate(rows):
        require_row_provenance(row.get("provenance"), f"{what} row {i}")
    require_one_producer(rows, what)
    _write_jsonl(path, rows)
    return path


def file_sha256(path) -> str:
    """SHA-256 over one file's bytes: what a receipt points at."""
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def seal_receipt(payload: dict) -> dict:
    """Add the self-hash that makes a receipt tamper-evident."""
    rec = {k: payload[k] for k in payload if k != RECEIPT_HASH_FIELD}
    rec[RECEIPT_HASH_FIELD] = digest_text(canon(rec))
    return rec


def receipt_seal_ok(rec: dict) -> bool:
    if not isinstance(rec, dict) or not rec.get(RECEIPT_HASH_FIELD):
        return False
    body = {k: rec[k] for k in rec if k != RECEIPT_HASH_FIELD}
    return digest_text(canon(body)) == rec[RECEIPT_HASH_FIELD]


def _load_receipt(path: pathlib.Path) -> dict | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return rec if isinstance(rec, dict) else None


def expected_rollout_ids(shard: dict) -> list[str]:
    """The exact (task_id, sample_index) pairs one shard owes, as sorted ids."""
    k = int(shard["k"])
    return sorted(f"{t}#{j}" for t in shard["task_ids"] for j in range(k))


def rollout_id(row: dict) -> str:
    return f"{row.get('task_id')}#{row.get('sample_index')}"


def shard_rows_gaps(shard: dict, rows: list) -> list[str]:
    """Why these rows are not the complete work of this shard. Empty means done.

    Checked against the PLAN, not against themselves: the exact rollout ids
    (task id x sample index) the shard owes, all of them, none extra, every row
    under the current environment contract, one producer identity.
    """
    gaps = []
    if not rows:
        return ["zero_rows"]
    stale = [r for r in rows if not contract_mod.is_current(r)]
    if stale:
        gaps.append(f"{len(stale)}_rows_under_another_environment_contract")
    want = expected_rollout_ids(shard)
    got = [rollout_id(r) for r in rows]
    if len(got) != len(want):
        gaps.append(f"{len(got)}_rollouts_of_{len(want)}")
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing:
        gaps.append(f"missing_rollouts:{','.join(missing[:4])}"
                    + (f"+{len(missing) - 4}" if len(missing) > 4 else ""))
    if extra:
        gaps.append(f"unplanned_rollouts:{','.join(extra[:4])}"
                    + (f"+{len(extra) - 4}" if len(extra) > 4 else ""))
    if len(set(got)) != len(got):
        gaps.append("duplicate_rollout_ids")
    if len(distinct_producers(rows)) > 1:
        gaps.append("more_than_one_producer_identity")
    return gaps


def build_shard_receipt(shard: dict, rows: list, path: pathlib.Path,
                        cfg: dict | None = None) -> dict:
    """The receipt that makes this shard's completion checkable.

    It states what the shard OWED (the planned task ids, the sample count, the
    expected rollout ids and their digest), what it actually holds, and the digest
    of the bytes that hold it. Resume then asks the receipt, and the receipt is
    bound to the file: a truncated or re-planned shard cannot satisfy it.
    """
    cfg = cfg or load_config()
    want = expected_rollout_ids(shard)
    return seal_receipt({
        "kind": SHARD_RECEIPT_KIND,
        "schema_version": RECEIPT_VERSION,
        "index": int(shard["index"]),
        "family": shard["family"],
        "horizon": int(shard["horizon"]),
        "split": shard.get("split") or rs_split(cfg),
        "samples_per_task": int(shard["k"]),
        "expected_task_ids": list(shard["task_ids"]),
        "expected_task_ids_sha256": digest_text(canon(list(shard["task_ids"]))),
        "expected_rollouts": len(want),
        "expected_rollout_ids_sha256": digest_text(canon(want)),
        "rollouts": len(rows),
        "rollout_ids_sha256": digest_text(canon(sorted(rollout_id(r) for r in rows))),
        "rows_path": path.name,
        "rows_sha256": file_sha256(path),
        contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
        "source_provenance": require_one_producer(rows, f"shard {shard['index']}"),
        "sessions": sorted({r["provenance"]["session_id"] for r in rows
                            if (r.get("provenance") or {}).get("session_id")}),
        "complete": True,
        "written_at_utc": now_utc(),
    })


def write_shard(shard: dict, rows: list, cfg: dict | None = None) -> pathlib.Path:
    """Write one shard and its receipt, refusing anything short of its plan.

    The rows are validated against the plan BEFORE the file is published, and the
    receipt is written after it, so the only shard that ever carries a receipt is
    a complete one.
    """
    index = int(shard["index"])
    gaps = shard_rows_gaps(shard, rows)
    if gaps:
        raise SystemExit(
            f"REFUSED: shard {index:04d} is not the work it was planned to do "
            f"({', '.join(gaps)}). A short shard on disk is read as a finished "
            f"one by resume, so it is not written at all: re-roll the shard.")
    path = _shard_path(index)
    receipt_path = shard_receipt_path(index)
    if receipt_path.exists():
        receipt_path.unlink()          # no stale receipt while the rows change
    write_attested_jsonl(path, rows, f"rejection-sampling shard {index:04d}")
    receipt = build_shard_receipt(shard, rows, path, cfg)
    _write_json(receipt_path, receipt)
    return path


def shard_is_current(index: int) -> bool:
    """Is this shard's file usable under THIS environment contract?

    A shard produced under the retired tokenless contract exists on disk and would
    otherwise be treated as done for ever, because resume is "does the file
    exist". It is treated as NOT done, so the shard is re-rolled under the current
    contract instead of being silently pooled with it.

    Contract currency is necessary and NOT sufficient for completion: see
    `shard_gaps`, which is what resume actually asks.
    """
    path = _shard_path(index)
    if not path.exists():
        return False
    try:
        rows = _read_jsonl(path)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(rows) and all(contract_mod.is_current(r) for r in rows)


def shard_gaps(shard: dict, cfg: dict | None = None) -> list[str]:
    """Why this shard is not complete. Empty means a validated receipt covers it.

    Resume asks this instead of `path.exists()`. It is deliberately cheap -- the
    receipt already asserts the parsed facts (expected ids, count, contract,
    producer) and the rows digest binds those assertions to these exact bytes, so
    a resume check hashes the file rather than re-parsing every rollout.
    """
    index = int(shard["index"])
    path, receipt_path = _shard_path(index), shard_receipt_path(index)
    if not path.exists():
        return ["rows_file_absent"]
    if not receipt_path.exists():
        return ["receipt_absent"]
    rec = _load_receipt(receipt_path)
    if rec is None:
        return ["receipt_unreadable"]
    if not receipt_seal_ok(rec):
        return ["receipt_self_hash_mismatch"]
    gaps = []
    if rec.get("kind") != SHARD_RECEIPT_KIND:
        gaps.append(f"receipt_kind_{rec.get('kind')!r}")
    if int(rec.get("schema_version") or 0) != RECEIPT_VERSION:
        gaps.append(f"receipt_schema_version_{rec.get('schema_version')!r}")
    if not rec.get("complete"):
        gaps.append("receipt_says_incomplete")
    if not contract_mod.is_current(rec):
        gaps.append("receipt_under_another_environment_contract")
    if int(rec.get("index", -1)) != index:
        gaps.append("receipt_names_another_shard")
    if (rec.get("family"), int(rec.get("horizon") or -1)) != (
            shard["family"], int(shard["horizon"])):
        gaps.append("receipt_names_another_cell")
    if rec.get("expected_task_ids_sha256") != digest_text(canon(list(shard["task_ids"]))):
        gaps.append("receipt_covers_another_task_set")
    want = expected_rollout_ids(shard)
    if int(rec.get("expected_rollouts") or -1) != len(want):
        gaps.append(f"receipt_expected_{rec.get('expected_rollouts')}_"
                    f"of_{len(want)}_rollouts")
    if int(rec.get("rollouts") or -1) != len(want):
        gaps.append(f"receipt_recorded_{rec.get('rollouts')}_of_{len(want)}_rollouts")
    if rec.get("rollout_ids_sha256") != digest_text(canon(want)):
        gaps.append("receipt_rollout_ids_differ_from_the_plan")
    if rec.get("rows_sha256") != file_sha256(path):
        gaps.append("rows_file_changed_since_the_receipt")
    return gaps


def shard_is_complete(shard: dict, cfg: dict | None = None) -> bool:
    """Completion is a validated receipt over the planned ids, count and digest."""
    return not shard_gaps(shard, cfg)


# --------------------------------------------------------------------------
# vLLM backend (GPU only; imported lazily)
# --------------------------------------------------------------------------

def engine_stage(args) -> str:
    """Which producer stage this engine attests as.

    Read off `args.stage` so the manifest a stage writes carries the stage's own
    name; a caller that sets nothing gets the generic label rather than another
    stage's, because a manifest filed under the wrong stage attests the wrong
    thing.
    """
    return str(getattr(args, "stage", None) or "rollout_engine")


def _vllm_engine(cfg: dict, args, frozen: str | None,
                 adapter: str | None = None) -> RolloutEngine:
    """Build ONE engine under the registered contract, optionally with an adapter.

    Every setting comes from `configio.engine_contract()` -- there is no
    per-module --gpu-frac / --max-model-len knob any more, because two knobs are
    two engines and S19 reads an engine_fingerprint that disagrees with the
    registered contract as a BUG.

    Startup measured 289.7 s on this card. An engine is therefore a STAGE-scoped
    resource: build it once here and feed it every pending work unit (see
    `run_units`), never once per shard. Fifty rejection-sampling shards paying
    their own startup was 4.02 GPU-hours of pure model loading.

    `adapter` serves a LoRA checkpoint alongside the base weights. The variance
    probe needs it: probing the base model while asserting the RS-SFT policy's
    group variance measures the wrong policy.
    """
    from vllm import LLM, SamplingParams

    from agentlab import env as labenv
    from agentlab.suite import configio
    from agentlab.suite.configio import engine_contract

    contract = engine_contract(cfg)
    stage = engine_stage(args)
    run_id = getattr(args, "run_id", None)
    # THE PRODUCER ATTESTATION. This process owns the card, so it -- not a later
    # consumer, and not the run lock -- is the authority on what produced the
    # tokens. `capture_runtime_manifest` runs the whole hardware veto (PCI order,
    # registered index, registered card, exclusivity, the run's UUID binding),
    # measures the card, records which OS process this is and what it serves, and
    # hashes the result BEFORE any model work happens.
    manifest_path, manifest = labenv.capture_runtime_manifest(
        stage=stage, cfg=cfg, run_id=run_id, model=args.model, adapter=adapter)
    proc = labenv.load_processor(args.model)
    tok = labenv.get_tokenizer(proc)
    kwargs = dict(model=args.model,
                  dtype=contract["dtype"],
                  max_model_len=contract["max_model_len"],
                  gpu_memory_utilization=contract["gpu_memory_utilization"],
                  max_num_seqs=contract["max_num_seqs"],
                  max_num_batched_tokens=contract["max_num_batched_tokens"],
                  enforce_eager=contract["enforce_eager"],
                  tensor_parallel_size=contract["tensor_parallel_size"])
    if contract["multimodal_inputs"] == "REJECTED":
        # The registered contract says multimodal inputs are REJECTED, not merely
        # unused, and until now only scripts/serve.sh honoured that -- it passed
        # `--limit-mm-per-prompt '{"image":0,"video":0}'` as a literal while this
        # offline path read every other contract key and never learned about this
        # one. So the two engines were NOT the same engine, which is the drift the
        # single contract exists to prevent.
        #
        # It is not cosmetic: with the vision tower live, vLLM's memory
        # `profile_run` builds a dummy MULTIMODAL batch, and on this hybrid
        # checkpoint that dummy forward dies inside the Gated DeltaNet path with
        # `AttributeError: 'NoneType' object has no attribute 'size'`
        # (qwen3_next.py forward, via profile_run -> _dummy_run). The engine never
        # reached its KV-cache sizing, so preflight probe 4 could not start an
        # engine at all while probe 3's server -- carrying the literal -- was fine.
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    lora_request = None
    if adapter:
        kwargs.update(enable_lora=True,
                      max_lora_rank=int(cfg["sft"]["lora_rank"]))
    llm = LLM(**kwargs)
    if adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("trained", 1, adapter)
    dec = cfg["decoding"]
    sp = SamplingParams(temperature=dec["temperature"], top_p=dec["top_p"],
                        top_k=dec["top_k"], max_tokens=dec["max_tokens_per_decision"])
    # The contract is the authority on thinking, and the effective value is
    # RECORDED rather than assumed: this checkpoint thinks by default.
    thinking = contract["enable_thinking"]

    def render(messages, schemas):
        from agentlab.suite.configio import reject_multimodal

        reject_multimodal(messages, cfg)
        return tok.apply_chat_template(messages, tools=schemas, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=thinking)

    def generate(prompts):
        outs = (llm.generate(prompts, sp, lora_request=lora_request) if lora_request
                else llm.generate(prompts, sp))
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]

    # The engine answered: countersign the attestation, then COPY it. The
    # fingerprint every row carries is the producer's measurement, never a
    # re-measurement by a later reader and never the registered A5000
    # expectation standing in for one.
    manifest = configio.mark_manifest_ready(manifest_path, run_id=run_id, cfg=cfg,
                                            stage=stage)
    fp = configio.fingerprint_from_manifest(manifest, cfg)
    fp.update({GPU_EXECUTION: True,
               "producer": stage,
               "runtime_manifest_sha256": manifest[configio.MANIFEST_HASH_FIELD],
               "session_id": manifest["session_id"],
               "producer_pid": manifest["pid"],
               "adapter": manifest["adapter"],
               "adapter_sha256": manifest["adapter_sha256"],
               "served_model": manifest["model"]})
    if bool(fp["enable_thinking_effective"]) != bool(thinking):
        raise SystemExit(
            f"REFUSED: the attested manifest renders with enable_thinking="
            f"{fp['enable_thinking_effective']!r} and this engine renders with "
            f"{thinking!r}. The rendered prompt and the recorded policy must be "
            f"the same policy.")
    require_row_provenance(fp, f"every rollout of stage {stage}")
    # THE run secret, shared with the prompt tournament, view construction and
    # evaluation: the recovery tokens and receipts the model sees are keyed with
    # it, so two consumers with different secrets are two environments.
    engine = RolloutEngine(cfg, render, generate, frozen_prompt=frozen,
                           provenance=fp,
                           secret=contract_mod.load_or_create_secret())
    engine.manifest_path = manifest_path
    engine.manifest = manifest
    return engine


def run_units(engine, units: list, *, run_one, is_done, budget_minutes: float,
              label: str, cfg: dict | None = None, work_unit: str = "unit") -> dict:
    """Feed every pending work unit to ONE long-lived engine.

    This is the persistent-engine contract:

      * the ENGINE lives for the whole stage; the 289.7 s startup is paid once
      * a WORK UNIT is still a short, atomic, resumable checkpoint on disk, so a
        kill costs at most one unit and `--auto` resumes at the first incomplete
        one
      * the stage stops launching new units once `budget_minutes` is spent, and
        reports what is left, so re-invoking is always the resume mechanism

    Returns {"units_done", "units_left", "minutes"} and ledgers nothing itself:
    the caller owns the ledger row, so a unit and its stage cannot both charge
    the same seconds.
    """
    t0 = time.time()
    done = 0
    for unit in units:
        if is_done(unit):
            continue
        if (time.time() - t0) / 60.0 >= budget_minutes and done:
            break
        run_one(unit)
        done += 1
    left = [u for u in units if not is_done(u)]
    minutes = (time.time() - t0) / 60.0
    print(f"[{label}] {done} {work_unit}s this pass, {len(left)} left, "
          f"{minutes:.1f} min on one engine")
    return {"units_done": done, "units_left": len(left), "minutes": minutes,
            "complete": not left}


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_plan(args) -> None:
    cfg = load_config()
    shards = plan_shards(args.split, args.shard_size, cfg)
    done = 0
    for s in shards:
        gaps = shard_gaps(s, cfg)
        done += not gaps
        mark = "done" if not gaps else ("    " if gaps == ["rows_file_absent"]
                                        else f"INCOMPLETE: {', '.join(gaps[:2])}")
        print(f"  shard {s['index']:04d}  {s['family']:13s} h{s['horizon']:<3d} "
              f"{len(s['task_ids']):3d} variants x k{s['k']} = "
              f"{s['expected_rollouts']:4d} rollouts  {mark}")
    print(f"[plan] {len(shards)} shards ({done} done, receipt-validated), "
          f"split={args.split or rs_split(cfg)}, shard_size={args.shard_size}")


def cmd_run(args) -> None:
    """Roll out pending shards on ONE engine.

    The engine start is charged to the ledger once, as its own row; each shard is
    then charged its own decode minutes. Neither overlaps the other, so the
    startup that used to be invisible (it happened before the per-shard timer
    began) is now an explicit budget line.
    """
    cfg = load_config()
    split = args.split or rs_split(cfg)
    shards = plan_shards(split, args.shard_size, cfg)
    if args.shard is not None:
        units = [shards[args.shard]]
        if shard_is_complete(shards[args.shard], cfg) and not args.force:
            print(f"[rs] shard {args.shard} already done (use --force to redo)")
            return
        forced = {args.shard} if args.force else set()
    else:
        units = shards
        forced = set()
    pending = [s for s in units
               if s["index"] in forced or not shard_is_complete(s, cfg)]
    # A shard file that exists but does not satisfy its receipt is RE-ROLLED, not
    # resumed: a retired-contract shard, a shard truncated by a kill, and a shard
    # planned at another --shard-size all land here rather than counting as done.
    unusable = {s["index"]: shard_gaps(s, cfg) for s in pending
                if s["index"] not in forced and _shard_path(s["index"]).exists()}
    if unusable:
        print(f"[rs] {len(unusable)} shard file(s) exist without a valid receipt "
              f"and will be RE-ROLLED, not resumed: "
              + "; ".join(f"{i:04d}: {', '.join(g[:2])}"
                          for i, g in sorted(unusable.items())[:8]))
    if not pending:
        print("[rs] all shards done")
        return
    ledger_guard("multidistill", args.budget_minutes, cfg)
    args.stage = "multidistill"

    frozen = None if args.no_frozen_prompt else load_frozen_prompt(cfg)
    by_id = {b.spec.task_id: b for b in load_split(split, cfg)}
    variants = ("canonical",) if args.no_frozen_prompt else ("canonical", "frozen")

    started = now_utc()
    t_engine = time.time()
    engine = _vllm_engine(cfg, args, frozen)
    startup_min = (time.time() - t_engine) / 60.0
    # The ledger row COPIES the producer session's snapshot, so the receipt that
    # charges these minutes carries the same UUID, driver, engine fingerprint and
    # session digest as the rollouts they produced.
    ledger_append("multidistill:engine_start", startup_min, cfg,
                  kind="engine_start", started_at=started,
                  manifest=engine.manifest_path,
                  work={"unit": "engine", "count": 1, "shards_pending": len(pending)})
    print(f"[rs] engine up in {startup_min:.1f} min; it serves all "
          f"{len(pending)} pending shards")

    def is_done(shard):
        return shard["index"] not in forced and shard_is_complete(shard, cfg)

    def run_one(shard):
        bundles = [by_id[t] for t in shard["task_ids"]]
        t0 = time.time()
        records = engine.run(engine.rollouts_for(bundles, variants=variants))
        # Validates the rollouts against the shard's plan, then writes the rows and
        # the receipt that certifies them; a short shard is never published.
        write_shard(shard, records, cfg)
        forced.discard(shard["index"])
        minutes = (time.time() - t0) / 60.0
        cumulative = ledger_append("multidistill", minutes, cfg, kind="shard",
                                   manifest=engine.manifest_path,
                                   work={"unit": "rollouts", "count": len(records),
                                         "shard": shard["index"],
                                         "variants": len(bundles)})
        print(f"[rs] shard {shard['index']:04d}: {len(records)} rollouts in "
              f"{minutes:.1f} min -> {_shard_path(shard['index'])} "
              f"(ledger {cumulative:.2f}h)")

    status = run_units(engine, units, run_one=run_one, is_done=is_done,
                       budget_minutes=args.budget_minutes, label="rs",
                       cfg=cfg, work_unit="shard")
    print(json.dumps(status))


def finalize(records: list, bundles: dict, cfg: dict) -> tuple[list, dict]:
    """Pure CPU acceptance pass -> (kept records, summary).

    Raw shards produced under a different model-visible environment are DROPPED,
    not resumed: `accept_record` refuses them through
    `contract.require_current`, and they are counted separately so the report
    says out loud that a regeneration is owed rather than reporting a quota miss.

    Rows that cannot say what produced them are dropped the same way
    (`missing_provenance:*`), and a corpus that mixes two producer identities is
    fatal: the accepted rows are the source snapshot the SFT views, the trainer
    manifest and the checkpoint lock all inherit, so an ambiguity here becomes an
    ambiguity in the locked checkpoint.
    """
    reasons: dict[str, int] = {}
    accepted: dict[str, dict] = {}
    records, stale = contract_mod.invalidate(records, "raw rejection-sampling row")
    if stale:
        reasons["stale_environment_contract"] = len(stale)
    for rec in records:
        ok, why = accept_record(rec, cfg, bundles)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        # One accepted trajectory per variant: prefer fewer calls, then the
        # earlier sample, deterministically.
        key = rec["task_id"]
        rank = (rec["verdict"]["calls"], rec["sample_index"])
        if key not in accepted or rank < (accepted[key]["verdict"]["calls"],
                                          accepted[key]["sample_index"]):
            if key in accepted:
                reasons["superseded_duplicate"] = reasons.get("superseded_duplicate", 0) + 1
            accepted[key] = rec

    kept = [accepted[k] for k in sorted(accepted)]
    # The corpus may descend from MANY producer sessions of one run (resumable
    # shards are the design) but from exactly ONE producer identity. A mixed
    # corpus is fatal here, not a footnote in the report: it is the S19 failure
    # itself, and every downstream artifact would inherit the ambiguity.
    source_provenance = require_one_producer(kept, "the accepted RS corpus")
    per_cell: dict[str, int] = {}
    for rec in kept:
        cell = f"{rec['family']}-h{rec['horizon']}"
        per_cell[cell] = per_cell.get(cell, 0) + 1

    measured_only = set(cfg["totals"].get("measured_only_cells") or ())
    quotas = {}
    for fam, m in cfg["mixture"].items():
        got = sum(v for k, v in per_cell.items()
                  if k.startswith(f"{fam}-") and k not in measured_only)
        quotas[fam] = {"accepted": got, "min_accepted": m["min_accepted"],
                       "ok": got >= m["min_accepted"],
                       "excluded_cells": sorted(c for c in measured_only
                                                if c.startswith(f"{fam}-"))}
    n_faulted = sum(1 for r in kept if r["fault_types"])
    faulted_min = cfg["totals"]["min_accepted_faulted"]
    quotas["_faulted"] = {"accepted": n_faulted, "min_accepted": faulted_min,
                          "ok": n_faulted >= faulted_min}
    summary = {"rollouts": len(records) + len(stale),
               "stale_environment_contract": len(stale),
               contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
               # The chain the SFT views, the trainer manifest and the checkpoint
               # lock inherit: which producer, which sessions, which attestations.
               "source_provenance": source_provenance,
               "source_sessions": sorted({r["provenance"]["session_id"] for r in kept
                                          if r["provenance"].get("session_id")}),
               "source_runtime_manifests": sorted(
                   {r["provenance"]["runtime_manifest_sha256"] for r in kept
                    if r["provenance"].get("runtime_manifest_sha256")}),
               "accepted": len(kept),
               "acceptance_rate": round(len(kept) / max(len(records), 1), 4),
               "per_cell": dict(sorted(per_cell.items())),
               "measured_only_cells": sorted(measured_only),
               "faulted_accepted": n_faulted,
               "quotas": quotas, "rejections": dict(sorted(reasons.items()))}
    return kept, summary


# --------------------------------------------------------------------------
# the accepted corpus: a receipt, or no corpus at all
# --------------------------------------------------------------------------

def accepted_receipt_path(path=None) -> pathlib.Path:
    """`accepted.receipt.json` beside the corpus it certifies."""
    return pathlib.Path(path or ACCEPTED_PATH).with_suffix(".receipt.json")


def quota_misses(summary: dict) -> list[dict]:
    """The quota rows that did NOT pass, worst deficit first."""
    out = []
    for name, q in (summary.get("quotas") or {}).items():
        if q.get("ok"):
            continue
        got, want = int(q.get("accepted") or 0), int(q.get("min_accepted") or 0)
        out.append({"quota": name, "accepted": got, "min_accepted": want,
                    "short_by": max(want - got, 0),
                    "excluded_cells": q.get("excluded_cells") or []})
    out.sort(key=lambda q: (-q["short_by"], q["quota"]))
    return out


def clear_accepted_corpus() -> list[str]:
    """Remove the corpus AND its receipt, so no resume can trust either."""
    removed = []
    for path in (accepted_receipt_path(), ACCEPTED_PATH):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def build_accepted_receipt(kept: list, summary: dict, shards: list,
                           path: pathlib.Path, cfg: dict | None = None) -> dict:
    """The receipt that lets a later stage trust this corpus.

    It names the shard census the corpus was assembled from (every shard receipt
    digest), the kept task ids and their digest, the row count, the digest of the
    corpus bytes and the quota verdicts. It exists ONLY for a corpus that passed
    every quota with every planned shard validated -- which is what makes its
    presence the completion marker.
    """
    cfg = cfg or load_config()
    task_ids = sorted(r["task_id"] for r in kept)
    receipts = []
    for shard in shards:
        rec = _load_receipt(shard_receipt_path(int(shard["index"]))) or {}
        receipts.append({"index": int(shard["index"]),
                         RECEIPT_HASH_FIELD: rec.get(RECEIPT_HASH_FIELD)})
    return seal_receipt({
        "kind": ACCEPTED_RECEIPT_KIND,
        "schema_version": RECEIPT_VERSION,
        "split": summary.get("split") or rs_split(cfg),
        "shards": len(shards),
        "shard_receipts": receipts,
        "rollouts": int(summary.get("rollouts") or 0),
        "accepted": len(kept),
        "task_ids": len(task_ids),
        "task_ids_sha256": digest_text(canon(task_ids)),
        "corpus_path": path.name,
        "corpus_sha256": file_sha256(path),
        "quotas": summary.get("quotas") or {},
        "quota_ok": True,
        "partial": False,
        "per_cell": summary.get("per_cell") or {},
        "faulted_accepted": int(summary.get("faulted_accepted") or 0),
        contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
        "source_provenance": summary.get("source_provenance") or {},
        "source_sessions": summary.get("source_sessions") or [],
        "complete": True,
        "written_at_utc": now_utc(),
    })


def accepted_corpus_gaps(path=None) -> list[str]:
    """Why the accepted corpus on disk may not be consumed. Empty means it may.

    Existence is not completion: the receipt must be sealed, current, quota-clean,
    non-partial, and must match the corpus bytes -- count, task-id digest and file
    digest -- that a consumer is about to read.
    """
    corpus = pathlib.Path(path or ACCEPTED_PATH)
    receipt_path = accepted_receipt_path(corpus)
    if not corpus.exists():
        return ["corpus_absent"]
    if not receipt_path.exists():
        return ["receipt_absent"]
    rec = _load_receipt(receipt_path)
    if rec is None:
        return ["receipt_unreadable"]
    if not receipt_seal_ok(rec):
        return ["receipt_self_hash_mismatch"]
    gaps = []
    if rec.get("kind") != ACCEPTED_RECEIPT_KIND:
        gaps.append(f"receipt_kind_{rec.get('kind')!r}")
    if int(rec.get("schema_version") or 0) != RECEIPT_VERSION:
        gaps.append(f"receipt_schema_version_{rec.get('schema_version')!r}")
    if not rec.get("complete"):
        gaps.append("receipt_says_incomplete")
    if rec.get("partial"):
        gaps.append("receipt_says_partial")
    if not rec.get("quota_ok"):
        gaps.append("receipt_says_quota_miss")
    if not contract_mod.is_current(rec):
        gaps.append("receipt_under_another_environment_contract")
    if rec.get("corpus_sha256") != file_sha256(corpus):
        gaps.append("corpus_changed_since_the_receipt")
        return gaps
    try:
        rows = _read_jsonl(corpus)
    except (json.JSONDecodeError, OSError):
        return gaps + ["corpus_unreadable"]
    if not rows:
        gaps.append("zero_accepted_rows")
    if int(rec.get("accepted") or -1) != len(rows):
        gaps.append(f"receipt_counts_{rec.get('accepted')}_of_{len(rows)}_rows")
    ids = sorted(r.get("task_id") for r in rows)
    if rec.get("task_ids_sha256") != digest_text(canon(ids)):
        gaps.append("receipt_names_another_task_set")
    return gaps


def require_accepted_corpus(path=None) -> dict:
    """Return the validated receipt, or refuse to let the next stage start.

    This is the "do not allow views/SFT" half of the quota rule: the view builder
    calls it before it reads a single trajectory, so a corpus that missed a quota
    (or was never finalized, or was truncated afterwards) stops the chain here
    instead of quietly training on whatever is on disk.
    """
    corpus = pathlib.Path(path or ACCEPTED_PATH)
    gaps = accepted_corpus_gaps(corpus)
    if gaps:
        raise SystemExit(
            f"REFUSED: {corpus} is not a completed accepted corpus "
            f"({', '.join(gaps)}).\n"
            f"  Completion is a validated receipt at {accepted_receipt_path(corpus)} "
            f"-- expected shards, kept task ids, row count, corpus digest and "
            f"PASSING quotas -- never a path that exists. Re-run `python -m "
            f"agentlab.multidistill run` for the shards that are short, then "
            f"`finalize`.")
    return _load_receipt(accepted_receipt_path(corpus))


def cmd_finalize(args) -> None:
    cfg = load_config()
    split = args.split or rs_split(cfg)
    shards = plan_shards(split, args.shard_size, cfg)
    # "Missing" is measured against each shard's RECEIPT, not its path: a shard
    # holding 3 of the 384 rollouts it owes would otherwise be finalized as if it
    # had run, and its cell would look like an honest quota miss.
    incomplete = {int(s["index"]): shard_gaps(s, cfg) for s in shards}
    incomplete = {i: g for i, g in incomplete.items() if g}
    if incomplete and not args.partial:
        detail = "; ".join(f"{i:04d}: {', '.join(g)}"
                           for i, g in sorted(incomplete.items())[:8])
        raise SystemExit(
            f"REFUSED: {len(incomplete)} of {len(shards)} shards have no valid "
            f"receipt ({detail}"
            f"{', ...' if len(incomplete) > 8 else ''}).\n"
            f"  Run them (`python -m agentlab.multidistill run`) or pass "
            f"--partial, which reports acceptance so far and deliberately "
            f"produces NO accepted corpus.")

    # Only receipt-validated shards are pooled, even in the --partial diagnostic:
    # rows from a truncated shard would distort the acceptance rate the operator is
    # about to read, and they may never reach a corpus in any case.
    records, pooled = [], []
    for s in shards:
        if int(s["index"]) in incomplete:
            continue
        records += _read_jsonl(_shard_path(s["index"]))
        pooled.append(int(s["index"]))

    bundles = {b.spec.task_id: b for b in load_split(split, cfg)}
    kept, summary = finalize(records, bundles, cfg)
    summary["split"] = split
    summary["shards"] = len(shards)
    summary["shards_pooled"] = len(pooled)
    summary["shards_incomplete"] = sorted(incomplete)
    summary["partial"] = bool(incomplete)
    summary["quota_misses"] = quota_misses(summary)
    summary["quota_ok"] = not summary["quota_misses"]
    summary["complete"] = summary["quota_ok"] and not summary["partial"]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))

    # THE HARD STOP. A quota miss (or a partial finalize) used to print a warning
    # and exit 0 after writing accepted.jsonl, so the next invocation of the chain
    # saw the file, skipped the stage, and trained on a corpus that had failed its
    # preregistered minimum. The quotas are registered numbers: the fix is more
    # rollouts in the short cells, never a smaller floor. So nothing is written,
    # anything a resume could trust is REMOVED, the reason is written down, and
    # the stage exits nonzero.
    if not summary["complete"]:
        removed = clear_accepted_corpus()
        failure = {
            "kind": "agentlab_rs_finalize_failure",
            "schema_version": RECEIPT_VERSION,
            "split": split,
            "reason": ("quota_miss" if summary["quota_misses"]
                       else "incomplete_shards"),
            "quota_misses": summary["quota_misses"],
            "shards": len(shards),
            "shards_incomplete": sorted(incomplete),
            "shard_gaps": {str(i): g for i, g in sorted(incomplete.items())},
            "accepted": len(kept),
            "per_cell": summary["per_cell"],
            "faulted_accepted": summary["faulted_accepted"],
            "removed_untrustworthy_artifacts": removed,
            contract_mod.STAMP_FIELD: contract_mod.environment_contract_sha256(),
            "written_at_utc": now_utc(),
        }
        _write_json(FAILURE_PATH, failure)
        lines = [f"REFUSED: the accepted RS corpus is not complete, so it was not "
                 f"written."]
        for q in summary["quota_misses"]:
            lines.append(f"  quota {q['quota']}: {q['accepted']} accepted, "
                         f"minimum {q['min_accepted']} (short by {q['short_by']})")
        if summary["partial"]:
            lines.append(f"  {len(incomplete)} of {len(shards)} shards have no "
                         f"valid receipt: {sorted(incomplete)[:8]}")
        if removed:
            lines.append(f"  removed so no resume can trust it: "
                         f"{', '.join(removed)}")
        lines.append(f"  why, as data: {FAILURE_PATH}")
        lines.append("  The quotas and the horizon strata are preregistered: roll "
                     "out more variants in the short cells (`python -m "
                     "agentlab.multidistill run`) and finalize again. A missing "
                     "deep-cell quota is never backfilled from easier cells, and "
                     "no minimum may be lowered.")
        raise SystemExit("\n".join(lines))

    # Horizon strata are structural: acceptance is one-per-variant and each
    # variant's horizon is fixed by the committed spec, so a missing deep-cell
    # quota can never be silently backfilled with easy cells.
    if FAILURE_PATH.exists():
        FAILURE_PATH.unlink()          # this finalize succeeded; the old why is void
    receipt_path = accepted_receipt_path()
    if receipt_path.exists():
        receipt_path.unlink()          # never a receipt for bytes being replaced
    write_attested_jsonl(ACCEPTED_PATH, kept, "the accepted RS corpus")
    receipt = build_accepted_receipt(kept, summary, shards, ACCEPTED_PATH, cfg)
    _write_json(receipt_path, receipt)
    # Read it back through the consumer's own gate: the receipt must describe the
    # bytes on disk, not the objects that were in memory a moment ago.
    require_accepted_corpus(ACCEPTED_PATH)
    print(f"[rs] accepted corpus {ACCEPTED_PATH} ({len(kept)} trajectories) "
          f"certified by {receipt_path} "
          f"(digest {receipt[RECEIPT_HASH_FIELD][:12]}...)")


def cmd_status(args) -> None:
    cfg = load_config()
    shards = plan_shards(args.split, args.shard_size, cfg)
    done = [s for s in shards if shard_is_complete(s, cfg)]
    started = [s for s in shards
               if s not in done and _shard_path(s["index"]).exists()]
    print(f"[status] shards {len(done)}/{len(shards)} done (receipt-validated), "
          f"{len(started)} file(s) present without a valid receipt")
    gaps = accepted_corpus_gaps()
    print(f"[status] accepted corpus: "
          + ("complete" if not gaps else f"NOT usable ({', '.join(gaps)})"))
    if FAILURE_PATH.exists():
        print(f"[status] last finalize refused; why: {FAILURE_PATH}")
    if SUMMARY_PATH.exists():
        print(SUMMARY_PATH.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split", default=None,
                        help="suite split to roll out (default: suite.rs_source)")
    common.add_argument("--shard-size", type=int, default=48,
                        help="variants per shard; keep one shard well under 8 minutes")

    sub.add_parser("plan", parents=[common])

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--shard", type=int, default=None,
                     help="one shard only; omit to serve every pending shard "
                          "from ONE engine (the default, and the reason engine "
                          "startup is paid once per stage rather than per shard)")
    run.add_argument("--auto", action="store_true",
                     help="accepted for compatibility; serving all pending "
                          "shards from one engine is now the default")
    run.add_argument("--force", action="store_true")
    run.add_argument("--model", default=None)
    run.add_argument("--run-id", default=None,
                     help="S19 run_id stamped on every row (default AGENTIC_RUN_ID)")
    # NO --gpu-frac / --max-model-len / --enforce-eager: the engine contract lives
    # in configs/multifaceted.yaml `engine:` and a per-invocation override is a
    # second engine. S19 reads a fingerprint that disagrees with the contract as
    # a BUG, so the knob would only ever produce an unusable trace.
    run.add_argument("--no-frozen-prompt", action="store_true",
                     help="SMOKE ONLY: canonical prompt for every attempt")
    run.add_argument("--budget-minutes", type=float, default=55.0,
                    help="stage budget for this pass: stop LAUNCHING new shards "
                         "after this many minutes. Each shard is still atomic and "
                         "resumable; re-invoke to continue.")

    fin = sub.add_parser("finalize", parents=[common])
    fin.add_argument("--partial", action="store_true",
                     help="DIAGNOSTIC ONLY: report acceptance over the shards "
                          "that have valid receipts. It writes the summary, "
                          "produces NO accepted corpus, and exits nonzero -- a "
                          "corpus assembled from part of the plan is not the "
                          "registered corpus and must never be trained on.")
    sub.add_parser("status", parents=[common])

    args = ap.parse_args()
    if args.cmd == "run":
        from agentlab import env as labenv

        args.model = args.model or labenv.MODEL
    {"plan": cmd_plan, "run": cmd_run, "finalize": cmd_finalize,
     "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
