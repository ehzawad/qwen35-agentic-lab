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
loading. A shard is still an atomic, resumable client-side work unit -- done when
its output file exists, at 48 variants (<=384 rollouts at k<=8, ~150
rollouts/min measured) so a kill costs at most one shard -- but it no longer pays
for an engine. The engine start is charged to the ledger as its own row and each
shard is charged its own decode minutes, so the two never overlap. `finalize` is
CPU-only.

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

from agentlab.chat import boxed_answer, numeric_answer, parse_tool_calls, strip_thinking
from agentlab.suite import runtime as rt_mod
from agentlab.suite.configio import (ROOT, ledger_append, ledger_guard, load_config,
                                     now_utc)
from agentlab.suite.generate import load_bundles
from agentlab.suite.schema import canon

MULTIFACE_DIR = ROOT / "data" / "multiface"
RAW_DIR = MULTIFACE_DIR / "raw"
ACCEPTED_PATH = MULTIFACE_DIR / "accepted.jsonl"
SUMMARY_PATH = MULTIFACE_DIR / "rs_summary.json"

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
_BOXED_ANY = re.compile(r"\\boxed")


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

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
                 frozen_prompt: str | None = None, provenance: dict | None = None):
        self.cfg = cfg
        self.render = render_fn
        self.generate = generate_fn
        self.frozen = frozen_prompt
        # The S19 fingerprint of the engine that produced every row this engine
        # emits: which card, which driver, which engine settings, which effective
        # thinking mode. None on the scripted CPU engines the tests inject.
        self.provenance = provenance
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
        runtime = rt_mod.EpisodeRuntime(spec, bundle.kb, bundle.nodes)
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

        prose = _TOOL_CALL_BLOCK.sub("", text).strip()
        msg = {"role": "assistant",
               "tool_calls": [{"type": "function",
                               "function": {"name": x["name"], "arguments": x["arguments"]}}
                              for x in calls]}
        if prose:
            msg["content"] = prose
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
                c["calls"].append({"decision_id": decision, "tool": name,
                                   "args": args, "exposed": result})
            tool_idx = len(c["messages"])
            c["messages"].append({"role": "tool", "name": name, "content": result})
            c["call_map"].append({"call_id": call_id,
                                  "assistant_msg_index": assistant_idx,
                                  "tool_msg_index": tool_idx})

    def _record(self, c: dict) -> dict:
        bundle, runtime = c["bundle"], c["runtime"]
        spec = bundle.spec
        verdict = runtime.verify(c["final"])
        fault = _fault_summary(runtime, spec, c["call_map"])
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
            # Which card and which engine produced this row (S19). Carried on the
            # ROW, not only in a nearby ledger: a row that cannot say what
            # produced it cannot support a same-card claim.
            "provenance": self.provenance,
        }


def _fault_summary(runtime, spec, call_map: list) -> dict | None:
    """Where the scheduled fault fired and where recovery happened, if at all.

    Read off the environment-side trace, never off model text: `fault_triggered`
    is stamped by the injector and `exposed_canonical` says the runtime later
    produced the node's true observation.
    """
    if not spec.faults:
        return None
    events = runtime.events
    fired = next((e for e in events if e.fault_triggered), None)
    by_call = {m["call_id"]: m for m in call_map}
    out = {"fired": fired is not None, "result_msg_index": None,
           "recovery_msg_index": None, "fault_decision": None,
           "recovery_decision": None, "post_fault_retries": 0}
    if fired is None:
        return out
    out["fault_decision"] = fired.decision_id
    out["result_msg_index"] = (by_call.get(fired.call_id) or {}).get("tool_msg_index")
    later = [e for e in events if e.call_id > fired.call_id
             and e.oracle_node == fired.oracle_node]
    out["post_fault_retries"] = len(later)
    recovered = next((e for e in later if e.exposed_canonical), None)
    if recovered is None and fired.fault_type == "malformed":
        # Ambiguous post-mutation case: a reservation-status query on the same
        # line establishes the state instead of re-observing the node.
        line = (fired.aux or {}).get("line")
        recovered = next((e for e in events if e.call_id > fired.call_id
                          and e.tool == "warehouse_query" and e.ok
                          and (e.aux or {}).get("resource") == "reservation"
                          and (e.aux or {}).get("line") == line), None)
    if recovered is not None:
        out["recovery_decision"] = recovered.decision_id
        out["recovery_msg_index"] = (by_call.get(recovered.call_id)
                                     or {}).get("assistant_msg_index")
    return out


# --------------------------------------------------------------------------
# exact replay verification (the faithfulness gate)
# --------------------------------------------------------------------------

def replay_record(rec: dict, bundle) -> tuple[bool, str]:
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
    ok, why = rt_mod.verify_replay(bundle.spec, bundle.kb, bundle.nodes, calls,
                                   rec["parity"])
    if not ok:
        return False, why
    runtime, _report = rt_mod.replay_trace(bundle.spec, bundle.kb, bundle.nodes,
                                           calls)
    if len(runtime.events) != len(calls):
        return False, f"replay_call_count:{len(runtime.events)}!={len(calls)}"
    for i, (event, call) in enumerate(zip(runtime.events, calls)):
        # The exposed digest is SHA-256 of the exact bytes the replay produced,
        # so this IS a byte comparison against what the model was shown.
        if digest_text(call.get("exposed", "")) != event.exposed_result_digest:
            return False, f"replay_observation_bytes@{i}"
    verdict = runtime.verify(rec["final"]).to_row()
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
    """Recovery must be mechanical, prompt, and must not launder the trap value."""
    fault = rec.get("fault") or {}
    if not rec["fault_types"]:
        return ""
    if not fault.get("fired"):
        return "fault_not_fired"
    if not rec["verdict"]["recovered"]:
        return "recovery_not_verified"
    rcfg = cfg["acceptance"]["recovery"]
    fd, rd = fault.get("fault_decision"), fault.get("recovery_decision")
    if fd is None or rd is None or rd - fd > rcfg["max_decisions_after_fault"]:
        return "recovery_too_late"
    if fault.get("post_fault_retries", 0) > rcfg["max_identical_retries"]:
        return "too_many_retries"
    if "wrong_unit" in rec["fault_types"]:
        # The remediation contract for a wrong-unit trap is "corrected target
        # required": a later conversion must explicitly request the unit the
        # oracle asked for. `exposed_canonical` on a post-fault event at the
        # target node is exactly that, decided by the canonical matcher rather
        # than by pattern-matching numbers out of the transcript.
        if not _corrected_conversion(rec):
            return "no_corrected_conversion"
        trapped = _trapped_value(rec)
        if trapped is not None and answers_match(trapped, rec["answer"]):
            return "trapped_value_used"
    return ""


def _corrected_conversion(rec: dict) -> bool:
    fired = next((e for e in rec["events"] if e["fault_triggered"]
                  and e["fault_type"] == "wrong_unit"), None)
    if fired is None:
        return False
    return any(e["call_id"] > fired["call_id"]
               and e["oracle_node"] == fired["oracle_node"]
               and e["exposed_canonical"]
               for e in rec["events"])


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
        obj = json.loads(rec["messages"][idx]["content"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    return obj.get("value") if isinstance(obj, dict) else None


def accept_record(rec: dict, cfg: dict | None = None, bundles: dict | None = None,
                  skip_replay: bool = False) -> tuple[bool, str]:
    """(accepted, rejection_reason).

    Order: cheap universal filters, then the STRICT VERIFIER (the single
    definition of task success), then the preregistered budget/recovery filters,
    then replay parity. The verifier is not re-implemented here.
    """
    cfg = cfg or load_config()
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
    if boxed_answer(rec["final"]) is None:
        # Commitment discipline: every family's prompt asks for \boxed{}. Answer
        # CORRECTNESS is decided by the verifier below and nowhere else.
        return False, "no_box"
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
    if not verdict["strict_success"]:
        return False, "verifier_rejected"

    why = _accept_budget(rec, cfg) or _accept_recovery(rec, cfg)
    if why:
        return False, why

    if not skip_replay:
        bundle = (bundles or {}).get(rec["task_id"])
        if bundle is None:
            return False, "replay_bundle_missing"
        ok, why = replay_record(rec, bundle)
        if not ok:
            return False, why
    return True, ""


# --------------------------------------------------------------------------
# shards
# --------------------------------------------------------------------------

def plan_shards(split: str | None = None, shard_size: int = 48,
                cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    split = split or rs_split(cfg)
    bundles = load_split(split, cfg)
    groups: dict[tuple, list] = {}
    for b in bundles:
        groups.setdefault((b.spec.family, b.spec.horizon), []).append(b)
    shards = []
    for (family, horizon) in sorted(groups):
        block = groups[(family, horizon)]
        for i in range(0, len(block), shard_size):
            chunk = block[i:i + shard_size]
            shards.append({"index": len(shards), "family": family, "horizon": horizon,
                           "task_ids": [b.spec.task_id for b in chunk]})
    return shards


def _shard_path(index: int) -> pathlib.Path:
    return RAW_DIR / f"shard-{index:04d}.jsonl"


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


# --------------------------------------------------------------------------
# vLLM backend (GPU only; imported lazily)
# --------------------------------------------------------------------------

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
    from agentlab.suite.configio import engine_contract, fingerprint

    contract = engine_contract(cfg)
    labenv.require_registered_gpu(cfg)
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

    fp = fingerprint(getattr(args, "run_id", None), cfg, enable_thinking=thinking)
    fp["adapter"] = adapter
    fp["served_model"] = args.model
    return RolloutEngine(cfg, render, generate, frozen_prompt=frozen,
                         provenance=fp)


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
    done = sum(1 for s in shards if _shard_path(s["index"]).exists())
    for s in shards:
        mark = "done" if _shard_path(s["index"]).exists() else "    "
        print(f"  shard {s['index']:04d}  {s['family']:13s} h{s['horizon']:<3d} "
              f"{len(s['task_ids']):3d} variants  {mark}")
    print(f"[plan] {len(shards)} shards ({done} done), "
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
        if _shard_path(args.shard).exists() and not args.force:
            print(f"[rs] shard {args.shard} already done (use --force to redo)")
            return
        forced = {args.shard} if args.force else set()
    else:
        units = shards
        forced = set()
    pending = [s for s in units
               if s["index"] in forced or not _shard_path(s["index"]).exists()]
    if not pending:
        print("[rs] all shards done")
        return
    ledger_guard("multidistill", args.budget_minutes, cfg)

    frozen = None if args.no_frozen_prompt else load_frozen_prompt(cfg)
    by_id = {b.spec.task_id: b for b in load_split(split, cfg)}
    variants = ("canonical",) if args.no_frozen_prompt else ("canonical", "frozen")

    started = now_utc()
    t_engine = time.time()
    engine = _vllm_engine(cfg, args, frozen)
    startup_min = (time.time() - t_engine) / 60.0
    ledger_append("multidistill:engine_start", startup_min, cfg,
                  kind="engine_start", started_at=started,
                  work={"unit": "engine", "count": 1, "shards_pending": len(pending)})
    print(f"[rs] engine up in {startup_min:.1f} min; it serves all "
          f"{len(pending)} pending shards")

    def is_done(shard):
        return shard["index"] not in forced and _shard_path(shard["index"]).exists()

    def run_one(shard):
        bundles = [by_id[t] for t in shard["task_ids"]]
        t0 = time.time()
        records = engine.run(engine.rollouts_for(bundles, variants=variants))
        _write_jsonl(_shard_path(shard["index"]), records)
        forced.discard(shard["index"])
        minutes = (time.time() - t0) / 60.0
        cumulative = ledger_append("multidistill", minutes, cfg, kind="shard",
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
    """Pure CPU acceptance pass -> (kept records, summary)."""
    reasons: dict[str, int] = {}
    accepted: dict[str, dict] = {}
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
    summary = {"rollouts": len(records), "accepted": len(kept),
               "acceptance_rate": round(len(kept) / max(len(records), 1), 4),
               "per_cell": dict(sorted(per_cell.items())),
               "measured_only_cells": sorted(measured_only),
               "faulted_accepted": n_faulted,
               "quotas": quotas, "rejections": dict(sorted(reasons.items()))}
    return kept, summary


def cmd_finalize(args) -> None:
    cfg = load_config()
    split = args.split or rs_split(cfg)
    shards = plan_shards(split, args.shard_size, cfg)
    missing = [s["index"] for s in shards if not _shard_path(s["index"]).exists()]
    if missing and not args.partial:
        raise SystemExit(f"missing shards {missing}; run them or pass --partial")

    records = []
    for s in shards:
        p = _shard_path(s["index"])
        if p.exists():
            records += _read_jsonl(p)

    bundles = {b.spec.task_id: b for b in load_split(split, cfg)}
    kept, summary = finalize(records, bundles, cfg)
    summary["partial"] = bool(missing)
    # Horizon strata are structural: acceptance is one-per-variant and each
    # variant's horizon is fixed by the committed spec, so a missing deep-cell
    # quota can never be silently backfilled with easy cells.
    _write_jsonl(ACCEPTED_PATH, kept)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(q["ok"] for q in summary["quotas"].values()):
        print("[rs] QUOTA MISS: at least one family is under its preregistered "
              "minimum; the RS-SFT stage must not proceed on this corpus.")


def cmd_status(args) -> None:
    cfg = load_config()
    shards = plan_shards(args.split, args.shard_size, cfg)
    done = [s for s in shards if _shard_path(s["index"]).exists()]
    print(f"[status] shards {len(done)}/{len(shards)} done")
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
    fin.add_argument("--partial", action="store_true")
    sub.add_parser("status", parents=[common])

    args = ap.parse_args()
    if args.cmd == "run":
        from agentlab import env as labenv

        args.model = args.model or labenv.MODEL
    {"plan": cmd_plan, "run": cmd_run, "finalize": cmd_finalize,
     "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
