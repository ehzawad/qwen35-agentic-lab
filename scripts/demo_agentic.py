#!/usr/bin/env python3
"""The fixed, unfiltered five-episode demonstration of the shipped tool loop.

Run it against a server YOU started (see docs/SERVING.md):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 \
    AGENTIC_RUN_ID=agentic-v1 PORT=8000 bash scripts/serve.sh Qwen/Qwen3.5-4B

    PYTHONPATH=src .venv/bin/python scripts/demo_agentic.py

This process opens no CUDA context and starts no engine: it is a pure HTTP
client, exactly like the claim-bearing evaluator.

WHY IT IS BUILT THIS WAY

  * THE PANEL IS FIXED IN THIS FILE, BEFORE THE SERVER IS CONTACTED. Five task
    ids, five conditions. `PANEL` below is a module constant and
    `derive_panel()` re-derives every id from the committed dev manifest by a
    declared cell/fault-class rule, refusing if the manifest disagrees. Nothing
    is selected, filtered, substituted, rerolled or ordered by its outcome, and
    there is no code path that could: the outcome does not exist until after the
    id is chosen and printed.
  * ONE OF THE FIVE CARRIES A REAL INJECTED TOOL FAILURE (the registered
    `rate_limit` fault of `dev-fulfillment-h8-0225`). Recovery is a claim about
    machinery, so the machinery is printed: the fault envelope the model saw
    verbatim, its `recovery_token`, the remediation the registered contract
    demands for that class, the call that attempted it, and the verifier's
    per-fault report. A clean-only demo cannot support a recovery claim.
  * ONE OF THE FIVE IS EXPECTED TO FAIL. `dev-fulfillment-h14-0000` sits at the
    deep-horizon boundary where the measured collapse must not be hidden. If it
    happens to pass, this demo does NOT substitute another task to manufacture a
    failure -- it prints the measured aggregate cliff next to it either way.
  * IT SHOWS THE MACHINERY, NOT A CHAT TRANSCRIPT. Every episode runs through
    `agentlab.suite.evaluate.run_episode`, the same function the registered
    evaluation calls, over `agentlab.suite.runtime.EpisodeRuntime`, the one
    runtime rejection sampling and the verifier use. This file adds no
    environment, no tool, no fault, no scoring and no second success predicate;
    it is a printer.
  * INFRASTRUCTURE FAILURE IS NOT AN EPISODE OUTCOME. A dead, unreachable or
    wrongly-loaded server aborts the demo loudly and is reported as
    infrastructure, never as something the model did.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agentlab import provenance                              # noqa: E402
from agentlab.suite import configio, contract, evaluate, faults  # noqa: E402

BASE_MODEL = "Qwen/Qwen3.5-4B"
DEV_SPECS = "data/suite/v1/certspecs/dev.jsonl"
FROZEN_PROMPT = "prompts/agentic/p2_plan_state_act.txt"
FROZEN_PROMPT_SHA256 = \
    "5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d"
SECRET_FILE = "out/agentic/run_secret.hex"

# ---------------------------------------------------------------------------
# the panel: fixed here, derived from declared rules, never from outcomes
# ---------------------------------------------------------------------------
# `rule` is what derive_panel() re-checks against the committed manifest:
#   first-in-cell        the FIRST task of that (family, horizon) cell in
#                        data/suite/v1/certspecs/dev.jsonl
#   first-in-cell+fault  the first task of that cell whose committed first fault
#                        is the named class
PANEL = (
    {"task_id": "dev-typed_relay-h4-0000", "condition": "clean",
     "family": "typed_relay", "horizon": 4, "rule": "first-in-cell",
     "purpose": "Four-call, all-tools quantitative composition"},
    {"task_id": "dev-lookup_chain-h8-0000", "condition": "clean",
     "family": "lookup_chain", "horizon": 8, "rule": "first-in-cell",
     "purpose": "Eight causally dependent lookups"},
    {"task_id": "dev-fulfillment-h8-0000", "condition": "clean",
     "family": "fulfillment", "horizon": 8, "rule": "first-in-cell",
     "purpose": "Stateful, irreversible, all-five-tools execution"},
    {"task_id": "dev-fulfillment-h8-0225", "condition": "faulted",
     "family": "fulfillment", "horizon": 8, "rule": "first-in-cell+fault",
     "fault_class": "rate_limit",
     "purpose": "First H8 fulfillment task with the registered rate_limit fault"},
    {"task_id": "dev-fulfillment-h14-0000", "condition": "clean",
     "family": "fulfillment", "horizon": 14, "rule": "first-in-cell",
     "purpose": "Deep-horizon boundary where the measured collapse must not be hidden"},
)

# The measured cliff, printed beside the H14 episode whatever it does. Snapshot
# figure from the 13 sealed distillation shards; replace with the final
# sealed-corpus count when it exists, and never silently mix the two.
H14_CLIFF = ("certified success 59/1,152 (5.1%) for the distillation "
             "`fulfillment-h14` cell, 13-shard snapshot of 2026-08-06")

BANNER = """\
================================================================================
 qwen35-agentic-lab -- fixed demonstration panel (SYNTHETIC DEV DEMONSTRATIONS)
================================================================================

WHAT THIS DEMONSTRATES

  * The shipped configuration really runs a tool loop: an OpenAI-compatible
    server, client-side tool execution, the committed five-tool synthetic suite,
    the registered observation form (canonical envelope plus a receipt line on
    every observation), and the exact verifier that decides certified success.
  * Behaviour under ONE REAL INJECTED TOOL FAILURE, printed as machinery: the
    fault envelope the model saw, its recovery token, the remediation the
    registered contract demands, the remedial call, and the verifier's per-fault
    report -- pass or fail.
  * Whatever these five fixed tasks happen to do, including failures.

WHAT THIS DOES NOT DEMONSTRATE

  * This is NOT a benchmark and NOT held-out evidence. These are five synthetic
    DEV demonstrations from the development split, and the development split was
    visible during prompt selection.
  * NOT general browser, shell, web, or arbitrary API competence.
  * NOT reliable execution beyond H8. H14/H20 are outside the supported
    reliability envelope and were measured-only cells.
  * NOT recovery from network outages, arbitrary exceptions, or tools that do not
    implement this repository's remediation contract. The faults here are
    deliberately injected under that contract: transient, rate-limit and
    malformed errors expose a recovery token that must be returned in the
    remedial call; rate-limit recovery must occur on a later assistant decision;
    wrong-unit recovery uses a corrected conversion target and no token.
  * NOT adapter superiority over the prompt-only base. If both model ids run
    here, they are shown side by side on identical tasks; neither is claimed
    better on this evidence.
  * NOT a 100% rate if all five of these illustrative examples happen to pass.
  * NO preregistered gate is computed here, and no capability claim follows from
    this output. The registered 7,800-episode evaluation is a separate stage.
"""


# ---------------------------------------------------------------------------
# inputs, each refused rather than guessed
# ---------------------------------------------------------------------------

def frozen_prompt(root: pathlib.Path) -> tuple[str, str]:
    """The frozen winning prompt, pinned by the hash the preregistration names."""
    path = root / FROZEN_PROMPT
    raw = path.read_text(encoding="utf-8")
    sha = provenance.observation_digest(raw)
    if sha != FROZEN_PROMPT_SHA256:
        raise SystemExit(
            f"REFUSED: {FROZEN_PROMPT} hashes to {sha}, not the frozen winner "
            f"{FROZEN_PROMPT_SHA256}. The shipped configuration IS the base "
            f"model plus these exact bytes; a different prompt is a different "
            f"configuration and must not be demonstrated as this one.")
    return raw.strip(), sha


def derive_panel(specs: list[dict]) -> list[tuple[dict, dict]]:
    """Re-derive every panel id from the committed manifest, or refuse.

    The point is that the panel is a consequence of declared rules over the
    committed dev manifest -- not a list someone chose after seeing results. If
    the manifest ever disagrees with the ids hard-coded above, this refuses
    instead of quietly demonstrating some other task.
    """
    by_id = {s["task_id"]: s for s in specs}
    out: list[tuple[dict, dict]] = []
    for row in PANEL:
        cell = [s for s in specs
                if s.get("family") == row["family"]
                and int(s.get("horizon") or 0) == row["horizon"]]
        if row["rule"] == "first-in-cell":
            derived = cell[0]["task_id"] if cell else None
        else:
            want = row["fault_class"]
            derived = next(
                (s["task_id"] for s in cell
                 if ((s.get("spec_row") or {}).get("faults") or [{}])[0]
                 .get("fault_type") == want), None)
        if derived != row["task_id"]:
            raise SystemExit(
                f"REFUSED: panel entry {row['task_id']} does not satisfy its "
                f"declared selection rule ({row['rule']}"
                + (f", {row.get('fault_class')}" if row.get("fault_class") else "")
                + f") over {DEV_SPECS}: that rule selects {derived!r}. A demo "
                f"panel whose ids no longer follow from a declared rule is an "
                f"outcome-selected panel.")
        spec = by_id[row["task_id"]]
        committed = (spec.get("spec_row") or {}).get("faults") or []
        need = contract.CONDITION_FAULTS[row["condition"]]
        if len(committed) < need:
            raise SystemExit(
                f"REFUSED: {row['task_id']} carries {len(committed)} committed "
                f"faults, condition {row['condition']!r} needs {need}. A "
                f"condition is never satisfied by inventing a fault the "
                f"generator did not commit.")
        out.append((row, spec))
    return out


# ---------------------------------------------------------------------------
# the printer
# ---------------------------------------------------------------------------

def _indent(text: str, pad: str = "      ") -> str:
    return "\n".join(pad + line for line in str(text).splitlines()) or pad


def _args_json(arguments) -> str:
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(arguments)


def print_panel(panel: list[tuple[dict, dict]], models: list[str]) -> None:
    print("FIXED PANEL (chosen before the server was contacted; nothing here is")
    print("outcome-selected, and every row below is reported whatever it does):")
    print()
    for i, (row, spec) in enumerate(panel, 1):
        fault = ((spec.get("spec_row") or {}).get("faults") or [None])[0]
        note = ""
        if row["condition"] != "clean" and fault:
            note = (f"  [INJECTED {fault['fault_type']} @ {fault['target_node']}"
                    f" params={_args_json(fault.get('params') or {})}]")
        print(f"  {i}. {row['task_id']:<28} condition={row['condition']:<8}"
              f" rule={row['rule']}")
        print(f"     {row['purpose']}{note}")
    print()
    print(f"model ids under test, same panel for each: {', '.join(models)}")
    print()


def print_episode_header(idx: int, total: int, row: dict, spec: dict,
                         model: str, arm: str, prompt_sha: str) -> None:
    print()
    print("=" * 80)
    print(f"EPISODE {idx}/{total}  {row['task_id']}  condition={row['condition']}")
    print(f"  model={model}  arm={arm}  purpose: {row['purpose']}")
    print(f"  family={spec.get('family')} horizon={spec.get('horizon')} "
          f"all_tools_required={bool(spec.get('all_tools_required'))} "
          f"template_cluster_id={spec.get('template_cluster_id')}")
    budgets = contract.budgets_for(int(spec.get("horizon") or 0), row["condition"])
    print(f"  budgets: max_decisions={budgets['max_decisions']} "
          f"max_calls={budgets['max_calls']}")
    print(f"  system prompt: {FROZEN_PROMPT} (sha256 {prompt_sha[:12]}…), sent "
          f"verbatim as message 0")
    committed = (spec.get("spec_row") or {}).get("faults") or []
    kept = committed[:contract.CONDITION_FAULTS[row["condition"]]]
    if not kept:
        print("  faults scheduled for this episode: NONE (clean condition)")
    for f in kept:
        cls = f["fault_type"]
        print(f"  fault scheduled: class={cls} target_node={f['target_node']} "
              f"params={_args_json(f.get('params') or {})}")
        print(f"    remediation the registered contract will demand: "
              f"{_args_json(faults.remediation_requirement(cls))}")
        print("    (the fault fires only on a credit-eligible call that reaches "
              "the target node: wrong calls do not consume it)")
    print("-" * 80)


def print_transcript(trace: dict) -> None:
    """The conversation, with the environment-side event for every dispatch."""
    events = list(trace.get("events") or [])
    ev_i = 0
    for msg in trace.get("messages") or []:
        role = msg.get("role")
        if role == "system":
            print("  [system] frozen winning prompt, "
                  f"{len(msg.get('content') or '')} chars (not repeated here)")
            continue
        if role == "user":
            print("  [user]")
            print(_indent(msg.get("content") or ""))
            continue
        if role == "assistant":
            content = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            print(f"  [assistant] decision, {len(calls)} tool call(s)")
            if content.strip():
                print(_indent(content))
            for c in calls:
                fn = c.get("function") or {}
                print(f"      -> CALL {fn.get('name')}"
                      f"({_args_json(fn.get('arguments'))})")
            continue
        if role == "tool":
            event = events[ev_i] if ev_i < len(events) else {}
            ev_i += 1
            print(f"  [tool:{msg.get('name')}] observation the model read, "
                  f"verbatim (call_id={event.get('call_id')}, "
                  f"decision={event.get('decision_id')}):")
            print(_indent(msg.get("content") or ""))
            print(f"      credited={event.get('credited')} "
                  f"oracle_node={event.get('oracle_node')} "
                  f"ok={event.get('ok')} "
                  f"exposed_canonical={event.get('exposed_canonical')}")
            if event.get("fault_triggered"):
                cls = event.get("fault_type")
                print(f"      !! FAULT INJECTED HERE: class={cls}. The bytes "
                      f"above are the fault envelope, NOT the canonical result.")
                print(f"         recovery_token emitted: "
                      f"{event.get('recovery_token')}")
                print(f"         remediation required: "
                      f"{_args_json(faults.remediation_requirement(cls))}")
            elif event.get("rate_limited"):
                print("      !! still rate-limited: the contract requires the "
                      "remedial call on a LATER assistant decision")
            if event.get("token_provided"):
                print(f"      recovery_token echoed by the model: "
                      f"{event.get('token_provided')}  (this is the remediation "
                      f"evidence the verifier looks for)")
            if event.get("replay"):
                print("      idempotent replay of an earlier mutation")
            if event.get("unsafe"):
                print("      !! UNSAFE state mutation outside the oracle plan")
            continue
        print(f"  [{role}] {msg.get('content')}")


def print_verdict(trace: dict, row: dict) -> None:
    verdict = trace.get("verdict") or {}
    score = trace.get("score") or {}
    runner = trace.get("runner") or {}
    print("-" * 80)
    print("  VERIFIER VERDICT (agentlab.suite.verify.verify_episode -- the one")
    print("  certified-success predicate; this demo defines no other):")
    print(f"    termination_reason={runner.get('termination_reason')} "
          f"decisions={runner.get('n_decisions')} calls={runner.get('n_calls')} "
          f"wall_s={runner.get('wall_s')}")
    print(f"    certified_success = {verdict.get('certified_success')}")
    print(f"    answer_ok={verdict.get('answer_ok')} "
          f"nodes={verdict.get('unique_valid_nodes')}/{verdict.get('nodes_total')} "
          f"within_budget={verdict.get('within_budget')} "
          f"state_ok={verdict.get('state_ok')} tokens_ok={verdict.get('tokens_ok')}")
    print(f"    receipts_ok={verdict.get('receipts_ok')} "
          f"consistent={verdict.get('consistent')} "
          f"unsafe_mutation={verdict.get('unsafe_mutation')} "
          f"runaway={verdict.get('runaway')} "
          f"hallucinated={verdict.get('hallucinated')}")
    for rep in verdict.get("fault_reports") or []:
        print(f"    fault report: {rep.get('fault_type')}@{rep.get('target_node')}"
              f" triggered={rep.get('triggered')} "
              f"attempted={rep.get('attempted')} "
              f"recovered={rep.get('recovered')} reason={rep.get('reason')}")
        print(f"      fired at decision {rep.get('fault_decision')}; "
              f"certifying recovery call_id={rep.get('recovery_call_id')} "
              f"at decision {rep.get('recovery_decision')}")
        if rep.get("reason") == "blind_retry":
            print("      blind_retry: the canonical value arrived without a "
                  "qualifying remediation event, so it is NOT certified recovery")
        if rep.get("reason") == "not_exposed":
            print("      not_exposed: the scheduled fault never fired, because "
                  "no credit-eligible call reached its target node")
    if verdict.get("fault_assigned"):
        print(f"    recovery_attempted={verdict.get('recovery_attempted')} "
              f"recovered={verdict.get('recovered')} "
              f"recovery_success={verdict.get('recovery_success')} "
              f"reason={verdict.get('recovery_reason')}")
    for reason in verdict.get("reasons") or []:
        print(f"    reason: {reason}")
    print(f"    ledger cross-check: raw_success={score.get('raw_success')} "
          f"certified_success={score.get('certified_success')} "
          f"verdict_agrees={score.get('verdict_agrees')}")
    if "recovery" in score:
        print(f"    recovery certification: {_args_json(score.get('recovery'))}")
    if "orchestration" in score:
        print(f"    orchestration certification: "
              f"{_args_json(score.get('orchestration'))}")
    if row["family"] == "fulfillment" and row["horizon"] >= 14:
        print()
        print(f"    **Descriptive only; not a preregistered claim:** {H14_CLIFF}.")
        print("    This estimate carries no registered decision threshold, does")
        print("    not change or replace any original gate, and must not be read")
        print("    as a confirmatory claim about training efficacy or general")
        print("    agentic capability. It is printed here whether this single")
        print("    episode passed or failed: no task was substituted to")
        print("    manufacture a failure, and none was dropped to hide one.")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(args) -> int:
    root = pathlib.Path(args.root or ROOT)
    cfg = configio.load_config()
    engine = configio.engine_contract(cfg)
    dec = cfg.get("eval_decoding") or {}
    decode = {"temperature": float(dec.get("temperature", 0.0)),
              "top_p": float(dec.get("top_p", 1.0)),
              "seed": int(dec.get("seed", 0xA61E0009)),
              "max_tokens": int(dec.get("max_tokens_per_decision", 1024)),
              "enable_thinking": bool(engine["enable_thinking"])}
    prompt, prompt_sha = frozen_prompt(root)
    # READ, never mint. Receipts and recovery tokens are keyed with the run
    # secret, and `load_or_create_secret` would CREATE one if the file were
    # absent -- after which the next study stage would adopt whatever this demo
    # happened to write. A demonstration does not get to author a study input.
    secret_path = root / args.secret_file
    if not secret_path.exists():
        raise SystemExit(
            f"REFUSED: {args.secret_file} does not exist, and this demo will "
            f"not MINT the run secret: receipts and recovery tokens are keyed "
            f"with it, so the next study stage would adopt whatever a demo "
            f"wrote there. Run the chain first, or point --secret-file at a "
            f"throwaway file you created yourself.")
    secret = contract.load_or_create_secret(secret_path)
    specs = evaluate.load_specs(root / DEV_SPECS)
    panel = derive_panel(specs)
    models = args.model or [BASE_MODEL]

    print(BANNER)
    print(f"server: {args.server}")
    print(f"engine contract: dtype={engine['dtype']} "
          f"max_model_len={engine['max_model_len']} "
          f"gpu_memory_utilization={engine['gpu_memory_utilization']} "
          f"max_num_seqs={engine['max_num_seqs']} "
          f"enforce_eager={engine['enforce_eager']} "
          f"enable_thinking={engine['enable_thinking']} (thinking DISABLED)")
    print(f"decode, per request: {_args_json(decode)}")
    print(f"environment contract: {contract.environment_contract_sha256()}")
    print(f"run secret: {args.secret_file} "
          f"(sha256 {provenance.observation_digest(secret.hex())[:12]}…) -- "
          f"receipts and recovery tokens are keyed with it")
    print()
    print(f"frozen winning prompt, {FROZEN_PROMPT} (sha256 {prompt_sha}), sent")
    print("verbatim as message 0 of every episode below. THE SHIPPED")
    print("CONFIGURATION IS THE BASE MODEL PLUS THESE EXACT BYTES; a bare")
    print("request without them is not this configuration:")
    print()
    print(_indent(prompt, "    | "))
    print()
    print_panel(panel, models)

    # A dead or wrongly-loaded server is discovered ONCE, here, before the first
    # episode -- and it is infrastructure, never an episode outcome.
    for model in models:
        evaluate.require_live_server(args.server, model, f"demo panel ({model})")

    from agentlab.suite.runtime import tool_schemas_for_family
    total = len(panel) * len(models)
    idx = 0
    results: list[dict] = []
    for row, spec in panel:
        for model in models:
            idx += 1
            arm = "BP" if model == args.base_id else "TP"
            print_episode_header(idx, total, row, spec, model, arm, prompt_sha)
            print("  tool surface offered (the one model-visible surface, from "
                  "runtime.tool_schemas_for_family):")
            print("      " + ", ".join(
                s["function"]["name"]
                for s in tool_schemas_for_family(spec.get("family"))))
            chat_fn = evaluate.make_http_chat(args.server, model, decode)
            run_meta = {"git_sha": configio.git_sha(), "server_model": model,
                        "requested_model": model, "base_id": args.base_id,
                        "adapter": None if arm == "BP" else args.adapter,
                        # This is a DEMONSTRATION, not a claim-bearing trace set:
                        # it carries no producer manifest and no hardware
                        # fingerprint, so nothing it prints can be mistaken for an
                        # evaluation row or pooled with one.
                        "demo": True}
            try:
                trace = evaluate.run_episode(
                    spec, arm=arm, condition=row["condition"], control="none",
                    secret=secret, fault_seed=evaluate.DEFAULT_FAULT_SEED,
                    system_prompt=prompt,
                    prompt_meta={"path": FROZEN_PROMPT, "sha256": prompt_sha},
                    chat_fn=chat_fn, decode=decode, run_meta=run_meta,
                    wall_limit_s=args.episode_wall_s)
            except evaluate.TransportFailure as exc:
                print()
                print("  INFRASTRUCTURE FAILURE, not an episode outcome: "
                      f"{exc.kind}")
                print(_indent(str(exc)))
                print("  No outcome is recorded for this episode and none is")
                print("  attributed to the model. Fix the server and re-run the")
                print("  whole panel; a demo that scored its own transport")
                print("  failures would be reporting harness bugs as model")
                print("  behaviour.")
                return 3
            print_transcript(trace)
            print_verdict(trace, row)
            verdict = trace.get("verdict") or {}
            results.append({
                "task_id": row["task_id"], "condition": row["condition"],
                "model": model,
                "certified_success": bool(verdict.get("certified_success")),
                "termination": (trace.get("runner") or {}).get("termination_reason"),
                "fault": verdict.get("recovery_reason"),
                "recovery_success": verdict.get("recovery_success")})

    print()
    print("=" * 80)
    print("PANEL SUMMARY -- every fixed row, whatever it did")
    print("=" * 80)
    for r in results:
        line = (f"  {r['task_id']:<28} {r['condition']:<8} {r['model']:<20} "
                f"certified_success={str(r['certified_success']):<5} "
                f"termination={r['termination']}")
        if r["fault"] is not None:
            line += (f"  recovery={r['fault']} "
                     f"recovery_success={r['recovery_success']}")
        print(line)
    n_ok = sum(1 for r in results if r["certified_success"])
    print()
    print(f"  {n_ok}/{len(results)} certified. That fraction is a property of "
          f"these five fixed synthetic dev")
    print("  tasks and nothing else: it is not a rate, not a benchmark score, "
          "not held-out")
    print("  evidence, and it establishes no preregistered claim. Nothing above "
          "was rerolled,")
    print("  substituted, filtered or reordered, and no episode was selected by "
          "its outcome.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000",
                    help="the OpenAI-compatible server YOU started "
                         "(scripts/serve.sh); this process starts none")
    ap.add_argument("--model", action="append", default=None,
                    help="served model id; repeat to run the SAME panel for the "
                         f"base and the adapter (default: {BASE_MODEL})")
    ap.add_argument("--base-id", default=BASE_MODEL,
                    help="which model id is the base configuration (arm BP); "
                         "any other id is labelled TP")
    ap.add_argument("--adapter", default="out/multiface/rssft-lora",
                    help="adapter path recorded for a non-base model id")
    ap.add_argument("--secret-file", default=SECRET_FILE)
    ap.add_argument("--episode-wall-s", type=float, default=240.0,
                    help="the registered per-episode wall limit; a wall_clock "
                         "termination is printed as such, not hidden")
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
