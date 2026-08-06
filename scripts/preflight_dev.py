#!/usr/bin/env python
"""The smallest credible dev-only preflight: five probes over six committed dev tasks.

This is the council's pre-production coherence check (round 5, "Smallest credible
dev-only preflight"). It does not replace the registered calibration or any
claim-bearing measurement; it establishes that the apparatus is coherent enough
to justify spending those hours. Nothing here is a gate on the study and nothing
here may relax one.

THE SIX TASKS are the council's exact minimum Cartesian coverage -- three
families x clean/faulted x low/high family horizon -- each once clean and once
faulted, 12 live episodes:

    dev-lookup_chain-h2-0000    transient    lookup_chain, low  horizon
    dev-lookup_chain-h12-0102   malformed    lookup_chain, high horizon
    dev-typed_relay-h2-0150     wrong_unit   typed_relay,  low  horizon
    dev-typed_relay-h12-0225    rate_limit   typed_relay,  high horizon
    dev-fulfillment-h4-0102     malformed    fulfillment,  low  horizon
                                             (the ambiguous mutation case)
    dev-fulfillment-h20-0225    rate_limit   fulfillment,  high horizon

They are collected into a DERIVED manifest (`out/preflight/dev6.jsonl`) whose
rows are copied byte for byte out of the committed `certspecs/dev.jsonl`, and the
source file is verified against the committed SHA256SUMS first: the committed dev
data is never edited.

    manifest   derive the six-task manifest and verify it against the commitment
    probe1     CPU  exact 12-row extractor replay + live grammar/scorer replay
    probe2     CPU  oracle-driven fault parity matrix (all four fault classes,
                    evaluation path vs canonical training path, remediation vs
                    bare retry)
    probe3     GPU  live 12-episode HTTP matrix, ONE server startup
    probe4     GPU  tiny offline-RS batch, both prompt variants, ONE engine
    probe5     GPU  view builder + one-optimizer-step SFT canary
    status     summarize the probe result files written so far

Each probe writes `results/agentic/preflight/<probe>.json` -- every check with its
pass/fail and the numbers behind it -- and exits non-zero on the first failing
probe. A failing probe is the point of a preflight: it stops the chain with a
reproducible case instead of spending calibration hours on an incoherent
apparatus.

GPU probes require the registered pin and refuse without it:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECT_GPU=A5000 \
        PYTHONPATH=src .venv/bin/python scripts/preflight_dev.py probe3

Every GPU probe charges its own measured minutes to the registered GPU ledger
through the producer session manifest, and stops every process it starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "out" / "preflight"
RESULTS = ROOT / "results" / "agentic" / "preflight"
DATA_DIR = ROOT / "data" / "suite" / "v1"
CERTSPECS = DATA_DIR / "certspecs"
DEV_SPLIT = "dev"
MANIFEST_PATH = OUT / "dev6.jsonl"
SECRET_FILE = ROOT / "out" / "agentic" / "run_secret.hex"
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-4B")
RUN_ID = os.environ.get("AGENTIC_RUN_ID") or "dev-preflight-v1"

# The neutral registered prompt and the probe-3/4 prompt. p8_combined is the
# candidate the council named for the live matrix; it is NOT the frozen
# tournament winner (the tournament runs after this preflight), so these episodes
# are apparatus evidence under a dev run id and are not a BP/TP arm.
NEUTRAL_PROMPT = ROOT / "prompts" / "agentic" / "p1_minimal.txt"
PROBE_PROMPT = ROOT / "prompts" / "agentic" / "p8_combined.txt"

# (task_id, registered fault class) -- the council's six.
SIX = (
    ("dev-lookup_chain-h2-0000", "transient"),
    ("dev-lookup_chain-h12-0102", "malformed"),
    ("dev-typed_relay-h2-0150", "wrong_unit"),
    ("dev-typed_relay-h12-0225", "rate_limit"),
    ("dev-fulfillment-h4-0102", "malformed"),
    ("dev-fulfillment-h20-0225", "rate_limit"),
)
SIX_IDS = tuple(t for t, _ in SIX)
FAULT_OF = dict(SIX)
CONDITIONS = ("clean", "faulted")

# The 12-episode trace of the run that exposed D1 and the answer-grammar defect.
D1_TRACE = ROOT / "out" / "verify-a5000" / "traces" / "B0.clean.none.jsonl"
D1_SECRET = ROOT / "out" / "verify-a5000" / "secret.hex"
EXTRACTION_FIXTURE = ROOT / "tests" / "suite" / "test_answer_extraction.py"


# ---------------------------------------------------------------------------
# result plumbing
# ---------------------------------------------------------------------------

class Probe:
    """One probe's checks, numbers and notes -> a committed result file."""

    def __init__(self, name: str, title: str) -> None:
        self.name = name
        self.title = title
        self.checks: list[dict] = []
        self.numbers: dict = {}
        self.notes: list[str] = []
        self.started = time.time()

    def check(self, key: str, ok, detail: str = "") -> bool:
        ok = bool(ok)
        self.checks.append({"check": key, "pass": ok, "detail": str(detail)[:4000]})
        print(f"  [{'ok  ' if ok else 'FAIL'}] {key}"
              + (f"  {detail}" if detail else ""), flush=True)
        return ok

    def number(self, key: str, value) -> None:
        self.numbers[key] = value

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  [note] {text}", flush=True)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["pass"] for c in self.checks)

    def finish(self) -> int:
        from agentlab.suite import configio, contract

        payload = {
            "probe": self.name, "title": self.title,
            "pass": self.passed,
            "run_id": RUN_ID,
            "git_sha": configio.git_sha(),
            "config_hash": configio.config_hash(),
            contract.STAMP_FIELD: contract.environment_contract_sha256(),
            "tasks": list(SIX_IDS),
            "elapsed_s": round(time.time() - self.started, 1),
            "finished_at_utc": configio.now_utc(),
            "numbers": self.numbers,
            "checks": self.checks,
            "notes": self.notes,
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{self.name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        failed = [c["check"] for c in self.checks if not c["pass"]]
        print(f"\n=== {self.name.upper()}: "
              f"{'PASS' if self.passed else 'FAIL'} "
              f"({len(self.checks) - len(failed)}/{len(self.checks)} checks) "
              f"-> {path}")
        if failed:
            print("    failed: " + ", ".join(failed))
        return 0 if self.passed else 1


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def committed_sha(rel: str) -> str | None:
    """The SHA256SUMS entry for one certspecs file, or None."""
    sums = CERTSPECS / "SHA256SUMS"
    if not sums.exists():
        return None
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == rel:
            return parts[0]
    return None


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in
            pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def secret_bytes() -> bytes:
    """THE run secret every probe shares, so envelopes are byte-comparable."""
    from agentlab.suite import contract

    return contract.load_or_create_secret(SECRET_FILE)


# ---------------------------------------------------------------------------
# the derived manifest
# ---------------------------------------------------------------------------

def build_manifest(probe: Probe | None = None) -> list[dict]:
    """Copy the six committed dev rows into a derived manifest, verbatim.

    The committed source is verified against its own SHA256SUMS entry first: a
    preflight that silently rewrote the dev data would prove nothing about the
    apparatus the study will run.
    """
    src = CERTSPECS / "dev.jsonl"
    if not src.exists():
        raise SystemExit(f"REFUSED: {src} is missing; run scripts/generate_suite.py "
                         f"and scripts/export_eval_specs.py first.")
    got, want = sha256_file(src), committed_sha("dev.jsonl")
    if probe is not None:
        probe.check("committed_dev_specs_unmodified", want is not None and got == want,
                    f"dev.jsonl sha256 {got[:16]} vs SHA256SUMS {str(want)[:16]}")
    by_id = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task_id"] in FAULT_OF:
            by_id[row["task_id"]] = (row, line)
    missing = [t for t in SIX_IDS if t not in by_id]
    if missing:
        raise SystemExit(f"REFUSED: the committed dev split does not carry "
                         f"{', '.join(missing)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        for task_id in SIX_IDS:
            fh.write(by_id[task_id][1] + "\n")
    rows = [by_id[t][0] for t in SIX_IDS]
    if probe is not None:
        probe.check("derived_manifest_is_the_six_tasks",
                    [r["task_id"] for r in rows] == list(SIX_IDS),
                    f"{len(rows)} rows -> {MANIFEST_PATH}")
        probe.check("derived_rows_are_byte_copies",
                    all(json.dumps(r, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False) == by_id[r["task_id"]][1]
                        for r in rows))
        classes = {r["task_id"]: [f["fault_type"] for f in r["spec_row"]["faults"]]
                   for r in rows}
        probe.check("registered_fault_classes",
                    all(classes[t] == [FAULT_OF[t]] for t in SIX_IDS),
                    json.dumps(classes, sort_keys=True))
        probe.number("manifest_sha256", sha256_file(MANIFEST_PATH))
    return rows


def manifest_rows() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return build_manifest()
    rows = read_jsonl(MANIFEST_PATH)
    if [r["task_id"] for r in rows] != list(SIX_IDS):
        return build_manifest()
    return rows


def dev_bundles() -> dict:
    """The committed dev bundles of the six tasks (the training path's input)."""
    from agentlab.suite.generate import load_bundles

    return {b.spec.task_id: b
            for b in load_bundles(str(DATA_DIR), DEV_SPLIT, task_ids=list(SIX_IDS))}


def bundle_for(bundle, condition: str):
    from agentlab.suite import contract

    return dataclasses.replace(
        bundle, spec=contract.spec_for_condition(bundle.spec, condition))


def cmd_manifest(args) -> int:
    probe = Probe("manifest", "the derived dev-only manifest (six committed tasks)")
    rows = build_manifest(probe)
    from agentlab.suite import contract
    from agentlab.suite.generate import certification_spec

    bundles = dev_bundles()
    same, budget_ok = [], []
    for row in rows:
        derived = certification_spec(bundles[row["task_id"]])
        same.append(json.dumps(derived, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
                    == json.dumps(row, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False))
        spec = bundles[row["task_id"]].spec
        faulted = contract.spec_for_condition(spec, "faulted")
        budget_ok.append(faulted.max_decisions == spec.max_decisions
                         and faulted.max_calls == spec.max_calls
                         and list(faulted.faults) == list(spec.faults))
    probe.check("certspec_rows_equal_the_bundles_they_came_from", all(same))
    probe.check("committed_budget_equals_the_registered_faulted_budget",
                all(budget_ok))
    probe.number("tasks", {t: {"family": r["family"], "horizon": r["horizon"],
                               "fault": FAULT_OF[t]}
                           for t, r in zip(SIX_IDS, rows)})
    probe.number("episodes", len(rows) * len(CONDITIONS))
    return probe.finish()


# ---------------------------------------------------------------------------
# PROBE 1 -- exact extractor replay (CPU only)
# ---------------------------------------------------------------------------

_PRE_FIX_ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\s`*]+)", re.IGNORECASE)
_PRE_FIX_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

# The seven episodes the defect mis-scored, and the four it did not, named
# individually so a grammar change cannot quietly move one between the sets.
RESCUED = ("dev-lookup_chain-h2-0004", "dev-lookup_chain-h2-0006",
           "dev-lookup_chain-h2-0002", "dev-lookup_chain-h2-0001",
           "dev-lookup_chain-h2-0005", "dev-lookup_chain-h2-0008",
           "dev-lookup_chain-h2-0009")
ALREADY_CORRECT = ("dev-lookup_chain-h2-0007", "dev-lookup_chain-h2-0003",
                   "dev-lookup_chain-h2-0011", "dev-lookup_chain-h2-0010")
STILL_WRONG = ("dev-lookup_chain-h2-0000",)


def pre_fix_extract(final_text: str) -> str | None:
    """The defective grammar, verbatim: "before" is measured, never recalled."""
    hits = _PRE_FIX_ANSWER_RE.findall(final_text or "")
    if hits:
        return hits[-1].strip().rstrip(".,;")
    boxed = _PRE_FIX_BOXED_RE.findall(final_text or "")
    if boxed:
        return boxed[-1].strip()
    return None


def _load_fixture_rows() -> dict:
    """`REGRESSION_ROWS` from the committed test module, keyed by task id."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("preflight_extraction_fixture",
                                                  EXTRACTION_FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {tid: (ans, final) for tid, ans, final in mod.REGRESSION_ROWS}


def rescore_d1_rows(rows: list[dict]) -> dict:
    """The exact offline rescore of the 12 recorded episodes."""
    from agentlab import provenance
    from agentlab.suite.schema import extract_committed_answer

    out = []
    for row in rows:
        final = provenance._final_assistant_text(row)
        answer = str(row["answer"])
        kind = row.get("answer_kind", "token")
        before = pre_fix_extract(final)
        after = extract_committed_answer(final)
        exposed = [v for e in row["events"] for v in provenance._exposed_values(e)]
        out.append({
            "task_id": row["task_id"],
            "recorded_raw_success": bool(row["score"]["raw_success"]),
            "recorded_hallucinated": bool(row["score"]["hallucinated"]),
            "before": before,
            "after": after,
            "before_correct": before is not None
                              and provenance._answers_equal(before, answer, kind),
            "after_correct": after is not None
                             and provenance._answers_equal(after, answer, kind),
            "before_sourced": before is not None and any(
                provenance._value_matches(before, v) for v in exposed),
            "after_sourced": after is not None and any(
                provenance._value_matches(after, v) for v in exposed),
            "final": final,
        })
    return {"rows": out}


def cmd_probe1(args) -> int:
    from agentlab import provenance
    from agentlab.suite import verify
    from agentlab.suite.runtime import run_oracle
    from agentlab.suite.schema import extract_committed_answer

    probe = Probe("probe1", "exact 12-row extractor replay (CPU only)")
    probe.note("falsifies: answer grammar, scorer consistency, false "
               "hallucination labelling")

    # ---- leg A: the 12 recorded episodes of the run that exposed the defect --
    if not D1_TRACE.exists():
        probe.check("d1_trace_present", False,
                    f"{D1_TRACE} is missing; the recorded 12-row evidence is "
                    f"required for the discriminating rescore")
        return probe.finish()
    rows = read_jsonl(D1_TRACE)
    probe.check("recorded_run_is_twelve_episodes", len(rows) == 12, f"{len(rows)} rows")
    probe.number("d1_trace_sha256", sha256_file(D1_TRACE))

    fixture = _load_fixture_rows()
    rep = rescore_d1_rows(rows)["rows"]
    by_id = {r["task_id"]: r for r in rep}
    probe.check("committed_fixture_matches_the_recorded_bytes",
                all(fixture.get(r["task_id"], (None, None))[1] == r["final"]
                    for r in rep)
                and set(fixture) == set(by_id),
                f"{len(fixture)} inlined rows")

    before_ok = {r["task_id"] for r in rep if r["before_correct"]}
    after_ok = {r["task_id"] for r in rep if r["after_correct"]}
    probe.check("the_before_tally_is_the_recorded_one",
                all(r["before_correct"] == r["recorded_raw_success"] for r in rep),
                "recorded raw_success == the defective grammar's reading")
    probe.check("before_4_of_12", before_ok == set(ALREADY_CORRECT),
                f"{len(before_ok)}/12 {sorted(before_ok)}")
    probe.check("after_11_of_12",
                after_ok == set(ALREADY_CORRECT) | set(RESCUED),
                f"{len(after_ok)}/12")
    probe.check("exactly_the_seven_named_rows_change",
                after_ok - before_ok == set(RESCUED),
                f"{sorted(after_ok - before_ok)}")
    probe.check("the_one_genuine_failure_stays_a_failure",
                all(not by_id[t]["after_correct"] and by_id[t]["after"] is None
                    for t in STILL_WRONG))
    probe.check("no_row_regresses", not (before_ok - after_ok))

    # the false hallucination label, and its mechanism
    false_labels = [t for t in RESCUED
                    if by_id[t]["recorded_hallucinated"]
                    and not by_id[t]["before_sourced"]
                    and by_id[t]["after_sourced"]]
    probe.check("seven_false_hallucination_labels_clear",
                len(false_labels) == 7 and set(false_labels) == set(RESCUED),
                f"{len(false_labels)} rows: the literal '\\boxed{{x}}' appears in "
                f"no validated observation, the inner token does")
    probe.check("already_correct_rows_were_never_labelled",
                all(not by_id[t]["recorded_hallucinated"] for t in ALREADY_CORRECT))
    probe.number("raw_success_before", len(before_ok))
    probe.number("raw_success_after", len(after_ok))
    probe.number("false_hallucination_labels_before",
                 sum(1 for r in rep if r["recorded_hallucinated"]))
    probe.number("hallucination_labels_after_over_rescued_rows", 0)
    probe.number("rescued_task_ids", sorted(RESCUED))

    # ---- one grammar, three readers -----------------------------------------
    battery = ["ANSWER: 42", "ANSWER: \\boxed{42}", "\\boxed{42}",
               "ANSWER: \\boxed{abc123}.", "no commitment here",
               "\\boxed{41}\n\nANSWER: 42",
               "The code is \\boxed{correct}\n\nANSWER: wrong",
               "ANSWER: correct\n\nANSWER: \\boxed{wrong}"]
    texts = battery + [r["final"] for r in rep]
    probe.check("certification_layer_reads_the_same_commitment",
                all(provenance.extract_final_answer(t) == extract_committed_answer(t)
                    for t in texts))

    class _S:
        answer_kind = "token"
        answer = "42"

    probe.check("strict_verifier_reads_the_same_commitment",
                all(verify._answer_ok(_S(), t)
                    is (extract_committed_answer(t) is not None
                        and str(extract_committed_answer(t)).lower() == "42")
                    for t in battery))

    # ---- leg B: the same grammar through the LIVE scorer, 12 dev episodes ---
    secret = secret_bytes()
    bundles = dev_bundles()
    grammar_rows = []
    for task_id in SIX_IDS:
        for condition in CONDITIONS:
            bundle = bundle_for(bundles[task_id], condition)
            rt, verdict = run_oracle(bundle.spec, bundle.kb, bundle.nodes,
                                     secret=secret)
            answer = str(bundle.spec.answer)
            forms = {
                "plain": f"ANSWER: {answer}",
                "answer_boxed": f"ANSWER: \\boxed{{{answer}}}",
                "boxed_only": f"the code is \\boxed{{{answer}}}",
                "trailing_dot": f"ANSWER: \\boxed{{{answer}}}.",
                "later_wrong": f"ANSWER: \\boxed{{{answer}}}\n\nANSWER: not-the-code",
                "wrong": "ANSWER: not-the-code",
            }
            scored = {}
            for name, text in forms.items():
                v = rt.verify(text)
                scored[name] = {"certified": bool(v.certified_success),
                                "answer_ok": bool(v.answer_ok),
                                "hallucinated": bool(v.hallucinated)}
            grammar_rows.append({"task_id": task_id, "condition": condition,
                                 "oracle_certified": bool(verdict.certified_success),
                                 "scored": scored})
    probe.check("oracle_certifies_all_twelve_derived_episodes",
                all(r["oracle_certified"] for r in grammar_rows),
                f"{sum(r['oracle_certified'] for r in grammar_rows)}/12")
    good = ("plain", "answer_boxed", "boxed_only", "trailing_dot")
    probe.check("every_correct_grammar_scores_identically",
                all(r["scored"][a]["certified"] and not r["scored"][a]["hallucinated"]
                    for r in grammar_rows for a in good),
                f"{len(good)} forms x 12 episodes")
    probe.check("a_later_wrong_commitment_is_not_rescued",
                all(not r["scored"]["later_wrong"]["certified"]
                    and not r["scored"]["later_wrong"]["answer_ok"]
                    for r in grammar_rows))
    probe.check("a_wrong_commitment_fails",
                all(not r["scored"]["wrong"]["certified"] for r in grammar_rows))
    probe.number("live_grammar_replays",
                 sum(len(r["scored"]) for r in grammar_rows))
    probe.number("live_episodes", len(grammar_rows))
    (OUT / "probe1_grammar.json").write_text(
        json.dumps({"d1_rescore": rep, "live": grammar_rows}, indent=2) + "\n",
        encoding="utf-8")
    return probe.finish()


# ---------------------------------------------------------------------------
# PROBE 2 -- oracle-driven fault parity matrix (CPU only)
# ---------------------------------------------------------------------------

class ScriptedPolicy:
    """A fixed decision script, replayed verbatim on both paths.

    Computed ONCE against a throwaway runtime, so neither path can influence what
    the "model" did: both face identical decisions and identical calls, and any
    difference that survives is the ENVIRONMENT's.

    mode="remediation"  the registered remediation: echo the emitted
                        recovery_token on the re-issued call (on a later decision
                        for rate_limit, since this policy takes one decision per
                        attempt) and re-request the original target unit after a
                        wrong-unit trap.
    mode="bare_retry"   re-issue the identical call WITHOUT the token exactly
                        once, and ACCEPT a wrong-unit result. Operationally this
                        often works; it must never be certified.
    """

    def __init__(self, spec, kb, nodes, secret: bytes, mode: str = "remediation"):
        self.mode = mode
        self.decisions = self._script(spec, kb, nodes, secret, mode)

    @staticmethod
    def _script(spec, kb, nodes, secret: bytes, mode: str) -> list[dict]:
        from agentlab.suite import runtime as rt_mod
        from agentlab.suite.faults import TOKEN_ARG

        rt = rt_mod.EpisodeRuntime(spec, kb, nodes, secret=secret)
        decisions: list[dict] = []
        for node in nodes:
            token = None
            retried_blind = False
            for _ in range(4):
                args = dict(node.args)
                if token is not None and mode == "remediation":
                    args[TOKEN_ARG] = token
                rt.begin_decision()
                decisions.append({"content": f"calling {node.tool}",
                                  "tool_calls": [{"name": node.tool,
                                                  "arguments": args}]})
                text = rt.dispatch(node.tool, args)
                token = rt_mod.recovery_token_in(text)
                if token is not None:
                    if mode == "bare_retry":
                        if retried_blind:
                            break
                        retried_blind = True
                    continue
                body = next((o for o in rt_mod.parse_observation(text)["objects"]
                             if isinstance(o, dict)), None)
                if body is None or not body.get("ok"):
                    break
                if (node.tool == "unit_convert"
                        and str(body.get("unit", "")).strip().lower()
                        != str(node.args["to_unit"]).strip().lower()):
                    if mode == "bare_retry":
                        break          # accept the trapped value
                    continue           # re-request the original target unit
                break
        decisions.append({"content": f"Done.\nANSWER: \\boxed{{{spec.answer}}}",
                          "tool_calls": []})
        return decisions

    def as_chat_fn(self):
        state = {"i": 0}

        def chat_fn(messages, tools):
            i = state["i"]
            state["i"] += 1
            step = (self.decisions[i] if i < len(self.decisions)
                    else {"content": "", "tool_calls": []})
            return {"content": step["content"],
                    "tool_calls": [dict(c) for c in step["tool_calls"]]}

        return chat_fn

    def as_generate_fn(self):
        state = {"i": 0}

        def generate(prompts):
            out = []
            for _ in prompts:
                i = state["i"]
                state["i"] += 1
                step = (self.decisions[i] if i < len(self.decisions)
                        else {"content": "", "tool_calls": []})
                text = step["content"]
                for call in step["tool_calls"]:
                    # sort_keys deliberately OFF: the evaluator receives the
                    # decision as a dict and the rollout engine parses it out of
                    # text, so re-ordering here would inject a difference the
                    # HARNESS made rather than the environment's.
                    payload = json.dumps({"name": call["name"],
                                          "arguments": call["arguments"]})
                    text += f"\n<tool_call>\n{payload}\n</tool_call>"
                out.append((text, "stop"))
            return out

        return generate


PARITY_SYSTEM = ("You can call tools. Echo a recovery_token when a tool error "
                 "supplies one. Finish with ANSWER: <value>.")


def _rendered_prefixes(tok, messages, schemas) -> list[list[int]] | None:
    """Token ids of the rendered prefix BEFORE each assistant decision.

    `tok=None` skips it (returning None on both paths, so the comparison is
    vacuous rather than half-made): the committed CPU test compares the byte and
    ledger surfaces without loading a tokenizer, and the probe itself always
    supplies one -- the rendered prefix is the only surface that catches a
    dropped assistant tool-call object or a missing tool `name`.
    """
    if tok is None:
        return None
    out = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        text = tok.apply_chat_template(messages[:i], tools=schemas, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def _event_surface(events: list[dict]) -> list[dict]:
    """Every recorded event field: the hidden ledger, compared in full."""
    return [dict(e) for e in events]


def _token_surface(events: list[dict]) -> list[dict]:
    return [{"call_id": e["call_id"], "decision_id": e["decision_id"],
             "tool": e["tool"], "token_provided": e["token_provided"],
             "recovery_token": e["recovery_token"],
             "fault_type": e["fault_type"],
             "fault_triggered": e["fault_triggered"],
             "requested_unit": e["requested_unit"]} for e in events]


def training_side(bundle, script, tok, condition: str, secret: bytes) -> dict:
    from agentlab.multidistill import RolloutEngine
    from agentlab.suite import runtime as rt_mod
    from agentlab.suite.configio import load_config

    spec = bundle.spec
    engine = RolloutEngine(load_config(), lambda m, s: m, script.as_generate_fn(),
                           secret=secret)
    convos = engine.rollouts_for([bundle], k_override=1, variants=("canonical",))
    for convo in convos:
        convo["messages"][0]["content"] = PARITY_SYSTEM
    rec = engine.run(convos, verbose=False)[0]
    schemas = rt_mod.tool_schemas_for_family(spec.family)
    return {
        "envelopes": [c["exposed"] for c in rec["calls"]],
        "events": _event_surface(rec["events"]),
        "tokens": _token_surface(rec["events"]),
        "verdict": rec["verdict"],
        "budgets": {"max_decisions": spec.max_decisions,
                    "max_calls": spec.max_calls},
        "progress": rec["parity"]["progress"],
        "episode_digest": rec["parity"]["episode"],
        "rendered": _rendered_prefixes(tok, rec["messages"], schemas),
        "termination": rec["verdict"].get("recovery_reason"),
        "record": rec,
    }


def evaluation_side(spec_row: dict, script, tok, condition: str,
                    secret: bytes) -> dict:
    from agentlab.suite import evaluate
    from agentlab.suite import runtime as rt_mod

    trace = evaluate.run_episode(
        spec_row, arm="B0", condition=condition, control="none", secret=secret,
        fault_seed=0xA61E0007, system_prompt=PARITY_SYSTEM,
        prompt_meta={"path": "-", "sha256": "-"}, chat_fn=script.as_chat_fn(),
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 1024,
                "enable_thinking": False},
        run_meta={"run_id": RUN_ID})
    schemas = rt_mod.tool_schemas_for_family(trace["family"])
    return {
        "envelopes": [c["exposed"] for c in trace["calls"]],
        "events": _event_surface(trace["events"]),
        "tokens": _token_surface(trace["events"]),
        "verdict": trace["verdict"],
        "budgets": trace["budgets"],
        "progress": trace["parity"]["progress"],
        "episode_digest": trace["parity"]["episode"],
        "rendered": _rendered_prefixes(tok, trace["messages"], schemas),
        "termination": trace["runner"]["termination_reason"],
        "trace": trace,
    }


PARITY_FIELDS = ("envelopes", "events", "tokens", "verdict", "budgets",
                 "progress", "episode_digest", "rendered")


def _analyzer_episodes(traces: list[dict], specs_by_id: dict,
                       secret: bytes) -> dict:
    """The analyzer's own episode structure, built the way it builds it.

    Mirrors `analyze.load_agentic_episodes` (including the canonical-verdict
    replay), so what S17 is handed here is what it would be handed by the real
    verdict path over these traces.
    """
    from agentlab import analyze, provenance

    out: dict = {}
    for trace in traces:
        key = (trace.get("arm"), trace.get("condition"),
               trace.get("control", "none"))
        rep = provenance.certify_episode(trace, secret)
        ep = {"task_id": trace["task_id"], "trace": trace, "rep": rep,
              "all_tools_required": bool(trace.get("all_tools_required")),
              "recomputed_verdict": analyze.recompute_canonical_verdict(
                  trace, specs_by_id, secret)}
        if trace.get("condition") in ("faulted", "stress"):
            ep["rec"] = provenance.certify_recovery(trace, secret, rep)
        if ep["all_tools_required"]:
            ep["orch"] = provenance.certify_orchestration(trace, secret, rep)
        out.setdefault(key, {})[ep["task_id"]] = ep
    return out


def _same_decision_episode(specs_by_id: dict, bundles: dict, secret: bytes) -> dict:
    """A CLEAN episode that answers correctly but batches both hops in ONE decision.

    The registered rule is that a dependency edge must cross a LATER assistant
    decision, so the second node is never credited and the strict verifier
    correctly refuses certification -- while the transcript alone (right answer,
    valid receipts, a validated source, no runaway, no fabrication) has nothing to
    object to. It is the fault-free member of the same class as a blind retry.
    """
    from agentlab.suite import evaluate
    from agentlab.suite import runtime as rt_mod

    task_id = "dev-lookup_chain-h2-0000"
    bundle = bundle_for(bundles[task_id], "clean")
    rt = rt_mod.EpisodeRuntime(bundle.spec, bundle.kb, bundle.nodes, secret=secret)
    calls = []
    for node in bundle.nodes:
        rt.begin_decision()
        rt.dispatch(node.tool, dict(node.args))
        calls.append({"name": node.tool, "arguments": dict(node.args)})
    steps = [{"content": "both hops in one decision", "tool_calls": calls},
             {"content": f"Done.\nANSWER: \\boxed{{{bundle.spec.answer}}}",
              "tool_calls": []}]
    state = {"i": 0}

    def chat_fn(messages, tools):
        i = state["i"]
        state["i"] += 1
        step = steps[i] if i < len(steps) else {"content": "", "tool_calls": []}
        return {"content": step["content"],
                "tool_calls": [dict(c) for c in step["tool_calls"]]}

    return evaluate.run_episode(
        specs_by_id[task_id], arm="B0", condition="clean", control="none",
        secret=secret, fault_seed=0xA61E0007, system_prompt=PARITY_SYSTEM,
        prompt_meta={"path": "-", "sha256": "-"}, chat_fn=chat_fn,
        decode={"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 1024,
                "enable_thinking": False},
        run_meta={"run_id": RUN_ID})


def cmd_probe2(args) -> int:
    from agentlab import provenance
    from agentlab.suite import contract

    probe = Probe("probe2", "oracle-driven fault parity matrix (CPU only)")
    probe.note("falsifies: the D2 reconciliation -- one runtime, one fault "
               "contract, one success predicate across evaluation and training")

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:      # the decisive wire-format assertion cannot run
        probe.check("tokenizer_available", False,
                    f"the rendered-prefix comparison needs the {MODEL} tokenizer "
                    f"({exc}); it is the only check that catches a dropped "
                    f"assistant tool-call object or a missing tool name")
        return probe.finish()

    secret = secret_bytes()
    rows = {r["task_id"]: r for r in manifest_rows()}
    bundles = dev_bundles()
    cases = ([(t, "clean", "remediation") for t in SIX_IDS]
             + [(t, "faulted", "remediation") for t in SIX_IDS]
             + [(t, "faulted", "bare_retry") for t in SIX_IDS])

    matrix = []
    diverged = []
    traces_by_mode: dict = {}
    for task_id, condition, mode in cases:
        bundle = bundle_for(bundles[task_id], condition)
        script = ScriptedPolicy(bundle.spec, bundle.kb, bundle.nodes, secret,
                                mode=mode)
        train = training_side(bundle, script, tok, condition, secret)
        script2 = ScriptedPolicy(bundle.spec, bundle.kb, bundle.nodes, secret,
                                 mode=mode)
        ev = evaluation_side(rows[task_id], script2, tok, condition, secret)
        differences = [f for f in PARITY_FIELDS if train[f] != ev[f]]
        if differences:
            diverged.append({"task_id": task_id, "condition": condition,
                             "mode": mode, "fields": differences})
        rec = ev["trace"].get("score", {}).get("recovery")
        traces_by_mode.setdefault(
            "oracle" if mode == "remediation" else mode, []).append(ev["trace"])
        matrix.append({
            "task_id": task_id, "condition": condition, "mode": mode,
            "fault_class": FAULT_OF[task_id] if condition == "faulted" else None,
            "parity_fields_equal": not differences,
            "differences": differences,
            "calls": len(train["events"]),
            "fault_assigned": train["verdict"]["fault_assigned"],
            "faults_triggered": train["verdict"]["faults_triggered"],
            "fault_fire_counts": train["verdict"]["fault_fire_counts"],
            "recovery_reason": train["verdict"]["recovery_reason"],
            "certified_success": train["verdict"]["certified_success"],
            "eval_certified_success": ev["verdict"]["certified_success"],
            "eval_certified_recovery": (rec or {}).get("certified_recovery"),
            "eval_recovery_reason": (rec or {}).get("reason"),
            "eval_termination": ev["termination"],
            "eval_score_raw": ev["trace"]["score"]["raw_success"],
            "eval_score_certified": ev["trace"]["score"]["certified_success"],
            "eval_verdict_agrees": ev["trace"]["score"]["verdict_agrees"],
            "eval_hallucinated": ev["trace"]["score"]["hallucinated"],
        })

    probe.check("evaluation_and_training_face_the_identical_environment",
                not diverged, json.dumps(diverged)[:900] if diverged
                else f"{len(matrix)} episode pairs, {len(PARITY_FIELDS)} surfaces "
                     f"each (envelope bytes, full event ledger, token arguments, "
                     f"verdict, budgets, progress, episode digest, rendered "
                     f"prefix token ids)")
    probe.check("no_spec_errors_on_the_evaluation_path",
                all(r["eval_termination"] != "spec_error" for r in matrix))
    probe.check("the_certifier_agrees_with_the_verdict_everywhere",
                all(r["eval_verdict_agrees"] for r in matrix))
    probe.check("one_success_predicate",
                all(r["certified_success"] == r["eval_certified_success"]
                    == r["eval_score_certified"] for r in matrix))

    clean = [r for r in matrix if r["condition"] == "clean"]
    remed = [r for r in matrix if r["mode"] == "remediation"
             and r["condition"] == "faulted"]
    blind = [r for r in matrix if r["mode"] == "bare_retry"]
    probe.check("clean_episodes_schedule_no_fault",
                all(r["fault_assigned"] == 0 and r["faults_triggered"] == 0
                    and r["certified_success"] for r in clean),
                f"{len(clean)} episodes")
    probe.check("every_scheduled_fault_fires_exactly_once",
                all(r["faults_triggered"] == 1
                    and set(r["fault_fire_counts"].values()) == {1}
                    for r in remed + blind),
                f"{len(remed) + len(blind)} faulted episodes")
    probe.check("registered_remediation_is_certified",
                all(r["recovery_reason"] == "ok" and r["certified_success"]
                    and r["eval_certified_recovery"] for r in remed),
                f"{len(remed)} episodes, classes "
                f"{sorted({r['fault_class'] for r in remed})}")
    from agentlab.suite import verify

    registered_labels = ((set(provenance.NON_RECOVERY_PRECEDENCE)
                          | set(verify.RECOVERY_REASON_ORDER)) - {"ok"})
    probe.check("BARE_RETRIES_ARE_NEVER_CERTIFIED",
                all(r["recovery_reason"] != "ok"
                    and not r["certified_success"]
                    and r["eval_certified_recovery"] is False
                    and r["eval_recovery_reason"] != "ok"
                    and r["eval_recovery_reason"] in registered_labels
                    for r in blind),
                json.dumps({r["fault_class"]: [r["recovery_reason"],
                                               r["eval_recovery_reason"]]
                            for r in blind}, sort_keys=True))
    probe.check("all_four_fault_classes_exercised",
                {r["fault_class"] for r in remed} == {"transient", "rate_limit",
                                                      "malformed", "wrong_unit"})
    stamps = {r.get(contract.STAMP_FIELD) for r in rows.values()}
    probe.check("the_derived_specs_carry_this_build_s_contract",
                stamps == {contract.environment_contract_sha256()},
                f"{sorted(str(s)[:16] for s in stamps)}")

    # ---- what the ANALYZER does with exactly these episodes ----------------
    # The certifier's `verdict_agrees` flag is consumed by the S17
    # HARNESS-INTEGRITY veto, and any BUG there vetoes every gate, claim and the
    # winner. So the parity matrix is only half the question: the other half is
    # whether a legitimately NON-certified episode reads as a harness defect.
    from agentlab import analyze

    s17 = {}
    for label, traces in sorted(traces_by_mode.items()):
        s17[label] = analyze.veto_s17_trace_summary(
            _analyzer_episodes(traces, rows, secret))
    batched = _same_decision_episode(rows, bundles, secret)
    s17["clean_batched_decision"] = analyze.veto_s17_trace_summary(
        _analyzer_episodes([batched], rows, secret))
    probe.number("s17_over_these_episodes",
                 {k: {"status": v["status"], "detail": v["detail"][:300],
                      "numbers": v["numbers"]} for k, v in s17.items()})
    probe.check("S17_accepts_the_registered_oracle_episodes",
                s17["oracle"]["status"] == "OK",
                f"{s17['oracle']['status']}: {s17['oracle']['detail'][:200]}")
    probe.check("S17_DOES_NOT_READ_A_LEGITIMATE_NON_CERTIFIED_EPISODE_AS_A_BUG",
                s17["bare_retry"]["status"] != "BUG"
                and s17["clean_batched_decision"]["status"] != "BUG",
                f"bare_retry -> {s17['bare_retry']['status']}, "
                f"clean same-decision batch -> "
                f"{s17['clean_batched_decision']['status']}; "
                f"{s17['bare_retry']['detail'][:200]}")
    disagreements = [f"{r['task_id']}:{r['condition']}:{r['mode']}"
                     for r in matrix if not r["eval_verdict_agrees"]]
    probe.number("verdict_agrees_disagreements", disagreements)

    probe.number("episode_pairs", len(matrix))
    probe.number("fault_classes",
                 sorted({r["fault_class"] for r in matrix if r["fault_class"]}))
    probe.number("bare_retry_reasons",
                 {r["fault_class"] + ":" + r["task_id"]: r["recovery_reason"]
                  for r in blind})
    probe.number("remediation_reasons",
                 {r["fault_class"] + ":" + r["task_id"]: r["recovery_reason"]
                  for r in remed})
    probe.number("recovery_precedence_labels", provenance.NON_RECOVERY_PRECEDENCE)
    (OUT / "probe2_matrix.json").write_text(
        json.dumps(matrix, indent=2, default=str) + "\n", encoding="utf-8")
    return probe.finish()


# ---------------------------------------------------------------------------
# GPU plumbing shared by probes 3-5
# ---------------------------------------------------------------------------

def require_pin() -> None:
    """The registered pin, or a refusal. No GPU probe defaults a device."""
    order = os.environ.get("CUDA_DEVICE_ORDER")
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if order != "PCI_BUS_ID" or visible != "0":
        raise SystemExit(
            "REFUSED: a GPU probe needs the registered pin.\n"
            "  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "
            "EXPECT_GPU=A5000\n"
            f"  got CUDA_DEVICE_ORDER={order!r} CUDA_VISIBLE_DEVICES={visible!r}")


def gpu_used_mib(index: str = "0") -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def wait_for_baseline(probe: Probe, baseline: int | None, label: str,
                      timeout_s: float = 180.0) -> None:
    """Confirm the pinned card came back to its baseline before returning."""
    if baseline is None:
        return
    deadline = time.time() + timeout_s
    used = gpu_used_mib()
    while used is not None and used > baseline + 200 and time.time() < deadline:
        time.sleep(5)
        used = gpu_used_mib()
    probe.check(f"{label}_card_returned_to_baseline",
                used is not None and used <= baseline + 200,
                f"baseline {baseline} MiB, now {used} MiB")
    probe.number(f"{label}_gpu_used_mib_after", used)


def ledger_note(stage: str, minutes: float, *, kind: str, work: dict,
                started_at: str | None = None, manifest=None) -> float:
    from agentlab.suite.configio import ledger_append

    return ledger_append(stage, minutes, kind=kind, work=work,
                         started_at=started_at, manifest=manifest)


class ServerSession:
    """scripts/serve.sh as the producer authority, always stopped on the way out."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.proc = None
        self.session_id = None
        self.manifest = None
        self.log = OUT / "serve.log"
        self.started_at = None
        self.t0 = None

    def __enter__(self):
        from agentlab.suite import configio

        OUT.mkdir(parents=True, exist_ok=True)
        self.session_id = configio.new_session_id("serve")
        self.manifest = configio.manifest_path("serve", RUN_ID, self.session_id)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = configio.now_utc()
        self.t0 = time.time()
        env = dict(os.environ, PORT=str(self.port), PYTHONPATH="src",
                   AGENTIC_RUN_ID=RUN_ID, AGENTIC_SESSION_ID=self.session_id,
                   RUNTIME_MANIFEST=str(self.manifest))
        fh = self.log.open("w", encoding="utf-8")
        self._fh = fh
        print(f"  [serve] starting vLLM (session {self.session_id}), "
              f"log {self.log}", flush=True)
        self.proc = subprocess.Popen(
            ["bash", "scripts/serve.sh", MODEL], cwd=str(ROOT), env=env,
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        return self

    def wait_ready(self, timeout_s: float = 1200.0) -> bool:
        import requests

        url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    print(f"  [serve] /v1/models answered after "
                          f"{time.time() - self.t0:.0f}s", flush=True)
                    return True
            except Exception:
                pass
            time.sleep(10)
        return False

    def countersign(self) -> dict:
        """The driver's `ready` signature, exactly as the supported chain does."""
        run_cmd([PY, "-m", "agentlab.env", "ready", "--manifest", str(self.manifest),
                 "--run-id", RUN_ID, "--stage", "serve", "--port", str(self.port)])
        from agentlab.suite import configio

        return configio.read_runtime_manifest(self.manifest)

    @property
    def minutes(self) -> float:
        return (time.time() - self.t0) / 60.0

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            print("  [serve] stopping vLLM", flush=True)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            for _ in range(60):
                if self.proc.poll() is not None:
                    break
                time.sleep(1)
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    self.proc.kill()
            self.proc.wait(timeout=60)
        try:
            self._fh.close()
        except Exception:
            pass
        return False


def run_streaming(cmd: list[str], log_path: pathlib.Path) -> int:
    """Run a long GPU command with its output going to a log file."""
    env = dict(os.environ, PYTHONPATH="src", AGENTIC_RUN_ID=RUN_ID,
               TOKENIZERS_PARALLELISM="false")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            raise


def run_cmd(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH="src", AGENTIC_RUN_ID=RUN_ID,
               TOKENIZERS_PARALLELISM="false")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True,
                          capture_output=True, **kw)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def run_soft(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a stage command WITHOUT raising on a non-zero exit.

    `run_cmd` aborts the interpreter, which is right for a setup step whose
    failure means the probe cannot be defined at all (the pin, the countersign).
    It is WRONG for a stage that has already spent GPU minutes: a SystemExit out
    of the `with ServerSession()` body stops the server correctly but skips both
    the ledger append and `probe.finish()`, so a failing GPU probe left NO result
    file and charged NOTHING for the card time it really burned. A preflight's
    whole product is the evidence from the run that failed, so the caller records
    the failure as a check instead of losing it.
    """
    env = dict(os.environ, PYTHONPATH="src", AGENTIC_RUN_ID=RUN_ID,
               TOKENIZERS_PARALLELISM="false")
    return subprocess.run(cmd, cwd=str(ROOT), env=env, text=True,
                          capture_output=True)


# ---------------------------------------------------------------------------
# PROBE 3 -- live 12-episode HTTP matrix, ONE server startup
# ---------------------------------------------------------------------------

def cmd_probe3(args) -> int:
    from agentlab import provenance
    from agentlab.suite import configio, contract
    from agentlab.suite import runtime as rt_mod
    from agentlab.suite.schema import OracleNode, TaskSpec

    require_pin()
    probe = Probe("probe3", "live 12-episode HTTP matrix, one server startup")
    probe.note("falsifies: producer attestation, fail-closed transport, "
               "clean/fault routing, request-contract equality, replay "
               "consistency. Capability success is DIAGNOSTIC here.")
    probe.note(f"prompt {PROBE_PROMPT.name} is a tournament CANDIDATE, not the "
               f"frozen winner: these dev episodes are apparatus evidence under "
               f"run_id {RUN_ID} and are not a BP/TP arm")

    rows_in = {r["task_id"]: r for r in manifest_rows()}
    traces_dir = OUT / "traces"
    for cond in CONDITIONS:
        stale = traces_dir / f"B0.{cond}.none.jsonl"
        if stale.exists():
            stale.unlink()
    baseline = gpu_used_mib()
    probe.number("gpu_used_mib_before", baseline)
    run_cmd([PY, "-m", "agentlab.env", "pin", "--run-id", RUN_ID])
    binding = json.loads(configio.hardware_lock_path().read_text(encoding="utf-8"))
    probe.number("run_binding", {k: binding.get(k) for k in
                                 ("gpu_name", "gpu_uuid", "cuda_visible_bytes",
                                  "driver_version", "pci_bus_id")})

    statuses = []
    with ServerSession() as server:
        up = server.wait_ready()
        if not up:
            tail = "\n".join(server.log.read_text(errors="replace")
                             .splitlines()[-25:])
            probe.check("server_answered_v1_models", False, tail[-1500:])
            return probe.finish()
        probe.check("server_answered_v1_models", True,
                    f"session {server.session_id}")
        manifest = server.countersign()
        probe.check("producer_manifest_is_whole_and_ready",
                    not configio.manifest_gaps(manifest)
                    and bool(manifest["ready_at_utc"]),
                    f"gaps {configio.manifest_gaps(manifest)}")
        probe.number("producer_manifest", {
            k: manifest[k] for k in
            ("run_id", "session_id", "stage", "gpu_name", "gpu_uuid",
             "cuda_visible_bytes", "driver_version", "pci_bus_id", "pid", "port",
             "enable_thinking_effective", configio.MANIFEST_HASH_FIELD)})
        completed = {}
        aborts = []
        for cond in CONDITIONS:
            for attempt in range(1, 21):
                proc = run_soft([
                    PY, "-m", "agentlab.suite.evaluate",
                    "--model", MODEL, "--base-id", MODEL, "--arm", "B0",
                    "--condition", cond, "--control", "none",
                    "--specs", str(MANIFEST_PATH), "--prompt", str(PROBE_PROMPT),
                    "--out", str(traces_dir), "--secret-file", str(SECRET_FILE),
                    "--run-id", RUN_ID,
                    "--server", f"http://127.0.0.1:{server.port}",
                    "--time-budget-s", "360",
                    "--runtime-manifest", str(server.manifest)])
                if proc.returncode != 0:
                    # The evaluator refused to write episodes (a transport abort,
                    # a spec refusal, ...). That refusal IS the finding: record it
                    # with the reproducible detail and stop asking the card for
                    # more, but still charge the minutes and write the result.
                    blob = ((proc.stdout or "") + (proc.stderr or "")).strip()
                    aborts.append({
                        "condition": cond, "attempt": attempt,
                        "returncode": proc.returncode,
                        "output_tail": "\n".join(blob.splitlines()[-40:]),
                    })
                    completed[cond] = False
                    print(f"  [eval] {cond} pass {attempt}: ABORTED rc="
                          f"{proc.returncode}", flush=True)
                    break
                status = json.loads(proc.stdout.strip().splitlines()[-1])
                print(f"  [eval] {cond} pass {attempt}: "
                      f"{json.dumps(status)}", flush=True)
                statuses.append(status)
                completed[cond] = bool(status["complete"])
                if status["complete"]:
                    break
            if aborts:
                break
        minutes = server.minutes
    probe.number("server_session_minutes", round(minutes, 2))
    # The work count is COUNTED, never assumed to be the 12 this probe intended.
    # A ledger row is the receipt calibration divides minutes by: charging 1.26
    # min for "12 episodes" when a transport abort wrote none would overstate
    # measured throughput without bound, and by exactly the factor that makes the
    # projection look affordable.
    written_now = 0
    for _cond in CONDITIONS:
        _p = traces_dir / f"B0.{_cond}.none.jsonl"
        if _p.exists():
            written_now += len(read_jsonl(_p))
    probe.number("episodes_written", written_now)
    cumulative = ledger_note("preflight_eval_http", minutes,
                             kind="server_session",
                             work={"unit": "episodes", "count": written_now,
                                   "intended": 12, "probe": "probe3"},
                             started_at=server.started_at,
                             manifest=server.manifest)
    probe.number("ledger_cumulative_h_after", round(cumulative, 4))
    wait_for_baseline(probe, baseline, "probe3")

    # ---- an evaluator refusal is a finding, not a lost run ------------------
    if aborts:
        transport = {}
        for cond in CONDITIONS:
            tlog = traces_dir / f"B0.{cond}.none.transport.log"
            if tlog.exists():
                # The transport log is APPENDED across sessions on purpose -- it
                # is standing infrastructure evidence, not a per-run scratch file.
                # So this session's finding is the rows THIS session wrote; a
                # count over the whole file would report a previous run's
                # failures as if this server had produced them.
                rows = [r for r in read_jsonl(tlog)
                        if r.get("session_id") == manifest["session_id"]]
                if not rows:
                    continue
                transport[cond] = {
                    "failures": len(rows),
                    "kinds": sorted({r.get("kind") for r in rows}),
                    "task_ids": sorted({r.get("task_id") for r in rows}),
                    "server_health_at_abort":
                        sorted({r.get("server_health_at_abort") for r in rows}),
                    "first_detail": rows[0].get("detail") if rows else None,
                }
        probe.number("evaluator_aborts", aborts)
        probe.number("transport_failures", transport)
        probe.check("NO_EVALUATOR_SHARD_ABORTED", False,
                    f"{len(aborts)} shard abort(s); transport evidence "
                    f"{json.dumps(transport)[:2000]}")
        probe.note("the evaluator fail-closed correctly: it wrote NO episode "
                   "row for any affected task, so nothing entered a denominator "
                   "and resume would re-run exactly those ids")
        return probe.finish()

    # ---- the trace set ------------------------------------------------------
    traces = {}
    for cond in CONDITIONS:
        path = traces_dir / f"B0.{cond}.none.jsonl"
        traces[cond] = read_jsonl(path) if path.exists() else []
    all_rows = traces["clean"] + traces["faulted"]
    probe.check("twelve_episodes",
                len(all_rows) == 12 and all(len(traces[c]) == 6 for c in CONDITIONS),
                f"clean {len(traces['clean'])}, faulted {len(traces['faulted'])}")
    if not all_rows:
        return probe.finish()
    probe.check("all_six_tasks_in_both_conditions",
                all({r["task_id"] for r in traces[c]} == set(SIX_IDS)
                    for c in CONDITIONS))
    probe.check("every_shard_reported_complete",
                all(completed.get(c) for c in CONDITIONS),
                f"{len(statuses)} evaluator invocations, complete {completed}")

    # provenance: complete, identical, and the producer's
    gaps = {r["task_id"]: configio.fingerprint_gaps(r["provenance"])
            for r in all_rows}
    probe.check("every_row_carries_a_complete_S19_fingerprint",
                not any(gaps.values()), json.dumps(gaps))
    ident = configio.fingerprint_identity(all_rows[0]["provenance"])
    conflicts = [configio.fingerprint_conflict(r["provenance"], ident)
                 for r in all_rows]
    probe.check("one_fingerprint_across_all_twelve",
                not any(conflicts), json.dumps(conflicts))
    uuids = {r["provenance"]["gpu_uuid"] for r in all_rows}
    manifests = {r["provenance"]["runtime_manifest_sha256"] for r in all_rows}
    sessions = {r["provenance"]["session_id"] for r in all_rows}
    probe.check("the_uuid_is_the_producer_measured_card",
                uuids == {manifest["gpu_uuid"]} == {binding["gpu_uuid"]},
                f"{sorted(uuids)}")
    probe.check("every_row_points_at_the_producer_session",
                manifests == {manifest[configio.MANIFEST_HASH_FIELD]}
                and sessions == {manifest["session_id"]})
    probe.number("gpu_uuid", sorted(uuids))
    probe.number("runtime_manifest_sha256", sorted(manifests))

    # no spec / parser / server errors
    terminations = {}
    for r in all_rows:
        terminations.setdefault(r["runner"]["termination_reason"], []).append(
            r["task_id"])
    probe.check("no_spec_parser_or_server_errors",
                not ({"spec_error", "parser_budget"} & set(terminations)),
                json.dumps({k: len(v) for k, v in terminations.items()},
                           sort_keys=True))
    probe.number("terminations", {k: len(v) for k, v in terminations.items()})

    # request-contract equality
    prompt_sha = hashlib.sha256(PROBE_PROMPT.read_bytes()).hexdigest()
    engine = configio.engine_contract()
    dec_cfg = configio.load_config()["eval_decoding"]
    decodes = {json.dumps(r["decode"], sort_keys=True) for r in all_rows}
    probe.check("identical_prompt_bytes_on_every_row",
                {r["prompt"]["sha256"] for r in all_rows} == {prompt_sha},
                prompt_sha[:16])
    probe.check("identical_decoding_on_every_row", len(decodes) == 1,
                decodes.pop() if len(decodes) == 1 else str(decodes)[:400])
    probe.check("decoding_is_the_registered_one",
                all(r["decode"]["temperature"] == float(dec_cfg["temperature"])
                    and r["decode"]["top_p"] == float(dec_cfg["top_p"])
                    and r["decode"]["seed"] == int(dec_cfg["seed"])
                    and r["decode"]["max_tokens"] ==
                    int(dec_cfg["max_tokens_per_decision"])
                    for r in all_rows))
    probe.check("thinking_disabled_in_request_and_engine",
                all(r["decode"]["enable_thinking"] is False for r in all_rows)
                and engine["enable_thinking"] is False
                and manifest["enable_thinking_effective"] is False)
    probe.check("engine_fingerprint_is_the_registered_contract",
                all(r["provenance"]["engine_fingerprint"] ==
                    configio.engine_fingerprint() for r in all_rows))
    probe.check("one_environment_contract_stamp",
                {r[contract.STAMP_FIELD] for r in all_rows} ==
                {contract.environment_contract_sha256()})
    probe.check("tool_surface_digest_matches_this_build",
                all(r["tool_schema_sha256"] == provenance.observation_digest(
                    rt_mod.tool_schema_bytes(r["family"])) for r in all_rows))
    probe.check("budgets_are_the_registered_ones",
                all(r["budgets"] == contract.budgets_for(r["horizon"],
                                                         r["condition"])
                    for r in all_rows))

    # clean / fault routing, and exactly-once firing
    clean_bad = [r["task_id"] for r in traces["clean"]
                 if r["faults"] or r["fault"]
                 or any(e["fault_triggered"] or e["fault_type"] for e in r["events"])]
    probe.check("clean_episodes_carry_no_fault_at_all", not clean_bad,
                json.dumps(clean_bad))
    routing = []
    fired_counts = {}
    for r in traces["faulted"]:
        faults = r["faults"] or []
        committed = rows_in[r["task_id"]]["spec_row"]["faults"]
        ok = (len(faults) == 1
              and faults[0]["class"] == FAULT_OF[r["task_id"]]
              and faults[0]["node"] == committed[0]["target_node"]
              and faults[0]["params"] == committed[0]["params"])
        routing.append(ok)
        fires = sum(1 for e in r["events"] if e["fault_triggered"])
        reached = any(e["fault_type"] == FAULT_OF[r["task_id"]]
                      for e in r["events"])
        fired_counts[r["task_id"]] = {
            "fault_triggered_events": fires, "reached": reached,
            "fire_counts": r["verdict"]["fault_fire_counts"],
            "recovery_reason": (r["score"].get("recovery") or {}).get("reason"),
            "certified_recovery": (r["score"].get("recovery") or {}).get(
                "certified_recovery"),
        }
    probe.check("faulted_episodes_carry_exactly_the_committed_fault",
                all(routing), json.dumps(fired_counts, sort_keys=True))
    probe.check("A_REACHED_FAULT_FIRES_EXACTLY_ONCE",
                all(v["fault_triggered_events"] == (1 if v["reached"] else 0)
                    and all(c <= 1 for c in v["fire_counts"].values())
                    for v in fired_counts.values()),
                json.dumps({k: v["fault_triggered_events"]
                            for k, v in fired_counts.items()}))
    probe.number("fault_firing", fired_counts)

    # trace replay consistency
    secret = secret_bytes()
    replay = {}
    for r in all_rows:
        spec_row = rows_in[r["task_id"]]
        spec = contract.spec_for_condition(TaskSpec.from_row(spec_row["spec_row"]),
                                           r["condition"])
        nodes = [OracleNode.from_row(n) for n in spec_row["oracle_nodes"]]
        ok, why = rt_mod.verify_replay(spec, spec_row.get("kb", {}), nodes,
                                       r["calls"], r["parity"], secret=secret)
        replay[f"{r['task_id']}:{r['condition']}"] = why if not ok else "ok"
    probe.check("every_episode_replays_to_identical_digests",
                set(replay.values()) == {"ok"},
                json.dumps({k: v for k, v in replay.items() if v != "ok"}))
    probe.check("the_certifier_agrees_with_every_recorded_verdict",
                all(r["score"]["verdict_agrees"] for r in all_rows))

    # capability: DIAGNOSTIC only
    diag = {
        "raw_success": sum(1 for r in all_rows if r["score"]["raw_success"]),
        "certified_success": sum(1 for r in all_rows
                                 if r["score"]["certified_success"]),
        "clean_certified": sum(1 for r in traces["clean"]
                               if r["score"]["certified_success"]),
        "faulted_certified": sum(1 for r in traces["faulted"]
                                 if r["score"]["certified_success"]),
        "certified_recovery": sum(
            1 for r in traces["faulted"]
            if (r["score"].get("recovery") or {}).get("certified_recovery")),
        "hallucinated": sum(1 for r in all_rows if r["score"]["hallucinated"]),
        "runaway": sum(1 for r in all_rows if r["score"]["runaway"]),
        "decisions": sum(r["runner"]["n_decisions"] for r in all_rows),
        "calls": sum(r["runner"]["n_calls"] for r in all_rows),
        "wall_s": round(sum(r["runner"]["wall_s"] for r in all_rows), 1),
    }
    probe.number("capability_diagnostic", diag)
    probe.note(f"capability (diagnostic, NOT a pass criterion): "
               f"{diag['certified_success']}/12 certified, "
               f"{diag['certified_recovery']}/6 certified recovery")
    return probe.finish()


# ---------------------------------------------------------------------------
# PROBE 4 -- tiny offline-RS batch, ONE engine startup
# ---------------------------------------------------------------------------

RS_DIR = OUT / "rs"
RS_K = 4          # per task/condition: 2 canonical + 2 frozen-variant attempts
RS_INFO = RS_DIR / "engine.json"


def cmd_rs_worker(args) -> int:
    """The GPU half of probe 4, in its own process.

    Separate so the parent can verify that the card came back to baseline: an
    in-process vLLM engine holds the allocation until the interpreter exits, and
    "the card was released" is not something this preflight may assume.
    """
    import argparse as _argparse

    from agentlab import multidistill as md
    from agentlab.suite import configio

    require_pin()
    cfg = configio.load_config()
    bundles = dev_bundles()
    RS_DIR.mkdir(parents=True, exist_ok=True)
    ns = _argparse.Namespace(model=MODEL, run_id=RUN_ID, stage="preflight_rs")
    started = configio.now_utc()
    t_engine = time.time()
    engine = md._vllm_engine(cfg, ns, PROBE_PROMPT.read_text(encoding="utf-8"))
    startup_min = (time.time() - t_engine) / 60.0
    ledger_note("preflight_rs:engine_start", startup_min, kind="engine_start",
                work={"unit": "engine", "count": 1, "probe": "probe4"},
                started_at=started, manifest=engine.manifest_path)
    t0 = time.time()
    written = {}
    for cond in CONDITIONS:
        batch = [bundle_for(bundles[t], cond) for t in SIX_IDS]
        convos = engine.rollouts_for(batch, k_override=RS_K,
                                     variants=("canonical", "frozen"))
        records = engine.run(convos, verbose=True)
        path = RS_DIR / f"{cond}.jsonl"
        md.write_attested_jsonl(path, records,
                                f"preflight rejection sampling ({cond})")
        written[cond] = len(records)
    minutes = (time.time() - t0) / 60.0
    cumulative = ledger_note("preflight_rs", minutes, kind="shard",
                             work={"unit": "rollouts",
                                   "count": sum(written.values()),
                                   "probe": "probe4"},
                             manifest=engine.manifest_path)
    RS_INFO.write_text(json.dumps({
        "engine_startup_min": round(startup_min, 2),
        "rollout_minutes": round(minutes, 2),
        "ledger_cumulative_h_after": round(cumulative, 4),
        "written": written, "k": RS_K,
        "runtime_manifest": str(engine.manifest_path),
        "runtime_manifest_sha256":
            engine.manifest[configio.MANIFEST_HASH_FIELD],
        "session_id": engine.manifest["session_id"],
        "started_at_utc": started,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rollouts": sum(written.values()),
                      "minutes": round(minutes, 2)}))
    return 0


def cmd_probe4(args) -> int:
    from agentlab import multidistill as md
    from agentlab.suite import contract
    from agentlab.suite import runtime as rt_mod
    from agentlab.suite.schema import digest_text

    require_pin()
    probe = Probe("probe4", "tiny offline-RS batch, both prompt variants, one engine")
    probe.note("falsifies: offline chat-template/tool-parsing drift and "
               "train/eval environment divergence")
    probe.note(f"the second prompt variant is {PROBE_PROMPT.name}, a tournament "
               f"CANDIDATE standing in for the frozen winner, which does not "
               f"exist yet by design (the tournament runs after this preflight)")

    secret = secret_bytes()
    bundles = dev_bundles()
    baseline = gpu_used_mib()
    probe.number("gpu_used_mib_before", baseline)
    RS_DIR.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        stale = RS_DIR / f"{cond}.jsonl"
        if stale.exists():
            stale.unlink()
    log = RS_DIR / "rs.log"
    print(f"  [rs] one engine, {len(SIX_IDS)} tasks x {len(CONDITIONS)} "
          f"conditions x {RS_K} samples (log {log})", flush=True)
    rc = run_streaming([PY, str(pathlib.Path(__file__).resolve()), "_rs_worker"],
                       log)
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        probe.check("rs_worker_completed", False, tail[-1500:])
        wait_for_baseline(probe, baseline, "probe4")
        return probe.finish()
    probe.check("rs_worker_completed", True, str(log))
    wait_for_baseline(probe, baseline, "probe4")

    info = json.loads(RS_INFO.read_text(encoding="utf-8"))
    rows_by_cond = {c: read_jsonl(RS_DIR / f"{c}.jsonl") for c in CONDITIONS}
    all_rows = rows_by_cond["clean"] + rows_by_cond["faulted"]
    probe.number("engine_startup_min", info["engine_startup_min"])
    probe.number("rollout_minutes", info["rollout_minutes"])
    probe.number("ledger_cumulative_h_after", info["ledger_cumulative_h_after"])
    probe.number("rollouts", len(all_rows))

    probe.check("expected_rollout_count",
                len(all_rows) == len(SIX_IDS) * len(CONDITIONS) * RS_K,
                f"{len(all_rows)} = 6 tasks x 2 conditions x {RS_K} samples")
    variants = {}
    for cond, recs in rows_by_cond.items():
        for r in recs:
            variants.setdefault(r["prompt_variant"], 0)
            variants[r["prompt_variant"]] += 1
    probe.check("both_prompt_variants_ran",
                set(variants) == {"canonical", "frozen"}
                and len(set(variants.values())) == 1, json.dumps(variants))

    # provenance
    gaps = [md.provenance_gaps(r.get("provenance")) for r in all_rows]
    probe.check("every_rollout_carries_complete_producer_provenance",
                not any(gaps), json.dumps([g for g in gaps if g]))
    ident = md.require_one_producer(all_rows, "the preflight RS batch")
    probe.check("one_gpu_producer_for_the_whole_batch",
                bool(ident.get("gpu_execution"))
                and ident.get("producer") == "preflight_rs"
                and ident.get("runtime_manifest_sha256")
                == info["runtime_manifest_sha256"]
                and ident.get("session_id") == info["session_id"],
                json.dumps({k: ident.get(k) for k in
                            ("producer", "gpu_execution", "gpu_uuid",
                             "session_id", "runtime_manifest_sha256")}))
    probe.number("producer_identity", ident)

    # replay parity
    parity = {}
    for cond, recs in rows_by_cond.items():
        for r in recs:
            bundle = bundle_for(bundles[r["task_id"]], cond)
            ok, why = md.replay_record(r, bundle, secret=secret)
            key = f"{r['task_id']}:{cond}:{r['prompt_variant']}:{r['sample_index']}"
            parity[key] = "ok" if ok else why
    probe.check("every_rollout_replays_to_identical_bytes_and_verdict",
                set(parity.values()) == {"ok"},
                json.dumps({k: v for k, v in parity.items() if v != "ok"})[:1200])

    # harness-level failures (capability outcomes are reported separately)
    failures = {
        "unknown_tool": [r["task_id"] for r in all_rows if r["unknown_tool"]],
        "arg_error": [r["task_id"] for r in all_rows if r["arg_error"]],
    }
    probe.check("no_unknown_tool_or_argument_failures",
                not failures["unknown_tool"] and not failures["arg_error"],
                json.dumps(failures))
    probe.number("capability_diagnostic", {
        "truncated": sum(1 for r in all_rows if r["truncated"]),
        "exhausted": sum(1 for r in all_rows if r["exhausted"]),
        "call_cap": sum(1 for r in all_rows if r["call_cap"]),
        "no_final": sum(1 for r in all_rows if not r["final"]),
        "certified_success": sum(1 for r in all_rows
                                 if r["verdict"]["certified_success"]),
        "recovered": sum(1 for r in rows_by_cond["faulted"]
                         if r["verdict"]["recovered"]),
    })

    # identical fault contracts to the evaluation path
    stamps = {r[contract.STAMP_FIELD] for r in all_rows}
    probe.check("one_environment_contract_with_this_build",
                stamps == {contract.environment_contract_sha256()},
                sorted(stamps)[0][:16])
    probe.check("tool_surface_digest_matches_this_build",
                all(r["tool_schema_sha256"] ==
                    digest_text(rt_mod.tool_schema_bytes(r["family"]))
                    for r in all_rows))
    eval_traces = {}
    for cond in CONDITIONS:
        path = OUT / "traces" / f"B0.{cond}.none.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                eval_traces[(row["task_id"], cond)] = row
    compared, mismatched = 0, []
    for r in rows_by_cond["faulted"]:
        trace = eval_traces.get((r["task_id"], "faulted"))
        if trace is None:
            continue
        rs_fault = next((e for e in r["events"] if e["fault_triggered"]), None)
        ev_fault = next((e for e in trace["events"] if e["fault_triggered"]), None)
        if rs_fault is None or ev_fault is None:
            continue
        compared += 1
        if (rs_fault["exposed_text"] != ev_fault["exposed_text"]
                or rs_fault["recovery_token"] != ev_fault["recovery_token"]
                or rs_fault["fault_type"] != ev_fault["fault_type"]):
            mismatched.append(r["task_id"])
    probe.check("fault_envelope_bytes_equal_the_evaluation_path",
                not mismatched,
                f"{compared} task(s) fired in both paths; mismatched "
                f"{mismatched}" if compared else
                "no task fired in both paths (nothing comparable)")
    probe.number("fault_envelope_comparisons", compared)
    probe.check("the_evaluation_traces_were_available_to_compare_against",
                bool(eval_traces),
                f"{len(eval_traces)} evaluation rows from probe3")
    wait_for_baseline(probe, baseline, "probe4")
    return probe.finish()


# ---------------------------------------------------------------------------
# PROBE 5 -- view builder + one-optimizer-step SFT canary
# ---------------------------------------------------------------------------

CANARY_DIR = OUT / "sft"
CANARY_VIEWS = CANARY_DIR / "canary_views.jsonl"
CANARY_ADAPTER = OUT / "sft-canary-lora"
TRL_COLUMNS = ("prompt", "completion", "tools", "chat_template_kwargs")
ONE_STEP_ROWS = 16      # sft.bsz 2 x sft.accum 8: one optimizer step


def _mask_report(tok, row: dict) -> dict:
    """TRL's own completion-only masking, recomputed through TRL's own helper.

    `trl.data_utils._tokenize` and `SFTTrainer`'s `tokenize_fn` are called here
    with the same arguments the trainer uses, so this is the mask the trainer
    will build, not a re-derivation of it: prompt ids with
    `add_generation_prompt=True`, prompt+completion ids, then
    `completion_mask = [0]*len(prompt_ids) + [1]*rest` and
    `labels[completion_mask == 0] = -100`
    (`trl/trainer/sft_trainer.py`). If the prompt ids are not a PREFIX of the
    prompt+completion ids, TRL only logs a warning and the mask then supervises
    the wrong tokens -- so the prefix property is checked, never assumed.
    """
    from trl.data_utils import _tokenize

    kwargs = {"chat_template": None, "tools": row["tools"],
              **(row.get("chat_template_kwargs") or {})}
    prompt_ids = _tokenize(tok, row["prompt"], add_generation_prompt=True,
                           **kwargs)["input_ids"]
    full_ids = _tokenize(tok, row["prompt"] + row["completion"],
                         **kwargs)["input_ids"]
    prefix_ok = list(full_ids[: len(prompt_ids)]) == list(prompt_ids)
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    supervised = tok.decode([i for i, m in zip(full_ids, mask) if m])
    return {"prompt_tokens": len(prompt_ids),
            "total_tokens": len(full_ids),
            "supervised_tokens": sum(mask),
            "prefix_ok": bool(prefix_ok),
            "supervised_text": supervised}


def cmd_probe5(args) -> int:
    from agentlab import multidistill as md
    from agentlab import sft as sft_mod
    from agentlab.suite import configio
    from agentlab.suite import datasets as ds_mod

    require_pin()
    probe = Probe("probe5", "view builder + one-optimizer-step SFT canary")
    probe.note("falsifies: the training-side provenance seam and dangerous "
               "trainer defaults")

    cfg = configio.load_config()
    secret = secret_bytes()
    bundles = dev_bundles()
    rows_by_cond = {}
    for cond in CONDITIONS:
        path = RS_DIR / f"{cond}.jsonl"
        if not path.exists():
            probe.check("probe4_rollouts_present", False,
                        f"{path} is missing; run probe4 first")
            return probe.finish()
        rows_by_cond[cond] = read_jsonl(path)
    probe.check("probe4_rollouts_present", True,
                f"{sum(len(v) for v in rows_by_cond.values())} rollouts")

    # ---- acceptance: the verified dev canaries ------------------------------
    accepted, reasons = [], {}
    for cond, recs in rows_by_cond.items():
        for r in recs:
            bundle = bundle_for(bundles[r["task_id"]], cond)
            ok, why = md.accept_record(r, cfg, bundles={r["task_id"]: bundle},
                                       secret=secret)
            key = f"{r['task_id']}:{cond}:{r['prompt_variant']}:{r['sample_index']}"
            reasons[key] = "accepted" if ok else why
            if ok:
                accepted.append((key, r))
    probe.number("acceptance", {"accepted": len(accepted),
                                "rejected": len(reasons) - len(accepted)})
    probe.number("rejection_reasons", reasons)
    probe.check("at_least_one_verified_dev_canary", bool(accepted),
                f"{len(accepted)} of {len(reasons)} rollouts accepted by the "
                f"strict verifier + registered budget/recovery filters")
    if not accepted:
        probe.note("the training-side seam cannot be exercised without a "
                   "GPU-attested, verifier-certified canary; a CPU-scripted "
                   "corpus is refused by design (datasets.require_views_chain)")
        return probe.finish()

    # ---- views -------------------------------------------------------------
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    counter = ds_mod.default_token_counter()
    picked, rows, meta, report = [], [], [], {}
    for key, rec in sorted(accepted, key=lambda kv: kv[0]):
        trial = picked + [rec]
        t_rows, t_meta, t_report = ds_mod.build_views(list(trial), counter, cfg)
        if picked and len(t_rows) > ONE_STEP_ROWS:
            break
        picked, rows, meta, report = trial, t_rows, t_meta, t_report
    probe.number("canary_trajectories", [r["task_id"] for r in picked])
    probe.number("view_rows", len(rows))
    probe.number("view_counts", report["view_counts"])
    probe.number("terminal_weight", report["terminal_weight"])
    probe.note(f"{len(picked)} of {len(accepted)} accepted canaries kept so the "
               f"corpus stays inside one optimizer step "
               f"({ONE_STEP_ROWS} = sft.bsz {cfg['sft']['bsz']} x sft.accum "
               f"{cfg['sft']['accum']}); the corpus-level terminal-weight gate "
               f"applies to the production corpus, not to this canary")

    probe.check("training_rows_carry_exactly_the_four_TRL_columns",
                all(tuple(sorted(r)) == tuple(sorted(TRL_COLUMNS)) for r in rows),
                f"{len(rows)} rows, columns "
                f"{sorted(rows[0]) if rows else []}")
    probe.check("view_metadata_is_one_to_one_with_the_rows",
                len(meta) == len(rows))
    checked = ds_mod.require_views_chain(rows, meta, report,
                                         require_gpu_source=True)
    probe.check("the_view_chain_names_one_attested_gpu_producer",
                bool(checked["source_provenance"].get("gpu_execution")),
                json.dumps({k: checked["source_provenance"].get(k) for k in
                            ("producer", "gpu_uuid", "session_id",
                             "runtime_manifest_sha256")}))

    # completion-only masking, recomputed exactly as TRL does it
    from agentlab import env as labenv

    tok = labenv.get_tokenizer(labenv.load_processor(MODEL))
    masks = [_mask_report(tok, r) for r in rows]
    probe.check("every_completion_is_exactly_one_assistant_message",
                all(len(r["completion"]) == 1
                    and r["completion"][0]["role"] == "assistant" for r in rows))
    probe.check("the_prompt_is_a_token_prefix_of_prompt_plus_completion",
                all(m["prefix_ok"] for m in masks),
                "TRL masks by prompt length; a non-prefix tokenization would "
                "train on the wrong tokens")
    probe.check("only_the_final_assistant_turn_is_supervised",
                all(m["supervised_tokens"] > 0 for m in masks)
                and not any("receipt: r-" in m["supervised_text"] for m in masks)
                and not any("<tool_response>" in m["supervised_text"]
                            for m in masks),
                f"supervised tokens per row: "
                f"{[m['supervised_tokens'] for m in masks]}")
    probe.check("no_view_exceeds_the_registered_view_budget",
                all(m["total_tokens"] <= int(cfg["acceptance"]["max_view_tokens"])
                    for m in masks),
                f"max {max(m['total_tokens'] for m in masks)} of "
                f"{cfg['acceptance']['max_view_tokens']}")
    probe.number("view_tokens", [m["total_tokens"] for m in masks])
    probe.number("supervised_tokens", [m["supervised_tokens"] for m in masks])

    with CANARY_VIEWS.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with CANARY_VIEWS.with_suffix(".meta.jsonl").open("w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    CANARY_VIEWS.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---- the trainer's dangerous defaults, before the card is touched ------
    from trl import SFTConfig

    ap = sft_mod.build_parser(cfg["sft"])
    train_args = ap.parse_args(["--model", MODEL,
                                "--distill-path", str(CANARY_VIEWS),
                                "--out", str(CANARY_ADAPTER),
                                "--run-id", RUN_ID])
    kwargs = sft_mod.sft_config_kwargs(train_args)
    sft_cfg = SFTConfig(**kwargs)
    probe.check("eval_batch_is_one_and_loss_only",
                kwargs["per_device_eval_batch_size"] == 1
                and kwargs["prediction_loss_only"] is True
                and kwargs["eval_accumulation_steps"] == 1,
                f"eval_bsz {kwargs['per_device_eval_batch_size']}, "
                f"prediction_loss_only {kwargs['prediction_loss_only']}")
    probe.check("completion_only_loss_is_left_to_the_dataset_shape",
                sft_cfg.completion_only_loss is None
                and all("prompt" in r and "completion" in r for r in rows),
                "SFTTrainer resolves completion_only_loss=True for a "
                "prompt-completion dataset")
    probe.check("packing_off_and_length_matches_the_view_budget",
                kwargs["packing"] is False
                and kwargs["max_length"] == int(cfg["acceptance"]["max_view_tokens"]),
                f"max_length {kwargs['max_length']}, packing {kwargs['packing']}")

    # ---- the one-step canary ----------------------------------------------
    baseline = gpu_used_mib()
    probe.number("gpu_used_mib_before", baseline)
    if CANARY_ADAPTER.exists():
        import shutil

        shutil.rmtree(CANARY_ADAPTER)
    receipt = sft_mod.training_manifest_path(CANARY_ADAPTER)
    if receipt.exists():
        receipt.unlink()
    started = configio.now_utc()
    t0 = time.time()
    sft_log = CANARY_DIR / "sft.log"
    rc = run_streaming([PY, "-m", "agentlab.sft", "--model", MODEL,
                    "--distill-path", str(CANARY_VIEWS),
                    "--out", str(CANARY_ADAPTER), "--run-id", RUN_ID,
                    "--rank", str(cfg["sft"]["lora_rank"]),
                    "--lora-alpha", str(cfg["sft"]["lora_alpha"]),
                    "--lora-dropout", str(cfg["sft"]["lora_dropout"]),
                    "--lr", str(cfg["sft"]["lr"]),
                    "--epochs", str(cfg["sft"]["epochs"]),
                    "--bsz", str(cfg["sft"]["bsz"]),
                    "--accum", str(cfg["sft"]["accum"]),
                    "--eval-bsz", str(cfg["sft"]["eval_bsz"]),
                    "--eval-accumulation-steps",
                    str(cfg["sft"]["eval_accumulation_steps"]),
                    "--prediction-loss-only",
                    str(cfg["sft"]["prediction_loss_only"]),
                    "--gradient-checkpointing",
                    str(cfg["sft"]["gradient_checkpointing"]),
                    "--packing", str(cfg["sft"]["packing"]),
                    "--max-length", str(cfg["sft"]["max_length"])], sft_log)
    minutes = (time.time() - t0) / 60.0
    probe.number("sft_minutes", round(minutes, 2))
    if rc != 0:
        tail = "\n".join(sft_log.read_text(errors="replace").splitlines()[-30:])
        probe.check("sft_canary_completed", False, tail[-1800:])
        wait_for_baseline(probe, baseline, "probe5")
        return probe.finish()
    probe.check("sft_canary_completed", True, str(sft_log))

    probe.check("checkpoint_written",
                (CANARY_ADAPTER / "adapter_model.safetensors").exists(),
                str(CANARY_ADAPTER))
    manifest = sft_mod.read_training_manifest(receipt)
    missing = [k for k in sft_mod.TRAINING_MANIFEST_FIELDS
               if manifest.get(k) is None or manifest.get(k) == ""]
    probe.check("trainer_manifest_is_complete", not missing, json.dumps(missing))
    tree = configio.checkpoint_tree_sha256(CANARY_ADAPTER)
    probe.check("checkpoint_hash_matches_the_bytes_on_disk",
                tree == manifest["checkpoint"]["checkpoint_sha256"],
                f"{str(tree)[:16]}")
    bound = configio.ledger_bound_uuid(cfg)
    binding = json.loads(configio.hardware_lock_path().read_text(encoding="utf-8"))
    probe.check("ledger_uuid_matches_the_training_card",
                manifest["hardware"]["gpu_uuid"] == bound == binding["gpu_uuid"]
                == manifest["ledger"]["gpu_uuid"],
                f"trainer {manifest['hardware']['gpu_uuid']}, ledger {bound}, "
                f"run binding {binding['gpu_uuid']}")
    probe.check("the_receipt_passes_the_checkpoint_locks_own_gate",
                bool(sft_mod.require_training_manifest(
                    receipt, checkpoint_path=CANARY_ADAPTER, cfg=cfg,
                    stage="rs_sft")))
    probe.check("one_optimizer_step", manifest["optimizer_steps"] == 1,
                f"{manifest['optimizer_steps']} step(s) over "
                f"{manifest['train_rows']} train rows")
    probe.check("the_corpus_the_receipt_names_is_the_corpus_on_disk",
                manifest["inputs"]["views_sha256"] == sha256_file(CANARY_VIEWS)
                and manifest["inputs"]["views_rows"] == len(rows))
    probe.check("the_receipt_inherits_the_rollout_producer",
                manifest["inputs"]["source_provenance"]["session_id"]
                == checked["source_provenance"]["session_id"]
                and manifest["inputs"]["source_provenance"]["gpu_uuid"]
                == checked["source_provenance"]["gpu_uuid"])
    probe.number("training_manifest", {
        "run_id": manifest["run_id"], "stage": manifest["stage"],
        "optimizer_steps": manifest["optimizer_steps"],
        "train_rows": manifest["train_rows"], "eval_rows": manifest["eval_rows"],
        "checkpoint_sha256": manifest["checkpoint"]["checkpoint_sha256"],
        "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
        "session_id": manifest["session_id"],
        "gpu_uuid": manifest["hardware"]["gpu_uuid"],
        "ledger": manifest["ledger"],
        "source_sessions": manifest["inputs"]["source_sessions"],
    })
    cumulative = ledger_note("preflight_sft_canary", minutes, kind="stage",
                             work={"unit": "optimizer_steps",
                                   "count": manifest["optimizer_steps"],
                                   "rows": manifest["train_rows"],
                                   "probe": "probe5"},
                             started_at=started,
                             manifest=manifest["runtime_manifest_path"])
    probe.number("ledger_cumulative_h_after", round(cumulative, 4))
    wait_for_baseline(probe, baseline, "probe5")
    return probe.finish()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    names = ("manifest", "probe1", "probe2", "probe3", "probe4", "probe5")
    worst = 0
    for name in names:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"  {name:<9} not run")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        failed = [c["check"] for c in rec["checks"] if not c["pass"]]
        print(f"  {name:<9} {'PASS' if rec['pass'] else 'FAIL'}  "
              f"{len(rec['checks']) - len(failed)}/{len(rec['checks'])} checks  "
              f"{rec['finished_at_utc']}")
        if failed:
            worst = 1
            print(f"            failed: {', '.join(failed)}")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("manifest", cmd_manifest), ("probe1", cmd_probe1),
                     ("probe2", cmd_probe2), ("probe3", cmd_probe3),
                     ("probe4", cmd_probe4), ("probe5", cmd_probe5),
                     ("_rs_worker", cmd_rs_worker), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
