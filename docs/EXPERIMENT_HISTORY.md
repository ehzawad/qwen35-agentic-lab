# Experiment history

Compact record of the round-1 GSM8K tool-loop study, the experiments excluded from
the shipped tree, and any round-2 measurement whose result the apparatus itself
changed. Everything summarized here from round 1 remains fully recoverable at the
pre-cleanup commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f` (`git show <sha>:<path>`).

## Excluded experiments

Single-turn xlam SFT reduced held-out GSM8K accuracy to 1/20 and ran away on 19/20
episodes because its corpus contained tool calls but no post-tool termination turns —
a 16x degradation of the base model. The reward-model run ended near chance; its
adapter-only checkpoint also reinitializes the trained score head on reload, making
the saved artifact unsafe for reuse. DPO and BudgetEnv proved trainer plumbing only:
DPO had no quality evaluation, and BudgetEnv saturated at reward 1.0 with zero
variance, validating the interface rather than any learning. Their runnable stages
and raw artifacts were removed from current `main`; the original record remains in
commits `95b8326` and `3ab3201`.

## Retired-but-validated GSM8K results

Held-out GSM8K, n=200 per arm, paired by problem; harness sanity checks clean and
all five preregistered gates passed. Full evidence (eval JSONs, traces, corpus,
verdict, analyzer) lives at commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f`.

| Arm | Accuracy | Statistics |
|---|---:|---|
| Base (default prompt) | 0.810 | reference arm |
| RS-SFT (rejection-sampled distillation) | 0.920 | vs base: paired McNemar p<0.001 |
| Concise-prompt control (zero training) | 0.900 | recovers 81.8% of the observed RS-SFT gain |
| RS-GRPO (GRPO after RS-SFT) | 0.930 | vs RS-SFT: +0.010, null (p=0.804) |

The controlling lesson: most of the measured "training gain" was elicitation of
latent behaviour, so every future capability claim requires a best-prompt base
control, and GRPO after successful SFT produced an informative null.

## Deferred — recorded so they are not rediscovered; neither is planned

- **RM-in-the-loop RLOO Goodhart experiment**: RLOO against a real 0.8B reward
  model, evaluated as a Goodhart curve. Blocked on a checkpoint fix — the RM must
  be merged via `merge_and_unload()` with a pre/post-merge logit-parity assertion
  (the adapter-only checkpoint silently reinitializes its score head on reload).
  TRL 1.9.2 has no PPOTrainer; RLOO would be the algorithm. DPO is not a dependency.
- **Inference-limits agenda, native context**: 262144-token native context costs
  ~8 GiB of KV cache on this model (32 KiB/token in bf16: 8 full-attention layers
  x 4 KV heads x head_dim 256 x K+V x 2 bytes).
- **Inference-limits agenda, YaRN extension**: a 1.01M-token batch-1 sequence needs
  ~30.87 GiB of paged hybrid cache and is predicted to fit this card at
  `gpu_memory_utilization` 0.89-0.90 (measured smoke at 0.85: weights 8.61 GiB,
  CUDA graphs 0.56 GiB, 29.34 GiB cache available).

## Nested-commitment extraction defect — NOT an outcome-blind repair

This entry exists because the repair **changed observed dev scoring**. It must not
be described as outcome-blind, even though what it fixes is a self-contradictory
answer grammar rather than a threshold, and even though no held-out result exists
yet for it to have been tuned against. The honest description is: a scoring bug
was found by reading a dev trace, and correcting it raised the dev tally.

**The contradiction.** The neutral system prompt (`prompts/agentic/p1_minimal.txt`)
requires a final `ANSWER: <value>` line. The suite's generated task prompts end
with "Reply with `\boxed{code}`". Both instructions are in the same context, so a
model that obeys both writes `ANSWER: \boxed{code}` — reasonable compliance, not a
format failure. The old extractor (`suite/schema.py`) took the last `ANSWER:` token
and stopped, committing the literal string `\boxed{code}`; the `\boxed{}` fallback
was unreachable whenever an `ANSWER:` line was present. That string matches no
ground truth, and it also matches no value in any tool observation, so the same
episodes were additionally labelled hallucinations by `provenance.py`.

**The repair.** Unwrap a full `\boxed{...}` anchored at the first character of the
selected `ANSWER:` value. It scrapes nothing out of prose, and the four frozen
precedence rules are unchanged and now individually tested: the last `ANSWER:`
bearing a value wins; `ANSWER: \boxed{x}` commits `x`; with no `ANSWER:` at all the
last `\boxed{x}` is accepted; a later wrong `ANSWER:` is never rescued by an
earlier correct box.

**Offline rescore, CPU only.** The 12-episode `verify-a5000` dev run
(`out/verify-a5000/traces/B0.clean.none.jsonl`, sha256 `cdeff6bf…`, run secret
sha256 `11744e05…`, git `0a1cda8`) was replayed through the real
`provenance.certify_episode`, which is a pure function of trace plus run secret.
Before the fix the replay reproduced the recorded tally exactly — 4/12 raw and
4/12 certified — which is what makes the after-tally credible rather than a
different code path.

| Episode | Final commitment as written | Before | After |
|---|---|---|---|
| dev-lookup_chain-h2-0000 | prose only, wrong value, 1 of 2 nodes | wrong | wrong |
| dev-lookup_chain-h2-0001 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0002 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0003 | `ANSWER: <code>` | correct | correct |
| dev-lookup_chain-h2-0004 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0005 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0006 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0007 | `ANSWER: <code>` | correct | correct |
| dev-lookup_chain-h2-0008 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0009 | `ANSWER: \boxed{<code>}` | wrong + hallucinated | correct |
| dev-lookup_chain-h2-0010 | `ANSWER: <code>` | correct | correct |
| dev-lookup_chain-h2-0011 | `ANSWER: <code>` | correct | correct |

Raw success 4/12 → 11/12; certified success 4/12 → 11/12; false hallucination
labels 7 → 0. The single remaining failure, `dev-lookup_chain-h2-0000`, is genuine:
it made one of two required calls, invented "12345", and committed nothing in
either accepted form, so the repair cannot and does not rescue it.

**What this measurement is and is not.** It is a fixed-trace rescore: identical
episodes, identical receipts, identical decoding (temperature 0, top-p 1, seed
2786983945, thinking disabled), only the reader of the final line changed. It is
*not* evidence about capability, and 11/12 on twelve clean H2 lookup episodes is
not a capability claim — it says the harness can now see a correct answer. Two
consequences carried forward: the earlier reasoning that arm B0 "mostly measures
format guessing" was an artifact of this defect and is withdrawn, and any dev
number recorded before this fix is void rather than merely stale.

The twelve final texts are inlined verbatim in
`tests/suite/test_answer_extraction.py` (`out/` is not committed) so the
discriminating rescore stays runnable without a GPU or the original run; the
full-trace certification replay runs additionally whenever those artifacts exist.

## 2026-08-05 — the verify-a5000 dev run is invalidated by the unified fault contract

The twelve-episode `verify-a5000` dev run above is now **excluded, clean-only
apparatus evidence** and is refused by the current certifier rather than
re-scored. It was produced under the retired environment: it carries no
`environment_contract_sha256`, its events use the retired field names
(`exposed_digest`, `decision`, `fault_emitted`), and it records no canonical
verdict at all, so oracle-node completion, the fulfillment final state,
capability-token provenance and the call budget were never checked for it.

Those twelve traces are also, usefully, the **direct evidence for the transcript
drift** the reconciliation closed. Every assistant turn that made a tool call is
recorded as `{"role": "assistant", "content": ""}` — the structured tool-call
object the offline rollout carried is simply absent — and every tool result is
nameless. Observation digests were blind to both; the chat template is not. That
is why the parity test compares rendered token ids per decision and not digests.

The answer-grammar rescore this run produced remains valid and remains runnable:
it is a property of the reader, not of the environment, and the twelve final texts
are inlined in `tests/suite/test_answer_extraction.py`. What changed is the second
half of that test — it no longer certifies the on-disk traces through the current
certification path. It asserts that they are **invalidated**, that their event
format is not silently accepted, and that both drifts are present in their real
bytes.

No number from that run enters any claim. It contained no fault-recovery outcomes
and never entered the study trace set.

## 2026-08-05 — size census under the unified contract

`scenario.tool_output_max_tokens` was 208, measured against the tokenless
payloads. Re-measured exhaustively (78,100 episodes, 742,500 model-visible
observations, 624,800 rendered views, all four committed train/dev splits × twelve
cells × seven fault variants × eight prompts) with the Qwen3.5-4B tokenizer:
the model-visible tool result peaks at **231 tokens / 474 chars** and the rendered
terminal view at **4,960 tokens**. Caps moved to 256 and 5,120 respectively;
`tool_output_max_chars` stayed at 512. Artifact and hash:
`results/agentic/token_census.json`,
`98379c42540a30d7c6c29fea53193a169714351f184f9c58b516484dc5896fa8`.

This is a measurement, not a result: it says how large the registered wire format
is, and the caps follow it. `tests/test_size_ceilings.py` fails if a cap ever
drops below what was measured, or if the census was taken under a different
environment contract.

## 2026-08-05 — dev-only preflight: probes 1-2 run, probe 2 STOPS the chain

The council's smallest credible pre-production check (round 5) was built as
`scripts/preflight_dev.py` over the six committed dev tasks it names — three
families × clean/faulted × low/high family horizon, each once clean and once
faulted, 12 episodes — collected into a derived manifest whose rows are byte
copies of `data/suite/v1/certspecs/dev.jsonl` (verified against the committed
`SHA256SUMS` first; the committed dev data is not edited).

**Probe 1, exact 12-row extractor replay, CPU only: PASS.** The recorded
`verify-a5000` trace rescores from **4/12 to 11/12** raw and certified answers;
exactly the seven named episodes change and no others; the one genuine failure
(`dev-lookup_chain-h2-0000`, which committed `12345`) stays a failure. The
before-tally is *measured*, not recalled: the defective grammar's reading equals
each row's recorded `raw_success`. The **seven false hallucination labels clear**,
with their mechanism shown — the literal `\boxed{x}` appears in no validated
observation, the inner token does. Three readers (`schema`, `provenance`,
`verify`) agree on the whole battery, and 12 live oracle episodes over the six
tasks certify identically under four correct commitment grammars (72 grammar
replays), while a later wrong commitment is never rescued.

**Probe 2, oracle-driven fault parity matrix, CPU only: FAIL.** 18 episode pairs
(six tasks × clean / faulted / faulted-with-a-bare-retry) were run through the
claim-bearing evaluation path and the canonical training path. The environment
itself reconciles: all 8 surfaces are equal on all 18 pairs — envelope bytes, the
whole hidden event ledger, token arguments, budgets, credited progress, episode
digest, verdict row, and the **rendered prefix token ids per decision**. Every
scheduled fault fires **exactly once**; the registered remediation is certified in
all four fault classes; and **bare retries are never certified** (`blind_retry`
for transient / rate_limit / malformed, and for the wrong-unit trap the frozen
precedence reports `hallucinated`, because accepting the trapped value leaves the
committed answer with no validated source).

What failed is the seam between that verdict and the analyzer.
`provenance.certify_episode` sets

    verdict_agrees = (verdict.certified_success == ledger_ok)

an EQUALITY between the canonical verdict and a strictly weaker transcript-only
predicate, and `analyze.veto_s17_trace_summary` turns any inequality into a
harness **BUG**, which vetoes every gate, every claim and the winner. So an
episode the strict verifier is *supposed* to refuse, whose transcript has nothing
to object to, reads as a broken harness. Two mechanisms were reproduced:

* **fault arm** — 5 of 6 bare-retry episodes: right answer, valid receipt chain,
  validated source, no runaway, no fabrication (`ledger_ok` true) while the
  verdict correctly refuses certification because recovery was a blind retry.
  S17 → BUG.
* **clean arm** — one episode that batches both lookup hops into a single
  decision: the registered "dependency edges need a LATER decision" rule refuses
  the second node, the answer is still correct and sourced. S17 → BUG. The defect
  is therefore not fault-specific.

The positive control passes in the same run: over the 12 clean and remediated
episodes S17 is **OK** with all 12 verdicts reproduced field-for-field by
canonical replay, so the BUG is not an artifact of how the probe builds the
analyzer's input.

Both behaviours are legitimate and near-certain in a production run, so the
preflight stopped there: **probes 3, 4 and 5 (the live 12-episode HTTP matrix,
the tiny offline-RS batch and the one-step SFT canary) were not run**, no GPU
process was started, and no GPU minutes were charged to the ledger. The council's
required ordering puts "pass the oracle parity matrix" before those three, and
spending calibration hours on an apparatus that cannot issue a verdict is exactly
what a preflight exists to prevent.

Results: `results/agentic/preflight/{manifest,probe1,probe2}.json` (every check
with its numbers). `tests/test_dev_preflight.py` keeps both CPU probes runnable
and pins the open defect as a `strict` xfail, so repairing the seam turns that
test into an unexpected pass and forces it to be flipped into a positive
assertion. The repair is a design decision for the certification/analysis owner:
the cross-check wants to be one-directional (a verdict claiming success the
ledger side refuses is tampering; a verdict refusing what the ledger side cannot
see is normal), and `certified_success` already ANDs the two, so no gate needs to
move.

### Forward correction, 2026-08-06 — the preflight is finished and green

The paragraph above stopped at probes 1–2 and said probes 3, 4 and 5 "were not
run". That is no longer the state, and the paragraph is left standing rather than
rewritten because it was true when written.

The S17 cross-predicate seam was closed one-directionally as described
(`bfc8543`, "a strict refusal is not a harness bug"), the four training-corpus
seams closed with it (`bbcdb00`), and the preflight then ran to completion:
**all five probes pass, 87 checks total** (`2ba52da`) —
probe1 16, probe2 12, probe3 24, probe4 13, probe5 22 — over
`results/agentic/preflight/probe{1..5}.json`. Probe 3 is the live 12-episode HTTP
matrix on one server startup, probe 4 the tiny offline-RS batch across both
prompt variants on one engine, probe 5 the view builder plus a one-optimizer-step
SFT canary. Those three did charge GPU minutes, and they are on the ledger.

## Round 2 write-up rules: the saturated secondary endpoints

The prompt tournament finished, `p2_plan_state_act` won, and the two secondary
endpoints turned out to be at ceiling on the development split: 295/300 on the H4
all-tools orchestration axis and 296/300 on the H8 execution axis, so even a
perfect trained arm could gain only 0.0167 and 0.0133 on those realized samples —
both under the registered `+0.05` margin.

I initially wrote that this meant "the preregistered consequence is INCONCLUSIVE
by saturation" and that "training had nothing to add" (commit message of
`4c840e3`, which stays as pushed). Both readings are wrong, and the referee's
corrections — the machine statuses the frozen analyzer actually produces, what a
saturated *development* endpoint does and does not establish, the required
pre-results wording, the per-gate reporting template, and why swapping the
ceiling for the deep-horizon floor (1/26 at fulfillment H14, 7/26 at H20, against
244/248 across the other ten cells) would be a bad trade — are recorded in
**[INTERPRETATION.md](INTERPRETATION.md)**. That file governs the write-up of the
held-out secondary results.
