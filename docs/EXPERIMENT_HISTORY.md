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
