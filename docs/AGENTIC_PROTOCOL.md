# Agentic evaluation protocol (v1, preregistered)

Committed **before** any held-out GPU result exists. After this commit, no
threshold, margin, sample size, seed, estimand, or interpretation rule below
may be edited in place. A genuinely broken gate gets a dated **AMENDMENT**
section appended here plus entirely fresh held-out seeds; the original gate is
still reported. The machine-readable mirror of this protocol is
[`configs/agentic_preregister.json`](../configs/agentic_preregister.json);
where prose and JSON could ever disagree, the JSON governs.

The verdict is computed, not narrated:

```bash
.venv/bin/python -m agentlab.analyze --agentic \
  --traces results/agentic/traces --secret out/agentic/run_secret.hex \
  --specs data/suite/v1/specs/eval.jsonl \
  --split-manifest train=data/suite/v1/specs/oracle_sft.jsonl \
  --split-manifest dev=data/suite/v1/specs/dev.jsonl \
  --split-manifest eval=data/suite/v1/specs/eval.jsonl \
  --save results/agentic/verdict.md --save-json results/agentic/verdict.json
```

## 1. Claim hierarchy (fixed; nothing may be promoted after results)

**PRIMARY — certified error recovery on the common-clean subset.**
On the subset `C` of held-out fault-assigned tasks where **both** BP and TP
achieve certified success in the clean replay, the paired difference (TP − BP)
in *certified recovery* on the injected replay must have a **one-sided 97.5%
template-clustered bootstrap lower bound above the +0.05 margin** (100,000
deterministic replicates, committed seed). The exact McNemar test is reported
alongside. `|C| ≥ 500` is required — a smaller `C` makes the primary claim
**INCONCLUSIVE**, no matter how favourable the point estimate. The
intention-to-treat difference over **all ≥ 900 assigned fault pairs** is also
reported and must be nonnegative.

**SECONDARY (each: same +0.05 margin; PASS only on a positive clustered lower
bound exceeding the margin):**

- (a) *Certified all-three-tools orchestration at fixed H4* on ≥ 600 pairs
  whose answer causally requires `kb_lookup`, `unit_convert`, and
  `calculator` (gates MT1–MT6).
- (b) *H8 execution reliability*: certified strict success on ≥ 400 paired
  clean H8 instances, families `lookup_chain` + `typed_relay` (gates HR1–HR3).

**DESCRIPTIVE (never gated, never claimed as capability):** full
success-vs-horizon curves with pointwise 95% Wilson bands — **no logistic or
any other extrapolation; H50 reported only if the observed curve crosses
50%**, otherwise left-/right-censored; the 280-task two-fault stress set as a
measured-only extrapolation probe; B0/T0 arms quantifying elicitation.

## 2. Arms and pairing

| Arm | Weights | System prompt |
|-----|---------|----------------|
| B0 | base | neutral (`p1_minimal.txt`) |
| BP | base | frozen tournament winner |
| T0 | trained (locked) | neutral |
| TP | trained (locked) | frozen tournament winner (identical bytes to BP's) |
| R0/RP | GRPO checkpoint | only if GRPO ran and was the locked selection |

The primary comparison is always **TP vs BP**: identical prompt, identical
task IDs, values, budgets, schemas, parser, server, and seeds (veto S8), so
the weight change is the only difference. Deterministic decoding: temperature
0.0, top_p 1.0, committed seed, 1024 tokens per decision.

The elicitation control is mandatory because round 1 of this lab measured a
one-sentence prompt recovering 81.8% of an SFT gain. Eight prompt candidates
are committed by SHA-256 in the preregistration; the winner is selected on
disjoint dev data (100 instances/axis for all eight, then 200/axis for the top
two; highest mean certified strict success on the combined 300, ties to the
shorter file then the lower index). The honest description is **"best of
eight preregistered system prompts under a fixed search budget"** — never
"best possible prompt".

## 3. Episode contract

Budgets: `H+3` assistant decisions clean, `H+5` single-fault, `H+8` stress;
hard tool-call cap `2H+4`; 240 s wall clock per episode. Final answers commit
via a last line `ANSWER: <value>` (`\boxed{}` tolerated as fallback).

Every tool observation carries an opaque receipt
`r-<hmac-sha256(run_secret, task|call|obs)[:32]>`. The model never sees the
secret, so receipts are unforgeable; the analyzer revalidates every chain
(S13). KB misses return only `no_entry` — never a key list.

**Runaway** (always counted as failure, never dropped): call cap reached;
three identical normalized calls each returning the identical error; four
consecutive calls that do not advance the ledger; runner termination for
token/wall-clock/parser budget. **Hallucinated result**: citing an unminted
receipt, tool-role content without an environment event, or a committed
answer value absent from every validated observation.

## 4. Fault protocol

Assigned deterministically per `(task_id, fault_seed)`, emitted **exactly
once** at the registered critical oracle node on the first semantically valid
call, always recoverable within the remaining budget. Groups balanced ≥ 300
assigned episodes each: transient/rate-limit, malformed, wrong-unit
(wrong-unit only on `unit_convert` nodes).

Error envelopes (transient, rate-limit, malformed) carry an unpredictable
128-bit `recovery_token` plus machine-readable remediation. **Certified
recovery** requires all of: fault actually emitted; fault-appropriate remedial
action incorporating the token (rate-limit: on a later decision; wrong-unit:
a corrected-target conversion — no token exists for a trap that is not an
error envelope); a new validated post-failure result for the faulted node; the
exact final answer derived from post-failure validated information; no
hallucinated result; no runaway.

The six preregistered **non-recovery** cases: guessing after an error
(`unvalidated_answer`), reusing a clean-run value (`unvalidated_answer` — the
clean and injected replays run in isolated processes), blind retries without
the remediation contract (`blind_retry`), answering from pre-fault information
(`pre_fault_answer`), inventing a tool response or receipt (`hallucinated`),
and answering after a runaway criterion (`runaway`). A correct answer without
the certificate is `lucky_success`, reported diagnostically, never recovery.

Both denominators are always reported: all assigned fault episodes
(intention-to-treat, headline) and actually exposed episodes (diagnostic).
Reporting only exposed episodes would reward policies that avoid tools.

## 5. Gates

**ER (primary)** — ER1 `|C| ≥ 500` (else INCONCLUSIVE); ER2 recovery diff on
C, one-sided 97.5% clustered LB > +0.05; ER3 TP recovery on C Wilson LB
≥ 0.60; ER4 clean non-inferiority, clustered LB > −0.03; ER5 ITT diff ≥ 0
over ≥ 900 assigned pairs; ER6 TP runaway Wilson UB ≤ 3%; ER7 TP
hallucinated-result Wilson UB ≤ 1% (both over all core TP episodes); ER8 no
fault group below −0.05 point diff.

**MT (secondary a)** — MT1 certified all-tools diff LB > +0.05 on ≥ 600 H4
pairs (else INCONCLUSIVE); MT2 TP Wilson LB ≥ 0.60; MT3 TP median calls
≤ oracle+2; MT4 runaway UB ≤ 3% and hallucination UB ≤ 1% on MT tasks; MT5
none of the six registered order patterns (`pattern_id` 0–5, ≥ 80 pairs each,
concrete sequences committed in the suite manifest before held-out results)
below −0.05; MT6 the absent-information control is clean.

**HR (secondary b)** — HR1 H8 clean certified diff LB > +0.05 on ≥ 400 pairs
with ≥ 20 discordant (else INCONCLUSIVE); HR2 TP H8 runaway UB ≤ 3%; HR3 TP
H8 hallucination UB ≤ 1%.

**Statistics.** Cluster = `template_id`; statistic = ratio of sums over
resampled clusters; 100,000 replicates from the committed SHAKE-256 stream
(seed 2786983947, label = gate name); lower bound = the floor(0.025·R)-th
order statistic. Exact binomial McNemar reported at any n; Holm-adjusted
across the two secondaries. Sample sizes must never be enlarged after looking
at results — an underpowered design is INCONCLUSIVE, not extendable.

## 6. Controls

**Absent information (S11).** ≥ 200 redacted instances per family on BP and
TP: the required lookup never returns the hidden value. Certified success is
zero *by construction*; **any raw exact success is a harness-leakage BUG**,
never model ability. Every scored task depends on an episode-specific hidden
value (≥ 48 bits entropy) generated outside the prompt; absence of a receipt
is never the only control — the answer itself must be unknowable from priors.

**Counterfactual permutation (S14).** ≥ 100 permuted replays (hidden terminal
values swapped between task IDs with the committed permutation seed) on BP and
TP: correct outputs must track the returned value, not the prompt identity.
All other scored tasks carry a generation-time `counterfactual_sensitive`
verification (mutating the hidden value changes the oracle answer and the
scorer decision).

## 7. Harness vetoes (checked before any gate)

S8 PAIRING · S9 ORACLE · S10 SPLITS · S11 ABSENT-INFO · S12 INJECTION ·
S13 RECEIPTS · S14 COUNTERFACTUAL · S15 ATTRITION · S16 CONTROL-INTEGRITY ·
S17 TRACE-SUMMARY · S18 TEST-BLINDNESS — semantics exactly as in the
preregistration JSON. **Any BUG vetoes every capability gate and the winner
rule.** Missing traces or an underpowered common-clean subset yield
INCONCLUSIVE — never a favourable interpretation. Outcome states everywhere:
PASS / FAIL / INCONCLUSIVE / BUG.

S18 uses a self-referential seed commitment: `heldout_seed =
SHA256(<preregistration commit sha> + ":agentic-heldout-v1")[:8]` as a big-
endian integer. The commit that introduces this protocol *is* the commitment;
`results/agentic/locks.json` (checkpoint + prompt winner) must exist before
`results/agentic/seed_reveal.json` is written.

## 8. Launch floors and winner rule

Whichever arm ships must clear **all** floors: overall clean certified
success ≥ 0.65; no family < 0.50 clean; overall faulted strict success
(intention-to-treat) ≥ 0.40; no family < 0.25 faulted; loop/crash rate
< 0.02.

Winner rule: the **trained arm ships only if** it clears every floor AND is
clean-non-inferior at −0.03 (ER4) AND has certified-recovery clustered LB
> +0.05 (ER2). Otherwise the **frozen prompted base** ships if it clears the
floors. Otherwise the honest verdict is **"no successful multifaceted
pipeline yet"** — that is a reportable scientific outcome, not a failure of
the harness.

## 9. Claims this workflow can never support

Even if every gate passes, the following are preregistered rejections (the
full list is in the JSON and is echoed verbatim into the machine verdict):
no general agentic competence; no general long-horizon planning; no arbitrary
eight-step planning; no robustness to arbitrary tool/API failures; no
real-world tool orchestration; no "training beats prompt engineering" without
the fixed eight-prompt qualification; no claim that the policy *understands*
recovery; no GRPO-improves-RS-SFT claim without a separately preregistered
stage-attribution experiment; no extrapolation beyond horizon eight; no
judge-model/user-simulator benchmark claims; no multimodal/long-context/
speculative-decoding claims. Successful dependency-chain execution does not
reveal an internal planning mechanism: the protocol supports narrow
*behavioural* claims inside one procedural generator distribution, nothing
more.
