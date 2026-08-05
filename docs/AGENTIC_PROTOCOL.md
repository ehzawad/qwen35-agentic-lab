# Agentic evaluation protocol (v1, preregistered)

Committed **before** any held-out GPU result exists. After this commit, no
threshold, margin, sample size, seed, estimand, or interpretation rule below
may be edited in place. A genuinely broken gate gets a dated **AMENDMENT**
section appended here plus entirely fresh held-out seeds; the original gate is
still reported. The machine-readable mirror of this protocol is
[`configs/agentic_preregister.json`](../configs/agentic_preregister.json);
where prose and JSON could ever disagree, the JSON governs.

**"The JSON governs" is now literally true.** Every threshold, margin, floor,
sample-size minimum and rate ceiling the analyzer applies lives in the
`machine` block of that file and is read from it at decision time
(`analyze.registered()`), with **no default and no fallback**: a missing field
raises rather than reverting to a literal. Until this commit the claim was
decorative — `analyze.py` carried its own `min_c = 500`, `min_assigned = 900`
and the literal margins `+0.05` / `-0.03` / `-0.05`, so editing the governing
file changed **not one verdict**. Three tests hold the line:
`test_no_gate_threshold_is_expressed_in_the_analyzer_source` (a JSON/code
parity scan that fails if any gate number reappears in code),
`test_every_registered_threshold_holds_its_unchanged_value` (every value is
byte-identical to the one the code used to carry — this was a governance fix,
not a threshold change), and
`test_editing_a_registered_threshold_changes_the_verdict`.

No held-out GPU result exists at the time of this commit, so every
specification repair recorded here is **outcome-blind by construction**: there
is no result it could have been fitted to.

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
  `calculator` (gates MT1–MT6). Registered generation: exactly 600 = six
  `pattern_id` values × 100, over ≥ 240 distinct `template_cluster_id` values
  with ≤ 5 value instantiations each.
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
call, always recoverable within the remaining budget. Groups balanced at the
registered **400 assigned episodes each** (1,200 total): transient/rate-limit,
malformed, wrong-unit (wrong-unit only on `unit_convert` nodes). ER8 is a gate
only when all three groups are present at that size (§5).

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
hallucinated-result Wilson UB ≤ 1%; ER8 no fault group below −0.05 point diff,
with **all three registered groups present at their registered 400 assigned
episodes each** — a missing or short group makes ER8 INCONCLUSIVE, because "no
group fell below the floor" computed over two of three groups is not the
registered gate. (Total assigned then stands at 1,200; the ≥ 900 floor is
unchanged. Raising the per-group cardinality from 300 to the registered 400 is
a tightening.)

**Stratum membership is frozen** (`machine.strata`, keyed on the spec/trace
`split` field). ER6/ER7 read "all core TP episodes" literally: the **core
stratum only**, control `none`, conditions clean and faulted. MT and H8
augmentation traces also run with `condition="clean"`, so an unrestricted
denominator let the *size* of a separately-sized secondary sample move the
primary claim's runaway and hallucination bounds. A trace whose `split` belongs
to no declared stratum is excluded from every gated sample and reported in the
emitted stratum census; exclusion can only widen an interval or lower a count,
never favour an arm.

**MT (secondary a)** — MT1 certified all-tools diff LB > +0.05 on ≥ 600 H4
pairs (else INCONCLUSIVE); MT2 TP Wilson LB ≥ 0.60; MT3 TP median calls
≤ oracle+2; MT4 runaway UB ≤ 3% and hallucination UB ≤ 1% on MT tasks; MT5
none of the six registered order patterns (`pattern_id` 0–5, ≥ 80 pairs each,
concrete sequences committed in the suite manifest before held-out results)
below −0.05; MT6 the absent-information control is clean.

**HR (secondary b)** — HR1 H8 clean certified diff LB > +0.05 on ≥ 400 pairs
with ≥ 20 discordant (else INCONCLUSIVE); HR2 TP H8 runaway UB ≤ 3%; HR3 TP
H8 hallucination UB ≤ 1%.

**Statistics.** Cluster = **`template_cluster_id`**, the *structural* template
identity (family + horizon + oracle-DAG shape + tool-order pattern + operand
roles) — **distinct from the paraphrase/wording `template_id`, and the sole
clustering field.** The old key (`template_id or task_id`) silently swapped
between two incomparable resampling units: with a two-wording paraphrase pool
it collapsed all 1,200 core tasks into **two** clusters, and where the field was
absent it fell back to `task_id`, i.e. one cluster per observation, which is not
a clustered bootstrap at all. Both mistakes make the interval *narrower* than
the design supports, so neither is reachable by a fallback.

Statistic = ratio of sums over resampled clusters; 100,000 replicates from the
committed SHAKE-256 stream (seed 2786983947, label = gate name); replicate
**chunk block frozen at 2,000** — each chunk draws from its own SHAKE label, so
the realised stream and therefore the bound depend on it, and it may not remain
an unregistered internal constant; lower bound = the floor(0.025·R)-th order
statistic.

Every interval report emits `n_clusters`. A gated sample needs at least
**30 structural clusters** (MT1: **240 clusters with ≤ 5 value instantiations
each**, per the registered MT contract), and any missing `template_cluster_id`
in the sample makes the gate **INCONCLUSIVE** — never a silently narrow
interval, and never a silent reclustering onto `task_id`. Thirty is the floor
because cluster-bootstrap coverage degrades sharply below roughly 20–30
independent clusters; below it the resampling distribution describes a handful
of templates rather than the generator distribution.

Exact binomial McNemar is reported at any n. **Holm-adjusted** exact McNemar
p-values are now actually computed and emitted across the two secondary claims
(family `{MT1, HR1}`, one-sided p, α 0.05) — specified since v1 and never
produced before. They are *reported, never gated*: the gate is always the
clustered lower bound. An absent or underpowered secondary enters the family
with p = 1.0 and is labelled INCONCLUSIVE rather than dropped, because dropping
a member shrinks the family and weakens the adjustment for the survivor.

**Median convention (MT3).** For an even sample the median is the **upper
middle** order statistic (`sorted[n//2]`, 0-based). It was unstated; it is now
registered, is the convention the analyzer already implemented, and is the more
conservative of the two choices for an at-most gate.

Sample sizes must never be enlarged after looking at results — an underpowered
design is INCONCLUSIVE, not extendable — and mandatory sample sizes may never be
*shrunk* after any result is visible (§9).

## 6. Controls

**Absent information (S11).** ≥ 200 redacted instances **per family per arm**
on BP and TP — counted per `(family, arm)` cell, never pooled across arms, and
every family appearing in a gated sample must have redacted coverage in every
arm. The required lookup never returns the hidden value. Certified success is
zero *by construction*; **any raw exact success is a harness-leakage BUG**,
never model ability. A control episode that never ran a single decision is a
second BUG: zero success over unattempted episodes is vacuous, not evidence that
the hidden value is required, and such traces contribute no coverage. Every
scored task depends on an episode-specific hidden value (≥ 48 bits entropy)
generated outside the prompt; absence of a receipt is never the only control —
the answer itself must be unknowable from priors.

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
preregistration JSON. **All eleven are MANDATORY**
(`machine.mandatory_harness_checks`).

**Official status precedence is `BUG > FAIL > INCONCLUSIVE > PASS`.** A real
FAIL outranks missing evidence; INCONCLUSIVE never reads as support; **`SKIP` is
forbidden** in the agentic path and any unknown state is treated as a BUG rather
than translated.

**A harness BUG overlays *every* official status** — each gate, each launch
floor, each claim, and the winner. The raw arithmetic survives only under the
explicitly non-verdict field **`observed_status`** (plus `measured_status` where
a gate had already been downgraded for underpower), so nothing is deleted, but
**no official field may read PASS anywhere in a verdict containing a BUG**. This
closes a real defect: previously only the aggregate claims and the winner were
overlaid, so in an S8-BUG run individual gates still printed PASS/FAIL and every
launch floor still printed PASS.

**A mandatory check left INCONCLUSIVE means NO VERDICT.** Previously only BUG
blocked the winner, so a winner could ship while S9 (oracle reachability), S10
(split isolation), S14 (counterfactual control) or S18 (test-blindness) was
simply unverified. Missing traces or an underpowered common-clean subset yield
INCONCLUSIVE — never a favourable interpretation. Outcome states everywhere:
PASS / FAIL / INCONCLUSIVE / BUG.

S16 additionally enforces the **exactly-one-locked-checkpoint** rule:
`locks.json` must name exactly one checkpoint path with exactly one stage from
`{rs_sft, grpo}`, and every TP/T0 trace's adapter must equal that path. An arm
labelled TP that is not demonstrably the locked checkpoint makes every trained
number unattributable, so a missing, ambiguous or disagreeing lock is a BUG.

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

**Floor denominators are frozen** (`machine.floors`): F1/F2 over the arm's
**core** clean episodes with control `none`; F3/F4 over its **core** faulted
episodes, intention-to-treat (every assigned fault episode stays in, including
timeouts, parser failures, crashes and runaways); F5 over the union of those two
core sets. MT and H8 augmentation, stress, redacted and permuted traces are
**outside every floor denominator** — they were not, and because an oversampled
secondary stratum could move a floor, and the floors decide the winner, the
size of a secondary sample could decide the winner. Floors are **point
estimates** against the threshold; Wilson intervals are reported beside them for
the reader but do not decide.

### The winner rule, as a frozen truth table

Ranks are evaluated in order and the first match decides. No discretion remains
anywhere in the table.

| Rank | Condition | Verdict |
|---|---|---|
| 1 | any mandatory harness check is **BUG** | **BUG**, no winner |
| 2 | any mandatory harness check is **INCONCLUSIVE** | **NO VERDICT** |
| 3 | exactly one locked trained checkpoint (`rs_sft` or `grpo`) maps to TP | precondition; a missing or ambiguous lock is a BUG (S16) |
| 4 | TP clears **every** launch floor **and** ER4 PASS **and** ER2 PASS | **TP ships** |
| 5 | otherwise BP clears every launch floor | **BP ships** — including when the trained comparison **FAILS** and when it is **underpowered** |
| 6 | floor evidence incomplete (any floor INCONCLUSIVE) | **NO VERDICT** |
| 7 | otherwise | **"no successful multifaceted pipeline yet"** |

Rank 5 is deliberate and load-bearing: the prompted-base-wins branch is a real
preregistered outcome and may not quietly degrade into "no successful pipeline"
just because the trained leg disappointed. Rank 7 is a reportable scientific
outcome, not a failure of the harness.

## 9. Budget: the 36-hour ceiling, and what is conditional on it

The ceiling is **36.0 measured GPU-hours** summed over every GPU stage —
measured from the ledger, never a nominal per-stage range and never a
caller-supplied projection.

**The final paired evaluation, not GRPO, is the stage most likely to consume
it.** Registered episode counts: **≈ 8,360** episodes for BP/TP alone (4,800
core = 1,200 tasks × clean/faulted × 2 arms; 1,200 MT; 400 H8 augmentation;
1,200 redacted; 200 permuted; 560 stress), **≈ 15,320** including B0/T0, and
**≈ 22,280** including R0/RP. Even at the historically favourable fully batched
rate of ~7.85 decision-generations/s with nine decisions per episode, the
all-arm evaluation is around seven GPU-hours — and the production evaluator runs
concurrency 8, longer contexts and up to 1,024 tokens per decision instead of
the 384-token rejection-sampling setting, so that rate is optimistic.
Production rejection sampling is the likely second-largest stage.

**MANDATORY (never budget-conditional):** BP and TP across core (clean and
faulted), MT, H8, and **all** controls.

**Cut order if calibration projects an overrun** — in this order and only this
order, and only on a projection made *before* the relevant stage runs, never
after seeing a held-out result from any arm:

1. the entire **GRPO branch**;
2. its **variance probe** and the **R0/RP** arms;
3. the descriptive **B0/T0** arms;
4. the two-fault **stress** set.

Everything in that list is declared budget-conditional **here, before the
push** — which is the only thing that makes dropping it honest. Anything not in
the list is mandatory.

**Mandatory sample sizes may NEVER shrink after any result is visible** — not by
one episode, and not for a budget reason discovered after a number was seen. **If
the mandatory work cannot fit the ceiling, the run STOPS and reports
INCOMPLETE / INCONCLUSIVE** rather than trimming, substituting a smaller sample,
or presenting a partial mandatory sample as the registered one. An unfinished
run is a reportable outcome; a silently shrunk one is not.

**Calibration.** Before the final paired evaluation, measure completed
episodes/hour *and* decision-generations/hour separately for base and adapter
arms, stratified by horizon (H2/H4/H8/H12/H14/H20) and clean/faulted, and project
the remaining registered work from those measurements including server restarts.

**Ledger accounting.** `results/agentic/gpu_ledger.jsonl` is append-only; every
GPU stage records stage name, start/end UTC, **measured** elapsed GPU seconds,
git sha, GPU identity, and the episode or step count actually completed. The
ceiling is enforced against the **sum of measured seconds**. A stage may not
start when its calibrated projection plus the ledger total would cross 36.0
hours. A shard that overruns its projection is still charged its measured time
and the overrun is reported, not absorbed. Every GPU consumer — **including the
final evaluator** — must read and append to the ledger; a stage absent from the
ledger is unaccounted time and makes the budget claim INCONCLUSIVE.

### GRPO (optional, cut rank 1)

Nothing in the primary or secondary claims depends on GRPO running.

**`scenario_seed` rule.** GRPO training data is a **map-style** dataset keyed by
`scenario_seed`. All G generations in a group must reconstruct byte-identically
the same prompt, KB, environment, fault target, fault class, node index and
budgets from that one seed; **only the sampling seed differs**. A group whose
members cannot be shown to share one reconstructed scenario is discarded, not
repaired.

**Variance probe** (`src/agentlab/variance.py`, which must exist and be tested
before GRPO is implementable): run on the locked RS-SFT checkpoint **before any
GRPO update**, on a disjoint committed probe split, with the exact GRPO decoding
settings and the **exact production reward function** — one shared reward
implementation imported by both probe and GRPO. **48 groups per family × 3
families = 144 groups × 8 generations = 1,152 rollouts**, eight distinct
committed sampling seeds per group with every non-sampling input identical.

Binding gates, evaluated **POOLED** over all 144 groups:

- fraction of groups with **nonzero total-reward SD ≥ 0.60** (sample SD of the
  total reward after summing every component and applying the final clamp;
  nonzero means SD > 1e-12);
- fraction of groups with **terminal-success disagreement ≥ 0.40** (both at
  least one strict success and at least one strict failure among the eight — not
  raw-answer disagreement; for faulted groups terminal success additionally
  requires *certified* recovery, token use and any later-decision requirement).

Per-family figures are reported descriptively and **may not** become an
eligibility route: "the gate passes for family X" would turn per-family
reporting into post-hoc family selection. Failed, crashed, truncated and
no-commit rollouts stay in every denominator. Numerator, denominator, fraction
and Wilson interval are reported for both gates.

**Diagnostic / operational only, NOT scientific gates:** mean strict success in
[0.15, 0.85] and truncation-or-clip fraction ≤ 0.05. `clip_frac` here can only
mean length-truncated/clipped completions — before an optimizer step there is no
PPO/DAPO policy-ratio clipping, so a ratio reading is meaningless in the probe
and would belong to a short GRPO preflight. These two were unregistered values
in the retired `multifaceted.yaml`; recording them as readiness checks means
neither they nor their absence can be presented as evidence.

**Checkpoint selection (RS-SFT vs GRPO).** Exactly one trained checkpoint is
locked as TP. The choice is made on the **committed dev split only**, before the
held-out seed reveal, and recorded in `locks.json` with its stage. Metric: **mean
certified strict success on the dev split, pooling clean and faulted dev
episodes with equal weight per episode**, under the frozen decoding settings and
the frozen tournament-winner prompt. **Tie rule:** a difference below **0.005
absolute** goes to **RS-SFT** (the shorter, cheaper pipeline) — a tie never goes
to GRPO, so an unresolved comparison cannot manufacture a GRPO claim.

## 10. Claims this workflow can never support

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
