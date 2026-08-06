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

## 0. Pre-run hardware pivot receipt — 2026-08-05

**This study is preregistered as a SINGLE-CARD NVIDIA RTX A5000 study.** The
RTX A6000 is **released completely**: it is not waited for, not reserved, not
probed, and no opportunistic branch runs on it. Its co-tenant processes are
never touched. A future A6000 experiment would be a **separate registered run**
with its own locks, seeds, ledger and hardware declaration, and may never be
spliced into this run's trace set.

**The state of the study at the moment the card changed:**

| Quantity | Value |
|---|---|
| measured study GPU-hours | **0.0** |
| rows in `results/agentic/gpu_ledger.jsonl` | **0** |
| held-out results in existence | **0** |
| trained checkpoints | **0** |
| `results/agentic/locks.json` | absent |
| `results/agentic/seed_reveal.json` | absent |
| verdicts computed | **0** |

**That is what makes this outcome-blind rather than a moved goalpost.** ZERO
study GPU-hours and ZERO held-out results existed when the card changed, so
there is no number this pivot could have been fitted to. The test of a moved
goalpost is whether a result was visible when the rule moved; none was.

**The registered hardware is exactly one card:**

| Field | Registered value |
|---|---|
| GPU count | 1 |
| GPU name | `NVIDIA RTX A5000` |
| CUDA-visible bytes | **25,282,805,760** (23.546 GiB) |
| single physical GPU for every stage | yes, one UUID bound for the whole run |
| conditional second-card branch | **none** |

**Registered engine contract for every inference stage** (`machine.engine_contract`;
the single copy the analyzer reads): dtype **bfloat16**, `gpu_memory_utilization`
**0.80** — *not* the 0.85 the probe used, and deliberately **not** compensated
upward to 0.8725 — `max_model_len` **8192**, `max_num_seqs` **8**,
`max_num_batched_tokens` **8192**, `enforce_eager` **false**, thinking
**DISABLED**, multimodal inputs **explicitly REJECTED** rather than merely
unused, `tensor_parallel_size` 1.

The live A5000 probe that fixed those settings (measured operational facts, not
thresholds): 25,282,805,760 CUDA-visible bytes = 23.546 GiB; at
`gpu_memory_utilization` 0.85 / `max_model_len` 8192, vLLM 0.25.1 reported
checkpoint 8.68 GiB, CUDA-graph pool 0.54 GiB, available KV cache 9.08 GiB =
242,741 tokens, 19.857 GiB used, 3.69 GiB free; engine startup 289.7 s; one
48-token greedy generation 1.08 s. Measured non-KV footprint =
19.857 − 9.080 = **10.777 GiB = 0.4577 of the card**. **Thinking is ON by
default in this checkpoint**, which is why the contract disables it explicitly
instead of relying on a default.

**Nothing scientific changed with the card.** No threshold, margin, sample size,
seed, estimand, cluster rule, launch floor, claim definition or training recipe
was edited as a consequence of the pivot. The only registered numeric change in
the whole preregistration is the operational budget ceiling (§9: 36.0 → 120.0
measured A5000 GPU-hours), which decides no gate. Two semantics are
*strengthened*: the new **S19 HARDWARE-INTEGRITY** veto (§7) and the explicit
`enable_thinking = false` engine contract. Both can only veto a result or make
it INCONCLUSIVE; neither can make any result more favourable.

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
| R0/RP | GRPO checkpoint | **absent by design** in this run (§9 GRPO disposition) |

The primary comparison is always **TP vs BP**: identical prompt, identical
task IDs, values, budgets, schemas, parser, server, and seeds (veto S8), **and
an identical hardware-and-engine fingerprint — same physical GPU UUID, GPU
model, CUDA-visible bytes, driver version, engine fingerprint and effective
thinking mode (veto S19)** — so the weight change is the only difference. That
last clause is not decoration: the protocol *asserts* weights are the only
difference, and the assertion is unverifiable unless the card and the engine are
held fixed too, which is why S19 is a mandatory veto and not a convention.
Deterministic decoding: temperature 0.0, top_p 1.0, committed seed, 1024 tokens
per decision, **thinking disabled**.

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
S17 TRACE-SUMMARY · S18 TEST-BLINDNESS · **S19 HARDWARE-INTEGRITY** — semantics
exactly as in the preregistration JSON. **All twelve are MANDATORY**
(`machine.mandatory_harness_checks`).

### S19 HARDWARE-INTEGRITY (new with the single-card pivot)

Every claim-bearing trace row carries the **frozen cross-agent fingerprint**:
`gpu_name`, `gpu_uuid`, `cuda_visible_bytes`, `driver_version`,
`engine_fingerprint` (vLLM/torch/transformers versions plus the §0 engine
contract), `enable_thinking_effective`, `run_id`, `git_sha`, `config_hash` and a
UTC timestamp. Every `gpu_ledger.jsonl` row carries the same fields. The
fingerprint reaches the analyzer through the trace row's `provenance` block —
the evaluator's `run_meta`, copied verbatim by the runtime layer — so the
analyzer only ever *reads* it and never infers a missing field.

| Condition | Outcome |
|---|---|
| every claim-bearing trace in one `run_id` carries the full fingerprint, one physical GPU UUID, one engine fingerprint, and every paired comparison matches | OK |
| **missing** hardware provenance on any claim-bearing trace | **INCONCLUSIVE**, no winner |
| a **known-wrong card** (GPU name or CUDA-visible byte count that is not the registered one) | **BUG** |
| **mixed** GPU UUID inside one `run_id`, more than one `run_id` in one trace set, or more than one engine fingerprint | **BUG** |
| a paired **BP/TP** or **B0/T0** comparison whose members disagree on any fingerprint field | **BUG** |
| an engine fingerprint that contradicts the registered engine contract (utilisation, context length, thinking) | **BUG** |

An **independent replication on another A5000** is legitimate science and
requires a **new `run_id`** with its own trace set; it may **never append** to an
existing trace set, because appending is precisely how two cards end up inside
one paired claim. Every direction S19 can move a verdict is unfavourable — it
vetoes or withholds, never promotes — which is what makes adding it before any
GPU-hour exists an outcome-blind strengthening.

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

S16 also owns the **missing-GRPO-checkpoint** rule. When the locked stage is not
`grpo`, no GRPO checkpoint exists, and the run must say why in the durable
disposition artifact `results/agentic/grpo_disposition.json` carrying an allowed
label (§9). **A missing checkpoint with no allowed disposition is a BUG and never
silently selects RS-SFT**: "GRPO was skipped for a reason nobody wrote down" is
indistinguishable from "GRPO ran and lost the dev comparison", and only one of
those is a reportable outcome.

S18 uses a self-referential seed commitment: `heldout_seed =
SHA256(<preregistration commit sha> + ":agentic-heldout-v1")[:8]` as a big-
endian integer. The commit that introduces this protocol *is* the commitment;
`results/agentic/locks.json` (checkpoint + prompt winner) must exist before
`results/agentic/seed_reveal.json` is written.

> **AMENDMENT 2026-08-05 (pre-run; zero GPU-hours, zero held-out results
> existed).** As implemented, S18 delivers **test-set commitment, not
> test-blindness**, and the veto must not be described as the latter until this
> is closed. The held-out specs, KBs, oracles and answers are absent from the
> repository and their hashes are pinned, so the set cannot be swapped after the
> fact — that part holds. But `scripts/generate_suite.py` is committed and the
> eval generation seed in `configs/suite_v1.toml` is public, and byte-identical
> regeneration is itself an acceptance criterion, so all 1,200 held-out answers
> are recoverable from this commit before any lock exists. The `heldout_seed`
> above is never consumed by the generator.
>
> The **registered target is real blindness**: derive the held-out *generation*
> seed from public entropy that exists only after `results/agentic/locks.json` is
> committed — the locks commit SHA — so the held-out set cannot be generated
> until the prompt winner and trained checkpoint are already frozen. Under that
> scheme this push pins train/dev only, and the held-out artifacts are pinned at
> reveal.
>
> Until that lands, `finalize-prereg` must not be run and no GPU stage may
> start. This amendment changes no threshold, margin, sample size or claim
> definition.

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

## 9. Budget: the 120-hour A5000 ceiling, and what is conditional on it

The ceiling is **120.0 measured A5000 GPU-hours** summed over every GPU stage —
measured from the ledger, never a nominal per-stage range and never a
caller-supplied projection. Server startup, compilation and CUDA-graph capture
count as measured occupancy.

**Hardware.** One RTX A5000, 25,282,805,760 CUDA-visible bytes, one physical
UUID bound for the whole run, no conditional second-card branch, engine contract
per §0 (S19 enforces all of it).

**Why the previous 36.0-hour envelope was raised, before any hour was spent.**
The mandatory BP/TP paired evaluation — which may never shrink — projects to
**75–99 A5000 GPU-hours** from *directly measured* prior evaluator rates:
**25.10 s/episode** for the base arm (5,019.8 s / 200 episodes) and
**44.27 s/episode** for the trained arm (8,854 s / 200 episodes), over the
**7,800** mandatory episodes below. That is 27.19–35.78 h for BP plus
47.96–63.10 h for TP; with the non-evaluation stages the whole-program evidence
envelope is roughly **87–118 h**. Keeping the old figure would not have made the
work cheaper — it would have silently forced a mandatory-sample shrinkage, which
this protocol forbids. **120.0 is the minimum defensible pre-calibration
budget.** Raising an accounting envelope while the ledger is empty, no checkpoint
exists and no held-out result exists is legitimate; shrinking a mandatory sample
after a number is visible is not.

**The ~14–21 hour figure is an unvalidated EXPECTATION, never a registered
measurement.** If the new concurrent vLLM evaluator validates a 3.3–3.9×
speed-up over the old serial evaluator, the whole program may finish in about
14–21 h. That number may not be cited as a measured cost, may not be used to
justify a smaller ceiling, and may not be entered into the ledger. Only measured
seconds are accounting.

**Mandatory episode census (corrected): 7,800.**

| Stratum | Episodes |
|---|---:|
| core: 1,200 tasks × clean/faulted × 2 arms | 4,800 |
| MT augmentation: 600 × 2 arms | 1,200 |
| H8: **200 ADDITIONAL** tasks × 2 arms (the other 200 of the registered 400 are already in core) | 400 |
| absent-information control: 600 × 2 arms (clean only) | 1,200 |
| permutation control: 100 × 2 arms | 200 |
| **mandatory total** | **7,800** |

Optional and outside that total: the two-fault stress set (280 × 2 = 560) and
the descriptive B0/T0 arms; R0/RP are absent by design. The controls are
registered as **clean** controls — adding faulted controls would be a **new
pre-registration choice** and is deliberately not made here. No registered
cardinality changed: this corrects an arithmetic census, not a sample size.

**The final paired evaluation, not GRPO, is the stage most likely to consume the
ceiling** — 75–99 projected hours for those 7,800 episodes. The earlier
"around seven GPU-hours at ~7.85 decision-generations/s" figure translated a
fully batched rejection-sampling rate into dependency-serial online tool
episodes and is superseded as a forecast; the registered episode counts it
referred to are unchanged. If continuous batching is validated, production
rejection sampling (≈ 16,400 rollouts, mean H ≈ 9.07) becomes the largest stage
instead — which is exactly why calibration is load-bearing.

**MANDATORY (never budget-conditional):** BP and TP across core (clean and
faulted), MT, H8, and **all** controls — the 7,800 episodes above.

**Cut order if calibration projects an overrun** — in this order and only this
order, and only on a projection made *before* the relevant stage runs, never
after seeing a held-out result from any arm:

1. the entire **GRPO branch**;
2. its **variance probe** and the **R0/RP** arms;
3. the descriptive **B0/T0** arms;
4. the two-fault **stress** set.

**The order is unchanged by the hardware pivot.** Ranks 1 and 2 are now *moot*
rather than reordered: GRPO is not run at all (the disposition below) and the
variance probe is `NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT`, so the first cut
actually available to a calibration overrun is rank 3, then rank 4. Nothing was
promoted, demoted or reclassified; no mandatory arm became optional and no
optional arm became mandatory.

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
**Calibration is load-bearing under this ceiling**: the pre-calibration evidence
envelope (87–118 h) and the unvalidated concurrent-evaluator expectation
(14–21 h) differ by roughly 5×, and which of them is true decides whether the
mandatory program fits. It runs on dev-only task IDs, before the held-out seed
reveal, on the exact production engine contract; concurrency is chosen from
timing, OOM/preemption and latency stability **only**, never from a capability
outcome. If the post-calibration projection exceeds the ceiling, cut optional
arms in the frozen order above; **if MANDATORY work still does not fit, STOP and
report INCOMPLETE / INCONCLUSIVE.** Mandatory samples may never shrink.

**Ledger accounting.** `results/agentic/gpu_ledger.jsonl` is append-only; every
GPU stage records stage name, start/end UTC, **measured** elapsed GPU seconds,
git sha, the full S19 fingerprint (GPU name, GPU UUID, CUDA-visible bytes, driver
version, engine fingerprint, effective thinking mode, `run_id`, config hash), and
the episode or step count actually completed. The ceiling is enforced against the
**sum of measured seconds**, and server startup, compilation and graph capture
count. A stage may not start when its calibrated projection plus the ledger total
would cross 120.0 hours. A shard that overruns its projection is still charged
its measured time and the overrun is reported, not absorbed. Every GPU consumer —
**including the final evaluator** — must read and append to the ledger; a stage
absent from the ledger is unaccounted time and makes the budget claim
INCONCLUSIVE. A ledger row whose GPU UUID disagrees with the run's first bound
UUID is fatal.

### GRPO: NOT RUN — `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`

Nothing in the primary or secondary claims depends on GRPO running.

**The registered GRPO recipe cannot instantiate on the registered card.** The
arithmetic, recorded before any GPU-hour was spent: the registered
`vllm_gpu_memory_utilization` 0.24 × 23.546 GiB = **5.651 GiB** is *smaller than*
vLLM's own **8.455 GiB** colocated policy copy — a 2.804 GiB shortfall before any
trainer weight — and the trainer's **9.420 GiB** plus that copy is already
**17.875 GiB** of 23.546 GiB before any KV cache, CUDA graphs, or the
**1.53 GiB per 3,072-token sequence** of logits. The configuration is infeasible,
not merely tight.

**The exact stage-outcome label is `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`.** It is a
**stage DISPOSITION**, never one of the gate states PASS / FAIL / INCONCLUSIVE /
BUG, is never written into a gate, claim, floor or winner field, and is
deliberately absent from `outcome_states`: "the trainer could not be
instantiated" is not a statement about the model.

It is **not interchangeable** with `GRPO_NOT_RUN_VARIANCE_GATE_CLOSED`, which
would mean the complete 144-group / 1,152-rollout probe ran and a binding pooled
gate failed — real evidence about the RS-SFT policy's reward and outcome
variance. **The variance probe is not run either**: its status is
`NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT`, and it may never be described as
"closed".

Frozen preconditions for the disposition:

1. the registered hardware is exactly one RTX A5000 with 25,282,805,760
   CUDA-visible bytes;
2. the registered GRPO recipe is **unchanged** — bf16 Qwen3.5-4B, LoRA r=32, 8
   generations, per-device batch 4, accumulation 2, `max_completion_length` 3072,
   `vllm_max_model_length` 6144, `vllm_gpu_memory_utilization` 0.24 (an
   infeasible recipe is still the *registered* recipe; rewriting it would define
   a different experiment);
3. a **deterministic preflight** records that 5.651 GiB < 8.455 GiB;
4. **no optimizer step and no held-out evaluation** precedes the disposition;
5. **no substituted variant** runs under this registered branch — not microbatch
   1, not 2,048-token completions, not no-vLLM generation, not quantization, not
   offload, not an alternate optimizer, not another card; each defines a
   **different treatment**;
6. a **durable artifact** `results/agentic/grpo_disposition.json` records GPU
   name, CUDA-visible bytes, config hash, checkpoint hash, git SHA, the
   arithmetic, the UTC time and **zero** GRPO optimizer steps;
7. **no GRPO checkpoint exists; R0/RP are absent by design; RS-SFT is the sole
   trained candidate** eligible for the checkpoint lock.

`agentlab.multigrpo` is **not written**: the GRPO branch is out of v1 scope. A
future GRPO study — on different hardware, or with a redesigned batch — is a
separate pre-registration with its own locks, seeds, ledger and hardware
declaration. The analyzer **requires** the artifact whenever the locked stage is
not `grpo` (§7, S16), so RS-SFT can never be selected by default and by silence.

**`scenario_seed` rule.** GRPO training data is a **map-style** dataset keyed by
`scenario_seed`. All G generations in a group must reconstruct byte-identically
the same prompt, KB, environment, fault target, fault class, node index and
budgets from that one seed; **only the sampling seed differs**. A group whose
members cannot be shown to share one reconstructed scenario is discarded, not
repaired.

**Variance probe** — registered, and **NOT RUN** in this study
(`NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT`; the hardware result already settles the
branch, and a probe result would answer a question no arm of this run asks). The
registered definition is preserved verbatim so that a future run cannot quietly
weaken it: `src/agentlab/variance.py`, which must exist and be tested before GRPO
is implementable, run on the locked RS-SFT checkpoint **before any GRPO update**,
on a disjoint committed probe split, with the exact GRPO decoding settings and the
**exact production reward function** — one shared reward implementation imported
by both probe and GRPO. **48 groups per family × 3 families = 144 groups × 8
generations = 1,152 rollouts**, eight distinct committed sampling seeds per group
with every non-sampling input identical.

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

**Checkpoint selection (RS-SFT vs GRPO).** In this run there is nothing to
select between: hardware-infeasible GRPO leaves **RS-SFT as the sole trained
candidate**, so the comparison below is never reached. The rule stays registered
verbatim because a future run that does produce a GRPO checkpoint must implement
it, and because deleting it would let "RS-SFT was locked" read as the outcome of a
comparison that never happened. Exactly one trained checkpoint is
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

## 11. AMENDMENT 2026-08-05 — one fault contract, one success predicate (D2)

> **This changes the registered training treatment.** Prompt selection,
> rejection sampling, SFT recovery views and any variance work will now expose
> recovery tokens, remediation text and receipts that the previous training
> runtime omitted, and will filter recovery with the registered remediation
> predicate. Recorded **before** any prompt tournament or production rollout.

### What was wrong

The repository implemented **two environments**.

| surface | training path (`suite.runtime.EpisodeRuntime` + `FaultEngine`) | evaluation path (`suite.evaluate.SpecRuntime` + `FaultInjector`) |
|---|---|---|
| error envelope | tokenless; `request_id`, sometimes `event_id` | `recovery_token` + `remediation` |
| tool schema | no `recovery_token` argument | locally augmented with it |
| token handling | never parsed or stripped | parsed and stripped |
| recovery | any later canonical result sufficed | token / timing / corrected-unit contract |
| task success | oracle nodes, final state, capability provenance, budgets | answer source, receipts, hallucination, runaway |
| clean result | included `event_id`, no receipt | no `event_id`, appended `receipt: r-…` |
| transcript | assistant tool-call object and tool `name` preserved | **tool-call object dropped**, tool result nameless |
| wrong unit | committed `FaultSpec.params["wrong_unit"]` | **re-derived** through `wrong_unit_target` |

Both imported "the canonical modules"; both passed their own tests. The
consequences were outcome-blind (no GPU stage had run) and material:

* the policy would have been **trained in a different environment from the one
  it is scored in**, for all four fault classes;
* two definitions of strict success survived and the **weaker one was what the
  SFT acceptance filter enforced**, so the corpus could contain trajectories the
  claim-bearing certifier labels `blind_retry`;
* `p4_error_repair` and `p8_combined` instruct the model about `recovery_token`,
  but the tournament that SELECTS the winning prompt ran tokenless, where that
  instruction is **inert**; `p5_provenance` promises a receipt on every tool
  result, and offline rollouts had none, so it was inert too;
* `provenance.certify_episode` — the function the gates are denominated in —
  contained **zero** references to oracle-node completion, the fulfillment final
  state, capability-token provenance or the call budget.

### What is registered now

**One runtime.** `suite.runtime.EpisodeRuntime` and its strict verifier, for
prompt selection, rejection sampling, SFT acceptance, variance work, evaluation
and analyzer recomputation. `SpecRuntime` and `FaultInjector` are deleted.

**One tool schema.** `tool_schemas_for_family` declares the optional
`recovery_token` argument on **every** tool of **every** family — the model does
not know in advance which tool will fault. Parsed and stripped in exactly one
place; it never reaches canonical tool semantics, oracle matching or the semantic
call digest.

**One model-visible observation form**, on every observation including clean ones:

```text
<canonical envelope, sorted keys, compact separators>
receipt: r-<32 hex>
```

No `event_id`, no `request_id` — call and event ids stay in the hidden ledger.

**One event format.** The `suite.schema.TraceEvent` superset, serialized by both
paths: `exposed_text`, `exposed_result_digest`, `receipt`,
`model_visible_digest`, `canonical_semantic_digest`, `token_provided`,
`recovery_token`, `requested_unit`, plus the existing oracle/credit/decision/
mutation/replay/state/capability fields.

**One remediation predicate**, in `suite.verify`. The event that ESTABLISHES the
post-fault canonical result must itself satisfy the contract: the exact emitted
token on the same stripped call identity (transient, malformed); additionally a
strictly later decision (rate_limit); a later `unit_convert` explicitly
requesting the original target unit (wrong_unit); a token-bearing idempotent
replay with `replay=True` (ambiguous malformed mutation). Labels:
`ok` · `blind_retry` · `no_remediation` · `no_post_fault_result` ·
`not_exposed`. **A status query after an ambiguous malformed mutation is no
longer certified recovery** — it neither reissues the same call nor echoes the
token; certifying it again would need its own amendment.

**One success predicate**, `certified_success`: committed exact answer ∧ every
oracle node completed in order ∧ every dependency crossed a later assistant
decision ∧ runtime/verifier credit maps agree ∧ the terminal answer came from a
validated observation ∧ receipt chain valid ∧ no hallucinated result ∧ no unsafe
mutation ∧ capability tokens observed before use ∧ fulfillment final state
equals `oracle_final` ∧ calls ≤ `max_calls` ∧ decisions ≤ `max_decisions` ∧ no
wall/parser/call-cap termination ∧ no runaway ∧ every assigned fault has a
certifying recovery event. `strict_success` is **removed** rather than left as a
second, weaker headline; `answer_ok`/`raw_success` and `task_success` remain
diagnostics.

**The equality-cap ruling.** An episode that succeeds using exactly `max_calls`
is within budget and is not runaway. Only exceeding the cap, or the runner
reporting that it terminated at the cap, is runaway. §3's "call cap reached"
means the latter.

**`environment_contract_sha256`** — a digest of the model-visible surface itself
(tool-schema bytes, observation form, every fault envelope, the remediation
predicate, the success predicate, the budget formulas) — is stamped into
certification specs, prompt-tournament rows, raw rejection-sampling shards,
accepted records, SFT view metadata and evaluation traces. **Any artifact whose
stamp is absent or stale is invalidated, not resumed.** That includes every
prompt-tournament row, raw shard, accepted record and SFT view produced before
this amendment, and the certification specs, which are regenerated.

### What is NOT changed

No threshold, margin, sample size, estimand, launch floor or held-out result
motivated or was touched by this amendment. The paired TP–BP contrast is not
retroactively invalid — both arms always faced the identical evaluation
environment — but "one fault engine with recovery tokens" was false as the tree
stood, and a reader would wrongly have concluded the trained arm was trained on
the contract it is scored against.

### Excluded prior evidence

The isolated **12-episode `verify-a5000` dev verification run** is disclosed as
**excluded, clean-only apparatus evidence**. It contained no fault-recovery
outcomes and never entered the study trace set. It carries no
`environment_contract_sha256`, uses the retired event format, and visibly
contains both transcript drifts (an empty assistant message where a tool-call
object belongs, and nameless tool results). It is refused by the invalidation
rule rather than re-certified; the answer-grammar regression it exposed is
preserved as inlined fixtures in `tests/suite/test_answer_extraction.py`.

### State of the study at the moment of this amendment

Zero registered study GPU-hours · zero optimizer steps · zero held-out results ·
no frozen prompt winner · no trained checkpoint · preregistration finalization
marker absent.

### Finalization refusal

`finalize-prereg` may not run unless the observational-equivalence test
(`tests/test_environment_parity.py`, all 12 family/horizon cells × clean + every
eligible fault class + the ambiguous malformed mutation) passes and the
tokenizer size census below has been recorded.

### Size census and the caps it moved

`scenario.tool_output_max_tokens` was **208 — measured against the smaller
tokenless payloads** (no recovery token, no remediation text, no receipt line),
so it was stale the moment the contract was unified. Re-measured **exhaustively**
by `scripts/token_census.py` with the Qwen3.5-4B tokenizer (`transformers`
5.14.1, `tokenizer.json` sha256 `5f9e4d49…cb42`, `chat_template.jinja` sha256
`a4aee8af…8f715`) over the four committed train/dev splits (`distill`,
`oracle_sft`, `grpo_train`, `dev`), all twelve family/horizon cells, the clean
case, every eligible fault class, the same-decision rate-limit repeat and the
ambiguous malformed mutation, crossed with **all eight** preregistered prompt
candidates:

| stratum | measured | n |
|---|---|---|
| model-visible tool result | **231 tokens**, 474 chars (`oracle_sft-fulfillment-h20-0112`, clean H20 order query — it must list every line with its capability token) | 742,500 observations |
| rendered terminal view | **4,960 tokens** (`distill-fulfillment-h20-0034`, same-decision rate-limit repeat under `p8_combined`, the longest prompt) | 624,800 views |
| episodes | | 78,100 |

Artifact `results/agentic/token_census.json`, sha256
`98379c42540a30d7c6c29fea53193a169714351f184f9c58b516484dc5896fa8`, carrying the
per-stratum maxima and the offending task ids.

| cap | was | now | why |
|---|---|---|---|
| `scenario.tool_output_max_tokens` | 208 | **256** | 231 measured; the registered definition says "model-visible tool result", so excluding the receipt would be misleading |
| `scenario.tool_output_max_chars` | 512 | **512** (unchanged) | 474 measured |
| `acceptance.max_view_tokens` | 4096 | **5120** | 4,960 measured; 4096 would structurally exclude otherwise valid H20 trajectories that fit under the retired tokenless treatment |
| `sft.max_length` | 4096 | **5120** | moves with the view budget — a view the builder accepts and the trainer truncates is a silently different training signal |

**Not changed by this census:** call budgets, decision budgets,
`decoding.max_tokens_per_decision` (384), `eval_decoding.max_tokens_per_decision`
(1024), the GRPO completion caps, and the **8,192-token serving context**.
Recovery still costs one call and one decision, and tool observations consume
*input* context, not assistant completion tokens.

**A5000 SFT arithmetic rechecked at 5,120 tokens.** One bf16 logit tensor over
the 248,320-token vocabulary grows from 2.034 GB (1.895 GiB) at 4,096 to 2.543 GB
(2.368 GiB) at 5,120, so train batch 2 peaks at 9.423 + 2 × 2.368 = **14.16 GiB**
of the 23.5 GiB card (was 13.21 GiB) and eval batch 1 at 11.79 GiB. The Hugging
Face default eval batch 8 would need 28.37 GiB and still does not fit, which is
why `eval_bsz` stays 1 with `prediction_loss_only`.

`tests/test_size_ceilings.py` binds every cap to the committed census — including
its environment-contract digest, so a census taken under a different environment
is rejected — and fails if a cap ever drops below what was measured.
