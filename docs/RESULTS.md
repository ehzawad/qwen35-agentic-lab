# Results — agentic-v1

**Status of this document: SKELETON. Dated 2026-08-06.** Every held-out slot below is
deliberately **UNFILLED**. No trained checkpoint exists, `L` does not exist, no held-out
byte has been generated, and no registered gate has been evaluated, so there is nothing
to fill them with. This file exists *now* so that the wording is fixed *before* the
numbers are visible; the `verdict` stage completes it.

Two rules govern every edit to this file:

1. **A bracketed slot is filled only from the analyzer's emitted output**, never from
   prose, memory, a projection or a plausible-looking round number. If the analyzer did
   not emit it, the slot stays unfilled and the reason is stated.
2. **The verbatim wording in §1–§5 may be added to but never softened, shortened or
   paraphrased.** It is quoted here from the referee ruling of council round 8 and, where
   round 8 says so, from [docs/INTERPRETATION.md](INTERPRETATION.md) §2, which fixed the
   same wording earlier. Prose blockquotes are hard-wrapped at 88 columns to match the
   rest of `docs/`; wrapping moves no word.

Everything measured today is **dev** or **distill** evidence, i.e. training-side. It is in
§6, under its own heading, and it is not a held-out result. Read §6 with §7's label
table.

---

## 0. Where the study actually is, as of 2026-08-06

| fact | state |
|---|---|
| `P` — preregistration finalized | commit `5844a97d2096fb55186e8559f4a4481dc3b75e9d`, pushed |
| `L` — checkpoint lock | **does not exist**; `scripts/agentic_locks.py status` REFUSED it (incomplete lock) |
| `R` — seed reveal | does not exist; the six held-out generation seeds are a function of `L` |
| registered 7,800-episode evaluation | **planned and being completed**, not descoped — see [docs/AMENDED_REPLICATION_NOT_RUN.md](AMENDED_REPLICATION_NOT_RUN.md) |
| optional cluster-balanced amended replication | **deliberately not run**; it was never registered, so it is not a deviation |
| held-out results | **none exist** |
| deviation notice | **none is required today**, and none is published; §1 states the exact condition that would change that |

The registered mandatory census is **7,800 BP/TP episodes**: core 4,800 (1,200 tasks ×
clean/faulted × 2 arms), MT augmentation 1,200, H8 augmentation 400, absent-information
control 1,200, counterfactual permutation control 200. Mandatory samples may never
shrink after a result is visible; if they do not fit, the run **STOPS** and reports
INCOMPLETE / INCONCLUSIVE. That is the registered rule and it is unchanged.

---

## 1. Study status — VERBATIM, conditional

**When this paragraph applies.** The referee's ruling is that the study is reported as
DEVIATED / PARTIALLY COMPLETED **only if the registered evaluation is not completed**.
The current plan is to complete the registered 7,800-episode evaluation — its measured
throughput puts it at roughly **6.4 A5000 GPU-hours**, arithmetic and caveats in
[docs/AMENDED_REPLICATION_NOT_RUN.md](AMENDED_REPLICATION_NOT_RUN.md) — so as of the date
of this file **the paragraph below is a template that does not yet apply, and no
deviation notice is due.**

**The tripwire.** If the registered 7,800-episode evaluation is *not* completed — for any
reason: budget, hardware, time, a reduced audit substituted for it — then the following
paragraph becomes the study status verbatim, a dated canonical notice is published at
`docs/DEVIATION_<date>.md`, a prominent link and one-sentence status go in the README, and
this paragraph is reproduced here. Silence would not be an option; a partial mandatory
sample may never be presented as the registered one.

> The preregistered 7,800-episode evaluation was not completed. This study is reported as
> a deliberate post-registration deviation and partial completion. No preregistered
> study-level winner is claimed. The reduced evaluation reported below, if any, was
> prospectively fixed before checkpoint lock but is post-registration descriptive
> evidence, not a substitute for the original confirmatory evaluation.

If a reduced held-out audit is ever proposed, the referee's six-condition reduced-size
rule applies in full and is recorded in
[docs/AMENDED_REPLICATION_NOT_RUN.md](AMENDED_REPLICATION_NOT_RUN.md) §4. Its first
condition is that the exact integer counts are committed before `L`; its second forbids
deriving the size from any visible capability rate.

---

## 2. Saturated secondaries — VERBATIM, must precede the held-out secondary results

This is the repository's already-fixed wording ([docs/INTERPRETATION.md](INTERPRETATION.md)
§2, quoted unchanged by council round 8):

> Before any trained checkpoint was evaluated and before the held-out suite was realized,
> the selected prompt-only control achieved 295/300 (0.9833) certified successes on the
> development H4 all-tools orchestration axis and 296/300 (0.9867) on the development H8
> execution axis. On those realized development samples, even perfect trained-arm
> performance could improve the respective rates by only 0.0167 and 0.0133, both below
> the preregistered +0.05 superiority margin. These development results diagnose ceiling
> limitation of the endpoints; they are not held-out estimates and do not determine the
> registered gate statuses.

---

## 3. Per-gate reporting template — VERBATIM, for every gate

> We evaluated the unchanged preregistered contrast on the observations available. BP
> achieved [kBP/n], TP achieved [kTP/n], the paired difference was [Δ], and the one-sided
> 97.5% structural-cluster bootstrap lower bound was [LB] over [G] clusters, with [d]
> discordant pairs. The preregistered gate status was [PASS/FAIL/INCONCLUSIVE], for the
> following registered reason: [reason]. A non-PASS means that improvement greater than
> five percentage points was not established on this endpoint. If the status is FAIL, the
> registered >0.05 superiority claim failed. Neither status is evidence that the training
> intervention had zero effect.

[docs/INTERPRETATION.md](INTERPRETATION.md) §2 records an earlier variant of this
template. Where the two differ, **use the round-8 text above**: it is strictly more
demanding, because it also requires the registered reason to be named and requires the
sentence "the registered >0.05 superiority claim failed" when the status is FAIL. Neither
version is softened by the other.

Every gate gets its own filled instance of that paragraph. The gates are registered; this
table is the checklist, not a place for numbers:

| gate | registered condition (abbreviated; `docs/AGENTIC_PROTOCOL.md` §5 governs) | status |
|---|---|---|
| ER1 | `\|C\| ≥ 500` common-clean pairs, else INCONCLUSIVE | [ ] |
| ER2 | recovery diff on `C`, one-sided 97.5% clustered LB > +0.05 | [ ] |
| ER3 | TP recovery on `C`, Wilson LB ≥ 0.60 | [ ] |
| ER4 | clean non-inferiority, clustered LB > −0.03 | [ ] |
| ER5 | ITT diff ≥ 0 over ≥ 900 assigned pairs | [ ] |
| ER6 | TP runaway Wilson UB ≤ 3% (core stratum only) | [ ] |
| ER7 | TP hallucinated-result Wilson UB ≤ 1% (core stratum only) | [ ] |
| ER8 | no fault group below −0.05, all three groups present at 400 assigned each | [ ] |
| MT1 | certified all-tools diff LB > +0.05 on ≥ 600 H4 pairs | [ ] |
| MT2 | TP Wilson LB ≥ 0.60 | [ ] |
| MT3 | TP median calls ≤ oracle+2 (median = upper middle) | [ ] |
| MT4 | runaway UB ≤ 3% and hallucination UB ≤ 1% on MT tasks | [ ] |
| MT5 | none of the six registered order patterns below −0.05, ≥ 80 pairs each | [ ] |
| MT6 | the absent-information control is clean | [ ] |
| HR1 | H8 clean certified diff LB > +0.05 on ≥ 400 pairs with ≥ 20 discordant | [ ] |
| HR2 | TP H8 runaway UB ≤ 3% | [ ] |
| HR3 | TP H8 hallucination UB ≤ 1% | [ ] |

Reported beside the gates, never instead of them: Holm-adjusted one-sided exact McNemar
p-values over the family `{MT1, HR1}` (α 0.05), `n_clusters` on every interval, and the
emitted stratum census including every excluded trace.

**Harness vetoes are checked before any gate** and all twelve are mandatory: S8 PAIRING ·
S9 ORACLE · S10 SPLITS · S11 ABSENT-INFO · S12 INJECTION · S13 RECEIPTS ·
S14 COUNTERFACTUAL · S15 ATTRITION · S16 CONTROL-INTEGRITY · S17 TRACE-SUMMARY ·
S18 POST-LOCK HELDOUT · S19 HARDWARE-INTEGRITY. A BUG vetoes every model-level gate,
floor and claim; an INCONCLUSIVE veto yields NO VERDICT. Their statuses are reported
exactly as emitted, before the gate table.

**Launch floors** (point estimates against the threshold, Wilson intervals reported
beside them but not deciding), each over its frozen core denominator: F1 overall clean
certified ≥ 0.65; F2 no family < 0.50 clean; F3 overall faulted ITT ≥ 0.40; F4 no family
< 0.25 faulted; F5 loop/crash < 0.02.

**Winner rule** — the frozen truth table in `docs/AGENTIC_PROTOCOL.md` §8, evaluated in
rank order, first match decides. Rank 5 ("BP ships") is a real preregistered outcome,
including when the trained comparison FAILS or is underpowered, and may not be reported
as "no successful pipeline".

---

## 4. Underpowered primary — VERBATIM

> The preregistered primary status was [PASS/FAIL/INCONCLUSIVE], with [|C|] common-clean
> pairs across [G] structural clusters and a TP−BP certified-recovery difference of [Δ],
> lower bound [LB]. [State every registered count, cluster, or gate condition that
> failed.] The structural audit conducted before held-out realization showed that
> plausible improvements concentrated in the two deep fulfillment clusters would have low
> probability of clearing the registered clustered lower-bound gate. This limits what the
> endpoint can establish; it does not establish that training is ineffective. Because the
> registered mandatory evaluation was not completed, this result does not supply a
> registered study-level winner.

The final sentence is conditional in exactly the same way as §1: it is used **only if**
the registered mandatory evaluation was in fact not completed. If the 7,800 mandatory
episodes are completed, that sentence is dropped and the registered winner rule decides
normally — the rest of the paragraph, including the audit sentence and the
"does not establish that training is ineffective" sentence, still applies verbatim.

The composition of `C` is reported alongside the primary, by family, horizon and
structural cluster: both-clean-successful, `BP`-only, `TP`-only, neither.

---

## 5. Any descriptive number — VERBATIM prefix

Every descriptive number published anywhere in this repository — this file, the README,
`docs/USAGE.md`, a commit message, a model card, a post — carries this prefix:

> **Descriptive only; not a preregistered claim:** [metric and exact
> numerator/denominator]. This estimate carries no registered decision threshold, does
> not change or replace any original gate, and must not be read as a confirmatory claim
> about training efficacy or general agentic capability.

---

## 6. Training-side descriptive observations (not held-out)

**Descriptive only; not a preregistered claim.** Everything in this section is measured on
the **distillation split**, from the **base model plus the frozen winning prompt**, with
**repeated samples per task**. It is the corpus the training data was built from. It is
not the adapter, not an independent task sample, and not a held-out estimate. No gate is
computed here and none of these numbers may be filled into §3 or §4.

**Snapshot.** The recorded snapshot is the **13 sealed rejection-sampling shards** as of
`2026-08-06T16:46:19Z`, containing exactly **3,952** raw rollouts and accounting for
**8.308** measured A5000 GPU-hours (`data/multiface/raw/shard-00{00..12}.receipt.json`;
cumulative ledger value in `results/agentic/gpu_ledger.jsonl`). Base model plus frozen
prompt produced **2,742/3,952 (69.4%)** certified successes over that snapshot. The stage
is still running, so §10 states how later counts replace these.

### 6.1 Fulfillment H4 and H8 certified success

| distillation cell | certified successes / rollouts | rate |
|---|---|---|
| `fulfillment-h4` | **1,174/1,200** | 97.8% |
| `fulfillment-h8` | **1,509/1,600** | 94.3% |

These are certified fulfillment-task rollouts from the **base model plus frozen prompt**,
with **repeated samples per task**. From the shard receipts' `samples_per_task` field and
the distinct task IDs in the rows: H4 is **6 samples × 200 tasks = 1,200**, H8 is
**8 samples × 200 tasks = 1,600**. They are **NOT the adapter**,
**NOT an independent task sample**, and **NOT evidence about arbitrary multi-tool use** —
the suite offers exactly five tools and `fulfillment` is the only family offered the two
warehouse operations. "94–98% tool calling" is not an acceptable restatement of these
rows; the numerators and denominators are the finding.

### 6.2 Certified recovery under the registered fault contract

**1,302/1,752 (74.3%)** fault-assigned rollouts met the exact **token-bearing recovery
predicate** registered in `docs/AGENTIC_PROTOCOL.md` §11 and implemented in
`agentlab.suite.verify`.

This is a **mixture over the observed distillation cells**. It is **NOT BP-on-`C`** (the
primary estimand conditions on the common-clean subset `C`, which this corpus never
constructs), **NOT a TP−BP contrast** (there is no TP), and **NOT recovery from arbitrary
production failures** (the faults are injected under one committed synthetic contract).

The mixture, exactly, at the same snapshot — published because the pooled rate moves with
the cell composition, not only with model behaviour:

| fault-assigned cell | recovery-predicate met / fault-assigned | rate |
|---|---|---|
| `fulfillment-h4` | 575/600 | 95.8% |
| `fulfillment-h8` | 714/800 | 89.3% |
| `fulfillment-h14` | 13/352 | 3.7% |
| **pooled** | **1,302/1,752** | **74.3%** |

And by declared fault class at the same snapshot (each fault-assigned rollout carries
exactly one class, so these four rows partition the 1,752):

| declared fault class | recovery-predicate met / fault-assigned |
|---|---|
| `transient` | 405/604 |
| `malformed` | 374/550 |
| `rate_limit` | 394/398 |
| `wrong_unit` | 129/200 |

### 6.3 The fulfillment H14 generation-yield cliff

**59/1,152 (5.1%)** certified success in the distillation `fulfillment-h14` cell.

That cell was **partially generated** at this snapshot: 1,152 rollouts is 8 samples × 144
distinct tasks, against a registered distillation plan of 200 tasks per cell. The cell
completed after the snapshot; §10 gives the completed count.

This is **solid evidence of a severe within-split generation-yield cliff**. It is **NOT
proof that horizon alone caused the collapse** (the deep cells also change state depth,
budget pressure and irreversible-commitment count together with horizon), **NOT proof
that every deep task fails** (59 rollouts were certified), and **NOT evidence that the
adapter will retain the same rate** (no adapter exists, and a corpus built by rejection
sampling deliberately keeps the successes).

### 6.4 Why no intervals are printed on any row in §6

The rows contain **repeated rollouts for the same tasks** — 6 samples per task in the H4
cell, 8 in the H8 and H14 cells, over 200, 200 and 144 distinct tasks respectively.
Ordinary binomial intervals treating all rollouts as independent would be
**misleading**: they would describe a sample far more independent than the one that was
actually drawn, and would come out too narrow. An interval that respects task and
structural clustering would be admissible; none is computed here. **So this section
publishes exact numerators and denominators without inferential language**, which is the
referee's instruction and the only honest option available without the clustered
machinery.

### 6.5 What §6 covers, and what it does not

At this snapshot the sealed shards contain the **`fulfillment` family only** — the
`typed_relay` and `lookup_chain` distillation cells had not been reached. §6 therefore
says nothing whatsoever about those two families on the distillation split, and no row
above may be read as a whole-suite rate.

---

## 7. Which evidence is which

| label | means | may support |
|---|---|---|
| **dev** | development split, task IDs public in the suite, used to select the prompt | calibration, instrument diagnosis, demos |
| **distill** | distillation split, repeated rollouts, the corpus training data is built from | descriptive training-side observations (§6) |
| **held-out** | generated only after `L` and `R`, seeds derived from the lock | the registered gates, floors and winner |
| **amended held-out** | any held-out sample under a dated amendment committed before `L` | post-registration descriptive evidence only |

The prompt winner was **selected on** the dev axes, so dev rates for `p2_plan_state_act`
are optimistic by selection. Every row published anywhere must carry its label and its
snapshot date; a row without both is not reportable.

---

## 8. Held-out results — EMPTY BY DESIGN

Nothing below is filled. The tables exist so that the shape of the report is fixed before
the numbers are: the analyzer writes into them, and a missing stratum is reported as
missing rather than dropped.

### 8.1 Per family × horizon, both arms, same tasks

The registered grid is **12 primary family/horizon cells**: `lookup_chain` and
`typed_relay` at H 2/4/8/12, `fulfillment` at H 4/8/14/20. The optional stress set runs
over the 7 cells at or above H8.

| family | horizon | condition | BP certified / n | TP certified / n | Δ | Wilson (BP) | Wilson (TP) | label |
|---|---|---|---|---|---|---|---|---|
| `lookup_chain` | 2 / 4 / 8 / 12 | clean, faulted | [ ] | [ ] | [ ] | [ ] | [ ] | held-out |
| `typed_relay` | 2 / 4 / 8 / 12 | clean, faulted | [ ] | [ ] | [ ] | [ ] | [ ] | held-out |
| `fulfillment` | 4 / 8 / 14 / 20 | clean, faulted | [ ] | [ ] | [ ] | [ ] | [ ] | held-out |

Horizon curves are reported with pointwise 95% Wilson bands and **no extrapolation of any
kind**; H50 is reported only if the observed curve crosses 50%, otherwise it is
left-/right-censored.

### 8.2 The four-arm descriptive decomposition

| contrast | reads as | value |
|---|---|---|
| `BP − B0` | prompt elicitation gain for base weights | [ ] |
| `T0 − B0` | training gain under the neutral prompt | [ ] |
| `TP − BP` | training gain once both arms receive the selected prompt | [ ] |
| `(TP − T0) − (BP − B0)` | descriptive prompt-by-training interaction / redundancy | [ ] |

`R0`/`RP` are **absent by design**, never merely missing. `B0`/`T0` are descriptive and
budget-conditional (cut rank 3).

### 8.3 Failure categories, complete denominators

Every episode lands in exactly one category, and the denominators sum to the stratum
census; a category is never reported as a fraction of the successes.

| category | BP / n | TP / n |
|---|---|---|
| certified success | [ ] | [ ] |
| answer wrong | [ ] | [ ] |
| oracle node missed or out of order | [ ] | [ ] |
| dependency not crossed by a later decision | [ ] | [ ] |
| blind retry (recovery attempted, no token) | [ ] | [ ] |
| no remediation after fault | [ ] | [ ] |
| no post-fault result | [ ] | [ ] |
| fault not exposed | [ ] | [ ] |
| budget: calls or decisions exceeded | [ ] | [ ] |
| runaway | [ ] | [ ] |
| hallucinated result | [ ] | [ ] |
| unsafe mutation | [ ] | [ ] |
| parser / wall-clock / crash termination | [ ] | [ ] |
| infrastructure failure (reported separately, never as a model failure) | [ ] | [ ] |

### 8.4 Controls

| control | registered requirement | observed |
|---|---|---|
| absent information (S11) | ≥ 200 redacted per family per arm; certified success is zero by construction; any raw exact success is a **BUG** | [ ] |
| counterfactual permutation (S14) | ≥ 100 permuted replays per arm; output must track the returned value | [ ] |

---

## 9. Corrections that travel with these results

These are known-wrong public records. They are listed here because a results document that
cites them without saying so would inherit the error.

1. **The prompt-tournament receipt `configs/frozen_prompt.json` is malformed, and must be
   corrected before `L`.** Its final ranking and `round2_candidates` name `p8` — a
   candidate with 300 observations that never ran round two — while the actual round-two
   finalist was `p6` at 900. The **winner is unaffected** (`p2` leads at 0.9367 over 900
   against `p6` at 0.9178) and no GPU work needs rerunning. Full entry: D1 in
   [docs/DEFERRED_REPAIRS.md](DEFERRED_REPAIRS.md). Shipping a "frozen winning prompt"
   whose public selection receipt names the wrong finalist is avoidably misleading, so
   this correction is ordered before `L` rather than after the verdict.
2. **The artifact index `ARTIFACTS.json` is mis-scoped and must be corrected or superseded
   before any artifact link is published.** The index declares `run_id: agentic-v1`, but
   **59 of its 75 entries carry `run_id: dev-preflight-v1`** and their remote paths sit
   under a `dev-preflight-v1/` prefix inside the `agentic-v1` dataset repository. The
   bytes, digests and card identity are unchanged, so no GPU rerun and no deletion of
   remote bytes is needed: preserve the historical preflight binding, add a dated
   `agentic-v1` correction, and regenerate or annotate the index.
3. **The earlier "BP recovery is 0.840, so there is 0.16 of headroom" framing is
   withdrawn** as an estimate of the primary control estimand. The tournament recovery
   axis scores fault-assigned dev episodes (252/300); the primary conditions on the
   common-clean subset `C`, which the tournament never constructs, and conditioning on `C`
   may materially change the control recovery rate. See
   [docs/INTERPRETATION.md](INTERPRETATION.md) §5.
4. **GRPO was not run** — disposition `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`, and the variance
   probe is `NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT`, not "closed". A stage disposition is
   never a gate state, and neither may be reported as a failed gate.

---

## 10. Snapshot dates versus final sealed-corpus results

The distillation stage is **live** while this file is being written, so §6 is a snapshot,
not a total. Two facts, both from receipts:

| snapshot | sealed shards | raw rollouts | measured GPU-h (cumulative ledger) |
|---|---|---|---|
| the recorded snapshot used in §6 and in `docs/USAGE.md` | 13 (`shard-0000`…`shard-0012`) | 3,952 | 8.308 |
| the corpus as of `2026-08-06T17:42:45Z` | 15 (`shard-0000`…`shard-0014`) | 4,400 | 9.249 |

**Snapshot figures are replaced by final sealed-corpus counts when the stage closes, and
are never silently mixed.** Any published row states which snapshot it came from.

Why this matters concretely, and not as boilerplate: shards `0013` and `0014` are further
`fulfillment-h14` work — they finish that cell at its registered 200 tasks × 8 samples —
and every one of their 448 rollouts is fault-assigned. Recomputed over 15 shards the same
pooled recovery statistic is **1,313/2,200 (59.7%)** and the `fulfillment-h14` certified
rate is **70/1,600 (4.4%)** — the pooled recovery number moved
by roughly fifteen points without any change to the model, the prompt or the contract,
purely because the cell mixture of the corpus changed. That is exactly why §6.2 publishes
the per-cell decomposition and why the pooled figure is labelled a mixture. Two snapshots
are shown here so that neither is quoted without its shard count; mixing them into one
number would be the error the rule exists to prevent.

---

## 11. Phrases that must not appear in this document

- "Training failed."
- "Training had nothing to add."
- "No training effect."
- "Both claims were inconclusive by saturation" — unless the registered machine
  conditions actually produce INCONCLUSIVE.

Also forbidden: converting a machine `FAIL` into `INCONCLUSIVE` in prose; describing a
`FAIL` as saturation without also saying `FAIL`; omitting a short or missing stratum
instead of reporting it; and describing the absence of a registered PASS as evidence that
the training intervention had zero effect.

---

## 12. Cross-references

- Reporting rules and the analyzer's `FAIL`/`INCONCLUSIVE` semantics:
  [docs/INTERPRETATION.md](INTERPRETATION.md).
- What a user must know before running this: [docs/USAGE.md](USAGE.md).
- The one documented serving path: [docs/SERVING.md](SERVING.md).
- What is deliberately not being run, and why it is not a deviation:
  [docs/AMENDED_REPLICATION_NOT_RUN.md](AMENDED_REPLICATION_NOT_RUN.md).
- Registered gates, floors, winner rule, census and budget:
  `docs/AGENTIC_PROTOCOL.md` (hash-pinned, unedited).
- Repairs a live producer forbade: [docs/DEFERRED_REPAIRS.md](DEFERRED_REPAIRS.md).
