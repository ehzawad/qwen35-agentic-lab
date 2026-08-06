# The amended cluster-balanced replication is NOT being run — and that is not a deviation

**Recorded:** 2026-08-06
**Original preregistration:** `P = 5844a97d2096fb55186e8559f4a4481dc3b75e9d`, finalized and
pushed
**State when this was recorded:** rejection sampling live on the pinned A5000; no trained
checkpoint; `L` does not exist; `R` does not exist; **no held-out byte and no held-out
result exist.**

This file is deliberately **not** named a deviation notice, because there is no deviation
to notice. It records one scope decision, the reason it costs the registered study nothing,
and the exact condition under which a real deviation notice would become mandatory.

---

## 1. What is not being run

The **optional cluster-balanced amended replication** proposed in council round 7 — a
post-preregistration design that would have rebalanced the deep fulfillment structural
clusters so that an effect concentrated in them could clear the registered whole-cluster
bootstrap — **will not be run.**

The referee's own notice text says why that costs nothing, and is quoted verbatim:

> The cluster-balanced replication proposed after preregistration was not part of `P`, was
> never registered, and will not be run.

## 2. Why this is not a deviation

1. **It was never registered.** It is absent from `configs/agentic_preregister.json`, from
   `docs/AGENTIC_PROTOCOL.md` and from the finalization marker
   `configs/preregistration_final.json` that hash-pins them. It was proposed *after* `P`
   was finalized and pushed.
2. **Only registered content can be deviated from.** The amendment rule in
   `configs/agentic_preregister.json` and `docs/AGENTIC_PROTOCOL.md` binds changes to
   registered thresholds, margins, sample sizes, estimands, strata, gates and floors.
   Declining to add an unregistered extra analysis changes none of those. No threshold,
   margin, sample size, seed, estimand, gate, floor or claim moves by not running it.
3. **Nothing published ever promised it.** No committed file states that the replication
   would run; there is nothing to retract, only this record so that a reader who saw the
   round-7 proposal knows what happened to it.
4. **It would have been descriptive at best.** Any post-registration sample is
   post-registration descriptive evidence and cannot establish an original primary,
   secondary, launch or winner claim. Not running it removes no registered claim, and
   running it could not have supplied one.
5. **The registered study is not being shrunk to pay for it.** The opposite: see §3.

## 3. What IS being completed — the registered 7,800-episode evaluation

The registered mandatory census is being executed **in full, unshrunk**:

| stratum | episodes |
|---|---:|
| core: 1,200 tasks × clean/faulted × 2 arms | 4,800 |
| MT augmentation: 600 × 2 arms | 1,200 |
| H8 augmentation: 200 additional tasks × 2 arms | 400 |
| absent-information control: 600 × 2 arms (clean only) | 1,200 |
| counterfactual permutation control: 100 × 2 arms | 200 |
| **mandatory total** | **7,800** |

### 3.1 What it is measured to cost

**~6.4 A5000 GPU-hours**, projected from a *measured* rate rather than from a forecast.
The measurement is the prompt-tournament stage in `results/agentic/gpu_ledger.jsonl`:

| ledger component | rows | minutes |
|---|---:|---:|
| `prompt_tournament` units (8 × 300 round-1 + 2 × 600 round-2 = 3,600 dev episodes) | 10 | 175.450 |
| `prompt_tournament:engine_start` | 2 | 1.850 |
| `prompt_tournament:uncharged_gpu_time` (session remainder) | 1 | 0.290 |
| **total for 3,600 episodes** | **13** | **177.590** |

That is **177.6 measured GPU-minutes per 3,600 episodes = 2.9598 s/episode**, and
7,800 × 2.9598 s = 23,086 s = **6.413 h ≈ 6.4 GPU-h**. Engine startup is inside that
figure, not outside it, because the ledger charges it.

### 3.2 The caveats that come with that number, stated plainly

- **It is a projection from a measured rate, not the measured cost of an evaluation that
  has not run.** Only measured seconds are accounting; this number may not be entered into
  the ledger.
- **It was measured on one arm.** The tournament ran base weights under eight candidate
  prompts on dev task IDs. The registered evaluation is a **paired BP/TP** evaluation, and
  the older serial rates that produced the protocol's pre-calibration envelope had the
  trained arm at **44.27 s/episode against 25.10 s/episode** for the base arm — a factor
  of 1.76. If that arm ratio held while the absolute rate stayed at the tournament's, the
  7,800 episodes split evenly across arms would come to about **8.9 GPU-h** (3.2 h BP plus
  5.6 h TP) rather than 6.4. Both figures are far below the ceiling; neither is a
  measurement of the evaluation.
- **The registered calibration step is what licenses the real projection.** The protocol
  requires measuring completed episodes/hour and decision-generations/hour separately for
  base and adapter arms, stratified by horizon and clean/faulted, on dev-only task IDs,
  before the seed reveal, on the exact production engine contract. That measurement — not
  this one — decides whether the mandatory program fits.
- **The 120.0-hour ceiling is unchanged.** This projection does not lower it, does not
  authorize shrinking any mandatory sample by one episode, and does not retire the
  protocol's pre-calibration 75–99 h evaluation envelope (which came from those older
  serial rates and remains the registered pre-calibration figure). The protocol's own
  words apply: an unvalidated speed-up "may not be cited as a measured cost, may not be
  used to justify a smaller ceiling, and may not be entered into the ledger."
- **Mandatory samples may never shrink after a result is visible.** If the mandatory work
  does not fit, the run **STOPS** and reports INCOMPLETE / INCONCLUSIVE. Optional arms are
  cut first, in the frozen order: GRPO branch (moot — not run at all), its variance probe
  and the R0/RP arms (moot — R0/RP are absent by design and the probe is
  `NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT`), then the descriptive B0/T0 arms, then the
  two-fault stress set.

## 4. If a reduced held-out audit is ever proposed instead

It is not proposed, and nothing here authorizes one. The referee's reduced-size rule is
recorded verbatim so that a future proposal meets it before, not after, the fact:

> A permissible reduction requires all of the following before `L`:
>
> 1. The exact integer counts per family, horizon, condition, control, and arm are
>    committed—not merely a maximum or range.
> 2. The rationale uses inputs independent of capability outcomes, such as an externally
>    fixed worst-case precision target or a fixed resource envelope plus measured
>    throughput. It must not use the 0.9833, 0.9867, 0.743, or 0.051 rates to solve for
>    `N`.
> 3. Selection is deterministic from a fresh derivation, balanced in fixed paired blocks,
>    and fixed before generation. A suitable rule is to rank eligible task IDs by
>    `SHA256(release_id || task_id)` within each predeclared cell and take the first exact
>    count.
> 4. BP and TP use identical selected task IDs. No result-dependent replacement or
>    reallocation is allowed.
> 5. There is no optional stopping, sample extension, or "continue if promising" clause.
> 6. The original 7,800-episode study remains incomplete regardless of the reduced result.
>
> If those conditions cannot be fixed before `L`, stop without realizing held-out data.

Additionally, per the same ruling: any such audit needs a dated `AMENDMENT` section
appended to `docs/AGENTIC_PROTOCOL.md`, committed and pushed **before `L`**, preserving all
prior text and all original gates, additive rather than rewriting the original size fields,
with a fresh held-out derivation label — and because that changes a pinned hash, the
finalization record must be refreshed through the repository's explicit amendment path
first. A narrative notice alone does not authorize a changed sample size.

## 5. The tripwire: when a deviation notice becomes mandatory

The referee's Q1 ruling in council round 8 was answered on the premise that the complete
7,800-episode evaluation **would not be executed**, and on that premise it required the
study to be reported as `DEVIATED / PARTIALLY COMPLETED / NO REGISTERED VERDICT` in a
canonical dated file. **That premise no longer holds**: the registered evaluation is being
completed, and what is dropped is the unregistered replication in §1. The classification
therefore does not apply today, which is why no `docs/DEVIATION_2026-08-06.md` exists.

It becomes mandatory the moment any of these is true:

- the registered 7,800 mandatory episodes are **not completed**, for any reason — budget,
  hardware, wall-clock, or a decision to stop;
- a **reduced** held-out evaluation is run in place of the registered one;
- any mandatory sample is **shrunk** after a result is visible;
- the run **stops** before `L` or before the evaluation finishes.

In any of those cases, the following happens and none of it is optional:

1. `docs/DEVIATION_<date>.md` is created as the canonical notice — not a commit message,
   not a mutable README block — and committed and pushed **before `L`** if `L` has not yet
   happened.
2. A prominent link and a one-sentence status go in the README.
3. The study-status paragraph in [docs/RESULTS.md](RESULTS.md) §1, already recorded
   verbatim, becomes the study status and is reproduced in the notice.
4. The unchanged registered analyzer is run over every applicable observation, and **every
   registered gate is reported with its actual machine status**. A machine `FAIL` stays
   `FAIL`; `INCONCLUSIVE` appears only when a registered condition makes it so. Saturation
   and post-hoc power analysis are not grounds for relabeling a `FAIL`. Missing or short
   strata are reported, never omitted.
5. Every estimate from a reduced run is labelled **post-registration descriptive
   evidence** and cannot establish any original primary, secondary, launch or winner
   claim.
6. The absence of a registered PASS is **not** described as "training failed", "training
   had nothing to add", "no training effect", or any equivalent.

## 6. What a reader may and may not conclude from this file

**May:** that one unregistered, optional post-hoc replication was proposed and declined;
that the registered evaluation is intended to run in full; that its cost is projected at
roughly 6.4 GPU-h from a measured single-arm rate; and that the conditions for a real
deviation notice are written down before the fact rather than after.

**May not:** that any held-out result exists (none does); that the registered evaluation
has run (it has not); that 6.4 GPU-h is a measured evaluation cost (it is a projection);
that declining the replication says anything about the trained adapter's effect (it says
nothing); or that the deep-cluster power problem the replication was meant to address has
gone away. It has not — it is a design-sensitivity diagnosis, recorded in
[docs/INTERPRETATION.md](INTERPRETATION.md) §4 and reflected in the underpowered-primary
template in [docs/RESULTS.md](RESULTS.md) §4.
