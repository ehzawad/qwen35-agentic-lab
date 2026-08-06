# Interpretation rules for the saturated secondary endpoints

**Status: apparatus/reporting record, dated 2026-08-06. Additive, written after the
preregistration was finalized at `P` = `5844a97d2096fb55186e8559f4a4481dc3b75e9d`, and
BEFORE any trained checkpoint, any held-out realization or any capability result
existed.** It amends nothing. `docs/AGENTIC_PROTOCOL.md` is hash-pinned by
`configs/preregistration_final.json` and is not edited here; no threshold, margin,
sample size, estimand, stratum, gate or launch floor changes. What this file fixes is
**my prose**, which had begun to describe the registered endpoints more favourably to
me than the machine will.

The prose being corrected is the pushed commit message of
`4c840e3b230cbaad70b94e3a6ad66b852ba8a12c` ("Freeze the prompt winner:
p2_plan_state_act, and record that both secondaries are saturated"), which asserted
that "the preregistered consequence is INCONCLUSIVE by saturation", that "a prompted 4B
base is already at ceiling here, so training had nothing to add", and that "0.840
leaves 0.16, so the registered +0.05 is reachable". **History is not rewritten**: that
commit stays exactly as pushed, and this file is the forward correction. Where the two
disagree, this file governs the write-up.

Referee ruling on the record: **proceed as registered.** Do not replace the H4/H8
secondaries with harder confirmatory strata in this run. Keep the original gates and
their original machine statuses, and interpret a non-PASS as *failure to establish
>5-point superiority on a ceiling-limited endpoint* — not as a tested-and-failed
training effect.

Every number below is labelled **Evidence** (measured, reproducible from committed or
receipted bytes), **Projection** (arithmetic from measured inputs) or **Estimate**
(assumption-bearing). Nothing here is a held-out result: none exists.

---

## 1. Three corrections to my framing

### 1.1 "Both secondaries will be INCONCLUSIVE by saturation" — wrong

I wrote that. It is not what the frozen analyzer does, and it flatters the outcome.

The analyzer decides on the **computable bound**, and reserves `INCONCLUSIVE` for
**registered non-computability conditions only** ([`_bootstrap_verdict`](../src/agentlab/analyze.py),
`_bootstrap_gate` reasons list):

| gate | INCONCLUSIVE only when | otherwise |
|---|---|---|
| `MT1` (all-tools H4 orchestration diff) | fewer than the registered **600** all-tools H4 clean pairs; or a registered non-computability reason fires — pairs missing `template_cluster_id`, fewer structural clusters than the registered minimum, a cluster larger than the registered maximum, or a degenerate resampling distribution | **PASS** if the one-sided 97.5% clustered lower bound **>** `+0.05`; **FAIL** if it is not |
| `HR1` (H8 clean certified-success diff) | fewer than the registered **400** H8 clean pairs; **or fewer than 20 discordant pairs**; or any of the same non-computability reasons | **PASS** if the lower bound **>** `+0.05`; **FAIL** if it is not |

So the honest prediction is: **`MT1` normally becomes FAIL** when its lower bound does
not exceed `+0.05`. `HR1` becomes `INCONCLUSIVE` **only if a registered condition such
as the fewer-than-20-discordant-pairs rule actually applies**; otherwise it too
becomes **FAIL**. `HR1`'s discordance condition is the more likely of the two escapes
precisely because the endpoint is saturated — near-ceiling paired arms produce few
discordant pairs — but that is a prediction, not a status.

**Rule: report the machine statuses the analyzer actually produces.** Do not convert a
machine `FAIL` into `INCONCLUSIVE` in prose, and do not describe a `FAIL` as
saturation without also saying `FAIL`.

### 1.2 "Training had nothing to add" — wrong

The development data establish something narrower: **the binary development endpoints
had almost no observable headroom.** They do **not** establish that training has zero
effect, and they do **not** establish held-out saturation.

- The dev endpoints are *binary* per episode. A training effect on trajectory quality,
  call economy, recovery behaviour or deep-horizon completion can be real and invisible
  to a saturated binary rate.
- Held-out is a *different realization*. A held-out `BP` below 0.95 could still leave
  point-estimate headroom above the `+0.05` margin. The development evidence makes that
  unlikely — it does not make it impossible.
- The prompt winner was **selected on** these dev axes, so its dev rates are optimistic
  by selection; the intervals below are informal for that reason.

### 1.3 "Cannot pass by construction" — true only of the realized dev samples

The arithmetic is right about the samples I have, and wrong about the samples I do not:

| axis | p2 dev result (**Evidence**) | max gain a perfect trained arm could show on *that realized sample* (**Projection**) | registered margin |
|---|---|---|---|
| H4 all-tools orchestration | 295/300 = 0.9833 | 5/300 = **0.0167** | `> +0.05` |
| H8 execution | 296/300 = 0.9867 | 4/300 = **0.0133** | `> +0.05` |

Ordinary unadjusted Wilson 95% intervals are approximately 0.962–0.993 and
0.966–0.995 (**Estimate**; informal because the candidate was selected on these data).
All eight candidates scored highly, which supports a genuine task ceiling rather than
one lucky prompt.

"Cannot pass by construction" is therefore **arithmetically true of the realized
development samples only**. It is **not logically true of unseen held-out data.**

### 1.4 Where the legitimate/illegitimate line sits

- Discovering and documenting the ceiling **on development data** is legitimate
  calibration — the control was explicitly selected on dev, and reporting the selected
  control's absolute level before held-out realization is proper.
- **Replacing the endpoint after seeing those numbers is an outcome-informed design
  change** — the start of a moved goalpost. Not done. Not doing it.
- A dated, additive amendment with independent held-out derivation could still be
  scientifically usable, because no trained-arm or held-out result has been seen — but
  it would be a *post-development, prospectively tested extension*, not the equal of
  the original preregistration, and it would have to preserve the original gates and
  expand multiplicity control.

---

## 2. Required results wording (VERBATIM — must appear before the held-out secondary results)

The following wording, or wording equally explicit, **must** appear before the held-out
secondary results:

> Before any trained checkpoint was evaluated and before the held-out suite was
> realized, the selected prompt-only control achieved 295/300 (0.9833) certified
> successes on the development H4 all-tools orchestration axis and 296/300 (0.9867) on
> the development H8 execution axis. On those realized development samples, even
> perfect trained-arm performance could improve the respective rates by only 0.0167 and
> 0.0133, both below the preregistered +0.05 superiority margin. These development
> results diagnose ceiling limitation of the endpoints; they are not held-out estimates
> and do not determine the registered gate statuses.

Then report **each** held-out gate as follows (VERBATIM template; bracketed slots are
filled from the analyzer's emitted numbers, never from prose):

> We nevertheless evaluated the unchanged preregistered contrast. BP achieved [kBP/n],
> TP achieved [kTP/n], the paired difference was [Δ], and the one-sided 97.5%
> structural-cluster bootstrap lower bound was [LB] over [G] clusters, with [d]
> discordant pairs. The preregistered gate status was [PASS/FAIL/INCONCLUSIVE]. A
> non-PASS means that improvement greater than five percentage points was not
> established on this endpoint. It must not be interpreted as evidence that the
> training intervention had zero effect.

**Superseded-in-part, forward, no deletion.** Council round 8 restated that template with
two additions: it also requires the registered reason to be named (`for the following
registered reason: [reason]`) and requires the sentence "the registered >0.05 superiority
claim failed" whenever the status is FAIL. That stricter version is recorded verbatim in
[docs/RESULTS.md](RESULTS.md) §3 and is the one to use. The block above is kept exactly as
first fixed — neither version softens the other, and rule 1 below already demanded the
FAIL sentence.

If held-out `BP` is itself at least 0.95, add:

> At the observed held-out BP rate, the largest arithmetically possible sample
> improvement was [value], below the +0.05 margin. The endpoint was therefore
> non-discriminating for the registered superiority claim on this realization.

Binding reporting rules that go with the template:

1. If the status is `FAIL`, say **"the registered >0.05 superiority claim failed."**
2. If `HR1` is `INCONCLUSIVE` because it has fewer than 20 discordant pairs, **name that
   exact registered reason** — and likewise for any other registered condition that
   fires (pair floor, cluster minimum, cluster maximum, missing cluster id, degenerate
   resampling).
3. Never convert a machine `FAIL` to `INCONCLUSIVE` in prose.

Phrases that must not appear:

- "Training failed."
- "Training had nothing to add."
- "No training effect."
- "Both claims were inconclusive by saturation" — unless the registered machine
  conditions actually produce `INCONCLUSIVE`.

---

## 3. What should carry the interpretation instead

**The prespecified horizon curves, descriptively.** They already cover H12/H14/H20.
Report the raw numerator, denominator and Wilson band at **every** family/horizon
point, and show where score headroom begins and whether BP/TP separation appears at
H12, H14 or H20. Any separation is **descriptive**: a selected deep cell may never be
promoted into a replacement confirmatory result, and the protocol already prohibits
both extrapolation and promotion.

**The four-arm layout as a descriptive 2x2 decomposition:**

| contrast | reads as |
|---|---|
| `BP − B0` | prompt elicitation gain for base weights |
| `T0 − B0` | training gain under the neutral prompt |
| `TP − BP` | training gain once both arms receive the selected prompt |
| `(TP − T0) − (BP − B0)` | descriptive prompt-by-training interaction / redundancy |

If `T0 > B0` while `TP ≈ BP`, the defensible reading is that **training changes
behaviour under neutral elicitation but overlaps with what the selected prompt already
elicits.** That is not "training did nothing". If separation appears only at deeper
horizons, say exactly that, and keep it descriptive.

---

## 4. Why harder strata would be a bad trade: the suite has a floor too

The tempting amendment — deeper horizons, more tools, tighter budgets — changes several
construct dimensions at once and risks **exchanging a ceiling for a floor**. The
development recovery axis already shows the floor.

p2 (`p2_plan_state_act`), recovery axis, 300 dev episodes, recomputed from the
tournament rollouts (**Evidence**;
`out/multiface/prompt_tournament/r1-p2_plan_state_act.txt.jsonl` +
`r2-…`, receipted in the artifact index):

| cell | certified recovery |
|---|---|
| `fulfillment-h14` | **1/26** |
| `fulfillment-h20` | **7/26** |
| the other ten cells, summed | **244/248** |
| whole axis | 252/300 = 0.840 |

Two cells at 1/26 and 7/26 next to ten cells at 244/248 is a floor, not a gradient. A
"harder strata" amendment would move the confirmatory endpoint from a region where the
prompt-only control is at 98% into a region where it is at 4–27%, and near-zero base
rates are their own measurement problem: the contrast becomes dominated by whether
either arm can start the task at all. **Swapping a ceiling for a floor is not an
improvement.** The cleaner paper is the one that admits the secondary endpoints were
miscalibrated and reports the machine statuses.

There is a second, statistical reason the deep cells cannot quietly become the claim:
each selected fulfillment horizon cell is **one 26-task structural cluster**, and the
registered bootstrap resamples whole structural clusters. A gain concentrated in one or
two clusters can be numerically large and still leave a 2.5th percentile at zero
(**Estimate**: with all positive differences in two clusters, a replicate omits both
with probability about `e^-2` = 13.5%).

---

## 5. Related correction: 0.840 is a planning proxy, not the primary control estimand

I have used "BP recovery is 0.840, so there is 0.16 of headroom". That statement is
retired.

The tournament recovery axis scores certified strict success on **fault-assigned
episodes** (**Evidence**: 252/300 above). The primary estimand is conditioned on the
**common-clean set `C`** — tasks where *both* `BP` and `TP` succeed cleanly before
faulted recovery is analyzed. The tournament never constructs `C`. Conditioning on `C`
will probably drop some of the hardest tasks and may push `BP` recovery **above**
0.840, **reducing** primary headroom (**Projection**). The `+0.05` margin remains
feasible at a hypothetical `BP`-on-`C` rate of 0.840 — the arithmetic ceiling is
`+0.160` — but that is coherence, not power, and `|C| ≥ 500` is a feasibility floor
rather than a power guarantee: power is driven by discordant pairs and by the number
and balance of effect-bearing structural clusters.

The composition of `C` must be reported alongside the primary: both clean-successful,
`BP`-only, `TP`-only, neither — by family, horizon and structural cluster.

---

## 6. Cross-references

- Machine semantics quoted in §1.1: `src/agentlab/analyze.py` (`_bootstrap_gate`,
  `_bootstrap_verdict`, the `MT1` and `HR1` branches).
- Registered gate definitions: `docs/AGENTIC_PROTOCOL.md`, "MT (secondary a)" and
  "HR (secondary b)" — hash-pinned, unedited.
- Dev rates and the tournament receipt: `configs/frozen_prompt.json`. **That receipt
  has a known malformation that must be corrected before `L`** — see
  `docs/DEFERRED_REPAIRS.md`.
- Study history and the round-1 elicitation lesson that motivated the frozen-prompt
  control: `docs/EXPERIMENT_HISTORY.md`.
