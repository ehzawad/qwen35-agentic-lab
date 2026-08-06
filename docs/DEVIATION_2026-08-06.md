# Deviation notice — 2026-08-06

This is the canonical notice the tripwire in [RESULTS.md](RESULTS.md) §4 requires. It is
published because the registered 7,800-episode held-out evaluation was **not completed**.
The tripwire names time and budget among the triggering reasons explicitly, so nothing
about this is a surprise to the protocol; the notice is the promised response.

## The study status, verbatim as the tripwire specifies

> The preregistered 7,800-episode evaluation was not completed. This study is reported as
> a deliberate post-registration deviation and partial completion. No preregistered
> study-level winner is claimed. The reduced evaluation reported below, if any, was
> prospectively fixed before checkpoint lock but is post-registration descriptive
> evidence, not a substitute for the original confirmatory evaluation.

**No reduced evaluation was run at all.** The clause above provides for one; this study
does not use it. There is no held-out evidence of any size, reduced or otherwise, because
the held-out suite was never realized — see "What does not exist" below.

## What was registered

`P` = `5844a97` hash-pinned the protocol, the machine-readable gates, the suite config and
the pipeline config. It registered: a primary claim (certified error-recovery superiority
of a trained checkpoint over the frozen prompt-only control, +0.05 margin, clustered
bootstrap lower bound, `|C| >= 500`), two secondary claims (all-tools orchestration at H4,
H8 execution reliability), a mandatory census of **7,800 BP/TP held-out episodes**, launch
floors, and a winner truth table that always permitted "the prompted base model ships" and
"no successful multifaceted pipeline yet".

## What was actually done

| stage | state |
|---|---|
| suite generation, validation, hashing | completed |
| five-probe dev preflight (87 checks) | completed, all green |
| prompt tournament, 8 hash-committed candidates, 3,600 dev episodes | completed |
| prompt winner locked under `P` | completed (`p2_plan_state_act`, `5facfd02997d`) |
| rejection sampling | **stopped at 15 of ~43 shards**: 4,400 attempts, 2,753 verified-correct trajectories |
| RS-SFT training run | **not run** |
| calibration and budget commitment | **not run** |
| `L` (checkpoint lock) | **not created** |
| `R` (seed reveal, held-out realization) | **not created** |
| held-out evaluation | **not run** |
| machine verdict | **not computed** |

## Why

Two reasons, both recorded when the decision was taken and neither of them a result.

**1. The owner set a time budget.** The remaining program was measured at ~26 GPU-hours,
dominated by rejection sampling at a measured 738 rollouts/GPU-h. The owner's stated goal
was a working, inspectable pipeline within about two hours, on the explicit reasoning that
the pipeline can be scaled later. Data collection was stopped, and the training and
held-out legs with it.

**2. Two of the three registered claims were already established as unable to pass, and
the third as under-powered — none of which depends on stopping.** Both facts were visible
from *development* data before any trained checkpoint existed:

- The selected prompt-only control measured **295/300 (0.9833)** on the development H4
  all-tools axis and **296/300 (0.9867)** on the development H8 axis. Even a perfect
  trained arm could improve those realized samples by 0.0167 and 0.0133 — both below the
  registered +0.05 margin. These are ceiling-limited endpoints.
- The registered bootstrap resamples whole structural clusters. A recomputation of the 300
  development recovery rows found 158 clusters overall but **one 26-task cluster per
  selected fulfillment horizon cell**, with nearly all failures concentrated in the H14 and
  H20 fulfillment cells. Gains concentrated there would be omitted by a replicate about
  13.5% of the time, so the lower bound could not clear +0.05 however many individual tasks
  improved. At a true lift of exactly +0.05 the power is ~2.5%: the margin sits on the null
  boundary.

Stopping therefore forfeited a confirmatory result that the instrument was already known to
be unable to deliver on two of three endpoints. That is an explanation, **not** a
justification for calling the study complete, and it is not offered as one.

## What a reader may conclude

- The pipeline runs end to end: task generation, agent execution, exact verification, fault
  injection with token-bearing recovery, prompt selection, corpus filtering, LoRA training
  (a 260 MB adapter was produced and provenance-verified by the preflight canary), serving,
  and demonstration.
- The descriptive observations in [RESULTS.md](RESULTS.md) and [USAGE.md](USAGE.md), with
  their exact numerators and denominators and their stated caveats.
- That the registered thresholds were public before any result existed, which remains true
  and is what `P` demonstrates.

## What a reader may **not** conclude

- Anything about whether training improves recovery, orchestration or long-horizon
  reliability. **That comparison was never run.** No trained checkpoint was evaluated
  against the control on any sample, held-out or otherwise.
- Any registered gate status. No gate was computed, because no held-out evidence exists.
  A gate that was never evaluated is not a FAIL, an INCONCLUSIVE, or a PASS.
- That the ceiling and power findings above are held-out results. They are development-split
  diagnostics.
- That the descriptive rates are held-out estimates, benchmark scores, or evidence about
  arbitrary real-world tools.

## What does not exist, and why that matters

`results/agentic/locks.json` records the prompt winner only; the checkpoint is unlocked and
`agentic_locks.py` still refuses `L` with *"the lock is INCOMPLETE, so there is nothing for
the held-out seed to be a function of"*. `seed_reveal.json` is absent. **No held-out bytes
were ever generated**, because the held-out generation seed is derived from the `L` commit
that does not exist. The test set was therefore never realized, never seen, and never
scored — the blindness guarantee held by construction rather than by discipline.

## Resuming

Nothing here is destructive. `make agentic` resumes: every stage decides from artifacts on
disk whether it is already done, the 15 sealed shards and their receipts are intact,
2,753 verified trajectories are banked in `data/multiface/raw`, and the remaining ~26
GPU-hours of collection would carry the study to `L`. A resumed run producing held-out
evidence would supersede this notice with a completed study; this notice would remain in
history as the record of the pause.
