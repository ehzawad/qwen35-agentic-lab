# Deferred repairs — council P0/P1 that a live producer forbade

**Dated 2026-08-06, written while rejection sampling is executing on the pinned A5000.**

This ledger exists because of one constraint, and it is worth stating plainly rather
than discovering later from a corpus that cannot be explained:

> A driver loop re-invokes `make agentic --only distill` until
> `data/multiface/accepted.jsonl` exists. **Every invocation re-imports the Python
> modules and re-reads the configuration files.** Editing anything on the distill import
> or read path would therefore change behaviour *between shards*, and the accepted
> corpus would be a corpus assembled by two different programs. That corruption is
> silent and unrecoverable: no receipt would show it, because each shard's receipt is
> internally consistent with whatever program wrote it.

So the following repairs were **identified, diagnosed, and deliberately not applied**.
Each row names the file it touches and why it waited. None of them is closed by this
commit series; this file is the promise, not the fix.

Frozen for the duration of the distill stage: `src/agentlab/multidistill.py`,
`src/agentlab/prompt_control.py`, `src/agentlab/chat.py`, `src/agentlab/provenance.py`,
`src/agentlab/suite/{datasets,runtime,verify,faults,schema,generate,rewards,kb,rng,contract,configio}.py`,
`src/agentlab/suite/envs/*`, `scripts/run_multifaceted_chain.sh`,
`configs/multifaceted.yaml`, `configs/suite_v1.toml`, `configs/agentic_preregister.json`,
`configs/frozen_prompt.json`.

Also untouched, for a different reason: `results/agentic/locks.json` and
`results/agentic/seed_reveal.json` are the S18 receipts. Each needs its own dedicated
commit, because the whole guarantee is `P < L < R <= E` **proved by git ancestry**. They
may never be swept into a mixed commit.

---

## D1 — The prompt-tournament receipt is malformed. **THIS MUST BE CORRECTED BEFORE `L`.**

**Files:** `configs/frozen_prompt.json` (the receipt) and
`src/agentlab/prompt_control.py` (the cause) — both frozen.

**What is wrong** (Evidence, all four facts independently reconfirmed from the tracked
receipt and the rollout files):

| field | says | actually |
|---|---|---|
| `round1_ranking` | `p2`, `p6`, `p8`, … | correct — this is the round-one order |
| `round2_candidates` | `p2`, **`p8`** | the round-two files that exist are `r2-p2_plan_state_act.txt.jsonl` and **`r2-p6_concise_termination.txt.jsonl`** |
| `per_candidate.p6.n` | 900 | 900 — i.e. p6 *did* run round two |
| `per_candidate.p8.n` | 300 | 300 — i.e. p8 has round-one observations **only** |
| `ranking` (final) | `p2`, **`p8`**, `p6`, … | ranks a 300-observation candidate above a 900-observation finalist |

**Cause:** finalization recomputes `top2` across *all* candidates after appending the
round-two rows, so it compares candidates with unequal sample sizes
(`src/agentlab/prompt_control.py`, the `finalize` path). p8's 300-episode `combined`
0.9300 outranks p6's 900-episode `combined` 0.9178, and the receipt then records p8 as a
round-two candidate although p8 never ran round two.

**What is NOT wrong:** the winner. p2 leads at **both** sample sizes, which is why this
is a provenance defect and not a result defect. Recomputed from the round-one rollout
files alone (300 episodes each, **Evidence**): p2 `combined` **0.9367**, p6 0.9333,
p8 0.9300 — exactly the `round1_ranking` the receipt already records. Over 900
observations: p2 **0.9367** against the actual finalist p6 at **0.9178**. The prompt in
use, `prompts/agentic/p2_plan_state_act.txt`
(`sha256 5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d`), is the
correct winner, and the lock on disk already names it. **No GPU work needs rerunning and
no result changes.** What is broken is the *ranking and finalist provenance* — a
reviewer should not have to reconcile unequal-sample candidates from raw rollout files
to discover who the finalists were.

**Why it waited:** `configs/frozen_prompt.json` is read on **every** distill invocation.
`multidistill.load_frozen_prompt()` calls `prompt_control.frozen_winner()`, which opens
the file and re-hashes the winning prompt before each shard. Rewriting the file
mid-stage means shard *n* and shard *n+1* read different bytes for the object that
defines the BP/TP system prompt. Even though the corrected file would carry the same
winner and the same prompt digest, "the receipt changed while the corpus was being
produced" is exactly the class of edit that cannot be audited after the fact.

**The required repair, once distill closes:** record `p2`/`p6` as the actual finalists,
restrict the final selection to them, preserve the round-one ranking separately from the
final ranking, add a regression test that a candidate with fewer than the round-two
sample size can never enter `round2_candidates` or outrank a finalist, and commit it as
a dated correction. **It must land before `L`.** `L` is the dedicated commit that
publishes the complete lock, the held-out seeds are a function of `L`, and a reviewer
reading backwards from the verdict must find a coherent tournament receipt at or before
that point — not a correction dated after it.

---

## D2 — The chain never calls the R gate it ships

**File:** `scripts/run_multifaceted_chain.sh` (frozen).

The `eval` stage's entire held-out precondition is a two-file existence test
(`run_multifaceted_chain.sh:1000`):

```sh
[[ -f results/agentic/locks.json && -f results/agentic/seed_reveal.json ]] \
  || die "... (S18). Run the lock stage."
```

Two files on disk are not a valid `R`. `scripts/verify_heldout_release.py` exists,
proves the real property — `P < L < R <= E` by git ancestry, the reveal rederiving from
the lock commit it names, the held-out manifest and checksums committed in that one
reveal commit, every released byte hashing to its committed value — and the chain does
not invoke it. Consequences: evaluation can allocate the card before any valid `R`
exists, and `R` creation (generate, export, seal, validate, commit the held-out release)
is not an explicit, tested stage at all — the `lock` stage reveals the seed and stops.

**Required repair:** call `verify-heldout-release` before any eval GPU startup and hard
refuse on failure; make `R` an explicit stage with its own validator; test the mandatory
pause at `L` and the resume through `R`; assert held-out bytes cannot exist before `L`
and are validated before eval.

**Why it waited:** the chain script is the program the running driver invokes. Editing it
between invocations changes the running stage's own orchestration.

---

## D3 — Stage completion accepts a path where a validated receipt is required

**File:** `scripts/run_multifaceted_chain.sh`, `stage_receipt()` at line 250 (frozen).

The chain's own header promises completion is decided "from a validated RECEIPT on disk
-- not from" a path. Three branches do not keep that promise:

| stage | check today | what it should run |
|---|---|---|
| `distill` | `[[ -s "$ACCEPTED_RECEIPT" ]]` — the receipt file is non-empty | `require_accepted_corpus`: quotas, per-cell counts, corpus digest, gap validation |
| `views` | `[[ -s "$VIEWS_REPORT" ]]` — the report exists | the view-chain validator: view rows against the accepted corpus they claim |
| `sft` | `[[ -s "$RSSFT/adapter_config.json" ]]` | the training-manifest / tree-digest validator |

The `sft` case is the dangerous one. If training writes root adapter weights and
`adapter_config.json` and then dies before publishing its training manifest, the chain
will skip `sft` **forever** as "complete", and the failure will surface much later as a
checkpoint-lock failure at `L`, where it is expensive and confusing. A non-empty receipt
file is also not a *valid* receipt: a torn or stale JSON passes `-s`.

**Required repair:** run the real validator on every skip/resume decision, for all three
stages. File existence must never be a resume or completion criterion.

**Why it waited:** same file as D2, invoked by the running driver.

---

## D4 — The driver has no stage→validator allowlist, and the running driver is not the committed one

**Files:** `scripts/run_stage.sh` (the committed driver) and the ignored
`out/driver/run_stage.sh` (the program actually executing).

**First, what is already fixed, so this entry is not read as worse than it is.** The
council reviewed the tree at `1f72685` (08:39). Commit `e3f9beb` (09:06, "Make the driver
behave like its own comments") landed three minutes after the report was written and
closed most of that P1 row: the committed driver now runs `set -euo pipefail`, **refuses
outright** if `results/agentic/locks.json` or `seed_reveal.json` exists untracked
(exit 3), stages an **explicit path list instead of `git add -A`**, and verifies
`git ls-remote origin refs/heads/main` equals local `HEAD` after pushing (exit 4 if not).
Those specific defects are not deferred; they are done.

**(a) What remains open in the committed driver.** Five things, and the first is the one
that matters:

1. **No stage→validator map.** It still takes an arbitrary `<completion-artifact>` path
   as its second argument and decides completion with `[ -e "$done_artifact" ]`
   (`run_stage.sh:60`, `:75`). A partial or torn file terminates a stage as "COMPLETE".
   The driver should accept a **stage name only** and own a fixed mapping from
   stage → native receipt validator → produced outputs → allowed git paths.
2. **One global path list, not per-stage allowlists.** `STAGE_PATHS` is the same nine
   entries for every stage, so a `distill` boundary would happily commit a changed
   `configs/frozen_prompt.json` or a preflight file if one were dirty. The staged path
   list should have to match *that stage's* allowlist exactly.
3. **No quiescence precondition.** It commits `gpu_ledger.jsonl` and `gpu_sessions.jsonl`
   without checking that no study or owned GPU process is still live, that every opened
   GPU session has a close event, or that the ledger and journal parse completely.
4. **No `PARTIAL_BUDGET` outcome.** The loop reports `COMPLETE` on artifact existence and
   exit 2 on running out of invocations; a cleanly-expired budget window with drained
   work and preserved progress has no distinct status, so a partial commit can only be
   described as a complete one.
5. **The push target is hardcoded** to `origin main` (`:119`) rather than the checked
   current upstream, so on any other branch it pushes the wrong ref — and the
   verification then compares `HEAD` against `origin/main`.

**Plus one defect the report did not name, found while checking the above:** the failure
diagnostic at `run_stage.sh:66-72` is **dead code**. Under `set -euo pipefail` a nonzero
`make agentic` exits the shell at line 65, so `rc=$?` is never reached and the
"Last 30 log lines" tail never prints. Verified directly: `set -euo pipefail; false;
rc=$?; echo reached` prints nothing and exits 1. The exit code still propagates, so this
costs diagnostics rather than correctness — but on an unattended overnight run,
diagnostics are the whole point of that branch.

**Required repair:** all five above, the dead branch, and
`tests/test_stage_driver.py` covering partial files, unrelated dirty files, `L`/`R`
exclusions, branch handling and writer-active refusal.

**(b) The executing program is the old ignored copy.** The distill loop is running
`bash out/driver/run_stage.sh distill data/multiface/accepted.jsonl 40` — 18 lines,
`sha256 8c7aeb5dc60cde7e27af99de7222223bf89577fcf1537b222229e30e2bd357d3`. The committed
driver is 130 lines,
`sha256 817b925603dc1bcc1cdf579359eb8ce4b93d70a370cc7d2e64c30fce40f4ec26`. The two
digests are recorded here so the invocation history of this stage stays reconstructible
from the pushed tree; the repaired driver may only take over **after** the current stage
closes, so that no single shard sequence has two plausible drivers.

This is not hypothetical. The distill producer opened its session at
`2026-08-06T11:45:04Z`, and `e3f9beb` rewrote the committed driver at 09:06 local — i.e.
**the committed driver already changed while this stage was running.** Treatment was not
affected, because the executing program is the `out/` copy and the chain script's bytes
did not move; but a reviewer reading only the pushed tree would reasonably assume the
130-line driver drove these shards, and it did not. Hence the two digests above.

**Why it waited:** repairing (a) now would put a *third* driver revision in the tree
while a driver loop is mid-stage. The repair is inert for the running loop and actively
misleading for the record, so it waits for the boundary — and when it lands, the stage
receipt should carry the executing driver's digest rather than leaving that fact in a
markdown file.

---

## D5 — One test consults the live repository instead of an isolated git history

**File:** `tests/test_chain.py:431`, `test_lock_prompt_refuses_before_P_exists`. This is
the single failing test in the current suite (**998 passed, 1 failed, 6 skipped**).

The test asserts that `lock-prompt` refuses *because no preregistration commit exists*:

```python
assert "no commit adds configs/preregistration_final.json" in (r.stdout + r.stderr)
```

It establishes that precondition by invoking the CLI against **this** repository rather
than against an isolated temporary git history. Now that `P` = `5844a97` is a real commit
on `main`, that message is unreachable, and the observed refusal is a *different*
refusal:

```
REFUSED: prompt sha 000000000000 is not one of the eight preregistered candidates (S16)
```

So `lock-prompt` still refuses the bogus lock — with a nonzero exit, on the S16 candidate
check — and the assertion about *which* refusal fires is what broke. The earlier
all-green receipt remains historically true: it was recorded pre-`P`, when the message
the test expects was the one that fired. The refusal machinery itself is demonstrably
intact; `scripts/agentic_locks.py status` reports
`L (locks commit) -- REFUSED: the lock is INCOMPLETE, so there is nothing for the
held-out seed to be a function of`, which is the same refusal family behaving correctly
on the live tree. **What is broken is the test's isolation, not the lock.**

Consequence worth stating: **current `main` is not green from a fresh clone**, and it
will stay that way until this is fixed. That is a real defect in the published tree even
though no gate depends on it.

**Required repair:** give the P/L/R tests their own temporary `git init` fixture so they
assert against a constructed history, and keep the live repository out of the assertion
entirely — then assert *both* refusal paths explicitly (no-`P` and non-candidate-sha)
instead of one message that changes meaning as the study advances.

**Why it waited:** the existing test files are outside the narrow edit allowlist that a
live producer permits. The fix is a test-only change and safe in itself, but it belongs
in the same pass as D4's `tests/test_stage_driver.py` so that the driver and lock tests
are isolated by one coherent fixture rather than two.

---

## Also open, from the same review round — smaller, same reason

These are not gate-bearing, and each is blocked by one of the frozen files above.

"Frozen" below means the file is on the running stage's import or read path, so an
edit changes behaviour between shards. "Allowlist" means the edit is safe in itself but
sits outside the narrow set of files this pass was permitted to touch while a producer
writes, and it belongs in a coherent later pass rather than a lone drive-by.

| item | file | why it waited |
|---|---|---|
| The prompt-tournament unit check uses file existence, with no receipt binding expected axis IDs and counts | `scripts/run_multifaceted_chain.sh` | frozen (D2/D3) |
| The variance probe's future RUN path uses cell-file existence | `scripts/run_multifaceted_chain.sh` | frozen |
| `verdict` always reruns; `ship` has no completion artifact at all | `scripts/run_multifaceted_chain.sh` | frozen |
| Evaluation has no final digest receipt binding the completed trace set (torn lines are skipped and missing tasks rerun, but nothing durable seals the set) | `src/agentlab/suite/evaluate.py` + chain | allowlist for `evaluate.py` (it is *not* on the distill import path); frozen for the chain half |
| The registered **calibration** stage and its producer do not exist, and `results/agentic/budget_commitment.json` is absent — yet calibration is the only mechanism allowed to cut optional arms, and its projection is load-bearing for the 120 h envelope | chain + `src/agentlab/suite/configio.py` (the reader exists, the writer does not) | frozen — `configio` is imported on every distill invocation. Must land before `eval`; it is not an `L` blocker |
| The lock-selection receipt has no writer | `scripts/agentic_locks.py` + chain | allowlist, and additionally: the lock path may not change while an incomplete lock sits on disk waiting to become `L` |
| `ledger_append` does not `fsync`, so a torn tail makes the next reader fail; the heartbeat helper can outlive its owner and overcharge idle time; `require_gpu` runs before ledger reconciliation; SFT has no budget guard | `src/agentlab/env.py` (imported by `multidistill`) + chain | frozen, and the ledger is being appended to by the live producer as this is written |
| SFT has one epoch, no `resume_from_checkpoint`, and no attempt-directory publication: a mid-epoch crash restarts from zero, and a crash between `save_model` and manifest creation can poison resume | `src/agentlab/sft.py` | allowlist; must land before the `sft` stage, and not while distill is mid-flight |

---

## What this ledger commits me to

1. Nothing above is fixed by pretending it is fixed. The README status block and
   `docs/INTERPRETATION.md` state the same facts this file does.
2. **D1 lands before `L`.** That is the one ordering constraint in this file that a
   reviewer can check by git ancestry, and it is the one I am most likely to be tempted
   to skip, because the winner is unaffected.
3. `L` and `R` remain dedicated single-purpose commits. No repair from this list is ever
   bundled into either of them.
4. Apparatus repairs are committed separately from experimental output, with the stage
   they follow named in the message.
