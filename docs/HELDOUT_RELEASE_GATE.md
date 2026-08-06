# The held-out release gate

`scripts/verify_heldout_release.py` is the check that has to pass before any
evaluation GPU is allocated. It exists because the chain's `eval` stage currently
proves nothing about the held-out set: it tests that two files exist.

```bash
[[ -f results/agentic/locks.json && -f results/agentic/seed_reveal.json ]] \
  || die "REFUSED: the held-out split stays blind until locks.json and \
seed_reveal.json exist (S18). Run the lock stage."
```

Two `-f` tests. A file can be written by anyone at any time; it carries no order,
no authorship and no commitment. Everything the study claims about its held-out
numbers rests on one relation:

```
P < L < R <= E
```

* **P** — the preregistration commit.
* **L** — the dedicated commit that adds the complete `results/agentic/locks.json`.
* **R** — the dedicated commit that adds `results/agentic/seed_reveal.json`
  together with `data/suite/v1/manifest.heldout.json` and
  `data/suite/v1/SHA256SUMS.heldout`.
* **E** — the commit the evaluation runs at.

The held-out master seed is `sha256(label ‖ L)`
(`agentlab.suite.generate.heldout_master_seed`), so the held-out realization is a
pure function of the lock commit's id. That is what makes "the held-out set was
fixed after the prompt winner and the checkpoint were frozen" checkable rather
than asserted.

## Ancestry, never timestamps

The gate reads no mtime and no `*_at` field. A timestamp is written by whoever can
write the file; an ancestry relation cannot be adjusted after the fact, because
changing a commit's parent changes its id — and changing L's id changes the seed,
hence the entire held-out realization. `locks.json` says this in its own note; the
gate is the part that enforces it.

Two consequences worth stating plainly:

* A lock or reveal sitting on a branch the evaluation is not running on proves
  nothing. The gate resolves L and R among commits **reachable from E**, not with
  `--all`, and says so when it finds one elsewhere.
* There is no way to commit a self-consistent reveal *before* L. The receipt's
  seed depends on L's id, and L's id depends on its parent, so a reveal at an
  earlier commit would have to predict a sha that depends on the commit
  containing the prediction. "R before L" therefore always surfaces as a receipt
  naming some other lock, or as a lock that is not an ancestor of R, or both.
  `tests/test_verify_heldout_release.py` asserts each form.

## What it refuses

1. E does not resolve, or either S18 receipt is missing from disk.
2. L is not unique among commits reachable from E, is edited after L, changes
   anything but the locks blob, already carries a reveal or a held-out
   commitment, or its committed bytes are not the bytes on disk.
3. R is not unique, does not add the manifest **and** the checksums in the same
   commit, adds anything outside the four permitted paths, or its committed bytes
   for any of the three are not the bytes on disk. The two commitment files must
   be *added at* R, not merely present there.
4. The receipt does not rederive (`load_reveal` recomputes the master seed and the
   release id), names a lock that is not L, or carries a `locks_blob_sha256` that
   is not L's committed lock.
5. Any leg of `P < L < R <= E` fails, strictly where it must be strict.
6. The manifest is unsealed, or disagrees with the receipt about the master seed,
   the release id, L, P or the receipt's own digest.
7. `SHA256SUMS.heldout` does not cover exactly the held-out phase — all eighteen
   payload files of the six held-out splits, the seven derived certspecs the
   evaluator actually reads, and the manifest itself — or it disagrees with the
   manifest, or the manifest was edited after it was sealed.
8. Any released file is missing on disk or does not hash to its committed value.
9. The retired whole-suite commitment (`manifest.json` / `SHA256SUMS`) is present.

`--require-published` additionally requires R to be an ancestor of
`refs/remotes/origin/main`. That reads the local remote-tracking ref and performs
no network I/O, the same choice `scripts/agentic_locks.py` already made: what a
clone can verify locally is the ref that follows the designated public branch.

The gate imports no torch, no vLLM, no network client and starts no CUDA context;
it is safe as the first statement of the evaluation stage, before a card is
pinned.

## Where the call goes

**Deferred.** `scripts/run_multifaceted_chain.sh` is on the forbidden list while
rejection sampling is running: the driver loop re-invokes `make agentic --only
distill`, each invocation re-imports the chain's Python modules, and editing
anything on that path would assemble one corpus out of two different programs.
The edit below is therefore recorded, not applied.

The call is **one line** in `stage_eval()`, as the last statement inside the
existing non-dry-run guard — after the two `-f` tests it strengthens, before the
adapter path is read out of `locks.json`, and before `require_gpu` / `pin_gpu` /
`gpu_session_open`:

```bash
stage_eval() {  # GPU -- the first and only stage that touches the held-out split
  say "eval: paired held-out evaluation, INTERLEAVED by immutable task block"
  if ! (( DRY_RUN )); then
    [[ -f results/agentic/locks.json && -f results/agentic/seed_reveal.json ]] \
      || die "REFUSED: the held-out split stays blind until locks.json and \
seed_reveal.json exist (S18). Run the lock stage."
    # >>> THE ONE ADDED LINE <<<
    "$PY" scripts/verify_heldout_release.py --require-published || die \
      "REFUSED: the held-out release is not provable by git ancestry \
(P < L < R <= E). No GPU is allocated. Fix the release, not the gate."
  fi
  local adapter; adapter="$("$PY" -c "
```

At the tree this note is written against, that is between line 1002 (the end of
the `die` string) and line 1003 (the closing `fi`) of
`scripts/run_multifaceted_chain.sh`. Line numbers move; the anchor that does not
is "last statement inside `stage_eval`'s `if ! (( DRY_RUN ))` block".

Two things the placement is doing:

* It runs **before** `require_gpu`, so a bad release costs nothing and never
  touches the shared card.
* It runs before the adapter path is read out of `locks.json`, so the evaluation
  cannot load a checkpoint named by a lock the gate was about to reject.

Drop `--require-published` only for an offline replay of an already published
study, where `refs/remotes/origin/main` is not available. Never pass
`--commitments-only` here: that mode skips hashing the payload and deliberately
does not print the verified-release line.

## Status

* **Evidence.** Run against this tree today the gate refuses, with
  `results/agentic/seed_reveal.json does not exist: nothing is revealed`. There
  is no R yet; `locks.json` itself is not yet committed, so there is no L either.
* **Evidence.** `tests/test_verify_heldout_release.py` is 33 tests, 31 of which
  build a whole synthetic study — its own P, its own L, its own R, its own
  held-out release — and assert the accept and every refusal above. The other two
  check that the gate imports no GPU or network stack, and that a directory which
  is not a repository is refused. One of the 31 releases is produced by the real
  `generate_phase` / `export_eval_specs.py` / `seal_phase`, so the gate cannot
  drift from the shape the generator writes.
* **Projection.** When the lock and reveal stages run, the gate is what turns
  their output into a claim. It has never been run against a real R, because none
  exists.

## The `.gitignore` half of this

R cannot commit what git ignores. The suite block in `.gitignore` used to name
`data/suite/v1/manifest.json` and `data/suite/v1/SHA256SUMS` as its exceptions —
the *retired* whole-suite commitment, which
`agentlab.suite.generate.LEGACY_COMMITMENTS` now refuses on sight — and named
neither live per-phase file. The live ones were not ignored either, because
`data/*` does not descend into `data/suite/v1/`, and the train/dev pair was
additionally safe because it is already tracked.

That is an accident, not a protection: one `data/**` or `data/suite/v1/*` written
above that block and `git add -A` would silently stop staging
`manifest.heldout.json` and `SHA256SUMS.heldout`, R would carry the receipt alone,
and the evaluated bytes would be pinned by nothing. The block is now
deny-by-default with the five commitment files named explicitly, and
`tests/test_release_gitignore.py` rebuilds R's file list in a clean checkout and
fails if either commitment file is missing from it.
