# Surviving host or disk loss

Council round 6 rated two things P0 that this document answers (Auditor,
"Outright blockers" 6 and 7; Architect, section 1):

> Expensive raw shards, accepted corpus, SFT views, adapters, and manifests
> under `out/` or `data/` are ignored. Git commits do not protect most GPU work.
> Process crashes are handled, but host/disk loss is not.

> The shared run secret is generated with `os.urandom(32)` and stored only under
> ignored `out/`. A disk loss makes the run impossible to resume or
> authenticate. A fresh clone gets different recovery tokens and receipts.

Both are Evidence, not projection: `.gitignore` excludes `out/` and `data/*`,
and `src/agentlab/suite/contract.py` writes the secret to
`out/agentic/run_secret.hex` and nowhere else.

## What git protects, and what it does not

`git push` protects the code, the configuration, the preregistration, the suite
seeds and manifests, the probe verdicts, the GPU ledger and the receipts under
`results/`. It protects none of the following, all of which cost GPU hours:

| Artifact | Where it lives | Ignored by |
|---|---|---|
| preflight evidence tree, canary LoRA | `out/preflight/**` | `out/` |
| prompt-tournament rollouts | `out/multiface/prompt_tournament/**` | `out/` |
| token census (original) | `out/agentic/token_census.json` | `out/` |
| rejection-sampling shards | `data/multiface/raw/**` | `data/*` |
| accepted corpus | `data/multiface/accepted.jsonl` | `data/*` |
| SFT views | `data/multiface/sft_views.jsonl` | `data/*` |
| **the locked adapter** | `out/multiface/rssft-lora/**` | `out/` |
| the run secret | `out/agentic/run_secret.hex` | `out/` |

(Those are the chain's own `ACCEPTED`, `VIEWS` and `RSSFT` variables. A test
asserts the group table matches them, because writing them down from memory is
how this file got two of them wrong the first time.)

Atomic writes plus sealed shard receipts already make a *process* crash
survivable. They do nothing about the loss of this disk.

## The remote

    hf://datasets/ehzawad/qwen35-agentic-lab-artifacts-agentic-v1

A **private** Hugging Face dataset repository, one per run. `ARTIFACTS.json` at
the repository root is the committed index: per file, the repo-relative source
path, the `hf://` URI, a second URI pinned to the dataset-repo commit that added
the bytes, the byte size, the SHA-256, the producer receipt fields the artifact
itself carries (`session_id`, `runtime_manifest_sha256`, `gpu_uuid`), and the
run scope the artifact declares. A mutable URL is not a durable reference, which
is why every entry also carries `remote_uri_pinned`.

    scripts/hf_artifacts.py plan            # what would go, no network
    scripts/hf_artifacts.py upload          # upload + index what exists now
    scripts/hf_artifacts.py verify          # re-digest what the index claims

The tool is deliberately not on any stage's import path and imports no
`agentlab` module.

### Running it while a producer is writing

A digest of a file that is still being appended to is a lie, and an index entry
built from one is worse than no entry: it points at bytes that have moved. So
every candidate is digested four times, at three checkpoints:

1. **twice before upload** — a prescan that stops an obviously live file from
   being transferred at all (`digest_moved_prescan`);
2. **again immediately after its own upload** (`digest_moved`);
3. **once more when the whole run ends** (`digest_moved_during_run`).

Checkpoint 3 exists because 1 and 2 only cover a file's own upload window. The
GPU session journal heartbeats about every 30 s and sailed straight through that
window on the first real run — `verify` showed its recorded digest was already
stale nine seconds later. So the end-of-run recheck also holds the observation
window open to at least `--settle-seconds` (default 45), because a window
shorter than the slowest appender's period is not evidence of quiescence, and
the difference between "did not move" and "was not watched long enough to move"
is the whole claim. Records written by *earlier* runs are left alone: they
correctly describe the bytes at an earlier stage boundary.

A file that fails any of those checks is **not recorded**. It goes to `skipped`
with both digests and the reason, and the next run picks it up. The GPU session
journal is therefore expected to sit in `skipped` for as long as a producer
holds the card.

Consequence: `hf_artifacts.py upload` is safe to run at any time, including
while rejection sampling holds the A5000. It never starts a GPU process, never
writes into an adapter tree (whose digest is a lock input), and never edits
anything on a running stage's import path.

### Stage boundaries

Groups that are not yet complete are declared but **not** swept by a bare
`upload`. `scripts/hf_artifacts.py plan --all-groups` prints the whole group
table with the run scope and boundary each one names. Run them at that boundary:

    scripts/hf_artifacts.py upload --group rs_raw --group accepted_corpus   # after distill
    scripts/hf_artifacts.py upload --group sft_views                        # after views
    scripts/hf_artifacts.py upload --group adapter                          # after SFT, before L
    scripts/hf_artifacts.py upload --group grpo_adapter                     # if that leg is admitted
    scripts/hf_artifacts.py upload --group traces                           # after eval
    scripts/hf_artifacts.py upload --group ship_smoke                       # after ship

The `adapter` group is the one that matters most: the `L` commit's checkpoint
hash is taken over those bytes, and a reviewer without this disk cannot
re-derive them. Its patterns — and every other group's — are pinned by a test
that parses the chain's own path variables, because the first version of the
table guessed three of them wrong and would have protected nothing while
reporting success.

A re-run at the same boundary transfers nothing and leaves the `files` array
byte-identical, so the file is evidence rather than churn. The `skipped` array
does move while a producer is live, which is the point of it.

### What it refuses, always

- **The held-out release.** Refused on path shape (`heldout`, `held_out`,
  `held-out`), not on a file list, so a future held-out artifact nobody thought
  to enumerate is refused too. It does not exist before `R` and must never be
  published by a bulk sweep.
- **`results/agentic/locks.json` and `results/agentic/seed_reveal.json`.** The
  S18 receipts. Each needs its own dedicated commit and its own deliberate
  publication step; `P < L < R` is proved by git ancestry and a bulk artifact
  sweep would destroy the point of it.
- **The plaintext run secret.** See below.

## The run secret: commitment now, reveal after the verdict

The secret is 32 bytes from `os.urandom`, created once per run, and every
recovery token the model sees and every receipt token is derived from it. Lose
it and the receipts are unverifiable and the run unresumable. Silently
substitute a new one and the model-visible environment has changed without a
record.

The ordering is fixed and each step is checkable by someone who does not trust
us:

1. **Commitment, before `L`.** `results/agentic/run_secret_commitment.json`
   records `sha256(secret)`, `sha256(secret.hex file)` and a run-bound
   commitment `sha256(domain ‖ 0 ‖ run_id ‖ 0 ‖ secret)` — **never the secret**.
   It is written once and the tool refuses to overwrite it with a different
   digest. Its git ancestry is the proof that the digest was fixed before any
   result existed.

       scripts/hf_artifacts.py commit-secret

2. **Encrypted off-host backup, now.** `out/artifacts/run_secret.enc.json` is
   an AES-256-GCM envelope (scrypt n=2^15, r=8, p=1; AAD binds the run id) and
   is uploaded to the private repo like any other artifact and indexed in
   `ARTIFACTS.json`. The envelope round-trips from disk before the tool claims a
   backup exists.

       scripts/hf_artifacts.py backup-secret
       scripts/hf_artifacts.py upload --group run_secret_backup

   The passphrase comes from `--passphrase-file`, from
   `$AGENTLAB_ARTIFACT_PASSPHRASE`, or is generated and written mode-0600 to
   `~/.config/agentlab/<run_id>.artifact-passphrase`. **A passphrase that lives
   only on this disk does not survive the disk loss the backup exists to
   survive** — it has to be copied to a password manager or another host. The
   tool says so loudly; it cannot do it for you.

3. **Reveal, after the verdict.** The plaintext secret is published as
   `results/agentic/run_secret_reveal.json` once evaluation is complete, and
   anyone recomputes the three digests in step 1 to confirm it is the secret the
   pre-result commitment names. This is **not** the S18 held-out seed reveal
   (`results/agentic/seed_reveal.json`), which is a different receipt with its
   own dedicated commit.

Publishing the secret before evaluation would let the model's recovery tokens be
anticipated; committing only after evaluation would let the secret be chosen to
fit the result. The commitment/reveal split is what removes both.

## The recovery drill

A backup nobody has restored is a hope. This was run end to end on
2026-08-06, while rejection sampling held the A5000:

```python
# 1. pull a file from the commit the index pins, not from the branch tip
from huggingface_hub import hf_hub_download
import hashlib, json
idx = json.load(open("ARTIFACTS.json"))
f = [x for x in idx["files"] if x["path"].endswith("run_secret.enc.json")][0]
p = hf_hub_download(repo_id=idx["remote"]["repo_id"], repo_type="dataset",
                    filename=f["remote_path"], revision=f["remote_commit"])
assert hashlib.sha256(open(p, "rb").read()).hexdigest() == f["sha256"]
```

Results:

- four files pulled from their pinned commits (the envelope, a tournament
  rollout, a preflight trace, the hardware lock) all hash to the digests the
  index records;
- the Hub's own server-side LFS `sha256` matches the index for every large
  object, including the 519 MB optimizer state — independent server-side
  confirmation, not our own digest read back;
- the run secret was recovered from the remote envelope and reproduces both
  `secret_sha256` and the run-bound `commitment.digest` in the committed
  commitment, and is byte-identical to the live secret.

So the claim "this run survives loss of this host" is tested, not asserted —
with the one stated exception that the *passphrase* still has to be copied
off-host by hand.

## What this does not fix

Deliberately out of scope here, and open in council round 6:

- The base model is still named `Qwen/Qwen3.5-4B` with no Hub revision or weight
  digest (Architect P0). Pinning it touches files a running stage imports.
- `results/agentic/hardware.json` declares `run_id: dev-preflight-v1` while the
  live run is `agentic-v1`; the index records what the file says rather than
  papering over it.
- The prompt-tournament rollout rows carry no provenance block at all, so their
  producing GPU session cannot be established from the files themselves. The
  index reports `"source": "unresolved"` rather than guessing from mtimes,
  because attributing an artifact to whichever session happened to be open would
  be an inference dressed up as a receipt.
- The environment is not hash-locked (`uv.lock`, pinned Python patch), so these
  bytes are reproducible-by-reference, not reproducible-by-rebuild.
