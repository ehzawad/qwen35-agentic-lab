#!/usr/bin/env python
"""The P -> L -> R state machine. CPU only, no GPU, no network.

S18 POST-LOCK HELDOUT is a relation between three commits, proved by GIT
ANCESTRY AND ARTIFACT HASHES and never by timestamps:

    P   the unique commit adding configs/preregistration_final.json
        (the finalization marker: it hash-pins the COMPLETED preregistration and
        the train/dev commitment)
          |  prompt selection and training, on train/dev only
    L   a unique DEDICATED commit adding the complete results/agentic/locks.json
        (prompt winner + checkpoint byte digest + the trainer's receipt digest);
        it changes nothing else, carries no reveal, and is published
          |  the held-out seed becomes derivable HERE, and not before
    R   a dedicated commit adding results/agentic/seed_reveal.json together with
        data/suite/v1/manifest.heldout.json and SHA256SUMS.heldout
          |
    E   evaluation traces whose git_sha is R or a descendant

    P < L < R <= E

THE SEED. The six held-out generation seeds are `heldout-master-v2` over L:

    master        = SHA256("qwen35-agentic-lab\\0agentlab-suite-v1\\0"
                           "heldout-master-v2\\0" || bytes.fromhex(L))
    split_seed(s) = int(SHA256("agentlab-heldout-split-v2\\0" || master
                              || "\\0" || s), "big")

implemented once, in `agentlab.suite.generate`, and imported here rather than
copied. Because the seed is a function of L, the held-out realization CANNOT be
generated until the prompt winner and the exact checkpoint bytes are frozen and
published. The retired v1 derivation hung off the preregistration commit, which
exists before any lock -- so every held-out answer was derivable in advance and
the generator never consumed the "seed" it published.

WHAT THIS DOES NOT CLAIM. A git commit id is author-influenceable (timestamps,
parents, message), so L is not a randomness beacon: an operator could construct
unpublished candidate lock commits, derive their held-out sets and publish the
preferred one. Requiring L to be published before the reveal makes that visible
in ordinary use, and nothing here can prove a negative about private caches or
what a human saw. docs/AGENTIC_PROTOCOL.md states the exact list of what a reader
may and may not conclude.

Ordering is ENFORCED here, not merely checked later: `reveal` refuses until L
exists as a valid dedicated published commit, `finalize-prereg` refuses unless
train/dev acceptance is closed and no held-out byte exists anywhere, and both
locks refuse to change after a reveal.

A checkpoint lock pins the checkpoint's BYTE DIGEST and the trainer's receipt, not
a mutable path: see `cmd_lock_checkpoint` below.

    lock-prompt              --file configs/frozen_prompt.json
    lock-checkpoint          --path out/multiface/rssft-lora --stage rs_sft
    grpo-disposition         [--verify]   the registered stage-disposition receipt
    finalize-prereg          the P gate: train/dev acceptance + no held-out byte
    verify-prereg
    reveal                   the R receipt; refuses without a valid published L
    verify-heldout-release   the R gate the evaluation stage must pass
    status
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "agentic"
LOCKS = RESULTS / "locks.json"
REVEAL = RESULTS / "seed_reveal.json"
PREREG = ROOT / "configs" / "agentic_preregister.json"
FINAL_MARKER = ROOT / "configs" / "preregistration_final.json"

# The state machine's paths, POSIX-relative, because git speaks in those.
LOCKS_REL = "results/agentic/locks.json"
REVEAL_REL = "results/agentic/seed_reveal.json"
MARKER_REL = "configs/preregistration_final.json"
SUITE_DIR_REL = "data/suite/v1"
HELDOUT_MANIFEST_REL = f"{SUITE_DIR_REL}/manifest.heldout.json"
HELDOUT_SUMS_REL = f"{SUITE_DIR_REL}/SHA256SUMS.heldout"
TRAIN_DEV_MANIFEST_REL = f"{SUITE_DIR_REL}/manifest.train-dev.json"
TRAIN_DEV_SUMS_REL = f"{SUITE_DIR_REL}/SHA256SUMS.train-dev"
SUITE_RELEASE_REL = f"{SUITE_DIR_REL}/suite_release.json"
# The study's output root: the chain's MULTIFACE_OUT. Everything the registered
# study writes lands under it, so "no trained checkpoint exists yet" is a question
# about this directory and not about disclosed pre-pivot history.
STUDY_OUT_REL = "out/multiface"
# Exactly what the reveal commit R may add, and nothing else. R must not touch the
# preregistration, the generator, the prompt, the locks or the evaluation code.
R_ALLOWED = (REVEAL_REL, HELDOUT_MANIFEST_REL, HELDOUT_SUMS_REL, SUITE_RELEASE_REL)

STUDY_ID = "agentic-v1"
LOCKS_SCHEMA = "agentic-locks-v2"
# Publication: the designated public ref is the remote branch; what a clone can
# VERIFY locally is the remote-tracking ref that follows it.
PUBLIC_BRANCH = "refs/heads/main"
PUBLIC_REF = "refs/remotes/origin/main"

# The registered gates finalization refuses without. The observational-equivalence
# test and the token census belong to the fault-contract layer (D1/D2); this reads
# their registered locations out of the preregistration rather than hardcoding a
# second copy of them.
PARITY_TEST_KEY = ("fault_contract_reconciliation_receipt",
                   "observational_equivalence_test", "path")
CENSUS_ARTIFACT_KEY = ("fault_contract_reconciliation_receipt", "caps",
                       "token_census_artifact")
CENSUS_SHA_KEY = ("fault_contract_reconciliation_receipt", "caps",
                  "token_census_sha256")
CENSUS_COMMITTED_REL = "results/agentic/token_census.json"

# The frozen set the marker hash-pins. Everything a claim depends on and nothing
# that a normal day's work touches: the preregistration, the protocol, the
# operational config that carries the counts, gates, hardware and engine
# contract, and the suite size authority.
FROZEN_SET = ("configs/agentic_preregister.json",
              "docs/AGENTIC_PROTOCOL.md",
              "configs/multifaceted.yaml",
              "configs/suite_v1.toml")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=30, check=True).stdout.strip()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes() -> dict:
    out = {}
    for rel in FROZEN_SET:
        p = ROOT / rel
        if p.exists():
            out[rel] = _sha256_file(p)
    return out


def _added_commit(rel: str) -> str | None:
    out = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
    shas = [line for line in out.splitlines() if line]
    return shas[-1] if shas else None


# ---------------------------------------------------------------------------
# git facts: the ONLY thing that establishes order
# ---------------------------------------------------------------------------
# Timestamps are recorded and never trusted. A timestamp can be backdated by
# anyone who can write a file; an ancestry relation cannot be, because changing a
# parent changes every descendant's id.

def _git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=30).returncode == 0


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          timeout=60, check=True).stdout


def adding_commits(rel: str) -> list[str]:
    """Every commit that ADDS `rel`, newest first."""
    out = _git("log", "--all", "--diff-filter=A", "--format=%H", "--", rel)
    return [line for line in out.splitlines() if line]


def touching_commits(rel: str) -> list[str]:
    """Every commit that changes `rel` in any way, newest first."""
    out = _git("log", "--all", "--format=%H", "--", rel)
    return [line for line in out.splitlines() if line]


def is_ancestor(a: str, b: str) -> bool:
    return _git_ok("merge-base", "--is-ancestor", a, b)


def commit_changed_files(sha: str) -> list[str]:
    out = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha)
    return sorted(line for line in out.splitlines() if line)


def tree_has(sha: str, rel: str) -> bool:
    return _git_ok("cat-file", "-e", f"{sha}:{rel}")


def blob_sha256(sha: str, rel: str) -> str:
    """SHA-256 of the file's CONTENT as committed (not the git blob id)."""
    return hashlib.sha256(_git_bytes("show", f"{sha}:{rel}")).hexdigest()


def ref_exists(ref: str) -> bool:
    return _git_ok("rev-parse", "--verify", "--quiet", ref)


def finalization() -> dict | None:
    """The committed finalization marker, or None."""
    if not FINAL_MARKER.exists():
        return None
    return json.loads(FINAL_MARKER.read_text(encoding="utf-8"))


def verify_finalization(strict: bool = True) -> dict:
    """Re-hash the frozen set against the marker.

    A drifted file is refused rather than reported: the whole value of a
    finalization marker is that edits AFTER it are visible.
    """
    marker = finalization()
    if marker is None:
        return {"present": False, "anchor": "oldest-addition-fallback",
                "drifted": [], "missing": []}
    recorded = marker.get("files") or {}
    current = frozen_hashes()
    drifted = sorted(k for k, v in recorded.items() if current.get(k) != v)
    missing = sorted(k for k in recorded if k not in current)
    committed = _added_commit(FINAL_MARKER.relative_to(ROOT).as_posix())
    result = {"present": True, "finalized_at": marker.get("finalized_at"),
              "files": recorded, "drifted": drifted, "missing": missing,
              "commit": committed,
              "anchor": "finalization-marker" if committed else "uncommitted-marker"}
    if strict and (drifted or missing):
        raise SystemExit(
            f"REFUSED: the preregistration was finalized and has drifted since: "
            f"{drifted + missing}. The marker exists precisely so that a forward "
            f"edit cannot pass unnoticed. Either revert the edit, or file a dated "
            f"AMENDMENT and finalize again with fresh held-out seeds.")
    return result


def preregistration_commit() -> str:
    """P: the UNIQUE commit adding configs/preregistration_final.json. No fallback.

    The old oldest-addition fallback is gone. It anchored on the first commit that
    ever added a preregistration FILE, which is fixed forever, so every later edit
    -- including the whole hardware pivot -- left the commitment untouched while
    the documentation claimed the opposite. P is now the commit that pins the
    COMPLETED preregistration and the train/dev commitment, and if it does not
    exist the answer is a refusal rather than a weaker anchor.
    """
    verify_finalization()                     # refuses on drift
    adds = adding_commits(MARKER_REL)
    if not adds:
        raise SystemExit(
            f"REFUSED: no commit adds {MARKER_REL}, so P does not exist. Run "
            f"`agentic_locks.py finalize-prereg` and commit the marker: until it "
            f"is in a commit it anchors nothing.")
    if len(adds) > 1:
        raise SystemExit(
            f"REFUSED: {len(adds)} commits add {MARKER_REL} ({[c[:12] for c in adds]}). "
            f"P must be unique -- 'the preregistration commit' cannot name two "
            f"different trees.")
    P = adds[0]
    on_disk = _sha256_file(FINAL_MARKER)
    if blob_sha256(P, MARKER_REL) != on_disk:
        raise SystemExit(
            f"REFUSED: {MARKER_REL} on disk does not match the bytes committed in "
            f"P ({P[:12]}). An edited marker anchors a tree that was never "
            f"committed.")
    return P


def _generate():
    """The suite generator: the ONE implementation of the held-out derivation."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite import generate

    return generate


def _registered(prereg: dict, keys) -> object:
    node = prereg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f"REFUSED: the preregistration does not register "
                             f"{'.'.join(keys)}")
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# the P gate
# ---------------------------------------------------------------------------
# finalize-prereg belongs BEFORE the first study GPU stage: P has to pin the
# train/dev realization and the held-out derivation RULE before the prompt
# tournament runs, or the tournament happens under an unpinned commitment. Every
# gate below is a positive fact about the tree, and every one of them refuses
# rather than warns.

def _run(cmd: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _py() -> str:
    return sys.executable


def gate_sibling_acceptance(prereg: dict) -> list[str]:
    """D1/D2's CPU acceptance: the parity test passes and the census is pinned.

    The observational-equivalence test is the decisive one for the fault-contract
    unification (it compares rendered prefix token ids per decision, not just
    observation digests), and the size caps are only meaningful against the
    committed tokenizer census. Neither is re-implemented here: the registered path
    is read out of the preregistration and executed.
    """
    problems: list[str] = []
    rel = str(_registered(prereg, PARITY_TEST_KEY))
    path = ROOT / rel
    if not path.exists():
        return [f"the registered observational-equivalence test {rel} is absent"]
    r = _run([_py(), "-m", "pytest", "-q", rel])
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        problems.append(f"the observational-equivalence test {rel} FAILS: {tail}")

    census_rel = str(_registered(prereg, CENSUS_ARTIFACT_KEY))
    want_sha = str(_registered(prereg, CENSUS_SHA_KEY))
    found = None
    for candidate in (census_rel, CENSUS_COMMITTED_REL):
        if (ROOT / candidate).exists():
            found = candidate
            break
    if not found:
        return problems + [
            f"the tokenizer size census is absent ({census_rel} and "
            f"{CENSUS_COMMITTED_REL}); the caps in configs/multifaceted.yaml are "
            f"only meaningful against a recorded measurement"]
    got = _sha256_file(ROOT / found)
    if got != want_sha:
        problems.append(f"{found} hashes {got[:12]} and the preregistration pins "
                        f"{want_sha[:12]}: the census on disk is not the one the "
                        f"caps were registered against")
    if not _git_ok("ls-files", "--error-unmatch", CENSUS_COMMITTED_REL):
        problems.append(f"{CENSUS_COMMITTED_REL} is not committed; an uncommitted "
                        f"census pins nothing")
    census = json.loads((ROOT / found).read_text(encoding="utf-8"))
    want_env = _registered(prereg, ("fault_contract_reconciliation_receipt",
                                    "environment_contract", "current_sha256"))
    if census.get("environment_contract_sha256") != want_env:
        problems.append("the census was measured under a different environment "
                        "contract than the registered one")
    return problems


def gate_train_dev_acceptance() -> list[str]:
    """The train/dev half of the acceptance criteria, run for real."""
    r = _run([_py(), "scripts/validate_suite.py", "--require-phase", "train-dev"])
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
        return [f"scripts/validate_suite.py --require-phase train-dev FAILS: {tail}"]
    if "train_dev_acceptance: PASS" not in r.stdout:
        return ["the validator did not report train_dev_acceptance: PASS"]
    if "heldout_acceptance:   DEFERRED_UNTIL_POST_LOCK_REVEAL" not in r.stdout:
        return ["the validator did not defer held-out acceptance; a run that claims "
                "held-out validation before the reveal is claiming nonexistent bytes"]
    return []


def gate_heldout_generator_refuses(data_dir: str) -> list[str]:
    """The refusal is TESTED, not documented: no reveal, no held-out byte."""
    generate = _generate()
    before = set(generate.stale_heldout_paths(data_dir))
    r = _run([_py(), "scripts/generate_suite.py", "--phase", "heldout"])
    problems = []
    if r.returncode == 0:
        problems.append("scripts/generate_suite.py --phase heldout SUCCEEDED "
                        "without a reveal receipt")
    if "REFUSED" not in (r.stdout + r.stderr):
        problems.append("the held-out generator did not refuse out loud")
    after = set(generate.stale_heldout_paths(data_dir))
    if after - before:
        problems.append(f"the refused run left files behind: {sorted(after - before)}")
    if (ROOT / HELDOUT_MANIFEST_REL).exists():
        problems.append(f"{HELDOUT_MANIFEST_REL} exists before any reveal")
    return problems


def gate_no_heldout_bytes(data_dir: str) -> list[str]:
    generate = _generate()
    problems = []
    stale = generate.stale_heldout_paths(data_dir)
    if stale:
        problems.append(
            f"{len(stale)} held-out file(s) are present in the consumed tree, e.g. "
            f"{stale[:3]}. Before the reveal there must be none: quarantine them "
            f"(scripts/generate_suite.py --quarantine-stale-heldout).")
    for rel in generate.LEGACY_COMMITMENTS:
        if (ROOT / SUITE_DIR_REL / rel).exists():
            problems.append(f"{SUITE_DIR_REL}/{rel} is the retired whole-suite "
                            f"commitment and discloses held-out hashes")
    for rel in (HELDOUT_MANIFEST_REL, HELDOUT_SUMS_REL, REVEAL_REL):
        if _git_ok("ls-files", "--error-unmatch", rel):
            problems.append(f"{rel} is TRACKED at P; the held-out commitments belong "
                            f"to R, which does not exist yet")
    for rel in (TRAIN_DEV_MANIFEST_REL, TRAIN_DEV_SUMS_REL):
        if not (ROOT / rel).exists():
            problems.append(f"{rel} is missing: P must pin the train/dev realization")
    return problems


def gate_nothing_has_run(cfg: dict) -> list[str]:
    """No GPU-hour, no checkpoint, no lock, no reveal, no held-out trace.

    finalize-prereg runs before the tournament, so a lock or a ledger row here
    means the ordering already went wrong and the marker would be pinning a
    commitment the study has partly executed against.
    """
    problems = []
    ledger = ROOT / str(cfg.get("budget", {}).get("ledger",
                                                  "results/agentic/gpu_ledger.jsonl"))
    if ledger.exists():
        rows = [json.loads(ln) for ln in
                ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # STUDY rows, not every row. Requiring an EMPTY ledger made finalization
        # unreachable: the same protocol requires all five dev preflight probes to
        # pass BEFORE the marker is written, and three of those probes run on the
        # GPU and charge their minutes here. Both rules cannot hold at once, so the
        # over-broad one is the defect -- and the fix is to name what actually must
        # not have happened rather than to skip the check.
        #
        # A preflight row is apparatus evidence: dev tasks only, never the held-out
        # split, never a claim-bearing arm, and the preregistration already
        # discloses it as excluded. A row from any stage below would mean the study
        # had begun executing against a commitment that is not yet pinned.
        study = [r for r in rows
                 if not str(r.get("stage", "")).startswith("preflight")]
        if study:
            stages = sorted({str(r.get("stage")) for r in study})
            problems.append(f"{_rel(ledger)} already has {len(study)} STUDY row(s) "
                            f"({', '.join(stages)}): study GPU work has run, so "
                            f"this is not a pre-run finalization")
        pre = [r for r in rows if r not in study]
        if pre:
            hours = sum(float(r.get("minutes") or 0) for r in pre) / 60.0
            print(f"  disclosed (not gated): {len(pre)} preflight ledger row(s), "
                  f"{hours:.3f} GPU-h of dev-only apparatus evidence -- registered "
                  f"as excluded, and required to have run before this marker")
    # The STUDY's output root (the chain's MULTIFACE_OUT). Adapters elsewhere under
    # out/ are the pre-pivot artifacts the preregistration already discloses as
    # excluded prior evidence -- they are reported, not gated on, because gating on
    # them would demand deleting disclosed history.
    study_adapters = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / STUDY_OUT_REL).glob("**/adapter_model.safetensors"))
    if study_adapters:
        problems.append(f"trained checkpoint(s) already exist under "
                        f"{STUDY_OUT_REL}: {study_adapters[:2]}")
    other = sorted(str(p.relative_to(ROOT))
                   for p in (ROOT / "out").glob("**/adapter_model.safetensors")
                   if STUDY_OUT_REL not in str(p))
    if other:
        print(f"  disclosed (not gated): {len(other)} pre-pivot adapter(s) outside "
              f"{STUDY_OUT_REL}, e.g. {other[:2]} -- registered as excluded prior "
              f"evidence, refused by the invalidation rule rather than re-certified")
    if LOCKS.exists():
        problems.append(
            f"{LOCKS_REL} already exists. finalize-prereg belongs BEFORE the prompt "
            f"tournament: P must pin the commitment the tournament and the training "
            f"run under, and L must be a strict DESCENDANT of P.")
    if REVEAL.exists():
        problems.append(f"{REVEAL_REL} exists, so held-out results are already "
                        f"unblinded; finalizing now would move P after the fact")
    return problems


def generation_closure(cfg: dict) -> dict:
    """Every byte that decides what the held-out realization will be.

    Frozen at P so that "the same L gives the same held-out set" is checkable: the
    generator, the loader, the family grammars, the serialization rules, the
    config, and the exact Python runtime.
    """
    rels = ["scripts/generate_suite.py", "scripts/validate_suite.py",
            "scripts/export_eval_specs.py", "configs/suite_v1.toml",
            "src/agentlab/tools.py"]
    suite_dir = ROOT / "src" / "agentlab" / "suite"
    rels += sorted(str(p.relative_to(ROOT)) for p in suite_dir.glob("**/*.py"))
    closure = {rel: _sha256_file(ROOT / rel) for rel in rels
               if (ROOT / rel).exists()}
    return {
        "files": closure,
        "digest": hashlib.sha256(
            json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "python": f"{platform.python_implementation()} "
                  f"{platform.python_version()}",
        "stdlib_only": "generation imports nothing outside the standard library",
    }


def cmd_finalize_prereg(args) -> int:
    """The P gate, then the marker. The OPERATOR commits it; that commit is P."""
    generate = _generate()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    cfg_suite = generate.load_suite_config(str(ROOT / "configs" / "suite_v1.toml"))
    data_dir = str(ROOT / cfg_suite["out_dir"])
    existing = finalization()
    if existing and not args.force:
        state = verify_finalization()
        print(json.dumps(state, indent=2, sort_keys=True))
        print(f"already finalized: {FINAL_MARKER.relative_to(ROOT)}")
        return 0

    gates = {
        "no_heldout_bytes_anywhere": gate_no_heldout_bytes(data_dir),
        "nothing_has_run_yet": gate_nothing_has_run(_load_cfg()),
        "heldout_generator_refuses_without_a_reveal":
            gate_heldout_generator_refuses(data_dir),
        "sibling_cpu_acceptance_d1_d2": gate_sibling_acceptance(prereg),
        "train_dev_acceptance": gate_train_dev_acceptance(),
    }
    print("the P gate:")
    for name, problems in gates.items():
        print(f"  [{'PASS' if not problems else 'FAIL'}] {name}")
        for p in problems:
            print(f"        {p}")
    if any(gates.values()):
        raise SystemExit(
            "REFUSED: finalization is the moment the commitment becomes binding, so "
            "every gate above has to be closed FIRST. Nothing here is a threshold "
            "that can be relaxed: fix the mechanism, then finalize.")

    hashes = frozen_hashes()
    missing = [rel for rel in FROZEN_SET if rel not in hashes]
    payload = {
        "finalized_at": now(),
        "head_at_finalization": _git("rev-parse", "HEAD"),
        "files": hashes,
        "absent": missing,
        "study_id": STUDY_ID,
        "train_dev_acceptance": "PASS",
        "heldout_acceptance": "DEFERRED_UNTIL_POST_LOCK_REVEAL",
        "heldout_acceptance_note":
            "No held-out byte exists at P and none may. This marker does NOT claim "
            "that held-out hashes, cardinalities, controls, the core cluster census "
            "or train/dev/eval cross-isolation were validated: they are established "
            "at R, over the release the locks commit determines.",
        "train_dev_commitment": {
            "manifest": TRAIN_DEV_MANIFEST_REL,
            "manifest_sha256": _sha256_file(ROOT / TRAIN_DEV_MANIFEST_REL),
            "sums": TRAIN_DEV_SUMS_REL,
            "sums_sha256": _sha256_file(ROOT / TRAIN_DEV_SUMS_REL),
        },
        "heldout_derivation": {
            "label": generate.HELDOUT_DERIVATION,
            "master_label_parts": list(generate.HELDOUT_MASTER_LABEL_PARTS),
            "split_label": generate.HELDOUT_SPLIT_LABEL,
            "release_label": generate.HELDOUT_RELEASE_LABEL,
            "generation_protocol": generate.GENERATION_PROTOCOL,
            "seed_source": "L: the unique dedicated commit adding "
                           + LOCKS_REL,
            "splits": list(generate.HELDOUT_SPLITS),
        },
        "heldout_plan_digest": hashlib.sha256(
            json.dumps(generate.heldout_plan(cfg_suite), sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "generation_closure": generation_closure(cfg_suite),
        "gates": {name: "PASS" for name in gates},
        "note": "The commit that ADDS this file is P. It pins the COMPLETED "
                "preregistration, the train/dev realization and the held-out "
                "derivation RULE -- not a held-out seed value, which cannot exist "
                "before L. No history is rewritten.",
        "seed_derivation": "master = SHA256('qwen35-agentic-lab\\0agentlab-suite-v1"
                           "\\0heldout-master-v2\\0' || bytes.fromhex(L)); "
                           "split_seed(s) = int(SHA256('agentlab-heldout-split-v2\\0'"
                           " || master || '\\0' || s), 'big')",
    }
    FINAL_MARKER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(f"\nwrote {FINAL_MARKER.relative_to(ROOT)} pinning {len(hashes)} frozen files")
    for rel, sha in sorted(hashes.items()):
        print(f"  {sha[:12]}  {rel}")
    print(f"  generation closure {payload['generation_closure']['digest'][:12]} "
          f"({len(payload['generation_closure']['files'])} files, "
          f"{payload['generation_closure']['python']})")
    print("\nCOMMIT IT -- an uncommitted marker anchors nothing, and its commit is P:")
    print(f"  git add {FINAL_MARKER.relative_to(ROOT)} {TRAIN_DEV_MANIFEST_REL} "
          f"{TRAIN_DEV_SUMS_REL}")
    print('  git commit -m "Finalize the preregistration: pin the completed '
          'commitment and the train/dev realization"')
    return 0


def grpo_disposition_payload(cfg: dict) -> dict:
    """The registered GRPO stage-disposition receipt.

    A stage DISPOSITION says what happened to a stage; it is never one of the gate
    states PASS/FAIL/INCONCLUSIVE/BUG. `GRPO_NOT_RUN_HARDWARE_INFEASIBLE` says the
    registered trainer cannot instantiate on the registered card and says NOTHING
    about whether the variance gate would have opened -- which is why it is not
    interchangeable with `GRPO_NOT_RUN_VARIANCE_GATE_CLOSED`, and why the variance
    gate is recorded as NOT_EVALUATED rather than closed.

    The arithmetic is part of the receipt: an artifact that does not SHOW the
    shortfall does not establish infeasibility.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import (config_hash, fingerprint, git_sha,
                                         hardware_contract, now_utc)

    grpo = cfg["grpo"]
    hw = hardware_contract(cfg)
    arith = dict(grpo["disposition_arithmetic"])
    if not arith["vllm_allocation_gib"] < arith["vllm_policy_copy_gib"]:
        raise SystemExit(
            "REFUSED: the disposition arithmetic does not show the shortfall "
            "(vllm_allocation_gib must be smaller than vllm_policy_copy_gib), so "
            "it does not establish infeasibility.")
    return {
        "branch": "grpo",
        "outcome": grpo["stage_disposition"],
        "variance_gate": cfg["variance_probe"]["stage_disposition"],
        "trainer_feasibility": "INFEASIBLE",
        "optimizer_steps": 0,
        "gpu_name": hw["expected_name"],
        "cuda_visible_bytes": hw["cuda_visible_bytes"],
        "config_hash": config_hash(),
        "checkpoint_hash": None,
        "git_sha": git_sha(),
        "arithmetic": arith,
        "timestamp_utc": now_utc(),
        "not_interchangeable_with": "GRPO_NOT_RUN_VARIANCE_GATE_CLOSED",
        "note": "A stage DISPOSITION, never a gate state. The registered colocate "
                "configuration cannot instantiate on the registered card, so no "
                "GRPO checkpoint exists, the R0/RP arms are ABSENT BY DESIGN, and "
                "RS-SFT is the sole trained candidate. No RS-SFT-versus-GRPO dev "
                "comparison is performed because there is no second candidate.",
        "forbidden_substitutions": ["microbatch 1", "2048-token completions",
                                    "no-vLLM (transformers) generation",
                                    "quantization", "CPU/disk offload",
                                    "an alternate optimizer", "another GPU model"],
        "provenance": fingerprint(),
    }


def require_grpo_disposition(cfg: dict, path: pathlib.Path) -> dict:
    """Read the artifact and refuse anything but the registered disposition.

    "GRPO was skipped for a reason nobody wrote down" is indistinguishable from
    "GRPO ran and lost", and only one of those is reportable -- so a missing or
    mislabelled artifact is an error here, never a silent selection of RS-SFT.
    """
    if not path.exists():
        raise SystemExit(f"REFUSED: no GRPO disposition artifact at {path}. A "
                         f"missing GRPO checkpoint with no disposition is an "
                         f"ERROR, not a silent fallback to RS-SFT.")
    rec = json.loads(path.read_text(encoding="utf-8"))
    want = cfg["grpo"]["stage_disposition"]
    if rec.get("outcome") != want:
        raise SystemExit(f"REFUSED: the disposition artifact says "
                         f"{rec.get('outcome')!r}; the registered disposition is "
                         f"{want!r}")
    gate = cfg["variance_probe"]["stage_disposition"]
    if rec.get("variance_gate") != gate:
        raise SystemExit(f"REFUSED: the artifact's variance_gate is "
                         f"{rec.get('variance_gate')!r}, registered {gate!r}. "
                         f"'closed' would claim the complete 144-group probe ran "
                         f"and a binding gate failed.")
    return rec


def _load_cfg() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import load_config

    return load_config()


def cmd_grpo_disposition(args) -> int:
    cfg = _load_cfg()
    path = ROOT / cfg["grpo"]["disposition_artifact"]
    if args.verify:
        rec = require_grpo_disposition(cfg, path)
        print(f"[grpo] {rec['outcome']} (variance gate: {rec['variance_gate']})")
        return 0
    if path.exists() and not args.force:
        rec = require_grpo_disposition(cfg, path)
        print(f"[grpo] already recorded: {rec['outcome']} -> {path}")
        return 0
    payload = grpo_disposition_payload(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"[grpo] {payload['outcome']} -> {path}")
    print(f"       variance gate: {payload['variance_gate']}")
    print(f"       vLLM carve {payload['arithmetic']['vllm_allocation_gib']} GiB "
          f"< its own policy copy {payload['arithmetic']['vllm_policy_copy_gib']} GiB")
    return 0


def cmd_verify_prereg(args) -> int:
    state = verify_finalization(strict=not args.report_only)
    print(json.dumps(state, indent=2, sort_keys=True))
    if not state["present"]:
        print("NOT FINALIZED: the anchor is still the oldest commit that added "
              "the preregistration, so forward edits do not change the "
              "commitment. Run `finalize-prereg` and commit the marker.")
        return 1 if args.require else 0
    if state["anchor"] == "uncommitted-marker":
        print("MARKER NOT COMMITTED: it anchors nothing until it is in a commit.")
        return 1 if args.require else 0
    print(f"finalized and clean; anchor commit {state['commit'][:12]}")
    return 0


def locks_payload_problems(locks: dict, prereg_commit: str) -> list[str]:
    """Is this a COMPLETE lock? -> problems.

    Structure only: the checkpoint's byte-level verification belongs to
    `verify_checkpoint_lock`, which owns the trainer-receipt grammar, and is
    called by the reveal rather than reimplemented here.
    """
    problems = []
    if locks.get("schema") != LOCKS_SCHEMA:
        problems.append(f"schema is {locks.get('schema')!r}, not {LOCKS_SCHEMA!r}")
    if locks.get("study_id") != STUDY_ID:
        problems.append(f"study_id is {locks.get('study_id')!r}, not {STUDY_ID!r}")
    if str(locks.get("preregistration_commit")) != prereg_commit:
        problems.append(f"preregistration_commit is "
                        f"{locks.get('preregistration_commit')!r}, and P is "
                        f"{prereg_commit}")
    prompt = locks.get("prompt_winner") or {}
    if not prompt.get("file") or len(str(prompt.get("sha256", ""))) != 64:
        problems.append("prompt_winner carries no file + sha256")
    ckpt = locks.get("checkpoint") or {}
    if len(str(ckpt.get("checkpoint_sha256", ""))) != 64:
        problems.append("checkpoint carries no 64-hex checkpoint_sha256: a lock on "
                        "a mutable PATH pins nothing")
    if not ckpt.get("checkpoint_files"):
        problems.append("checkpoint carries no per-file digest manifest")
    if ckpt.get("stage") not in ("rs_sft", "grpo"):
        problems.append(f"checkpoint stage {ckpt.get('stage')!r} is not one of "
                        f"rs_sft/grpo")
    if not locks.get("selection"):
        problems.append("no selection receipt: a lock must say whether the "
                        "checkpoint won a dev comparison or was the sole candidate")
    return problems


def locks_commit(*, prereg_commit: str | None = None) -> dict:
    """L, with every condition it has to satisfy, or a refusal that says which.

    L is identified EXTERNALLY -- as the unique commit that adds the locks blob --
    so `locks.json` never has to contain a field pretending to hold its own commit
    sha, which is impossible.
    """
    P = prereg_commit or preregistration_commit()
    if not LOCKS.exists():
        raise SystemExit(f"REFUSED: {LOCKS_REL} does not exist. Lock the prompt "
                         f"winner and the checkpoint first; that ordering IS S18.")
    locks = read_locks()
    problems = locks_payload_problems(locks, P)
    if problems:
        raise SystemExit(
            "REFUSED: the lock is INCOMPLETE, so there is nothing for the held-out "
            "seed to be a function of:\n  - " + "\n  - ".join(problems))
    adds = adding_commits(LOCKS_REL)
    if not adds:
        raise SystemExit(
            f"REFUSED: {LOCKS_REL} exists on disk but no commit adds it. L must be "
            f"a real commit: the seed derives from its sha, and an uncommitted file "
            f"has none. Commit it ALONE:\n"
            f"    git add {LOCKS_REL}\n"
            f'    git commit -m "Lock the prompt winner and the trained checkpoint"')
    if len(adds) > 1:
        raise SystemExit(f"REFUSED: {len(adds)} commits add {LOCKS_REL} "
                         f"({[c[:12] for c in adds]}); L must be unique")
    L = adds[0]
    touching = touching_commits(LOCKS_REL)
    facts = {}
    problems = []
    if len(touching) != 1 or touching[0] != L:
        problems.append(
            f"{len(touching)} commits change {LOCKS_REL}; a later edit means the "
            f"published lock is not the lock in the tree, and the seed would "
            f"derive from a superseded one")
    changed = commit_changed_files(L)
    if changed != [LOCKS_REL]:
        problems.append(f"L changes {changed}; it must be a DEDICATED commit that "
                        f"changes only {LOCKS_REL}")
    if not is_ancestor(P, L) or P == L:
        problems.append(f"L ({L[:12]}) is not a strict descendant of P ({P[:12]}): "
                        f"the relation P < L is what makes the lock post-"
                        f"preregistration")
    for rel in (REVEAL_REL, HELDOUT_MANIFEST_REL, HELDOUT_SUMS_REL):
        if tree_has(L, rel):
            problems.append(f"L already carries {rel}: a lock commit that contains "
                            f"a reveal or a held-out commitment is not a lock, it "
                            f"is the reveal")
    on_disk = _sha256_file(LOCKS)
    committed = blob_sha256(L, LOCKS_REL)
    if committed != on_disk:
        problems.append(f"{LOCKS_REL} on disk hashes {on_disk[:12]} and L committed "
                        f"{committed[:12]}: the seed must derive from the PUBLISHED "
                        f"lock, not from an edited copy")
    if not ref_exists(PUBLIC_REF):
        problems.append(f"the designated public ref {PUBLIC_REF} does not exist "
                        f"here, so publication cannot be verified")
    elif not is_ancestor(L, PUBLIC_REF):
        problems.append(
            f"L ({L[:12]}) is not published on {PUBLIC_REF}. Push it BEFORE "
            f"revealing: an unpublished lock leaves no evidence that candidate lock "
            f"commits were not ground and discarded until a favourable held-out "
            f"seed appeared.")
    if problems:
        raise SystemExit("REFUSED: L is not a valid lock commit:\n  - "
                         + "\n  - ".join(problems))
    facts.update({"commit": L, "preregistration_commit": P,
                  "locks_blob_sha256": on_disk,
                  "changed_files": changed,
                  "published_on": PUBLIC_REF,
                  "public_branch": PUBLIC_BRANCH})
    return facts


def read_locks() -> dict:
    return json.loads(LOCKS.read_text(encoding="utf-8")) if LOCKS.exists() else {}


def write_locks(locks: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOCKS.write_text(json.dumps(locks, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")


def _refuse_if_revealed(what: str) -> None:
    if REVEAL.exists() or (ROOT / HELDOUT_MANIFEST_REL).exists():
        raise SystemExit(
            f"REFUSED: {REVEAL.relative_to(ROOT)} already exists, so held-out "
            f"results have been unblinded. Changing the {what} lock now is the "
            f"S18 violation this script exists to prevent. A genuinely new "
            f"candidate needs a dated AMENDMENT and fresh held-out seeds.")


def _stamp_locks_header(locks: dict, prereg_commit: str) -> dict:
    """The lock's own identity: schema, study, and the P it descends from.

    Deliberately NOT its own commit sha -- a file cannot contain the hash of the
    commit that adds it. L is identified externally, as the unique commit adding
    this blob.
    """
    locks["schema"] = LOCKS_SCHEMA
    locks["study_id"] = STUDY_ID
    locks["preregistration_commit"] = prereg_commit
    locks["note"] = ("L is the unique DEDICATED commit that adds this file; the six "
                     "held-out generation seeds are heldout-master-v2 over that "
                     "commit sha. Timestamps here are informational: order is "
                     "established by git ancestry, never by them.")
    return locks


def cmd_lock_prompt(args) -> int:
    _refuse_if_revealed("prompt_winner")
    # Every prompt/training stage requires a committed, clean P: the tournament
    # runs AFTER finalization, so if P does not exist yet the ordering is wrong.
    prereg_commit = preregistration_commit()
    frozen = json.loads(pathlib.Path(args.file).read_text(encoding="utf-8"))
    winner = frozen.get("winner") or frozen
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    block = prereg["prompt_candidates"]
    # `prompt_control finalize` records the winner as the candidate FILE NAME plus
    # the file's sha256; the directory is the preregistered one, so the path is
    # derived rather than trusted from the tournament output.
    name = winner.get("candidate") or winner.get("file") or winner.get("path")
    sha = winner.get("sha256")
    if not name or not sha:
        raise SystemExit(f"{args.file} carries no winner candidate/sha256; run "
                         f"`python -m agentlab.prompt_control finalize` first")
    path = f"{block['directory']}/{pathlib.PurePosixPath(name).name}"
    registered = set(block["sha256"].values())
    if sha not in registered:
        raise SystemExit(f"REFUSED: prompt sha {sha[:12]} is not one of the eight "
                         f"preregistered candidates (S16)")
    on_disk = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    if on_disk != sha:
        raise SystemExit(f"REFUSED: {path} on disk hashes {on_disk[:12]}, the "
                         f"tournament recorded {sha[:12]}")
    locks = _stamp_locks_header(read_locks(), prereg_commit)
    locks["prompt_winner"] = {"file": path, "sha256": sha, "locked_at": now(),
                              "commit": _git("rev-parse", "HEAD")}
    write_locks(locks)
    print(f"locked prompt winner {path} ({sha[:12]})")
    print(f"  under P {prereg_commit[:12]}")
    return 0


# --------------------------------------------------------------------------
# the checkpoint lock: BYTES, and the chain that produced them
# --------------------------------------------------------------------------
#
# The lock used to record {path, stage, locked_at, commit}. A path is mutable: the
# adapter behind it can be retrained, overwritten or swapped after the held-out
# seed is revealed, and the lock would still "hold". Worse, nothing connected the
# locked adapter to the trajectories, the card or the run that produced it, so
# S18 blindness and S19 hardware integrity both stopped at this file.
#
# A lock now pins the checkpoint's CONTENT DIGEST and the digest of the trainer's
# receipt, and it refuses to exist without them. `verify_checkpoint_lock` re-hashes
# the bytes, so a post-lock swap is detected rather than assumed away.

CHECKPOINT_LOCK_FIELDS = ("path", "stage", "locked_at", "commit",
                          "checkpoint_sha256", "training_manifest",
                          "training_manifest_sha256", "gpu_uuid",
                          "environment_contract_sha256", "config_hash",
                          "views_sha256", "source_provenance")


def _sft_module():
    """The trainer owns the training-manifest grammar; this reads it, never a copy."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab import sft

    return sft


def cmd_lock_checkpoint(args) -> int:
    _refuse_if_revealed("checkpoint")
    p = ROOT / args.path
    if not p.exists():
        raise SystemExit(f"REFUSED: {args.path} does not exist")
    sft = _sft_module()
    cfg = _load_cfg()
    override = getattr(args, "training_manifest", None)
    manifest_path = (ROOT / override) if override else sft.training_manifest_path(p)
    # The whole chain, or no lock: the receipt hashes to itself, was written under
    # this environment contract, the checkpoint bytes still hash to the digest it
    # recorded, its card is this run's card and the ledger's card, and the corpus
    # it names was produced by an attested GPU session.
    rec = sft.require_training_manifest(manifest_path, checkpoint_path=p, cfg=cfg,
                                        stage=args.stage)
    inputs = rec["inputs"]
    locks = read_locks()
    locks["checkpoint"] = {
        "path": args.path,
        "stage": args.stage,
        "locked_at": now(),
        "commit": _git("rev-parse", "HEAD"),
        # WHAT is locked: the bytes, not the name of the directory holding them.
        "checkpoint_sha256": rec["checkpoint"]["checkpoint_sha256"],
        "checkpoint_files": rec["checkpoint"]["files"],
        # WHO produced them, and from what.
        "training_manifest": _rel(manifest_path),
        "training_manifest_sha256": rec[_hash_field()],
        "trained_at_utc": rec["finished_at_utc"],
        "optimizer_steps": rec["optimizer_steps"],
        "gpu_uuid": rec["hardware"]["gpu_uuid"],
        "gpu_name": rec["hardware"]["gpu_name"],
        "runtime_manifest_sha256": rec["runtime_manifest_sha256"],
        "session_id": rec["session_id"],
        "git_sha_trained": rec["git_sha"],
        "config_hash": rec["config_hash"],
        "environment_contract_sha256": rec["environment_contract_sha256"],
        "views_sha256": inputs["views_sha256"],
        "views_rows": inputs["views_rows"],
        "source_provenance": inputs["source_provenance"],
        "source_runtime_manifests": inputs["source_runtime_manifests"],
    }
    write_locks(locks)
    print(f"locked checkpoint {args.path} (selected stage: {args.stage})")
    print(f"  checkpoint sha256   {locks['checkpoint']['checkpoint_sha256']}")
    print(f"  training manifest   {locks['checkpoint']['training_manifest_sha256']}"
          f"  ({locks['checkpoint']['training_manifest']})")
    print(f"  trained on          {locks['checkpoint']['gpu_name']} "
          f"{locks['checkpoint']['gpu_uuid']}")
    print(f"  corpus              {inputs['views_rows']} rows, "
          f"{inputs['source_trajectories']} trajectories, "
          f"views {inputs['views_sha256'][:12]}")
    return 0


def _rel(path: pathlib.Path) -> str:
    p = pathlib.Path(path)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _hash_field() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import MANIFEST_HASH_FIELD

    return MANIFEST_HASH_FIELD


def verify_checkpoint_lock(locks: dict | None = None, *, cfg: dict | None = None,
                           verify_bytes: bool = True) -> dict:
    """Re-check a recorded checkpoint lock. Used before any held-out verdict.

    A path-only lock -- the shape this script used to write -- is REFUSED here
    rather than reported, because a verdict that cites "the locked checkpoint"
    while the lock names only a mutable directory cites nothing.
    """
    locks = read_locks() if locks is None else locks
    rec = (locks or {}).get("checkpoint")
    if not rec:
        raise SystemExit("REFUSED: no checkpoint is locked; that ordering IS S18.")
    missing = [k for k in CHECKPOINT_LOCK_FIELDS
               if rec.get(k) is None or rec.get(k) == ""]
    if missing:
        raise SystemExit(
            f"REFUSED: the checkpoint lock is missing {', '.join(missing)}. A lock "
            f"that records only a PATH pins nothing: the bytes behind that path can "
            f"be retrained or swapped after the held-out seed is revealed, and "
            f"nothing ties them to the trajectories, the card or the run that "
            f"produced them. Re-lock with `agentic_locks.py lock-checkpoint`, which "
            f"requires the trainer's receipt.")
    sft = _sft_module()
    cfg = cfg or _load_cfg()
    if verify_bytes:
        sft.require_training_manifest(ROOT / rec["training_manifest"],
                                      checkpoint_path=ROOT / rec["path"],
                                      cfg=cfg, stage=rec["stage"])
        manifest = sft.read_training_manifest(ROOT / rec["training_manifest"])
        if manifest[_hash_field()] != rec["training_manifest_sha256"]:
            raise SystemExit(
                f"REFUSED: the training manifest now hashes "
                f"{manifest[_hash_field()]} and the lock recorded "
                f"{rec['training_manifest_sha256']}; the receipt changed after the "
                f"lock was taken.")
        if manifest["checkpoint"]["checkpoint_sha256"] != rec["checkpoint_sha256"]:
            raise SystemExit(
                f"REFUSED: the locked checkpoint digest {rec['checkpoint_sha256']} "
                f"is not the digest the receipt records "
                f"({manifest['checkpoint']['checkpoint_sha256']}).")
    return rec


def reveal_payload(facts: dict, generate) -> dict:
    """The receipt. It RECORDS a derivation; it does not release a secret.

    Once L is public the seed is mathematically derivable by anyone, which is the
    point: the reveal is a receipt saying which L this study's held-out set came
    from, not a key handed out at a chosen moment.
    """
    master = generate.heldout_master_seed(facts["commit"])
    return {
        "schema": generate.REVEAL_SCHEMA,
        "study_id": STUDY_ID,
        "preregistration_commit": facts["preregistration_commit"],
        "locks_commit": facts["commit"],
        "locks_blob_sha256": facts["locks_blob_sha256"],
        "public_ref": PUBLIC_BRANCH,
        "public_ref_verified": facts["published_on"],
        "master_seed_hex": master.hex(),
        "heldout_release_id": generate.heldout_release_id(master),
        "derivation_label": generate.HELDOUT_DERIVATION,
        "generation_protocol": generate.GENERATION_PROTOCOL,
        "generator_commit": facts["preregistration_commit"],
        "revealed_at": now(),
        "timestamps_are_informational": True,
        "ordering_is_established_by": "git ancestry P < L < R <= E, never timestamps",
        "note": "The six held-out split seeds and the release id are derived from "
                "master; none of them is stored here as a value to be trusted. "
                "A commit sha is author-influenceable, so this is a "
                "post-lock-derivation receipt, not a randomness beacon.",
    }


def cmd_reveal(args) -> int:
    generate = _generate()
    P = preregistration_commit()
    facts = locks_commit(prereg_commit=P)
    # The checkpoint's BYTES, re-hashed now: a checkpoint mutated after L was
    # published would otherwise be revealed against.
    verify_checkpoint_lock(read_locks(), verify_bytes=not args.no_bytes)
    cfg = generate.load_suite_config(str(ROOT / "configs" / "suite_v1.toml"))
    data_dir = str(ROOT / cfg["out_dir"])
    generate.assert_no_stale_heldout(data_dir)
    payload = reveal_payload(facts, generate)
    if REVEAL.exists():
        existing = json.loads(REVEAL.read_text(encoding="utf-8"))
        for key in ("locks_commit", "master_seed_hex", "heldout_release_id",
                    "preregistration_commit"):
            if existing.get(key) != payload[key]:
                raise SystemExit(
                    f"REFUSED: {REVEAL_REL} already exists and its {key} is "
                    f"{existing.get(key)!r}, but this state machine derives "
                    f"{payload[key]!r}. A second, different reveal for the same "
                    f"study is a replay: it needs a dated AMENDMENT, not an "
                    f"overwrite.")
        print(f"already revealed (identical receipt): {REVEAL_REL}")
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    REVEAL.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"revealed the held-out derivation for L {facts['commit'][:12]} "
          f"(published on {facts['published_on']})")
    print(f"  P                 {facts['preregistration_commit'][:12]}")
    print(f"  release id        {payload['heldout_release_id']}")
    print(f"  derivation        {payload['derivation_label']}")
    print("\nNEXT, in this order:")
    print(f"  scripts/generate_suite.py --phase heldout --reveal {REVEAL_REL}")
    print("  scripts/export_eval_specs.py --splits "
          + " ".join(generate.PHASES["heldout"]))
    print("  scripts/generate_suite.py --phase heldout --seal")
    print("  scripts/validate_suite.py --require-phase heldout")
    print(f"  git add {REVEAL_REL} {HELDOUT_MANIFEST_REL} {HELDOUT_SUMS_REL}")
    print('  git commit -m "Reveal the held-out derivation and pin the release"')
    print("  git push          # R must be published before any evaluation")
    return 0


def heldout_release_state() -> dict:
    """R: the reveal commit and every condition the evaluation stage requires."""
    generate = _generate()
    state: dict = {"reveal_present": REVEAL.exists(), "problems": []}
    if not REVEAL.exists():
        state["problems"].append(f"{REVEAL_REL} does not exist: nothing is revealed")
        return state
    receipt = generate.load_reveal(str(REVEAL))
    P = preregistration_commit()
    facts = locks_commit(prereg_commit=P)
    L = facts["commit"]
    problems: list[str] = []
    if receipt["locks_commit"] != L:
        problems.append(f"the receipt names L {receipt['locks_commit'][:12]} and the "
                        f"repository's unique lock commit is {L[:12]}")
    if receipt["preregistration_commit"] != P:
        problems.append(f"the receipt names P {receipt['preregistration_commit'][:12]}"
                        f" and P is {P[:12]}")
    if receipt["locks_blob_sha256"] != facts["locks_blob_sha256"]:
        problems.append("the receipt's locks_blob_sha256 is not the committed lock")
    adds = adding_commits(REVEAL_REL)
    R = None
    if not adds:
        problems.append(f"no commit adds {REVEAL_REL}: R does not exist yet. Commit "
                        f"the receipt together with the held-out commitments.")
    elif len(adds) > 1:
        problems.append(f"{len(adds)} commits add {REVEAL_REL}; R must be unique")
    else:
        R = adds[0]
        changed = set(commit_changed_files(R))
        must = {REVEAL_REL, HELDOUT_MANIFEST_REL, HELDOUT_SUMS_REL}
        if not must <= changed:
            problems.append(
                f"R adds {sorted(changed)}; it must add the receipt AND the held-out "
                f"commitments together ({sorted(must)}) -- a reveal without its "
                f"manifest lets the evaluated bytes be chosen after the fact")
        stray = sorted(changed - set(R_ALLOWED))
        if stray:
            problems.append(f"R also changes {stray}; the reveal commit may not "
                            f"touch the preregistration, the generator, the prompt, "
                            f"the lock or the evaluation code")
        if not is_ancestor(L, R) or L == R:
            problems.append(f"R ({R[:12]}) is not a strict descendant of L ({L[:12]})")
        if not ref_exists(PUBLIC_REF):
            problems.append(f"{PUBLIC_REF} does not exist, so R's publication cannot "
                            f"be verified")
        elif not is_ancestor(R, PUBLIC_REF):
            problems.append(f"R ({R[:12]}) is not published on {PUBLIC_REF}; "
                            f"evaluation waits until it is")
        for rel in sorted(must):
            if blob_sha256(R, rel) != _sha256_file(ROOT / rel):
                problems.append(f"{rel} on disk is not the bytes R committed")
    cfg = generate.load_suite_config(str(ROOT / "configs" / "suite_v1.toml"))
    data_dir = str(ROOT / cfg["out_dir"])
    manifest = generate.heldout_release(data_dir)
    if manifest is None:
        problems.append(f"{HELDOUT_MANIFEST_REL} is absent: the release is not pinned")
    else:
        if not manifest.get("sealed"):
            problems.append("the held-out manifest is UNSEALED: the certspecs the "
                            "evaluator reads are not pinned by it")
        if manifest.get("heldout_release_id") != receipt["heldout_release_id"]:
            problems.append("the manifest's release id is not the receipt's")
        if manifest.get("locks_commit") != receipt["locks_commit"]:
            problems.append("the manifest's locks commit is not the receipt's")
        listed = generate.read_sums(generate.phase_sums_path(data_dir, "heldout"))
        bad = [rel for rel, digest in sorted(listed.items())
               if not os.path.exists(os.path.join(data_dir, rel))
               or generate.file_sha256(os.path.join(data_dir, rel)) != digest]
        if bad:
            problems.append(f"{len(bad)} released file(s) do not match "
                            f"SHA256SUMS.heldout, e.g. {bad[:3]}")
        stale = generate.stale_heldout_paths(data_dir)
        if stale:
            problems.append(f"held-out files outside the release: {stale[:3]}")
    state.update({"P": P, "L": L, "R": R,
                  "heldout_release_id": receipt["heldout_release_id"],
                  "problems": problems})
    return state


def cmd_verify_heldout_release(args) -> int:
    state = heldout_release_state()
    if state["problems"]:
        print("REFUSED: the held-out release is not verified:")
        for p in state["problems"]:
            print(f"  - {p}")
        return 1
    print(f"held-out release VERIFIED: P {state['P'][:12]} < L {state['L'][:12]} < "
          f"R {state['R'][:12]}, all published on {PUBLIC_REF}")
    print(f"  release id {state['heldout_release_id']}")
    return 0


def cmd_status(args) -> int:
    def line(label: str, value: str) -> None:
        print(f"  {label:<20} {value}")

    locks = read_locks()
    state = verify_finalization(strict=False)
    try:
        P = preregistration_commit()
    except SystemExit as exc:
        P = None
        line("P (prereg)", f"-- {exc}")
    if P:
        line("P (prereg)", f"{P[:12]}  [{state['anchor']}]"
             + (f"  DRIFTED={state['drifted']}" if state.get("drifted") else ""))
    for key in ("prompt_winner", "checkpoint"):
        rec = locks.get(key)
        line(key, "-- not locked" if not rec else json.dumps(rec)[:160])
    if P:
        try:
            facts = locks_commit(prereg_commit=P)
            line("L (locks commit)", f"{facts['commit'][:12]}  published on "
                                     f"{facts['published_on']}")
        except SystemExit as exc:
            line("L (locks commit)", f"-- {str(exc).splitlines()[0]}")
    line("reveal", "-- not revealed" if not REVEAL.exists()
         else json.loads(REVEAL.read_text(encoding="utf-8"))["heldout_release_id"])
    if REVEAL.exists() and P:
        rstate = heldout_release_state()
        line("R (reveal commit)",
             (rstate.get("R") or "-- not committed")
             + (f"  problems={len(rstate['problems'])}" if rstate["problems"] else "  OK"))
        for p in rstate["problems"][:6]:
            print(f"      - {p}")
    else:
        line("R (reveal commit)", "-- nothing revealed yet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("lock-prompt")
    lp.add_argument("--file", default="configs/frozen_prompt.json")
    lp.set_defaults(fn=cmd_lock_prompt)
    lc = sub.add_parser("lock-checkpoint")
    lc.add_argument("--path", required=True)
    lc.add_argument("--stage", default="rs_sft", choices=("rs_sft", "grpo"))
    lc.add_argument("--training-manifest", default=None,
                    help="the trainer's receipt (default: "
                         "<path>.agentlab_training_manifest.json). A checkpoint "
                         "with no receipt cannot be locked: the lock pins the "
                         "checkpoint's BYTE DIGEST and the chain that produced "
                         "them, never a mutable path.")
    lc.set_defaults(fn=cmd_lock_checkpoint)
    fp = sub.add_parser("finalize-prereg")
    fp.add_argument("--force", action="store_true",
                    help="re-pin the frozen set (a dated AMENDMENT, not a fixup)")
    fp.set_defaults(fn=cmd_finalize_prereg)
    gd = sub.add_parser("grpo-disposition")
    gd.add_argument("--verify", action="store_true",
                    help="read and check the artifact instead of writing it")
    gd.add_argument("--force", action="store_true")
    gd.set_defaults(fn=cmd_grpo_disposition)
    vp = sub.add_parser("verify-prereg")
    vp.add_argument("--require", action="store_true",
                    help="exit non-zero when the preregistration is not finalized")
    vp.add_argument("--report-only", action="store_true",
                    help="report drift instead of refusing")
    vp.set_defaults(fn=cmd_verify_prereg)
    rv = sub.add_parser("reveal")
    rv.add_argument("--require-finalization", action="store_true",
                    help="accepted and always enforced: P (the committed "
                         "finalization marker) is now a precondition, not an option")
    rv.add_argument("--no-bytes", action="store_true",
                    help="skip re-hashing the checkpoint bytes (diagnostics only; "
                         "the chain never passes it)")
    rv.set_defaults(fn=cmd_reveal)
    vh = sub.add_parser("verify-heldout-release")
    vh.set_defaults(fn=cmd_verify_heldout_release)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
