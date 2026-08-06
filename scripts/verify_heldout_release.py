#!/usr/bin/env python3
"""The gate that must pass BEFORE any evaluation GPU is allocated.

It answers one question and refuses if the answer is not yes:

    is the held-out release the study is about to be evaluated on PROVED to have
    been fixed after the lock, by git ancestry?

`P < L < R <= E` is the whole claim.

    P   the preregistration commit
    L   the dedicated commit that adds the complete results/agentic/locks.json
    R   the dedicated commit that adds results/agentic/seed_reveal.json together
        with data/suite/v1/manifest.heldout.json and SHA256SUMS.heldout
    E   the commit the evaluation is about to run at (HEAD, or --head)

ANCESTRY, NEVER TIMESTAMPS. This tool reads no mtime and no field named `*_at`.
A timestamp can be written by anyone who can write a file; an ancestry relation
cannot be forged after the fact, because changing a commit's parent changes its
id, and the held-out seed IS a function of L's id
(`agentlab.suite.generate.heldout_master_seed`). So a held-out set that verifies
here could not have been chosen before the prompt winner and the checkpoint were
published -- that is the only reason the numbers mean anything.

What it proves, in order, and refuses on:

  1. E resolves, and the two S18 receipts exist on disk.
  2. L is the UNIQUE commit reachable from E that adds the locks blob, is never
     edited afterwards, changes nothing but that blob, carries no reveal or
     held-out commitment of its own, and its committed bytes are the bytes on
     disk.
  3. R is the UNIQUE commit reachable from E that adds the reveal receipt, adds
     the held-out manifest and checksums in the SAME commit, adds nothing
     outside the permitted four paths, and its committed bytes for all three are
     the bytes on disk.
  4. The receipt rederives: the master seed is recomputed from the locks commit
     the receipt names, the release id is recomputed from that seed, and the
     receipt's locks commit must BE L and its locks_blob_sha256 must be L's
     committed lock. A receipt that names some other lock is a receipt for some
     other seed.
  5. P < L < R <= E, each leg strict where it must be, by `merge-base
     --is-ancestor` alone.
  6. The manifest is sealed, agrees with the receipt, and its commitment
     (SHA256SUMS.heldout) covers EXACTLY the held-out phase -- every payload and
     certspec file of the six held-out splits plus the manifest itself, and
     nothing belonging to train/dev.
  7. The held-out bytes on disk hash to their committed values, and no held-out
     file exists outside the release.
  8. The retired whole-suite commitment (manifest.json / SHA256SUMS) is absent.

No GPU, no network, no model, no torch import: it is safe to call as the first
statement of the evaluation stage, before a card is pinned. `--require-published`
additionally requires R to be published on the designated public ref, which is
read from the local remote-tracking ref and still performs no network I/O.

Exit 0 = proved. Exit 1 = refused, with every reason listed. Exit 2 = usage.

Where the call goes: see docs/HELDOUT_RELEASE_GATE.md. It is ONE line at the end
of the existing non-dry-run guard in `stage_eval()`, before the adapter path is
read out of locks.json and before `require_gpu`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

TOOL_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(TOOL_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT / "src"))

# The derivation is imported, never reimplemented: one authority for the master
# seed, the release id, the phase file names and the commitment format. A second
# copy of the seed derivation living in a gate would be a way for the gate and
# the generator to disagree about what was evaluated.
from agentlab.suite.generate import (  # noqa: E402
    HELDOUT_DERIVATION, HELDOUT_PHASE, HELDOUT_SPLITS, LEGACY_COMMITMENTS,
    PHASE_MANIFEST, PHASE_SUMS, TRAIN_DEV_PHASE, certspec_rels,
    heldout_release_id, load_reveal, read_sums, split_rels)

SUITE_DIR_REL = "data/suite/v1"
LOCKS_REL = "results/agentic/locks.json"
REVEAL_REL = "results/agentic/seed_reveal.json"
HELDOUT_MANIFEST_REL = f"{SUITE_DIR_REL}/{PHASE_MANIFEST[HELDOUT_PHASE]}"
HELDOUT_SUMS_REL = f"{SUITE_DIR_REL}/{PHASE_SUMS[HELDOUT_PHASE]}"
SUITE_RELEASE_REL = f"{SUITE_DIR_REL}/suite_release.json"
# Exactly what R may change. Three are mandatory; suite_release.json is the only
# rider the reveal state machine permits.
R_REQUIRED = (REVEAL_REL, HELDOUT_MANIFEST_REL, HELDOUT_SUMS_REL)
R_ALLOWED = R_REQUIRED + (SUITE_RELEASE_REL,)
# Written by the lock stage; may not be present at L.
L_FORBIDDEN_AT_L = R_REQUIRED

DEFAULT_PUBLIC_REF = "refs/remotes/origin/main"
STUDY_ID = "agentic-v1"


# ---------------------------------------------------------------------------
# git, read-only
# ---------------------------------------------------------------------------
# --no-optional-locks so this can be called while another process holds the
# repository: a gate must never be the thing that writes to the index.

class Git:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

    def _run(self, *args: str, text: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "--no-optional-locks", *args],
                              cwd=str(self.root), capture_output=True,
                              text=text, timeout=120)

    def out(self, *args: str) -> str:
        r = self._run(*args)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def ok(self, *args: str) -> bool:
        return self._run(*args).returncode == 0

    def raw(self, *args: str) -> bytes:
        r = self._run(*args, text=False)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed")
        return r.stdout

    def is_repo(self) -> bool:
        return self.ok("rev-parse", "--git-dir")

    def commit_of(self, ref: str) -> str | None:
        r = self._run("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        sha = r.stdout.strip()
        return sha if r.returncode == 0 and len(sha) == 40 else None

    def adding_commits(self, rel: str, *, since: str) -> list[str]:
        """Commits reachable from `since` that ADD `rel`, newest first.

        Reachable from `since`, not `--all`: a lock or a reveal sitting on a
        branch the evaluation is not running on proves nothing about the
        evaluation, and `--all` would silently accept exactly that.
        """
        out = self.out("log", "--format=%H", "--diff-filter=A", since, "--", rel)
        return [ln for ln in out.splitlines() if ln]

    def touching_commits(self, rel: str, *, since: str) -> list[str]:
        out = self.out("log", "--format=%H", since, "--", rel)
        return [ln for ln in out.splitlines() if ln]

    def changed_files(self, sha: str) -> list[str]:
        out = self.out("diff-tree", "--no-commit-id", "--name-only", "-r",
                       "--root", sha)
        return sorted(ln for ln in out.splitlines() if ln)

    def tree_has(self, sha: str, rel: str) -> bool:
        return self.ok("cat-file", "-e", f"{sha}:{rel}")

    def blob_sha256(self, sha: str, rel: str) -> str:
        return hashlib.sha256(self.raw("show", f"{sha}:{rel}")).hexdigest()

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.ok("merge-base", "--is-ancestor", a, b)

    def ref_exists(self, ref: str) -> bool:
        return self.ok("rev-parse", "--verify", "--quiet", ref)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short(sha: str | None) -> str:
    return sha[:12] if sha else "-"


# ---------------------------------------------------------------------------
# the required file list of the held-out commitment
# ---------------------------------------------------------------------------

def heldout_commitment_rels() -> tuple[str, ...]:
    """Every relative path SHA256SUMS.heldout must list, and no other.

    Eighteen payload files (kb/oracles/specs for the six held-out splits), the
    derived certspecs the evaluator actually reads, and the manifest itself.
    Derived from the generator's own constants, so a newly registered held-out
    split cannot slip out of the commitment without this list growing.
    """
    rels: list[str] = []
    for split in HELDOUT_SPLITS:
        rels.extend(split_rels(split))
    rels.extend(certspec_rels(HELDOUT_PHASE))
    rels.append(PHASE_MANIFEST[HELDOUT_PHASE])
    return tuple(sorted(rels))


# ---------------------------------------------------------------------------
# the verification itself
# ---------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.facts: dict = {"P": None, "L": None, "R": None, "E": None,
                            "heldout_release_id": None, "files_verified": 0,
                            "payload_checked": True, "publication_checked": False}

    def refuse(self, msg: str) -> None:
        self.problems.append(msg)

    @property
    def verified(self) -> bool:
        return not self.problems


def verify(root: pathlib.Path, *, head: str = "HEAD",
           suite_dir_rel: str = SUITE_DIR_REL,
           require_published: bool = False,
           public_ref: str = DEFAULT_PUBLIC_REF,
           commitments_only: bool = False,
           study_id: str = STUDY_ID) -> Report:
    rep = Report()
    rep.facts["payload_checked"] = not commitments_only
    git = Git(root)

    if not git.is_repo():
        rep.refuse(f"{root} is not a git repository, so there is no ancestry to "
                   f"read and the release cannot be proved at all")
        return rep

    E = git.commit_of(head)
    if E is None:
        rep.refuse(f"{head!r} does not resolve to a commit; E is the commit the "
                   f"evaluation runs at and it has to exist")
        return rep
    rep.facts["E"] = E

    locks_path = root / LOCKS_REL
    reveal_path = root / REVEAL_REL
    manifest_rel = f"{suite_dir_rel}/{PHASE_MANIFEST[HELDOUT_PHASE]}"
    sums_rel = f"{suite_dir_rel}/{PHASE_SUMS[HELDOUT_PHASE]}"
    manifest_path = root / manifest_rel
    sums_path = root / sums_rel
    r_required = (REVEAL_REL, manifest_rel, sums_rel)
    r_allowed = set(r_required) | {f"{suite_dir_rel}/suite_release.json"}

    if not locks_path.exists():
        rep.refuse(f"{LOCKS_REL} does not exist: there is no L, so the held-out "
                   f"seed is a function of nothing")
    if not reveal_path.exists():
        rep.refuse(f"{REVEAL_REL} does not exist: nothing is revealed, so there "
                   f"is no held-out release to evaluate")
    if rep.problems:
        return rep

    # -- L ------------------------------------------------------------------
    L = _resolve_dedicated(git, rep, LOCKS_REL, E, kind="L")
    if L is not None:
        changed = git.changed_files(L)
        if changed != [LOCKS_REL]:
            rep.refuse(f"L ({short(L)}) changes {changed}; L must be a DEDICATED "
                       f"commit that changes only {LOCKS_REL}. Anything riding "
                       f"along is material chosen at the same moment as the lock.")
        for rel in (REVEAL_REL, manifest_rel, sums_rel):
            if git.tree_has(L, rel):
                rep.refuse(f"L ({short(L)}) already carries {rel}: a lock commit "
                           f"holding a reveal or a held-out commitment is not a "
                           f"lock, it is the reveal")
        committed = git.blob_sha256(L, LOCKS_REL)
        if committed != sha256_file(locks_path):
            rep.refuse(f"{LOCKS_REL} on disk is not the bytes L committed: the "
                       f"seed derives from the PUBLISHED lock, and an edited copy "
                       f"is a different lock")
        rep.facts["L"] = L

    # -- R ------------------------------------------------------------------
    R = _resolve_dedicated(git, rep, REVEAL_REL, E, kind="R")
    if R is not None:
        changed = set(git.changed_files(R))
        missing = sorted(set(r_required) - changed)
        if missing:
            rep.refuse(
                f"R ({short(R)}) does not add {missing}. The receipt and the "
                f"held-out manifest and checksums must land in ONE commit: a "
                f"reveal without its commitment lets the evaluated bytes be "
                f"chosen after the seed is public.")
        stray = sorted(changed - r_allowed)
        if stray:
            rep.refuse(f"R ({short(R)}) also changes {stray}; the reveal commit "
                       f"may not touch the preregistration, the generator, the "
                       f"prompt, the lock, the docs or the evaluation code")
        for rel in r_required:
            if not git.tree_has(R, rel):
                continue
            path = root / rel
            if not path.exists():
                rep.refuse(f"{rel} is committed at R but missing on disk")
            elif git.blob_sha256(R, rel) != sha256_file(path):
                rep.refuse(f"{rel} on disk is not the bytes R committed; these "
                           f"are not the released held-out commitment")
        # Committed AT R, not merely present at R.
        for rel in (manifest_rel, sums_rel):
            adds = git.adding_commits(rel, since=E)
            if not adds:
                rep.refuse(f"no commit reachable from E adds {rel}: the held-out "
                           f"commitment is not committed, so R proves nothing "
                           f"about the bytes being evaluated")
            elif len(adds) > 1:
                rep.refuse(f"{len(adds)} commits add {rel}; the held-out "
                           f"commitment must be added exactly once, at R")
            elif adds[0] != R:
                rep.refuse(f"{rel} was added at {short(adds[0])}, not at R "
                           f"({short(R)}); the commitment and the reveal must be "
                           f"the same commit")
        rep.facts["R"] = R

    # -- the receipt rederives ---------------------------------------------
    receipt = None
    try:
        receipt = load_reveal(str(reveal_path), study_id=study_id)
    except SystemExit as exc:              # load_reveal refuses via SystemExit
        rep.refuse(f"the reveal receipt does not rederive: {exc}")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        rep.refuse(f"the reveal receipt is unreadable: {exc}")

    P = None
    if receipt is not None:
        rep.facts["heldout_release_id"] = receipt["heldout_release_id"]
        named_L = str(receipt["locks_commit"]).strip().lower()
        if L is not None and named_L != L:
            rep.refuse(
                f"the receipt derives its seed from locks commit "
                f"{short(named_L)} and the lock on this line of history is "
                f"{short(L)}. The seed is a function of L; a receipt naming "
                f"another commit reveals another suite.")
        if L is not None and named_L == L:
            claimed = str(receipt.get("locks_blob_sha256", "")).strip().lower()
            actual = git.blob_sha256(L, LOCKS_REL)
            if claimed != actual:
                rep.refuse(
                    f"the receipt's locks_blob_sha256 {claimed[:12] or '-'} is "
                    f"not L's committed lock {actual[:12]}: the receipt is for a "
                    f"different lock content than the commit it names")
        P = str(receipt["preregistration_commit"]).strip().lower()
        rep.facts["P"] = P
        try:
            locks_doc = json.loads(locks_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            locks_doc = None
            rep.refuse(f"{LOCKS_REL} is unreadable: {exc}")
        if isinstance(locks_doc, dict):
            lock_P = str(locks_doc.get("preregistration_commit", "")).strip().lower()
            if lock_P != P:
                rep.refuse(f"the lock names P {short(lock_P) or '-'} and the "
                           f"receipt names P {short(P)}; they must be the same "
                           f"preregistration")
        if git.commit_of(P) is None:
            rep.refuse(f"P ({short(P)}) is not a commit in this repository, so "
                       f"P < L cannot be read")
            P = None

    # -- P < L < R <= E, by ancestry only ---------------------------------
    if P is not None and L is not None:
        if P == L:
            rep.refuse(f"P and L are the same commit ({short(L)}): a lock that IS "
                       f"the preregistration was not made after it")
        elif not git.is_ancestor(P, L):
            rep.refuse(f"P ({short(P)}) is not an ancestor of L ({short(L)}): "
                       f"P < L is what makes the lock post-preregistration")
    if L is not None and R is not None:
        if L == R:
            rep.refuse(f"L and R are the same commit ({short(R)}): the reveal must "
                       f"be a strict descendant of the lock it derives from")
        elif not git.is_ancestor(L, R):
            rep.refuse(f"L ({short(L)}) is not an ancestor of R ({short(R)}): "
                       f"L < R is what makes the held-out set post-lock")
    if R is not None and not git.is_ancestor(R, E):
        rep.refuse(f"R ({short(R)}) is not an ancestor of E ({short(E)}): the "
                   f"evaluation would run at a commit that does not contain the "
                   f"reveal, so R <= E does not hold")

    # -- the retired whole-suite commitment -------------------------------
    for legacy in LEGACY_COMMITMENTS:
        rel = f"{suite_dir_rel}/{legacy}"
        if (root / rel).exists():
            rep.refuse(f"{rel} is present: that is the RETIRED whole-suite "
                       f"commitment, which pinned held-out hashes at a commit "
                       f"where the held-out bytes must not have existed")
        if R is not None and git.tree_has(R, rel):
            rep.refuse(f"R carries {rel}, the retired whole-suite commitment")

    # -- the manifest and its commitment ----------------------------------
    if not manifest_path.exists():
        rep.refuse(f"{manifest_rel} is absent: the release is not pinned")
        return rep
    if not sums_path.exists():
        rep.refuse(f"{sums_rel} is absent: {manifest_rel} exists but nothing "
                   f"commits its bytes")
        return rep
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        rep.refuse(f"{manifest_rel} is unreadable: {exc}")
        return rep
    if not isinstance(manifest, dict):
        rep.refuse(f"{manifest_rel} is not a manifest object")
        return rep

    if manifest.get("phase") != HELDOUT_PHASE:
        rep.refuse(f"{manifest_rel} declares phase {manifest.get('phase')!r}, "
                   f"not {HELDOUT_PHASE!r}")
    if manifest.get("sealed") is not True:
        rep.refuse(f"{manifest_rel} is UNSEALED: the certspecs the evaluator "
                   f"actually reads are not pinned by it, so 'the suite is "
                   f"pinned' is not true of the bytes being evaluated")
    if manifest.get("derivation_label") != HELDOUT_DERIVATION:
        rep.refuse(f"{manifest_rel} derivation_label "
                   f"{manifest.get('derivation_label')!r} != "
                   f"{HELDOUT_DERIVATION!r}")
    master_hex = str(manifest.get("master_seed_hex", "")).strip().lower()
    try:
        rederived = heldout_release_id(bytes.fromhex(master_hex))
    except (ValueError, TypeError):
        rederived = None
        rep.refuse(f"{manifest_rel} carries no usable master seed, so its release "
                   f"id rederives from nothing")
    if rederived is not None and manifest.get("heldout_release_id") != rederived:
        rep.refuse(f"{manifest_rel}: release id "
                   f"{str(manifest.get('heldout_release_id'))[:12]} does not "
                   f"rederive from its own master seed ({rederived[:12]})")
    if receipt is not None:
        if master_hex != receipt["master_seed"].hex():
            rep.refuse(f"{manifest_rel} was generated from master seed "
                       f"{master_hex[:12] or '-'} and the receipt reveals "
                       f"{receipt['master_seed'].hex()[:12]}: the released bytes "
                       f"are not the revealed seed's bytes")
        if manifest.get("heldout_release_id") != receipt["heldout_release_id"]:
            rep.refuse(f"{manifest_rel} release id is not the receipt's")
        if str(manifest.get("locks_commit", "")).strip().lower() != \
                str(receipt["locks_commit"]).strip().lower():
            rep.refuse(f"{manifest_rel} locks_commit is not the receipt's")
        if str(manifest.get("preregistration_commit", "")).strip().lower() != \
                str(receipt["preregistration_commit"]).strip().lower():
            rep.refuse(f"{manifest_rel} preregistration_commit is not the "
                       f"receipt's")
        if "reveal_sha256" in manifest:
            on_disk = sha256_file(reveal_path)
            if str(manifest["reveal_sha256"]).strip().lower() != on_disk:
                rep.refuse(f"{manifest_rel} was sealed against reveal "
                           f"{str(manifest['reveal_sha256'])[:12]} and the receipt "
                           f"on disk hashes {on_disk[:12]}")

    # -- SHA256SUMS.heldout covers exactly the held-out phase --------------
    try:
        listed = read_sums(str(sums_path))
    except (ValueError, OSError) as exc:
        rep.refuse(f"{sums_rel} is not a usable checksum file: {exc}")
        return rep

    required = set(heldout_commitment_rels())
    missing = sorted(required - set(listed))
    if missing:
        rep.refuse(f"{sums_rel} does not cover {len(missing)} held-out file(s), "
                   f"e.g. {missing[:3]}: a held-out file outside the release "
                   f"commitment is a cached early value, not evidence")
    extra = sorted(set(listed) - required)
    if extra:
        rep.refuse(f"{sums_rel} lists {extra[:3]} ({len(extra)} file(s)) that are "
                   f"not part of the held-out phase; the two phases have two "
                   f"commitments and neither may seal the other's bytes")
    for rel in (PHASE_MANIFEST[TRAIN_DEV_PHASE], PHASE_SUMS[TRAIN_DEV_PHASE]):
        if rel in listed:
            rep.refuse(f"{sums_rel} pins {rel}, the train/dev commitment")

    declared: dict[str, str] = {}
    for block in ("files", "certspecs"):
        entries = manifest.get(block)
        if not isinstance(entries, dict):
            rep.refuse(f"{manifest_rel} has no {block!r} block; it is not a "
                       f"sealed release manifest")
            continue
        for rel, meta in entries.items():
            if isinstance(meta, dict) and "sha256" in meta:
                declared[rel] = str(meta["sha256"]).lower()
    disagree = sorted(rel for rel, digest in declared.items()
                      if listed.get(rel) != digest)
    if disagree:
        rep.refuse(f"{manifest_rel} and {sums_rel} disagree about "
                   f"{len(disagree)} file(s), e.g. {disagree[:3]}: the manifest "
                   f"and the checksum file are one commitment and must agree")
    self_digest = listed.get(PHASE_MANIFEST[HELDOUT_PHASE])
    if self_digest is not None and self_digest != sha256_file(manifest_path):
        rep.refuse(f"{sums_rel} pins {PHASE_MANIFEST[HELDOUT_PHASE]} at "
                   f"{self_digest[:12]} and it hashes "
                   f"{sha256_file(manifest_path)[:12]}: the manifest was edited "
                   f"after it was sealed")

    # -- the bytes on disk -------------------------------------------------
    if commitments_only:
        rep.facts["files_verified"] = 0
    else:
        absent: list[str] = []
        wrong: list[str] = []
        checked = 0
        for rel, digest in sorted(listed.items()):
            path = root / suite_dir_rel / rel
            if not path.exists():
                absent.append(rel)
                continue
            if sha256_file(path) != digest:
                wrong.append(rel)
            else:
                checked += 1
        rep.facts["files_verified"] = checked
        if absent:
            rep.refuse(f"{len(absent)} released file(s) are missing on disk, e.g. "
                       f"{absent[:3]}: regenerate the held-out phase from the "
                       f"reveal, then run this gate again")
        if wrong:
            rep.refuse(f"{len(wrong)} released file(s) do not hash to their "
                       f"committed value, e.g. {wrong[:3]}: these bytes are not "
                       f"the revealed held-out set")

    # -- publication --------------------------------------------------------
    if require_published:
        rep.facts["publication_checked"] = True
        if not git.ref_exists(public_ref):
            rep.refuse(f"{public_ref} does not exist here, so R's publication "
                       f"cannot be verified locally")
        elif R is not None and not git.is_ancestor(R, public_ref):
            rep.refuse(f"R ({short(R)}) is not published on {public_ref}; the "
                       f"evaluation waits until it is")
    return rep


def _resolve_dedicated(git: Git, rep: Report, rel: str, E: str, *,
                       kind: str) -> str | None:
    """The unique commit reachable from E that adds `rel`, or a refusal.

    Uniqueness and reachability are both load-bearing. Two adding commits mean
    the receipt cannot say which one the seed came from; an adding commit that E
    cannot reach means the evaluation is not running on the history that contains
    it, and `P < L < R <= E` is false however good the file looks.
    """
    adds = git.adding_commits(rel, since=E)
    if not adds:
        elsewhere = [ln for ln in
                     git.out("log", "--all", "--format=%H", "--diff-filter=A",
                             "--", rel).splitlines() if ln]
        if elsewhere:
            rep.refuse(
                f"{kind}: {rel} is added at {short(elsewhere[0])}, which is NOT "
                f"reachable from E ({short(E)}). The commit exists on some other "
                f"line of history, so it establishes no ordering for this "
                f"evaluation: ancestry is the proof, not the presence of a file.")
        else:
            rep.refuse(
                f"{kind}: no commit adds {rel}. It exists on disk with no commit "
                f"behind it, so it has no sha to order and nothing to derive "
                f"from. Commit it in its own dedicated commit.")
        return None
    if len(adds) > 1:
        rep.refuse(f"{kind}: {len(adds)} commits add {rel} "
                   f"({[short(c) for c in adds]}); {kind} must be unique")
        return None
    sha = adds[0]
    touching = git.touching_commits(rel, since=E)
    if touching != [sha]:
        rep.refuse(f"{kind}: {len(touching)} commits change {rel}; a later edit "
                   f"means the published receipt is not the receipt in the tree")
    return sha


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="verify_heldout_release.py",
        description="Refuse evaluation unless P < L < R <= E is proved by git "
                    "ancestry and the held-out bytes verify against the "
                    "commitment R published.")
    ap.add_argument("--repo", default=str(TOOL_ROOT),
                    help="repository to verify (default: this tool's repository)")
    ap.add_argument("--head", default="HEAD",
                    help="E, the commit the evaluation would run at "
                         "(default: HEAD)")
    ap.add_argument("--suite-dir", default=SUITE_DIR_REL,
                    help=f"suite directory, repo-relative (default: "
                         f"{SUITE_DIR_REL})")
    ap.add_argument("--require-published", action="store_true",
                    help="also require R to be an ancestor of the designated "
                         "public ref (read locally; no network)")
    ap.add_argument("--public-ref", default=DEFAULT_PUBLIC_REF,
                    help=f"the designated public ref (default: "
                         f"{DEFAULT_PUBLIC_REF})")
    ap.add_argument("--commitments-only", action="store_true",
                    help="verify ancestry and the commitment without hashing the "
                         "payload. NOT sufficient as the evaluation gate.")
    ap.add_argument("--study-id", default=STUDY_ID)
    ap.add_argument("--json", action="store_true",
                    help="emit the verdict as JSON on stdout")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.repo).resolve()
    if not root.is_dir():
        print(f"REFUSED: {root} is not a directory", file=sys.stderr)
        return 2

    rep = verify(root, head=args.head, suite_dir_rel=args.suite_dir,
                 require_published=args.require_published,
                 public_ref=args.public_ref,
                 commitments_only=args.commitments_only,
                 study_id=args.study_id)

    if args.json:
        print(json.dumps({"verified": rep.verified, "problems": rep.problems,
                          **rep.facts}, indent=2, sort_keys=True))
        return 0 if rep.verified else 1

    if not rep.verified:
        print("REFUSED: the held-out release is not provable. No GPU should be "
              "allocated.", file=sys.stderr)
        for p in rep.problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"  (order is read from git ancestry only; no timestamp was "
              f"consulted. E = {short(rep.facts['E'])})", file=sys.stderr)
        return 1

    f = rep.facts
    head = ("held-out release VERIFIED" if f["payload_checked"]
            else "held-out COMMITMENT verified (payload NOT hashed -- this is "
                 "not the evaluation gate)")
    print(f"{head}: P {short(f['P'])} < L {short(f['L'])} < R {short(f['R'])} "
          f"<= E {short(f['E'])}")
    print(f"  proved by git ancestry alone; no timestamp was consulted")
    print(f"  release id {f['heldout_release_id']}")
    if f["payload_checked"]:
        print(f"  {f['files_verified']} released file(s) hash to their committed "
              f"values")
    if f["publication_checked"]:
        print(f"  R is published on the designated public ref")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
