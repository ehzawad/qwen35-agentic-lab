"""R's file list, constructed from a clean checkout, must contain the held-out
commitment.

The hole this closes was an accident that looked like a rule. `.gitignore` named
`data/suite/v1/manifest.json` and `data/suite/v1/SHA256SUMS` as its two
exceptions -- the RETIRED whole-suite commitment, which
`agentlab.suite.generate.LEGACY_COMMITMENTS` now refuses on sight -- and named
neither of the live per-phase files. The live ones were not ignored either,
because `data/*` does not descend into `data/suite/v1/`, and the train/dev pair
was additionally safe because it is already tracked. So the intended allowlist
protected two files that must never exist and said nothing about the two that R
absolutely must carry.

That is not a protection. It is one `data/**` away from dropping
`manifest.heldout.json` and `SHA256SUMS.heldout` out of R with no error anywhere:
`git add -A` would simply not stage them, the reveal commit would carry the
receipt alone, and the evaluated bytes would be pinned by nothing.

These tests read the `.gitignore` in the working tree -- the artifact under test
-- and rebuild a clean checkout around it in a temporary repository. Nothing here
consults this repository's own P/L/R state, its HEAD, or its index, so the tests
say the same thing before and after the commit that introduces them.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agentlab.suite.generate import (HELDOUT_PHASE, HELDOUT_SPLITS,
                                     LEGACY_COMMITMENTS, PHASE_MANIFEST,
                                     PHASE_SUMS, TRAIN_DEV_PHASE, certspec_rels,
                                     split_rels)

REPO = pathlib.Path(__file__).resolve().parents[1]
GITIGNORE = REPO / ".gitignore"
SUITE_DIR = "data/suite/v1"

# The four paths the reveal state machine is allowed to add at R. Three are
# mandatory; suite_release.json is the only permitted rider.
R_FILES = (
    "results/agentic/seed_reveal.json",
    f"{SUITE_DIR}/{PHASE_MANIFEST[HELDOUT_PHASE]}",
    f"{SUITE_DIR}/{PHASE_SUMS[HELDOUT_PHASE]}",
    f"{SUITE_DIR}/suite_release.json",
)
# The two that carry the commitment. If either is missing from R, the held-out
# bytes are pinned by nothing.
R_COMMITMENT = (f"{SUITE_DIR}/{PHASE_MANIFEST[HELDOUT_PHASE]}",
                f"{SUITE_DIR}/{PHASE_SUMS[HELDOUT_PHASE]}")


def _git(root: pathlib.Path, *args: str) -> str:
    r = subprocess.run(["git", "--no-optional-locks", *args], cwd=str(root),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr}"
    return r.stdout


def _write(root: pathlib.Path, rel: str, text: str = "x\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def clean_checkout(tmp_path: pathlib.Path) -> pathlib.Path:
    """A fresh repository carrying this repository's `.gitignore` and the
    train/dev commitment that a clean checkout of P already tracks.

    The held-out files are deliberately NOT seeded here: in R they are brand-new
    untracked files, and the point is that a clean checkout stages them anyway
    rather than relying on their being tracked already.
    """
    root = tmp_path / "clone"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / ".gitignore").write_bytes(GITIGNORE.read_bytes())
    _write(root, f"{SUITE_DIR}/{PHASE_MANIFEST[TRAIN_DEV_PHASE]}", "{}\n")
    _write(root, f"{SUITE_DIR}/{PHASE_SUMS[TRAIN_DEV_PHASE]}", "")
    _git(root, "add", "--", ".gitignore",
         f"{SUITE_DIR}/{PHASE_MANIFEST[TRAIN_DEV_PHASE]}",
         f"{SUITE_DIR}/{PHASE_SUMS[TRAIN_DEV_PHASE]}")
    _git(root, "commit", "-q", "-m", "P: train/dev commitment")
    return root


def materialize_R(root: pathlib.Path) -> None:
    """Everything the reveal stage writes: the four R files, the whole held-out
    payload it generates alongside them, and the retired commitment a stale tree
    might still carry."""
    for rel in R_FILES:
        _write(root, rel)
    for split in HELDOUT_SPLITS:
        for rel in split_rels(split):
            _write(root, f"{SUITE_DIR}/{rel}")
    for rel in certspec_rels(HELDOUT_PHASE):
        _write(root, f"{SUITE_DIR}/{rel}")
    for legacy in LEGACY_COMMITMENTS:
        _write(root, f"{SUITE_DIR}/{legacy}")


def staged_file_list(root: pathlib.Path) -> list[str]:
    """R's file list: exactly what `git add -A` would put in the reveal commit."""
    _git(root, "add", "-A")
    return sorted(ln for ln in
                  _git(root, "diff", "--cached", "--name-only").splitlines() if ln)


# ---------------------------------------------------------------------------
# the P0 itself
# ---------------------------------------------------------------------------

def test_R_file_list_from_a_clean_checkout_carries_the_heldout_commitment(tmp_path):
    root = clean_checkout(tmp_path)
    materialize_R(root)
    listed = staged_file_list(root)
    missing = [rel for rel in R_COMMITMENT if rel not in listed]
    assert not missing, (
        f"R would be committed WITHOUT {missing}. The reveal commit must carry "
        f"the held-out manifest and its checksums, or the evaluated bytes are "
        f"pinned by nothing. Staged: {listed}")


@pytest.mark.parametrize("rel", R_COMMITMENT)
def test_each_heldout_commitment_file_is_individually_permitted(tmp_path, rel):
    """One assertion per file, so a failure names the file that went missing."""
    root = clean_checkout(tmp_path)
    _write(root, rel)
    assert rel in staged_file_list(root), f"{rel} is ignored in a clean checkout"


def test_the_heldout_commitment_is_not_merely_surviving_by_being_tracked(tmp_path):
    """The train/dev pair survives because it is tracked; that must not be the
    mechanism protecting the held-out pair, which cannot be tracked before R."""
    root = clean_checkout(tmp_path)
    tracked = set(_git(root, "ls-files").split())
    for rel in R_COMMITMENT:
        assert rel not in tracked, f"{rel} must not exist before R"
    materialize_R(root)
    listed = staged_file_list(root)
    for rel in R_COMMITMENT:
        assert rel in listed


def test_R_stages_nothing_beyond_the_four_permitted_paths(tmp_path):
    """Deny-by-default cuts both ways: the payload the reveal stage generates is
    44 MB of regenerable bytes and the retired whole-suite commitment is refused
    on sight, so neither may ride along into R."""
    root = clean_checkout(tmp_path)
    materialize_R(root)
    listed = staged_file_list(root)
    assert sorted(listed) == sorted(R_FILES), (
        f"R would stage {sorted(set(listed) - set(R_FILES))} beyond the four "
        f"permitted paths")


@pytest.mark.parametrize("legacy", LEGACY_COMMITMENTS)
def test_the_retired_whole_suite_commitment_stays_ignored(tmp_path, legacy):
    root = clean_checkout(tmp_path)
    rel = f"{SUITE_DIR}/{legacy}"
    _write(root, rel)
    assert rel not in staged_file_list(root), (
        f"{rel} is the retired whole-suite commitment; a tree carrying it is a "
        f"tree pinned by a commitment that disclosed held-out hashes")


def test_the_heldout_payload_stays_out_of_git(tmp_path):
    root = clean_checkout(tmp_path)
    materialize_R(root)
    listed = set(staged_file_list(root))
    payload = [f"{SUITE_DIR}/{rel}" for split in HELDOUT_SPLITS
               for rel in split_rels(split)]
    payload += [f"{SUITE_DIR}/{rel}" for rel in certspec_rels(HELDOUT_PHASE)]
    leaked = sorted(rel for rel in payload if rel in listed)
    assert not leaked, f"held-out payload would be committed: {leaked[:3]}"


def test_the_producers_output_is_still_ignored(tmp_path):
    """The .gitignore change must not start tracking what the running study
    writes: the rejection-sampling corpus and everything under out/."""
    root = clean_checkout(tmp_path)
    _write(root, "data/multiface/accepted.jsonl")
    _write(root, "data/multiface/raw.shard0.jsonl")
    _write(root, "out/multiface/rs_sft/adapter_model.safetensors")
    _write(root, "data/distill.jsonl")
    listed = set(staged_file_list(root))
    for rel in ("data/multiface/accepted.jsonl", "data/multiface/raw.shard0.jsonl",
                "out/multiface/rs_sft/adapter_model.safetensors"):
        assert rel not in listed, f"{rel} must stay out of git"
    assert "data/distill.jsonl" in listed, "the committed seed corpus is tracked"


def test_the_gitignore_names_the_live_commitment_files_explicitly(tmp_path):
    """Not just behaviour: the file has to SAY it, so the next editor of this
    block cannot delete the protection by accident."""
    text = GITIGNORE.read_text(encoding="utf-8")
    for rel in R_COMMITMENT:
        assert f"!{rel}" in text, f"{rel} is permitted only by accident"
    # and the train/dev pair the study already committed at P
    assert f"!{SUITE_DIR}/{PHASE_MANIFEST[TRAIN_DEV_PHASE]}" in text
    assert f"!{SUITE_DIR}/{PHASE_SUMS[TRAIN_DEV_PHASE]}" in text


def test_this_repository_tracks_the_train_dev_commitment(tmp_path):
    """The one fact the clean-checkout fixture asserts about the real tree: the
    train/dev commitment is committed, which is why R only has to add the
    held-out one."""
    tracked = set(_git(REPO, "ls-files", "--", SUITE_DIR).split())
    for rel in (f"{SUITE_DIR}/{PHASE_MANIFEST[TRAIN_DEV_PHASE]}",
                f"{SUITE_DIR}/{PHASE_SUMS[TRAIN_DEV_PHASE]}"):
        assert rel in tracked, f"{rel} is not tracked"
