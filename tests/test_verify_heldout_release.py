"""`scripts/verify_heldout_release.py`: the gate that runs before any evaluation
GPU is allocated.

Every test here builds a whole synthetic study in a temporary git repository --
its own P, its own L, its own R, its own held-out release -- and runs the real
tool against it with `--repo`. Nothing reads this repository's P/L/R state, so
these tests keep saying the same thing while the live study advances underneath
them, and they can construct the failures the live repository must never be put
into.

The four cases the task names are `test_a_well_formed_release_is_accepted`,
`test_a_reveal_committed_before_the_lock_is_refused`,
`test_a_reveal_without_a_committed_lock_is_refused` and the three
`test_..._digest_...` cases. The rest are the neighbouring holes.

One note on what CANNOT be tested, because it is the mechanism working. There is
no way to build a repository in which R is literally an earlier commit than L and
the receipt at R still rederives: the receipt's seed is
`sha256(label || L)`, and L's own id depends on its parent, so a receipt naming L
cannot be committed before L exists without predicting a sha that depends on the
commit containing the prediction. So "R before L" always shows up as a receipt
naming some OTHER lock, or as a lock that is not an ancestor of R, or both --
which is exactly what the two ordering tests assert.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

import pytest

from agentlab.suite.generate import (GENERATION_PROTOCOL, HELDOUT_DERIVATION,
                                     HELDOUT_PHASE, HELDOUT_SPLITS,
                                     LEGACY_COMMITMENTS, PHASE_MANIFEST,
                                     PHASE_SUMS, REVEAL_SCHEMA, certspec_rels,
                                     heldout_master_seed, heldout_release_id,
                                     split_rels)

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
TOOL = REPO / "scripts" / "verify_heldout_release.py"
SUITE_DIR = "data/suite/v1"
LOCKS_REL = "results/agentic/locks.json"
REVEAL_REL = "results/agentic/seed_reveal.json"
MANIFEST_REL = f"{SUITE_DIR}/{PHASE_MANIFEST[HELDOUT_PHASE]}"
SUMS_REL = f"{SUITE_DIR}/{PHASE_SUMS[HELDOUT_PHASE]}"
STUDY_ID = "agentic-v1"
# A 40-hex string that is a valid commit id shape and is not a commit here. Used
# where a receipt has to name a lock the repository does not have.
FOREIGN_L = "b" * 40


# ---------------------------------------------------------------------------
# building a synthetic study
# ---------------------------------------------------------------------------

def _git(root: pathlib.Path, *args: str) -> str:
    r = subprocess.run(["git", "--no-optional-locks", *args], cwd=str(root),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr}"
    return r.stdout.strip()


def _write(root: pathlib.Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _sha256(root: pathlib.Path, rel: str) -> str:
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def _commit(root: pathlib.Path, rels, msg: str) -> str:
    _git(root, "add", "--", *rels)
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


def _init(tmp_path: pathlib.Path, name: str = "repo") -> pathlib.Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    return root


def _locks_text(P: str) -> str:
    return json.dumps({
        "schema": "agentic-locks-v2", "study_id": STUDY_ID,
        "preregistration_commit": P,
        "prompt_winner": {"file": "prompts/agentic/p2_plan_state_act.txt",
                          "sha256": "5" * 64},
        "checkpoint": {"path": "out/multiface/rs_sft", "sha256": "6" * 64},
        "selection": {"sole_candidate": "rs_sft"},
    }, indent=2, sort_keys=True) + "\n"


def _payload_rels() -> list[str]:
    rels: list[str] = []
    for split in HELDOUT_SPLITS:
        rels.extend(split_rels(split))
    return rels


def _reveal_text(*, P: str, L: str, locks_blob: str) -> str:
    master = heldout_master_seed(L)
    return json.dumps({
        "schema": REVEAL_SCHEMA, "study_id": STUDY_ID,
        "preregistration_commit": P, "locks_commit": L,
        "locks_blob_sha256": locks_blob, "public_ref": "refs/heads/main",
        "master_seed_hex": master.hex(),
        "heldout_release_id": heldout_release_id(master),
        "derivation_label": HELDOUT_DERIVATION,
        "generation_protocol": GENERATION_PROTOCOL,
        "generator_commit": "0" * 40, "revealed_at": "2026-08-06T00:00:00Z",
    }, indent=2, sort_keys=True) + "\n"


def _seal(root: pathlib.Path, *, P: str, L: str, master: bytes, sealed: bool,
          sums_omit=(), sums_extra=None) -> None:
    """Write the held-out payload, manifest and checksums the way
    `agentlab.suite.generate._write_commitment` writes them: the manifest is
    hashed into its own SHA256SUMS last, so an edit after sealing is visible."""
    rid = heldout_release_id(master)
    files, certs = {}, {}
    for rel in _payload_rels():
        _write(root, f"{SUITE_DIR}/{rel}", f"{rel}\n{rid}\n")
        files[rel] = {"sha256": _sha256(root, f"{SUITE_DIR}/{rel}"),
                      "bytes": (root / SUITE_DIR / rel).stat().st_size}
    for rel in certspec_rels(HELDOUT_PHASE):
        _write(root, f"{SUITE_DIR}/{rel}", f"{rel}\n{rid}\n")
        certs[rel] = {"sha256": _sha256(root, f"{SUITE_DIR}/{rel}"),
                      "bytes": (root / SUITE_DIR / rel).stat().st_size}
    manifest = {
        "suite": "agentlab-suite-v1", "version": "1.0.0",
        "generator": "agentlab.suite.generate",
        "generation_protocol": GENERATION_PROTOCOL,
        "phase": HELDOUT_PHASE, "files": files,
        "certspecs": certs if sealed else "PENDING_EXPORT",
        "sealed": bool(sealed),
        "heldout_release_id": rid, "master_seed_hex": master.hex(),
        "derivation_label": HELDOUT_DERIVATION,
        "locks_commit": L, "preregistration_commit": P,
        "reveal_sha256": _sha256(root, REVEAL_REL),
    }
    _write(root, MANIFEST_REL,
           json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    listed = {rel: meta["sha256"] for rel, meta in files.items()}
    if sealed:
        listed.update({rel: meta["sha256"] for rel, meta in certs.items()})
    listed[PHASE_MANIFEST[HELDOUT_PHASE]] = _sha256(root, MANIFEST_REL)
    for rel in sums_omit:
        listed.pop(rel, None)
    if sums_extra:
        listed.update(sums_extra)
    _write(root, SUMS_REL,
           "".join(f"{d}  {r}\n" for r, d in sorted(listed.items())))


def build(tmp_path, *, extra_in_L=None, extra_in_R=None, r_omits=(),
          sealed=True, sums_omit=(), sums_extra=None, commit_locks=True,
          receipt_locks=None, legacy=False) -> dict:
    """A whole synthetic study: P, then a dedicated L, then a dedicated R.

    Every knob exists because it is the shape of one real failure. Defaults give
    a well-formed release.
    """
    root = _init(tmp_path)
    _write(root, "configs/agentic_preregister.json", '{"study_id": "agentic-v1"}\n')
    P = _commit(root, ["configs/agentic_preregister.json"], "P: preregistration")

    _write(root, LOCKS_REL, _locks_text(P))
    L = None
    if commit_locks:
        rels = [LOCKS_REL]
        if extra_in_L:
            for rel in extra_in_L:
                _write(root, rel, "rider\n")
            rels += list(extra_in_L)
        L = _commit(root, rels, "L: lock the prompt winner and the checkpoint")

    named_L = receipt_locks or L or FOREIGN_L
    locks_blob = _sha256(root, LOCKS_REL)
    _write(root, REVEAL_REL,
           _reveal_text(P=P, L=named_L, locks_blob=locks_blob))
    _seal(root, P=P, L=named_L, master=heldout_master_seed(named_L),
          sealed=sealed, sums_omit=sums_omit, sums_extra=sums_extra)
    if legacy:
        for name in LEGACY_COMMITMENTS:
            _write(root, f"{SUITE_DIR}/{name}", "retired\n")

    rels = [r for r in (REVEAL_REL, MANIFEST_REL, SUMS_REL) if r not in r_omits]
    if extra_in_R:
        for rel in extra_in_R:
            _write(root, rel, "rider\n")
        rels += list(extra_in_R)
    R = _commit(root, rels, "R: reveal the held-out release")
    return {"root": root, "P": P, "L": L, "R": R, "named_L": named_L}


def run(root: pathlib.Path, *args: str):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([str(PY), str(TOOL), "--repo", str(root), *args],
                          capture_output=True, text=True, timeout=300, env=env)


def refusal(root: pathlib.Path, *args: str) -> str:
    r = run(root, *args)
    assert r.returncode == 1, (
        f"expected a refusal, got {r.returncode}\n{r.stdout}\n{r.stderr}")
    assert "REFUSED" in r.stderr
    return r.stderr


# ---------------------------------------------------------------------------
# 1. a well-formed release is accepted
# ---------------------------------------------------------------------------

def test_a_well_formed_release_is_accepted(tmp_path):
    st = build(tmp_path)
    r = run(st["root"])
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "held-out release VERIFIED" in r.stdout
    for sha in (st["P"], st["L"], st["R"]):
        assert sha[:12] in r.stdout
    assert "no timestamp was consulted" in r.stdout


def test_the_accepted_verdict_reports_the_whole_commitment(tmp_path):
    st = build(tmp_path)
    r = run(st["root"], "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["verified"] is True and out["problems"] == []
    assert (out["P"], out["L"], out["R"]) == (st["P"], st["L"], st["R"])
    assert out["E"] == st["R"], "E is R when the evaluation runs at the reveal"
    master = heldout_master_seed(st["L"])
    assert out["heldout_release_id"] == heldout_release_id(master)
    # 18 payload + 7 certspecs + the manifest itself
    assert out["files_verified"] == len(_payload_rels()) \
        + len(certspec_rels(HELDOUT_PHASE)) + 1


def test_a_release_written_by_the_real_generator_is_accepted(tmp_path):
    """The one test that is not hand-built.

    Everything above seals a synthetic commitment, which proves the gate's logic
    and nothing about whether the generator writes that shape. Here the real
    `generate_phase` / `export_eval_specs.py` / `seal_phase` produce a tiny
    held-out release from a receipt naming this repository's actual L, and the
    gate has to accept it. If the manifest shape ever drifts -- a renamed field, a
    dropped `sealed`, a certspec that stops being covered -- this fails instead of
    the gate quietly passing something it no longer understands.
    """
    from agentlab.suite.generate import (PHASES, generate_phase, load_reveal,
                                         load_suite_config, seal_phase)

    root = _init(tmp_path)
    _write(root, "configs/agentic_preregister.json", '{"study_id": "agentic-v1"}\n')
    P = _commit(root, ["configs/agentic_preregister.json"], "P: preregistration")
    _write(root, LOCKS_REL, _locks_text(P))
    L = _commit(root, [LOCKS_REL], "L: lock")

    _write(root, REVEAL_REL,
           _reveal_text(P=P, L=L, locks_blob=_sha256(root, LOCKS_REL)))
    reveal = load_reveal(str(root / REVEAL_REL))
    cfg = load_suite_config(str(REPO / "configs" / "suite_v1.toml"))
    cfg = dict(cfg, sizes={k: 2 for k in cfg["sizes"]})   # two per cell: a test
    out = root / SUITE_DIR
    generate_phase(cfg, str(out), HELDOUT_PHASE, reveal=reveal)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([str(PY), str(REPO / "scripts" / "export_eval_specs.py"),
                        "--data", str(out), "--splits", *PHASES[HELDOUT_PHASE]],
                       capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    seal_phase(cfg, str(out), HELDOUT_PHASE, reveal=reveal)

    R = _commit(root, [REVEAL_REL, MANIFEST_REL, SUMS_REL], "R: reveal")
    got = run(root)
    assert got.returncode == 0, f"{got.stdout}\n{got.stderr}"
    assert "held-out release VERIFIED" in got.stdout
    assert R[:12] in got.stdout
    # 18 payload + 7 certspecs + the manifest itself, the same 26 the generator's
    # own commitment test counts
    assert "26 released file(s)" in got.stdout


def test_the_gate_touches_no_gpu_and_no_network(tmp_path):
    """It is called before a card is pinned, so it must not import the stack that
    needs one."""
    text = TOOL.read_text(encoding="utf-8")
    for banned in ("import torch", "import vllm", "requests", "urllib.request",
                   "nvidia-smi", "huggingface_hub"):
        assert banned not in text, f"the gate must not reach for {banned}"


# ---------------------------------------------------------------------------
# 2. ordering: R before L
# ---------------------------------------------------------------------------

def test_a_reveal_committed_before_the_lock_is_refused(tmp_path):
    """R is a genuine ancestor of the commit that adds the lock.

    Linear history: P -> R -> L. The receipt at R therefore cannot name L (see
    the module docstring), so it names a foreign lock; both the ordering and the
    naming are refused, and the ordering refusal is the one that matters.
    """
    root = _init(tmp_path)
    _write(root, "configs/agentic_preregister.json", '{"study_id": "agentic-v1"}\n')
    P = _commit(root, ["configs/agentic_preregister.json"], "P: preregistration")
    _write(root, LOCKS_REL, _locks_text(P))
    locks_blob = _sha256(root, LOCKS_REL)
    _write(root, REVEAL_REL,
           _reveal_text(P=P, L=FOREIGN_L, locks_blob=locks_blob))
    _seal(root, P=P, L=FOREIGN_L, master=heldout_master_seed(FOREIGN_L),
          sealed=True)
    R = _commit(root, [REVEAL_REL, MANIFEST_REL, SUMS_REL], "R: reveal")
    L = _commit(root, [LOCKS_REL], "L: lock, too late")

    err = refusal(root)
    assert f"L ({L[:12]}) is not an ancestor of R ({R[:12]})" in err
    assert "L < R is what makes the held-out set post-lock" in err


def test_a_lock_on_another_line_of_history_is_refused(tmp_path):
    """The same violation dressed as a branch: the lock is published somewhere,
    just not on the history the evaluation is running on. Presence of the file is
    not the proof; ancestry is."""
    root = _init(tmp_path)
    _write(root, "configs/agentic_preregister.json", '{"study_id": "agentic-v1"}\n')
    P = _commit(root, ["configs/agentic_preregister.json"], "P: preregistration")
    _git(root, "checkout", "-q", "-b", "lockline")
    _write(root, LOCKS_REL, _locks_text(P))
    locks_blob = _sha256(root, LOCKS_REL)
    L = _commit(root, [LOCKS_REL], "L: lock on a side branch")
    _git(root, "checkout", "-q", "main")
    # git removed the lock from the worktree with the branch; the operator who
    # tries this puts it back by hand.
    _write(root, LOCKS_REL, _locks_text(P))
    _write(root, REVEAL_REL, _reveal_text(P=P, L=L, locks_blob=locks_blob))
    _seal(root, P=P, L=L, master=heldout_master_seed(L), sealed=True)
    _commit(root, [REVEAL_REL, MANIFEST_REL, SUMS_REL], "R: reveal")

    err = refusal(root)
    assert f"{LOCKS_REL} is added at {L[:12]}" in err
    assert "NOT reachable from E" in err


def test_a_lock_that_does_not_descend_from_P_is_refused(tmp_path):
    """P < L is the leg that makes the lock post-preregistration."""
    root = _init(tmp_path)
    _write(root, "seed.txt", "root\n")
    base = _commit(root, ["seed.txt"], "base")
    _write(root, "configs/agentic_preregister.json", '{"study_id": "agentic-v1"}\n')
    P = _commit(root, ["configs/agentic_preregister.json"], "P: preregistration")
    # branch off BEFORE P, so P is not an ancestor of the lock
    _git(root, "checkout", "-q", "-b", "sideline", base)
    _write(root, LOCKS_REL, _locks_text(P))
    locks_blob = _sha256(root, LOCKS_REL)
    L = _commit(root, [LOCKS_REL], "L: lock that skipped P")
    _write(root, REVEAL_REL, _reveal_text(P=P, L=L, locks_blob=locks_blob))
    _seal(root, P=P, L=L, master=heldout_master_seed(L), sealed=True)
    _commit(root, [REVEAL_REL, MANIFEST_REL, SUMS_REL], "R: reveal")

    err = refusal(root)
    assert f"P ({P[:12]}) is not an ancestor of L ({L[:12]})" in err


def test_an_evaluation_at_a_head_that_predates_R_is_refused(tmp_path):
    """R <= E. Evaluating at a commit that does not contain the reveal proves
    nothing about what the reveal fixed."""
    st = build(tmp_path)
    err = refusal(st["root"], "--head", st["L"])
    assert "NOT reachable from E" in err or "is not an ancestor of E" in err


# ---------------------------------------------------------------------------
# 3. a reveal with no lock behind it
# ---------------------------------------------------------------------------

def test_a_reveal_without_a_committed_lock_is_refused(tmp_path):
    """locks.json sits in the worktree with no commit behind it, so it has no sha
    for the seed to be a function of."""
    st = build(tmp_path, commit_locks=False)
    err = refusal(st["root"])
    assert f"no commit adds {LOCKS_REL}" in err
    assert "no sha to order" in err


def test_a_reveal_without_any_lock_at_all_is_refused(tmp_path):
    st = build(tmp_path)
    (st["root"] / LOCKS_REL).unlink()
    err = refusal(st["root"])
    assert f"{LOCKS_REL} does not exist" in err
    assert "there is no L" in err


def test_a_reveal_committed_without_its_manifest_is_refused(tmp_path):
    """The council's R rule: the receipt and the commitment land together, or the
    evaluated bytes can be chosen after the seed is public."""
    st = build(tmp_path, r_omits=(MANIFEST_REL, SUMS_REL))
    err = refusal(st["root"])
    assert "does not add" in err and PHASE_MANIFEST[HELDOUT_PHASE] in err


def test_a_manifest_added_in_a_later_commit_than_R_is_refused(tmp_path):
    st = build(tmp_path, r_omits=(MANIFEST_REL, SUMS_REL))
    later = _commit(st["root"], [MANIFEST_REL, SUMS_REL], "the commitment, later")
    err = refusal(st["root"])
    assert f"was added at {later[:12]}, not at R" in err


# ---------------------------------------------------------------------------
# 4. digest mismatches
# ---------------------------------------------------------------------------

def test_a_payload_digest_mismatch_is_refused(tmp_path):
    st = build(tmp_path)
    victim = st["root"] / SUITE_DIR / "specs" / "eval.jsonl"
    victim.write_text(victim.read_text(encoding="utf-8") + "tampered\n",
                      encoding="utf-8")
    err = refusal(st["root"])
    assert "do not hash to their committed value" in err
    assert "not the revealed held-out set" in err


def test_a_manifest_edited_after_R_is_refused(tmp_path):
    """The manifest is hashed into its own SHA256SUMS and committed at R, so an
    edit afterwards fails twice: against R's blob and against its own sums line."""
    st = build(tmp_path)
    p = st["root"] / MANIFEST_REL
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["files"]["specs/eval.jsonl"]["sha256"] = "0" * 64
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    err = refusal(st["root"])
    assert f"{MANIFEST_REL} on disk is not the bytes R committed" in err
    assert "was edited after it was sealed" in err


def test_a_manifest_and_sums_that_disagree_are_refused(tmp_path):
    """Both files are committed at R, so the only way to disagree is to have been
    sealed against different bytes -- which the gate must not average over."""
    st = build(tmp_path, sums_extra={"specs/eval.jsonl": "0" * 64})
    err = refusal(st["root"])
    assert "disagree about" in err


def test_a_lock_edited_after_L_is_refused(tmp_path):
    st = build(tmp_path)
    p = st["root"] / LOCKS_REL
    p.write_text(p.read_text(encoding="utf-8").replace('"rs_sft"', '"grpo"'),
                 encoding="utf-8")
    err = refusal(st["root"])
    assert f"{LOCKS_REL} on disk is not the bytes L committed" in err


def test_a_missing_released_file_is_refused(tmp_path):
    st = build(tmp_path)
    (st["root"] / SUITE_DIR / "kb" / "eval_mt.json").unlink()
    err = refusal(st["root"])
    assert "missing on disk" in err


# ---------------------------------------------------------------------------
# 5. the receipt has to be a receipt for THIS lock
# ---------------------------------------------------------------------------

def test_a_receipt_naming_another_lock_is_refused(tmp_path):
    st = build(tmp_path, receipt_locks=FOREIGN_L)
    err = refusal(st["root"])
    assert f"derives its seed from locks commit {FOREIGN_L[:12]}" in err
    assert f"the lock on this line of history is {st['L'][:12]}" in err


def test_a_receipt_with_the_wrong_locks_blob_is_refused(tmp_path):
    st = build(tmp_path)
    p = st["root"] / REVEAL_REL
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["locks_blob_sha256"] = "7" * 64
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    err = refusal(st["root"])
    assert "locks_blob_sha256" in err


def test_a_receipt_carrying_its_own_seed_is_refused(tmp_path):
    """`load_reveal` forbids a supplied seed; the gate must surface that refusal
    rather than crash on it."""
    st = build(tmp_path)
    p = st["root"] / REVEAL_REL
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["eval_seed"] = 12345
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    err = refusal(st["root"])
    assert "does not rederive" in err


def test_a_manifest_generated_from_another_seed_is_refused(tmp_path):
    """The receipt is honest, the commitment is honest, and they are commitments
    to different bytes. That is the case a file-existence check cannot see."""
    st = build(tmp_path, receipt_locks=None)
    root = st["root"]
    # reseal the release from a foreign master seed, then re-commit it
    _seal(root, P=st["P"], L=st["L"], master=heldout_master_seed(FOREIGN_L),
          sealed=True)
    _commit(root, [MANIFEST_REL, SUMS_REL], "reseal from another seed")
    err = refusal(root)
    assert "are not the revealed seed's bytes" in err


# ---------------------------------------------------------------------------
# 6. dedication: nothing rides along
# ---------------------------------------------------------------------------

def test_a_nondedicated_lock_commit_is_refused(tmp_path):
    st = build(tmp_path, extra_in_L=("docs/notes.md",))
    err = refusal(st["root"])
    assert "must be a DEDICATED commit" in err


def test_a_stray_file_in_R_is_refused(tmp_path):
    st = build(tmp_path, extra_in_R=("src/agentlab/eval.py",))
    err = refusal(st["root"])
    assert "also changes" in err and "src/agentlab/eval.py" in err


def test_the_permitted_rider_at_R_is_accepted(tmp_path):
    """suite_release.json is the one extra path the state machine allows."""
    st = build(tmp_path, extra_in_R=(f"{SUITE_DIR}/suite_release.json",))
    r = run(st["root"])
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


# ---------------------------------------------------------------------------
# 7. the commitment has to cover the phase, and only the phase
# ---------------------------------------------------------------------------

def test_an_unsealed_manifest_is_refused(tmp_path):
    st = build(tmp_path, sealed=False)
    err = refusal(st["root"])
    assert "UNSEALED" in err


def test_sums_that_omit_a_heldout_file_are_refused(tmp_path):
    st = build(tmp_path, sums_omit=("specs/eval_h8.jsonl",))
    err = refusal(st["root"])
    assert "does not cover" in err and "specs/eval_h8.jsonl" in err


def test_sums_that_pin_a_train_dev_file_are_refused(tmp_path):
    st = build(tmp_path, sums_extra={"specs/dev.jsonl": "0" * 64})
    err = refusal(st["root"])
    assert "not part of the held-out phase" in err


@pytest.mark.parametrize("legacy", LEGACY_COMMITMENTS)
def test_the_retired_whole_suite_commitment_is_refused(tmp_path, legacy):
    st = build(tmp_path)
    _write(st["root"], f"{SUITE_DIR}/{legacy}", "retired\n")
    err = refusal(st["root"])
    assert "RETIRED whole-suite commitment" in err


# ---------------------------------------------------------------------------
# 8. publication, and the mode that is not the gate
# ---------------------------------------------------------------------------

def test_require_published_refuses_until_R_is_on_the_public_ref(tmp_path):
    st = build(tmp_path)
    err = refusal(st["root"], "--require-published")
    assert "cannot be verified locally" in err
    _git(st["root"], "update-ref", "refs/remotes/origin/main", st["R"])
    r = run(st["root"], "--require-published")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "published on the designated public ref" in r.stdout


def test_require_published_refuses_an_unpushed_reveal(tmp_path):
    st = build(tmp_path)
    _git(st["root"], "update-ref", "refs/remotes/origin/main", st["L"])
    err = refusal(st["root"], "--require-published")
    assert "is not published on" in err


def test_commitments_only_does_not_claim_to_be_the_gate(tmp_path):
    """A mode that skips the payload must not print the sentence the evaluation
    stage relies on."""
    st = build(tmp_path)
    (st["root"] / SUITE_DIR / "specs" / "eval.jsonl").unlink()
    r = run(st["root"], "--commitments-only")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "held-out release VERIFIED" not in r.stdout
    assert "not the evaluation gate" in r.stdout
    # and the default mode still refuses the same tree
    refusal(st["root"])


def test_a_non_repository_is_refused(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    r = run(plain)
    assert r.returncode == 1
    assert "not a git repository" in r.stderr
