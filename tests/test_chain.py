"""The one supported entry point: `make agentic` -> scripts/run_multifaceted_chain.sh.

These are the guards on the properties a driver script loses silently: that a
mere LOOK at the pipeline never touches a GPU, that the stage order is the one
the README promises, that the held-out split stays blind until the locks exist,
and that the S18 lock/reveal ordering is enforced by the tooling rather than by
whoever happens to be running it.

Every test here is CPU-only and starts no server.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "run_multifaceted_chain.sh"
LOCKS = REPO / "scripts" / "agentic_locks.py"
PY = REPO / ".venv" / "bin" / "python"

# The chain must be reachable without a GPU pin; a stage that needs one refuses.
NO_GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")

EXPECTED_STAGES = ["suite", "prompt", "baselock", "distill", "views", "sft",
                   "probe", "grpo", "lock", "eval", "verdict", "ship"]


def chain(*args, env=None, timeout=300):
    return subprocess.run(["bash", str(CHAIN), *args], cwd=REPO,
                          env=env or NO_GPU_ENV, capture_output=True,
                          text=True, timeout=timeout)


def test_the_chain_script_is_valid_bash_and_executable():
    assert os.access(CHAIN, os.X_OK)
    r = subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_help_and_list_and_dry_run_never_touch_a_gpu():
    """The property that matters on a shared box: looking costs nothing.

    No CUDA import, no vLLM server, no allocation -- so all three must succeed
    with CUDA_VISIBLE_DEVICES empty, which would make any real GPU stage refuse.
    """
    for args in (["--help"], ["--list"], ["--dry-run"]):
        r = chain(*args)
        assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-2000:])
        combined = r.stdout + r.stderr
        assert "CUDA out of memory" not in combined
        assert "Traceback" not in combined, (args, combined[-2000:])
    # a dry run announces itself and runs nothing
    r = chain("--dry-run")
    assert "DRY RUN: nothing runs, no GPU is touched" in r.stdout


def test_stage_order_is_the_pipeline_the_readme_promises():
    listed = [ln.strip() for ln in chain("--list").stdout.splitlines()
              if ln.startswith("  ")]
    assert listed == EXPECTED_STAGES
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    # every stage the driver runs is described in the README's one-entry-point
    # section, so the promise and the script cannot drift apart unnoticed
    for phrase in ("generate", "prompt", "verified trajectories", "RS-SFT",
                   "GRPO", "evaluation", "verdict"):
        assert phrase in readme


def test_dry_run_reports_every_stage_and_names_its_pending_inputs():
    out = chain("--dry-run").stdout
    for stage in EXPECTED_STAGES:
        assert f"=== {stage}" in out, stage
    # a chain started from scratch has no tournament winner yet, and the dry run
    # must SAY so rather than pretending the stage would run
    assert "configs/frozen_prompt.json" in out
    assert "prerequisite not present yet" in out


def test_unknown_stage_is_rejected():
    r = chain("--only", "definitely-not-a-stage")
    assert r.returncode != 0
    assert "unknown stage" in (r.stdout + r.stderr)


def test_from_and_to_select_a_contiguous_window():
    out = chain("--dry-run", "--from", "views", "--to", "probe").stdout
    assert "stages=views sft probe" in out
    assert "=== suite" not in out and "=== eval" not in out


def test_the_heldout_split_refuses_to_run_before_the_locks_exist():
    """S18 is enforced, not documented: no locks, no held-out evaluation."""
    if (REPO / "results" / "agentic" / "locks.json").exists():
        pytest.skip("this repo already has locks.json; the refusal path is moot")
    r = chain("--only", "eval")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "REFUSED" in combined and "blind" in combined
    # and it refused BEFORE looking at a card, so no GPU was involved
    assert "GPU ok" not in combined


def test_a_gpu_stage_refuses_an_unregistered_index_before_touching_a_card():
    """This run owns ONE card. A different index is a different study.

    The refusal happens before the nvidia-smi query, so it cannot read -- let alone
    disturb -- a card this run does not own.
    """
    r = subprocess.run(["bash", str(CHAIN), "--check-gpu"], cwd=REPO,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="7",
                               CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "registered on index 0" in combined
    assert "SEPARATE registered run" in combined
    assert "GPU ok" not in combined


def test_a_gpu_stage_refuses_without_an_explicit_pin():
    r = subprocess.run(["bash", str(CHAIN), "--check-gpu"], cwd=REPO,
                       env={k: v for k, v in os.environ.items()
                            if k != "CUDA_VISIBLE_DEVICES"},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "needs an explicit pin" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# the S16/S18 lock receipts
# ---------------------------------------------------------------------------

def locks_cmd(*args, timeout=60):
    return subprocess.run([str(PY), str(LOCKS), *args], cwd=REPO,
                          capture_output=True, text=True, timeout=timeout)


def test_the_heldout_seed_cannot_be_revealed_before_the_locks():
    if (REPO / "results" / "agentic" / "seed_reveal.json").exists():
        pytest.skip("already revealed in this repo")
    r = locks_cmd("reveal")
    assert r.returncode != 0
    assert "REFUSED" in (r.stdout + r.stderr)


def test_the_heldout_seed_derives_from_the_LOCKS_commit_not_the_prereg_one():
    """The derivation is what makes the set unavailable EARLY.

    The retired v1 seed hung off the preregistration commit, which exists before
    any lock -- so the held-out set was derivable in advance. v2 hangs off L, and
    nothing but a full 40-hex commit id is accepted: a prefix would let two
    different commits derive the same suite.
    """
    import hashlib

    from agentlab.suite.generate import (HELDOUT_MASTER_LABEL_PARTS,
                                         HELDOUT_SPLIT_LABEL, HELDOUT_SPLITS,
                                         heldout_master_seed, heldout_release_id,
                                         heldout_split_seed)

    L = "9" * 40
    label = "\0".join(HELDOUT_MASTER_LABEL_PARTS) + "\0"
    want = hashlib.sha256(label.encode() + bytes.fromhex(L)).digest()
    assert heldout_master_seed(L) == want
    assert len(want) == 32                      # a full 256-bit master seed
    seeds = {s: heldout_split_seed(want, s) for s in HELDOUT_SPLITS}
    assert len(set(seeds.values())) == len(HELDOUT_SPLITS)
    for split, seed in seeds.items():
        assert seed == int.from_bytes(hashlib.sha256(
            (HELDOUT_SPLIT_LABEL + "\0").encode() + want + b"\0"
            + split.encode()).digest(), "big")
    # one bit of L moves every split seed and the release id
    other = heldout_master_seed("8" + "9" * 39)
    assert other != want
    assert heldout_release_id(other) != heldout_release_id(want)
    assert all(heldout_split_seed(other, s) != seeds[s] for s in HELDOUT_SPLITS)
    # and a shortened or symbolic reference is refused outright
    for bad in ("9" * 39, "9" * 12, "HEAD", "", None):
        with pytest.raises(ValueError, match="40-hex"):
            heldout_master_seed(bad)


def _locks_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("agentic_locks", LOCKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_repo(tmp_path):
    """A repo with the frozen files, and a `git` callable. -> (repo, git)."""
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "agentic_preregister.json").write_text('{"v": 1}')
    (repo / "configs" / "multifaceted.yaml").write_text("version: 1\n")

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True,
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-q", "-m", "add the preregistration")
    return repo, git


def _point_module_at(mod, repo):
    mod.ROOT = repo
    mod.PREREG = repo / "configs" / "agentic_preregister.json"
    mod.FINAL_MARKER = repo / "configs" / "preregistration_final.json"
    mod.FROZEN_SET = ("configs/agentic_preregister.json", "configs/multifaceted.yaml")
    mod.RESULTS = repo / "results" / "agentic"
    mod.LOCKS = mod.RESULTS / "locks.json"
    mod.REVEAL = mod.RESULTS / "seed_reveal.json"
    return mod


def test_P_is_the_unique_marker_commit_with_no_fallback_anchor(tmp_path):
    """The defect: the anchor was the OLDEST commit that added the prereg file.

    That commit is fixed forever, so every later edit -- including the whole A5000
    hardware pivot -- left the commitment untouched while the docs claimed the
    opposite. P is now the commit that adds the FINALIZATION MARKER, it must be
    unique, and there is no weaker fallback to fall back to: no marker, no P.
    """
    mod = _locks_module()
    repo, git = _tiny_repo(tmp_path)
    _point_module_at(mod, repo)

    # a later edit to the preregistration: under the retired rule this changed
    # nothing about the anchor, and under this one there is no anchor at all yet
    (repo / "configs" / "agentic_preregister.json").write_text('{"v": 2}')
    git("add", "-A")
    git("commit", "-q", "-m", "hardware pivot")
    with pytest.raises(SystemExit, match="no commit adds"):
        mod.preregistration_commit()

    mod.FINAL_MARKER.write_text(json.dumps({"files": {}}))
    # an UNCOMMITTED marker anchors nothing, and says so
    assert mod.verify_finalization()["anchor"] == "uncommitted-marker"
    with pytest.raises(SystemExit, match="no commit adds"):
        mod.preregistration_commit()

    git("add", "-A")
    git("commit", "-q", "-m", "finalize the preregistration")
    P = git("rev-parse", "HEAD")
    assert mod.preregistration_commit() == P
    # an EDITED marker no longer matches the bytes P committed
    mod.FINAL_MARKER.write_text(json.dumps({"files": {}, "sneak": 1}))
    with pytest.raises(SystemExit, match="does not match the bytes committed"):
        mod.preregistration_commit()


def test_a_forward_edit_after_finalization_is_visible(tmp_path):
    mod = _locks_module()
    repo, git = _tiny_repo(tmp_path)
    _point_module_at(mod, repo)
    mod.FINAL_MARKER.write_text(json.dumps({"files": mod.frozen_hashes()}))
    git("add", "-A")
    git("commit", "-q", "-m", "finalize")
    assert mod.verify_finalization()["anchor"] == "finalization-marker"
    (repo / "configs" / "agentic_preregister.json").write_text('{"v": 3}')
    with pytest.raises(SystemExit, match="drifted"):
        mod.verify_finalization()


def test_finalization_refuses_once_anything_has_run(tmp_path):
    """The P gate: finalization happens BEFORE the tournament, or not at all.

    A reveal, a lock or a ledger row at finalization time means the ordering
    already went wrong, and the marker would be pinning a commitment the study has
    partly executed against.
    """
    mod = _locks_module()
    repo, _git = _tiny_repo(tmp_path)
    _point_module_at(mod, repo)
    mod.RESULTS.mkdir(parents=True)
    cfg = {"budget": {"ledger": "results/agentic/gpu_ledger.jsonl"}}
    assert mod.gate_nothing_has_run(cfg) == []

    mod.LOCKS.write_text("{}")
    assert any("belongs BEFORE the prompt tournament" in p
               for p in mod.gate_nothing_has_run(cfg))
    mod.REVEAL.write_text("{}")
    assert any("already unblinded" in p for p in mod.gate_nothing_has_run(cfg))
    (mod.RESULTS / "gpu_ledger.jsonl").write_text('{"gpu_hours": 0.1}\n')
    assert any("GPU work has run" in p for p in mod.gate_nothing_has_run(cfg))


@pytest.mark.parametrize("cmd,args,what", [
    ("cmd_lock_prompt", {"file": "ignored"}, "prompt_winner"),
    ("cmd_lock_checkpoint", {"path": "ignored", "stage": "rs_sft"}, "checkpoint"),
])
def test_neither_lock_can_be_changed_after_the_seed_is_revealed(tmp_path, cmd,
                                                                args, what):
    """Re-locking after a reveal is the one move that would make S18 worthless.

    Once held-out results are unblinded, swapping in a different checkpoint or a
    different prompt turns a preregistered comparison into a search over the test
    set. Both locks refuse BEFORE reading their input, so a re-lock cannot even be
    attempted with a valid file.
    """
    mod = _locks_module()
    mod.ROOT = tmp_path
    mod.RESULTS = tmp_path
    mod.REVEAL = tmp_path / "seed_reveal.json"
    mod.REVEAL.write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED") as e:
        getattr(mod, cmd)(argparse.Namespace(**args))
    assert what in str(e.value)
    assert "AMENDMENT" in str(e.value)


def _lock_chain(tmp_path, *, dedicated=True, publish=True, extra_in_L=None):
    """Build a P -> L repo through the state machine's own rules. -> (mod, repo, L)."""
    mod = _locks_module()
    repo, git = _tiny_repo(tmp_path)
    _point_module_at(mod, repo)
    mod.FINAL_MARKER.write_text(json.dumps({"files": mod.frozen_hashes()}))
    git("add", "-A")
    git("commit", "-q", "-m", "finalize")
    P = git("rev-parse", "HEAD")
    mod.RESULTS.mkdir(parents=True, exist_ok=True)
    mod.LOCKS.write_text(json.dumps({
        "schema": mod.LOCKS_SCHEMA, "study_id": mod.STUDY_ID,
        "preregistration_commit": P,
        "prompt_winner": {"file": "prompts/agentic/p8_combined.txt",
                          "sha256": "a" * 64},
        "checkpoint": {"path": "out/multiface/rssft-lora", "stage": "rs_sft",
                       "checkpoint_sha256": "b" * 64,
                       "checkpoint_files": {"adapter_model.safetensors":
                                            {"sha256": "c" * 64, "bytes": 1}}},
        "selection": {"kind": "sole_candidate"},
    }, indent=2, sort_keys=True) + "\n")
    if extra_in_L:
        (repo / extra_in_L).parent.mkdir(parents=True, exist_ok=True)
        (repo / extra_in_L).write_text("x\n")
    if dedicated:
        git("add", mod.LOCKS_REL)
        if extra_in_L:
            git("add", extra_in_L)
    else:
        git("add", "-A")
    git("commit", "-q", "-m", "lock")
    L = git("rev-parse", "HEAD")
    if publish:
        git("update-ref", "refs/remotes/origin/main", L)
    return mod, repo, L


def test_L_must_be_unique_dedicated_published_and_a_descendant_of_P(tmp_path):
    """Every condition on L, each one refused by name.

    Order comes from ancestry, not from timestamps: nothing in this test writes a
    timestamp, and the state machine never reads one.
    """
    mod, repo, L = _lock_chain(tmp_path)
    facts = mod.locks_commit()
    assert facts["commit"] == L
    assert facts["changed_files"] == [mod.LOCKS_REL]
    assert mod.is_ancestor(facts["preregistration_commit"], L)

    # an edit to the lock AFTER the commit: the published lock is not the tree's
    mod.LOCKS.write_text(mod.LOCKS.read_text().replace('"rs_sft"', '"grpo"'))
    with pytest.raises(SystemExit, match="on disk hashes"):
        mod.locks_commit()


def test_an_undedicated_lock_commit_is_refused(tmp_path):
    mod, repo, L = _lock_chain(tmp_path, extra_in_L="configs/extra.txt")
    with pytest.raises(SystemExit, match="DEDICATED commit"):
        mod.locks_commit()


def test_an_unpublished_lock_commit_is_refused(tmp_path):
    """Publication is a gate, and an ABSENT public ref is refused, not assumed."""
    mod, repo, L = _lock_chain(tmp_path, publish=False)
    with pytest.raises(SystemExit, match="publication cannot be verified"):
        mod.locks_commit()
    # a public ref that exists but does not contain L is the other half
    subprocess.run(["git", "update-ref", mod.PUBLIC_REF, f"{L}^"], cwd=repo,
                   check=True, capture_output=True, timeout=30)
    with pytest.raises(SystemExit, match="not published"):
        mod.locks_commit()


def test_an_incomplete_lock_cannot_be_a_lock_commit(tmp_path):
    """A lock on a mutable PATH pins nothing the seed can be a function of."""
    mod, repo, L = _lock_chain(tmp_path)
    locks = json.loads(mod.LOCKS.read_text())
    locks["checkpoint"] = {"path": "out/multiface/rssft-lora", "stage": "rs_sft"}
    mod.LOCKS.write_text(json.dumps(locks, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        mod.locks_commit()


def test_reveal_always_requires_a_committed_P(tmp_path):
    """The chain's lock stage passes --require-finalization; P is unconditional."""
    chain_text = CHAIN.read_text(encoding="utf-8")
    assert "reveal --require-finalization" in chain_text
    src = LOCKS.read_text(encoding="utf-8")
    assert "--require-finalization" in src
    mod = _locks_module()
    repo, git = _tiny_repo(tmp_path)
    _point_module_at(mod, repo)
    mod.RESULTS.mkdir(parents=True)
    mod.LOCKS.write_text("{}")
    with pytest.raises(SystemExit, match="no commit adds"):
        mod.cmd_reveal(argparse.Namespace(require_finalization=False, no_bytes=True))
    assert not mod.REVEAL.exists()


def test_lock_prompt_refuses_a_prompt_outside_the_preregistered_eight(tmp_path,
                                                                     monkeypatch):
    mod = _locks_module()
    monkeypatch.setattr(mod, "preregistration_commit", lambda: "f" * 40)
    bogus = tmp_path / "frozen.json"
    bogus.write_text(json.dumps({"winner": {"candidate": "p8_combined.txt",
                                            "sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="not one of the eight preregistered"):
        mod.cmd_lock_prompt(argparse.Namespace(file=str(bogus)))


# ---------------------------------------------------------------------------
# the certification-spec export the eval arms and the leakage veto consume
# ---------------------------------------------------------------------------

def test_split_leakage_veto_passes_on_the_group_manifests_not_the_split_ones():
    """The wiring bug that would have made every verdict a NO VERDICT.

    The three training splits deliberately share template ids 0-7, and eval shares
    10-11 with eval_stress, so handing S10 one manifest per split reports template
    overlap as a harness BUG. The veto is about the train/dev/eval GROUPS.

    Before the reveal only the TRAIN and DEV group manifests exist -- the eval group
    is exported at R, from the released held-out bytes -- so this checks what is
    present rather than pretending the held-out group is there.
    """
    from agentlab.analyze import _load_jsonl, veto_s10_splits

    groups_dir = REPO / "data" / "suite" / "v1" / "certspecs" / "groups"
    if not groups_dir.exists():
        pytest.skip("run scripts/export_eval_specs.py first")
    groups = {p.stem: _load_jsonl(p) for p in sorted(groups_dir.glob("*.jsonl"))}
    assert {"train", "dev"} <= set(groups)
    if "eval" not in groups:
        # the pre-reveal state: no held-out certspec may exist yet
        assert not (groups_dir.parent / "eval.jsonl").exists()
    assert veto_s10_splits(groups)["status"] == "OK"

    per_split = {p.stem: _load_jsonl(p) for p in
                 sorted(groups_dir.parent.glob("*.jsonl"))}
    if len(per_split) >= 4:
        # documents WHY the grouping is required rather than incidental
        assert veto_s10_splits(per_split)["status"] == "BUG"


def test_exported_specs_replay_and_declare_their_horizons():
    from agentlab.analyze import _load_jsonl, veto_s9_oracle

    p = REPO / "data" / "suite" / "v1" / "certspecs" / "eval.jsonl"
    if not p.exists():
        pytest.skip("run scripts/export_eval_specs.py first")
    specs = _load_jsonl(p)
    assert len(specs) == 1200
    assert veto_s9_oracle(specs)["status"] == "OK"
