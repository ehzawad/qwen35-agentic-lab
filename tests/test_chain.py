"""The one supported entry point: `make agentic` -> scripts/run_multifaceted_chain.sh.

These are the guards on the properties a driver script loses silently: that a
mere LOOK at the pipeline never touches a GPU, that the stage order is the one
the README promises, that the held-out split stays blind until the locks exist,
and that the S18 lock/reveal ordering is enforced by the tooling rather than by
whoever happens to be running it.

Every test here is CPU-only and starts no server.
"""

from __future__ import annotations

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


def test_the_heldout_seed_derives_from_the_preregistration_commit():
    """The commitment is only worth something if it is DERIVED, not supplied."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("agentic_locks", LOCKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    commit = mod.preregistration_commit()
    assert len(commit) == 40
    # the commit that introduced the preregistration really does contain it
    listed = subprocess.run(["git", "show", "--stat", "--format=", commit],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert "configs/agentic_preregister.json" in listed.stdout
    # and the seed is a pure function of that sha, matching the analyzer's check
    import hashlib
    want = int.from_bytes(hashlib.sha256(
        (commit + ":agentic-heldout-v1").encode()).digest()[:8], "big")
    assert mod.heldout_seed(commit) == want


def test_lock_prompt_refuses_a_prompt_outside_the_preregistered_eight(tmp_path):
    bogus = tmp_path / "frozen.json"
    bogus.write_text(json.dumps({"winner": {"candidate": "p8_combined.txt",
                                            "sha256": "0" * 64}}))
    r = locks_cmd("lock-prompt", "--file", str(bogus))
    assert r.returncode != 0
    assert "not one of the eight preregistered candidates" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# the certification-spec export the eval arms and the leakage veto consume
# ---------------------------------------------------------------------------

def test_split_leakage_veto_passes_on_the_group_manifests_not_the_split_ones():
    """The wiring bug that would have made every verdict a NO VERDICT.

    The three training splits deliberately share template ids 0-7, and eval shares
    10-11 with eval_stress, so handing S10 one manifest per split reports template
    overlap as a harness BUG. The veto is about the train/dev/eval GROUPS.
    """
    from agentlab.analyze import _load_jsonl, veto_s10_splits

    groups_dir = REPO / "data" / "suite" / "v1" / "certspecs" / "groups"
    if not groups_dir.exists():
        pytest.skip("run scripts/export_eval_specs.py first")
    groups = {p.stem: _load_jsonl(p) for p in sorted(groups_dir.glob("*.jsonl"))}
    assert set(groups) == {"train", "dev", "eval"}
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
