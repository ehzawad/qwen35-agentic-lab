"""D3: the held-out split cannot be generated before the lock is published.

These are the guards on the mechanism itself, not on its documentation. Each test
names the exact move it forbids:

  * no held-out seed may live in the config, and the loader refuses one that does;
  * `generate_all` -- the one-pass writer that made every held-out answer
    derivable from the preregistration commit -- refuses;
  * `--phase` is mandatory and `--phase heldout` refuses without a receipt, before
    a staging directory exists, so a refusal leaves no partial file;
  * a receipt whose seed does not rederive from L is refused, and no receipt may
    carry a seed of its own;
  * the old-seed bundles that existed in this workspace cannot be loaded,
    exported or sealed -- their release id is the thing that fails, because their
    task ids and file names look exactly right;
  * the two phase commitments cover their own phase and nothing else, and the
    real external `sha256sum -c` passes over both.

Everything here is CPU-only, uses a clearly labelled SENTINEL derivation, and
validates the MECHANISM. The designated held-out realization is validated once,
at R, by `scripts/validate_suite.py --require-phase heldout`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from agentlab.suite.generate import (HELDOUT_PHASE, HELDOUT_SPLITS, PHASE_MANIFEST,
                                     PHASE_OF, PHASE_SUMS, PHASES, TRAIN_DEV_PHASE,
                                     TRAIN_DEV_SPLITS, certification_spec,
                                     certspec_rels, generate_all, generate_phase,
                                     heldout_master_seed, heldout_release,
                                     heldout_release_id, heldout_split_seed,
                                     load_bundles, load_reveal, load_suite_config,
                                     quarantine_stale_heldout, read_sums,
                                     seal_phase, split_rels, split_seed,
                                     stale_heldout_paths)

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
CONFIG = REPO / "configs" / "suite_v1.toml"
FAKE_L = "7" * 40
# Two per cell: enough for every structural invariant, small enough to be a test.
TINY = 2


def cfg_tiny():
    cfg = load_suite_config(str(CONFIG))
    return dict(cfg, sizes={k: TINY for k in cfg["sizes"]})


def sentinel_receipt(tmp_path, *, locks_commit=FAKE_L, **overrides) -> pathlib.Path:
    """A reveal receipt written the way the state machine writes one."""
    master = heldout_master_seed(locks_commit)
    payload = {
        "schema": "agentic-heldout-reveal-v2", "study_id": "agentic-v1",
        "preregistration_commit": "0" * 40, "locks_commit": locks_commit,
        "locks_blob_sha256": "1" * 64, "public_ref": "refs/heads/main",
        "master_seed_hex": master.hex(),
        "heldout_release_id": heldout_release_id(master),
        "derivation_label": "heldout-master-v2", "generation_protocol": 2,
        "generator_commit": "0" * 40, "revealed_at": "2026-08-05T00:00:00Z",
    }
    payload.update(overrides)
    path = tmp_path / "seed_reveal.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_script(script: str, *args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([str(PY), str(REPO / "scripts" / script), *args],
                          cwd=str(cwd or REPO), capture_output=True, text=True,
                          timeout=900, env=env)


# ---------------------------------------------------------------------------
# 1. the seeds are gone from the config, and the loader refuses their return
# ---------------------------------------------------------------------------

def test_the_committed_config_carries_no_heldout_seed():
    raw = CONFIG.read_text(encoding="utf-8")
    cfg = load_suite_config(str(CONFIG))
    assert sorted(cfg["seeds"]) == sorted(TRAIN_DEV_SPLITS)
    for retired in ("0xA61E0005", "0xA61E0006", "0xA61E0011", "0xA61E0012",
                    "0xA61E0013", "0xA61E0014"):
        assert retired.lower() not in raw.lower(), retired
    assert cfg["heldout"]["derivation"] == "heldout-master-v2"
    assert cfg["heldout"]["no_fallback"] is True


@pytest.mark.parametrize("injected", [
    'eval = 0xA61E0005\n',                     # smuggled into the train/dev table
    'eval_absent = 0xA61E0013\neval_perm = 0xA61E0014\n',
])
def test_a_config_that_reintroduces_a_heldout_seed_is_refused(tmp_path, injected):
    doctored = tmp_path / "suite.toml"
    text = CONFIG.read_text(encoding="utf-8")
    doctored.write_text(text.replace("dev = 0xA61E0004\n",
                                     "dev = 0xA61E0004\n" + injected))
    with pytest.raises(SystemExit, match="held-out seed value"):
        load_suite_config(str(doctored))


def test_a_flat_retired_config_is_refused_rather_than_read(tmp_path):
    """The retired shape put all ten seeds at the top level. It cannot be read."""
    doctored = tmp_path / "suite.toml"
    doctored.write_text('suite = "agentlab-suite-v1"\n'
                        "oracle_sft = 0xA61E0001\ndistill = 0xA61E0002\n"
                        "grpo_train = 0xA61E0003\ndev = 0xA61E0004\n"
                        "eval = 0xA61E0005\nstress = 0xA61E0006\n")
    with pytest.raises(SystemExit, match="held-out seed value"):
        load_suite_config(str(doctored))


def test_a_config_that_edits_the_frozen_derivation_is_refused(tmp_path):
    doctored = tmp_path / "suite.toml"
    doctored.write_text(CONFIG.read_text(encoding="utf-8").replace(
        'split_label = "agentlab-heldout-split-v2"',
        'split_label = "something-else"'))
    with pytest.raises(SystemExit, match="frozen"):
        load_suite_config(str(doctored))


def test_split_seed_has_no_fallback_for_a_heldout_split():
    cfg = cfg_tiny()
    for split in TRAIN_DEV_SPLITS:
        assert split_seed(cfg, split) == cfg["seeds"][split]
    for split in HELDOUT_SPLITS:
        with pytest.raises(RuntimeError, match="no fallback"):
            split_seed(cfg, split)


# ---------------------------------------------------------------------------
# 2. one pass over ten splits is gone
# ---------------------------------------------------------------------------

def test_generate_all_refuses_and_names_the_two_phases(tmp_path):
    with pytest.raises(RuntimeError) as exc:
        generate_all(cfg_tiny(), str(tmp_path / "v1"))
    assert "train-dev" in str(exc.value) and "heldout" in str(exc.value)


def test_the_cli_requires_an_explicit_phase(tmp_path):
    r = run_script("generate_suite.py", "--out", str(tmp_path / "v1"))
    assert r.returncode != 0
    assert "REFUSED: --phase is required" in (r.stdout + r.stderr)
    assert "no --phase all" in (r.stdout + r.stderr)
    assert not (tmp_path / "v1").exists()


def test_the_heldout_phase_refuses_without_a_receipt_and_leaves_nothing(tmp_path):
    out = tmp_path / "v1"
    r = run_script("generate_suite.py", "--phase", "heldout", "--out", str(out))
    assert r.returncode != 0
    assert "REFUSED" in (r.stdout + r.stderr)
    # the refusal happens before a staging directory exists
    assert not out.exists()
    assert not list(tmp_path.glob("*staging*"))


def test_there_is_no_seed_flag_and_an_environment_seed_is_refused(tmp_path):
    src = (REPO / "scripts" / "generate_suite.py").read_text(encoding="utf-8")
    assert '"--seed"' not in src and "'--seed'" not in src
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
               AGENTLAB_HELDOUT_SEED="12345")
    r = subprocess.run([str(PY), str(REPO / "scripts" / "generate_suite.py"),
                        "--phase", "train-dev", "--out", str(tmp_path / "v1")],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300,
                       env=env)
    assert r.returncode != 0
    assert "unregistered seed input" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# 3. the receipt is rederived, never trusted
# ---------------------------------------------------------------------------

def test_a_receipt_whose_seed_does_not_rederive_is_refused(tmp_path):
    path = sentinel_receipt(tmp_path, master_seed_hex="a" * 64)
    with pytest.raises(SystemExit, match="derived, never supplied"):
        load_reveal(str(path))


def test_a_receipt_that_carries_its_own_split_seed_is_refused(tmp_path):
    path = sentinel_receipt(tmp_path, heldout_seed=999)
    with pytest.raises(SystemExit, match="carries seed field"):
        load_reveal(str(path))


def test_a_retired_v1_receipt_is_refused(tmp_path):
    path = tmp_path / "seed_reveal.json"
    path.write_text(json.dumps({"revealed_at": "x", "heldout_seed": 1,
                                "preregistration_commit": "0" * 40}))
    with pytest.raises(SystemExit, match="schema"):
        load_reveal(str(path))


def test_a_receipt_from_another_study_is_refused(tmp_path):
    path = sentinel_receipt(tmp_path, study_id="some-other-study")
    with pytest.raises(SystemExit, match="belongs to study"):
        load_reveal(str(path))


def test_the_receipt_derives_six_distinct_split_seeds(tmp_path):
    reveal = load_reveal(str(sentinel_receipt(tmp_path)))
    master = heldout_master_seed(FAKE_L)
    assert reveal["master_seed"] == master
    assert reveal["split_seeds"] == {s: heldout_split_seed(master, s)
                                     for s in HELDOUT_SPLITS}
    assert len(set(reveal["split_seeds"].values())) == 6
    assert reveal["heldout_release_id"] == heldout_release_id(master)


# ---------------------------------------------------------------------------
# 4. two phases, two commitments, and a real sha256sum -c over each
# ---------------------------------------------------------------------------

def _both_phases(tmp_path):
    """Generate and seal both phases in a scratch tree. -> (out_dir, reveal)."""
    out = tmp_path / "v1"
    cfg = cfg_tiny()
    reveal = load_reveal(str(sentinel_receipt(tmp_path)))
    generate_phase(cfg, str(out), TRAIN_DEV_PHASE)
    generate_phase(cfg, str(out), HELDOUT_PHASE, reveal=reveal)
    for phase in (TRAIN_DEV_PHASE, HELDOUT_PHASE):
        _export_certspecs(out, phase)
        seal_phase(cfg, str(out), phase,
                   reveal=reveal if phase == HELDOUT_PHASE else None)
    return out, reveal


def _export_certspecs(out: pathlib.Path, phase: str) -> None:
    r = run_script("export_eval_specs.py", "--data", str(out),
                   "--splits", *PHASES[phase])
    assert r.returncode == 0, r.stdout + r.stderr


def test_each_phase_commitment_covers_its_own_phase_only(tmp_path):
    out, reveal = _both_phases(tmp_path)
    for phase, other in ((TRAIN_DEV_PHASE, HELDOUT_PHASE),
                         (HELDOUT_PHASE, TRAIN_DEV_PHASE)):
        listed = read_sums(str(out / PHASE_SUMS[phase]))
        for split in PHASES[phase]:
            for rel in split_rels(split):
                assert rel in listed, (phase, rel)
        for split in PHASES[other]:
            for rel in split_rels(split):
                assert rel not in listed, (phase, rel)
        for rel in certspec_rels(phase):
            assert rel in listed, (phase, rel)
        assert PHASE_MANIFEST[phase] in listed
    # the held-out commitment covers 18 sources + 6 certspecs + the eval group
    # certspec S10 reads + itself
    assert len(read_sums(str(out / PHASE_SUMS[HELDOUT_PHASE]))) == 26
    assert len(read_sums(str(out / PHASE_SUMS[TRAIN_DEV_PHASE]))) == 19


def test_the_real_external_sha256sum_c_passes_over_both_commitments(tmp_path):
    out, _reveal = _both_phases(tmp_path)
    for phase in (TRAIN_DEV_PHASE, HELDOUT_PHASE):
        r = subprocess.run(["sha256sum", "-c", "--quiet", PHASE_SUMS[phase]],
                           cwd=str(out), capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr


def test_the_train_dev_manifest_discloses_no_realized_heldout_value(tmp_path):
    out = tmp_path / "v1"
    generate_phase(cfg_tiny(), str(out), TRAIN_DEV_PHASE)
    text = (out / PHASE_MANIFEST[TRAIN_DEV_PHASE]).read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest["heldout_acceptance"] == "DEFERRED_UNTIL_POST_LOCK_REVEAL"
    for split in HELDOUT_SPLITS:
        assert split not in manifest["splits"]
    # the retired global commitment is gone
    assert not (out / "manifest.json").exists()
    assert not (out / "SHA256SUMS").exists()


def test_a_phase_is_not_pinned_until_its_certspecs_are_sealed(tmp_path):
    out = tmp_path / "v1"
    cfg = cfg_tiny()
    generate_phase(cfg, str(out), TRAIN_DEV_PHASE)
    manifest = json.loads((out / PHASE_MANIFEST[TRAIN_DEV_PHASE]).read_text())
    assert manifest["sealed"] is False
    assert manifest["certspecs"] == "PENDING_EXPORT"
    with pytest.raises(RuntimeError, match="derived certspecs"):
        seal_phase(cfg, str(out), TRAIN_DEV_PHASE)
    _export_certspecs(out, TRAIN_DEV_PHASE)
    sealed = seal_phase(cfg, str(out), TRAIN_DEV_PHASE)
    assert sealed["sealed"] is True
    assert set(sealed["certspecs"]) == set(certspec_rels(TRAIN_DEV_PHASE))


def test_generating_one_phase_does_not_disturb_the_other(tmp_path):
    """Selective replacement: a whole-directory swap would destroy the other phase."""
    out, reveal = _both_phases(tmp_path)
    before = {rel: (out / rel).read_bytes()
              for split in HELDOUT_SPLITS for rel in split_rels(split)}
    before[PHASE_SUMS[HELDOUT_PHASE]] = (out / PHASE_SUMS[HELDOUT_PHASE]).read_bytes()
    generate_phase(cfg_tiny(), str(out), TRAIN_DEV_PHASE)
    for rel, data in before.items():
        assert (out / rel).read_bytes() == data, rel


def test_a_heldout_regeneration_is_byte_identical(tmp_path):
    cfg = cfg_tiny()
    reveal = load_reveal(str(sentinel_receipt(tmp_path)))
    a, b = tmp_path / "a", tmp_path / "b"
    m1 = generate_phase(cfg, str(a), HELDOUT_PHASE, reveal=reveal)
    m2 = generate_phase(cfg, str(b), HELDOUT_PHASE, reveal=reveal)
    assert {k: v["sha256"] for k, v in m1["files"].items()} == \
           {k: v["sha256"] for k, v in m2["files"].items()}
    # a DIFFERENT L gives a different realization, which is the whole mechanism
    other = load_reveal(str(sentinel_receipt(tmp_path, locks_commit="6" * 40)))
    m3 = generate_phase(cfg, str(tmp_path / "c"), HELDOUT_PHASE, reveal=other)
    assert m3["files"]["specs/eval.jsonl"]["sha256"] != \
           m1["files"]["specs/eval.jsonl"]["sha256"]


# ---------------------------------------------------------------------------
# 5. the release id: what makes a stale old-seed file fail
# ---------------------------------------------------------------------------

def test_every_heldout_spec_and_certspec_carries_the_release_id(tmp_path):
    out, reveal = _both_phases(tmp_path)
    rid = reveal["heldout_release_id"]
    for split in HELDOUT_SPLITS:
        rows = [json.loads(ln) for ln in
                (out / f"specs/{split}.jsonl").read_text().splitlines() if ln]
        assert rows and all(r["heldout_release_id"] == rid for r in rows)
    for rel in certspec_rels(HELDOUT_PHASE):
        rows = [json.loads(ln) for ln in
                (out / rel).read_text().splitlines() if ln]
        assert rows and all(r["heldout_release_id"] == rid for r in rows)
    # train/dev rows carry none: they are pinned at P, not by a release
    for split in TRAIN_DEV_SPLITS:
        rows = [json.loads(ln) for ln in
                (out / f"specs/{split}.jsonl").read_text().splitlines() if ln]
        assert all("heldout_release_id" not in r for r in rows)


def test_a_heldout_split_cannot_be_loaded_without_a_release(tmp_path):
    out = tmp_path / "v1"
    cfg = cfg_tiny()
    reveal = load_reveal(str(sentinel_receipt(tmp_path)))
    generate_phase(cfg, str(out), TRAIN_DEV_PHASE)
    generate_phase(cfg, str(out), HELDOUT_PHASE, reveal=reveal)
    # train/dev loads with no release at all
    assert load_bundles(str(out), "dev")
    # remove the release commitment: the bytes are still there and still look right
    (out / PHASE_MANIFEST[HELDOUT_PHASE]).unlink()
    with pytest.raises(RuntimeError, match="no revealed release"):
        load_bundles(str(out), "eval")


def test_bytes_that_do_not_match_the_release_cannot_be_loaded(tmp_path):
    out, reveal = _both_phases(tmp_path)
    assert load_bundles(str(out), "eval")
    path = out / "specs/eval.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
    rows[0]["answer"] = "TAMPERED"
    path.write_text("".join(json.dumps(r, sort_keys=True,
                                       separators=(",", ":")) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="not the revealed held-out set"):
        load_bundles(str(out), "eval")


def test_a_row_from_another_release_cannot_be_loaded(tmp_path):
    """The exact shape of a stale old-seed file: right name, wrong release."""
    out, reveal = _both_phases(tmp_path)
    other = load_reveal(str(sentinel_receipt(tmp_path, locks_commit="6" * 40)))
    stale = tmp_path / "stale"
    generate_phase(cfg_tiny(), str(stale), HELDOUT_PHASE, reveal=other)
    for rel in split_rels("eval"):
        (out / rel).write_bytes((stale / rel).read_bytes())
    with pytest.raises(RuntimeError, match="not the revealed"):
        load_bundles(str(out), "eval")


def test_a_heldout_certspec_cannot_be_exported_without_a_release(tmp_path):
    from agentlab.suite.generate import build_task

    bundle = build_task("agentlab-suite-v1", 1234, "eval", "lookup_chain", 2, 0, None)
    assert bundle.release_id is None
    with pytest.raises(RuntimeError, match="no heldout_release_id"):
        certification_spec(bundle)
    bundle.release_id = "0" * 64
    assert certification_spec(bundle)["heldout_release_id"] == "0" * 64


def test_sealing_refuses_a_certspec_from_another_release(tmp_path):
    out, reveal = _both_phases(tmp_path)
    other = load_reveal(str(sentinel_receipt(tmp_path, locks_commit="6" * 40)))
    stale = tmp_path / "stale"
    generate_phase(cfg_tiny(), str(stale), HELDOUT_PHASE, reveal=other)
    _export_certspecs(stale, HELDOUT_PHASE)
    (out / "certspecs" / "eval.jsonl").write_bytes(
        (stale / "certspecs" / "eval.jsonl").read_bytes())
    with pytest.raises(RuntimeError, match="exported from other bytes"):
        seal_phase(cfg_tiny(), str(out), HELDOUT_PHASE, reveal=reveal)


def test_sealing_refuses_a_group_manifest_that_mixes_phases(tmp_path):
    out, reveal = _both_phases(tmp_path)
    # the eval group certspec, but built from train/dev rows
    (out / "certspecs" / "groups" / "eval.jsonl").write_bytes(
        (out / "certspecs" / "groups" / "dev.jsonl").read_bytes())
    with pytest.raises(RuntimeError, match="not in phase heldout"):
        seal_phase(cfg_tiny(), str(out), HELDOUT_PHASE, reveal=reveal)


# ---------------------------------------------------------------------------
# 6. invalidating the values that were already cached in this workspace
# ---------------------------------------------------------------------------

def test_stale_heldout_bytes_are_detected_and_refuse_a_train_dev_generation(tmp_path):
    out = tmp_path / "v1"
    cfg = cfg_tiny()
    generate_phase(cfg, str(out), TRAIN_DEV_PHASE)
    assert stale_heldout_paths(str(out)) == []
    # exactly the shape the workspace was in: held-out payloads from a retired
    # derivation, sitting beside a valid train/dev tree
    (out / "specs" / "eval.jsonl").write_text('{"task_id": "eval-old-0000"}\n')
    (out / "SHA256SUMS").write_text("deadbeef  specs/eval.jsonl\n")
    assert "specs/eval.jsonl" in stale_heldout_paths(str(out))
    with pytest.raises(RuntimeError, match="cached values"):
        generate_phase(cfg, str(out), TRAIN_DEV_PHASE)


def test_quarantine_moves_them_out_with_a_receipt(tmp_path):
    out = tmp_path / "v1"
    generate_phase(cfg_tiny(), str(out), TRAIN_DEV_PHASE)
    (out / "specs" / "eval.jsonl").write_text('{"task_id": "eval-old-0000"}\n')
    (out / "certspecs").mkdir(exist_ok=True)
    (out / "certspecs" / "eval.jsonl").write_text('{"task_id": "eval-old-0000"}\n')
    (out / "manifest.json").write_text("{}")
    dest = tmp_path / "quarantine"
    receipt = quarantine_stale_heldout(str(out), str(dest))
    assert set(receipt["files"]) >= {"specs/eval.jsonl", "certspecs/eval.jsonl",
                                     "manifest.json"}
    for rel in receipt["files"]:
        assert not (out / rel).exists()
        assert (dest / rel).exists()
    assert (dest / "QUARANTINE.json").exists()
    assert receipt["current_derivation"] == "heldout-master-v2"
    assert stale_heldout_paths(str(out)) == []
    # and the generation it was blocking now runs
    generate_phase(cfg_tiny(), str(out), TRAIN_DEV_PHASE)


def test_the_workspace_holds_no_unreleased_heldout_byte():
    """The live tree, not a fixture: the old-seed values are really gone."""
    cfg = load_suite_config(str(CONFIG))
    data_dir = str(REPO / cfg["out_dir"])
    assert stale_heldout_paths(data_dir) == []
    assert heldout_release(data_dir) is None
    for rel in ("manifest.json", "SHA256SUMS"):
        assert not (REPO / cfg["out_dir"] / rel).exists()
    for split in HELDOUT_SPLITS:
        for rel in split_rels(split):
            assert not (REPO / cfg["out_dir"] / rel).exists(), rel


def test_the_quarantine_receipt_records_what_was_invalidated():
    receipt = REPO / "out" / "quarantine" / "stale-heldout-v1" / "QUARANTINE.json"
    if not receipt.exists():
        pytest.skip("this clone never held the retired public-seed bundles")
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    assert rec["current_derivation"] == "heldout-master-v2"
    assert "retired public held-out seeds" in rec["why"]
    # the 18 payloads, the 6 certspecs, the eval group certspec, both retired
    # whole-suite commitment files
    assert len(rec["files"]) >= 27
    for rel, meta in rec["files"].items():
        assert len(meta["sha256"]) == 64 and meta["bytes"] > 0


# ---------------------------------------------------------------------------
# 7. the validator says which phase it validated, and defers the other
# ---------------------------------------------------------------------------

def test_the_validator_defers_heldout_acceptance_before_the_reveal():
    r = run_script("validate_suite.py", "--require-phase", "train-dev")
    assert r.returncode == 0, r.stdout[-3000:]
    assert "train_dev_acceptance: PASS" in r.stdout
    assert "heldout_acceptance:   DEFERRED_UNTIL_POST_LOCK_REVEAL" in r.stdout
    for check in ("14 absent-information", "15 permutation", "16 every held-out"):
        assert f"[DEFERRED] {check}" in r.stdout or any(
            line.startswith("[DEFERRED]") and check in line
            for line in r.stdout.splitlines())
    assert "[PASS] 1 regeneration" in r.stdout
    # and it never claims a held-out PASS
    assert "heldout_acceptance:   PASS" not in r.stdout


def test_the_validator_refuses_the_heldout_phase_with_no_release():
    r = run_script("validate_suite.py", "--phase", "heldout")
    assert r.returncode != 0
    assert "no verified held-out release" in (r.stdout + r.stderr)


def test_phase_membership_is_exhaustive_and_disjoint():
    assert set(PHASES) == {TRAIN_DEV_PHASE, HELDOUT_PHASE}
    assert set(PHASE_OF) == set(TRAIN_DEV_SPLITS) | set(HELDOUT_SPLITS)
    assert not set(TRAIN_DEV_SPLITS) & set(HELDOUT_SPLITS)
    assert len(HELDOUT_SPLITS) == 6
