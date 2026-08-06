"""The apparatus records: what the study ran ON, pinned by bytes.

Three tracked records exist because the study named its apparatus instead of
identifying it:

  env/model_revision.json     the base model was registered as the NAME
                              `Qwen/Qwen3.5-4B` with no Hub revision and no
                              weight digest, so an upstream re-upload would have
                              silently changed the subject. This record pins the
                              revision the cache resolved and the SHA-256 of
                              every shard, the weight index, the config, the
                              tokenizer files and the processor configs.

  env/requirements.lock.txt   the environment was described by loose ranges
                              (`transformers>=5.5.3`) and an unhashed snapshot,
                              so `scripts/setup.sh` rebuilt a DIFFERENT
                              environment each time it ran. This lock pins 213
                              distributions with hashes on CPython 3.12.13.

  env/host_apparatus.json     the OS, driver, CUDA runtime and registered card,
                              plus the landmine: flash-linear-attention and
                              causal-conv1d SEGFAULT the forward pass on this
                              stack, so their absence is a property of the
                              apparatus and not an oversight.

What is actually pinned here:

  internal consistency   each record's aggregate digest must follow from its own
                         file list, so a hand-edited record is detectable
                         without touching the cache.
  cross-artifact         the five tokenizer digests already frozen in
                         `results/agentic/token_census.json` must equal the ones
                         in the model record. They identify the same subject or
                         one of them is wrong.
  self-check             where the hub cache names a blob by its content hash,
                         that name must equal the SHA-256 we computed -- an
                         independent check on our own hashing.
  disclosure             the model record must state, in words, that it is an
                         ADDITIVE record made AFTER the preregistration, and it
                         must not touch `configs/agentic_preregister.json`.
  lock discipline        every requirement pinned with `==` and at least one
                         hash; the pin set equal to the older unhashed snapshot;
                         neither segfaulting package present.
  setup refusal          `scripts/setup.sh` must refuse to install over an
                         existing venv, because doing that mid-run would change
                         the program between shards.

Byte-level agreement with the local hub cache is checked too, but SKIPPED when
the cache is absent: a fresh clone has no 8.7 GiB snapshot and must still be
able to run the suite.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL_RECORD = REPO / "env" / "model_revision.json"
HOST_RECORD = REPO / "env" / "host_apparatus.json"
LOCK = REPO / "env" / "requirements.lock.txt"
PYVERSION = REPO / ".python-version"
OLD_SNAPSHOT = REPO / "requirements-lock.txt"
CENSUS = REPO / "results" / "agentic" / "token_census.json"
REPRODUCE = REPO / "docs" / "REPRODUCE.md"
SETUP = REPO / "scripts" / "setup.sh"

PINNED_PYTHON = "3.12.13"
SEGFAULTERS = ("flash-linear-attention", "causal-conv1d")


def _load(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rmr():
    return _load("record_model_revision")


@pytest.fixture(scope="module")
def rha():
    return _load("record_host_apparatus")


@pytest.fixture(scope="module")
def model_record():
    assert MODEL_RECORD.exists(), (
        f"{MODEL_RECORD.relative_to(REPO)} is missing: the study subject would "
        f"be a mutable Hub name again. Run scripts/record_model_revision.py record."
    )
    return json.loads(MODEL_RECORD.read_text())


@pytest.fixture(scope="module")
def host_record():
    assert HOST_RECORD.exists(), (
        f"{HOST_RECORD.relative_to(REPO)} is missing. Run "
        f"scripts/record_host_apparatus.py record."
    )
    return json.loads(HOST_RECORD.read_text())


def _requirements(text: str) -> dict[str, list[str]]:
    """-> {name==version: [hashes]}, from a pip/uv requirements file."""
    out: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        # A pinned requirement and each of its hashes are separate physical
        # lines joined by a trailing backslash; drop the continuation first.
        line = raw.strip().removesuffix("\\").strip()
        if not line or line.startswith(("#", "--index-url", "--extra-index-url")):
            continue
        if line.startswith("--hash="):
            assert current is not None, f"stray hash line: {line}"
            out[current].append(line.removeprefix("--hash="))
            continue
        current = line
        out.setdefault(current, [])
    return out


# ---------------------------------------------------------------------------
# the model record: shape, internal consistency, disclosure
# ---------------------------------------------------------------------------

def test_model_record_pins_an_immutable_revision(model_record, rmr):
    assert model_record["schema"] == rmr.SCHEMA
    subject = model_record["subject"]
    assert subject["hub_repo_id"] == "Qwen/Qwen3.5-4B"
    assert re.fullmatch(r"[0-9a-f]{40}", subject["hub_revision"]), (
        "a branch or tag name is exactly the mutable pointer this record removes"
    )
    # The record must agree with the config it claims to identify, or it pins
    # the wrong model.
    assert subject["configured_as"] == rmr.configured_model_base()
    assert subject["hub_revision"] in subject["revision_url"]


def test_model_record_covers_the_weights_not_just_the_tokenizer(model_record):
    roles = {}
    for entry in model_record["files"]:
        roles.setdefault(entry["role"], []).append(entry)
    for required in ("weight_shard", "weight_index", "model_config", "tokenizer"):
        assert roles.get(required), f"no {required} in the record"
    # Two shards and an index: a record of shards without the index cannot prove
    # the shard set is complete.
    assert len(roles["weight_shard"]) == model_record["weights"]["shard_count"] >= 1
    for entry in model_record["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"]
        assert entry["size_bytes"] > 0, entry["path"]
    assert (model_record["total_bytes"]
            == sum(e["size_bytes"] for e in model_record["files"]))
    assert (model_record["weights"]["shard_bytes"]
            == sum(e["size_bytes"] for e in model_record["files"]
                   if e["role"] == "weight_shard"))


def test_model_record_aggregates_follow_from_its_own_file_list(model_record, rmr):
    """A hand-edited record is caught without reading a single model byte."""
    assert model_record["manifest_sha256"] == rmr.manifest_digest(model_record["files"])
    assert model_record["weights"]["manifest_sha256"] == rmr.manifest_digest(
        model_record["files"], ("weight_shard", "weight_index"))
    # And the aggregate must actually depend on the digests it covers.
    tampered = json.loads(json.dumps(model_record["files"]))
    tampered[0]["sha256"] = "0" * 64
    assert rmr.manifest_digest(tampered) != model_record["manifest_sha256"]


def test_hub_blob_names_confirm_our_own_hashing(model_record):
    """The hub stores large files under their SHA-256. Free cross-check."""
    checked = 0
    for entry in model_record["files"]:
        blob = entry.get("blob") or ""
        if re.fullmatch(r"[0-9a-f]{64}", blob):
            assert blob == entry["sha256"], (
                f"{entry['path']}: cache blob is named {blob[:12]} but we hashed "
                f"{entry['sha256'][:12]}"
            )
            checked += 1
    assert checked >= 2, "expected the weight shards to be content-addressed blobs"


def test_model_record_agrees_with_the_frozen_token_census(model_record):
    """Same subject, or one of the two artifacts is lying.

    The census tokenizer digests are pinned by the preregistration amendment, so
    this is the one place the new record touches frozen evidence -- by agreeing
    with it, never by changing it.
    """
    if not CENSUS.exists():
        pytest.skip("token census not present in this tree")
    census = json.loads(CENSUS.read_text())["tokenizer"]
    assert census["model"] == model_record["subject"]["hub_repo_id"]
    recorded = {e["path"]: e["sha256"] for e in model_record["files"]}
    for name, digest in census["files_sha256"].items():
        assert recorded.get(name) == digest, (
            f"{name}: census says {digest[:12]}, the model record says "
            f"{str(recorded.get(name))[:12]}"
        )


def test_model_record_discloses_that_it_came_after_P(model_record, rmr):
    """It must read as a dated amendment, not as part of the registration."""
    disclosure = model_record["disclosure"]
    assert disclosure == rmr.DISCLOSURE
    lowered = disclosure.lower()
    for phrase in ("additive", "after p", "not registered earlier",
                   "does not amend", "name only"):
        assert phrase in lowered, f"disclosure does not say {phrase!r}"
    assert model_record["kind"] == "post_finalization_apparatus_identification"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        model_record["recorded_at_utc"])
    prereg = model_record["preregistration"]
    assert prereg["preregistration_file"] == "configs/agentic_preregister.json"
    assert "configs/agentic_preregister.json" in prereg["not_edited_by_this_tool"]


def test_the_preregistration_files_really_were_not_touched(model_record):
    """Ancestry, not assertion: P must predate the commit that adds the record.

    Skipped outside a git checkout. When the record is still uncommitted the
    ordering cannot be read from history yet, so only the P side is checked.
    """
    prereg = model_record["preregistration"]
    P = prereg.get("finalization_commit_P")
    if not P:
        pytest.skip("no finalization marker commit recorded")
    def git(*args):
        return subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True,
                              text=True)
    if git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")
    assert git("cat-file", "-e", f"{P}^{{commit}}").returncode == 0, (
        f"recorded P {P[:12]} is not a commit in this repository")
    adds = git("log", "--reverse", "--format=%H", "--",
               "env/model_revision.json").stdout.split()
    if not adds:
        pytest.skip("record not committed yet")
    assert git("merge-base", "--is-ancestor", P, adds[0]).returncode == 0, (
        f"P {P[:12]} is not an ancestor of the commit that adds the record "
        f"({adds[0][:12]}): the record would not be post-finalization")


def test_model_record_says_loader_propagation_is_still_deferred(model_record):
    """The honest gap, kept visible instead of implied closed."""
    offline = model_record["offline_resolution"]
    note = offline["loader_revision_propagation"].lower()
    assert "deferred" in note
    assert "revision=" in offline["loader_revision_propagation"]
    assert "snapshots/" in offline["snapshot_relpath"]
    assert model_record["subject"]["hub_revision"] in offline["snapshot_relpath"]


# ---------------------------------------------------------------------------
# the model record against the actual bytes (skipped without the cache)
# ---------------------------------------------------------------------------

def test_recorded_bytes_are_still_the_bytes_on_disk(model_record, rmr):
    cache = rmr.hub_cache_root()
    snap = cache / model_record["offline_resolution"]["snapshot_relpath"]
    if not snap.is_dir():
        pytest.skip(f"hub snapshot not materialized at {snap}")
    if os.environ.get("AGENTLAB_SKIP_SLOW_HASHES"):
        pytest.skip("AGENTLAB_SKIP_SLOW_HASHES set")
    # Hash the small files always; the 8.7 GiB of shards only when asked, so the
    # default suite stays a CPU-seconds test rather than a disk-minutes one.
    heavy = os.environ.get("AGENTLAB_VERIFY_MODEL_SHARDS") == "1"
    for entry in model_record["files"]:
        if entry["role"] == "weight_shard" and not heavy:
            target = (snap / entry["path"]).resolve()
            assert target.exists(), f"{entry['path']} is a dangling cache symlink"
            assert target.stat().st_size == entry["size_bytes"], entry["path"]
            continue
        digest, size = rmr.sha256_file((snap / entry["path"]).resolve())
        assert digest == entry["sha256"], f"{entry['path']} MOVED"
        assert size == entry["size_bytes"], entry["path"]


def test_the_pinned_revision_loads_offline(model_record):
    """The point of the pin: these bytes are reachable with the network off."""
    rev = model_record["subject"]["hub_revision"]
    repo_id = model_record["subject"]["hub_repo_id"]
    transformers = pytest.importorskip("transformers")
    rmr = _load("record_model_revision")
    snap = rmr.hub_cache_root() / model_record["offline_resolution"]["snapshot_relpath"]
    if not snap.is_dir():
        pytest.skip("hub snapshot not materialized")
    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        cfg = transformers.AutoConfig.from_pretrained(repo_id, revision=rev)
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior
    assert cfg.model_type


# ---------------------------------------------------------------------------
# the environment lock
# ---------------------------------------------------------------------------

def test_python_is_pinned_to_a_patch_version():
    assert PYVERSION.exists(), ".python-version is missing"
    assert PYVERSION.read_text().strip() == PINNED_PYTHON
    assert PINNED_PYTHON in LOCK.read_text(), (
        "the lock must name the interpreter it was resolved for")


def test_every_requirement_is_pinned_and_hashed():
    reqs = _requirements(LOCK.read_text())
    assert len(reqs) >= 200, f"only {len(reqs)} requirements in the lock"
    for req, hashes in reqs.items():
        assert "==" in req, f"{req} is not pinned to one version"
        assert hashes, f"{req} carries no hash, so --require-hashes cannot bind it"
        for h in hashes:
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", h), f"{req}: {h}"


def test_the_lock_is_the_same_graph_as_the_older_unhashed_snapshot():
    """The lock adds hashes; it must not quietly change versions."""
    locked = set(_requirements(LOCK.read_text()))
    old = {line.strip() for line in OLD_SNAPSHOT.read_text().splitlines()
           if line.strip() and not line.startswith("#")}
    assert locked == old, (
        "env/requirements.lock.txt and requirements-lock.txt disagree:\n"
        f"  only in the lock: {sorted(locked - old)[:8]}\n"
        f"  only in the snapshot: {sorted(old - locked)[:8]}"
    )


def test_the_segfaulting_packages_are_absent_everywhere():
    """Their absence is a property of the apparatus, not an omission."""
    locked = {req.split("==")[0].lower() for req in _requirements(LOCK.read_text())}
    for pkg in SEGFAULTERS:
        assert pkg not in locked, f"{pkg} is in the lock and segfaults the forward pass"
        assert pkg.replace("-", "_") not in locked
    setup = SETUP.read_text()
    assert "flash-linear-attention" in setup and "causal-conv1d" in setup, (
        "setup.sh must keep saying why it omits them")


def test_the_landmine_is_recorded_machine_readably(host_record, rha):
    landmine = host_record["landmine"]
    assert set(landmine["packages_that_must_stay_absent"]) == set(SEGFAULTERS)
    assert landmine["import_names"] == sorted(("causal_conv1d",
                                               "flash_linear_attention"))
    assert "139" in landmine["symptom"] and "segv" in landmine["symptom"].lower()
    assert host_record["forbidden_packages_clean"] is True
    assert host_record["forbidden_packages_present"] == []
    # And the invariant on whatever host is running this suite.
    assert rha.forbidden_present() == [], (
        "a segfaulting fast-path package is importable in this environment")


# ---------------------------------------------------------------------------
# the host record
# ---------------------------------------------------------------------------

def test_host_record_identifies_what_python_cannot_pin(host_record, rha):
    assert host_record["schema"] == rha.SCHEMA
    apparatus = host_record["tested_apparatus"]
    assert apparatus["os"]["pretty_name"] and apparatus["os"]["kernel_release"]
    assert apparatus["os"]["glibc"]
    driver = apparatus["driver"]
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", driver["nvidia_driver_version"])
    stack = apparatus["python_stack"]
    assert stack["python_version"] == PINNED_PYTHON
    assert stack["torch_cuda_runtime"], "no CUDA runtime recorded"
    assert "+cu" in stack["torch_version"], (
        "torch must be recorded with its CUDA build tag, not the bare version")
    assert apparatus["registered_gpu"]["name"] == "NVIDIA RTX A5000"
    assert apparatus["registered_gpu"]["count"] == 1
    assert apparatus["capacity"]["ram_total_gib"] > 0


def test_host_record_states_a_driver_policy_both_ways(host_record):
    """The council asked for one of two positions. It must pick one and say it."""
    policy = host_record["driver_policy"]
    assert policy["exact_tested_driver"] == (
        host_record["tested_apparatus"]["driver"]["nvidia_driver_version"])
    assert "replay" in policy["for_original_replay"].lower()
    independent = policy["for_independent_replication"].lower()
    assert "new run" in independent and "never an append" in independent


def test_host_record_carries_no_home_directory():
    """Tracked evidence must not be pinned to one user's paths."""
    apparatus = json.loads(HOST_RECORD.read_text())["tested_apparatus"]
    blob = json.dumps(apparatus)
    assert "/home/" not in blob
    assert str(pathlib.Path.home()) not in blob


# ---------------------------------------------------------------------------
# setup.sh must not install over a live environment
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_tree(tmp_path):
    """A repository-shaped tree with a .venv, so setup.sh has something to refuse."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "env").mkdir()
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "scripts" / "setup.sh").write_bytes(SETUP.read_bytes())
    (tmp_path / ".python-version").write_text(PINNED_PYTHON + "\n")
    (tmp_path / "env" / "requirements.lock.txt").write_text("# stub\n")
    return tmp_path


def _run_setup(tree: pathlib.Path, *args: str):
    return subprocess.run(("bash", str(tree / "scripts" / "setup.sh")) + args,
                          capture_output=True, text=True, timeout=120)


def test_setup_refuses_to_overwrite_an_existing_venv(fake_tree):
    r = _run_setup(fake_tree, "--frozen")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "already exists" in r.stderr
    assert "--recreate" in r.stderr


def test_setup_defaults_to_frozen_and_rejects_unknown_flags(fake_tree):
    r = _run_setup(fake_tree, "--nonsense")
    assert r.returncode == 2
    assert "unknown argument" in r.stderr
    # `--frozen` is the default: no argument at all must take the same path,
    # which the existing-venv refusal proves it does.
    bare = _run_setup(fake_tree)
    assert bare.returncode == 1 and "already exists" in bare.stderr


def test_setup_refuses_while_a_process_is_using_that_venv(fake_tree):
    """Swapping wheels under a running stage is the unrecoverable case."""
    fake = fake_tree / ".venv" / "bin" / "python"
    # A process whose cmdline contains "<tree>/.venv/bin/", which is what the
    # guard greps for -- no real interpreter needed.
    proc = subprocess.Popen(["bash", "-c", f'exec -a "{fake} -m stub" sleep 20'])
    try:
        # --recreate must NOT be enough to get past a live venv.
        r = _run_setup(fake_tree, "--frozen", "--recreate")
        assert r.returncode == 1, r.stdout + r.stderr
        assert "something is running out of" in r.stderr
    finally:
        proc.kill()
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# the fresh-clone document
# ---------------------------------------------------------------------------

def test_reproduce_doc_covers_what_a_stranger_needs():
    assert REPRODUCE.exists(), "docs/REPRODUCE.md is missing"
    text = REPRODUCE.read_text()
    for needed in ("env/requirements.lock.txt", ".python-version",
                   "scripts/record_model_revision.py", "HF_HUB_CACHE",
                   "flash-linear-attention", "causal-conv1d",
                   "git clone", "RAM", "disk"):
        assert needed in text, f"docs/REPRODUCE.md does not mention {needed}"
    lowered = text.lower()
    # The two words the council insisted be kept apart.
    assert "original replay" in lowered and "independent replication" in lowered
