"""The off-host artifact index: shape, the digest-moved veto, and idempotency.

`scripts/hf_artifacts.py` exists because `.gitignore` excludes `out/` and
`data/*`, so a git push protects almost none of this study's GPU work. The tool
copies those bytes to a private Hugging Face dataset repository and commits
`ARTIFACTS.json` as the only in-tree record that they exist.

Three properties are load-bearing and are pinned here:

  shape        every recorded entry carries the repo-relative source path, the
               remote URI and a commit-pinned remote URI, the byte size, the
               SHA-256, the producer receipt fields the artifact actually
               carries, and the run scope. A record missing any of those is
               not a durable reference.

  digest-moved a digest of a file whose producer is still appending is a lie.
               The tool digests, uploads, and re-digests, and REFUSES to record
               a file whose digest moved -- which is what lets it be run while
               rejection sampling is appending to the GPU ledger.

  idempotency  a second run at the same stage boundary transfers nothing and
               leaves the committed index byte-identical, so the file stops
               being evidence the moment it starts churning.

Plus the refusals: the held-out release must never be published (it must not
even exist before R), the S18 receipts belong to their own dedicated commits,
and the plaintext run secret is published only after the verdict.

No network, no GPU: the uploader is injected.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hf_artifacts.py"


@pytest.fixture(scope="module")
def ha():
    spec = importlib.util.spec_from_file_location("hf_artifacts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclass field resolution looks the module up in sys.modules, so a
    # by-path import has to register itself before it executes.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# a miniature run tree
# ---------------------------------------------------------------------------

SESSION = "multidistill-20260806T114504Z-4137477-38358c17"
MANIFEST_SHA = "c" * 64
GPU_UUID = "GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567"


def _write(p: pathlib.Path, text: str) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def tree(tmp_path):
    """A tree whose paths match the real group patterns."""
    root = tmp_path / "repo"
    _write(root / "results/agentic/hardware.json",
           json.dumps({"run_id": "dev-preflight-v1", "gpu_uuid": GPU_UUID}))
    _write(root / "results/agentic/manifests/m1.json",
           json.dumps({"run_id": "agentic-v1", "session_id": SESSION,
                       "manifest_sha256": MANIFEST_SHA, "gpu_uuid": GPU_UUID}))
    _write(root / "results/agentic/gpu_ledger.jsonl",
           json.dumps({"run_id": "agentic-v1", "cumulative_h": 1.0}) + "\n")
    _write(root / "out/multiface/prompt_tournament.json",
           json.dumps({"winner": {"candidate": "p2_plan_state_act.txt"}}))
    _write(root / "out/multiface/prompt_tournament/r1-p1_minimal.txt.jsonl",
           json.dumps({"task_id": "t0", "success": True}) + "\n")
    _write(root / "out/preflight/traces/B0.clean.none.jsonl",
           json.dumps({"task_id": "d0", "provenance": {
               "run_id": "dev-preflight-v1", "session_id": SESSION,
               "runtime_manifest_sha256": MANIFEST_SHA,
               "gpu_uuid": GPU_UUID}}) + "\n")
    return root


class FakeUploader:
    """Records the batches, and can tamper with a file mid-upload."""

    def __init__(self, tamper: str | None = None, tamper_with: bytes = b"more\n"):
        self.batches: list[list[str]] = []
        self.sent: list[str] = []
        self.tamper = tamper
        self.tamper_with = tamper_with
        self.n = 0

    def __call__(self, batch):
        self.n += 1
        self.batches.append([p.rel for p in batch])
        self.sent.extend(p.rel for p in batch)
        for p in batch:
            if self.tamper is not None and p.rel == self.tamper:
                with p.abs.open("ab") as fh:      # the producer appends a row
                    fh.write(self.tamper_with)
        return f"oid{self.n:04d}"


def publish(ha, root, uploader, groups=None, remote=None, **kw):
    kw.setdefault("settle", 0.0)          # no wall-clock waiting in the tests
    pub = ha.Publisher(root, "ehzawad/test-artifacts", uploader,
                       remote_files=set() if remote is None else remote,
                       log=lambda *a, **k: None, **kw)
    idx = pub.run(groups or list(ha.DEFAULT_GROUPS))
    return pub, idx


# ---------------------------------------------------------------------------
# 1. ARTIFACTS.json shape
# ---------------------------------------------------------------------------

REQUIRED = ("path", "group", "run_id", "bytes", "sha256", "remote_path",
            "remote_uri", "remote_uri_pinned", "remote_commit", "receipt",
            "recorded_at_utc")


def test_index_shape_is_a_durable_reference(ha, tree):
    up = FakeUploader()
    pub, idx = publish(ha, tree, up)

    assert idx["kind"] == "agentlab_artifact_index"
    assert idx["run_id"] == "agentic-v1"
    assert idx["remote"] == {"provider": "huggingface", "repo_type": "dataset",
                             "repo_id": "ehzawad/test-artifacts",
                             "private": True}
    assert idx["updated_at_utc"]
    assert idx["files"], "nothing was recorded"

    for f in idx["files"]:
        missing = [k for k in REQUIRED if k not in f]
        assert not missing, f"{f['path']} missing {missing}"
        # repo-relative source path, resolving to the bytes that were digested
        assert not f["path"].startswith("/")
        src = tree / f["path"]
        assert src.is_file()
        assert f["bytes"] == src.stat().st_size
        assert f["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        # the URI names the remote; the pinned URI names the commit that added it
        assert f["remote_uri"] == (f"hf://datasets/ehzawad/test-artifacts/"
                                  f"{f['remote_path']}")
        assert f["remote_commit"] == f["remote_uri_pinned"].split("@")[1].split(
            "/")[0]
        assert f["remote_path"].startswith(f["run_id"] + "/")
        assert isinstance(f["receipt"], dict) and f["receipt"]["source"]

    # totals are the sum of what is actually recorded
    assert idx["totals"]["files"] == len(idx["files"])
    assert idx["totals"]["bytes"] == sum(f["bytes"] for f in idx["files"])
    # and the file on disk is what the caller got back
    on_disk = json.loads((tree / ha.INDEX_REL).read_text(encoding="utf-8"))
    assert on_disk == idx


def test_receipt_fields_come_from_the_artifact_not_from_a_guess(ha, tree):
    up = FakeUploader()
    _, idx = publish(ha, tree, up)
    by = {f["path"]: f for f in idx["files"]}

    # a self-describing runtime manifest: session, manifest digest, GPU, run
    m = by["results/agentic/manifests/m1.json"]["receipt"]
    assert m["source"] == "self_describing"
    assert m["session_id"] == SESSION
    assert m["runtime_manifest_sha256"] == MANIFEST_SHA
    assert m["gpu_uuid"] == GPU_UUID
    assert by["results/agentic/manifests/m1.json"]["run_id"] == "agentic-v1"

    # rows carrying a provenance block
    t = by["out/preflight/traces/B0.clean.none.jsonl"]
    assert t["receipt"]["source"] == "embedded_provenance"
    assert t["receipt"]["session_id"] == SESSION
    assert t["receipt"]["gpu_uuid"] == GPU_UUID
    assert t["run_id"] == "dev-preflight-v1"     # the file's own scope, not the study's

    # a tournament rollout carries no provenance: reported unresolved, never
    # attributed to whichever session happened to be open at its mtime
    r = by["out/multiface/prompt_tournament/r1-p1_minimal.txt.jsonl"]
    assert r["receipt"]["source"] == "unresolved"
    assert "session_id" not in r["receipt"]
    assert r["run_id"] == "agentic-v1"           # falls back to the group scope


def test_sidecar_receipt_is_preferred_when_present(ha, tmp_path):
    d = tmp_path / "data/multiface/raw"
    _write(d / "shard-0000.jsonl", json.dumps({"task_id": "x"}) + "\n")
    _write(d / "shard-0000.receipt.json",
           json.dumps({"session_id": SESSION, "gpu_uuid": GPU_UUID,
                       "runtime_manifest_sha256": MANIFEST_SHA,
                       "run_id": "agentic-v1"}))
    got = ha.resolve_receipt(d / "shard-0000.jsonl")
    assert got["source"] == "sidecar_receipt"
    assert got["session_id"] == SESSION
    assert got["runtime_manifest_sha256"] == MANIFEST_SHA


# ---------------------------------------------------------------------------
# 2. the digest-moved refusal
# ---------------------------------------------------------------------------

LIVE = "results/agentic/gpu_ledger.jsonl"


def test_a_file_written_during_upload_is_refused_not_recorded(ha, tree):
    up = FakeUploader(tamper=LIVE)
    pub, idx = publish(ha, tree, up)

    assert LIVE in up.sent, "the test needs the live file to have been attempted"
    assert LIVE not in [f["path"] for f in idx["files"]], (
        "a file that changed while it was being uploaded was recorded anyway")

    skipped = {s["path"]: s for s in idx["skipped"]}
    assert LIVE in skipped
    s = skipped[LIVE]
    assert s["reason"] == "digest_moved"
    assert s["sha256_before"] != s["sha256_after"]
    assert s["sha256_after"] == hashlib.sha256(
        (tree / LIVE).read_bytes()).hexdigest()
    assert pub.stats["skipped"] == 1

    # every other artifact still went: one live producer must not block the run
    assert len(idx["files"]) >= 5


def test_a_file_growing_before_upload_is_not_even_sent(ha, tree, monkeypatch):
    """The prescan digests twice, so an obviously live file is never shipped."""
    real = ha.sha256_file
    seen = {"n": 0}

    def flaky(p):
        got = real(p)
        if p == tree / LIVE:
            seen["n"] += 1
            if seen["n"] == 1:            # the producer appends between the reads
                with p.open("ab") as fh:
                    fh.write(b'{"cumulative_h": 2.0}\n')
        return got

    monkeypatch.setattr(ha, "sha256_file", flaky)
    up = FakeUploader()
    pub, idx = publish(ha, tree, up)

    assert LIVE not in up.sent, "a live file was uploaded before being vetoed"
    skipped = {s["path"]: s for s in idx["skipped"]}
    assert skipped[LIVE]["reason"] == "digest_moved_prescan"
    assert LIVE not in [f["path"] for f in idx["files"]]


def test_a_slow_appender_is_caught_by_the_end_of_run_recheck(ha, tree):
    """The session journal heartbeats twice a minute: its own window is not enough.

    The tamper lands on a file uploaded EARLY, long after which other groups are
    still being transferred -- exactly the shape that let a live journal through
    a per-file-only check.
    """
    up = FakeUploader()
    pub = ha.Publisher(tree, "ehzawad/test-artifacts", up, remote_files=set(),
                       log=lambda *a, **k: None, settle=0.0)
    real_record = pub._record

    def record_then_grow(p, oid):
        real_record(p, oid)
        if p.rel == LIVE:                    # a heartbeat lands after recording
            with p.abs.open("ab") as fh:
                fh.write(b'{"event": "heartbeat"}\n')
    pub._record = record_then_grow
    idx = pub.run(list(ha.DEFAULT_GROUPS))

    assert LIVE not in [f["path"] for f in idx["files"]]
    s = {x["path"]: x for x in idx["skipped"]}[LIVE]
    assert s["reason"] == "digest_moved_during_run"
    assert s["sha256_after"] == hashlib.sha256(
        (tree / LIVE).read_bytes()).hexdigest()
    assert pub.stats["skipped"] == 1
    assert pub.stats["recorded"] == len(idx["files"])
    assert idx["totals"]["files"] == len(idx["files"])
    assert idx["totals"]["bytes"] == sum(f["bytes"] for f in idx["files"])


def test_the_observation_window_is_held_open_to_the_settle_floor(ha, tree,
                                                                monkeypatch):
    """A window shorter than the appender's period is not evidence of quiescence."""
    slept: list[float] = []
    monkeypatch.setattr(ha.time, "sleep", lambda s: slept.append(s))
    up = FakeUploader()
    pub, idx = publish(ha, tree, up, settle=45.0)
    assert slept and 40 < slept[0] <= 45, slept
    assert idx["files"]

    # nothing recorded means nothing to watch, so nothing to wait for
    slept.clear()
    remote = {f["remote_path"] for f in idx["files"]}
    publish(ha, tree, FakeUploader(), remote=remote, settle=45.0)
    assert slept == []


def test_an_earlier_runs_record_is_not_demoted_by_a_later_change(ha, tree):
    """A record from a previous boundary correctly describes the bytes then."""
    _, idx1 = publish(ha, tree, FakeUploader())
    remote = {f["remote_path"] for f in idx1["files"]}
    stale = {f["path"]: f["sha256"] for f in idx1["files"]}[LIVE]

    # nothing to do this time except one unrelated new file
    _write(tree / "results/agentic/manifests/m2.json",
           json.dumps({"run_id": "agentic-v1", "session_id": SESSION}))
    up2 = FakeUploader(tamper="results/agentic/manifests/m2.json")
    _, idx2 = publish(ha, tree, up2, remote=remote)

    kept = {f["path"]: f for f in idx2["files"]}
    assert LIVE in kept and kept[LIVE]["sha256"] == stale
    assert "results/agentic/manifests/m2.json" in [s["path"]
                                                   for s in idx2["skipped"]]


def test_a_previously_recorded_file_that_goes_live_loses_its_record(ha, tree):
    """A stale record is worse than no record: it points at bytes that moved."""
    _, idx = publish(ha, tree, FakeUploader())
    assert LIVE in [f["path"] for f in idx["files"]]

    with (tree / LIVE).open("ab") as fh:
        fh.write(b'{"cumulative_h": 3.0}\n')
    remote = {f["remote_path"] for f in idx["files"]}
    _, idx2 = publish(ha, tree, FakeUploader(tamper=LIVE), remote=remote)

    assert LIVE not in [f["path"] for f in idx2["files"]]
    assert LIVE in [s["path"] for s in idx2["skipped"]]


# ---------------------------------------------------------------------------
# 3. idempotency and resumability
# ---------------------------------------------------------------------------

def test_second_run_uploads_nothing_and_leaves_the_index_byte_identical(ha, tree):
    up1 = FakeUploader()
    _, idx1 = publish(ha, tree, up1)
    first_bytes = (tree / ha.INDEX_REL).read_bytes()
    n = len(idx1["files"])
    assert up1.sent

    remote = {f["remote_path"] for f in idx1["files"]}
    up2 = FakeUploader()
    pub2, idx2 = publish(ha, tree, up2, remote=remote)

    assert up2.sent == [], f"re-uploaded {up2.sent}"
    assert pub2.stats["unchanged"] == n
    assert pub2.stats["recorded"] == 0
    assert (tree / ha.INDEX_REL).read_bytes() == first_bytes, (
        "a no-op run rewrote the committed index")


def test_recheck_digests_still_uploads_nothing_when_content_matches(ha, tree):
    up1 = FakeUploader()
    _, idx1 = publish(ha, tree, up1)
    first_bytes = (tree / ha.INDEX_REL).read_bytes()
    remote = {f["remote_path"] for f in idx1["files"]}

    up2 = FakeUploader()
    pub2, _ = publish(ha, tree, up2, remote=remote, recheck=True)
    assert up2.sent == []
    assert (tree / ha.INDEX_REL).read_bytes() == first_bytes


def test_a_record_whose_remote_copy_is_absent_is_re_uploaded(ha, tree):
    """Resumability: a crash between the transfer and the index write recovers."""
    _, idx1 = publish(ha, tree, FakeUploader())
    remote = {f["remote_path"] for f in idx1["files"]}
    lost = sorted(remote)[0]
    up2 = FakeUploader()
    pub2, idx2 = publish(ha, tree, up2, remote=remote - {lost})
    assert len(up2.sent) == 1
    assert up2.sent[0] == lost.split("/", 1)[1]
    assert len(idx2["files"]) == len(idx1["files"])


def test_changed_content_is_re_uploaded_and_re_recorded(ha, tree):
    _, idx1 = publish(ha, tree, FakeUploader())
    remote = {f["remote_path"] for f in idx1["files"]}
    target = "out/multiface/prompt_tournament.json"
    (tree / target).write_text('{"winner": {"candidate": "p6.txt"}}',
                               encoding="utf-8")
    up2 = FakeUploader()
    _, idx2 = publish(ha, tree, up2, remote=remote)
    assert up2.sent == [target]
    new = {f["path"]: f for f in idx2["files"]}[target]
    assert new["sha256"] == hashlib.sha256(
        (tree / target).read_bytes()).hexdigest()
    assert new["remote_commit"] == "oid0001"


# ---------------------------------------------------------------------------
# 4. what may never be published
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "data/suite/v1/manifest.heldout.json",
    "data/suite/v1/SHA256SUMS.heldout",
    "data/suite/v1/specs/heldout.jsonl",
    "results/agentic/traces/HELDOUT/x.jsonl",
    "out/whatever/held_out-rows.jsonl",
])
def test_heldout_bytes_are_refused_by_path_shape(ha, rel):
    why = ha.refusal_reason(rel)
    assert why and "held-out" in why


@pytest.mark.parametrize("rel", ["results/agentic/locks.json",
                                 "results/agentic/seed_reveal.json"])
def test_the_s18_receipts_are_refused(ha, rel):
    why = ha.refusal_reason(rel)
    assert why and "dedicated" in why


def test_the_plaintext_run_secret_is_refused(ha):
    why = ha.refusal_reason(ha.SECRET_REL)
    assert why and "after the verdict" in why


def test_group_patterns_stay_inside_the_repo(ha):
    for g in ha.GROUPS:
        assert g.patterns, g.name
        for pat in g.patterns:
            assert not pat.startswith("/"), (g.name, pat)
            assert ".." not in pat, (g.name, pat)


def test_no_default_group_sweeps_up_the_s18_receipts(ha, tree):
    """`results/agentic/*.json` is deliberately NOT a pattern anywhere."""
    _write(tree / "results/agentic/seed_reveal.json", "{}")
    _write(tree / "results/agentic/locks.json", "{}")
    up = FakeUploader()
    _, idx = publish(ha, tree, up)
    assert not set(up.sent) & set(ha.S18_RECEIPTS)
    assert all(f["path"] not in ha.S18_RECEIPTS for f in idx["files"])


def test_a_refused_path_reached_by_a_group_is_logged_and_never_sent(ha, tree,
                                                                   monkeypatch):
    """Belt and braces: even a group that DOES match a refused path is stopped."""
    _write(tree / "results/agentic/locks.json", "{}")
    _write(tree / "results/agentic/manifest.heldout.json", "{}")
    reckless = ha.Group("reckless", ("results/agentic/*.json",), "agentic-v1",
                        "any", True, "a group somebody widened by accident")
    monkeypatch.setitem(ha.GROUPS_BY_NAME, "reckless", reckless)

    up = FakeUploader()
    pub, idx = publish(ha, tree, up, groups=["reckless"])

    refused = {r["path"]: r["reason"] for r in idx["refused"]}
    assert "results/agentic/locks.json" in refused
    assert "dedicated" in refused["results/agentic/locks.json"]
    assert "results/agentic/manifest.heldout.json" in refused
    assert "held-out" in refused["results/agentic/manifest.heldout.json"]
    assert not set(up.sent) & set(refused)
    assert not set(f["path"] for f in idx["files"]) & set(refused)
    assert pub.stats["refused"] == 2
    # hardware.json is in the same directory and is perfectly publishable
    assert "results/agentic/hardware.json" in up.sent


def test_stage_gated_groups_are_not_swept_by_a_bare_upload(ha):
    gated = {g.name for g in ha.GROUPS if not g.default}
    assert {"rs_raw", "accepted_corpus", "sft_views", "adapter",
            "traces"} <= gated
    assert not gated & set(ha.DEFAULT_GROUPS)
    # ... and each names the boundary at which it becomes complete
    for name in gated:
        assert ha.GROUPS_BY_NAME[name].stage in {
            "distill", "views", "sft", "eval"}


# ---------------------------------------------------------------------------
# 5. the run-secret commitment
# ---------------------------------------------------------------------------

SECRET = bytes(range(32))


def test_commitment_digests_are_independently_recomputable(ha):
    hexfile = SECRET.hex().encode() + b"\n"
    p = ha.commitment_payload(SECRET, hexfile, "agentic-v1", "2026-08-06T00:00:00Z")
    assert p["kind"] == "agentlab_run_secret_commitment"
    assert p["secret_bytes"] == 32
    assert p["secret_sha256"] == hashlib.sha256(SECRET).hexdigest()
    assert p["hex_file_sha256"] == hashlib.sha256(hexfile).hexdigest()
    assert p["commitment"]["digest"] == hashlib.sha256(
        b"agentlab-run-secret-commitment-v1" + bytes(1) + b"agentic-v1"
        + bytes(1) + SECRET).hexdigest()
    # the secret itself must not be anywhere in the committed file
    assert SECRET.hex() not in json.dumps(p)
    assert SECRET.hex()[:16] not in json.dumps(p)
    # the commitment is bound to the run, so one run's commitment cannot be
    # replayed as another's
    other = ha.commitment_payload(SECRET, hexfile, "other-run", "t")
    assert other["commitment"]["digest"] != p["commitment"]["digest"]
    assert other["secret_sha256"] == p["secret_sha256"]


def test_commitment_is_write_once(ha, tmp_path, capsys):
    root = tmp_path / "repo"
    _write(root / ha.SECRET_REL, SECRET.hex() + "\n")

    class A:
        pass
    a = A()
    a.root, a.run_id = str(root), "agentic-v1"

    assert ha.cmd_commit_secret(a) == 0
    doc = json.loads((root / ha.COMMITMENT_REL).read_text(encoding="utf-8"))
    assert ha.cmd_commit_secret(a) == 0            # idempotent
    assert json.loads((root / ha.COMMITMENT_REL).read_text()) == doc

    (root / ha.SECRET_REL).write_text(bytes(32).hex() + "\n")
    with pytest.raises(SystemExit) as e:
        ha.cmd_commit_secret(a)
    assert "write-once" in str(e.value)
    assert json.loads((root / ha.COMMITMENT_REL).read_text()) == doc


def test_a_missing_secret_is_never_manufactured(ha, tmp_path):
    class A:
        pass
    a = A()
    a.root, a.run_id = str(tmp_path), "agentic-v1"
    with pytest.raises(SystemExit) as e:
        ha.cmd_commit_secret(a)
    assert "do not manufacture" in str(e.value)


def test_the_envelope_round_trips_and_is_bound_to_the_run(ha):
    blob = ha.encrypt_secret(SECRET, b"pass-phrase", "agentic-v1", "t")
    assert blob["cipher"] == "AES-256-GCM"
    assert SECRET.hex() not in json.dumps(blob)
    assert ha.decrypt_secret(blob, b"pass-phrase") == SECRET

    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        ha.decrypt_secret(blob, b"wrong-phrase")
    tampered = dict(blob, aad="agentlab-run-secret/other-run")
    with pytest.raises(InvalidTag):
        ha.decrypt_secret(tampered, b"pass-phrase")
