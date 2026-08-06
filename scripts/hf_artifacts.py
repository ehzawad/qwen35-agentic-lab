#!/usr/bin/env python
"""Push the run's expensive artifacts off this host, and index them in git.

Git does not protect this study. `.gitignore` excludes `out/` and `data/*`,
which is where every GPU-hour of work lands: the preflight evidence tree, the
prompt-tournament rollouts, the token census, the rejection-sampling shards,
the accepted corpus, the SFT views and the adapter. A process crash is already
survivable -- shards are written to temporaries and atomically renamed -- but
host or disk loss is not. Nothing in the pushed tree can reconstruct those
bytes, and a fresh clone cannot re-hash or re-evaluate a checkpoint it does not
have.

This tool copies those artifacts to a PRIVATE Hugging Face dataset repository
and writes `ARTIFACTS.json` at the repository root recording, per file:

    path         repo-relative source path (where the file lives here)
    remote_uri   hf:// URI, and a second URI pinned to the commit that added it
    bytes        size at the moment the digest was taken
    sha256       content digest
    receipt      the producer receipt fields -- session_id,
                 runtime_manifest_sha256, gpu_uuid -- where the artifact or its
                 sidecar actually carries them, plus how they were resolved
    run_id       the run scope the artifact itself declares

`ARTIFACTS.json` is committed. The bytes are not. A mutable URL is not a
durable reference, so every entry also carries `remote_uri_pinned`, which
names the dataset-repo commit oid that added the file.

THE WRITER RULE. A digest of a file that is still being appended to is a lie.
Every candidate is digested twice before upload (a cheap prescan), uploaded,
then digested a THIRD time, and finally re-digested once more when the whole
run ends -- because a slow appender such as the GPU session journal, which
heartbeats about twice a minute, can sail through its own upload window and
still be live. A file whose digest moved at ANY of those points is NOT
recorded: it goes to `skipped` with both digests and the reason. The remote may
hold a snapshot of such a file; it is not referenced by the index, and the next
run re-uploads and overwrites it. This is why the tool can be run while
rejection sampling is appending to the GPU ledger: the ledger simply fails to
record until its producer is quiescent.

Re-runnable at every stage boundary. Recorded files are matched by (size,
mtime, digest) and skipped without re-uploading, so a second run transfers
nothing and leaves `ARTIFACTS.json` byte-identical.

    # what would go, no network
    scripts/hf_artifacts.py plan

    # commit the run-secret hash commitment (writes git, no network, no secret)
    scripts/hf_artifacts.py commit-secret

    # encrypted off-host backup of the run secret
    scripts/hf_artifacts.py backup-secret

    # upload + index the artifacts that exist now
    scripts/hf_artifacts.py upload

    # later stage boundaries
    scripts/hf_artifacts.py upload --group rs_raw --group accepted_corpus
    scripts/hf_artifacts.py upload --all-groups

    # re-digest what the index claims
    scripts/hf_artifacts.py verify

Nothing here imports `agentlab`. The tool must keep working from a fresh clone
with only `huggingface_hub` and `cryptography` available, and it must never be
on the import path of a stage that is running.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import pathlib
import secrets
import sys
import time
from typing import Callable, Iterable, Sequence

TOOL = "scripts/hf_artifacts.py"
TOOL_VERSION = 1
INDEX_KIND = "agentlab_artifact_index"
COMMITMENT_KIND = "agentlab_run_secret_commitment"
SECRET_BLOB_KIND = "agentlab_encrypted_run_secret"

ROOT = pathlib.Path(__file__).resolve().parents[1]

STUDY_RUN_ID = "agentic-v1"
DEFAULT_REPO_ID = f"ehzawad/qwen35-agentic-lab-artifacts-{STUDY_RUN_ID}"

INDEX_REL = "ARTIFACTS.json"
COMMITMENT_REL = "results/agentic/run_secret_commitment.json"
REVEAL_REL = "results/agentic/run_secret_reveal.json"
SECRET_REL = "out/agentic/run_secret.hex"
# A directory no stage owns: writing the envelope next to the plaintext secret
# would put a new file into a directory the live run reads from.
SECRET_BLOB_REL = "out/artifacts/run_secret.enc.json"

COMMITMENT_DOMAIN = "agentlab-run-secret-commitment-v1"

CHUNK = 1 << 20
MAX_OPS_PER_COMMIT = 48
MAX_BYTES_PER_COMMIT = 1_500_000_000

# A window shorter than the slowest appender's period is not evidence of
# quiescence. The GPU session journal heartbeats about every 30 s, and on the
# first real run a small upload finished inside that gap and recorded a digest
# that was already stale nine seconds later. So the end-of-run recheck holds the
# observation window open to at least this long. It costs nothing at a stage
# boundary and is what makes "these bytes did not move while the run watched
# them" a claim rather than a coincidence.
SETTLE_SECONDS = 45.0


# ---------------------------------------------------------------------------
# what may never be published, whatever a caller asks for
# ---------------------------------------------------------------------------

# The S18 receipts. Each needs its own dedicated commit and its own deliberate
# publication step at the L / R boundary, and `locks.json` in the live tree is
# incomplete until then. Publishing either from a bulk artifact sweep would
# destroy the point of the P < L < R ancestry proof.
S18_RECEIPTS = ("results/agentic/locks.json", "results/agentic/seed_reveal.json")

# The held-out suite does not exist yet and MUST NOT exist before R. There is
# no legitimate reason for this tool to carry any held-out byte, so the refusal
# is on the path shape rather than on a file list: a future held-out artifact
# nobody thought to enumerate here is refused too.
HELDOUT_MARKERS = ("heldout", "held_out", "held-out")


def refusal_reason(rel: str) -> str | None:
    """Why this path may never be uploaded, or None."""
    rel = rel.replace(os.sep, "/").lstrip("./")
    low = rel.lower()
    if any(m in low for m in HELDOUT_MARKERS):
        return ("held-out release: it must not exist before R and must never be "
                "published by a bulk artifact sweep")
    if rel in S18_RECEIPTS:
        return ("S18 receipt: belongs to its own dedicated L/R commit and its "
                "own publication step, not to an artifact sweep")
    if rel == SECRET_REL:
        return ("plaintext run secret: use `backup-secret` (encrypted) now and "
                "publish the reveal only after the verdict")
    return None


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Group:
    name: str
    patterns: tuple[str, ...]
    run_id: str            # the run scope these artifacts were produced under
    stage: str             # the stage boundary at which the group is complete
    default: bool          # included by a bare `upload`
    note: str


GROUPS: tuple[Group, ...] = (
    Group("hardware", ("results/agentic/hardware.json",), "dev-preflight-v1",
          "hardware_lock", True,
          "the GPU identity binding every later receipt is checked against"),
    Group("census", ("out/agentic/token_census.json",
                     "out/agentic/token_census.json.sha256",
                     "out/agentic/token_census.log",
                     "results/agentic/token_census.json",
                     "results/agentic/token_census.json.sha256"),
          "dev-preflight-v1", "census", True,
          "the measured size ceilings; the out/ copy is the ignored original"),
    Group("ledger", ("results/agentic/gpu_ledger.jsonl",
                     "results/agentic/gpu_sessions.jsonl"),
          STUDY_RUN_ID, "any", True,
          "GPU accounting; APPENDED LIVE, so it records only when quiescent"),
    Group("manifests", ("results/agentic/manifests/*.json",), STUDY_RUN_ID,
          "any", True, "one runtime manifest per GPU session"),
    Group("preflight_probes", ("results/agentic/preflight/*.json",),
          "dev-preflight-v1", "preflight", True,
          "the five committed probe verdicts (already tracked in git)"),
    Group("preflight_evidence",
          ("out/preflight/*.json", "out/preflight/*.jsonl",
           "out/preflight/*.log", "out/preflight/traces/**/*",
           "out/preflight/rs/**/*", "out/preflight/sft/**/*"),
          "dev-preflight-v1", "preflight", True,
          "the ignored preflight evidence the probe verdicts were computed from"),
    Group("preflight_canary_adapter", ("out/preflight/sft-canary-lora/**/*",),
          "dev-preflight-v1", "preflight", True,
          "the canary LoRA tree: proves the training path ran end to end"),
    Group("prompt_tournament",
          ("out/multiface/prompt_tournament.json",
           "out/multiface/prompt_tournament/*.jsonl"),
          STUDY_RUN_ID, "prompt", True,
          "every tournament rollout the frozen prompt was selected from"),
    Group("run_secret_backup", (SECRET_BLOB_REL,), STUDY_RUN_ID, "prompt", True,
          "AES-256-GCM envelope around the run secret; never the plaintext"),

    # --- not yet complete: run these AT the named stage boundary -----------
    Group("rs_raw", ("data/multiface/raw/*.jsonl",
                     "data/multiface/raw/*.receipt.json"),
          STUDY_RUN_ID, "distill", False,
          "rejection-sampling shards; a producer is appending until distill ends"),
    Group("accepted_corpus", ("data/multiface/accepted.jsonl",
                              "data/multiface/accepted.*.json"),
          STUDY_RUN_ID, "distill", False, "the accepted corpus and its receipt"),
    Group("sft_views", ("out/multiface/views/**/*",), STUDY_RUN_ID, "views",
          False, "the SFT views and the view chain report"),
    Group("adapter", ("out/qwen35-4b-rssft-lora/**/*",
                      "out/*.agentlab_training_manifest.json"),
          STUDY_RUN_ID, "sft", False,
          "THE locked adapter and its training manifest; the L commit hashes it"),
    Group("traces", ("results/agentic/traces/**/*",), STUDY_RUN_ID, "eval", False,
          "the evaluation trace archive"),
)

GROUPS_BY_NAME = {g.name: g for g in GROUPS}
DEFAULT_GROUPS = tuple(g.name for g in GROUPS if g.default)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: pathlib.Path) -> tuple[str, int]:
    """(digest, bytes read). Read once; the size is what was digested."""
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            n += len(b)
            h.update(b)
    return h.hexdigest(), n


def write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def rel_of(p: pathlib.Path, root: pathlib.Path) -> str:
    return str(p.relative_to(root)).replace(os.sep, "/")


def expand(root: pathlib.Path, patterns: Sequence[str]) -> list[str]:
    """Repo-relative files matching the patterns, sorted, deduplicated."""
    seen: dict[str, None] = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and not p.is_symlink():
                seen.setdefault(rel_of(p, root), None)
    return list(seen)


# ---------------------------------------------------------------------------
# producer receipts
# ---------------------------------------------------------------------------

RECEIPT_FIELDS = ("session_id", "runtime_manifest_sha256", "gpu_uuid", "run_id")
_MANIFEST_ALIASES = {"runtime_manifest_sha256": ("runtime_manifest_sha256",
                                                 "manifest_sha256")}
_JSONL_SCAN_ROWS = 200


def _pick(d: dict) -> dict:
    out = {}
    for f in RECEIPT_FIELDS:
        for k in _MANIFEST_ALIASES.get(f, (f,)):
            if d.get(k):
                out[f] = d[k]
                break
    return out


def resolve_receipt(path: pathlib.Path,
                    root: pathlib.Path | None = None) -> dict:
    """The producer receipt fields this artifact actually carries.

    Resolution is by CONTENT only -- a sidecar receipt, the file's own header,
    or the provenance block its rows carry. There is deliberately no
    guess-by-timestamp fallback: attributing an artifact to whichever GPU
    session happened to be open when its mtime landed would be an inference
    dressed up as a receipt. Unresolved is reported as unresolved.
    """
    out: dict = {"source": "unresolved"}

    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    if not sidecar.exists() and path.suffix == ".jsonl":
        sidecar = path.with_suffix(".receipt.json")
    if sidecar.exists():
        try:
            got = _pick(json.loads(sidecar.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            got = {}
        if got:
            base = root if root is not None else ROOT
            try:
                where = rel_of(sidecar, base)
            except ValueError:
                where = sidecar.name
            return {"source": "sidecar_receipt", "sidecar": where, **got}

    if path.suffix == ".json":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            doc = None
        if isinstance(doc, dict):
            got = _pick(doc)
            if got:
                return {"source": "self_describing", **got}
            prov = doc.get("provenance")
            if isinstance(prov, dict):
                got = _pick(prov)
                if got:
                    return {"source": "embedded_provenance", **got}
        return out

    if path.suffix == ".jsonl":
        seen: dict[str, set] = {f: set() for f in RECEIPT_FIELDS}
        rows = 0
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if rows >= _JSONL_SCAN_ROWS:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    rows += 1
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    src = row.get("provenance")
                    got = _pick(src if isinstance(src, dict) else row)
                    for k, v in got.items():
                        seen[k].add(v)
        except OSError:
            return out
        got = {k: (sorted(v)[0] if len(v) == 1 else sorted(v))
               for k, v in seen.items() if v}
        if got:
            return {"source": "embedded_provenance",
                    "rows_scanned": rows, **got}
    return out


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------

def empty_index(repo_id: str, run_id: str = STUDY_RUN_ID) -> dict:
    return {
        "kind": INDEX_KIND,
        "schema_version": 1,
        "run_id": run_id,
        "generated_by": {"tool": TOOL, "tool_version": TOOL_VERSION},
        "remote": {"provider": "huggingface", "repo_type": "dataset",
                   "repo_id": repo_id, "private": True},
        "durability_note": (
            "Most paths below are excluded from git by .gitignore. This index is "
            "the only committed record that the bytes exist; the remote copy is "
            "the only copy that survives loss of this host."),
        "run_secret": {
            "commitment_file": COMMITMENT_REL,
            "encrypted_backup_path": SECRET_BLOB_REL,
            "plaintext_published": False,
            "ordering": ("hash commitment committed before L; the encrypted "
                         "envelope is backed up off-host now; the plaintext "
                         f"secret is published as {REVEAL_REL} only after the "
                         "verdict, and is then checked against the commitment"),
        },
        "updated_at_utc": None,
        "totals": {"files": 0, "bytes": 0},
        "files": [],
        "skipped": [],
        "refused": [],
    }


def load_index(root: pathlib.Path, repo_id: str) -> dict:
    p = root / INDEX_REL
    if not p.exists():
        return empty_index(repo_id)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("kind") != INDEX_KIND:
        raise SystemExit(f"REFUSED: {INDEX_REL} is not a {INDEX_KIND}")
    have = doc.get("remote", {}).get("repo_id")
    if have and have != repo_id:
        raise SystemExit(
            f"REFUSED: {INDEX_REL} already indexes {have}, not {repo_id}. One "
            f"index names one remote; moving the artifacts is a deliberate "
            f"migration, not a flag.")
    fresh = empty_index(repo_id, doc.get("run_id", STUDY_RUN_ID))
    for key in ("files", "skipped", "refused"):
        doc.setdefault(key, [])
    # static explanatory blocks always come from the tool, so a doc written by
    # an older version is upgraded rather than half-stale.
    for key in ("durability_note", "run_secret", "remote", "generated_by"):
        doc[key] = fresh[key]
    return doc


def finalize_index(index: dict) -> dict:
    """Sort, total, and bump the timestamp ONLY if something actually moved.

    A run that uploads nothing must leave the committed file byte-identical --
    otherwise `upload` produces a git diff at every stage boundary and the
    index stops being evidence of anything.
    """
    index["files"].sort(key=lambda f: f["path"])
    index["skipped"].sort(key=lambda f: f["path"])
    index["refused"].sort(key=lambda f: f["path"])
    index["totals"] = {"files": len(index["files"]),
                       "bytes": sum(int(f["bytes"]) for f in index["files"])}
    index["generated_by"] = {"tool": TOOL, "tool_version": TOOL_VERSION}
    return index


def index_fingerprint(index: dict) -> str:
    """Everything except the timestamp, so 'did anything change' is decidable."""
    body = {k: v for k, v in index.items() if k != "updated_at_utc"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Pending:
    rel: str
    abs: pathlib.Path
    group: str
    run_id: str
    sha256: str
    bytes: int
    mtime_ns: int
    remote_path: str
    receipt: dict


Uploader = Callable[[list[Pending]], str]


class Publisher:
    """Group -> digest -> upload -> re-digest -> record.

    `uploader` takes a batch and returns the remote commit oid that added it.
    Injecting it keeps the whole decision path -- refusals, the digest-moved
    veto, idempotency -- testable without a network or a GPU.
    """

    def __init__(self, root: pathlib.Path, repo_id: str, uploader: Uploader,
                 remote_files: set[str] | None = None,
                 now: Callable[[], str] = now_utc,
                 recheck: bool = False, log=print,
                 settle: float = SETTLE_SECONDS):
        self.root = root
        self.settle = float(settle)
        self.repo_id = repo_id
        self.uploader = uploader
        self.remote = set() if remote_files is None else set(remote_files)
        self.now = now
        self.recheck = recheck
        self.log = log
        self.index = load_index(root, repo_id)
        self.prior = {f["path"]: f for f in self.index["files"]}
        self.stats = {"recorded": 0, "unchanged": 0, "skipped": 0,
                      "refused": 0, "uploaded_bytes": 0}
        self.touched: set[str] = set()

    # -- URIs ----------------------------------------------------------------
    def uri(self, remote_path: str, oid: str | None = None) -> str:
        at = f"@{oid}" if oid else ""
        return f"hf://datasets/{self.repo_id}{at}/{remote_path}"

    # -- bookkeeping ---------------------------------------------------------
    def _note(self, bucket: str, entry: dict) -> None:
        rows = [r for r in self.index[bucket] if r["path"] != entry["path"]]
        rows.append(entry)
        self.index[bucket] = rows

    def _drop(self, bucket: str, rel: str) -> None:
        self.index[bucket] = [r for r in self.index[bucket] if r["path"] != rel]

    def _record(self, p: Pending, oid: str | None) -> None:
        self._drop("skipped", p.rel)
        self._drop("refused", p.rel)
        entry = {
            "path": p.rel,
            "group": p.group,
            "run_id": p.run_id,
            "bytes": p.bytes,
            "sha256": p.sha256,
            "mtime_ns": p.mtime_ns,
            "remote_path": p.remote_path,
            "remote_uri": self.uri(p.remote_path),
            "remote_uri_pinned": self.uri(p.remote_path, oid),
            "remote_commit": oid,
            "receipt": p.receipt,
            "recorded_at_utc": self.now(),
        }
        self._note("files", entry)
        self.prior[p.rel] = entry
        self.remote.add(p.remote_path)
        self.touched.add(p.rel)
        self.stats["recorded"] += 1

    def _demote(self, rel: str, group: str, before: str, after: str,
                reason: str, detail: str) -> None:
        self._drop("files", rel)
        self.prior.pop(rel, None)
        self.touched.discard(rel)
        self._note("skipped", {"path": rel, "group": group, "reason": reason,
                               "sha256_before": before, "sha256_after": after,
                               "detail": detail,
                               "observed_at_utc": self.now()})
        self.stats["recorded"] = max(0, self.stats["recorded"] - 1)
        self.stats["skipped"] += 1
        self.log(f"  LIVE     {rel}: {reason}")

    def _reverify(self) -> None:
        """Everything recorded in this run must still hash the same at the end.

        The per-file check only covers the window between a file's own digest
        and its own upload. A slow appender -- the GPU session journal
        heartbeats about twice a minute -- can sail through that window and
        still be live. Requiring stability across the WHOLE run is the honest
        claim: the index says these bytes did not move while this run watched
        them. Records made by EARLIER runs are left alone; they correctly
        describe the bytes at an earlier stage boundary.
        """
        if not self.touched:
            return
        left = self.settle - (time.monotonic() - self.t0)
        if left > 0:
            self.log(f"  settle   holding the window open {left:.0f}s so a slow "
                     f"appender cannot pass as quiescent")
            time.sleep(left)
        for rel in sorted(self.touched):
            rec = self.prior.get(rel)
            if not rec:
                continue
            p = self.root / rel
            if not p.exists():
                self._demote(rel, rec.get("group", ""), rec["sha256"], "",
                             "vanished_after_upload",
                             "the file was removed before the run finished")
                continue
            after, _ = sha256_file(p)
            if after != rec["sha256"]:
                self._demote(
                    rel, rec.get("group", ""), rec["sha256"], after,
                    "digest_moved_during_run",
                    "the digest was stable across this file's own upload but "
                    "moved before the run finished, so its producer is still "
                    "appending; the remote copy is a snapshot and is "
                    "deliberately not referenced")

    # -- the one interesting decision ---------------------------------------
    def _stable_digest(self, path: pathlib.Path) -> tuple[str, int, int] | dict:
        """Digest twice. A file that changed between the two reads is live."""
        d1, n1 = sha256_file(path)
        st = path.stat()
        d2, n2 = sha256_file(path)
        if d1 != d2:
            return {"reason": "digest_moved_prescan", "sha256_before": d1,
                    "sha256_after": d2, "bytes_before": n1, "bytes_after": n2}
        return d1, n1, st.st_mtime_ns

    def run(self, groups: Sequence[str]) -> dict:
        self.t0 = time.monotonic()
        for name in groups:
            g = GROUPS_BY_NAME[name]
            pending: list[Pending] = []
            for rel in expand(self.root, g.patterns):
                why = refusal_reason(rel)
                if why:
                    self._drop("files", rel)
                    self._note("refused", {"path": rel, "group": g.name,
                                           "reason": why,
                                           "observed_at_utc": self.now()})
                    self.stats["refused"] += 1
                    self.log(f"  REFUSED  {rel}: {why}")
                    continue
                abs_p = self.root / rel
                prior = self.prior.get(rel)
                receipt = resolve_receipt(abs_p, self.root)
                got_run = receipt.get("run_id")
                run_id = got_run if isinstance(got_run, str) else g.run_id
                remote_path = f"{run_id}/{rel}"

                if (prior and not self.recheck
                        and prior.get("remote_path") in self.remote):
                    try:
                        st = abs_p.stat()
                    except OSError:
                        st = None
                    if (st and prior.get("mtime_ns") == st.st_mtime_ns
                            and prior.get("bytes") == st.st_size):
                        self.stats["unchanged"] += 1
                        continue

                got = self._stable_digest(abs_p)
                if isinstance(got, dict):
                    self._drop("files", rel)
                    self._note("skipped", {"path": rel, "group": g.name,
                                           **got,
                                           "observed_at_utc": self.now()})
                    self.stats["skipped"] += 1
                    self.log(f"  LIVE     {rel}: {got['reason']}")
                    continue
                digest, size, mtime_ns = got

                if (prior and prior.get("sha256") == digest
                        and prior.get("remote_path") in self.remote):
                    prior["mtime_ns"] = mtime_ns
                    self.stats["unchanged"] += 1
                    continue

                pending.append(Pending(
                    rel=rel, abs=abs_p, group=g.name, run_id=run_id,
                    sha256=digest, bytes=size, mtime_ns=mtime_ns,
                    remote_path=remote_path, receipt=receipt))

            for batch in self._batches(pending):
                total = sum(p.bytes for p in batch)
                self.log(f"  upload   {g.name}: {len(batch)} file(s), "
                         f"{total:,} bytes")
                oid = self.uploader(batch)
                self.stats["uploaded_bytes"] += total
                for p in batch:
                    after, _ = sha256_file(p.abs)
                    if after != p.sha256:
                        self._drop("files", p.rel)
                        self._note("skipped", {
                            "path": p.rel, "group": p.group,
                            "reason": "digest_moved",
                            "sha256_before": p.sha256, "sha256_after": after,
                            "detail": ("the file changed while it was being "
                                       "uploaded; the remote copy may be torn "
                                       "and is deliberately not referenced"),
                            "observed_at_utc": self.now()})
                        self.stats["skipped"] += 1
                        self.log(f"  LIVE     {p.rel}: digest moved during upload")
                        continue
                    self._record(p, oid)
                self._flush()
        self._reverify()
        return self._flush()

    @staticmethod
    def _batches(pending: list[Pending]) -> Iterable[list[Pending]]:
        batch: list[Pending] = []
        acc = 0
        for p in pending:
            if batch and (len(batch) >= MAX_OPS_PER_COMMIT
                          or acc + p.bytes > MAX_BYTES_PER_COMMIT):
                yield batch
                batch, acc = [], 0
            batch.append(p)
            acc += p.bytes
        if batch:
            yield batch

    def _flush(self) -> dict:
        before = self.index.get("updated_at_utc")
        was = index_fingerprint(self.index)
        finalize_index(self.index)
        if index_fingerprint(self.index) != was or before is None:
            self.index["updated_at_utc"] = self.now()
        write_json_atomic(self.root / INDEX_REL, self.index)
        return self.index


# ---------------------------------------------------------------------------
# the Hub
# ---------------------------------------------------------------------------

def hf_api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def ensure_repo(api, repo_id: str) -> None:
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)


def remote_file_set(api, repo_id: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception:
        return set()


def hf_uploader(api, repo_id: str, run_id: str) -> Uploader:
    from huggingface_hub import CommitOperationAdd

    def _upload(batch: list[Pending]) -> str:
        ops = [CommitOperationAdd(path_in_repo=p.remote_path,
                                  path_or_fileobj=str(p.abs)) for p in batch]
        info = api.create_commit(
            repo_id, repo_type="dataset", operations=ops,
            commit_message=(f"{run_id}: add {len(ops)} artifact(s) "
                            f"[{batch[0].group}]"))
        return getattr(info, "oid", None) or ""
    return _upload


def dry_uploader(log=print) -> Uploader:
    def _upload(batch: list[Pending]) -> str:
        for p in batch:
            log(f"  DRY-RUN  {p.rel} -> {p.remote_path} ({p.bytes:,} B)")
        return "DRY-RUN"
    return _upload


# ---------------------------------------------------------------------------
# the run secret: commitment now, reveal after the verdict
# ---------------------------------------------------------------------------

def read_secret(root: pathlib.Path) -> bytes:
    p = root / SECRET_REL
    if not p.exists():
        raise SystemExit(
            f"REFUSED: no run secret at {SECRET_REL}. It is created once per "
            f"run by suite/contract.py; do not manufacture one here.")
    raw = p.read_text(encoding="utf-8").strip()
    return bytes.fromhex(raw)


def commitment_payload(secret: bytes, hex_file: bytes, run_id: str,
                       when: str) -> dict:
    dom = COMMITMENT_DOMAIN.encode("ascii")
    digest = hashlib.sha256(dom + b"\0" + run_id.encode("ascii") + b"\0"
                            + secret).hexdigest()
    return {
        "kind": COMMITMENT_KIND,
        "schema_version": 1,
        "run_id": run_id,
        "committed_at_utc": when,
        "why": (
            "The run secret is 32 bytes from os.urandom, created once per run by "
            "src/agentlab/suite/contract.py and stored ONLY under gitignored "
            "out/. Every recovery token the model sees and every receipt token "
            "is derived from it. Losing it makes the receipts unverifiable and "
            "the run unresumable; substituting a new one silently changes the "
            "model-visible environment. This file fixes the secret's digest in "
            "git BEFORE the locks commit, so the secret can be published after "
            "the verdict and checked against a commitment that predates the "
            "result."),
        "secret_bytes": len(secret),
        "secret_sha256": hashlib.sha256(secret).hexdigest(),
        "hex_file_sha256": hashlib.sha256(hex_file).hexdigest(),
        "commitment": {
            "scheme": "sha256(domain || 0x00 || run_id || 0x00 || secret_bytes)",
            "domain": COMMITMENT_DOMAIN,
            "digest": digest,
        },
        "ordering": {
            "1_commitment": ("this file, committed before L; its git ancestry is "
                             "the proof that the digest predates the result"),
            "2_backup": ("an AES-256-GCM envelope of the same bytes pushed "
                         "off-host to the private artifact repo, indexed in "
                         "ARTIFACTS.json"),
            "3_reveal": (f"after evaluation, the plaintext secret is published "
                         f"as {REVEAL_REL} and anyone recomputes the digests "
                         f"below to confirm it is the secret this file commits "
                         f"to. NOT the S18 held-out seed reveal "
                         f"(results/agentic/seed_reveal.json), which is a "
                         f"different receipt with its own dedicated commit."),
        },
        "how_to_check": {
            "hex_file_sha256": "sha256sum run_secret.hex",
            "secret_sha256":
                "python -c \"import hashlib,pathlib;"
                "s=bytes.fromhex(pathlib.Path('run_secret.hex').read_text()"
                ".strip());print(hashlib.sha256(s).hexdigest())\"",
            "commitment_digest":
                "python -c \"import hashlib,pathlib;"
                "s=bytes.fromhex(pathlib.Path('run_secret.hex').read_text()"
                f".strip());print(hashlib.sha256(b'{COMMITMENT_DOMAIN}'"
                f"+bytes(1)+b'{run_id}'+bytes(1)+s).hexdigest())\"",
        },
    }


def cmd_commit_secret(args) -> int:
    root = pathlib.Path(args.root).resolve()
    secret = read_secret(root)
    hex_file = (root / SECRET_REL).read_bytes()
    payload = commitment_payload(secret, hex_file, args.run_id, now_utc())
    out = root / COMMITMENT_REL
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        same = (old.get("commitment", {}).get("digest")
                == payload["commitment"]["digest"]
                and old.get("secret_sha256") == payload["secret_sha256"])
        if not same:
            raise SystemExit(
                f"REFUSED: {COMMITMENT_REL} already commits to a DIFFERENT "
                f"secret ({old.get('secret_sha256')} != "
                f"{payload['secret_sha256']}). A commitment is write-once. "
                f"Either the secret was regenerated -- which invalidates every "
                f"receipt minted under the old one -- or you are pointing at "
                f"another run. Resolve that, do not overwrite this file.")
        print(f"unchanged: {COMMITMENT_REL} already commits to this secret")
        print(f"commitment: {payload['commitment']['digest']}")
        return 0
    write_json_atomic(out, payload)
    print(f"wrote {COMMITMENT_REL}")
    print(f"  secret_sha256:      {payload['secret_sha256']}")
    print(f"  commitment.digest:  {payload['commitment']['digest']}")
    print("  the secret itself is NOT in this file and must not be committed "
          "until after the verdict")
    return 0


# -- encryption -------------------------------------------------------------

SCRYPT = {"n": 1 << 15, "r": 8, "p": 1, "dklen": 32}
# OpenSSL's default 32 MiB guard is exactly the working set of these parameters;
# it is a library ceiling, not part of the derivation, so it is not recorded.
SCRYPT_MAXMEM = 1 << 27


def derive_key(passphrase: bytes, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase, salt=salt, maxmem=SCRYPT_MAXMEM, **SCRYPT)


def encrypt_secret(secret: bytes, passphrase: bytes, run_id: str,
                   when: str) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    aad = f"agentlab-run-secret/{run_id}".encode("ascii")
    ct = AESGCM(derive_key(passphrase, salt)).encrypt(nonce, secret, aad)
    return {
        "kind": SECRET_BLOB_KIND,
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": when,
        "cipher": "AES-256-GCM",
        "kdf": {"name": "scrypt", **SCRYPT, "salt_b64":
                base64.b64encode(salt).decode()},
        "aad": aad.decode(),
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ciphertext_b64": base64.b64encode(ct).decode(),
        "plaintext_sha256": hashlib.sha256(secret).hexdigest(),
        "plaintext_bytes": len(secret),
        "how_to_decrypt": (
            "scrypt(passphrase, salt, n, r, p, dklen=32) -> AES-256-GCM key; "
            "decrypt ciphertext with nonce and aad; the result must hash to "
            "plaintext_sha256 and match run_secret_commitment.json."),
    }


def decrypt_secret(blob: dict, passphrase: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    kdf = blob["kdf"]
    key = hashlib.scrypt(passphrase,
                         salt=base64.b64decode(kdf["salt_b64"]),
                         n=kdf["n"], r=kdf["r"], p=kdf["p"],
                         dklen=kdf["dklen"], maxmem=SCRYPT_MAXMEM)
    out = AESGCM(key).decrypt(base64.b64decode(blob["nonce_b64"]),
                              base64.b64decode(blob["ciphertext_b64"]),
                              blob["aad"].encode("ascii"))
    if hashlib.sha256(out).hexdigest() != blob["plaintext_sha256"]:
        raise SystemExit("REFUSED: decrypted plaintext does not match "
                         "plaintext_sha256")
    return out


def resolve_passphrase(args) -> tuple[bytes, str, pathlib.Path | None]:
    """(passphrase, how, where it was written). Never inside the repo."""
    if args.passphrase_file:
        p = pathlib.Path(args.passphrase_file).expanduser()
        return p.read_bytes().strip(), f"passphrase-file {p}", None
    env = os.environ.get("AGENTLAB_ARTIFACT_PASSPHRASE")
    if env:
        return env.strip().encode(), "env AGENTLAB_ARTIFACT_PASSPHRASE", None
    home = pathlib.Path(
        os.environ.get("XDG_CONFIG_HOME", str(pathlib.Path.home() / ".config")))
    dest = home / "agentlab" / f"{args.run_id}.artifact-passphrase"
    if dest.exists():
        return dest.read_bytes().strip(), f"existing key file {dest}", dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    phrase = base64.urlsafe_b64encode(secrets.token_bytes(32)).strip()
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(phrase + b"\n")
    return phrase, f"generated, written to {dest}", dest


def cmd_backup_secret(args) -> int:
    root = pathlib.Path(args.root).resolve()
    secret = read_secret(root)
    phrase, how, dest = resolve_passphrase(args)
    blob_path = root / SECRET_BLOB_REL
    if blob_path.exists() and not args.force:
        old = json.loads(blob_path.read_text(encoding="utf-8"))
        if old.get("plaintext_sha256") == hashlib.sha256(secret).hexdigest():
            print(f"unchanged: {SECRET_BLOB_REL} already wraps this secret")
            print("  upload it with: hf_artifacts.py upload "
                  "--group run_secret_backup")
            return 0
        raise SystemExit(f"REFUSED: {SECRET_BLOB_REL} wraps a different secret; "
                         f"pass --force only if you understand why.")
    blob = encrypt_secret(secret, phrase, args.run_id, now_utc())
    write_json_atomic(blob_path, blob)
    # round-trip from the file on disk before claiming a backup exists
    back = decrypt_secret(json.loads(blob_path.read_text(encoding="utf-8")),
                          phrase)
    if back != secret:
        blob_path.unlink(missing_ok=True)
        raise SystemExit("REFUSED: the envelope did not round-trip; no backup "
                         "was written")
    print(f"wrote {SECRET_BLOB_REL} (AES-256-GCM, scrypt n={SCRYPT['n']})")
    print(f"  passphrase: {how}")
    print(f"  plaintext_sha256: {blob['plaintext_sha256']}")
    if dest is not None:
        print("")
        print("  ACTION REQUIRED -- the passphrase is on THIS disk. Copy it to "
              "a password manager or another host, or the encrypted off-host "
              "backup does not survive the disk loss it exists to survive.")
        print(f"  {dest}")
    return 0


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def chosen_groups(args) -> list[str]:
    if args.all_groups:
        return [g.name for g in GROUPS]
    if args.group:
        bad = [g for g in args.group if g not in GROUPS_BY_NAME]
        if bad:
            raise SystemExit(f"REFUSED: unknown group(s) {bad}; known: "
                             f"{sorted(GROUPS_BY_NAME)}")
        return list(args.group)
    return list(DEFAULT_GROUPS)


def cmd_plan(args) -> int:
    root = pathlib.Path(args.root).resolve()
    total = 0
    count = 0
    for name in chosen_groups(args):
        g = GROUPS_BY_NAME[name]
        rels = expand(root, g.patterns)
        keep = [r for r in rels if not refusal_reason(r)]
        drop = [r for r in rels if refusal_reason(r)]
        size = sum((root / r).stat().st_size for r in keep)
        total += size
        count += len(keep)
        flag = "" if g.default else "  (stage-gated)"
        print(f"{name:26s} {len(keep):4d} file(s) {size:>14,} B  "
              f"[{g.run_id} @ {g.stage}]{flag}")
        if args.verbose:
            for r in keep:
                print(f"    {r}")
        for r in drop:
            print(f"    REFUSED {r}: {refusal_reason(r)}")
    print(f"{'TOTAL':26s} {count:4d} file(s) {total:>14,} B")
    return 0


def cmd_upload(args) -> int:
    root = pathlib.Path(args.root).resolve()
    groups = chosen_groups(args)
    if args.dry_run:
        pub = Publisher(root, args.repo_id, dry_uploader(), remote_files=set(),
                        recheck=args.recheck_digests, settle=0.0)
        # a dry run must not rewrite the committed index
        pub._flush = lambda: pub.index  # type: ignore[method-assign]
        idx = pub.run(groups)
        print(json.dumps({"dry_run": True, "stats": pub.stats,
                          "would_record": len(idx["files"])}, indent=2))
        return 0
    api = hf_api(args.token)
    who = api.whoami()
    print(f"authenticated as {who['name']} ({who.get('type')})")
    ensure_repo(api, args.repo_id)
    remote = remote_file_set(api, args.repo_id)
    print(f"repo {args.repo_id} (private dataset): {len(remote)} file(s) present")
    pub = Publisher(root, args.repo_id,
                    hf_uploader(api, args.repo_id, args.run_id),
                    remote_files=remote, recheck=args.recheck_digests,
                    settle=args.settle_seconds)
    idx = pub.run(groups)
    # the index itself belongs in the remote too, so the bucket is self-describing
    try:
        api.upload_file(path_or_fileobj=str(root / INDEX_REL),
                        path_in_repo=INDEX_REL, repo_id=args.repo_id,
                        repo_type="dataset",
                        commit_message=f"{args.run_id}: refresh {INDEX_REL}")
    except Exception as exc:                       # non-fatal: git has the copy
        print(f"  note: could not mirror {INDEX_REL} to the repo: {exc}")
    print(json.dumps({"repo": f"hf://datasets/{args.repo_id}",
                      "stats": pub.stats, "totals": idx["totals"],
                      "skipped": len(idx["skipped"]),
                      "refused": len(idx["refused"])}, indent=2))
    return 0


def cmd_verify(args) -> int:
    root = pathlib.Path(args.root).resolve()
    idx = load_index(root, args.repo_id)
    bad, gone, ok = [], [], 0
    for f in idx["files"]:
        p = root / f["path"]
        if not p.exists():
            gone.append(f["path"])
            continue
        d, n = sha256_file(p)
        if d != f["sha256"] or n != f["bytes"]:
            bad.append({"path": f["path"], "recorded": f["sha256"], "found": d})
        else:
            ok += 1
    print(json.dumps({"records": len(idx["files"]), "match": ok,
                      "mismatched": bad, "absent_locally": gone,
                      "skipped_records": [s["path"] for s in idx["skipped"]]},
                     indent=2))
    return 1 if bad else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--run-id", default=STUDY_RUN_ID)
    ap.add_argument("--token", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_groups(p):
        p.add_argument("--group", action="append", default=None)
        p.add_argument("--all-groups", action="store_true")

    p = sub.add_parser("plan", help="what would go, no network")
    add_groups(p)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("upload", help="upload and index")
    add_groups(p)
    p.add_argument("--dry-run", action="store_true",
                   help="no network and no index write; assumes an empty remote, "
                        "so it lists everything a first upload would send")
    p.add_argument("--recheck-digests", action="store_true",
                   help="ignore the (size, mtime) fast path")
    p.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS,
                   help="minimum observation window before the end-of-run "
                        "digest recheck; a window shorter than the slowest "
                        "appender's period is not evidence of quiescence")
    p.set_defaults(fn=cmd_upload)

    p = sub.add_parser("commit-secret",
                       help="write the write-once run-secret commitment")
    p.set_defaults(fn=cmd_commit_secret)

    p = sub.add_parser("backup-secret", help="encrypt the run secret for upload")
    p.add_argument("--passphrase-file", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_backup_secret)

    p = sub.add_parser("verify", help="re-digest what the index claims")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
