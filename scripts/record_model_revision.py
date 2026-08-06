#!/usr/bin/env python
"""Identify the study subject by its bytes, not by a mutable name.

`configs/multifaceted.yaml` names the base model as `Qwen/Qwen3.5-4B` and
nothing else. A Hub repository name is a moving target: the owner can re-upload
`main` at any time, and every loader in this repository calls
`from_pretrained` without a `revision`. Nothing committed would notice. The
tokenizer digests in `results/agentic/token_census.json` do not close this --
they identify the *text pipeline*, not the weights, and a re-upload can keep
the tokenizer byte-identical while changing every shard.

This tool records what the study actually ran against:

    revision      the immutable Hub commit the local cache resolved `main` to
    per file      SHA-256 + byte size of every file in that snapshot, including
                  both weight shards and `model.safetensors.index.json`
    manifest      one aggregate digest over the whole file list, and a second
                  over the weight shards + index alone
    offline       the cache-relative snapshot path, so a later load can be
                  pointed at these exact bytes with the network switched off

WHAT THIS RECORD IS NOT. It is not a preregistration. `configs/agentic_preregister.json`
was finalized (P) before this record existed, it is hash-pinned by the
finalization marker, and it is NOT edited by this tool. This is an ADDITIVE,
DATED APPARATUS-IDENTIFICATION RECORD MADE AFTER P: the revision was recovered
from the local cache after the fact and disclosed, not registered earlier. The
record says so in its own `disclosure` field, and anything that reads it must
carry that wording forward rather than presenting the pin as original.

    # write env/model_revision.json from the local hub cache
    scripts/record_model_revision.py record

    # re-hash the cache and refuse if a single byte moved
    scripts/record_model_revision.py verify

    # where the pinned bytes are, for an offline load
    scripts/record_model_revision.py resolve
    eval "$(scripts/record_model_revision.py env)"

Standard library only, and it does not import `agentlab` or `huggingface_hub`.
It must keep working from a fresh clone, and it must never be importable from a
stage that is running.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "env" / "model_revision.json"
DEFAULT_CONFIG = ROOT / "configs" / "multifaceted.yaml"
SCHEMA = "model_revision/1"
CHUNK = 8 * 1024 * 1024

# The disclosure is a constant, not a template. It is the one sentence that
# keeps this record from being mistaken for part of the frozen registration,
# so it is not caller-supplied and not reworded per invocation.
DISCLOSURE = (
    "This is an ADDITIVE, DATED APPARATUS-IDENTIFICATION RECORD MADE AFTER P "
    "(the preregistration finalization commit). The base model was registered "
    "by NAME ONLY, as Qwen/Qwen3.5-4B, with no Hub revision and no weight "
    "digest. This file recovers the revision that the local Hugging Face cache "
    "had already resolved for the run and hashes those bytes so the subject "
    "stops being mutable. It was NOT registered earlier, it does NOT amend "
    "configs/agentic_preregister.json (which is hash-pinned by the "
    "finalization marker and untouched by this tool), and it must never be "
    "presented as though the revision had been pinned before the study began. "
    "It changes no threshold, margin, sample size, seed, estimand or claim."
)

ROLES = (
    # (role, matcher) -- first match wins, so the shard rule precedes the
    # generic safetensors rule.
    ("weight_index", lambda n: n.endswith("index.json") and "safetensors" in n),
    ("weight_shard", lambda n: n.endswith(".safetensors")),
    ("weight_shard", lambda n: n.endswith(".bin") and "pytorch_model" in n),
    ("model_config", lambda n: n == "config.json"),
    ("generation_config", lambda n: n == "generation_config.json"),
    ("chat_template", lambda n: n.startswith("chat_template")),
    ("tokenizer", lambda n: n.startswith("tokenizer") or n in ("vocab.json", "merges.txt")),
    ("processor_config", lambda n: "preprocessor_config" in n or "processor_config" in n),
)

# Roles a record MUST carry before it can claim to identify a subject. A record
# with no shard digest identifies nothing; a record with shards but no index
# cannot prove the shard set is complete.
REQUIRED_ROLES = ("weight_shard", "weight_index", "model_config", "tokenizer")


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify(name: str) -> str:
    for role, matches in ROLES:
        if matches(name):
            return role
    return "other"


def sha256_file(path: pathlib.Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def manifest_digest(entries: list[dict], roles: tuple[str, ...] | None = None) -> str:
    """One digest over a file list, so a subject has a single name.

    Sorted by path and rendered as fixed text: the digest must not depend on
    directory iteration order or on JSON key order.
    """
    lines = []
    for e in sorted(entries, key=lambda e: e["path"]):
        if roles is not None and e["role"] not in roles:
            continue
        lines.append(f"{e['role']}|{e['path']}|{e['sha256']}|{e['size_bytes']}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def hub_cache_root() -> pathlib.Path:
    """The hub cache, resolved the way huggingface_hub itself resolves it."""
    if os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return pathlib.Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return pathlib.Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = pathlib.Path(xdg).expanduser() if xdg else pathlib.Path.home() / ".cache"
    return base / "huggingface" / "hub"


def repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def configured_model_base(config: pathlib.Path = DEFAULT_CONFIG) -> str | None:
    """`model.base` out of the registered config, without a YAML dependency.

    Read-only, and deliberately narrow: it looks for the `base:` key inside the
    top-level `model:` block and gives up rather than guessing.
    """
    if not config.exists():
        return None
    in_model = False
    for raw in config.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^model:\s*$", line):
            in_model = True
            continue
        if in_model:
            if not line.startswith((" ", "\t")):
                break
            m = re.match(r"^\s+base:\s*(\S+)\s*$", line)
            if m:
                return m.group(1).strip("'\"")
    return None


def resolve_revision(cache: pathlib.Path, repo_id: str, revision: str | None) -> tuple[str, str]:
    """-> (revision, how). An explicit revision wins; otherwise read refs/main."""
    if revision:
        return revision, "argument"
    ref = cache / repo_folder(repo_id) / "refs" / "main"
    if ref.exists():
        return ref.read_text().strip(), "cache refs/main"
    raise SystemExit(
        f"REFUSED: no revision given and {ref} does not exist. Pass "
        f"--revision <40-hex commit>, or point --hf-hub-cache at the cache "
        f"that holds {repo_id}."
    )


def snapshot_dir(cache: pathlib.Path, repo_id: str, revision: str) -> pathlib.Path:
    return cache / repo_folder(repo_id) / "snapshots" / revision


def scan_snapshot(snap: pathlib.Path) -> list[dict]:
    if not snap.is_dir():
        raise SystemExit(
            f"REFUSED: {snap} is not a directory. The pinned revision is not "
            f"materialized in this cache, so there are no bytes to hash."
        )
    entries: list[dict] = []
    for path in sorted(p for p in snap.rglob("*") if not p.is_dir()):
        rel = path.relative_to(snap).as_posix()
        target = path.resolve()
        if not target.exists():
            raise SystemExit(
                f"REFUSED: {rel} points at {target}, which does not exist. A "
                f"dangling cache symlink means the snapshot is not usable "
                f"offline; re-download before recording."
            )
        digest, size = sha256_file(target)
        entries.append({
            "path": rel,
            "role": classify(path.name),
            "sha256": digest,
            "size_bytes": size,
            # The blob name is the cache's own content address. For large files
            # the hub stores them under their SHA-256, so this doubles as an
            # independent check on our own hashing.
            "blob": target.name if target != path else None,
        })
    return entries


def index_total_size(snap: pathlib.Path, entries: list[dict]) -> int | None:
    for e in entries:
        if e["role"] == "weight_index":
            try:
                blob = json.loads((snap / e["path"]).read_text())
            except Exception:
                return None
            total = (blob.get("metadata") or {}).get("total_size")
            return int(total) if isinstance(total, int) else None
    return None


def git_out(*args: str) -> str | None:
    try:
        return subprocess.run(("git", "-C", str(ROOT)) + args, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def preregistration_anchor() -> dict:
    """P, named the same way `agentic_locks.py verify-prereg` names it.

    Recorded so the ordering claim in `disclosure` is checkable by git ancestry
    rather than by trusting the timestamp in this file.
    """
    marker_rel = "configs/preregistration_final.json"
    marker = ROOT / marker_rel
    anchor: dict = {
        "preregistration_file": "configs/agentic_preregister.json",
        "finalization_marker": marker_rel,
        "not_edited_by_this_tool": [
            "configs/agentic_preregister.json",
            marker_rel,
            "configs/multifaceted.yaml",
            "configs/suite_v1.toml",
            "docs/AGENTIC_PROTOCOL.md",
        ],
    }
    if marker.exists():
        try:
            blob = json.loads(marker.read_text())
            anchor["finalized_at_utc"] = blob.get("finalized_at")
        except Exception:
            pass
        commit = git_out("log", "--reverse", "--format=%H", "--", marker_rel)
        if commit:
            anchor["finalization_commit_P"] = commit.splitlines()[0]
    anchor["head_when_recorded"] = git_out("rev-parse", "HEAD")
    return anchor


def acquisition_window(entries: list[dict], snap: pathlib.Path) -> dict:
    """When these bytes landed on this host, from the cache's own mtimes.

    Filesystem evidence, not a Hub receipt: it says when the download finished
    here, which is what dates the apparatus. A cache copied between hosts can
    carry mtimes forward, so this is corroboration, never proof.
    """
    stamps = []
    for e in entries:
        try:
            stamps.append((snap / e["path"]).resolve().stat().st_mtime)
        except OSError:
            continue
    if not stamps:
        return {"derived_from": "unavailable"}
    fmt = lambda t: _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return {
        "downloaded_at_utc_first": fmt(min(stamps)),
        "downloaded_at_utc_last": fmt(max(stamps)),
        "derived_from": "mtime of the cache blobs on the recording host",
    }


def build_record(*, repo_id: str, cache: pathlib.Path, revision: str | None,
                 license_id: str) -> dict:
    rev, how = resolve_revision(cache, repo_id, revision)
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(
            f"REFUSED: {rev!r} is not a 40-hex Hub commit. A branch or tag name "
            f"is exactly the mutable pointer this record exists to remove."
        )
    snap = snapshot_dir(cache, repo_id, rev)
    entries = scan_snapshot(snap)
    roles = {e["role"] for e in entries}
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise SystemExit(
            f"REFUSED: the snapshot carries no {', '.join(missing)} file. This "
            f"record would not identify the weights it claims to identify."
        )
    shard_roles = ("weight_shard", "weight_index")
    return {
        "schema": SCHEMA,
        "kind": "post_finalization_apparatus_identification",
        "recorded_at_utc": _utcnow(),
        "disclosure": DISCLOSURE,
        "preregistration": preregistration_anchor(),
        "subject": {
            "hub_repo_id": repo_id,
            "hub_revision": rev,
            "revision_source": how,
            "revision_url": f"https://huggingface.co/{repo_id}/tree/{rev}",
            "configured_as": configured_model_base(),
            "license": {
                "spdx": license_id,
                "observed_from": (
                    "the Hub model card metadata for this repository, read on "
                    "the recording date. It is NOT verified from the cached "
                    "bytes -- the snapshot carries no LICENSE file."
                ),
            },
            "acquisition": acquisition_window(entries, snap),
        },
        "offline_resolution": {
            "cache_layout": "huggingface_hub hub cache",
            "snapshot_relpath": snap.relative_to(cache).as_posix(),
            "how": (
                "export HF_HUB_OFFLINE=1 and point HF_HUB_CACHE at a cache "
                "containing snapshot_relpath, or load from the absolute "
                "snapshot directory printed by `record_model_revision.py "
                "resolve`. Either way `verify` must pass first: the path is "
                "not the subject, the digests are."
            ),
            "loader_revision_propagation": (
                "DEFERRED, not done. src/agentlab/env.py, src/agentlab/multidistill.py, "
                "src/agentlab/sft.py, scripts/serve.sh, scripts/token_census.py and "
                "scripts/preflight_dev.py still call from_pretrained without "
                "revision=. They are on the import path of the rejection-sampling "
                "stage that is running, so they are not edited here. Until that "
                "propagation lands, offline pinning is an OPERATOR obligation "
                "enforced by this record plus `verify`, not by the loaders."
            ),
        },
        "weights": {
            "shard_count": sum(1 for e in entries if e["role"] == "weight_shard"),
            "shard_bytes": sum(e["size_bytes"] for e in entries if e["role"] == "weight_shard"),
            "index_total_size_bytes": index_total_size(snap, entries),
            "manifest_sha256": manifest_digest(entries, shard_roles),
        },
        "file_count": len(entries),
        "total_bytes": sum(e["size_bytes"] for e in entries),
        "manifest_sha256": manifest_digest(entries),
        "files": entries,
        "host_observation": {
            "note": (
                "Informational only. The identity of the subject is the digests "
                "above; these fields say where they were read on the recording "
                "host and are not part of any check."
            ),
            "hf_hub_cache": str(cache),
            "hostname": os.uname().nodename,
        },
    }


def cmd_record(args) -> int:
    cache = pathlib.Path(args.hf_hub_cache).expanduser() if args.hf_hub_cache else hub_cache_root()
    repo_id = args.repo_id or configured_model_base() or "Qwen/Qwen3.5-4B"
    record = build_record(repo_id=repo_id, cache=cache, revision=args.revision,
                          license_id=args.license)
    out = pathlib.Path(args.output)
    if out.exists() and not args.force:
        old = json.loads(out.read_text())
        stable = ("manifest_sha256", "file_count", "total_bytes")
        moved = [k for k in stable if old.get(k) != record.get(k)]
        moved += ([] if (old.get("subject") or {}).get("hub_revision")
                  == record["subject"]["hub_revision"] else ["subject.hub_revision"])
        if not moved:
            print(f"unchanged: {out} already records revision "
                  f"{record['subject']['hub_revision'][:12]} "
                  f"({record['file_count']} files, {record['manifest_sha256'][:12]})")
            return 0
        raise SystemExit(
            f"REFUSED: {out} exists and disagrees on {', '.join(moved)}. This "
            f"record dates an apparatus claim; overwriting it silently would "
            f"restate history. Re-run with --force only if you intend to "
            f"replace the claim, and say why in the commit message."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"  revision          {record['subject']['hub_revision']}")
    print(f"  files             {record['file_count']}  "
          f"({record['total_bytes']:,} bytes)")
    print(f"  weights manifest  {record['weights']['manifest_sha256']}")
    print(f"  full manifest     {record['manifest_sha256']}")
    return 0


def cmd_verify(args) -> int:
    rec = json.loads(pathlib.Path(args.record).read_text())
    if rec.get("schema") != SCHEMA:
        raise SystemExit(f"REFUSED: {args.record} is schema {rec.get('schema')!r}, "
                         f"not {SCHEMA!r}")
    cache = pathlib.Path(args.hf_hub_cache).expanduser() if args.hf_hub_cache else hub_cache_root()
    repo_id = rec["subject"]["hub_repo_id"]
    rev = rec["subject"]["hub_revision"]
    snap = snapshot_dir(cache, repo_id, rev)

    # Recomputed from the record, not trusted: a record whose own aggregate
    # disagrees with its own file list is corrupt no matter what is on disk.
    for key, roles in (("manifest_sha256", None),
                       ("weights", ("weight_shard", "weight_index"))):
        want = rec[key] if key == "manifest_sha256" else rec["weights"]["manifest_sha256"]
        got = manifest_digest(rec["files"], roles)
        if want != got:
            raise SystemExit(f"REFUSED: {args.record} is internally inconsistent: "
                            f"{key} says {want[:12]}, its file list digests to {got[:12]}")

    problems: list[str] = []
    if not snap.is_dir():
        problems.append(f"snapshot {snap} is absent, so the pinned bytes are not here")
    else:
        on_disk = {}
        for path in sorted(p for p in snap.rglob("*") if not p.is_dir()):
            on_disk[path.relative_to(snap).as_posix()] = path
        for e in rec["files"]:
            path = on_disk.pop(e["path"], None)
            if path is None:
                problems.append(f"{e['path']}: recorded but MISSING")
                continue
            target = path.resolve()
            if not target.exists():
                problems.append(f"{e['path']}: dangling symlink -> {target}")
                continue
            digest, size = sha256_file(target)
            if digest != e["sha256"]:
                problems.append(f"{e['path']}: sha256 {digest[:12]} != recorded "
                                f"{e['sha256'][:12]} -- THE SUBJECT MOVED")
            elif size != e["size_bytes"]:
                problems.append(f"{e['path']}: {size} bytes != recorded {e['size_bytes']}")
        for extra in sorted(on_disk):
            problems.append(f"{extra}: present in the snapshot but NOT recorded")

    # Upstream drift is reported, never treated as a failure: the whole point of
    # pinning by revision is that `main` is allowed to move underneath us.
    ref = cache / repo_folder(repo_id) / "refs" / "main"
    if ref.exists():
        now = ref.read_text().strip()
        state = "unchanged" if now == rev else f"MOVED to {now}"
        print(f"refs/main: {state} (pinned {rev[:12]})")
        if now != rev:
            print("  the pin still holds -- this is exactly the silent "
                  "substitution the record exists to make visible")

    if problems:
        print(f"FAIL: {len(problems)} problem(s) against {args.record}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {rec['file_count']} files, {rec['total_bytes']:,} bytes match "
          f"{args.record}")
    print(f"  revision {rev}")
    print(f"  manifest {rec['manifest_sha256']}")
    return 0


def cmd_resolve(args) -> int:
    rec = json.loads(pathlib.Path(args.record).read_text())
    cache = pathlib.Path(args.hf_hub_cache).expanduser() if args.hf_hub_cache else hub_cache_root()
    snap = cache / rec["offline_resolution"]["snapshot_relpath"]
    print(snap)
    return 0 if snap.is_dir() else 1


def cmd_env(args) -> int:
    rec = json.loads(pathlib.Path(args.record).read_text())
    cache = pathlib.Path(args.hf_hub_cache).expanduser() if args.hf_hub_cache else hub_cache_root()
    snap = cache / rec["offline_resolution"]["snapshot_relpath"]
    print(f"export HF_HUB_CACHE={cache}")
    print("export HF_HUB_OFFLINE=1")
    print(f"export AGENTLAB_MODEL_REVISION={rec['subject']['hub_revision']}")
    print(f"export AGENTLAB_MODEL_SNAPSHOT={snap}")
    print("# loaders do not read these yet -- see "
          "offline_resolution.loader_revision_propagation in the record.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-hub-cache", default=None,
                    help="hub cache root (default: HF_HUB_CACHE / HF_HOME/hub / "
                         "~/.cache/huggingface/hub)")
    ap.add_argument("--record", default=str(DEFAULT_RECORD))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="write the revision + byte record")
    r.add_argument("--repo-id", default=None)
    r.add_argument("--revision", default=None)
    r.add_argument("--output", default=str(DEFAULT_RECORD))
    r.add_argument("--license", default="apache-2.0",
                   help="SPDX id from the Hub model card (default: apache-2.0, "
                        "which is what Qwen/Qwen3.5-4B declares)")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_record)

    v = sub.add_parser("verify", help="re-hash the cache against the record")
    v.set_defaults(fn=cmd_verify)

    s = sub.add_parser("resolve", help="print the pinned snapshot directory")
    s.set_defaults(fn=cmd_resolve)

    e = sub.add_parser("env", help="print shell exports for an offline load")
    e.set_defaults(fn=cmd_env)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
