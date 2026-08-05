#!/usr/bin/env python
"""Write the S16/S18 lock and reveal receipts. CPU only, no GPU, no network.

The verdict analyzer refuses to certify test-blindness (S18) unless two files
exist and are ordered:

  results/agentic/locks.json        the prompt winner and the trained checkpoint,
                                    each with a timestamp and the commit that
                                    locked it
  results/agentic/seed_reveal.json  written strictly AFTER the last lock, and
                                    carrying the held-out seed derived from the
                                    preregistration commit:
                                      SHA256(<prereg commit> + ':agentic-heldout-v1')[:8]

The derivation is the point: the preregistration commit is the commit that first
introduced configs/agentic_preregister.json, and changing any frozen content
changes that sha, so the seed cannot be chosen to favour an arm. This script
DERIVES that commit from git history rather than accepting it as an argument.

Ordering is enforced here, not just checked later: `reveal` refuses to run
before both locks exist, and `lock-checkpoint` refuses to overwrite a lock that a
reveal has already been issued against -- locking a different checkpoint after
seeing held-out data is the exact failure S18 exists to catch.

    lock-prompt      --file configs/frozen_prompt.json
    lock-checkpoint  --path out/multiface/rssft-lora --stage rs_sft
    reveal
    status
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "agentic"
LOCKS = RESULTS / "locks.json"
REVEAL = RESULTS / "seed_reveal.json"
PREREG = ROOT / "configs" / "agentic_preregister.json"
HELDOUT_LABEL = ":agentic-heldout-v1"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=30, check=True).stdout.strip()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def preregistration_commit() -> str:
    """The commit that introduced configs/agentic_preregister.json."""
    rel = PREREG.relative_to(ROOT).as_posix()
    out = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
    shas = [line for line in out.splitlines() if line]
    if not shas:
        raise SystemExit(f"cannot find the commit that added {rel}; "
                         f"the preregistration must be committed before locking")
    return shas[-1]  # oldest addition


def heldout_seed(commit: str) -> int:
    return int.from_bytes(
        hashlib.sha256((str(commit) + HELDOUT_LABEL).encode()).digest()[:8], "big")


def read_locks() -> dict:
    return json.loads(LOCKS.read_text(encoding="utf-8")) if LOCKS.exists() else {}


def write_locks(locks: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOCKS.write_text(json.dumps(locks, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")


def _refuse_if_revealed(what: str) -> None:
    if REVEAL.exists():
        raise SystemExit(
            f"REFUSED: {REVEAL.relative_to(ROOT)} already exists, so held-out "
            f"results have been unblinded. Changing the {what} lock now is the "
            f"S18 violation this script exists to prevent. A genuinely new "
            f"candidate needs a dated AMENDMENT and fresh held-out seeds.")


def cmd_lock_prompt(args) -> int:
    _refuse_if_revealed("prompt_winner")
    frozen = json.loads(pathlib.Path(args.file).read_text(encoding="utf-8"))
    winner = frozen.get("winner") or frozen
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    block = prereg["prompt_candidates"]
    # `prompt_control finalize` records the winner as the candidate FILE NAME plus
    # the file's sha256; the directory is the preregistered one, so the path is
    # derived rather than trusted from the tournament output.
    name = winner.get("candidate") or winner.get("file") or winner.get("path")
    sha = winner.get("sha256")
    if not name or not sha:
        raise SystemExit(f"{args.file} carries no winner candidate/sha256; run "
                         f"`python -m agentlab.prompt_control finalize` first")
    path = f"{block['directory']}/{pathlib.PurePosixPath(name).name}"
    registered = set(block["sha256"].values())
    if sha not in registered:
        raise SystemExit(f"REFUSED: prompt sha {sha[:12]} is not one of the eight "
                         f"preregistered candidates (S16)")
    on_disk = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    if on_disk != sha:
        raise SystemExit(f"REFUSED: {path} on disk hashes {on_disk[:12]}, the "
                         f"tournament recorded {sha[:12]}")
    locks = read_locks()
    locks["prompt_winner"] = {"file": path, "sha256": sha, "locked_at": now(),
                              "commit": _git("rev-parse", "HEAD")}
    write_locks(locks)
    print(f"locked prompt winner {path} ({sha[:12]})")
    return 0


def cmd_lock_checkpoint(args) -> int:
    _refuse_if_revealed("checkpoint")
    p = ROOT / args.path
    if not p.exists():
        raise SystemExit(f"REFUSED: {args.path} does not exist")
    locks = read_locks()
    locks["checkpoint"] = {"path": args.path, "stage": args.stage,
                           "locked_at": now(),
                           "commit": _git("rev-parse", "HEAD")}
    write_locks(locks)
    print(f"locked checkpoint {args.path} (selected stage: {args.stage})")
    return 0


def cmd_reveal(args) -> int:
    locks = read_locks()
    missing = [k for k in ("prompt_winner", "checkpoint") if k not in locks]
    if missing:
        raise SystemExit(f"REFUSED: cannot reveal the held-out seed before "
                         f"locking {missing}; that ordering IS S18")
    if REVEAL.exists():
        print(f"already revealed: {REVEAL.relative_to(ROOT)}")
        return 0
    commit = preregistration_commit()
    payload = {"revealed_at": now(), "preregistration_commit": commit,
               "heldout_seed": heldout_seed(commit),
               "derivation": f"int.from_bytes(SHA256(commit + '{HELDOUT_LABEL}')"
                             f"[:8], 'big')"}
    latest = max(locks[k]["locked_at"] for k in ("prompt_winner", "checkpoint"))
    if payload["revealed_at"] <= latest:
        # Same-second locking would make the reveal look non-strictly-after.
        raise SystemExit(f"REFUSED: reveal timestamp {payload['revealed_at']} is "
                         f"not strictly after the last lock {latest}; wait a "
                         f"second and retry rather than backdating anything")
    RESULTS.mkdir(parents=True, exist_ok=True)
    REVEAL.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"revealed held-out seed for preregistration commit {commit[:12]}")
    return 0


def cmd_status(args) -> int:
    locks = read_locks()
    for key in ("prompt_winner", "checkpoint"):
        rec = locks.get(key)
        print(f"  {key:<14} {'-- not locked' if not rec else json.dumps(rec)}")
    print(f"  {'reveal':<14} "
          + ("-- not revealed" if not REVEAL.exists()
             else REVEAL.read_text(encoding='utf-8').strip().replace("\n", " ")))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("lock-prompt")
    lp.add_argument("--file", default="configs/frozen_prompt.json")
    lp.set_defaults(fn=cmd_lock_prompt)
    lc = sub.add_parser("lock-checkpoint")
    lc.add_argument("--path", required=True)
    lc.add_argument("--stage", default="rs_sft", choices=("rs_sft", "grpo"))
    lc.set_defaults(fn=cmd_lock_checkpoint)
    sub.add_parser("reveal").set_defaults(fn=cmd_reveal)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
