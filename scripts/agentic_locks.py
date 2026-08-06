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

The derivation is the point: changing any frozen content changes the anchor sha,
so the seed cannot be chosen to favour an arm. This script DERIVES that commit
from git history rather than accepting it as an argument.

THE ANCHOR, AND WHY IT NEEDED FIXING. The anchor used to be the OLDEST commit
that added configs/agentic_preregister.json. That commit is fixed forever, so
every later edit to the preregistration -- including the whole A5000 hardware
pivot -- left the commitment untouched while the documentation claimed the
opposite. The fix is a FINALIZATION MARKER, not a history rewrite:

    finalize-prereg   hash every frozen preregistration file into
                      configs/preregistration_final.json; the operator commits
                      that file, and its ADDING commit becomes the anchor
    verify-prereg     re-hash and refuse if any frozen file drifted afterwards

So the anchor is the commit that pins the COMPLETED preregistration, forward
edits after finalization are detected rather than ignored, and no commit is ever
rewritten. Until the marker exists the old oldest-addition anchor is used and
`reveal` says so out loud (and refuses under --require-finalization, which the
chain passes).

Ordering is enforced here, not just checked later: `reveal` refuses to run
before both locks exist, and `lock-checkpoint` refuses to overwrite a lock that a
reveal has already been issued against -- locking a different checkpoint after
seeing held-out data is the exact failure S18 exists to catch.

A checkpoint lock pins the checkpoint's BYTE DIGEST and the trainer's receipt, not
a mutable path: see `cmd_lock_checkpoint` below.

    lock-prompt        --file configs/frozen_prompt.json
    lock-checkpoint    --path out/multiface/rssft-lora --stage rs_sft
    grpo-disposition   [--verify]   the registered stage-disposition receipt
    finalize-prereg
    verify-prereg
    reveal [--require-finalization]
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
FINAL_MARKER = ROOT / "configs" / "preregistration_final.json"
HELDOUT_LABEL = ":agentic-heldout-v1"

# The frozen set the marker hash-pins. Everything a claim depends on and nothing
# that a normal day's work touches: the preregistration, the protocol, the
# operational config that carries the counts, gates, hardware and engine
# contract, and the suite size authority.
FROZEN_SET = ("configs/agentic_preregister.json",
              "docs/AGENTIC_PROTOCOL.md",
              "configs/multifaceted.yaml",
              "configs/suite_v1.toml")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=30, check=True).stdout.strip()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes() -> dict:
    out = {}
    for rel in FROZEN_SET:
        p = ROOT / rel
        if p.exists():
            out[rel] = _sha256_file(p)
    return out


def _added_commit(rel: str) -> str | None:
    out = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
    shas = [line for line in out.splitlines() if line]
    return shas[-1] if shas else None


def finalization() -> dict | None:
    """The committed finalization marker, or None."""
    if not FINAL_MARKER.exists():
        return None
    return json.loads(FINAL_MARKER.read_text(encoding="utf-8"))


def verify_finalization(strict: bool = True) -> dict:
    """Re-hash the frozen set against the marker.

    A drifted file is refused rather than reported: the whole value of a
    finalization marker is that edits AFTER it are visible.
    """
    marker = finalization()
    if marker is None:
        return {"present": False, "anchor": "oldest-addition-fallback",
                "drifted": [], "missing": []}
    recorded = marker.get("files") or {}
    current = frozen_hashes()
    drifted = sorted(k for k, v in recorded.items() if current.get(k) != v)
    missing = sorted(k for k in recorded if k not in current)
    committed = _added_commit(FINAL_MARKER.relative_to(ROOT).as_posix())
    result = {"present": True, "finalized_at": marker.get("finalized_at"),
              "files": recorded, "drifted": drifted, "missing": missing,
              "commit": committed,
              "anchor": "finalization-marker" if committed else "uncommitted-marker"}
    if strict and (drifted or missing):
        raise SystemExit(
            f"REFUSED: the preregistration was finalized and has drifted since: "
            f"{drifted + missing}. The marker exists precisely so that a forward "
            f"edit cannot pass unnoticed. Either revert the edit, or file a dated "
            f"AMENDMENT and finalize again with fresh held-out seeds.")
    return result


def preregistration_commit() -> str:
    """The commit this run's held-out seed derives from.

    Preference: the commit that ADDED configs/preregistration_final.json, which
    hash-pins the COMPLETED preregistration. Fallback: the oldest commit that
    added configs/agentic_preregister.json -- historically the only anchor, and
    the reason forward edits did not change the commitment.
    """
    state = verify_finalization()
    if state["present"] and state.get("commit"):
        return state["commit"]
    rel = PREREG.relative_to(ROOT).as_posix()
    sha = _added_commit(rel)
    if not sha:
        raise SystemExit(f"cannot find the commit that added {rel}; "
                         f"the preregistration must be committed before locking")
    return sha


def cmd_finalize_prereg(args) -> int:
    """Write the finalization marker. The OPERATOR commits it; that is the anchor."""
    if REVEAL.exists():
        raise SystemExit(
            f"REFUSED: {REVEAL.relative_to(ROOT)} exists, so held-out results are "
            f"already unblinded. Finalizing the preregistration now would move the "
            f"anchor -- and the seed -- after the fact.")
    existing = finalization()
    if existing and not args.force:
        state = verify_finalization()
        print(json.dumps(state, indent=2, sort_keys=True))
        print(f"already finalized: {FINAL_MARKER.relative_to(ROOT)}")
        return 0
    hashes = frozen_hashes()
    missing = [rel for rel in FROZEN_SET if rel not in hashes]
    payload = {
        "finalized_at": now(),
        "head_at_finalization": _git("rev-parse", "HEAD"),
        "files": hashes,
        "absent": missing,
        "note": "The commit that ADDS this file is the preregistration anchor: "
                "the held-out seed derives from it, so it pins the COMPLETED "
                "preregistration rather than the first commit that happened to "
                "introduce a preregistration file. No history is rewritten.",
        "seed_derivation": f"int.from_bytes(SHA256(<this file's adding commit> + "
                           f"'{HELDOUT_LABEL}')[:8], 'big')",
    }
    FINAL_MARKER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(f"wrote {FINAL_MARKER.relative_to(ROOT)} pinning {len(hashes)} frozen files")
    for rel, sha in sorted(hashes.items()):
        print(f"  {sha[:12]}  {rel}")
    print("\nCOMMIT IT -- an uncommitted marker anchors nothing:")
    print(f"  git add {FINAL_MARKER.relative_to(ROOT)}")
    print('  git commit -m "Finalize the preregistration: hash-pin the completed '
          'commitment"')
    return 0


def grpo_disposition_payload(cfg: dict) -> dict:
    """The registered GRPO stage-disposition receipt.

    A stage DISPOSITION says what happened to a stage; it is never one of the gate
    states PASS/FAIL/INCONCLUSIVE/BUG. `GRPO_NOT_RUN_HARDWARE_INFEASIBLE` says the
    registered trainer cannot instantiate on the registered card and says NOTHING
    about whether the variance gate would have opened -- which is why it is not
    interchangeable with `GRPO_NOT_RUN_VARIANCE_GATE_CLOSED`, and why the variance
    gate is recorded as NOT_EVALUATED rather than closed.

    The arithmetic is part of the receipt: an artifact that does not SHOW the
    shortfall does not establish infeasibility.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import (config_hash, fingerprint, git_sha,
                                         hardware_contract, now_utc)

    grpo = cfg["grpo"]
    hw = hardware_contract(cfg)
    arith = dict(grpo["disposition_arithmetic"])
    if not arith["vllm_allocation_gib"] < arith["vllm_policy_copy_gib"]:
        raise SystemExit(
            "REFUSED: the disposition arithmetic does not show the shortfall "
            "(vllm_allocation_gib must be smaller than vllm_policy_copy_gib), so "
            "it does not establish infeasibility.")
    return {
        "branch": "grpo",
        "outcome": grpo["stage_disposition"],
        "variance_gate": cfg["variance_probe"]["stage_disposition"],
        "trainer_feasibility": "INFEASIBLE",
        "optimizer_steps": 0,
        "gpu_name": hw["expected_name"],
        "cuda_visible_bytes": hw["cuda_visible_bytes"],
        "config_hash": config_hash(),
        "checkpoint_hash": None,
        "git_sha": git_sha(),
        "arithmetic": arith,
        "timestamp_utc": now_utc(),
        "not_interchangeable_with": "GRPO_NOT_RUN_VARIANCE_GATE_CLOSED",
        "note": "A stage DISPOSITION, never a gate state. The registered colocate "
                "configuration cannot instantiate on the registered card, so no "
                "GRPO checkpoint exists, the R0/RP arms are ABSENT BY DESIGN, and "
                "RS-SFT is the sole trained candidate. No RS-SFT-versus-GRPO dev "
                "comparison is performed because there is no second candidate.",
        "forbidden_substitutions": ["microbatch 1", "2048-token completions",
                                    "no-vLLM (transformers) generation",
                                    "quantization", "CPU/disk offload",
                                    "an alternate optimizer", "another GPU model"],
        "provenance": fingerprint(),
    }


def require_grpo_disposition(cfg: dict, path: pathlib.Path) -> dict:
    """Read the artifact and refuse anything but the registered disposition.

    "GRPO was skipped for a reason nobody wrote down" is indistinguishable from
    "GRPO ran and lost", and only one of those is reportable -- so a missing or
    mislabelled artifact is an error here, never a silent selection of RS-SFT.
    """
    if not path.exists():
        raise SystemExit(f"REFUSED: no GRPO disposition artifact at {path}. A "
                         f"missing GRPO checkpoint with no disposition is an "
                         f"ERROR, not a silent fallback to RS-SFT.")
    rec = json.loads(path.read_text(encoding="utf-8"))
    want = cfg["grpo"]["stage_disposition"]
    if rec.get("outcome") != want:
        raise SystemExit(f"REFUSED: the disposition artifact says "
                         f"{rec.get('outcome')!r}; the registered disposition is "
                         f"{want!r}")
    gate = cfg["variance_probe"]["stage_disposition"]
    if rec.get("variance_gate") != gate:
        raise SystemExit(f"REFUSED: the artifact's variance_gate is "
                         f"{rec.get('variance_gate')!r}, registered {gate!r}. "
                         f"'closed' would claim the complete 144-group probe ran "
                         f"and a binding gate failed.")
    return rec


def _load_cfg() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import load_config

    return load_config()


def cmd_grpo_disposition(args) -> int:
    cfg = _load_cfg()
    path = ROOT / cfg["grpo"]["disposition_artifact"]
    if args.verify:
        rec = require_grpo_disposition(cfg, path)
        print(f"[grpo] {rec['outcome']} (variance gate: {rec['variance_gate']})")
        return 0
    if path.exists() and not args.force:
        rec = require_grpo_disposition(cfg, path)
        print(f"[grpo] already recorded: {rec['outcome']} -> {path}")
        return 0
    payload = grpo_disposition_payload(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"[grpo] {payload['outcome']} -> {path}")
    print(f"       variance gate: {payload['variance_gate']}")
    print(f"       vLLM carve {payload['arithmetic']['vllm_allocation_gib']} GiB "
          f"< its own policy copy {payload['arithmetic']['vllm_policy_copy_gib']} GiB")
    return 0


def cmd_verify_prereg(args) -> int:
    state = verify_finalization(strict=not args.report_only)
    print(json.dumps(state, indent=2, sort_keys=True))
    if not state["present"]:
        print("NOT FINALIZED: the anchor is still the oldest commit that added "
              "the preregistration, so forward edits do not change the "
              "commitment. Run `finalize-prereg` and commit the marker.")
        return 1 if args.require else 0
    if state["anchor"] == "uncommitted-marker":
        print("MARKER NOT COMMITTED: it anchors nothing until it is in a commit.")
        return 1 if args.require else 0
    print(f"finalized and clean; anchor commit {state['commit'][:12]}")
    return 0


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


# --------------------------------------------------------------------------
# the checkpoint lock: BYTES, and the chain that produced them
# --------------------------------------------------------------------------
#
# The lock used to record {path, stage, locked_at, commit}. A path is mutable: the
# adapter behind it can be retrained, overwritten or swapped after the held-out
# seed is revealed, and the lock would still "hold". Worse, nothing connected the
# locked adapter to the trajectories, the card or the run that produced it, so
# S18 blindness and S19 hardware integrity both stopped at this file.
#
# A lock now pins the checkpoint's CONTENT DIGEST and the digest of the trainer's
# receipt, and it refuses to exist without them. `verify_checkpoint_lock` re-hashes
# the bytes, so a post-lock swap is detected rather than assumed away.

CHECKPOINT_LOCK_FIELDS = ("path", "stage", "locked_at", "commit",
                          "checkpoint_sha256", "training_manifest",
                          "training_manifest_sha256", "gpu_uuid",
                          "environment_contract_sha256", "config_hash",
                          "views_sha256", "source_provenance")


def _sft_module():
    """The trainer owns the training-manifest grammar; this reads it, never a copy."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab import sft

    return sft


def cmd_lock_checkpoint(args) -> int:
    _refuse_if_revealed("checkpoint")
    p = ROOT / args.path
    if not p.exists():
        raise SystemExit(f"REFUSED: {args.path} does not exist")
    sft = _sft_module()
    cfg = _load_cfg()
    override = getattr(args, "training_manifest", None)
    manifest_path = (ROOT / override) if override else sft.training_manifest_path(p)
    # The whole chain, or no lock: the receipt hashes to itself, was written under
    # this environment contract, the checkpoint bytes still hash to the digest it
    # recorded, its card is this run's card and the ledger's card, and the corpus
    # it names was produced by an attested GPU session.
    rec = sft.require_training_manifest(manifest_path, checkpoint_path=p, cfg=cfg,
                                        stage=args.stage)
    inputs = rec["inputs"]
    locks = read_locks()
    locks["checkpoint"] = {
        "path": args.path,
        "stage": args.stage,
        "locked_at": now(),
        "commit": _git("rev-parse", "HEAD"),
        # WHAT is locked: the bytes, not the name of the directory holding them.
        "checkpoint_sha256": rec["checkpoint"]["checkpoint_sha256"],
        "checkpoint_files": rec["checkpoint"]["files"],
        # WHO produced them, and from what.
        "training_manifest": _rel(manifest_path),
        "training_manifest_sha256": rec[_hash_field()],
        "trained_at_utc": rec["finished_at_utc"],
        "optimizer_steps": rec["optimizer_steps"],
        "gpu_uuid": rec["hardware"]["gpu_uuid"],
        "gpu_name": rec["hardware"]["gpu_name"],
        "runtime_manifest_sha256": rec["runtime_manifest_sha256"],
        "session_id": rec["session_id"],
        "git_sha_trained": rec["git_sha"],
        "config_hash": rec["config_hash"],
        "environment_contract_sha256": rec["environment_contract_sha256"],
        "views_sha256": inputs["views_sha256"],
        "views_rows": inputs["views_rows"],
        "source_provenance": inputs["source_provenance"],
        "source_runtime_manifests": inputs["source_runtime_manifests"],
    }
    write_locks(locks)
    print(f"locked checkpoint {args.path} (selected stage: {args.stage})")
    print(f"  checkpoint sha256   {locks['checkpoint']['checkpoint_sha256']}")
    print(f"  training manifest   {locks['checkpoint']['training_manifest_sha256']}"
          f"  ({locks['checkpoint']['training_manifest']})")
    print(f"  trained on          {locks['checkpoint']['gpu_name']} "
          f"{locks['checkpoint']['gpu_uuid']}")
    print(f"  corpus              {inputs['views_rows']} rows, "
          f"{inputs['source_trajectories']} trajectories, "
          f"views {inputs['views_sha256'][:12]}")
    return 0


def _rel(path: pathlib.Path) -> str:
    p = pathlib.Path(path)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _hash_field() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from agentlab.suite.configio import MANIFEST_HASH_FIELD

    return MANIFEST_HASH_FIELD


def verify_checkpoint_lock(locks: dict | None = None, *, cfg: dict | None = None,
                           verify_bytes: bool = True) -> dict:
    """Re-check a recorded checkpoint lock. Used before any held-out verdict.

    A path-only lock -- the shape this script used to write -- is REFUSED here
    rather than reported, because a verdict that cites "the locked checkpoint"
    while the lock names only a mutable directory cites nothing.
    """
    locks = read_locks() if locks is None else locks
    rec = (locks or {}).get("checkpoint")
    if not rec:
        raise SystemExit("REFUSED: no checkpoint is locked; that ordering IS S18.")
    missing = [k for k in CHECKPOINT_LOCK_FIELDS
               if rec.get(k) is None or rec.get(k) == ""]
    if missing:
        raise SystemExit(
            f"REFUSED: the checkpoint lock is missing {', '.join(missing)}. A lock "
            f"that records only a PATH pins nothing: the bytes behind that path can "
            f"be retrained or swapped after the held-out seed is revealed, and "
            f"nothing ties them to the trajectories, the card or the run that "
            f"produced them. Re-lock with `agentic_locks.py lock-checkpoint`, which "
            f"requires the trainer's receipt.")
    sft = _sft_module()
    cfg = cfg or _load_cfg()
    if verify_bytes:
        sft.require_training_manifest(ROOT / rec["training_manifest"],
                                      checkpoint_path=ROOT / rec["path"],
                                      cfg=cfg, stage=rec["stage"])
        manifest = sft.read_training_manifest(ROOT / rec["training_manifest"])
        if manifest[_hash_field()] != rec["training_manifest_sha256"]:
            raise SystemExit(
                f"REFUSED: the training manifest now hashes "
                f"{manifest[_hash_field()]} and the lock recorded "
                f"{rec['training_manifest_sha256']}; the receipt changed after the "
                f"lock was taken.")
        if manifest["checkpoint"]["checkpoint_sha256"] != rec["checkpoint_sha256"]:
            raise SystemExit(
                f"REFUSED: the locked checkpoint digest {rec['checkpoint_sha256']} "
                f"is not the digest the receipt records "
                f"({manifest['checkpoint']['checkpoint_sha256']}).")
    return rec


def cmd_reveal(args) -> int:
    locks = read_locks()
    missing = [k for k in ("prompt_winner", "checkpoint") if k not in locks]
    if missing:
        raise SystemExit(f"REFUSED: cannot reveal the held-out seed before "
                         f"locking {missing}; that ordering IS S18")
    if REVEAL.exists():
        print(f"already revealed: {REVEAL.relative_to(ROOT)}")
        return 0
    state = verify_finalization()
    if getattr(args, "require_finalization", False) and (
            not state["present"] or state["anchor"] != "finalization-marker"):
        raise SystemExit(
            "REFUSED: the preregistration is not finalized (or its marker is not "
            "committed), so the held-out seed would anchor to the OLDEST commit "
            "that added a preregistration file -- an anchor that every later edit, "
            "including the whole hardware pivot, left untouched. Run "
            "`agentic_locks.py finalize-prereg`, commit the marker, then reveal.")
    if not state["present"]:
        print("WARNING: not finalized. The anchor is the oldest commit that added "
              "the preregistration, so forward edits did not change it.")
    commit = preregistration_commit()
    payload = {"revealed_at": now(), "preregistration_commit": commit,
               "heldout_seed": heldout_seed(commit),
               "anchor": state["anchor"],
               "finalization": state,
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
    state = verify_finalization(strict=False)
    print(f"  {'preregistration':<14} anchor={state['anchor']}"
          + (f" commit={state['commit'][:12]}" if state.get("commit") else "")
          + (f" DRIFTED={state['drifted']}" if state.get("drifted") else ""))
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
    lc.add_argument("--training-manifest", default=None,
                    help="the trainer's receipt (default: "
                         "<path>.agentlab_training_manifest.json). A checkpoint "
                         "with no receipt cannot be locked: the lock pins the "
                         "checkpoint's BYTE DIGEST and the chain that produced "
                         "them, never a mutable path.")
    lc.set_defaults(fn=cmd_lock_checkpoint)
    fp = sub.add_parser("finalize-prereg")
    fp.add_argument("--force", action="store_true",
                    help="re-pin the frozen set (a dated AMENDMENT, not a fixup)")
    fp.set_defaults(fn=cmd_finalize_prereg)
    gd = sub.add_parser("grpo-disposition")
    gd.add_argument("--verify", action="store_true",
                    help="read and check the artifact instead of writing it")
    gd.add_argument("--force", action="store_true")
    gd.set_defaults(fn=cmd_grpo_disposition)
    vp = sub.add_parser("verify-prereg")
    vp.add_argument("--require", action="store_true",
                    help="exit non-zero when the preregistration is not finalized")
    vp.add_argument("--report-only", action="store_true",
                    help="report drift instead of refusing")
    vp.set_defaults(fn=cmd_verify_prereg)
    rv = sub.add_parser("reveal")
    rv.add_argument("--require-finalization", action="store_true",
                    help="refuse unless a COMMITTED finalization marker anchors "
                         "the completed preregistration")
    rv.set_defaults(fn=cmd_reveal)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
