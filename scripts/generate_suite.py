#!/usr/bin/env python
"""Generate ONE suite v1 phase. There is no --phase all, and no --seed.

Generation is two phases with two seed sources (D3, the post-lock held-out state
machine):

    # before any lock: the public committed seeds
    PYTHONPATH=src .venv/bin/python scripts/generate_suite.py --phase train-dev

    # only after L (the dedicated commit adding the complete locks.json) is
    # published and `agentic_locks.py reveal` has written the receipt
    PYTHONPATH=src .venv/bin/python scripts/generate_suite.py --phase heldout \
        --reveal results/agentic/seed_reveal.json

    # after scripts/export_eval_specs.py has written the phase's certspecs, pin
    # them into the phase commitment (idempotent)
    PYTHONPATH=src .venv/bin/python scripts/generate_suite.py --phase train-dev --seal

    # move the retired public-seed held-out bytes out of the consumed tree
    PYTHONPATH=src .venv/bin/python scripts/generate_suite.py --quarantine-stale-heldout

`--phase` is required: the retired default emitted all ten splits in one pass,
which is exactly how all 1,200 held-out answers became derivable from the
preregistration commit alone. The held-out phase REFUSES without a verified
reveal receipt and creates no staging directory, so a failed check leaves no
partial held-out files. There is no way to supply a held-out seed here -- not by
flag, not by environment variable, not by fallback.

CPU-only and fully deterministic within a phase: rerunning a phase produces
byte-identical artifacts (scripts/validate_suite.py enforces that).
"""

from __future__ import annotations

import argparse
import os
import sys

# Any of these in the environment is an attempt to supply a seed the derivation
# is supposed to own. Refused rather than ignored.
_FORBIDDEN_SEED_ENV = ("AGENTLAB_HELDOUT_SEED", "HELDOUT_SEED",
                       "AGENTLAB_SUITE_SEED", "SUITE_SEED",
                       "AGENTLAB_HELDOUT_MASTER_SEED")


def _refuse_env_seeds() -> None:
    supplied = sorted(k for k in _FORBIDDEN_SEED_ENV if os.environ.get(k))
    if supplied:
        raise SystemExit(
            f"REFUSED: the environment supplies {supplied}. Held-out seeds derive "
            f"from L through the frozen derivation and from nowhere else; an "
            f"unregistered seed input is the one thing this generator must never "
            f"accept.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/suite_v1.toml")
    ap.add_argument("--out", default=None,
                    help="output directory (default: [layout].out_dir)")
    ap.add_argument("--phase", choices=("train-dev", "heldout"), default=None,
                    help="REQUIRED for generation; there is deliberately no 'all'")
    ap.add_argument("--reveal", default=None,
                    help="the reveal receipt (required for --phase heldout)")
    ap.add_argument("--seal", action="store_true",
                    help="pin the phase's exported certspecs into its commitment")
    ap.add_argument("--quarantine-stale-heldout", action="store_true",
                    help="move held-out bytes that belong to no verified release "
                         "out of the consumed tree, with a receipt")
    args = ap.parse_args()

    _refuse_env_seeds()

    from agentlab.suite.generate import (PHASES, generate_phase, heldout_release,
                                         load_reveal, load_suite_config,
                                         quarantine_stale_heldout, seal_phase,
                                         stale_heldout_paths)

    cfg = load_suite_config(args.config)
    out_dir = args.out or cfg["out_dir"]

    if args.quarantine_stale_heldout:
        stale = stale_heldout_paths(out_dir)
        if not stale:
            print(f"nothing to quarantine: {out_dir} holds no unreleased held-out "
                  f"files")
            return 0
        receipt = quarantine_stale_heldout(out_dir)
        print(f"quarantined {len(receipt['files'])} file(s) from {out_dir}")
        for rel, meta in sorted(receipt["files"].items()):
            print(f"  {meta['sha256'][:12]}  {meta['bytes']:>10}  {rel}")
        print(f"  -> {receipt['quarantined_to']}  (see QUARANTINE.json)")
        print("They were generated from the RETIRED public held-out seeds and can "
              "never become the designated set: the registered derivation is now "
              f"{receipt['current_derivation']} over L, and every held-out spec "
              "and certspec carries that release's id.")
        return 0

    if not args.phase:
        raise SystemExit(
            "REFUSED: --phase is required. Generation is two phases with two seed "
            "sources:\n"
            "  --phase train-dev                       public committed seeds\n"
            "  --phase heldout --reveal <receipt>      seeds derived from L\n"
            "There is no --phase all: emitting every split in one pass is the "
            "behaviour that made the held-out answers derivable before any lock.")

    reveal = None
    if args.phase == "heldout":
        if not args.reveal and not args.seal:
            raise SystemExit(
                "REFUSED: --phase heldout needs --reveal <results/agentic/"
                "seed_reveal.json>. The receipt is produced by `agentic_locks.py "
                "reveal`, which refuses until L -- the dedicated commit adding the "
                "complete results/agentic/locks.json -- is committed and published.")
        if args.reveal:
            reveal = load_reveal(args.reveal)
    elif args.reveal:
        raise SystemExit("REFUSED: --reveal belongs to --phase heldout only")

    if args.seal:
        if args.phase == "heldout" and heldout_release(out_dir) is None:
            raise SystemExit("REFUSED: there is no held-out release to seal")
        manifest = seal_phase(cfg, out_dir, args.phase, reveal=reveal)
        print(f"sealed {args.phase}: {len(manifest['files'])} source files + "
              f"{len(manifest['certspecs'])} certspec files")
        for rel, meta in sorted(manifest["certspecs"].items()):
            print(f"  {meta['sha256'][:12]}  {meta['bytes']:>10}  {rel}")
        print(f"  -> {out_dir}")
        return 0

    manifest = generate_phase(cfg, out_dir, args.phase, reveal=reveal)
    total = 0
    for split in PHASES[args.phase]:
        meta = manifest["splits"][split]
        # The held-out seeds are full 256-bit values; only a prefix is displayed,
        # and the manifest carries the whole thing.
        seed = meta["seed"]
        shown = seed if len(seed) <= 18 else f"{seed[:18]}..."
        print(f"  {split:<12} {meta['count']:>6} specs "
              f"(seed {shown}, templates {meta['template_ids']})")
        total += meta["count"]
    print(f"  {'total':<12} {total:>6} specs -> {out_dir}")
    if args.phase == "heldout":
        print(f"  release {manifest['heldout_release_id']}")
        print(f"  from L  {manifest['locks_commit']}")
    print(f"  commitment {manifest['phase']}: unsealed until the certspecs are "
          f"exported and `--phase {args.phase} --seal` has run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
