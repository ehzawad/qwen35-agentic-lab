#!/usr/bin/env python
"""Generate suite v1: specs, per-task KBs, oracle DAGs, manifest, SHA256SUMS.

CPU-only and fully deterministic: rerunning this script produces byte-identical
artifacts (validate_suite.py enforces that). Usage:

    PYTHONPATH=src .venv/bin/python scripts/generate_suite.py \
        [--config configs/suite_v1.toml] [--out data/suite/v1]
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/suite_v1.toml")
    ap.add_argument("--out", default=None,
                    help="output directory (default: [layout].out_dir)")
    args = ap.parse_args()

    from agentlab.suite.generate import generate_all, load_suite_config

    cfg = load_suite_config(args.config)
    out_dir = args.out or cfg["out_dir"]
    manifest = generate_all(cfg, out_dir)

    total = 0
    for split, meta in manifest["splits"].items():
        print(f"  {split:<12} {meta['count']:>6} specs "
              f"(seed {meta['seed']}, templates {meta['template_ids']})")
        total += meta["count"]
    print(f"  {'total':<12} {total:>6} specs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
