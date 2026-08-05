#!/usr/bin/env python
"""Export committed suite bundles into the certification-layer spec contract.

`agentlab.suite.evaluate` consumes a flat JSONL spec manifest (one
`generate.certification_spec(bundle)` row per line). This is the ONLY writer of
that manifest: it reads the committed specs/kb/oracles through
`generate.load_bundles` and adapts them, so the evaluation arms and the training
path cannot end up looking at differently-built tasks.

The output is a derived view of committed data, so it is regenerated rather than
committed, and it carries its own SHA256SUMS so a shard run can prove which
bytes it evaluated.

    PYTHONPATH=src .venv/bin/python scripts/export_eval_specs.py \
        [--data data/suite/v1] [--out data/suite/v1/certspecs] \
        [--splits eval eval_mt eval_h8 eval_absent eval_perm ...]

With no --splits it exports every registered split, including the four
claim/control splits (eval_mt, eval_h8, eval_absent, eval_perm). The two control
splits are already IN their controlled form: eval_absent rows carry
`control: "redacted"` with the family-specific redaction applied, and eval_perm
rows carry `control: "permuted"` with the committed derangement applied, so the
arm that runs them must not transform them a second time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


# The split-leakage veto (S10) is about the train / dev / eval GROUPS, not about
# the individual splits: the three training splits deliberately share template
# ids 0-7, and eval shares 10-11 with eval_stress, so handing the analyzer one
# manifest per split makes it report template overlap as a harness BUG and no
# verdict can ever issue. `group_of` is the same grouping scripts/validate_suite.py
# uses, so both checkers mean the same thing by "split".
# Taken from the generator so a newly registered split cannot end up ungrouped
# and therefore silently excluded from the leakage veto.
def _group_of() -> dict:
    from agentlab.suite.generate import SPLIT_KIND

    return dict(SPLIT_KIND)


def _write(path: str, rows) -> dict:
    # Written the same way the suite writes its own artifacts: sorted keys,
    # compact separators, trailing newline -- so the export is byte-stable.
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False) + "\n")
    return {"rows": len(rows), "bytes": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def export(data_dir: str, out_dir: str, splits, groups: bool = True) -> dict:
    from agentlab.suite.generate import certification_spec, load_bundles

    os.makedirs(out_dir, exist_ok=True)
    group_of = _group_of()
    written = {}
    by_group: dict = {}
    for split in splits:
        rows = [certification_spec(b) for b in load_bundles(data_dir, split)]
        written[f"{split}.jsonl"] = _write(
            os.path.join(out_dir, f"{split}.jsonl"), rows)
        by_group.setdefault(group_of.get(split, split), []).extend(rows)
    if groups:
        gdir = os.path.join(out_dir, "groups")
        os.makedirs(gdir, exist_ok=True)
        for group, rows in sorted(by_group.items()):
            written[f"groups/{group}.jsonl"] = _write(
                os.path.join(gdir, f"{group}.jsonl"), rows)
    sums = "".join(f"{meta['sha256']}  {rel}\n"
                   for rel, meta in sorted(written.items()))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write(sums)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/suite/v1")
    ap.add_argument("--out", default=None, help="default: <data>/certspecs")
    # Every split by default. The evaluation arms need dev/eval/eval_stress plus
    # the four registered claim/control splits (eval_mt, eval_h8, eval_absent,
    # eval_perm), and the verdict's S10 split-leakage veto compares whatever
    # manifests it is given, so exporting the training splits too lets it check
    # every pair rather than only dev-vs-eval.
    ap.add_argument("--splits", nargs="+", default=None,
                    help="default: every registered split")
    ap.add_argument("--no-groups", action="store_true",
                    help="skip the merged train/dev/eval group manifests")
    args = ap.parse_args()

    from agentlab.suite.generate import SPLITS

    out_dir = args.out or os.path.join(args.data, "certspecs")
    written = export(args.data, out_dir, args.splits or list(SPLITS),
                     groups=not args.no_groups)
    for rel, meta in sorted(written.items()):
        print(f"  {rel:<20} {meta['rows']:>6} specs  {meta['bytes']:>10} bytes  "
              f"{meta['sha256'][:12]}")
    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
