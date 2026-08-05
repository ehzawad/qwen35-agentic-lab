"""Split-leakage checks (harness veto S10).

No train/dev/test overlap is tolerated in any of:

  * task IDs
  * KB namespaces and concrete KB keys
  * structural template hashes (template_id / template_hash)
  * graph-isomorphism signatures, when specs carry them

The checker is pure: it takes lists of spec dicts per split and returns every
violation it finds. Any violation is a BUG that vetoes all capability claims.
"""

from __future__ import annotations


_FIELDS = (
    ("task_id", "task_id"),
    ("kb_namespace", "kb_namespace"),
    ("template", "template_id"),
    ("template_hash", "template_hash"),
    ("graph_signature", "graph_signature"),
)


def _collect(specs: list[dict], field: str) -> set:
    out = set()
    for s in specs:
        v = s.get(field)
        if v is not None:
            out.add(v)
    return out


def _kb_keys(specs: list[dict]) -> set:
    keys = set()
    for s in specs:
        kb = s.get("kb")
        if isinstance(kb, dict):
            keys.update(kb.keys())
    return keys


def check_split_leakage(splits: dict[str, list[dict]]) -> list[dict]:
    """splits: {"train": [...specs], "dev": [...], "eval": [...]} -> violations."""
    names = sorted(splits)
    violations = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for label, field in _FIELDS:
                inter = _collect(splits[a], field) & _collect(splits[b], field)
                if inter:
                    violations.append({
                        "kind": label, "splits": [a, b],
                        "examples": sorted(str(x) for x in inter)[:5],
                        "count": len(inter),
                    })
            kb_inter = _kb_keys(splits[a]) & _kb_keys(splits[b])
            if kb_inter:
                violations.append({
                    "kind": "kb_key", "splits": [a, b],
                    "examples": sorted(str(x) for x in kb_inter)[:5],
                    "count": len(kb_inter),
                })
    return violations
