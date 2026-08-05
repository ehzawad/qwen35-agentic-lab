"""Isolated per-episode knowledge bases with a no-key-leak miss contract.

Every task owns its own KB view; nothing is shared between episodes and
nothing falls back to the global demo KB in agentlab.tools (whose miss message
lists every valid key -- that behavior must never reach suite episodes, since
it would leak the search space).

Miss contract (binding): a lookup miss returns ONLY

    {"ok": false, "error": "no_entry"}

never a key list, never a hint, never a partial match.
"""

from __future__ import annotations

MISS_PAYLOAD = {"ok": False, "error": "no_entry"}


class KBView:
    """Read-only view over one task's isolated key -> record mapping."""

    def __init__(self, entries: dict) -> None:
        # Copy so runtime instances can never alias generator state.
        self._entries = {str(k): v for k, v in (entries or {}).items()}
        # Keys in this suite are uppercase (base32 tokens, PREFIX-<digits>,
        # SPEC-...); accept a case-insensitive hit without leaking anything.
        self._upper = {k.upper(): k for k in self._entries}

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return self._entries.keys()

    def lookup(self, key: str) -> dict:
        """Semantic payload for one lookup. Misses return only no_entry."""
        k = str(key).strip()
        rec = self._entries.get(k)
        if rec is None:
            canonical = self._upper.get(k.upper())
            rec = self._entries.get(canonical) if canonical is not None else None
        if rec is None:
            return dict(MISS_PAYLOAD)
        return {"ok": True, "record": rec}


def load_split_kb(path: str) -> dict:
    """kb/<split>.json is {task_id: {key: record}}."""
    from .schema import read_json

    return read_json(path)
