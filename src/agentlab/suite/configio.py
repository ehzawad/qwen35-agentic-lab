"""Loader for configs/multifaceted.yaml, the single source of every count and gate.

Lives in its own module (not suite/__init__.py) so the training-path modules and
the measurement modules can evolve without editing each other's files.
"""

from __future__ import annotations

import functools
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "multifaceted.yaml"


@functools.lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    """Parse the multifaceted config once per path; callers must not mutate it."""
    import yaml

    p = pathlib.Path(path) if path else CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# GPU ledger: one line per completed chunk, hard ceiling for the whole run
# --------------------------------------------------------------------------

def ledger_path(cfg: dict | None = None) -> pathlib.Path:
    cfg = cfg or load_config()
    return ROOT / cfg["budget"]["ledger"]


def ledger_hours(cfg: dict | None = None) -> float:
    """Cumulative measured GPU hours so far (0.0 when no ledger exists yet)."""
    import json

    p = ledger_path(cfg)
    if not p.exists():
        return 0.0
    total = 0.0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            total += float(json.loads(line).get("minutes", 0.0)) / 60.0
    return total


def ledger_append(stage: str, minutes: float, cfg: dict | None = None) -> float:
    """Append one completed chunk; returns cumulative hours after the append."""
    import json

    cfg = cfg or load_config()
    p = ledger_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cumulative = ledger_hours(cfg) + minutes / 60.0
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": stage, "minutes": round(minutes, 2),
                             "cumulative_h": round(cumulative, 3)}) + "\n")
    return cumulative


def ledger_guard(stage: str, projected_minutes: float, cfg: dict | None = None) -> None:
    """Refuse to start GPU work whose projection would cross the hard ceiling."""
    cfg = cfg or load_config()
    ceiling = float(cfg["budget"]["gpu_hours_ceiling"])
    projected = ledger_hours(cfg) + projected_minutes / 60.0
    if projected > ceiling:
        raise SystemExit(
            f"BUDGET: {stage} projects {projected:.1f} GPU-hours against the "
            f"{ceiling:.0f}h ceiling; stopping before touching the card."
        )
