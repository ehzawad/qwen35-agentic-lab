"""Shared helpers for the suite v1 tests: tiny deterministic task bundles."""

from __future__ import annotations

import pytest

from agentlab.suite.generate import build_task

SUITE = "agentlab-suite-v1"
SEED = 0xA61E0005  # the committed eval seed; tests never depend on GPU state


def mk_bundle(family: str, horizon: int, entries=None, split: str = "eval",
              index: int = 0):
    """One deterministic TaskBundle; entries = [(fault_type, ambiguous)]."""
    return build_task(SUITE, SEED, split, family, horizon, index, entries)


@pytest.fixture
def bundle_factory():
    return mk_bundle
