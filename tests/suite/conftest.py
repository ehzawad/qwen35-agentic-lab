"""Shared helpers for the suite v1 tests: tiny deterministic task bundles."""

from __future__ import annotations

import pytest

import os

from agentlab.suite.generate import build_task, load_suite_config

SUITE = "agentlab-suite-v1"
# The run secret every suite test uses. Recovery tokens and receipts are keyed
# with it, so it is part of the model-visible observation bytes: a fixture
# without one could not mint the tokens the registered predicate requires.
SECRET = bytes.fromhex("5c" * 32)
SEED = 0xA61E0005  # the committed eval seed; tests never depend on GPU state

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "suite_v1.toml")
# Read the committed seeds rather than copying them: a duplicated seed table is
# how a new split ends up generated from the wrong stream, or not at all.
SEEDS = load_suite_config(CONFIG_PATH)["seeds"]


def mk_bundle(family: str, horizon: int, entries=None, split: str = "eval",
              index: int = 0):
    """One deterministic TaskBundle; entries = [(fault_type, ambiguous)]."""
    return build_task(SUITE, SEED, split, family, horizon, index, entries)


@pytest.fixture
def bundle_factory():
    return mk_bundle
