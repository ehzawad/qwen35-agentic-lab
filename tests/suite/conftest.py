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

# THE SENTINEL. Held-out splits have no seed until L exists, so these tests
# exercise the GENERATOR MECHANISM with an obviously fake, clearly labelled seed.
# Nothing about the designated held-out realization is validated by using it: that
# happens once, at R, over the release the locks commit determines
# (scripts/validate_suite.py --require-phase heldout). The value is deliberately
# not one of the retired public seeds, so a test can never accidentally reproduce
# a quarantined old-seed bundle.
SENTINEL_HELDOUT_SEED = 0x5E4714E1_5E4714E1_5E4714E1_5E4714E1
SEED = SENTINEL_HELDOUT_SEED  # what mk_bundle draws with (split defaults to eval)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "suite_v1.toml")
# Read the committed TRAIN/DEV seeds rather than copying them: a duplicated seed
# table is how a split ends up generated from the wrong stream, or not at all.
# There are no held-out seeds to read -- the loader refuses a config that has any.
SEEDS = load_suite_config(CONFIG_PATH)["seeds"]


def seed_for(split: str) -> int:
    """The train/dev committed seed, or the labelled sentinel for a held-out split."""
    from agentlab.suite.generate import PHASE_OF, TRAIN_DEV_PHASE

    if PHASE_OF[split] == TRAIN_DEV_PHASE:
        return SEEDS[split]
    return SENTINEL_HELDOUT_SEED


def mk_bundle(family: str, horizon: int, entries=None, split: str = "eval",
              index: int = 0):
    """One deterministic TaskBundle; entries = [(fault_type, ambiguous)]."""
    return build_task(SUITE, SEED, split, family, horizon, index, entries)


@pytest.fixture
def bundle_factory():
    return mk_bundle
