"""Every oracle trajectory executes to strict success, clean and faulted."""

from __future__ import annotations

import pytest

from agentlab.suite.runtime import run_oracle
from agentlab.suite.schema import CELLS

from .conftest import mk_bundle


@pytest.mark.parametrize("family,horizon", CELLS)
def test_clean_oracle_succeeds(family, horizon):
    b = mk_bundle(family, horizon)
    _, verdict = run_oracle(b.spec, b.kb, b.nodes)
    assert verdict.strict_success
    assert verdict.unique_valid_nodes == horizon
    assert verdict.excess_calls == 0
    assert verdict.fault_assigned == 0


@pytest.mark.parametrize("family,horizon,entries", [
    ("lookup_chain", 4, [("transient", False)]),
    ("lookup_chain", 8, [("malformed", False)]),
    ("lookup_chain", 12, [("rate_limit", False)]),
    ("typed_relay", 2, [("wrong_unit", False)]),
    ("typed_relay", 8, [("malformed", False)]),
    ("typed_relay", 12, [("rate_limit", False)]),
    ("fulfillment", 4, [("transient", False)]),
    ("fulfillment", 4, [("malformed", True)]),
    ("fulfillment", 8, [("wrong_unit", False)]),
    ("fulfillment", 8, [("malformed", True)]),
    ("fulfillment", 14, [("rate_limit", False)]),
    ("fulfillment", 20, [("malformed", True)]),
])
def test_faulted_oracle_recovers(family, horizon, entries):
    b = mk_bundle(family, horizon, entries)
    _, verdict = run_oracle(b.spec, b.kb, b.nodes)
    assert verdict.strict_success
    assert verdict.faults_triggered == len(entries)
    assert verdict.recovered
    assert verdict.recovery_success
    assert verdict.excess_calls == len(entries)  # exactly one retry per fault


@pytest.mark.parametrize("family,horizon,entries", [
    ("lookup_chain", 8, [("transient", False), ("rate_limit", False)]),
    ("typed_relay", 12, [("wrong_unit", False), ("malformed", False)]),
    ("fulfillment", 20, [("malformed", True), ("wrong_unit", False)]),
])
def test_two_fault_stress_oracle(family, horizon, entries):
    b = mk_bundle(family, horizon, entries, split="eval_stress")
    _, verdict = run_oracle(b.spec, b.kb, b.nodes)
    assert verdict.strict_success
    assert verdict.faults_triggered == 2
    assert verdict.excess_calls == 2


def test_clean_arm_of_faulted_spec():
    """The paired counterfactual: the same base task runs clean."""
    b = mk_bundle("fulfillment", 8, [("wrong_unit", False)])
    clean = b.spec.without_faults()
    assert clean.faults == []
    assert clean.max_decisions == 8 + 3
    _, verdict = run_oracle(clean, b.kb, b.nodes)
    assert verdict.strict_success
    assert verdict.fault_assigned == 0
