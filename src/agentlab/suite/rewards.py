"""Rewards for suite v1: binary strict evaluation, bounded GRPO shaping.

All REPORTED evaluation uses binary strict success only. The shaping reward
exists for GRPO training and is deliberately mechanical and bounded:

    if unsafe or invalid mutation: 0
    else: 0.8 * strict_success + 0.2 * unique_valid_oracle_nodes / H

Partial credit can never exceed 0.2, repeated nodes never add reward (the
verifier counts unique dependency-valid nodes only), and no failed trajectory
can outscore a valid success (a success scores >= 0.8; a failure <= 0.2).
"""

from __future__ import annotations


def binary_reward(verdict) -> float:
    """Strict binary success: the only reward evaluation ever reports."""
    return 1.0 if verdict.strict_success else 0.0


def shaped_reward(verdict, horizon: int | None = None) -> float:
    """Bounded mechanical shaping for GRPO only. Never reported as a result."""
    if verdict.unsafe_mutation or not verdict.consistent:
        return 0.0
    h = horizon if horizon is not None else verdict.nodes_total
    if h <= 0:
        return 0.0
    frac = min(verdict.unique_valid_nodes, h) / h
    return 0.8 * (1.0 if verdict.strict_success else 0.0) + 0.2 * frac
