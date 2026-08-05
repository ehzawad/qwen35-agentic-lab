"""Statistics for the preregistered agentic gates.

Every procedure here is fixed by configs/agentic_preregister.json before any
held-out GPU result exists; the implementations are deterministic so the
verdict is a pure function of the traces and the committed seeds.

  * wilson           95% score interval (re-exported from agentlab.analyze)
  * mcnemar_exact    exact binomial McNemar on discordant pairs (all n)
  * holm             Holm step-down adjusted p-values
  * cluster_bootstrap_lb
                     one-sided template-clustered bootstrap lower confidence
                     bound for a paired risk difference, 100k deterministic
                     replicates from the committed SHAKE-256 stream
"""

from __future__ import annotations

import math

from agentlab.analyze import wilson  # noqa: F401  (single Wilson implementation)
from agentlab.suite import rng


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar on discordant pairs; b = only-first-arm success, c = only-second.

    Returns two-sided and one-sided (P[X >= c] under p=1/2, i.e. evidence that
    the SECOND arm wins) exact binomial p-values. Unlike analyze.mcnemar this
    never falls back to a normal approximation: the primary claim requires the
    exact test at any n.
    """
    n = b + c
    if n == 0:
        return {"n_discordant": 0, "b": b, "c": c, "p_two_sided": 1.0, "p_one_sided": 1.0}
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2**n
    two = min(1.0, 2 * tail)
    one = sum(math.comb(n, i) for i in range(c, n + 1)) / 2**n
    return {"n_discordant": n, "b": b, "c": c, "p_two_sided": two, "p_one_sided": one}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, val)  # enforce monotonicity
        adj[idx] = running
    return adj


def _quantile_lower(sorted_vals, alpha: float) -> float:
    """Committed quantile rule: the floor(alpha*R)-th order statistic (0-based).

    Deterministic and slightly conservative for a lower bound; preregistered in
    configs/agentic_preregister.json so it cannot be swapped post hoc.
    """
    r = len(sorted_vals)
    idx = int(math.floor(alpha * r))
    idx = min(max(idx, 0), r - 1)
    return float(sorted_vals[idx])


def cluster_bootstrap_lb(
    diffs: list[float],
    clusters: list[str],
    seed: int,
    label: str,
    replicates: int = 100_000,
    alpha: float = 0.025,
    block: int = 2_000,
) -> dict:
    """One-sided (1-alpha) lower confidence bound for the mean paired difference.

    Resamples whole STRUCTURAL clusters (`template_cluster_id`) with replacement
    -- value instantiations of one structural template never count as
    independent evidence, and neither do paraphrases, which is why the cluster
    key is the structural id and never the wording `template_id`. The statistic
    per replicate is the ratio of sums: sum(resampled cluster diff sums) /
    sum(resampled cluster sizes). Replicate index draws come from the committed
    SHAKE-256 stream, so the bound is bit-reproducible from (seed, label).

    `block` is the replicate chunk size. It is NOT an implementation detail: each
    chunk draws from its own SHAKE label, so the realised stream -- and therefore
    the reported bound -- depends on it. Callers in the preregistered path pass it
    from configs/agentic_preregister.json
    (`statistics.clustered_bootstrap.chunk_block`), where it is frozen at the
    2000 that produced every committed bound. The default here mirrors that
    registered value so a direct caller cannot silently pick a different stream.
    """
    if len(diffs) != len(clusters):
        raise ValueError("diffs and clusters must align")
    if block <= 0:
        raise ValueError("bootstrap chunk block must be positive")
    if not diffs:
        return {"n": 0, "n_clusters": 0, "point": 0.0, "lb": float("-inf"),
                "alpha": alpha, "replicates": replicates, "degenerate": True}

    import numpy as np

    cluster_ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(cluster_ids)}
    k = len(cluster_ids)
    sums = np.zeros(k)
    sizes = np.zeros(k)
    for d, c in zip(diffs, clusters):
        sums[pos[c]] += d
        sizes[pos[c]] += 1
    point = float(sums.sum() / sizes.sum())

    if k == 1:
        # A single cluster cannot be resampled informatively; the bound is the
        # point estimate and the caller must treat the gate as INCONCLUSIVE.
        return {"n": len(diffs), "n_clusters": 1, "point": point, "lb": point,
                "alpha": alpha, "replicates": replicates, "degenerate": True}

    stats = np.empty(replicates)
    done = 0
    chunk_no = 0
    while done < replicates:
        b = min(block, replicates - done)
        idx = rng.choice_indices(seed, f"{label}:chunk{chunk_no}", b * k, k).reshape(b, k)
        num = sums[idx].sum(axis=1)
        den = sizes[idx].sum(axis=1)
        stats[done : done + b] = num / den
        done += b
        chunk_no += 1
    stats.sort()
    lb = _quantile_lower(stats, alpha)
    return {"n": len(diffs), "n_clusters": k, "point": point, "lb": lb,
            "alpha": alpha, "replicates": replicates, "degenerate": False}
