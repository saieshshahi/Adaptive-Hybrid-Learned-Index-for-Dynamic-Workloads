"""
Workload generators.

A workload is a 1-D np.ndarray of float64 query keys, in the order they should
be issued. We generate two flavors:

  - hit-mostly: most queries are present in the array; some are slightly off
    (synthetic "missing" keys) controlled by `miss_rate`
  - drift-aware: queries follow a multi-phase schedule, with each phase drawn
    from the corresponding phase's data
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def workload_hits(keys: np.ndarray, n_queries: int, seed: int = 0,
                  miss_rate: float = 0.0) -> np.ndarray:
    """Sample queries from `keys`. With probability `miss_rate`, perturb the
    sampled key by a small offset that is unlikely to match any existing key."""
    rng = np.random.default_rng(seed)
    n = len(keys)
    idx = rng.integers(0, n, size=n_queries)
    q = keys[idx].astype(np.float64).copy()
    if miss_rate > 0:
        flips = rng.random(n_queries) < miss_rate
        # nudge by a tiny amount based on local key spacing
        if n > 1:
            spacing = float((keys[-1] - keys[0]) / max(n - 1, 1))
        else:
            spacing = 1e-6
        offsets = rng.uniform(-0.49, 0.49, size=n_queries) * spacing
        q[flips] = q[flips] + offsets[flips]
    return q


def drift_workload(phase_keys: Sequence[np.ndarray], queries_per_phase: int,
                   seed: int = 0, miss_rate: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a contiguous query stream that follows the phase order. Returns:
      queries: 1-D array of query keys
      phase_id: same length, integer phase index per query

    The data in each phase is provided by the caller (one np.ndarray per phase).
    """
    rng = np.random.default_rng(seed)
    parts = []
    ids = []
    for i, keys in enumerate(phase_keys):
        sub = workload_hits(keys, queries_per_phase, seed=int(rng.integers(0, 1 << 31)),
                            miss_rate=miss_rate)
        parts.append(sub)
        ids.append(np.full(queries_per_phase, i, dtype=np.int32))
    return np.concatenate(parts), np.concatenate(ids)
