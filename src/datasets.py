"""
Synthetic key generators for the Adaptive Hybrid Learned Index project.

Every generator returns a sorted, unique numpy array of float64 keys. This is the
storage format the indexes operate on (sorted array + bisect / learned model).
We keep the data in memory; nothing here touches disk or downloads anything.

The point is to study learned-index behavior under controlled distributions, so
the generators are deliberately small and explicit rather than realistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _finalize(keys: np.ndarray) -> np.ndarray:
    """Dedup + sort + cast to float64. Keys must end up strictly increasing."""
    keys = np.unique(keys.astype(np.float64))
    return keys


def make_uniform(n: int, low: float = 0.0, high: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(low, high, size=int(n * 1.05))
    return _finalize(raw)[:n]


def make_linear(n: int, low: float = 0.0, high: float = 1.0, seed: int = 0) -> np.ndarray:
    """Evenly spaced keys plus tiny jitter; the easiest case for a linear model."""
    rng = np.random.default_rng(seed)
    base = np.linspace(low, high, n)
    jitter = rng.normal(0, (high - low) / (n * 50.0), size=n)
    return _finalize(base + jitter)[:n]


def make_lognormal(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mu, sigma, size=int(n * 1.05))
    return _finalize(raw)[:n]


def make_piecewise(n: int, segments: int = 4, seed: int = 0) -> np.ndarray:
    """
    Concatenate `segments` ranges, each filled with its own distribution. Designed
    to break a single global linear model: any straight line through this CDF will
    miss badly in some segment.
    """
    rng = np.random.default_rng(seed)
    per = n // segments
    chunks = []
    cursor = 0.0
    for i in range(segments):
        width = 1.0 + 0.5 * i  # widening segments
        if i % 3 == 0:
            chunk = rng.uniform(cursor, cursor + width, size=per)
        elif i % 3 == 1:
            # dense cluster
            chunk = rng.normal(cursor + width / 2, width / 12, size=per)
            chunk = np.clip(chunk, cursor, cursor + width)
        else:
            # sparse tail
            chunk = cursor + rng.exponential(width / 4, size=per)
        chunks.append(chunk)
        cursor += width + 0.1
    raw = np.concatenate(chunks)
    return _finalize(raw)[:n]


# --- drift phases ---------------------------------------------------------

@dataclass(frozen=True)
class PhaseSpec:
    """One phase of a drift schedule.

    name        : label for plots / CSVs
    n           : number of keys in this phase's sorted array
    generator   : callable (n, seed) -> sorted np.ndarray
    n_queries   : how many lookups to issue against the phase
    """
    name: str
    n: int
    generator: Callable[[int, int], np.ndarray]
    n_queries: int


def default_drift_schedule(n: int = 50_000, queries_per_phase: int = 5_000) -> list[PhaseSpec]:
    """
    A four-phase schedule that exercises a learned model:
      1. linear   -- model trained here is near-perfect
      2. lognormal-- distribution shifts; old model errors blow up
      3. piecewise-- structurally hard for any single linear model
      4. uniform  -- different scale; tests recovery via retraining
    """
    return [
        PhaseSpec("linear", n, lambda n, s: make_linear(n, 0, 1, seed=s), queries_per_phase),
        PhaseSpec("lognormal", n, lambda n, s: make_lognormal(n, 0, 1, seed=s), queries_per_phase),
        PhaseSpec("piecewise", n, lambda n, s: make_piecewise(n, 4, seed=s), queries_per_phase),
        PhaseSpec("uniform", n, lambda n, s: make_uniform(n, 0, 100, seed=s), queries_per_phase),
    ]


DATASET_BUILDERS: dict[str, Callable[[int, int], np.ndarray]] = {
    "uniform": lambda n, seed: make_uniform(n, 0, 1, seed=seed),
    "linear": lambda n, seed: make_linear(n, 0, 1, seed=seed),
    "lognormal": lambda n, seed: make_lognormal(n, 0, 1, seed=seed),
    "piecewise": lambda n, seed: make_piecewise(n, 4, seed=seed),
}
