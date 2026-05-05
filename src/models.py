"""
Learned rank-prediction models.

Both models share an interface:

    model.fit(keys)              # keys: 1-D sorted np.ndarray
    model.predict(query_keys)    # returns predicted ranks (float ndarray)
    model.max_error              # max absolute error on training data (int)
    model.p95_error              # p95 absolute error on training data (int)

`max_error` is the safety envelope used by the indexes: if a learned model says
"key probably has rank ~p", we search [p - max_error, p + max_error] and we
are guaranteed to find the key in that window (when it exists).

We deliberately avoid sklearn's full pipeline machinery: we want explainable,
fast-to-fit predictors so retraining cost is small.
"""

from __future__ import annotations

import numpy as np


class LinearRankModel:
    """Single global linear model: rank = a * key + b."""

    def __init__(self) -> None:
        self.a: float = 0.0
        self.b: float = 0.0
        self.max_error: int = 0
        self.p95_error: int = 0
        self.n: int = 0

    def fit(self, keys: np.ndarray) -> None:
        n = len(keys)
        ranks = np.arange(n, dtype=np.float64)
        # closed-form OLS, vectorized
        x_mean = keys.mean()
        y_mean = ranks.mean()
        denom = float(((keys - x_mean) ** 2).sum())
        if denom == 0.0:
            self.a = 0.0
        else:
            self.a = float(((keys - x_mean) * (ranks - y_mean)).sum() / denom)
        self.b = float(y_mean - self.a * x_mean)
        self.n = n
        # measure error envelope on training data
        preds = self.a * keys + self.b
        errors = np.abs(preds - ranks)
        self.max_error = int(np.ceil(errors.max())) if n else 0
        self.p95_error = int(np.ceil(np.quantile(errors, 0.95))) if n else 0

    def predict(self, qkeys: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(qkeys, dtype=np.float64) + self.b


class PiecewiseLinearModel:
    """
    A small two-stage model: split sorted keys into `n_segments` equal-size buckets
    by index, fit a linear model in each. At predict time, we use the global linear
    fit to *route* the query to a segment, then the segment's local linear fit gives
    the final rank prediction.

    This is intentionally a stripped-down RMI: one routing model + n leaf models.
    """

    def __init__(self, n_segments: int = 8) -> None:
        self.n_segments = n_segments
        self.router = LinearRankModel()
        self.leaf_a: np.ndarray = np.zeros(n_segments)
        self.leaf_b: np.ndarray = np.zeros(n_segments)
        self.max_error: int = 0
        self.p95_error: int = 0
        self.n: int = 0

    def fit(self, keys: np.ndarray) -> None:
        n = len(keys)
        self.n = n
        if n == 0:
            self.max_error = 0
            self.p95_error = 0
            return
        seg = max(1, min(self.n_segments, n))
        self.n_segments = seg
        self.leaf_a = np.zeros(seg)
        self.leaf_b = np.zeros(seg)
        # router maps key -> segment index in [0, seg)
        self.router.fit(keys)
        ranks = np.arange(n, dtype=np.float64)
        # split sorted keys into seg roughly-equal buckets by index
        boundaries = np.linspace(0, n, seg + 1, dtype=int)
        for i in range(seg):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 2:
                # degenerate segment: use router as a fallback for this leaf
                self.leaf_a[i] = self.router.a
                self.leaf_b[i] = self.router.b
                continue
            xs = keys[lo:hi]
            ys = ranks[lo:hi]
            xm, ym = xs.mean(), ys.mean()
            denom = float(((xs - xm) ** 2).sum())
            if denom == 0.0:
                self.leaf_a[i] = 0.0
                self.leaf_b[i] = ym
            else:
                a = float(((xs - xm) * (ys - ym)).sum() / denom)
                self.leaf_a[i] = a
                self.leaf_b[i] = float(ym - a * xm)
        # error envelope across the whole training set
        preds = self.predict(keys)
        errors = np.abs(preds - ranks)
        self.max_error = int(np.ceil(errors.max()))
        self.p95_error = int(np.ceil(np.quantile(errors, 0.95)))

    def predict(self, qkeys: np.ndarray) -> np.ndarray:
        qkeys = np.asarray(qkeys, dtype=np.float64)
        if self.n == 0:
            return np.zeros_like(qkeys)
        # route via router prediction, clamped to segment range
        routed = self.router.a * qkeys + self.router.b
        seg_size = self.n / self.n_segments
        seg_idx = np.clip((routed / max(seg_size, 1e-12)).astype(int), 0, self.n_segments - 1)
        # apply per-segment linear model
        a = self.leaf_a[seg_idx]
        b = self.leaf_b[seg_idx]
        return a * qkeys + b


MODEL_BUILDERS = {
    "linear": lambda: LinearRankModel(),
    "piecewise8": lambda: PiecewiseLinearModel(n_segments=8),
    "piecewise32": lambda: PiecewiseLinearModel(n_segments=32),
}
