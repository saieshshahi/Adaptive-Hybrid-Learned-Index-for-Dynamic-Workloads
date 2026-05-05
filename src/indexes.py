"""
Three indexes that all expose the same lookup interface:

    BisectIndex          -- robust baseline, sorted array + bisect
    LearnedOnlyIndex     -- model predicts rank, search local window, NO fallback
    HybridIndex          -- predict + local search; bisect fallback on miss; rolling
                            drift detection triggers retraining

All three share storage: a sorted np.ndarray of unique float64 keys. We keep the
storage trivial on purpose -- the project is about lookup behavior, not about
implementing a B-tree.

`LookupResult` collects logical metrics (window size, prediction error, steps,
fallback) so the rest of the pipeline can analyze runs without needing to time
individual queries reliably (Python timing is noisy at microsecond scale).
"""

from __future__ import annotations

import bisect
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .models import LinearRankModel, PiecewiseLinearModel


@dataclass
class LookupResult:
    found: bool
    pos: int                # position of key (or insertion point if missing)
    pred_error: int         # |predicted_rank - true_rank|; 0 for bisect
    window_size: int        # search window the index actually used
    steps: int              # logical comparisons (analytic; see _bin_steps)
    fell_back: bool         # bisect fallback triggered
    elapsed_ns: int         # wall-clock for this lookup (noisy but we keep it)


def _bin_steps(window: int) -> int:
    """Logical comparisons for binary search over `window` items."""
    if window <= 1:
        return 1
    return int(math.ceil(math.log2(window)))


class BisectIndex:
    """Sorted array + Python's bisect. Robust, the upper bound on correctness."""

    name = "bisect"

    def __init__(self) -> None:
        self.keys: np.ndarray = np.empty(0, dtype=np.float64)
        self.retrain_count = 0
        self.retrain_time_ns = 0

    def build(self, keys: np.ndarray) -> None:
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)

    def update_data(self, keys: np.ndarray) -> None:
        self.build(keys)

    def lookup(self, key: float) -> LookupResult:
        n = len(self.keys)
        t0 = time.perf_counter_ns()
        # bisect uses the C implementation under the hood; we feed it the ndarray
        pos = bisect.bisect_left(self.keys, key)
        found = pos < n and self.keys[pos] == key
        elapsed = time.perf_counter_ns() - t0
        return LookupResult(
            found=found, pos=pos, pred_error=0,
            window_size=n, steps=_bin_steps(n),
            fell_back=False, elapsed_ns=elapsed,
        )


class LearnedOnlyIndex:
    """
    Train a model once on the keys; for each lookup, predict rank, then bisect
    inside [pred - max_error, pred + max_error]. We do NOT fall back to a global
    bisect even on miss -- that's the point: under drift, this index degrades.
    """

    name = "learned_only"

    def __init__(self, model_factory: Callable[[], object] = LinearRankModel) -> None:
        self.model_factory = model_factory
        self.model = None
        self.keys: np.ndarray = np.empty(0, dtype=np.float64)
        self.retrain_count = 0
        self.retrain_time_ns = 0

    def build(self, keys: np.ndarray) -> None:
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)
        self.model = self.model_factory()
        t0 = time.perf_counter_ns()
        self.model.fit(self.keys)
        self.retrain_time_ns += (time.perf_counter_ns() - t0)
        self.retrain_count += 1

    def update_data(self, keys: np.ndarray) -> None:
        # learned-only is static: it sees new data but doesn't refit.
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)

    def lookup(self, key: float) -> LookupResult:
        n = len(self.keys)
        if n == 0 or self.model is None:
            return LookupResult(False, 0, 0, 0, 1, False, 0)
        t0 = time.perf_counter_ns()
        pred_arr = self.model.predict(np.array([key]))
        pred = int(round(float(pred_arr[0])))
        max_err = max(1, int(self.model.max_error))
        lo = max(0, pred - max_err)
        hi = min(n, pred + max_err + 1)
        # local bisect with bounds
        local_pos = bisect.bisect_left(self.keys, key, lo, hi)
        # we accept whatever the local search returns -- this index has no fallback
        if local_pos < hi and self.keys[local_pos] == key:
            found, pos = True, local_pos
        elif local_pos < n and self.keys[local_pos] == key:
            found, pos = True, local_pos
        else:
            found, pos = False, local_pos
        elapsed = time.perf_counter_ns() - t0
        # report logical-error vs the true sorted position
        true_pos = bisect.bisect_left(self.keys, key)
        pred_error = abs(pred - true_pos)
        window = hi - lo
        return LookupResult(
            found=found, pos=pos, pred_error=pred_error,
            window_size=window, steps=_bin_steps(window),
            fell_back=False, elapsed_ns=elapsed,
        )


class HybridIndex:
    """
    Adaptive hybrid:

      1. Predict rank with the model.
      2. Search a local window whose half-width is min(model.max_error,
         confidence_threshold). The threshold caps the local search cost.
      3. If the local search misses (key not in the window, or key is genuinely
         absent), fall back to a global bisect over the full sorted array.
      4. Track rolling fallback rate over the last `drift_window` queries; if
         it exceeds `drift_threshold`, retrain on the current keys. To avoid
         pointless retraining when the model just structurally can't fit the
         data, we stop retraining once a freshly trained model still has
         max_error > confidence_threshold (the "give up" rule).

    Knobs: confidence_threshold, drift_window, drift_threshold, model_factory.
    """

    name = "hybrid"

    def __init__(
        self,
        model_factory: Callable[[], object] = LinearRankModel,
        confidence_threshold: int = 1024,
        drift_window: int = 200,
        drift_threshold: float = 0.25,
        min_queries_between_retrains: int = 200,
    ) -> None:
        self.model_factory = model_factory
        self.confidence_threshold = confidence_threshold
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.min_queries_between_retrains = min_queries_between_retrains

        self.model = None
        self.keys: np.ndarray = np.empty(0, dtype=np.float64)
        self.retrain_count = 0
        self.retrain_time_ns = 0
        self._fallback_history: deque[int] = deque(maxlen=drift_window)
        self._queries_since_retrain = 0
        self._gave_up: bool = False  # True when last retrain still produced max_error > threshold

    def build(self, keys: np.ndarray) -> None:
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)
        self._train()

    def update_data(self, keys: np.ndarray) -> None:
        # Phase change: data changes, but we do NOT auto-retrain. Drift detection
        # in lookup() is responsible for triggering retraining when warranted.
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)
        # New data is a fresh chance for the model to fit; clear the give-up flag.
        self._gave_up = False
        self._fallback_history.clear()
        self._queries_since_retrain = 0

    def _train(self) -> None:
        self.model = self.model_factory()
        t0 = time.perf_counter_ns()
        self.model.fit(self.keys)
        self.retrain_time_ns += (time.perf_counter_ns() - t0)
        self.retrain_count += 1
        self._queries_since_retrain = 0
        self._fallback_history.clear()
        # Note: we do NOT set _gave_up here. Initial / phase-change builds always
        # leave the model available; the give-up flag is only set after a *retrain*
        # whose new model still can't meet the confidence threshold (see
        # _maybe_retrain). This way an initial build with tight tau still exercises
        # the learned path (with reactive fallback), which is what the threshold
        # sweep is meant to study.

    def _maybe_retrain(self) -> None:
        if self._gave_up:
            return
        if self._queries_since_retrain < self.min_queries_between_retrains:
            return
        if len(self._fallback_history) < self.drift_window:
            return
        rate = sum(self._fallback_history) / len(self._fallback_history)
        if rate >= self.drift_threshold:
            self._train()
            # if even a fresh fit cannot meet the confidence threshold, give up
            # until update_data() is called.
            if self.model.max_error > self.confidence_threshold:
                self._gave_up = True

    def lookup(self, key: float) -> LookupResult:
        n = len(self.keys)
        if n == 0:
            return LookupResult(False, 0, 0, 0, 1, False, 0)
        t0 = time.perf_counter_ns()

        fell_back = False
        steps = 0
        window = 0
        pred = -1
        pos = 0
        found = False

        # If we've decided the model can't help on this data, just bisect.
        if self.model is None or self._gave_up:
            fell_back = True
            window = n
            steps += _bin_steps(n)
            pos = bisect.bisect_left(self.keys, key)
            found = pos < n and self.keys[pos] == key
        else:
            pred_arr = self.model.predict(np.array([key]))
            pred_raw = int(round(float(pred_arr[0])))
            # clamp to a valid index so the window math always produces lo<=hi
            pred = max(0, min(n - 1, pred_raw))
            # threshold caps the local search half-width; if the model thinks it
            # needs more, we accept some misses (and a fallback) instead of paying
            # the wide-window cost up front.
            cap = max(1, int(self.confidence_threshold))
            max_err = min(max(1, int(self.model.max_error)), cap)
            lo = max(0, pred - max_err)
            hi = min(n, pred + max_err + 1)
            if hi < lo:
                hi = lo  # paranoia; pred is clamped above so this shouldn't trigger
            window = hi - lo
            steps += _bin_steps(max(window, 1))
            local_pos = bisect.bisect_left(self.keys, key, lo, hi)
            if local_pos < hi and self.keys[local_pos] == key:
                found, pos = True, local_pos
            else:
                fell_back = True
                steps += _bin_steps(n)
                pos = bisect.bisect_left(self.keys, key)
                found = pos < n and self.keys[pos] == key
            # restore pred for error reporting (use the clamped value)

        elapsed = time.perf_counter_ns() - t0

        # update drift state and possibly retrain
        self._fallback_history.append(1 if fell_back else 0)
        self._queries_since_retrain += 1
        self._maybe_retrain()

        # logical prediction error wrt current data
        true_pos = bisect.bisect_left(self.keys, key)
        pred_error = abs(pred - true_pos) if pred >= 0 else 0

        return LookupResult(
            found=found, pos=pos, pred_error=pred_error,
            window_size=window, steps=steps,
            fell_back=fell_back, elapsed_ns=elapsed,
        )


INDEX_BUILDERS: dict[str, Callable[..., object]] = {
    "bisect": lambda **kw: BisectIndex(),
    "learned_only_linear": lambda **kw: LearnedOnlyIndex(model_factory=LinearRankModel),
    "learned_only_piecewise8": lambda **kw: LearnedOnlyIndex(
        model_factory=lambda: PiecewiseLinearModel(n_segments=8)
    ),
    "hybrid_linear": lambda **kw: HybridIndex(model_factory=LinearRankModel, **kw),
    "hybrid_piecewise8": lambda **kw: HybridIndex(
        model_factory=lambda: PiecewiseLinearModel(n_segments=8), **kw
    ),
}
