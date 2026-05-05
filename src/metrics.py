"""
Run a workload against an index, collect per-query records, summarize.

Per-query records are intentionally cheap to collect (one row in a list per
query). Summarization happens once at the end.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .indexes import LookupResult


def run_workload(index, queries: np.ndarray,
                 phase_id: Optional[np.ndarray] = None,
                 collect_per_query: bool = True) -> pd.DataFrame:
    """Run all queries against `index`. Returns a DataFrame with one row per
    query (if collect_per_query) plus a `phase` column when provided."""
    n = len(queries)
    rows = []

    # Reduce GC noise during timing. Re-enable on exit to be polite.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        wall_t0 = time.perf_counter_ns()
        for i in range(n):
            res: LookupResult = index.lookup(float(queries[i]))
            if collect_per_query:
                rows.append((
                    res.found, res.pred_error, res.window_size,
                    res.steps, res.fell_back, res.elapsed_ns,
                ))
        wall_total = time.perf_counter_ns() - wall_t0
    finally:
        if gc_was_enabled:
            gc.enable()

    if collect_per_query:
        df = pd.DataFrame(rows, columns=[
            "found", "pred_error", "window_size", "steps", "fell_back", "elapsed_ns",
        ])
    else:
        df = pd.DataFrame({"elapsed_ns_total": [wall_total]})
    if phase_id is not None and collect_per_query:
        df["phase"] = phase_id[:len(df)]
    df.attrs["wall_total_ns"] = wall_total
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Reduce a per-query DataFrame to a single-row dict of headline metrics."""
    if len(df) == 0:
        return {}
    elapsed = df["elapsed_ns"].to_numpy()
    pred_err = df["pred_error"].to_numpy()
    window = df["window_size"].to_numpy()
    steps = df["steps"].to_numpy()
    return {
        "n_queries": int(len(df)),
        "found_rate": float(df["found"].mean()),
        "fallback_rate": float(df["fell_back"].mean()),
        "median_latency_ns": float(np.median(elapsed)),
        "mean_latency_ns": float(np.mean(elapsed)),
        "p95_latency_ns": float(np.quantile(elapsed, 0.95)),
        "throughput_qps": float(1e9 / max(np.mean(elapsed), 1.0)),
        "mean_pred_error": float(np.mean(pred_err)),
        "p95_pred_error": float(np.quantile(pred_err, 0.95)),
        "max_pred_error": float(np.max(pred_err)),
        "mean_window_size": float(np.mean(window)),
        "max_window_size": float(np.max(window)),
        "mean_steps": float(np.mean(steps)),
        "p95_steps": float(np.quantile(steps, 0.95)),
    }


def per_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-phase headline metrics. Requires a `phase` column."""
    if "phase" not in df.columns:
        return pd.DataFrame()
    rows = []
    for phase, grp in df.groupby("phase"):
        s = summarize(grp)
        s["phase"] = int(phase)
        rows.append(s)
    out = pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)
    return out


def rolling_metric(df: pd.DataFrame, column: str, window: int = 200) -> np.ndarray:
    """Simple rolling mean over a column. Useful for plotting drift."""
    if len(df) == 0:
        return np.array([])
    return df[column].astype(float).rolling(window, min_periods=1).mean().to_numpy()
