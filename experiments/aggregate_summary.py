"""
Combine results from all three experiments into a single summary_metrics.csv.

Columns: experiment, dataset/phase, index, threshold (where relevant), and the
standard headline metrics. This is the report-friendly view; per-experiment
CSVs stay around for the detailed slice-and-dice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parts = []

    static_path = Path("results/static_summary.csv")
    if static_path.exists():
        df = pd.read_csv(static_path)
        df.insert(0, "experiment", "static")
        df.insert(2, "phase", "")
        df.insert(3, "threshold", "")
        parts.append(df)

    drift_path = Path("results/drift_per_phase.csv")
    if drift_path.exists():
        df = pd.read_csv(drift_path)
        df.insert(0, "experiment", "drift")
        df["phase"] = df["phase_name"]
        df["dataset"] = df["phase_name"]
        df["threshold"] = ""
        parts.append(df)

    sweep_path = Path("results/threshold_sweep.csv")
    if sweep_path.exists():
        df = pd.read_csv(sweep_path)
        df.insert(0, "experiment", "threshold_sweep")
        df["dataset"] = "piecewise"
        df["index"] = "hybrid_linear"
        df["phase"] = ""
        parts.append(df)

    if not parts:
        print("no inputs found in results/; run the three experiments first")
        return

    common_cols = [
        "experiment", "dataset", "phase", "index", "threshold",
        "n_queries", "found_rate", "fallback_rate",
        "median_latency_ns", "mean_latency_ns", "p95_latency_ns",
        "throughput_qps",
        "mean_pred_error", "p95_pred_error", "max_pred_error",
        "mean_window_size", "max_window_size",
        "mean_steps", "p95_steps",
        "retrain_count", "retrain_time_ns",
    ]

    combined = pd.concat(parts, ignore_index=True, sort=False)
    for c in common_cols:
        if c not in combined.columns:
            combined[c] = ""
    out = combined[common_cols]
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/summary_metrics.csv", index=False)
    print(f"wrote results/summary_metrics.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
