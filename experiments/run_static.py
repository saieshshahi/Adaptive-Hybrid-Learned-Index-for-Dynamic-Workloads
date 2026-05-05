"""
Static distribution comparison.

For each (dataset, index) pair: build the index on the dataset, run a hit-mostly
workload, summarize. Writes:

  results/static_summary.csv          -- one row per (dataset, index)
  plots/static_lookup_latency.png     -- mean latency, ns
  plots/static_logical_steps.png      -- mean comparisons (analytic)
  plots/static_pred_error.png         -- p95 prediction error (learned variants)
  plots/static_window.png             -- mean local search window

Run from repo root:
    python experiments/run_static.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.datasets import DATASET_BUILDERS
from src.indexes import INDEX_BUILDERS
from src.metrics import run_workload, summarize
from src.plotting import bar_compare
from src.workloads import workload_hits


DATASETS = ["uniform", "linear", "lognormal", "piecewise"]
INDEXES = [
    "bisect",
    "learned_only_linear",
    "learned_only_piecewise8",
    "hybrid_linear",
    "hybrid_piecewise8",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000, help="keys per dataset")
    parser.add_argument("--q", type=int, default=5000, help="queries per index per dataset")
    parser.add_argument("--miss-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results/static_summary.csv")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    rows = []
    for ds_name in DATASETS:
        keys = DATASET_BUILDERS[ds_name](args.n, args.seed)
        queries = workload_hits(keys, args.q, seed=args.seed + 1, miss_rate=args.miss_rate)
        for idx_name in INDEXES:
            index = INDEX_BUILDERS[idx_name]()
            index.build(keys)
            df = run_workload(index, queries)
            s = summarize(df)
            s["dataset"] = ds_name
            s["index"] = idx_name
            s["n_keys"] = len(keys)
            s["retrain_count"] = getattr(index, "retrain_count", 0)
            s["retrain_time_ns"] = getattr(index, "retrain_time_ns", 0)
            rows.append(s)
            print(f"[{ds_name:>10}] {idx_name:<25} "
                  f"med={s['median_latency_ns']:7.0f}ns  "
                  f"mean_steps={s['mean_steps']:5.2f}  "
                  f"fallback={s['fallback_rate']:.3f}  "
                  f"p95_err={s['p95_pred_error']:.0f}")

    out = pd.DataFrame(rows)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.results, index=False)
    print(f"\nwrote {args.results} ({len(out)} rows)")

    # Plots
    bar_compare(out, "median_latency_ns",
                "Static lookup: median latency",
                f"{args.plot_dir}/static_lookup_latency.png",
                ylabel="ns / lookup")
    bar_compare(out, "mean_steps",
                "Static lookup: mean logical comparisons (analytic)",
                f"{args.plot_dir}/static_logical_steps.png",
                ylabel="comparisons / lookup")
    bar_compare(out, "p95_pred_error",
                "Static lookup: p95 model prediction error",
                f"{args.plot_dir}/static_pred_error.png",
                ylabel="|pred - true_rank|")
    bar_compare(out, "mean_window_size",
                "Static lookup: mean local search window",
                f"{args.plot_dir}/static_window.png",
                ylabel="window size")
    print(f"wrote plots to {args.plot_dir}/")


if __name__ == "__main__":
    main()
