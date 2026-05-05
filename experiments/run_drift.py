"""
Drift experiment: phase-by-phase distribution shift.

For each phase: replace the underlying sorted array (all indexes), then issue
that phase's queries. Track per-query metrics across the entire run, with a
`phase` column to slice on later.

Outputs:
  results/drift_per_query.csv         -- per-query records (long)
  results/drift_per_phase.csv         -- per (phase, index) summary
  plots/drift_error_over_time.png     -- rolling pred error vs query index
  plots/drift_fallback_rate.png       -- rolling fallback rate vs query index
  plots/drift_steps_over_time.png     -- rolling logical steps

Run from repo root:
    python experiments/run_drift.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.datasets import default_drift_schedule
from src.indexes import INDEX_BUILDERS
from src.metrics import run_workload, summarize, rolling_metric
from src.plotting import line_over_time
from src.workloads import drift_workload


# Indexes we want to compare under drift. Bisect stays as a sanity baseline; the
# story is between learned_only and hybrid.
INDEXES = [
    "bisect",
    "learned_only_linear",
    "hybrid_linear",
    "hybrid_piecewise8",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30000, help="keys per phase")
    parser.add_argument("--q", type=int, default=4000, help="queries per phase")
    parser.add_argument("--miss-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rolling", type=int, default=200)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    schedule = default_drift_schedule(n=args.n, queries_per_phase=args.q)
    # materialize phase data
    phase_keys = []
    for i, ph in enumerate(schedule):
        keys = ph.generator(ph.n, args.seed + i)
        phase_keys.append(keys)
    queries, phase_id = drift_workload(phase_keys, args.q,
                                       seed=args.seed + 100,
                                       miss_rate=args.miss_rate)

    phase_boundaries = [args.q * (i + 1) for i in range(len(schedule) - 1)]

    # Run each index across the schedule
    per_index_dfs = {}
    summary_rows = []
    rolling_err = {}
    rolling_fb = {}
    rolling_steps = {}
    for idx_name in INDEXES:
        index = INDEX_BUILDERS[idx_name]()
        # initialize on phase 0 keys (so model is trained on phase 0 only)
        index.build(phase_keys[0])

        all_records = []
        for i, ph in enumerate(schedule):
            if i > 0:
                index.update_data(phase_keys[i])
            sub_q = queries[i * args.q:(i + 1) * args.q]
            sub_phase = phase_id[i * args.q:(i + 1) * args.q]
            df_phase = run_workload(index, sub_q, phase_id=sub_phase)
            df_phase["phase"] = i
            all_records.append(df_phase)
        df = pd.concat(all_records, ignore_index=True)
        df["index"] = idx_name
        per_index_dfs[idx_name] = df

        # per-phase summary
        for i in range(len(schedule)):
            sub = df[df["phase"] == i]
            s = summarize(sub)
            s["phase"] = i
            s["phase_name"] = schedule[i].name
            s["index"] = idx_name
            s["retrain_count"] = getattr(index, "retrain_count", 0)
            s["retrain_time_ns"] = getattr(index, "retrain_time_ns", 0)
            summary_rows.append(s)
            print(f"[phase {i}={schedule[i].name:<10}] {idx_name:<25} "
                  f"med={s['median_latency_ns']:7.0f}ns  "
                  f"mean_steps={s['mean_steps']:6.2f}  "
                  f"fallback={s['fallback_rate']:.3f}  "
                  f"p95_err={s['p95_pred_error']:.0f}")

        rolling_err[idx_name] = rolling_metric(df, "pred_error", args.rolling)
        rolling_fb[idx_name] = rolling_metric(df, "fell_back", args.rolling)
        rolling_steps[idx_name] = rolling_metric(df, "steps", args.rolling)

    # write per-query records (concat all indexes)
    long = pd.concat(list(per_index_dfs.values()), ignore_index=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    long.to_csv(f"{args.results_dir}/drift_per_query.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(f"{args.results_dir}/drift_per_phase.csv",
                                      index=False)

    # plots
    line_over_time(rolling_err,
                   f"Drift: rolling prediction error (window={args.rolling}, log y)",
                   f"{args.plot_dir}/drift_error_over_time.png",
                   xlabel="query #", ylabel="|pred - true_rank| (rolling mean)",
                   phase_boundaries=phase_boundaries, log_y=True)
    line_over_time(rolling_fb,
                   f"Drift: rolling fallback rate (window={args.rolling})",
                   f"{args.plot_dir}/drift_fallback_rate.png",
                   xlabel="query #", ylabel="fallback rate (rolling)",
                   phase_boundaries=phase_boundaries)
    line_over_time(rolling_steps,
                   f"Drift: rolling logical steps (window={args.rolling})",
                   f"{args.plot_dir}/drift_steps_over_time.png",
                   xlabel="query #", ylabel="comparisons / lookup (rolling)",
                   phase_boundaries=phase_boundaries)
    print(f"\nwrote {args.results_dir}/drift_*.csv and {args.plot_dir}/drift_*.png")


if __name__ == "__main__":
    main()
