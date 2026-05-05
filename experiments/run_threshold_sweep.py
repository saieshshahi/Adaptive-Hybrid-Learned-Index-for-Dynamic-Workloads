"""
Threshold sweep: vary the hybrid index's `confidence_threshold` and observe the
fallback-rate / search-window tradeoff. We use the piecewise dataset because a
single linear model has structural error there -- exactly the regime where the
threshold knob matters.

Outputs:
  results/threshold_sweep.csv
  plots/threshold_tradeoff.png

Run from repo root:
    python experiments/run_threshold_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.datasets import make_piecewise
from src.indexes import HybridIndex
from src.metrics import run_workload, summarize
from src.models import LinearRankModel
from src.plotting import threshold_tradeoff
from src.workloads import workload_hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--q", type=int, default=5000)
    parser.add_argument("--miss-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--results", default="results/threshold_sweep.csv")
    parser.add_argument("--plot", default="plots/threshold_tradeoff.png")
    args = parser.parse_args()

    keys = make_piecewise(args.n, segments=4, seed=args.seed)
    queries = workload_hits(keys, args.q, seed=args.seed + 1, miss_rate=args.miss_rate)

    # Sweep covers "always fall back" (tau=0) through "essentially never preemptively
    # fall back" (tau >= n). The piecewise dataset has large model error on a
    # global linear fit, so the choice of tau is meaningful.
    thresholds = [0, 16, 64, 256, 1024, 4096, 16384, args.n * 2]

    rows = []
    for tau in thresholds:
        index = HybridIndex(model_factory=LinearRankModel,
                            confidence_threshold=tau,
                            drift_window=200,
                            drift_threshold=0.999,  # disable retraining for this sweep
                            min_queries_between_retrains=10**9)
        index.build(keys)
        df = run_workload(index, queries)
        s = summarize(df)
        s["threshold"] = tau
        s["model_max_error"] = int(getattr(index.model, "max_error", 0))
        s["retrain_count"] = index.retrain_count
        rows.append(s)
        print(f"tau={tau:>8}  fallback={s['fallback_rate']:.3f}  "
              f"mean_window={s['mean_window_size']:7.0f}  "
              f"mean_steps={s['mean_steps']:6.2f}  "
              f"med_lat={s['median_latency_ns']:7.0f}ns")

    out = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.results, index=False)
    print(f"\nwrote {args.results}")

    threshold_tradeoff(out, args.plot,
                       title="Hybrid: fallback rate vs local search window")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
