# Adaptive Hybrid Learned Index for Dynamic Workloads

A small Python framework for studying the behavior of a learned-index lookup
policy that falls back to a robust baseline (`bisect` on a sorted array) when
its model is not confident. The point is **explainable behavior under
distributional drift**, not raw lookup speed: Python's microbenchmarks are too
noisy and too far from a real database engine to make speed claims that
generalize.

## Layout

```
src/
  datasets.py        synthetic key generators (uniform, linear, lognormal, piecewise) + drift phases
  models.py          LinearRankModel, PiecewiseLinearModel (RMI-lite)
  indexes.py         BisectIndex, LearnedOnlyIndex, HybridIndex (with drift detection / retrain)
  workloads.py       hit-mostly query streams + drift workload
  metrics.py         per-query records and headline summarization
  plotting.py        bar / line / twin-axis helpers
experiments/
  run_static.py             static distribution comparison
  run_drift.py              phase-shift drift experiment
  run_threshold_sweep.py    threshold-vs-fallback tradeoff
  aggregate_summary.py      combine all three CSVs into summary_metrics.csv
results/             CSV outputs from each experiment
plots/               PNGs referenced in the report
requirements.txt
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Tested with Python 3.13 + numpy 1.24+, pandas 2+, scikit-learn 1.3+,
matplotlib 3.7+. The project deliberately avoids any C/C++ extension code
beyond what numpy and Python's `bisect` already give us.

## Reproducing the results

From the repo root:

```
python experiments/run_static.py
python experiments/run_drift.py
python experiments/run_threshold_sweep.py
python experiments/aggregate_summary.py
```

Each script accepts a `--help` for the relevant knobs (n keys, queries per
phase, miss rate, seed). The defaults are sized to run in well under a minute
on a laptop and match the numbers in this README.

## Generated files

CSVs (`results/`):
- `static_summary.csv` — one row per (dataset, index): latency stats,
  fallback rate, prediction error stats, mean window, mean steps, retrain count.
- `drift_per_query.csv` — long-form per-query records across all four phases
  for every index (used by the rolling-mean plots).
- `drift_per_phase.csv` — per (phase, index) summary; this is the table to
  cite in the report.
- `threshold_sweep.csv` — one row per `confidence_threshold` value.
- `summary_metrics.csv` — concatenated headline view of all three experiments.

Plots (`plots/`):
- `static_lookup_latency.png` — median ns/lookup by (dataset, index). **Noisy;
  see "What to claim" below.**
- `static_logical_steps.png` — analytic mean comparisons per lookup. **The
  cleaner story.**
- `static_pred_error.png` — p95 |predicted_rank − true_rank| per (dataset, index).
- `static_window.png` — mean local search window.
- `drift_error_over_time.png` — rolling prediction error vs query index, with
  vertical lines at phase boundaries (log-scale y-axis).
- `drift_fallback_rate.png` — rolling fallback rate vs query index.
- `drift_steps_over_time.png` — rolling mean logical steps.
- `threshold_tradeoff.png` — twin-axis: fallback rate (left) and mean window
  (right) vs `confidence_threshold` on a log x-axis.

## How the indexes work

All three store a single sorted `np.ndarray` of float64 keys. They differ only
in how they search.

- **BisectIndex** — `bisect.bisect_left` over the full array.
  Logical cost ≈ `ceil(log2(n))` per lookup.

- **LearnedOnlyIndex** — fits a model on the keys at build time and records
  the model's `max_error` on training data. Each lookup predicts a rank,
  searches the window `[pred − max_error, pred + max_error]` with a bounded
  bisect, and **does not fall back** even if it misses. This is the regime
  where, under drift, the index degrades to garbage.

- **HybridIndex** — like learned-only, but:
  1. The local search half-width is `min(model.max_error, confidence_threshold)`.
     The threshold is a *cap* on the local search cost.
  2. If the local search misses, fall back to a global bisect.
  3. The rolling fallback rate over the last `drift_window` queries is tracked.
     If it exceeds `drift_threshold`, retrain on the current keys.
  4. If a retrain produces a model that *still* doesn't satisfy
     `max_error <= confidence_threshold`, set a "give up" flag and stop using
     the model until `update_data` is called with new keys. This avoids
     infinite retrain churn when a linear model is trying to fit a structurally
     non-linear distribution.

Drift detection is intentionally simple: rolling mean of a 0/1 fallback signal.
This is enough to demonstrate the policy without inviting the question
"is the detector itself the contribution?"

## Key observations

### 1. Static comparison (`run_static.py`)

| dataset   | bisect steps | learned_only_linear | learned_only_piecewise8 | hybrid_linear | hybrid_piecewise8 |
|-----------|--------------|---------------------|-------------------------|---------------|-------------------|
| linear    | 15           | 2.0                 | 2.0                     | 2.8           | 2.8               |
| uniform   | 15           | 8.0                 | 7.0                     | 8.8           | 7.8               |
| lognormal | 15           | 14.3                | 11.9                    | 15.3          | 13.5              |
| piecewise | 15           | 12.8                | 13.7                    | 15.2          | 14.8              |

(`mean_steps` from `static_summary.csv`. n=20000 keys, q=5000 queries.)

- Learned models reduce logical comparisons substantially on smooth
  distributions (`linear`, `uniform`).
- On `lognormal` and `piecewise`, a single linear model has too much error
  for the local-search window to be small; the piecewise model helps but
  doesn't recover the linear case.
- `hybrid_linear` on `lognormal`/`piecewise` ends up *slightly worse than
  bisect* because of the cost of attempting the local search and falling back.
  This is exactly why the give-up flag exists.

### 2. Drift (`run_drift.py`)

Four phases, each with its own data distribution:
`linear → lognormal → piecewise → uniform(0,100)`. The model is trained on
phase 0 only at startup; underlying data is replaced at each phase boundary.

| phase     | bisect found | learned_only found | hybrid_linear found | hybrid_linear fallback | hybrid_piecewise8 fallback |
|-----------|--------------|--------------------|---------------------|------------------------|----------------------------|
| linear    | 0.95         | 0.95               | 0.95                | 0.05                   | 0.05                       |
| lognormal | 0.95         | **0.00**           | 0.95                | 1.00                   | 1.00                       |
| piecewise | 0.95         | **0.00**           | 0.95                | 1.00                   | 1.00                       |
| uniform   | 0.95         | **0.00**           | 0.95                | 0.09                   | 0.09                       |

(`drift_per_phase.csv`. found_rate=0.95 matches the 5% miss rate.)

- **`learned_only_linear` collapses to 0% found rate** in any phase past
  phase 0. With no fallback, predictions land in a 1-key-wide window that
  doesn't contain the key, and the lookup just returns the wrong position.
- **`hybrid_*` matches bisect's found rate in every phase**: the fallback
  catches every case the model can't handle.
- In phases 1 and 2 the linear model can't fit; `give_up` triggers after
  one futile retrain and hybrid behaves exactly like bisect for the rest of
  the phase. In phase 3 (uniform), the retrained model fits, give-up clears,
  and hybrid uses the local search again (~9% fallback rate; the rest is
  the 5% miss rate plus a small calibration gap).
- Total retrain cost: `hybrid_linear` retrained 4 times for 2.4 ms total;
  `hybrid_piecewise8` retrained 4 times for 7.8 ms total. Negligible compared
  to the workload time.

### 3. Threshold sweep (`run_threshold_sweep.py`)

On the `piecewise` dataset where the linear model has `max_error = 4513`:

| threshold τ | fallback rate | mean window | mean steps |
|-------------|---------------|-------------|------------|
| 0           | 1.000         | 3           | 16.9       |
| 16          | 0.996         | 32          | 20.9       |
| 64          | 0.975         | 125         | 22.6       |
| 256         | 0.891         | 494         | 23.3       |
| 1024        | 0.501         | 1960        | 19.4       |
| 4096        | 0.000         | 4572        | 12.9       |
| 16384       | 0.000         | 4572        | 12.9       |
| 40000       | 0.000         | 4572        | 12.9       |

- **The tradeoff is non-monotonic in mean steps.** Tight thresholds force most
  queries to do *both* a doomed local search and a global bisect, costing more
  than bisect alone (peak ~23 steps vs bisect's 15).
- Once `τ` exceeds the model's actual `max_error`, fallback rate drops to 0
  and the local search dominates. This is the regime where hybrid actually
  beats bisect.
- The implication is practical: **set `τ` based on the observed `max_error`
  after training, not as a hand-picked constant.** Picking τ too low costs
  more than just disabling the model would.
