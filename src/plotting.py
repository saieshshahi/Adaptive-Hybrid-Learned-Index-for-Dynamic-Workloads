"""
Plot helpers for the experiments. Each function writes one PNG to `out_path`.

Plots aim to be simple and report-ready: clear axes, labels, no marketing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed for batch plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def bar_compare(df: pd.DataFrame, metric: str, title: str, out_path: str,
                ylabel: Optional[str] = None, group_col: str = "dataset",
                index_col: str = "index") -> None:
    """One bar per (dataset, index) cell."""
    _ensure_parent(out_path)
    pivot = df.pivot_table(index=group_col, columns=index_col, values=metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel or metric)
    ax.set_xlabel(group_col)
    ax.legend(title=index_col, loc="best", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def line_over_time(records: dict[str, np.ndarray], title: str, out_path: str,
                   xlabel: str = "query #", ylabel: str = "value",
                   phase_boundaries: Optional[Iterable[int]] = None,
                   log_y: bool = False) -> None:
    """One line per index; optional vertical lines at phase boundaries.

    If log_y is True, use a symmetric-log y-axis so series spanning many orders
    of magnitude (e.g., learned_only's pred error under drift) are readable
    alongside well-behaved series.
    """
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, ys in records.items():
        ax.plot(np.arange(len(ys)), ys, label=label, linewidth=1.2)
    if phase_boundaries:
        for x in phase_boundaries:
            ax.axvline(x, color="grey", linestyle="--", alpha=0.5)
    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def threshold_tradeoff(df: pd.DataFrame, out_path: str,
                       title: str = "Confidence threshold tradeoff") -> None:
    """Twin-axis plot: fallback rate vs threshold (left), mean window size (right)."""
    _ensure_parent(out_path)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xscale("log")
    ax1.plot(df["threshold"], df["fallback_rate"], marker="o", color="tab:blue",
             label="fallback rate")
    ax1.set_xlabel("confidence_threshold (max acceptable model error)")
    ax1.set_ylabel("fallback rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["mean_window_size"], marker="s", color="tab:red",
             label="mean window")
    ax2.set_ylabel("mean local search window", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
