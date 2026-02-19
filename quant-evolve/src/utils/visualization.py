"""
Visualization utilities for QuantEvolve.

These helpers are intentionally lightweight and tolerant to partially
available inputs so the pipeline can always emit plots without crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _prepare_output_path(output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, output_path: Union[str, Path]) -> None:
    path = _prepare_output_path(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _safe_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.dropna()
    if isinstance(values, (list, tuple, np.ndarray)):
        return pd.Series(values).dropna()
    return pd.Series(dtype=float)


def plot_feature_map_evolution(
    feature_map_snapshots: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Plot occupancy/diversity evolution from feature-map snapshots."""
    snapshots = list(feature_map_snapshots or [])
    fig, ax = plt.subplots(figsize=(10, 5))

    if not snapshots:
        ax.text(0.5, 0.5, "No feature map snapshots", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    x = np.arange(len(snapshots))
    occupancy = [float(s.get("occupancy_rate", s.get("feature_map_occupancy_rate", 0.0))) for s in snapshots]
    diversity = [float(s.get("diversity_score", 0.0)) for s in snapshots]

    ax.plot(x, occupancy, label="Occupancy Rate", linewidth=2)
    ax.plot(x, diversity, label="Diversity Score", linewidth=2)
    ax.set_title("Feature Map Evolution")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_sharpe_vs_generation(
    generation_metrics: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Plot Sharpe-like metric by generation."""
    metrics = list(generation_metrics or [])
    fig, ax = plt.subplots(figsize=(10, 5))

    if not metrics:
        ax.text(0.5, 0.5, "No generation metrics", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    generations = [int(m.get("generation", i)) for i, m in enumerate(metrics)]
    sharpe = [
        float(
            m.get("best_sharpe_ratio", m.get("sharpe_ratio", m.get("best_sharpe", 0.0)))
        )
        for m in metrics
    ]

    ax.plot(generations, sharpe, color="tab:blue", linewidth=2)
    ax.set_title("Sharpe vs Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_cumulative_returns(
    strategy_returns: Union[pd.Series, Sequence[float]],
    benchmark_returns: Union[pd.Series, Sequence[float]],
    output_path: Union[str, Path],
) -> None:
    """Plot cumulative returns for strategy and benchmark."""
    strat = _safe_series(strategy_returns)
    bench = _safe_series(benchmark_returns)

    fig, ax = plt.subplots(figsize=(10, 5))

    if strat.empty and bench.empty:
        ax.text(0.5, 0.5, "No return series available", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    if not strat.empty:
        strat_cum = (1.0 + strat).cumprod() - 1.0
        ax.plot(strat_cum.values, label="Strategy", linewidth=2)

    if not bench.empty:
        bench_cum = (1.0 + bench).cumprod() - 1.0
        ax.plot(bench_cum.values, label="Benchmark", linewidth=2)

    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_category_distribution(
    category_distribution: Dict[str, int],
    output_path: Union[str, Path],
) -> None:
    """Plot distribution of strategies across categories."""
    distribution = category_distribution or {}
    fig, ax = plt.subplots(figsize=(10, 5))

    if not distribution:
        ax.text(0.5, 0.5, "No category distribution data", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    names = list(distribution.keys())
    counts = [int(distribution[k]) for k in names]
    ax.bar(names, counts, color="tab:green", alpha=0.8)
    ax.set_title("Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_insight_timeline(
    insights_timeline: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Plot number of insights over generations."""
    timeline = list(insights_timeline or [])
    fig, ax = plt.subplots(figsize=(10, 5))

    if not timeline:
        ax.text(0.5, 0.5, "No insight timeline data", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    gen_counts: Dict[int, int] = {}
    for item in timeline:
        gen = int(item.get("generation", 0))
        gen_counts[gen] = gen_counts.get(gen, 0) + 1

    generations = sorted(gen_counts.keys())
    counts = [gen_counts[g] for g in generations]
    ax.plot(generations, counts, marker="o", linewidth=2, color="tab:orange")
    ax.set_title("Insight Timeline")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Insights Count")
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_insight_evolution(
    insights_timeline: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Backward-compatible alias."""
    plot_insight_timeline(insights_timeline, output_path)


def plot_3d_feature_map(
    points: Sequence[Union[Tuple[float, float, float], Dict[str, float]]],
    output_path: Union[str, Path],
) -> None:
    """Plot a simple 3D scatter for feature-map coordinates."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for p in points or []:
        if isinstance(p, dict):
            xs.append(float(p.get("x", 0.0)))
            ys.append(float(p.get("y", 0.0)))
            zs.append(float(p.get("z", 0.0)))
        elif isinstance(p, (tuple, list)) and len(p) >= 3:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))

    if xs:
        ax.scatter(xs, ys, zs, alpha=0.8)
    else:
        ax.text2D(0.35, 0.5, "No 3D feature points", transform=ax.transAxes)

    ax.set_title("3D Feature Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _save_figure(fig, output_path)


def plot_dimension_pair_heatmap(
    matrix_like: Any,
    output_path: Union[str, Path],
) -> None:
    """Plot a 2D heatmap from a matrix-like object."""
    fig, ax = plt.subplots(figsize=(7, 6))

    if isinstance(matrix_like, pd.DataFrame):
        matrix = matrix_like.values
    else:
        matrix = np.asarray(matrix_like) if matrix_like is not None else np.array([])

    if matrix.size == 0:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, output_path)
        return

    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Dimension Pair Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, output_path)
