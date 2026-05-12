"""Figures for results/figures/*."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_equity_curve(equity: pd.Series, out: Path, *, title: str = "Equity") -> None:
    ensure_dir(out.parent)
    plt.figure(figsize=(10, 4))
    equity.astype(float).plot(ax=plt.gca(), color="#1f77b4")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_equity_curve_multi(
    equities: dict[str, pd.Series],
    out: Path,
    *,
    title: str = "Equity curves",
) -> None:
    """Strategy + benchmarks on one chart (PLAN §14.1 ``equity_curve.png``)."""
    ensure_dir(out.parent)
    plt.figure(figsize=(11, 4))
    ax = plt.gca()
    for name, eq in equities.items():
        if eq is None or eq.empty:
            continue
        eq.astype(float).plot(ax=ax, label=name, alpha=0.85)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_drawdown(equity: pd.Series, out: Path, *, title: str = "Drawdown") -> None:
    ensure_dir(out.parent)
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    mdd_idx = dd.idxmin()
    plt.figure(figsize=(10, 3))
    dd.plot(ax=plt.gca(), color="#d62728")
    plt.title(title)
    plt.ylabel("Drawdown")
    if np.isfinite(mdd) and mdd_idx is not None and not pd.isna(mdd_idx):
        plt.annotate(
            f"max DD {mdd:.2%}",
            xy=(mdd_idx, mdd),
            xytext=(10, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.5),
            fontsize=9,
        )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_spread_zscore(
    spread: pd.Series,
    z: pd.Series,
    out: Path,
    *,
    title: str = "Spread and z-score",
) -> None:
    ensure_dir(out.parent)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    spread.astype(float).plot(ax=ax1, color="gray", alpha=0.8, label="spread")
    ax1.set_ylabel("spread")
    ax2 = ax1.twinx()
    z.astype(float).plot(ax=ax2, color="#ff7f0e", alpha=0.9, label="z-score")
    ax2.set_ylabel("z-score")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_rolling_beta(beta: pd.Series, out: Path, *, title: str = "Rolling beta") -> None:
    ensure_dir(out.parent)
    plt.figure(figsize=(10, 3))
    beta.astype(float).plot(ax=plt.gca(), color="#2ca02c")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_rolling_correlation(corr: pd.Series, out: Path, *, title: str = "Rolling correlation") -> None:
    ensure_dir(out.parent)
    plt.figure(figsize=(10, 3))
    corr.astype(float).plot(ax=plt.gca(), color="#9467bd")
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_trades_on_spread(
    spread: pd.Series,
    position: pd.Series,
    out: Path,
    *,
    title: str = "Spread with position overlay",
    round_trips: pd.DataFrame | None = None,
) -> None:
    ensure_dir(out.parent)
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    spread.astype(float).plot(ax=ax, color="gray", alpha=0.7, label="spread")
    pos = position.reindex(spread.index).fillna(0).astype(float)
    ax.fill_between(pos.index, pos.min(), pos.max(), where=(pos > 0), color="green", alpha=0.08)
    ax.fill_between(pos.index, pos.min(), pos.max(), where=(pos < 0), color="red", alpha=0.08)

    if round_trips is not None and not round_trips.empty and "entry_time" in round_trips.columns:
        s_aligned = spread.reindex(spread.index)
        for _, row in round_trips.iterrows():
            try:
                te = pd.Timestamp(row["entry_time"])
                tx = pd.Timestamp(row["exit_time"])
                if te in s_aligned.index:
                    ax.scatter([te], [float(s_aligned.loc[te])], color="blue", s=22, marker="^", zorder=5)
                if tx in s_aligned.index:
                    ax.scatter([tx], [float(s_aligned.loc[tx])], color="black", s=18, marker="x", zorder=5)
            except Exception:  # noqa: BLE001
                continue

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_benchmark_comparison(
    equity_strategy: pd.Series,
    equity_benchmark: pd.Series,
    out: Path,
    *,
    title: str = "Strategy vs benchmark",
) -> None:
    ensure_dir(out.parent)
    plt.figure(figsize=(10, 4))
    equity_strategy.astype(float).plot(ax=plt.gca(), label="strategy")
    equity_benchmark.astype(float).plot(ax=plt.gca(), label="benchmark")
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_spread_zscore_detailed(
    spread: pd.Series,
    z: pd.Series,
    out: Path,
    *,
    rolling_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    title: str = "Spread / z-score / thresholds",
) -> None:
    """PLAN §15.1 — spread with rolling mean; z-score with entry/exit bands."""
    ensure_dir(out.parent)
    spread_mean = spread.astype(float).rolling(rolling_window, min_periods=rolling_window).mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    spread.astype(float).plot(ax=axes[0], color="gray", alpha=0.65, label="spread")
    spread_mean.plot(ax=axes[0], color="blue", alpha=0.75, label=f"rolling mean ({rolling_window})")
    axes[0].set_ylabel("spread")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    zf = z.astype(float)
    zf.plot(ax=axes[1], color="#ff7f0e", alpha=0.85, label="z-score")
    axes[1].axhline(entry_z, color="red", linestyle="--", alpha=0.7, label="±entry")
    axes[1].axhline(-entry_z, color="red", linestyle="--", alpha=0.7)
    axes[1].axhline(exit_z, color="green", linestyle=":", alpha=0.7, label="±exit")
    axes[1].axhline(-exit_z, color="green", linestyle=":", alpha=0.7)
    axes[1].axhline(stop_z, color="purple", linestyle="-.", alpha=0.6, label="±stop")
    axes[1].axhline(-stop_z, color="purple", linestyle="-.", alpha=0.6)
    axes[1].set_ylabel("z-score")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_benchmark_grid(equities: dict[str, pd.Series], out: Path, *, title: str = "Benchmarks") -> None:
    """Overlay multiple normalized equity curves (start at 1.0)."""
    ensure_dir(out.parent)
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    for name, eq in equities.items():
        if eq is None or eq.empty:
            continue
        s = eq.astype(float)
        if len(s) == 0:
            continue
        norm = s / float(s.iloc[0])
        norm.plot(ax=ax, label=name, alpha=0.85)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_walk_forward_equity(wf_results: pd.DataFrame, out: Path, *, ycol: str = "test_cumulative_return") -> None:
    """Bar chart of per-fold test metric (PLAN §15 walk-forward figure)."""
    ensure_dir(out.parent)
    if wf_results.empty or ycol not in wf_results.columns:
        plt.figure(figsize=(8, 3))
        plt.title("Walk-forward (no rows)")
        plt.savefig(out, dpi=120)
        plt.close()
        return
    plt.figure(figsize=(11, 4))
    x = range(len(wf_results))
    plt.bar(x, wf_results[ycol].astype(float).values, color="steelblue", alpha=0.85)
    plt.xticks(x, [f"{i}" for i in x])
    plt.ylabel(ycol)
    plt.xlabel("fold")
    plt.title("Walk-forward test segments")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_all_standard(
    *,
    equity: pd.Series,
    spread: pd.Series,
    z: pd.Series,
    beta: pd.Series | None,
    position: pd.Series,
    benchmark_equity: pd.Series | None,
    corr_roll: pd.Series | None,
    figures_dir: Path,
    round_trips: pd.DataFrame | None = None,
) -> None:
    figures_dir = Path(figures_dir)
    plot_equity_curve(equity, figures_dir / "equity_curve.png")
    plot_drawdown(equity, figures_dir / "drawdown.png")
    plot_spread_zscore(spread, z, figures_dir / "spread_zscore.png")
    plot_trades_on_spread(
        spread, position, figures_dir / "trades_on_spread.png", round_trips=round_trips
    )
    if beta is not None:
        plot_rolling_beta(beta, figures_dir / "rolling_beta.png")
    if corr_roll is not None:
        plot_rolling_correlation(corr_roll, figures_dir / "rolling_correlation.png")
    if benchmark_equity is not None:
        plot_benchmark_comparison(equity, benchmark_equity, figures_dir / "benchmark_comparison.png")
