"""Performance and risk metrics (PLAN §8, §14)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pair_trading.backtester import BacktestResult


def validate_equity_curve(equity: pd.Series) -> dict[str, Any]:
    """
    Inspect an equity path for non-positive values (PLAN §8.2).

    Returns keys: ``has_negative_equity``, ``capital_depleted`` (any bar at/below zero),
    ``first_depletion_time``, ``min_equity``.
    """
    eq = equity.astype(float)
    if eq.empty:
        return {
            "has_negative_equity": False,
            "capital_depleted": False,
            "first_depletion_time": None,
            "min_equity": float("nan"),
        }
    min_e = float(eq.min())
    le0 = eq <= 0
    has = bool(le0.any())
    first: pd.Timestamp | None = None
    if has:
        first = pd.Timestamp(eq.loc[le0].index[0])
    return {
        "has_negative_equity": bool((eq < 0).any()),
        "capital_depleted": has,
        "first_depletion_time": first,
        "min_equity": min_e,
    }


def safe_returns(equity: pd.Series) -> pd.Series:
    """
    Per-bar simple returns only while strictly positive equity; stops after first
    non-positive level (PLAN §8.2).
    """
    eq = equity.astype(float)
    if eq.empty:
        return pd.Series(dtype=float)
    out = pd.Series(np.nan, index=eq.index, dtype=float)
    for i in range(len(eq)):
        if eq.iloc[i] <= 0:
            out.iloc[i:] = np.nan
            break
        if i == 0:
            out.iloc[i] = 0.0
            continue
        prev = float(eq.iloc[i - 1])
        if prev <= 0:
            out.iloc[i:] = np.nan
            break
        out.iloc[i] = float(eq.iloc[i] / prev - 1.0)
    return out.rename("return")


def max_drawdown_capped(equity: pd.Series, *, capital_depleted: bool) -> float:
    """Max drawdown not below -1.0; returns -1.0 when ``capital_depleted`` (PLAN §3.4)."""
    if capital_depleted:
        return -1.0
    m = max_drawdown(equity)
    if not np.isfinite(m):
        return float(m)
    return float(max(m, -1.0))


def annualization_factor(timeframe: str) -> float:
    """Approximate bars per year from timeframe string like ``1h``, ``4h``, ``1d``."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        n = int(tf[:-1])
        return (365 * 24 * 60) / n
    if tf.endswith("h"):
        n = int(tf[:-1])
        return (365 * 24) / n
    if tf.endswith("d"):
        n = int(tf[:-1])
        return 365 / n
    return 365 * 24


def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(equity: pd.Series, *, timeframe: str = "1h") -> float:
    r = total_return(equity)
    n = len(equity.dropna())
    if n < 2:
        return float("nan")
    ann = annualization_factor(timeframe)
    years = n / ann
    if years <= 0:
        return float("nan")
    base = 1.0 + r
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    out = float(base ** (1.0 / years) - 1.0)
    return out


def volatility(returns: pd.Series, *, timeframe: str = "1h") -> float:
    r = returns.astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    ann = annualization_factor(timeframe)
    return float(r.std(ddof=1) * np.sqrt(ann))


def sharpe_ratio(
    returns: pd.Series,
    *,
    timeframe: str = "1h",
    risk_free: float = 0.0,
) -> float:
    r = returns.astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    ann = annualization_factor(timeframe)
    mu = float(r.mean()) * ann - risk_free
    sd = float(r.std(ddof=1)) * np.sqrt(ann)
    if sd == 0:
        return float("nan")
    return mu / sd


def sortino_ratio(
    returns: pd.Series,
    *,
    timeframe: str = "1h",
    risk_free: float = 0.0,
) -> float:
    r = returns.astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    ann = annualization_factor(timeframe)
    neg = r[r < 0]
    if len(neg) < 2:
        down_dev = float("nan")
    else:
        down_dev = float(neg.std(ddof=1)) * np.sqrt(ann)
    mu = float(r.mean()) * ann - risk_free
    if down_dev == 0 or np.isnan(down_dev):
        return float("nan")
    return mu / down_dev


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def calmar_ratio(equity: pd.Series, *, timeframe: str = "1h") -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    ar = annualized_return(equity, timeframe=timeframe)
    if np.isnan(ar):
        return float("nan")
    return float(ar / mdd)


def cumulative_return(equity: pd.Series) -> float:
    return total_return(equity)


def timeframe_to_bar_hours(timeframe: str) -> float:
    """Approximate hours per bar for ``1h``, ``4h``, ``15m``, etc."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(int(tf[:-1]))
    if tf.endswith("d"):
        return float(int(tf[:-1])) * 24.0
    return 1.0


def win_rate_from_trades_df(trades: pd.DataFrame) -> float:
    """Share of round-trips with positive ``net_pnl``."""
    if trades.empty or "net_pnl" not in trades.columns:
        return float("nan")
    pnl = trades["net_pnl"].astype(float)
    if len(pnl) == 0:
        return float("nan")
    return float((pnl > 0).mean())


def profit_factor_from_trades_df(trades: pd.DataFrame) -> float:
    """Gross wins / gross losses on ``net_pnl`` of round-trips."""
    if trades.empty or "net_pnl" not in trades.columns:
        return float("nan")
    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def average_trade_return_from_trades_df(trades: pd.DataFrame) -> float:
    if trades.empty or "net_pnl" not in trades.columns:
        return float("nan")
    return float(trades["net_pnl"].astype(float).mean())


def average_trade_duration_bars_from_trades_df(trades: pd.DataFrame) -> float:
    if trades.empty or "duration_bars" not in trades.columns:
        return float("nan")
    return float(trades["duration_bars"].astype(float).mean())


def profit_factor(returns: pd.Series) -> float:
    r = returns.astype(float).dropna()
    gains = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / abs(losses))


def win_rate(returns: pd.Series) -> float:
    r = returns.astype(float).dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def exposure_time(positions: pd.Series) -> float:
    p = positions.reindex(positions.index).fillna(0).astype(float)
    if len(p) == 0:
        return float("nan")
    return float((p != 0).mean())


def fees_paid(costs: pd.Series) -> float:
    return float(costs.astype(float).sum())


def number_of_trades(trades: pd.DataFrame) -> int:
    return int(len(trades.index))


def average_trade_return(equity: pd.Series, trades: pd.DataFrame) -> float:
    """Mean equity change at rebalance timestamps (approximate per-trade edge)."""
    if trades.empty or equity.empty:
        return float("nan")
    changes: list[float] = []
    for ts in trades["timestamp"]:
        try:
            loc = equity.index.get_loc(ts)
        except KeyError:
            continue
        if isinstance(loc, slice):
            continue
        i = int(loc) if isinstance(loc, (int, np.integer)) else int(loc.start)
        if i <= 0:
            continue
        changes.append(float(equity.iloc[i] - equity.iloc[i - 1]))
    if not changes:
        return float("nan")
    return float(np.mean(changes))


def average_trade_duration(trades: pd.DataFrame, bar_delta_hours: float = 1.0) -> float:
    """Hours between consecutive rebalance events (1h bars -> delta 1)."""
    if len(trades) < 2:
        return float("nan")
    ts = pd.to_datetime(trades["timestamp"], utc=True)
    dt = ts.diff().dt.total_seconds() / 3600.0
    return float(dt.iloc[1:].mean()) if len(dt) > 1 else float("nan")


def summarize_backtest(
    equity: pd.Series,
    returns: pd.Series,
    *,
    timeframe: str = "1h",
) -> dict[str, float]:
    """Compact summary (backward compatible keys)."""
    return {
        "cumulative_return": cumulative_return(equity),
        "total_return": total_return(equity),
        "sharpe": sharpe_ratio(returns, timeframe=timeframe),
        "max_drawdown": max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "annualized_return": annualized_return(equity, timeframe=timeframe),
        "volatility": volatility(returns, timeframe=timeframe),
    }


def detailed_performance(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    costs: pd.Series,
    trades: pd.DataFrame,
    *,
    timeframe: str = "1h",
    initial_capital: float = 10_000.0,
    round_trips: pd.DataFrame | None = None,
    bt: Any = None,
) -> dict[str, float | bool | str | None]:
    """Extended metrics per PLAN §8.2; pass ``bt`` (:class:`~pair_trading.backtester.BacktestResult`) for depletion-aware fields."""
    capital_depleted = bool(getattr(bt, "capital_depleted", False)) if bt is not None else False
    eq_for_safe = (
        getattr(bt, "equity_accounting", None) if bt is not None and getattr(bt, "equity_accounting", None) is not None else equity
    )
    safe_ret = safe_returns(eq_for_safe.astype(float)).dropna()
    max_dd_raw_val = float(getattr(bt, "max_drawdown_raw", float("nan"))) if bt is not None else max_drawdown(
        eq_for_safe.astype(float)
    )
    min_eq_bt = float(getattr(bt, "min_equity", float("nan"))) if bt is not None else float("nan")
    dep_ts = getattr(bt, "capital_depletion_time", None) if bt is not None else None
    dep_str = dep_ts.isoformat() if dep_ts is not None else ""

    final_raw = float(equity.iloc[-1]) if len(equity) else float("nan")
    if bt is not None:
        fr = float(getattr(bt, "final_equity_raw", float("nan")))
        if np.isfinite(fr):
            final_raw = fr
    total_return_capped = float(equity.iloc[-1] / float(initial_capital) - 1.0) if len(equity) else float("nan")
    total_return_raw = (
        float(final_raw / float(initial_capital) - 1.0) if np.isfinite(final_raw) else total_return_capped
    )

    if capital_depleted:
        base = {
            "cumulative_return": total_return_capped,
            "total_return": total_return_capped,
            "sharpe": float("nan"),
            "max_drawdown": -1.0,
            "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
            "annualized_return": float("nan"),
            "volatility": float("nan"),
        }
        sharpe_ud = sharpe_ratio(safe_ret, timeframe=timeframe)
        sortino_ud = sortino_ratio(safe_ret, timeframe=timeframe)
    else:
        base = summarize_backtest(equity, returns, timeframe=timeframe)
        sharpe_ud = sharpe_ratio(safe_ret, timeframe=timeframe) if len(safe_ret) > 1 else float("nan")
        sortino_ud = sortino_ratio(safe_ret, timeframe=timeframe) if len(safe_ret) > 1 else float("nan")

    ann = annualization_factor(timeframe)
    bars_per_year = ann
    years = len(returns.dropna()) / bars_per_year if bars_per_year else float("nan")

    if round_trips is not None and not round_trips.empty:
        wr = win_rate_from_trades_df(round_trips)
        pf = profit_factor_from_trades_df(round_trips)
        atr = average_trade_return_from_trades_df(round_trips)
        dur_bars = average_trade_duration_bars_from_trades_df(round_trips)
        bar_h = timeframe_to_bar_hours(timeframe)
        avg_dur_h = dur_bars * bar_h if np.isfinite(dur_bars) else float("nan")
        n_tr = int(len(round_trips))
    else:
        wr = win_rate(returns)
        pf = profit_factor(returns)
        atr = average_trade_return(equity, trades)
        avg_dur_h = average_trade_duration(trades)
        n_tr = number_of_trades(trades)

    mdd_capped = max_drawdown_capped(equity.astype(float), capital_depleted=capital_depleted)

    extra = {
        "initial_equity": float(initial_capital),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "sortino": float("nan") if capital_depleted else sortino_ratio(returns, timeframe=timeframe),
        "sortino_ratio": float("nan") if capital_depleted else sortino_ratio(returns, timeframe=timeframe),
        "calmar": float("nan") if capital_depleted else calmar_ratio(equity, timeframe=timeframe),
        "calmar_ratio": float("nan") if capital_depleted else calmar_ratio(equity, timeframe=timeframe),
        "profit_factor": pf,
        "win_rate": wr,
        "exposure_time": exposure_time(positions),
        "fees_paid": fees_paid(costs),
        "number_of_trades": float(n_tr),
        "average_trade_return": atr,
        "average_trade_duration": float(avg_dur_h),
        "average_trade_duration_hours": float(avg_dur_h),
        "initial_capital": float(initial_capital),
        "total_return_raw": total_return_raw,
        "total_return_capped": total_return_capped,
        "capital_depleted": capital_depleted,
        "capital_depletion_time": dep_str,
        "min_equity": min_eq_bt if np.isfinite(min_eq_bt) else validate_equity_curve(eq_for_safe)["min_equity"],
        "stopped_early": capital_depleted,
        "max_drawdown_raw": max_dd_raw_val,
        "max_drawdown_capped": mdd_capped,
        "sharpe_until_depletion": sharpe_ud,
        "sortino_until_depletion": sortino_ud,
    }
    base.update(extra)
    _ = years
    return base
