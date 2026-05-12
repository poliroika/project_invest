"""Benchmark comparisons — PLAN §9–10."""

from __future__ import annotations

import pandas as pd

from pair_trading.backtester import run_two_leg_backtest
from pair_trading.metrics import (
    annualized_return,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    volatility,
)
from pair_trading.preprocessing import log_prices
from pair_trading.signals import generate_positions
from pair_trading.spread import build_spread_and_zscore


def buy_and_hold(
    price: pd.Series,
    *,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Alias for single-asset buy-and-hold (PLAN §9)."""
    return buy_and_hold_asset(price, initial_capital=initial_capital)


def buy_and_hold_asset(price: pd.Series, *, initial_capital: float = 10_000.0) -> pd.Series:
    """100% capital in a single asset; equity = initial / P0 * Pt."""
    p = price.astype(float).dropna()
    if p.empty:
        return pd.Series(dtype=float)
    q = float(initial_capital) / float(p.iloc[0])
    return (q * p).rename("bh_equity")


def equal_weight_buy_and_hold(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Split capital 50/50, buy and hold both legs."""
    pa = price_a.astype(float)
    pb = price_b.astype(float)
    df = pd.concat([pa, pb], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    half = float(initial_capital) / 2.0
    qa = half / float(df.iloc[0, 0])
    qb = half / float(df.iloc[0, 1])
    equity = qa * df.iloc[:, 0] + qb * df.iloc[:, 1]
    return equity.rename("benchmark_equity")


def no_trade_baseline(index: pd.Index, *, initial_capital: float = 10_000.0) -> pd.Series:
    """Flat equity curve at ``initial_capital`` (PLAN §9)."""
    ic = float(initial_capital)
    return pd.Series(ic, index=index, dtype=float).rename("no_trade_equity")


def naive_mean_reversion_without_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    strategy_cfg: object,
) -> pd.Series:
    """
    Same two-leg execution as the main strategy but **hedge ratio fixed at 1**
    (no OLS / Engle–Granger): spread on log prices ``log A - log B``.
    """
    st = strategy_cfg
    df = pd.concat([price_a, price_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    ya = log_prices(df.iloc[:, 0])
    yb = log_prices(df.iloc[:, 1])
    spread, _, z = build_spread_and_zscore(
        ya,
        yb,
        rolling_window=st.rolling_window,
        use_dynamic_beta=False,
        static_beta=1.0,
        min_train_periods=st.min_train_periods,
    )
    pos = generate_positions(
        z,
        entry_z=st.entry_z,
        exit_z=st.exit_z,
        stop_z=st.stop_z,
    )
    bt = run_two_leg_backtest(
        df.iloc[:, 0],
        df.iloc[:, 1],
        pos,
        hedge_ratio=1.0,
        transaction_fee=st.transaction_fee,
        slippage=st.slippage,
    )
    return bt.equity


def simple_mean_reversion_no_coint(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    strategy_cfg: object,
) -> pd.Series:
    """PLAN name for ``naive_mean_reversion_without_cointegration``."""
    return naive_mean_reversion_without_cointegration(
        price_a, price_b, strategy_cfg=strategy_cfg
    )


def benchmark_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def build_six_benchmark_series(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    strategy_cfg: object,
    initial_capital: float = 10_000.0,
) -> dict[str, pd.Series]:
    """PLAN §9.1 — equity curves for six named benchmarks + strategy slot filled by caller."""
    idx = pd.concat([price_a, price_b], axis=1).dropna().index
    return {
        "buy_hold_asset_a": buy_and_hold_asset(price_a, initial_capital=initial_capital),
        "buy_hold_asset_b": buy_and_hold_asset(price_b, initial_capital=initial_capital),
        "equal_weight_buy_hold": equal_weight_buy_and_hold(
            price_a, price_b, initial_capital=initial_capital
        ),
        "no_trade": no_trade_baseline(idx, initial_capital=initial_capital),
        "simple_mean_reversion_no_coint": simple_mean_reversion_no_coint(
            price_a, price_b, strategy_cfg=strategy_cfg
        ),
    }


def build_four_benchmark_series(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    strategy_cfg: object,
    initial_capital: float = 10_000.0,
) -> dict[str, pd.Series]:
    """Return equity curves for BH A, BH B, equal-weight, naive mean-reversion (β=1)."""
    six = build_six_benchmark_series(
        price_a, price_b, strategy_cfg=strategy_cfg, initial_capital=initial_capital
    )
    return {
        "buy_hold_asset_a": six["buy_hold_asset_a"],
        "buy_hold_asset_b": six["buy_hold_asset_b"],
        "equal_weight_buy_hold": six["equal_weight_buy_hold"],
        "naive_mean_reversion_beta1": six["simple_mean_reversion_no_coint"],
    }


def build_benchmark_comparison_variant_b(
    *,
    pair_label: str,
    strategy_equity: pd.Series,
    benchmark_equities: dict[str, pd.Series],
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Wide table: strategy, pair, returns, risk metrics (PLAN §8.3 variant B / §9.1)."""
    rows: list[dict[str, object]] = []
    all_curves = {"pair_trading": strategy_equity, **benchmark_equities}
    for name, eq in all_curves.items():
        if eq is None or eq.empty:
            continue
        r = benchmark_returns(eq)
        rows.append(
            {
                "strategy": name,
                "pair": pair_label,
                "total_return": total_return(eq),
                "annualized_return": annualized_return(eq, timeframe=timeframe),
                "volatility": volatility(r, timeframe=timeframe),
                "sharpe": sharpe_ratio(r, timeframe=timeframe),
                "sortino": sortino_ratio(r, timeframe=timeframe),
                "max_drawdown": max_drawdown(eq),
                "calmar": calmar_ratio(eq, timeframe=timeframe),
            }
        )
    return pd.DataFrame(rows)
