"""Spread construction and rolling statistics (z-score inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling OLS slope beta from Cov(y,x)/Var(x) on aligned series."""
    mp = min_periods or window
    c = y.rolling(window, min_periods=mp).cov(x)
    v = x.rolling(window, min_periods=mp).var()
    beta = c / v.replace(0, np.nan)
    return beta.rename("beta")


def rolling_zscore(spread: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Z-score of spread vs rolling mean/std."""
    mp = min_periods or window
    if mp > window:
        mp = window
    m = spread.rolling(window, min_periods=mp).mean()
    s = spread.rolling(window, min_periods=mp).std(ddof=1)
    z = (spread - m) / s.replace(0, np.nan)
    return z.rename("zscore")


def build_spread_and_zscore(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    *,
    rolling_window: int,
    use_dynamic_beta: bool,
    static_beta: float | None = None,
    min_train_periods: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return ``spread``, ``beta_series``, ``zscore``.

    If ``use_dynamic_beta``, beta is rolling Cov/Var; else ``static_beta`` must be set.
    """
    y = log_price_a.astype(float)
    x = log_price_b.astype(float)

    if use_dynamic_beta:
        beta_s = rolling_hedge_ratio(y, x, rolling_window, min_periods=min_train_periods)
        spread = y - beta_s * x
    else:
        if static_beta is None:
            raise ValueError("static_beta required when use_dynamic_beta is False")
        beta_s = pd.Series(float(static_beta), index=y.index)
        spread = y - float(static_beta) * x

    z = rolling_zscore(spread, rolling_window, min_periods=min_train_periods)
    return spread.rename("spread"), beta_s, z
