"""Log prices, returns, quality filters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_prices(close: pd.Series) -> pd.Series:
    """Natural log of close; invalid/zero prices become NaN."""
    s = close.astype(float)
    return np.log(s.where(s > 0))


def log_prices_df(closes: pd.DataFrame) -> pd.DataFrame:
    """Apply log to each column (non-positive values become NaN)."""
    x = closes.astype(float)
    return np.log(x.where(x > 0))


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def filter_timerange_index(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    """Slice DataFrame/Series index to ``[start, end]`` (UTC)."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return df.loc[(df.index >= s) & (df.index <= e)]


def pair_quality_mask(
    y: pd.Series,
    x: pd.Series,
    *,
    min_observations: int,
    max_nan_fraction: float,
) -> bool:
    """Return True if aligned (y, x) passes length and NaN fraction checks."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < min_observations:
        return False
    combined = pd.concat([y, x], axis=1)
    nan_frac = float(combined.isna().any(axis=1).mean())
    return nan_frac <= max_nan_fraction
