"""Tests for spread / z-score helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.spread import build_spread_and_zscore, rolling_zscore


def test_rolling_zscore_constant_spread_nan_std():
    idx = pd.date_range("2020-01-01", periods=50, freq="h", tz="UTC")
    s = pd.Series(np.ones(50), index=idx)
    z = rolling_zscore(s, window=10, min_periods=10)
    assert np.isnan(z.iloc[9])


def test_build_spread_static_beta():
    idx = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.Series(np.linspace(1, 5, 120), index=idx) + rng.normal(0, 0.01, size=120)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.01, size=120)
    spread, beta_s, z = build_spread_and_zscore(
        y,
        x,
        rolling_window=20,
        use_dynamic_beta=False,
        static_beta=2.0,
        min_train_periods=30,
    )
    assert float(beta_s.iloc[0]) == 2.0
    assert len(spread) == len(z)
