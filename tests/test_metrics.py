"""Tests for metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.metrics import annualized_return, max_drawdown, sharpe_ratio


def test_max_drawdown_simple():
    eq = pd.Series([100.0, 110.0, 90.0, 95.0])
    assert max_drawdown(eq) < 0


def test_sharpe_finite_on_random_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0001, 0.001, size=500))
    s = sharpe_ratio(r, timeframe="1h")
    assert not np.isnan(s)


def test_annualized_return_nan_when_equity_wiped():
    eq = pd.Series([10000.0, 1000.0, -500.0])
    ar = annualized_return(eq, timeframe="1h")
    assert np.isnan(ar)
