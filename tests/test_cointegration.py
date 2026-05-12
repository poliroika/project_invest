"""Tests for cointegration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.cointegration import (
    calculate_half_life,
    calculate_spread,
    estimate_hedge_ratio,
    run_adf_test,
    run_cointegration_test,
)


def test_estimate_hedge_ratio_linear():
    rng = np.random.default_rng(0)
    x = pd.Series(np.linspace(1, 10, 200)) + rng.normal(0, 0.01, size=200)
    y = 1.5 + 2.0 * x + rng.normal(0, 0.01, size=200)
    beta = estimate_hedge_ratio(y, x)
    assert abs(beta - 2.0) < 0.05


def test_spread_matches_beta():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([2.0, 4.0, 6.0])
    sp = calculate_spread(y, x, beta=2.0)
    assert np.allclose(sp.astype(float).values, [0.0, 0.0, 0.0])


def test_half_life_positive_on_mean_reverting_ar1():
    rng = np.random.default_rng(1)
    phi = 0.85
    s = np.zeros(500)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    hl = calculate_half_life(pd.Series(s))
    assert hl > 0
    assert hl < 500


def test_adf_stationary_series_low_pvalue():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=800))
    res = run_adf_test(x)
    assert res["pvalue"] < 0.05


def test_cointegration_test_handles_ndarray_critical_values():
    """statsmodels may return ndarray for ``coint`` critical values."""
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=300).cumsum())
    y = pd.Series(0.5 * x.values + rng.normal(size=300))
    res = run_cointegration_test(y, x)
    assert isinstance(res["critical_values"], dict)
    assert "statistic" in res


def test_run_cointegration_test_returns_expected_keys():
    rng = np.random.default_rng(42)
    x = pd.Series(np.cumsum(rng.normal(size=500)))
    y = 1.5 * x + rng.normal(scale=0.5, size=500)
    res = run_cointegration_test(y, x)
    assert "statistic" in res
    assert "pvalue" in res
    assert "critical_values" in res
    assert isinstance(res["pvalue"], float)
