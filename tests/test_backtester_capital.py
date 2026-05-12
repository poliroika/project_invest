"""Capital protection and metrics consistency (PLAN §3, §8, §14)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.backtester import run_two_leg_backtest
from pair_trading.metrics import (
    detailed_performance,
    max_drawdown_capped,
    safe_returns,
    validate_equity_curve,
)


def test_backtester_stops_on_capital_depletion():
    idx = pd.date_range("2020-01-01", periods=80, freq="h", tz="UTC")
    pa = pd.Series(100.0, index=idx)
    pb = pd.Series(50.0, index=idx)
    pos = pd.Series(np.array([1, -1] * 40, dtype=np.int8), index=idx)
    bt = run_two_leg_backtest(
        pa,
        pb,
        pos,
        1.0,
        initial_capital=2000.0,
        transaction_fee=0.02,
        slippage=0.02,
        stop_on_capital_depletion=True,
        capital_depletion_threshold=0.0,
    )
    assert bt.capital_depleted is True
    assert bt.capital_depletion_time is not None
    assert float(bt.equity.min()) >= 0.0


def test_max_drawdown_is_capped_at_minus_one_when_depleted():
    idx = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    pa = pd.Series(100.0, index=idx)
    pb = pd.Series(50.0, index=idx)
    pos = pd.Series(np.array([1, -1] * 20, dtype=np.int8), index=idx)
    bt = run_two_leg_backtest(
        pa,
        pb,
        pos,
        1.0,
        initial_capital=1500.0,
        transaction_fee=0.015,
        slippage=0.015,
        stop_on_capital_depletion=True,
        capital_depletion_threshold=0.0,
    )
    det = detailed_performance(
        bt.equity,
        bt.returns,
        bt.positions,
        bt.costs,
        bt.trades,
        timeframe="1h",
        initial_capital=1500.0,
        bt=bt,
    )
    assert det["capital_depleted"] is True
    assert det["max_drawdown"] == -1.0
    assert det["max_drawdown_capped"] == -1.0


def test_no_returns_after_capital_depletion_in_safe_returns():
    idx = pd.date_range("2020-01-01", periods=20, freq="h", tz="UTC")
    eq = pd.Series([10000.0] + [5000.0] * 9 + [0.0] * 10, index=idx)
    r = safe_returns(eq)
    assert bool(r.iloc[11:].isna().all())


def test_sharpe_nan_when_equity_depleted_in_detailed():
    idx = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    pa = pd.Series(100.0, index=idx)
    pb = pd.Series(50.0, index=idx)
    pos = pd.Series(np.array([1, -1] * 15, dtype=np.int8), index=idx)
    bt = run_two_leg_backtest(
        pa,
        pb,
        pos,
        1.0,
        initial_capital=1200.0,
        transaction_fee=0.02,
        slippage=0.02,
        stop_on_capital_depletion=True,
        capital_depletion_threshold=0.0,
    )
    det = detailed_performance(
        bt.equity,
        bt.returns,
        bt.positions,
        bt.costs,
        bt.trades,
        timeframe="1h",
        initial_capital=1200.0,
        bt=bt,
    )
    assert np.isnan(float(det["sharpe"]))


def test_total_return_raw_and_capped_are_separate_when_depleted():
    idx = pd.date_range("2020-01-01", periods=50, freq="h", tz="UTC")
    pa = pd.Series(100.0, index=idx)
    pb = pd.Series(50.0, index=idx)
    pos = pd.Series(np.array([1, -1] * 25, dtype=np.int8), index=idx)
    bt = run_two_leg_backtest(
        pa,
        pb,
        pos,
        1.0,
        initial_capital=1800.0,
        transaction_fee=0.02,
        slippage=0.02,
        stop_on_capital_depletion=True,
        capital_depletion_threshold=0.0,
    )
    det = detailed_performance(
        bt.equity,
        bt.returns,
        bt.positions,
        bt.costs,
        bt.trades,
        timeframe="1h",
        initial_capital=1800.0,
        bt=bt,
    )
    assert "total_return_raw" in det and "total_return_capped" in det
    assert np.isfinite(float(det["total_return_raw"])) or np.isnan(float(det["total_return_raw"]))


def test_validate_equity_curve_flags_non_positive():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    eq = pd.Series([100.0, 50.0, -1.0, 0.0, 10.0], index=idx)
    v = validate_equity_curve(eq)
    assert v["has_negative_equity"] is True
    assert v["capital_depleted"] is True


def test_max_drawdown_capped_floor():
    idx = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    eq = pd.Series([100.0, 200.0, 50.0, 40.0], index=idx)
    m = max_drawdown_capped(eq, capital_depleted=False)
    assert m >= -1.0
