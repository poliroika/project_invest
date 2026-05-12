"""Tests for two-leg backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.backtester import build_round_trip_trades, run_two_leg_backtest
from pair_trading.signals import generate_positions_with_reasons


def test_flat_markets_flat_position_constant_equity_minus_costs():
    idx = pd.date_range("2020-01-01", periods=20, freq="h", tz="UTC")
    pa = pd.Series(100.0, index=idx)
    pb = pd.Series(50.0, index=idx)
    pos = pd.Series(0, index=idx, dtype=np.int8)
    bt = run_two_leg_backtest(pa, pb, pos, hedge_ratio=1.0, initial_capital=10_000.0)
    assert np.allclose(bt.equity.values, 10_000.0)


def test_long_spread_scales_with_price_move():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    pa = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    pb = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0], index=idx)
    pos = pd.Series([1, 1, 1, 1, 1], index=idx, dtype=np.int8)
    bt = run_two_leg_backtest(pa, pb, pos, hedge_ratio=2.0, initial_capital=10_000.0)
    assert bt.equity.iloc[-1] != bt.equity.iloc[0]


def test_round_trip_trade_on_open_close():
    idx = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    pa = pd.Series([100.0, 100.0, 100.0, 101.0, 101.0, 101.0], index=idx)
    pb = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0, 50.0], index=idx)
    z = pd.Series([0.0, 0.0, -2.5, -2.0, -0.1, 0.0], index=idx)
    pos, reasons = generate_positions_with_reasons(z, entry_z=2.0, exit_z=0.5, stop_z=5.0)
    bt = run_two_leg_backtest(pa, pb, pos, hedge_ratio=1.0, initial_capital=10_000.0)
    rt = build_round_trip_trades(pa, pb, bt, reasons)
    assert len(rt) >= 1
    assert "net_pnl" in rt.columns
