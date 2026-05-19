"""Tests for walk-forward evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.config import StrategyConfig
from pair_trading.walk_forward import walk_forward_backtest


def test_bar_walk_forward_accepts_explicit_risk_settings():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=240, freq="h", tz="UTC")
    base = 100 + rng.normal(0.0, 0.2, size=len(idx)).cumsum()
    hedge = 50 + (base - base[0]) * 0.45 + rng.normal(0.0, 0.05, size=len(idx)).cumsum()
    pa = pd.Series(base, index=idx)
    pb = pd.Series(hedge, index=idx)
    st = StrategyConfig(
        rolling_window=20,
        min_train_periods=20,
        entry_z=1.5,
        exit_z=0.25,
        stop_z=4.0,
        min_signal_bars=1,
    )

    out = walk_forward_backtest(
        pa,
        pb,
        train_len=80,
        test_len=40,
        step=40,
        strategy_cfg=st,
        initial_capital=5_000.0,
        leg_capital_fraction=0.25,
        max_position_size_pct=0.25,
        stop_on_capital_depletion=True,
    )

    assert not out.empty
    assert "test_sortino" in out.columns
    assert "capital_depleted" in out.columns
