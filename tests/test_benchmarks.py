"""Tests for benchmark helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.benchmarks import (
    build_benchmark_comparison_variant_b,
    build_six_benchmark_series,
    no_trade_baseline,
)
from pair_trading.config import StrategyConfig


def _dummy_strategy_cfg() -> StrategyConfig:
    return StrategyConfig(
        rolling_window=20,
        min_train_periods=30,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=5.0,
    )


def test_no_trade_baseline_flat():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    s = no_trade_baseline(idx, initial_capital=1234.0)
    assert np.allclose(s.values, 1234.0)


def test_build_six_benchmark_series_keys():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    pa = pd.Series(100 + rng.normal(size=len(idx)).cumsum() * 0.01, index=idx)
    pb = pd.Series(50 + rng.normal(size=len(idx)).cumsum() * 0.01, index=idx)
    st = _dummy_strategy_cfg()
    out = build_six_benchmark_series(pa, pb, strategy_cfg=st, initial_capital=10_000.0)
    for k in (
        "buy_hold_asset_a",
        "buy_hold_asset_b",
        "equal_weight_buy_hold",
        "no_trade",
        "simple_mean_reversion_no_coint",
    ):
        assert k in out


def test_benchmark_comparison_variant_b_shape():
    idx = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
    pa = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    pb = pd.Series(np.linspace(50, 48, len(idx)), index=idx)
    st = _dummy_strategy_cfg()
    strat = pa * 10
    bench = build_six_benchmark_series(pa, pb, strategy_cfg=st)
    df = build_benchmark_comparison_variant_b(
        pair_label="A|B",
        strategy_equity=strat,
        benchmark_equities=bench,
        timeframe="1h",
    )
    assert "strategy" in df.columns and "pair" in df.columns
    assert len(df) >= 6
