"""Tests for discrete position logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trading.signals import generate_positions


def test_enter_long_spread_when_z_low():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    z = pd.Series([np.nan, np.nan, -2.5, -2.5, -0.1], index=idx)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=5.0)
    assert int(pos.iloc[2]) == 1
    assert int(pos.iloc[4]) == 0  # exit


def test_stop_closes_position():
    idx = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    z = pd.Series([-2.5, -2.5, -4.0, 0.0], index=idx)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    assert int(pos.iloc[1]) == 1
    assert int(pos.iloc[2]) == 0
