"""
Freqtrade strategy: directional spread / z-score on Asset A vs informative Asset B.

Trading pair (``metadata["pair"]``) is Asset A; ``partner_pair`` is Asset B (informative).

This is a **research approximation**, not full atomic two-leg execution: signals drive **only leg A**
while B supplies the hedge reference via merged OHLCV. **Funding rates, cross-margin, and
synchronized hedge fills are not modeled** (see project report / README).

Requires: ``pip install freqtrade`` (optional extra in ``pyproject.toml``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from freqtrade.strategy import IStrategy
    from pandas import DataFrame
except ImportError:  # pragma: no cover - type stubs without freqtrade
    IStrategy = object  # type: ignore[misc, assignment]

    class DataFrame:  # type: ignore[no-redef]
        pass


class PairTradingCointegrationStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    partner_pair = "BTC/USDT:USDT"
    hedge_ratio = 1.0
    entry_z = 2.0
    exit_z = 0.5
    stop_z = 3.5
    rolling_window = 100
    startup_candle_count = 110

    def informative_pairs(self):
        return [(self.partner_pair, self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        partner = self.partner_pair
        out = dataframe.copy()
        if self.dp:
            inf = self.dp.get_pair_dataframe(pair=partner, timeframe=self.timeframe)
            right = inf[["date", "close", "volume"]].rename(
                columns={"close": "partner_close", "volume": "partner_volume"}
            )
            idx_order = out.sort_values("date").index
            left = out.loc[idx_order, ["date", "close", "volume"]].reset_index(drop=True)
            r2 = right.sort_values("date").reset_index(drop=True)
            merged = pd.merge_asof(left, r2, on="date", direction="backward")
            out.loc[idx_order, "partner_close"] = merged["partner_close"].values
            out.loc[idx_order, "partner_volume"] = merged["partner_volume"].values
        else:
            out["partner_close"] = out["close"]
            out["partner_volume"] = out["volume"]

        ok = (out["volume"].astype(float) > 0) & (out["partner_volume"].astype(float) > 0)

        out["log_close_a"] = np.log(out["close"].astype(float))
        out["log_close_b"] = np.log(out["partner_close"].astype(float))
        beta = float(self.hedge_ratio)
        out["spread"] = out["log_close_a"] - beta * out["log_close_b"]
        rolling_mean = out["spread"].rolling(self.rolling_window).mean()
        rolling_std = out["spread"].rolling(self.rolling_window).std().replace(0, np.nan)
        out["rolling_mean"] = rolling_mean
        out["rolling_std"] = rolling_std
        out["zscore"] = (out["spread"] - rolling_mean) / rolling_std
        out["signal_ok"] = ok.astype(int)
        return out

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        z = dataframe["zscore"]
        ok = dataframe["signal_ok"] == 1
        dataframe.loc[(z < -self.entry_z) & ok, "enter_long"] = 1
        dataframe.loc[(z > self.entry_z) & ok, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        z = dataframe["zscore"]
        za = z.abs()
        ok = dataframe["signal_ok"] == 1
        dataframe.loc[(za < self.exit_z) & ok, "exit_long"] = 1
        dataframe.loc[(za < self.exit_z) & ok, "exit_short"] = 1
        dataframe.loc[(za > self.stop_z) & ok, "exit_long"] = 1
        dataframe.loc[(za > self.stop_z) & ok, "exit_short"] = 1
        return dataframe
