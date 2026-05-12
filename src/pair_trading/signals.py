"""Z-score trading signals (states 0, 1, -1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_positions_with_reasons(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    spread: pd.Series | None = None,
    min_signal_bars: int = 1,
    cooldown_bars_after_stop: int = 0,
    max_holding_bars: int = 1_000_000,
    allow_position_flip: bool = True,
    min_spread_volatility: float | None = None,
    max_spread_volatility: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Map z-score to discrete position in {-1, 0, 1} with exit reasons.

    When ``spread`` is provided with volatility bounds, rolling std of spread
    (same window length as ``z``'s effective history) gates new entries only.
    ``min_signal_bars`` requires that many consecutive bars beyond entry threshold
    before opening. ``cooldown_bars_after_stop`` blocks new entries after a stop.
    ``max_holding_bars`` forces a flat after that many bars in a trade.
    If ``allow_position_flip`` is False, an opposite-direction entry first goes flat.
    """
    pos = pd.Series(np.int8(0), index=z.index, dtype=np.int8)
    reasons = pd.Series("", index=z.index, dtype=object)
    current = 0
    cool = 0
    consec_long = 0
    consec_short = 0
    entry_bar_idx: int | None = None

    def _vol_ok(i: int) -> bool:
        if spread is None:
            return True
        if min_spread_volatility is None and max_spread_volatility is None:
            return True
        w = min(100, max(5, i + 1))
        seg = spread.iloc[max(0, i - w + 1) : i + 1].astype(float)
        if len(seg) < 2:
            return True
        v = float(seg.std(ddof=1))
        if min_spread_volatility is not None and v < float(min_spread_volatility):
            return False
        if max_spread_volatility is not None and v > float(max_spread_volatility):
            return False
        return True

    for i in range(len(z)):
        zi = z.iloc[i]
        if zi is None or (isinstance(zi, float) and np.isnan(zi)):
            pos.iloc[i] = current
            continue
        zi_f = float(zi)
        vol_ok = _vol_ok(i)

        if cool > 0:
            cool -= 1

        if current != 0:
            bars_in_trade = (i - entry_bar_idx) if entry_bar_idx is not None else 0
            if bars_in_trade >= max_holding_bars:
                current = 0
                reasons.iloc[i] = "max_holding_exit"
                consec_long = consec_short = 0
                entry_bar_idx = None
            elif abs(zi_f) > stop_z:
                current = 0
                reasons.iloc[i] = "stop_z_exit"
                consec_long = consec_short = 0
                entry_bar_idx = None
                if cooldown_bars_after_stop > 0:
                    cool = int(cooldown_bars_after_stop)
            elif abs(zi_f) < exit_z:
                current = 0
                reasons.iloc[i] = "mean_reversion_exit"
                consec_long = consec_short = 0
                entry_bar_idx = None
            else:
                want_short = zi_f > entry_z
                want_long = zi_f < -entry_z
                if not allow_position_flip and ((current == 1 and want_short) or (current == -1 and want_long)):
                    current = 0
                    reasons.iloc[i] = "no_flip_exit"
                    consec_long = consec_short = 0
                    entry_bar_idx = None
        else:
            if cool > 0 or not vol_ok:
                consec_long = consec_short = 0
            else:
                if zi_f < -entry_z:
                    consec_long += 1
                    consec_short = 0
                elif zi_f > entry_z:
                    consec_short += 1
                    consec_long = 0
                else:
                    consec_long = consec_short = 0

                if consec_long >= min_signal_bars:
                    current = 1
                    entry_bar_idx = i
                    consec_long = consec_short = 0
                elif consec_short >= min_signal_bars:
                    current = -1
                    entry_bar_idx = i
                    consec_long = consec_short = 0

        pos.iloc[i] = current

    return pos.rename("position"), reasons.rename("exit_reason")


def generate_positions(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    spread: pd.Series | None = None,
    min_signal_bars: int = 1,
    cooldown_bars_after_stop: int = 0,
    max_holding_bars: int = 1_000_000,
    allow_position_flip: bool = True,
    min_spread_volatility: float | None = None,
    max_spread_volatility: float | None = None,
) -> pd.Series:
    """
    Map z-score to discrete position in {-1, 0, 1}.

    1 = long spread (long A, short B)
    -1 = short spread (short A, long B)
    0 = flat

    Rules (exit/stop evaluated before new entries when in a position).
    """
    pos, _ = generate_positions_with_reasons(
        z,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        spread=spread,
        min_signal_bars=min_signal_bars,
        cooldown_bars_after_stop=cooldown_bars_after_stop,
        max_holding_bars=max_holding_bars,
        allow_position_flip=allow_position_flip,
        min_spread_volatility=min_spread_volatility,
        max_spread_volatility=max_spread_volatility,
    )
    return pos
