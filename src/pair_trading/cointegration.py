"""
Engle–Granger, ADF, half-life, pair screening.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint

from pair_trading.preprocessing import log_prices_df, pair_quality_mask


def _format_critical_values(crit: Any) -> dict[str, float]:
    """Normalize ``coint`` critical values (dict or ndarray across statsmodels versions)."""
    if crit is None:
        return {}
    if hasattr(crit, "items"):
        return {str(k): float(v) for k, v in crit.items()}
    arr = np.asarray(crit, dtype=float).ravel()
    labels = ("1%", "5%", "10%")
    return {
        label: float(value)
        for label, value in zip(labels, arr, strict=False)
        if np.isfinite(value)
    }


# Backwards compatibility
_coint_critical_values_to_dict = _format_critical_values


def _is_valid_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v))


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """
    Estimate hedge ratio using OLS regression:
    y = alpha + beta * x + error.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    yv = df.iloc[:, 0].astype(float).values
    xv = df.iloc[:, 1].astype(float).values
    x_design = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(x_design, yv, rcond=None)[0][1]
    return float(beta)


def calculate_spread(y: pd.Series, x: pd.Series, beta: float) -> pd.Series:
    """
    Calculate spread as y - beta * x (aligned index).
    """
    out = y.astype(float) - float(beta) * x.astype(float)
    return out.rename("spread")


def run_cointegration_test(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    """
    Run Engle–Granger cointegration test.
    Return test statistic, p-value, critical values.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 20:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "critical_values": {},
        }
    yv = df.iloc[:, 0].astype(float).values
    xv = df.iloc[:, 1].astype(float).values
    coint_t, pvalue, crit = coint(yv, xv)
    return {
        "statistic": float(coint_t),
        "pvalue": float(pvalue),
        "critical_values": _format_critical_values(crit),
    }


def run_adf_test(series: pd.Series, *, autolag: str = "AIC") -> dict[str, Any]:
    """
    Run Augmented Dickey-Fuller test.
    Return statistic, p-value, used lags, critical values.
    """
    s = series.dropna().astype(float).values
    if len(s) < 20:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "usedlag": None,
            "critical_values": {},
        }
    res = adfuller(s, autolag=autolag)
    stat, pvalue, usedlag, _, crit, _ = res
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "critical_values": {str(k): float(v) for k, v in crit.items()},
    }


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using:
    delta_spread_t = a + b * spread_{t-1} + error_t
    half_life = -ln(2) / b  (TZ §7.8)
    """
    s = spread.dropna().astype(float)
    if len(s) < 30:
        return float("nan")
    lag = s.shift(1)
    delta = s.diff()
    df = pd.concat([delta, lag], axis=1).dropna()
    df.columns = ["delta", "lag"]
    if len(df) < 20:
        return float("nan")
    if df["lag"].std() == 0:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(df["lag"].values, df["delta"].values)
    b = float(slope)
    if b >= 0:
        return float("nan")
    return float(-np.log(2.0) / b)


def screen_pairs(price_df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """
    Iterate over all asset combinations and return pair statistics.

    ``config`` must expose ``data``, ``screening`` sections compatible with
    :class:`pair_trading.config.ProjectConfig`.
    """
    data_cfg = getattr(config, "data")
    scr = getattr(config, "screening")

    cols = list(price_df.columns)
    rows: list[dict[str, Any]] = []

    logp = log_prices_df(price_df)

    pair_list = list(itertools.combinations(cols, 2))
    n_pairs = len(pair_list)
    log = logging.getLogger(__name__)

    for k, (a, b) in enumerate(pair_list):
        if k == 0 or (k + 1) % 25 == 0 or k == n_pairs - 1:
            log.info("Pair screening %s / %s (%s vs %s)", k + 1, n_pairs, a, b)
        y = logp[a]
        x = logp[b]
        start = price_df.index.min()
        end = price_df.index.max()

        if not pair_quality_mask(
            y,
            x,
            min_observations=data_cfg.min_observations,
            max_nan_fraction=data_cfg.max_nan_fraction,
        ):
            rows.append(_empty_row(a, b, start, end))
            continue

        aligned = pd.concat([y, x], axis=1).dropna()
        y_a = aligned.iloc[:, 0]
        x_b = aligned.iloc[:, 1]
        mx = getattr(scr, "max_bars_for_hypothesis_tests", None)
        if mx is not None and mx > 0 and len(y_a) > mx:
            y_a = y_a.iloc[-mx:]
            x_b = x_b.iloc[-mx:]

        if scr.use_log_returns_for_correlation:
            lr = pd.concat(
                [
                    np.log(price_df[a].astype(float)).diff(),
                    np.log(price_df[b].astype(float)).diff(),
                ],
                axis=1,
            ).dropna()
            corr = float(lr.iloc[:, 0].corr(lr.iloc[:, 1])) if len(lr) > 5 else float("nan")
        else:
            corr = float(y_a.corr(x_b))

        beta = estimate_hedge_ratio(y_a, x_b)
        spread = calculate_spread(y_a, x_b, beta)

        coint_res = run_cointegration_test(y_a, x_b)
        adf_res = run_adf_test(spread)
        hl = calculate_half_life(spread)

        obs = int(len(y_a))
        cp = coint_res["pvalue"]
        ap = adf_res["pvalue"]
        selected = (
            _is_valid_number(corr)
            and _is_valid_number(cp)
            and _is_valid_number(ap)
            and _is_valid_number(hl)
            and _is_valid_number(beta)
            and float(corr) >= scr.correlation_min
            and float(cp) < scr.coint_pvalue_max
            and float(ap) < scr.adf_pvalue_max
            and scr.half_life_min < float(hl) < scr.half_life_max_candles
        )

        rows.append(
            {
                "asset_a": a,
                "asset_b": b,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "observations": obs,
                "correlation": corr,
                "coint_stat": coint_res["statistic"],
                "coint_pvalue": coint_res["pvalue"],
                "adf_stat": adf_res["statistic"],
                "adf_pvalue": adf_res["pvalue"],
                "hedge_ratio": beta,
                "half_life": hl,
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std(ddof=1)) if obs > 1 else float("nan"),
                "selected": selected,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["selected"] = out["selected"].astype(bool)
    return out


def _empty_row(asset_a: str, asset_b: str, start: Any, end: Any) -> dict[str, Any]:
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "start_date": start.isoformat() if hasattr(start, "isoformat") else str(start),
        "end_date": end.isoformat() if hasattr(end, "isoformat") else str(end),
        "observations": 0,
        "correlation": float("nan"),
        "coint_stat": float("nan"),
        "coint_pvalue": float("nan"),
        "adf_stat": float("nan"),
        "adf_pvalue": float("nan"),
        "hedge_ratio": float("nan"),
        "half_life": float("nan"),
        "spread_mean": float("nan"),
        "spread_std": float("nan"),
        "selected": False,
    }
