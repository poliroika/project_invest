"""Download OHLCV at runtime into Freqtrade-compatible JSON (Binance USDT-M via CCXT)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _timerange_to_ms(start: str, end: str) -> tuple[int, int]:
    import pandas as pd

    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return int(s.timestamp() * 1000), int(e.timestamp() * 1000)


def _expected_json_path(
    ohlcv_root: str | Path,
    exchange: str,
    pair: str,
    timeframe: str,
    trading_mode: str,
) -> Path:
    from pair_trading.data_loader import pair_to_filename_stem

    stem = pair_to_filename_stem(pair)
    root = Path(ohlcv_root) / exchange
    if trading_mode.lower() == "futures":
        return root / "futures" / f"{stem}-{timeframe}-futures.json"
    return root / f"{stem}-{timeframe}.json"


def fetch_binance_futures_ohlcv(
    pair: str,
    timeframe: str,
    ms_start: int,
    ms_end: int,
    *,
    limit_per_call: int = 1500,
) -> list[list[float]]:
    """
    Fetch OHLCV from Binance USDT-M using CCXT (public REST, no API keys).

    Returns rows ``[timestamp_ms, open, high, low, close, volume]`` sorted ascending.
    """
    try:
        import ccxt  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Runtime download requires the `ccxt` package. Install: pip install ccxt"
        ) from e

    ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    tf_ms = int(ex.parse_timeframe(timeframe) * 1000)

    out: list[list[float]] = []
    since = ms_start
    for _ in range(500):
        batch: list[list[float]] = ex.fetch_ohlcv(
            pair,
            timeframe,
            since=since,
            limit=limit_per_call,
        )
        if not batch:
            break
        for row in batch:
            ts = int(row[0])
            if ts < ms_start:
                continue
            if ts > ms_end:
                continue
            out.append(
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
            )
        last_ts = int(batch[-1][0])
        if last_ts >= ms_end:
            break
        nxt = last_ts + tf_ms
        if nxt <= since:
            nxt = since + tf_ms
        since = nxt
        time.sleep(max(ex.rateLimit / 1000.0, 0.05) if ex.rateLimit else 0.05)

    seen: set[int] = set()
    unique: list[list[float]] = []
    for row in sorted(out, key=lambda r: r[0]):
        ts = int(row[0])
        if ts not in seen:
            seen.add(ts)
            unique.append(row)
    return unique


def save_freqtrade_json(path: Path, ohlcv: list[list[Any]]) -> None:
    """Write Freqtrade-style JSON: list of ``[ms, o, h, l, c, v]``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(ohlcv, f)
    tmp.replace(path)


def ensure_pair_ohlcv_file(
    *,
    ohlcv_root: str | Path,
    exchange: str,
    pair: str,
    timeframe: str,
    trading_mode: str,
    timerange_start: str,
    timerange_end: str,
) -> Path:
    """
    Download missing OHLCV and save next to where ``load_pair_close`` expects it.

    Raises ``RuntimeError`` if nothing was fetched (symbol/network issue).
    """
    if exchange.lower() != "binance":
        raise NotImplementedError(
            f"Runtime download is implemented only for exchange=binance, got {exchange}"
        )
    if trading_mode.lower() != "futures":
        raise NotImplementedError(
            "Runtime download supports futures mode (Binance USDT-M) only in this project."
        )

    out = _expected_json_path(ohlcv_root, exchange, pair, timeframe, trading_mode)
    ms_start, ms_end = _timerange_to_ms(timerange_start, timerange_end)

    logger.info(
        "Downloading OHLCV %s %s from Binance USDT-M (%s → %s) → %s",
        pair,
        timeframe,
        timerange_start,
        timerange_end,
        out,
    )

    rows = fetch_binance_futures_ohlcv(pair, timeframe, ms_start, ms_end)
    if not rows:
        raise RuntimeError(
            f"No candles returned for {pair} — check symbol and network access to Binance."
        )

    save_freqtrade_json(out, rows)
    logger.info("Saved %s candles to %s", len(rows), out)
    return out
