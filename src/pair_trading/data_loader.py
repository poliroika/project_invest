"""Load and align OHLCV series for multiple symbols (Freqtrade-style layout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def pair_to_filename_stem(pair: str) -> str:
    """Freqtrade file stem: ``BTC/USDT:USDT`` -> ``BTC_USDT_USDT``."""
    return pair.replace("/", "_").replace(":", "_")


def discover_ohlcv_path(
    ohlcv_root: str | Path,
    exchange: str,
    pair: str,
    timeframe: str,
    trading_mode: str,
) -> Path | None:
    """
    Locate a single OHLCV file for ``pair`` under Freqtrade-style ``user_data/data``.

    Tries common layouts (futures subfolder, ``-futures`` suffix in filename).
    """
    root = Path(ohlcv_root)
    stem = pair_to_filename_stem(pair)
    mode = trading_mode.lower()
    candidates: list[Path] = []

    sub = root / exchange
    if mode == "futures":
        candidates.extend(
            [
                sub / "futures" / f"{stem}-{timeframe}-futures.feather",
                sub / "futures" / f"{stem}-{timeframe}-futures.json",
                sub / "futures" / f"{stem}-{timeframe}-futures.json.gz",
            ]
        )
    candidates.extend(
        [
            sub / f"{stem}-{timeframe}.feather",
            sub / f"{stem}-{timeframe}.json",
            sub / f"{stem}-{timeframe}.json.gz",
            sub / f"{stem}-{timeframe}-futures.feather",
            sub / f"{stem}-{timeframe}-futures.json",
            sub / f"{stem}-{timeframe}-futures.json.gz",
        ]
    )

    for c in candidates:
        if c.is_file():
            return c

    # Broad glob fallback (slow but tolerant)
    if sub.is_dir():
        for pat in (f"{stem}-{timeframe}*.json*", f"{stem}-{timeframe}*.feather"):
            hits = list(sub.rglob(pat))
            if hits:
                return hits[0]
    return None


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lowercase OHLCV columns and a ``date`` column."""
    lower_map = {str(c).lower(): c for c in df.columns}
    date_key = None
    for key in ("date", "timestamp"):
        if key in lower_map:
            date_key = lower_map[key]
            break
    if date_key is None:
        raise ValueError("OHLCV frame must contain date/timestamp column")

    out = df.rename(columns={date_key: "date"})
    for need in ("open", "high", "low", "close", "volume"):
        if need not in out.columns:
            alt = lower_map.get(need)
            if alt is not None:
                out = out.rename(columns={alt: need})
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return out[["date", "open", "high", "low", "close", "volume"]]


def _read_ohlcv_file(path: Path) -> pd.DataFrame:
    """Read JSON / feather OHLCV into a DataFrame indexed by UTC datetime."""
    suf = path.suffix.lower()
    if suf == ".feather":
        try:
            df = pd.read_feather(path)
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading .feather requires pyarrow support.") from e
        df = _normalize_ohlcv_columns(df)
    elif suf == ".json" or path.name.endswith(".json.gz"):
        raw = path.read_bytes()
        if path.name.endswith(".gz"):
            import gzip

            raw = gzip.decompress(raw)
        data = json.loads(raw.decode("utf-8"))
        if not data:
            raise ValueError(f"Empty OHLCV JSON: {path}")
        first = data[0]
        if isinstance(first, (list, tuple)) and len(first) >= 6:
            df = pd.DataFrame(
                data, columns=["date", "open", "high", "low", "close", "volume"]
            )
        elif isinstance(first, dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unrecognized OHLCV JSON shape in {path}")
        df = _normalize_ohlcv_columns(df)
    else:
        raise ValueError(f"Unsupported OHLCV format: {path}")

    col = df["date"]
    if pd.api.types.is_datetime64_any_dtype(col):
        ts = pd.to_datetime(col, utc=True)
    else:
        ts = pd.to_datetime(col, unit="ms", utc=True)
    df = df.assign(date=ts).set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_pair_close(
    ohlcv_root: str | Path,
    exchange: str,
    pair: str,
    timeframe: str,
    trading_mode: str,
    *,
    auto_download: bool = False,
    timerange_start: str | None = None,
    timerange_end: str | None = None,
) -> pd.Series:
    """Load close series for one pair; optionally fetch Binance futures OHLCV via CCXT."""
    path = discover_ohlcv_path(ohlcv_root, exchange, pair, timeframe, trading_mode)
    if path is None and auto_download and timerange_start and timerange_end:
        from pair_trading.runtime_download import ensure_pair_ohlcv_file

        ensure_pair_ohlcv_file(
            ohlcv_root=ohlcv_root,
            exchange=exchange,
            pair=pair,
            timeframe=timeframe,
            trading_mode=trading_mode,
            timerange_start=timerange_start,
            timerange_end=timerange_end,
        )
        path = discover_ohlcv_path(ohlcv_root, exchange, pair, timeframe, trading_mode)
    if path is None:
        raise FileNotFoundError(
            f"No OHLCV file for {pair} under {ohlcv_root}/{exchange} "
            f"(timeframe={timeframe}, mode={trading_mode}). "
            "Enable data.auto_download in config (requires ccxt + internet) or run scripts/download_data.ps1."
        )
    df = _read_ohlcv_file(path)
    return df["close"].rename(pair)


def load_close_matrix(
    ohlcv_root: str | Path,
    exchange: str,
    pairs: Iterable[str],
    timeframe: str,
    trading_mode: str,
    *,
    inner_join: bool = True,
    auto_download: bool = False,
    timerange_start: str | None = None,
    timerange_end: str | None = None,
) -> pd.DataFrame:
    """
    Load ``close`` columns for many pairs and concat on time index.

    Missing pairs are skipped with a warning-like empty behavior: only loads
    pairs that exist on disk.
    """
    series_list: list[pd.Series] = []
    for pair in pairs:
        try:
            series_list.append(
                load_pair_close(
                    ohlcv_root,
                    exchange,
                    pair,
                    timeframe,
                    trading_mode,
                    auto_download=auto_download,
                    timerange_start=timerange_start,
                    timerange_end=timerange_end,
                )
            )
        except FileNotFoundError:
            continue

    if not series_list:
        return pd.DataFrame()

    out = pd.concat(series_list, axis=1)
    if inner_join:
        out = out.dropna(how="any")
    return out.sort_index()
