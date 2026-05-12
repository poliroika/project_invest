"""Offline tests for runtime download helpers (no network)."""

from __future__ import annotations

from pair_trading.runtime_download import _expected_json_path, _timerange_to_ms


def test_timerange_ms():
    a, b = _timerange_to_ms("2021-01-01", "2021-02-01")
    assert a < b


def test_expected_json_path_futures():
    p = _expected_json_path("user_data/data", "binance", "ETH/USDT:USDT", "1h", "futures")
    assert p.name == "ETH_USDT_USDT-1h-futures.json"
    assert "futures" in p.parts
