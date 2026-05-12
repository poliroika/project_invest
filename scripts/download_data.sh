#!/usr/bin/env bash
# Download OHLCV via Freqtrade (Binance USDT-M futures, 1h).
# Prefer ``--pairs-file``; fall back to explicit ``--pairs`` if unsupported.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAIRS_FILE="configs/pairs_top_crypto.txt"
EXPLICIT_PAIRS=(
  BTC/USDT:USDT ETH/USDT:USDT BNB/USDT:USDT SOL/USDT:USDT XRP/USDT:USDT
  ADA/USDT:USDT DOGE/USDT:USDT AVAX/USDT:USDT LINK/USDT:USDT DOT/USDT:USDT
  LTC/USDT:USDT BCH/USDT:USDT ATOM/USDT:USDT NEAR/USDT:USDT APT/USDT:USDT
  ARB/USDT:USDT OP/USDT:USDT
)

if freqtrade download-data --help 2>/dev/null | grep -q pairs-file; then
  freqtrade download-data \
    --exchange binance \
    --trading-mode futures \
    --pairs-file "$PAIRS_FILE" \
    --timeframes 1h \
    --timerange 20210101-20260101
else
  freqtrade download-data \
    --exchange binance \
    --trading-mode futures \
    --pairs "${EXPLICIT_PAIRS[@]}" \
    --timeframes 1h \
    --timerange 20210101-20260101
fi
