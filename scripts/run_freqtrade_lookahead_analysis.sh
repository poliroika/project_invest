#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results
OUT="results/freqtrade_lookahead_analysis.txt"

freqtrade lookahead-analysis \
  --config configs/config_futures.example.json \
  --strategy PairTradingCointegrationStrategy \
  --strategy-path user_data/strategies \
  --timerange 20210101-20260101 \
  | tee "$OUT"
