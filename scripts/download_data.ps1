# Download OHLCV via Freqtrade (Binance USDT-M futures, 1h). Run from repo root or any cwd.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

freqtrade download-data `
  --exchange binance `
  --trading-mode futures `
  --pairs BTC/USDT:USDT ETH/USDT:USDT BNB/USDT:USDT SOL/USDT:USDT XRP/USDT:USDT ADA/USDT:USDT DOGE/USDT:USDT AVAX/USDT:USDT LINK/USDT:USDT DOT/USDT:USDT LTC/USDT:USDT BCH/USDT:USDT ATOM/USDT:USDT NEAR/USDT:USDT APT/USDT:USDT ARB/USDT:USDT OP/USDT:USDT `
  --timeframes 1h `
  --timerange 20210101-20260101
