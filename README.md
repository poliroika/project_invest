# Cointegration-Based Pair Trading Strategy with Freqtrade Backtesting

Research-grade **two-leg** pair trading on Python (cointegration, ADF, z-score) plus a **Freqtrade** strategy that trades **one leg** (Asset A) using an informative pair (Asset B) as a hedge reference.

## Project Overview

This repository screens futures pairs (Binance USDT-M, 1h), estimates a hedge ratio, builds a mean-reverting spread and z-score, runs a **synchronous two-leg** backtest with transaction fees and slippage, compares against benchmarks, and supports **calendar walk-forward** (single pair or re-screened universe).

## Research Question

Do log-price spreads from cointegrated crypto futures pairs exhibit exploitable mean reversion **after** realistic two-leg costs?

## Methodology

1. Engle–Granger / ADF / half-life filters (`screen_pairs`).
2. Spread \( \log A - \beta \log B \), rolling z-score, discrete positions in \(\{-1,0,1\}\).
3. Custom `run_two_leg_backtest` with per-leg notionals from `risk` config.
4. Round-trip **trade ledger** (`results/trades.csv`) with exit reasons (`mean_reversion_exit`, `stop_z_exit`, `end_of_backtest_exit`).
5. Benchmarks: buy-and-hold each leg, equal-weight hold, flat **no_trade**, **simple_mean_reversion_no_coint** (β=1, no cointegration filter).

## Repository Structure

| Path | Role |
| --- | --- |
| `src/pair_trading/` | Library: data, cointegration, spread, signals, backtester, metrics, benchmarks, walk-forward, plotting |
| `configs/project_config.yaml` | Exchange, timerange, screening, strategy, **risk**, walk-forward |
| `configs/pairs_top_crypto.txt` | Universe for screening |
| `scripts/` | CLI: `check_data`, `screen_pairs`, `run_single_pair_backtest`, `run_walk_forward`, `run_walk_forward_screening`, `generate_report`, Freqtrade shell helpers |
| `user_data/strategies/` | `PairTradingCointegrationStrategy` (Freqtrade) |
| `results/` | CSV outputs and `figures/` |

## Installation

Using **uv** (recommended):

```powershell
cd <repo-root>
uv sync
```

Using **pip** (PLAN-compatible):

```powershell
pip install -e ".[dev]"
```

Optional Freqtrade CLI:

```powershell
uv sync --extra freqtrade
```

## Data Download

Freqtrade (if installed):

```bash
bash scripts/download_data.sh
```

With `data.auto_download: true` in `configs/project_config.yaml`, missing OHLCV can be fetched at runtime via **CCXT** into `user_data/data/`.

## Pair Screening

```powershell
uv run python scripts/check_data.py --config configs/project_config.yaml
uv run python scripts/screen_pairs.py --config configs/project_config.yaml
```

Outputs: `results/data_availability.csv`, `results/pair_screening_results.csv`, `results/selected_pairs.csv`.

## Custom Two-Leg Backtest

Explicit pair:

```powershell
uv run python scripts/run_single_pair_backtest.py ETH/USDT:USDT BTC/USDT:USDT --config configs/project_config.yaml
```

From `selected_pairs.csv` (top N by correlation):

```powershell
uv run python scripts/run_single_pair_backtest.py --from-selected --top-n 3 --config configs/project_config.yaml
```

Writes: `backtest_summary.csv` (variant B row), `benchmark_comparison.csv`, `trades.csv` (round trips), `equity_curve.csv`, `spread_zscore.csv`, figures under `results/figures/`. For `--top-n` > 1, outputs go to `results/multi_pair/<slug>/` per pair.

## Walk-Forward Validation

Single pair (calendar):

```powershell
uv run python scripts/run_walk_forward.py --asset-a ETH/USDT:USDT --asset-b BTC/USDT:USDT --config configs/project_config.yaml
```

Re-screen each train window (mode B); outputs under `results/walk_forward_screening/`:

```powershell
uv run python scripts/run_walk_forward_screening.py --config configs/project_config.yaml
```

Holdout (train 2021–2024, test 2024–2026) and sensitivity grid:

```powershell
uv run python scripts/run_holdout_experiment.py --config configs/project_config.yaml
uv run python scripts/run_sensitivity_analysis.py --config configs/project_config.yaml
```

## Freqtrade Backtest

Requires Freqtrade installed. Example config: `configs/config_futures.example.json`.

```bash
bash scripts/run_freqtrade_backtest.sh
bash scripts/run_freqtrade_lookahead_analysis.sh
```

The Freqtrade layer is a **practical approximation**: it is not identical to the two-leg Python model (execution, funding, and simultaneous hedge are simplified). The Python backtester is the reference for dollar-neutral two-leg PnL.

## Results

### Results Summary

The current run screened **136** pair combinations (see `results/pair_screening_results.csv`) and selected **2** pairs in `results/selected_pairs.csv`:

- XRP/USDT:USDT — LTC/USDT:USDT  
- DOGE/USDT:USDT — AVAX/USDT:USDT  

The DOGE/AVAX two-leg backtest in the repository snapshot was **negative after transaction costs** (see `results/backtest_summary.csv` or re-run `run_single_pair_backtest.py`). That outcome supports the view that **statistical cointegration alone is not sufficient** for profitable crypto pair trading under the modeled fees and rules.

**Important:** the current strategy is **not** presented as production-ready alpha. Treat the repo as a **research / backtesting framework**, not a turnkey trading system.

After running the pipeline, inspect `results/*.csv` and `results/figures/*.png`. Refresh the narrative section of the report:

```powershell
uv run python scripts/generate_report.py --config configs/project_config.yaml
```

Benchmarks and per-strategy returns are in `results/benchmark_comparison.csv` (and under each `results/multi_pair/<slug>/` when applicable).

### Final Multi-Pair Summary

| Pair | Total Return | Sharpe | Max Drawdown | Win Rate | Profit Factor | Capital Depleted |
|---|---:|---:|---:|---:|---:|---|
| DOGE/AVAX | -30.10% | -0.861 | -37.64% | 54.15% | 0.698 | False |
| XRP/LTC | +50.42% | 0.783 | -18.19% | 66.54% | 1.293 | False |

The results are mixed: one statistically selected pair was profitable, while the other remained unprofitable after costs. This supports the main conclusion that cointegration and ADF filters are useful for candidate selection, but they are not sufficient to guarantee profitable trading.

## Key Findings

Interpretation depends on your data window and screening thresholds. If **selected_pairs.csv** is empty, relax `screening.*` in YAML and document thresholds in `reports/report.md`.

## Limitations

- Past performance does not guarantee future results.
- Crypto futures: funding, liquidation, latency, and depth are not fully modeled.
- Freqtrade strategy ≠ full two-leg replication.

## Future Work

Funding-aware PnL, cross-exchange validation, alternative cointegration tests, and portfolio-level capital allocation across multiple pairs.

## Disclaimer

This project is **educational and research-oriented**. It is **not** financial advice. Trading involves substantial risk of loss.

## Reproduction pipeline

```powershell
uv run python scripts/check_data.py --config configs/project_config.yaml
uv run python scripts/screen_pairs.py --config configs/project_config.yaml
uv run python scripts/run_single_pair_backtest.py --from-selected --top-n 2 --config configs/project_config.yaml
uv run python scripts/run_walk_forward.py --asset-a ETH/USDT:USDT --asset-b BTC/USDT:USDT --config configs/project_config.yaml
uv run python scripts/run_walk_forward_screening.py --config configs/project_config.yaml
uv run python scripts/run_holdout_experiment.py --config configs/project_config.yaml
uv run python scripts/run_sensitivity_analysis.py --config configs/project_config.yaml
uv run python scripts/generate_report.py --config configs/project_config.yaml
uv run pytest -q
uv run ruff check src scripts tests
```

Aggregate metrics for several selected pairs: `results/multi_pair_summary.csv` (created when `--top-n` ≥ 2 with `--from-selected`).
