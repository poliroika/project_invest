"""Load and validate project configuration (YAML + defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TimerangeConfig(BaseModel):
    start: str = "2021-01-01"
    end: str = "2026-01-01"


class DataConfig(BaseModel):
    ohlcv_root: str = "user_data/data"
    min_observations: int = 500
    max_nan_fraction: float = 0.02
    # If True and OHLCV files are missing, fetch from Binance USDT-M via CCXT into ``ohlcv_root``.
    auto_download: bool = True


class ScreeningConfig(BaseModel):
    correlation_min: float = 0.75
    use_log_returns_for_correlation: bool = False
    coint_pvalue_max: float = 0.05
    adf_pvalue_max: float = 0.05
    half_life_min: float = 0.0
    half_life_max_candles: float = 200.0
    # If set, use only the last N aligned bars for cointegration/ADF/half-life/beta (much faster on long history).
    max_bars_for_hypothesis_tests: int | None = None


class StrategyConfig(BaseModel):
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    rolling_window: int = 100
    min_train_periods: int = 500
    use_log_prices: bool = True
    use_dynamic_beta: bool = False
    transaction_fee: float = 0.0004
    slippage: float = 0.0002
    min_signal_bars: int = 2
    cooldown_bars_after_stop: int = 24
    max_holding_bars: int = 240
    allow_position_flip: bool = False
    min_spread_volatility: float | None = None
    max_spread_volatility: float | None = None


class PathsConfig(BaseModel):
    pairs_file: str = "configs/pairs_top_crypto.txt"
    results_dir: str = "results"
    figures_dir: str = "results/figures"


class RiskConfig(BaseModel):
    """Position sizing (PLAN §7.3). ``initial_capital`` overrides ``project.initial_capital`` when set."""

    initial_capital: float | None = None
    leg_capital_fraction: float = 0.5
    max_position_size_pct: float = 0.25
    stop_on_capital_depletion: bool = True
    capital_depletion_threshold: float = 0.0
    max_drawdown_stop: float = 0.5
    stop_after_consecutive_losses: int = 5
    min_equity_to_trade: float = 1000.0


class ProjectMeta(BaseModel):
    """Optional project-level defaults (see PLAN §16)."""

    name: str = "Cointegration Pair Trading"
    initial_capital: float = 10_000.0
    base_currency: str = "USDT"


class WalkForwardConfig(BaseModel):
    """Calendar walk-forward (PLAN §11.2)."""

    enabled: bool = True
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    min_pairs_per_window: int = 1
    max_pairs_per_window: int = 5


class ProjectConfig(BaseModel):
    exchange: str = "binance"
    trading_mode: str = "futures"
    timeframe: str = "1h"
    timerange: TimerangeConfig = Field(default_factory=TimerangeConfig)
    project: ProjectMeta = Field(default_factory=ProjectMeta)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load YAML config and return a validated ``ProjectConfig``."""
    p = Path(path)
    raw: dict[str, Any]
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProjectConfig.model_validate(raw)


def load_pairs_list(path: str | Path) -> list[str]:
    """Load newline-separated pair symbols (Freqtrade convention)."""
    lines: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def effective_initial_capital(cfg: ProjectConfig) -> float:
    """``risk.initial_capital`` if set, otherwise ``project.initial_capital``."""
    if cfg.risk.initial_capital is not None:
        return float(cfg.risk.initial_capital)
    return float(cfg.project.initial_capital)
