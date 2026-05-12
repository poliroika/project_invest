#!/usr/bin/env python3
"""CLI: screen cointegrated pairs."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from pair_trading.cointegration import screen_pairs
from pair_trading.config import load_pairs_list, load_project_config
from pair_trading.data_loader import load_close_matrix
from pair_trading.preprocessing import filter_timerange_index

app = typer.Typer(help="Screen pairs (Engle–Granger, ADF, half-life).")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config", help="YAML config"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_project_config(config)
    pairs = load_pairs_list(cfg.paths.pairs_file)

    closes = load_close_matrix(
        cfg.data.ohlcv_root,
        cfg.exchange,
        pairs,
        cfg.timeframe,
        cfg.trading_mode,
        inner_join=True,
        auto_download=cfg.data.auto_download,
        timerange_start=cfg.timerange.start,
        timerange_end=cfg.timerange.end,
    )
    if closes.empty:
        typer.echo(
            "No OHLCV data loaded. Check internet, install ccxt (`pip install ccxt`), "
            f"or place files under {cfg.data.ohlcv_root}/{cfg.exchange}. "
            "Set data.auto_download: true in config to fetch from Binance.",
            err=True,
        )
        raise typer.Exit(code=1)

    closes = filter_timerange_index(
        closes,
        start=cfg.timerange.start,
        end=cfg.timerange.end,
    )

    out_df = screen_pairs(closes, cfg)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "pair_screening_results.csv"
    out_df.to_csv(out_path, index=False)

    selected = out_df[out_df["selected"]].copy()
    selected_path = results_dir / "selected_pairs.csv"
    selected.to_csv(selected_path, index=False)

    typer.echo(f"Wrote {out_path} ({len(out_df)} rows).")
    typer.echo(f"Selected pairs -> {selected_path} ({len(selected)} rows).")


if __name__ == "__main__":
    app()
