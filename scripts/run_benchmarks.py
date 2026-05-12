#!/usr/bin/env python3
"""CLI: benchmarks vs strategy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from pair_trading.benchmarks import benchmark_returns, equal_weight_buy_and_hold
from pair_trading.config import load_project_config
from pair_trading.data_loader import load_pair_close
from pair_trading.metrics import summarize_backtest

app = typer.Typer(help="Compute equal-weight buy-and-hold benchmark metrics.")


@app.command()
def main(
    asset_a: str | None = typer.Option(
        None,
        "--asset-a",
        help="First leg; if omitted, first row of results/selected_pairs.csv is used",
    ),
    asset_b: str | None = typer.Option(None, "--asset-b"),
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
) -> None:
    cfg = load_project_config(config)
    results_dir = Path(cfg.paths.results_dir)

    if asset_a is None or asset_b is None:
        sel = results_dir / "selected_pairs.csv"
        if not sel.is_file():
            typer.echo("Provide --asset-a/--asset-b or create selected_pairs.csv via screen_pairs.", err=True)
            raise typer.Exit(code=1)
        row = pd.read_csv(sel).iloc[0]
        asset_a = str(row["asset_a"])
        asset_b = str(row["asset_b"])

    pa = load_pair_close(
        cfg.data.ohlcv_root,
        cfg.exchange,
        asset_a,
        cfg.timeframe,
        cfg.trading_mode,
        auto_download=cfg.data.auto_download,
        timerange_start=cfg.timerange.start,
        timerange_end=cfg.timerange.end,
    )
    pb = load_pair_close(
        cfg.data.ohlcv_root,
        cfg.exchange,
        asset_b,
        cfg.timeframe,
        cfg.trading_mode,
        auto_download=cfg.data.auto_download,
        timerange_start=cfg.timerange.start,
        timerange_end=cfg.timerange.end,
    )

    df = pd.concat([pa, pb], axis=1).dropna()
    bench_eq = equal_weight_buy_and_hold(df.iloc[:, 0], df.iloc[:, 1])
    bench_summ = summarize_backtest(bench_eq, benchmark_returns(bench_eq), timeframe=cfg.timeframe)

    rows = [{"name": "equal_weight_buy_and_hold", **bench_summ}]
    out = results_dir / "benchmark_comparison.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        prev = pd.read_csv(out)
        prev = prev[prev["name"] != "equal_weight_buy_and_hold"]
        merged = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
        merged.to_csv(out, index=False)
    else:
        pd.DataFrame(rows).to_csv(out, index=False)

    typer.echo(f"Appended benchmark metrics -> {out}")


if __name__ == "__main__":
    app()
