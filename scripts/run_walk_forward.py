#!/usr/bin/env python3
"""CLI: walk-forward analysis (calendar or bar-based)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from pair_trading.config import load_pairs_list, load_project_config
from pair_trading.data_loader import load_close_matrix, load_pair_close
from pair_trading.preprocessing import filter_timerange_index
from pair_trading.walk_forward import (
    walk_forward_backtest,
    walk_forward_calendar_single_pair,
    walk_forward_screen_pairs_calendar,
)
from pair_trading.plotting import plot_walk_forward_equity

app = typer.Typer(help="Walk-forward rolling train/test evaluation.")


@app.command()
def main(
    asset_a: str | None = typer.Option(
        None,
        "--asset-a",
        help="Leg A; omit with --screen-all to only screen pairs per calendar fold",
    ),
    asset_b: str | None = typer.Option(None, "--asset-b"),
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
    use_calendar: bool = typer.Option(
        True,
        "--calendar/--bars",
        help="Use walk_forward.train_months/test_months/step_months from config",
    ),
    train_len: int = typer.Option(1500, help="[--bars] Training bars per fold"),
    test_len: int = typer.Option(500, help="[--bars] Test bars per fold"),
    step: int = typer.Option(250, help="[--bars] Step between folds"),
    screen_all_pairs: bool = typer.Option(
        False,
        "--screen-all",
        help="Run pair screening on each train window (writes walk_forward_selected_pairs.csv)",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_project_config(config)
    results_dir = Path(cfg.paths.results_dir)
    figures_dir = Path(cfg.paths.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if screen_all_pairs:
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
        closes = filter_timerange_index(
            closes,
            start=cfg.timerange.start,
            end=cfg.timerange.end,
        )
        all_df, sel_df = walk_forward_screen_pairs_calendar(closes, cfg)
        all_path = results_dir / "walk_forward_screening_all_folds.csv"
        sel_path = results_dir / "walk_forward_selected_pairs.csv"
        all_df.to_csv(all_path, index=False)
        sel_df.to_csv(sel_path, index=False)
        typer.echo(f"Wrote {all_path} ({len(all_df)} rows).")
        typer.echo(f"Wrote {sel_path} ({len(sel_df)} rows).")
        if asset_a and asset_b:
            typer.echo("(Also running single-pair WF below.)")
        else:
            return

    if asset_a is None or asset_b is None:
        if not screen_all_pairs:
            typer.echo("Provide --asset-a / --asset-b or use --screen-all.", err=True)
            raise typer.Exit(code=1)
        return

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
    df = filter_timerange_index(df, start=cfg.timerange.start, end=cfg.timerange.end)

    if use_calendar and cfg.walk_forward.enabled:
        wf = walk_forward_calendar_single_pair(
            df.iloc[:, 0],
            df.iloc[:, 1],
            cfg=cfg,
            timeframe=cfg.timeframe,
        )
    else:
        wf = walk_forward_backtest(
            df.iloc[:, 0],
            df.iloc[:, 1],
            train_len=train_len,
            test_len=test_len,
            step=step,
            strategy_cfg=cfg.strategy,
            timeframe=cfg.timeframe,
        )

    out = results_dir / "walk_forward_results.csv"
    if not wf.empty:
        wf = wf.copy()
        wf["asset_a"] = asset_a
        wf["asset_b"] = asset_b
    wf.to_csv(out, index=False)
    typer.echo(f"Wrote {out} ({len(wf)} folds).")

    plot_walk_forward_equity(wf, figures_dir / "walk_forward_equity.png")


if __name__ == "__main__":
    app()
