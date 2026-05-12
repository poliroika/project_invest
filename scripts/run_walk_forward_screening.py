#!/usr/bin/env python3
"""CLI: walk-forward mode B — re-screen pairs each train window (PLAN §6)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from pair_trading.config import load_pairs_list, load_project_config
from pair_trading.data_loader import load_close_matrix
from pair_trading.plotting import plot_walk_forward_equity
from pair_trading.preprocessing import filter_timerange_index
from pair_trading.walk_forward import walk_forward_portfolio_screening

app = typer.Typer(help="Walk-forward with per-fold pair screening.")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override max pairs per fold"),
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
    closes = filter_timerange_index(
        closes,
        start=cfg.timerange.start,
        end=cfg.timerange.end,
    )
    if closes.empty:
        typer.echo("No OHLCV loaded for walk-forward screening.", err=True)
        raise typer.Exit(code=1)

    wf_df, sel_df, eq_df = walk_forward_portfolio_screening(
        closes, cfg, top_k=top_k, timeframe=cfg.timeframe
    )

    results_root = Path(cfg.paths.results_dir)
    out_dir = results_root / "walk_forward_screening"
    fold_dir = out_dir / "fold_pair_screening"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    wf_df.to_csv(out_dir / "walk_forward_results.csv", index=False)
    sel_df.to_csv(out_dir / "walk_forward_selected_pairs.csv", index=False)
    if not eq_df.empty:
        eq_df.to_csv(out_dir / "walk_forward_equity.csv", index=False)

    metric_col = "test_total_return" if "test_total_return" in wf_df.columns else "test_sharpe"
    plot_walk_forward_equity(wf_df, fig_dir / "walk_forward_equity.png", ycol=metric_col)

    typer.echo(f"Wrote {out_dir / 'walk_forward_results.csv'} ({len(wf_df)} rows).")
    typer.echo(f"Wrote {out_dir / 'walk_forward_selected_pairs.csv'} ({len(sel_df)} rows).")


if __name__ == "__main__":
    app()
