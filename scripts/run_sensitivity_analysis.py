#!/usr/bin/env python3
"""Parameter sensitivity grid (PLAN §10) — full grid, no cherry-picking."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from pair_trading.backtester import build_round_trip_trades, run_two_leg_backtest
from pair_trading.cointegration import estimate_hedge_ratio
from pair_trading.config import effective_initial_capital, load_project_config
from pair_trading.data_loader import load_pair_close
from pair_trading.metrics import detailed_performance
from pair_trading.preprocessing import filter_timerange_index, log_prices
from pair_trading.signals import generate_positions_with_reasons
from pair_trading.spread import build_spread_and_zscore

app = typer.Typer(help="Sensitivity grid over strategy and cost parameters.")

ENTRY_GRID = (1.5, 2.0, 2.5)
EXIT_GRID = (0.25, 0.5, 0.75)
STOP_GRID = (3.0, 3.5, 4.0)
ROLL_GRID = (50, 100, 200)
FEE_GRID = (0.0002, 0.0004, 0.0008)
SLIP_GRID = (0.0001, 0.0002, 0.0005)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
    asset_a: str = typer.Option("DOGE/USDT:USDT", "--asset-a"),
    asset_b: str = typer.Option("AVAX/USDT:USDT", "--asset-b"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_project_config(config)
    st0 = cfg.strategy
    ic = effective_initial_capital(cfg)

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
    if df.empty or len(df) < st0.min_train_periods:
        typer.echo("Insufficient data for sensitivity run.", err=True)
        raise typer.Exit(code=1)

    ya = log_prices(df.iloc[:, 0])
    yb = log_prices(df.iloc[:, 1])
    beta = estimate_hedge_ratio(ya, yb)

    out_dir = Path(cfg.paths.results_dir) / "sensitivity"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for entry_z, exit_z, stop_z, rolling_window, fee_rate, slippage in itertools.product(
        ENTRY_GRID, EXIT_GRID, STOP_GRID, ROLL_GRID, FEE_GRID, SLIP_GRID
    ):
        spread, _, z = build_spread_and_zscore(
            ya,
            yb,
            rolling_window=int(rolling_window),
            use_dynamic_beta=st0.use_dynamic_beta,
            static_beta=beta if not st0.use_dynamic_beta else None,
            min_train_periods=st0.min_train_periods,
        )
        pos, exit_reasons = generate_positions_with_reasons(
            z,
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            spread=spread,
            min_signal_bars=st0.min_signal_bars,
            cooldown_bars_after_stop=st0.cooldown_bars_after_stop,
            max_holding_bars=st0.max_holding_bars,
            allow_position_flip=st0.allow_position_flip,
            min_spread_volatility=st0.min_spread_volatility,
            max_spread_volatility=st0.max_spread_volatility,
        )
        bt = run_two_leg_backtest(
            df.iloc[:, 0],
            df.iloc[:, 1],
            pos,
            beta,
            initial_capital=ic,
            transaction_fee=float(fee_rate),
            slippage=float(slippage),
            leg_capital_fraction=cfg.risk.leg_capital_fraction,
            max_position_size_pct=cfg.risk.max_position_size_pct,
            stop_on_capital_depletion=cfg.risk.stop_on_capital_depletion,
            capital_depletion_threshold=cfg.risk.capital_depletion_threshold,
        )
        rt = build_round_trip_trades(
            df.iloc[:, 0],
            df.iloc[:, 1],
            bt,
            exit_reasons,
            transaction_fee=float(fee_rate),
            slippage=float(slippage),
        )
        det = detailed_performance(
            bt.equity,
            bt.returns,
            bt.positions,
            bt.costs,
            bt.trades,
            timeframe=cfg.timeframe,
            initial_capital=ic,
            round_trips=rt,
            bt=bt,
        )
        rows.append(
            {
                "asset_a": asset_a,
                "asset_b": asset_b,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "rolling_window": rolling_window,
                "fee_rate": fee_rate,
                "slippage": slippage,
                "total_return_capped": det.get("total_return_capped", det.get("total_return")),
                "sharpe_until_depletion": det.get("sharpe_until_depletion"),
                "max_drawdown": det.get("max_drawdown"),
                "capital_depleted": det.get("capital_depleted", False),
                "fees_paid": det.get("fees_paid"),
                "number_of_trades": det.get("number_of_trades"),
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "sensitivity_results.csv", index=False)

    tr = res["total_return_capped"].astype(float)
    prof = float((tr > 0).mean())
    dep = float(res["capital_depleted"].astype(bool).mean())
    summ = pd.DataFrame(
        [
            {
                "median_total_return": float(tr.median()),
                "q25_total_return": float(tr.quantile(0.25)),
                "q75_total_return": float(tr.quantile(0.75)),
                "worst_total_return": float(tr.min()),
                "best_total_return": float(tr.max()),
                "pct_profitable_configs": prof,
                "pct_capital_depleted": dep,
                "n_configs": int(len(res)),
            }
        ]
    )
    summ.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        pivot = res.pivot_table(
            values="total_return_capped",
            index="exit_z",
            columns="entry_z",
            aggfunc="median",
        )
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        plt.colorbar(label="median total return")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("entry_z")
        plt.ylabel("exit_z")
        plt.title("Sensitivity heatmap (median over other params)")
        plt.tight_layout()
        plt.savefig(fig_dir / "heatmap_entry_exit.png")
        plt.close()

        sub = res[res["rolling_window"] == 100]
        if not sub.empty:
            plt.figure(figsize=(6, 4))
            for fee in sorted(sub["fee_rate"].unique()):
                m = sub[sub["fee_rate"] == fee].groupby("slippage")["total_return_capped"].median()
                plt.plot(m.index.astype(float), m.values, marker="o", label=f"fee={fee}")
            plt.xlabel("slippage")
            plt.ylabel("median total_return_capped")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "fee_sensitivity.png")
            plt.close()
    except Exception as e:
        logging.warning("Figure export skipped: %s", e)

    typer.echo(f"Wrote {out_dir / 'sensitivity_results.csv'} ({len(res)} rows).")


if __name__ == "__main__":
    app()
