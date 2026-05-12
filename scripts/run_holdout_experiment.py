#!/usr/bin/env python3
"""Holdout experiment: screen on train, backtest on test (PLAN §5)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from pair_trading.backtester import build_equity_curve_dataframe, build_round_trip_trades, run_two_leg_backtest
from pair_trading.benchmarks import build_benchmark_comparison_variant_b, build_six_benchmark_series
from pair_trading.cointegration import estimate_hedge_ratio, screen_pairs
from pair_trading.config import effective_initial_capital, load_pairs_list, load_project_config
from pair_trading.data_loader import load_close_matrix
from pair_trading.metrics import detailed_performance
from pair_trading.plotting import plot_all_standard, plot_benchmark_grid, plot_equity_curve_multi, plot_spread_zscore_detailed
from pair_trading.preprocessing import filter_timerange_index, log_prices
from pair_trading.signals import generate_positions_with_reasons
from pair_trading.spread import build_spread_and_zscore

app = typer.Typer(help="Train/test holdout pair trading experiment.")

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-01-01"
TEST_START = "2024-01-01"
TEST_END = "2026-01-01"


def _pair_slug(a: str, b: str) -> str:
    return f"{a.replace('/', '_').replace(':', '_')}__{b.replace('/', '_').replace(':', '_')}"


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_project_config(config)
    st = cfg.strategy
    ic = effective_initial_capital(cfg)
    pairs = load_pairs_list(cfg.paths.pairs_file)

    closes = load_close_matrix(
        cfg.data.ohlcv_root,
        cfg.exchange,
        pairs,
        cfg.timeframe,
        cfg.trading_mode,
        inner_join=True,
        auto_download=cfg.data.auto_download,
        timerange_start=TRAIN_START,
        timerange_end=TEST_END,
    )
    closes = filter_timerange_index(closes, start=TRAIN_START, end=TEST_END)
    closes = closes[~closes.index.duplicated(keep="first")]
    if closes.empty:
        typer.echo("No OHLCV for holdout range.", err=True)
        raise typer.Exit(code=1)

    train_df = filter_timerange_index(closes, start=TRAIN_START, end=TRAIN_END)
    test_df = filter_timerange_index(closes, start=TEST_START, end=TEST_END)
    if len(train_df) < cfg.data.min_observations or test_df.empty:
        typer.echo("Insufficient train or test data for holdout.", err=True)
        raise typer.Exit(code=1)

    out_dir = Path(cfg.paths.results_dir) / "holdout"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    screened = screen_pairs(train_df, cfg)
    screened.to_csv(out_dir / "pair_screening_train.csv", index=False)
    selected = screened[screened["selected"]].copy()
    selected.to_csv(out_dir / "selected_pairs_train.csv", index=False)

    if selected.empty:
        typer.echo("No pairs passed screening on train window; wrote screening CSV only.", err=True)
        pd.DataFrame().to_csv(out_dir / "backtest_summary_test.csv", index=False)
        return

    summ_rows: list[dict[str, object]] = []
    bench_rows: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_eq: list[pd.DataFrame] = []
    warmup_need = int(max(st.rolling_window, st.min_train_periods))

    for _, row in selected.iterrows():
        a = str(row["asset_a"])
        b = str(row["asset_b"])
        try:
            tr = train_df[[a, b]].dropna()
            te = test_df[[a, b]].dropna()
        except KeyError:
            continue
        tr = tr[~tr.index.duplicated(keep="first")]
        te = te[~te.index.duplicated(keep="first")]
        if len(tr) < cfg.data.min_observations or te.empty:
            continue

        ya_tr = log_prices(tr.iloc[:, 0])
        yb_tr = log_prices(tr.iloc[:, 1])
        beta = estimate_hedge_ratio(ya_tr, yb_tr)

        warm_len = min(warmup_need, len(tr))
        warm = tr.iloc[-warm_len:]
        chunk = pd.concat([warm, te])
        chunk = chunk[~chunk.index.duplicated(keep="first")]
        ya_c = log_prices(chunk.iloc[:, 0])
        yb_c = log_prices(chunk.iloc[:, 1])

        spread, beta_ser, z = build_spread_and_zscore(
            ya_c,
            yb_c,
            rolling_window=st.rolling_window,
            use_dynamic_beta=st.use_dynamic_beta,
            static_beta=beta if not st.use_dynamic_beta else None,
            min_train_periods=st.min_train_periods,
        )

        pos, exit_reasons = generate_positions_with_reasons(
            z,
            entry_z=st.entry_z,
            exit_z=st.exit_z,
            stop_z=st.stop_z,
            spread=spread,
            min_signal_bars=st.min_signal_bars,
            cooldown_bars_after_stop=st.cooldown_bars_after_stop,
            max_holding_bars=st.max_holding_bars,
            allow_position_flip=st.allow_position_flip,
            min_spread_volatility=st.min_spread_volatility,
            max_spread_volatility=st.max_spread_volatility,
        )

        bt = run_two_leg_backtest(
            chunk.iloc[:, 0],
            chunk.iloc[:, 1],
            pos,
            beta,
            initial_capital=ic,
            transaction_fee=st.transaction_fee,
            slippage=st.slippage,
            leg_capital_fraction=cfg.risk.leg_capital_fraction,
            max_position_size_pct=cfg.risk.max_position_size_pct,
            stop_on_capital_depletion=cfg.risk.stop_on_capital_depletion,
            capital_depletion_threshold=cfg.risk.capital_depletion_threshold,
        )

        eq_test = bt.equity.reindex(te.index)
        ret_test = bt.returns.reindex(te.index)
        rt = build_round_trip_trades(
            chunk.iloc[:, 0],
            chunk.iloc[:, 1],
            bt,
            exit_reasons,
            transaction_fee=st.transaction_fee,
            slippage=st.slippage,
        )

        bench = build_six_benchmark_series(
            te.iloc[:, 0], te.iloc[:, 1], strategy_cfg=st, initial_capital=ic
        )
        pair_label = f"{a}|{b}"
        cmp = build_benchmark_comparison_variant_b(
            pair_label=pair_label,
            strategy_equity=eq_test,
            benchmark_equities={k: v.reindex(te.index) for k, v in bench.items() if v is not None},
            timeframe=cfg.timeframe,
        )

        det = detailed_performance(
            eq_test,
            ret_test,
            bt.positions.reindex(eq_test.index).fillna(0),
            bt.costs.reindex(eq_test.index).fillna(0.0),
            bt.trades,
            timeframe=cfg.timeframe,
            initial_capital=ic,
            round_trips=rt,
            bt=bt,
        )
        summ_rows.append({"pair": pair_label, **det})
        bench_rows.append(cmp)
        all_trades.append(rt.assign(pair=pair_label))
        all_eq.append(build_equity_curve_dataframe(bt).assign(pair=pair_label))

        pfig = fig_dir / _pair_slug(a, b)
        pfig.mkdir(parents=True, exist_ok=True)
        plot_all_standard(
            equity=bt.equity,
            spread=spread.reindex(chunk.index),
            z=z.reindex(chunk.index),
            beta=beta_ser,
            position=pos.reindex(chunk.index),
            benchmark_equity=None,
            corr_roll=ya_c.rolling(st.rolling_window, min_periods=min(st.rolling_window, st.min_train_periods)).corr(
                yb_c
            ),
            figures_dir=pfig,
            round_trips=rt,
        )

        bench_curves = build_six_benchmark_series(
            te.iloc[:, 0], te.iloc[:, 1], strategy_cfg=st, initial_capital=ic
        )
        bench_curves["pair_trading"] = bt.equity.reindex(te.index)
        plot_equity_curve_multi(
            bench_curves,
            pfig / "equity_curve.png",
            title=f"Equity test: {pair_label}",
        )
        plot_spread_zscore_detailed(
            spread.reindex(chunk.index),
            z.reindex(chunk.index),
            pfig / "spread_zscore_detailed.png",
            rolling_window=st.rolling_window,
            entry_z=st.entry_z,
            exit_z=st.exit_z,
            stop_z=st.stop_z,
        )
        plot_benchmark_grid(
            {k: v for k, v in bench_curves.items() if v is not None and not getattr(v, "empty", False)},
            pfig / "benchmark_comparison.png",
            title="Strategy vs benchmarks (normalized)",
        )

    if summ_rows:
        pd.DataFrame(summ_rows).to_csv(out_dir / "backtest_summary_test.csv", index=False)
        pd.concat(bench_rows, ignore_index=True).to_csv(out_dir / "benchmark_comparison_test.csv", index=False)
        pd.concat(all_trades, ignore_index=True).to_csv(out_dir / "trades_test.csv", index=False)
        pd.concat(all_eq, ignore_index=True).to_csv(out_dir / "equity_curve_test.csv", index=False)

    typer.echo(f"Holdout results -> {out_dir}")


if __name__ == "__main__":
    app()
