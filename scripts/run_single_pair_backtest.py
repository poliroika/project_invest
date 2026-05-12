#!/usr/bin/env python3
"""CLI: two-leg backtest for one pair (explicit symbols or from ``selected_pairs.csv``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from pair_trading.backtester import build_equity_curve_dataframe, build_round_trip_trades, run_two_leg_backtest
from pair_trading.benchmarks import build_benchmark_comparison_variant_b, build_six_benchmark_series
from pair_trading.config import effective_initial_capital, load_project_config
from pair_trading.data_loader import load_pair_close
from pair_trading.metrics import detailed_performance, summarize_backtest
from pair_trading.plotting import (
    plot_all_standard,
    plot_benchmark_grid,
    plot_equity_curve_multi,
    plot_spread_zscore_detailed,
)
from pair_trading.preprocessing import filter_timerange_index, log_prices
from pair_trading.signals import generate_positions_with_reasons
from pair_trading.spread import build_spread_and_zscore
from pair_trading.cointegration import estimate_hedge_ratio

app = typer.Typer(help="Run custom two-leg pair backtest.")


def pair_slug(a: str, b: str) -> str:
    return f"{a.replace('/', '_').replace(':', '_')}__{b.replace('/', '_').replace(':', '_')}"


@app.command()
def main(
    asset_a: str | None = typer.Argument(None, help="Leg A (e.g. ETH/USDT:USDT)"),
    asset_b: str | None = typer.Argument(None, help="Leg B (e.g. BTC/USDT:USDT)"),
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
    from_selected: bool = typer.Option(False, "--from-selected", help="Pick pair(s) from results/selected_pairs.csv"),
    top_n: int = typer.Option(1, "--top-n", help="With --from-selected: run top N rows by correlation"),
    respect_selected_period: bool = typer.Option(
        True,
        "--respect-selected-period/--no-respect-selected-period",
        help="With --from-selected: clip backtest to each row's start_date/end_date (PLAN §4).",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_project_config(config)
    st = cfg.strategy
    ic = effective_initial_capital(cfg)

    results_dir = Path(cfg.paths.results_dir)
    figures_dir = Path(cfg.paths.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairs_to_run: list[tuple[str, str, str | None, str | None]] = []
    if from_selected:
        sel_path = results_dir / "selected_pairs.csv"
        if not sel_path.is_file():
            typer.echo(f"Missing {sel_path}. Run screen_pairs first.", err=True)
            raise typer.Exit(code=1)
        sel = pd.read_csv(sel_path)
        if sel.empty or "asset_a" not in sel.columns:
            typer.echo("selected_pairs.csv is empty or malformed.", err=True)
            raise typer.Exit(code=1)
        if "correlation" in sel.columns:
            sel = sel.sort_values("correlation", ascending=False)
        for _, row in sel.head(top_n).iterrows():
            sd = str(row["start_date"]) if respect_selected_period and "start_date" in sel.columns else None
            ed = str(row["end_date"]) if respect_selected_period and "end_date" in sel.columns else None
            pairs_to_run.append((str(row["asset_a"]), str(row["asset_b"]), sd, ed))
    else:
        if not asset_a or not asset_b:
            typer.echo("Provide asset_a asset_b or use --from-selected.", err=True)
            raise typer.Exit(code=1)
        pairs_to_run = [(asset_a, asset_b, None, None)]

    use_multi = from_selected and top_n >= 2
    summary_rows: list[dict[str, object]] = []

    for a_sym, b_sym, sel_start, sel_end in pairs_to_run:
        if use_multi:
            sub = pair_slug(a_sym, b_sym)
            rdir = results_dir / "multi_pair" / sub
            fdir = figures_dir / "multi_pair" / sub
            rdir.mkdir(parents=True, exist_ok=True)
            fdir.mkdir(parents=True, exist_ok=True)
        else:
            rdir = results_dir
            fdir = figures_dir
        row_summary = _run_one_pair(
            a_sym,
            b_sym,
            cfg,
            st,
            ic,
            rdir,
            fdir,
            selected_start=sel_start,
            selected_end=sel_end,
            collect_summary=use_multi,
        )
        if row_summary is not None and use_multi:
            summary_rows.append(row_summary)

    if use_multi and summary_rows:
        pd.DataFrame(summary_rows).to_csv(results_dir / "multi_pair_summary.csv", index=False)
        typer.echo(f"Wrote {results_dir / 'multi_pair_summary.csv'}")


def _run_one_pair(
    asset_a: str,
    asset_b: str,
    cfg: object,
    st: object,
    ic: float,
    results_dir: Path,
    figures_dir: Path,
    *,
    selected_start: str | None = None,
    selected_end: str | None = None,
    collect_summary: bool = False,
) -> dict[str, object] | None:
    tr_start = selected_start or cfg.timerange.start
    tr_end = selected_end or cfg.timerange.end

    pa = load_pair_close(
        cfg.data.ohlcv_root,
        cfg.exchange,
        asset_a,
        cfg.timeframe,
        cfg.trading_mode,
        auto_download=cfg.data.auto_download,
        timerange_start=tr_start,
        timerange_end=tr_end,
    )
    pb = load_pair_close(
        cfg.data.ohlcv_root,
        cfg.exchange,
        asset_b,
        cfg.timeframe,
        cfg.trading_mode,
        auto_download=cfg.data.auto_download,
        timerange_start=tr_start,
        timerange_end=tr_end,
    )

    df = pd.concat([pa, pb], axis=1).dropna()
    df = filter_timerange_index(df, start=tr_start, end=tr_end)
    if df.empty or len(df) < st.min_train_periods:
        typer.echo(f"Insufficient overlapping history for {asset_a} vs {asset_b}.", err=True)
        raise typer.Exit(code=1)

    ya = log_prices(df.iloc[:, 0])
    yb = log_prices(df.iloc[:, 1])
    beta = estimate_hedge_ratio(ya, yb)

    spread, beta_series, z = build_spread_and_zscore(
        ya,
        yb,
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
        df.iloc[:, 0],
        df.iloc[:, 1],
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

    round_trips = build_round_trip_trades(
        df.iloc[:, 0],
        df.iloc[:, 1],
        bt,
        exit_reasons,
        transaction_fee=st.transaction_fee,
        slippage=st.slippage,
    )

    bench_curves = build_six_benchmark_series(
        df.iloc[:, 0], df.iloc[:, 1], strategy_cfg=st, initial_capital=ic
    )

    pair_label = f"{asset_a}|{asset_b}"
    cmp_df = build_benchmark_comparison_variant_b(
        pair_label=pair_label,
        strategy_equity=bt.equity,
        benchmark_equities=bench_curves,
        timeframe=cfg.timeframe,
    )

    summ = summarize_backtest(bt.equity, bt.returns, timeframe=cfg.timeframe)
    detail = detailed_performance(
        bt.equity,
        bt.returns,
        bt.positions,
        bt.costs,
        bt.trades,
        timeframe=cfg.timeframe,
        initial_capital=ic,
        round_trips=round_trips,
        bt=bt,
    )

    summary_row = {"strategy": "pair_trading", "pair": pair_label, **detail}
    pd.DataFrame([summary_row]).to_csv(results_dir / "backtest_summary.csv", index=False)
    cmp_df.to_csv(results_dir / "benchmark_comparison.csv", index=False)

    if not round_trips.empty:
        round_trips.to_csv(results_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "entry_price_a",
                "entry_price_b",
                "exit_price_a",
                "exit_price_b",
                "qty_a",
                "qty_b",
                "gross_pnl",
                "fees",
                "slippage_cost",
                "net_pnl",
                "return_pct",
                "duration_bars",
                "exit_reason",
            ]
        ).to_csv(results_dir / "trades.csv", index=False)

    build_equity_curve_dataframe(bt).to_csv(results_dir / "equity_curve.csv", index=False)

    pd.DataFrame(
        {
            "timestamp": spread.index,
            "spread": spread.astype(float).values,
            "zscore": z.astype(float).values,
        }
    ).to_csv(results_dir / "spread_zscore.csv", index=False)

    roll_mp = min(st.rolling_window, st.min_train_periods)
    corr_roll = ya.rolling(st.rolling_window, min_periods=roll_mp).corr(yb)

    bench_curves["pair_trading"] = bt.equity
    plot_all_standard(
        equity=bt.equity,
        spread=spread.reindex(df.index),
        z=z.reindex(df.index),
        beta=beta_series if st.use_dynamic_beta else beta_series,
        position=pos.reindex(df.index),
        benchmark_equity=None,
        corr_roll=corr_roll.reindex(df.index),
        figures_dir=figures_dir,
        round_trips=round_trips,
    )

    plot_equity_curve_multi(
        bench_curves,
        figures_dir / "equity_curve.png",
        title="Equity: strategy vs benchmarks",
    )

    plot_spread_zscore_detailed(
        spread.reindex(df.index),
        z.reindex(df.index),
        figures_dir / "spread_zscore_detailed.png",
        rolling_window=st.rolling_window,
        entry_z=st.entry_z,
        exit_z=st.exit_z,
        stop_z=st.stop_z,
    )

    plot_benchmark_grid(
        {k: v for k, v in bench_curves.items() if v is not None and not getattr(v, "empty", False)},
        figures_dir / "benchmark_comparison.png",
        title="Strategy vs benchmarks (normalized)",
    )

    typer.echo("--- Backtest summary (pair_trading) ---")
    typer.echo(f"pair: {pair_label}")
    for k in (
        "total_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "number_of_trades",
        "win_rate",
        "fees_paid",
    ):
        if k in detail:
            typer.echo(f"  {k}: {detail[k]}")
    typer.echo(f"Summary -> {results_dir / 'backtest_summary.csv'}")
    typer.echo(f"Figures -> {figures_dir}")

    if not collect_summary:
        return None

    sel_rank = float("nan")
    m = pd.DataFrame()
    sel_path = Path(cfg.paths.results_dir) / "selected_pairs.csv"
    if sel_path.is_file():
        seld = pd.read_csv(sel_path)
        m = seld[(seld["asset_a"] == asset_a) & (seld["asset_b"] == asset_b)]
        if not m.empty and "correlation" in seld.columns:
            seld2 = seld.sort_values("correlation", ascending=False).reset_index(drop=True)
            hit = seld2[(seld2["asset_a"] == asset_a) & (seld2["asset_b"] == asset_b)]
            if not hit.empty:
                sel_rank = float(hit.index[0])

    def _fcol(name: str) -> float:
        if m.empty or name not in m.columns:
            return float("nan")
        return float(m[name].iloc[0])

    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "selected_rank": sel_rank,
        "coint_pvalue": _fcol("coint_pvalue"),
        "adf_pvalue": _fcol("adf_pvalue"),
        "correlation": _fcol("correlation"),
        "half_life": _fcol("half_life"),
        "total_return": detail.get("total_return_capped", detail.get("total_return")),
        "sharpe": detail.get("sharpe"),
        "sortino": detail.get("sortino"),
        "max_drawdown": detail.get("max_drawdown"),
        "win_rate": detail.get("win_rate"),
        "profit_factor": detail.get("profit_factor"),
        "fees_paid": detail.get("fees_paid"),
        "number_of_trades": detail.get("number_of_trades"),
        "capital_depleted": detail.get("capital_depleted", False),
    }


if __name__ == "__main__":
    app()
