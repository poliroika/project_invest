"""Walk-forward / rolling estimation windows."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd

from pair_trading.backtester import build_round_trip_trades, run_two_leg_backtest
from pair_trading.cointegration import (
    calculate_spread,
    estimate_hedge_ratio,
    run_adf_test,
    run_cointegration_test,
    screen_pairs,
)
from pair_trading.config import effective_initial_capital
from pair_trading.metrics import (
    detailed_performance,
    profit_factor_from_trades_df,
    summarize_backtest,
    win_rate_from_trades_df,
)
from pair_trading.preprocessing import log_prices
from pair_trading.signals import generate_positions, generate_positions_with_reasons
from pair_trading.spread import build_spread_and_zscore


def walk_forward_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    train_len: int,
    test_len: int,
    step: int,
    strategy_cfg: Any,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Rolling train (estimate hedge ratio) + out-of-sample test segment.

    For each fold, beta is estimated on the train window only. Signals and PnL are
    computed on ``warmup + test`` where ``warmup`` is the tail of the train window
    long enough for rolling z-score stabilization; metrics are taken **only** on
    the test index (no lookahead beyond the train/test boundary for trading).
    """
    pa = price_a.astype(float).rename("a")
    pb = price_b.astype(float).rename("b")
    df = pd.concat([pa, pb], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    st = strategy_cfg
    warmup_need = int(max(st.rolling_window, st.min_train_periods))

    rows: list[dict[str, Any]] = []
    n = len(df)
    i = train_len
    while i + test_len <= n:
        train = df.iloc[i - train_len : i]
        test = df.iloc[i : i + test_len]

        ya_tr = log_prices(train.iloc[:, 0])
        xb_tr = log_prices(train.iloc[:, 1])
        beta = estimate_hedge_ratio(ya_tr, xb_tr)

        warm_len = min(warmup_need, len(train))
        warm = train.iloc[-warm_len:]
        chunk = pd.concat([warm, test])
        ya_c = log_prices(chunk.iloc[:, 0])
        yb_c = log_prices(chunk.iloc[:, 1])

        spread, _, z = build_spread_and_zscore(
            ya_c,
            yb_c,
            rolling_window=st.rolling_window,
            use_dynamic_beta=st.use_dynamic_beta,
            static_beta=beta if not st.use_dynamic_beta else None,
            min_train_periods=st.min_train_periods,
        )

        pos = generate_positions(
            z,
            entry_z=st.entry_z,
            exit_z=st.exit_z,
            stop_z=st.stop_z,
        )

        bt = run_two_leg_backtest(
            chunk.iloc[:, 0],
            chunk.iloc[:, 1],
            pos,
            beta,
            transaction_fee=st.transaction_fee,
            slippage=st.slippage,
            leg_capital_fraction=cfg.risk.leg_capital_fraction,
            max_position_size_pct=cfg.risk.max_position_size_pct,
            stop_on_capital_depletion=cfg.risk.stop_on_capital_depletion,
            capital_depletion_threshold=cfg.risk.capital_depletion_threshold,
        )

        eq_all = bt.equity.reindex(chunk.index)
        ret_all = bt.returns.reindex(chunk.index)

        eq_test = eq_all.loc[test.index]
        ret_test = ret_all.loc[test.index]

        summ = summarize_backtest(eq_test, ret_test, timeframe=timeframe)

        rows.append(
            {
                "train_start": train.index[0].isoformat(),
                "train_end": train.index[-1].isoformat(),
                "test_start": test.index[0].isoformat(),
                "test_end": test.index[-1].isoformat(),
                "hedge_ratio_train": beta,
                "test_sharpe": summ["sharpe"],
                "test_max_drawdown": summ["max_drawdown"],
                "test_cumulative_return": summ["cumulative_return"],
                "test_final_equity": summ["final_equity"],
            }
        )

        i += step

    return pd.DataFrame(rows)


def iter_calendar_folds(
    index: pd.DatetimeIndex,
    *,
    train_months: int,
    test_months: int,
    step_months: int,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield ``(train_index, test_index)`` slices following PLAN §11 (calendar windows)."""
    idx = pd.DatetimeIndex(sorted(index.unique())).sort_values()
    if len(idx) == 0:
        return
    end_limit = idx.max()
    cursor = idx.min()
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end_limit + pd.DateOffset(days=1):
            break
        tr = idx[(idx >= cursor) & (idx < train_end)]
        te = idx[(idx >= test_start) & (idx < test_end)]
        if len(tr) > 0 and len(te) > 0:
            yield tr, te
        cursor = cursor + pd.DateOffset(months=step_months)


def walk_forward_calendar_single_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    cfg: Any,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Walk-forward using ``walk_forward.*_months`` from config (single traded pair)."""
    wf = getattr(cfg, "walk_forward")
    st = getattr(cfg, "strategy")

    df = pd.concat([price_a.astype(float), price_b.astype(float)], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    warmup_need = int(max(st.rolling_window, st.min_train_periods))

    for tr_idx, te_idx in iter_calendar_folds(
        pd.DatetimeIndex(df.index),
        train_months=wf.train_months,
        test_months=wf.test_months,
        step_months=wf.step_months,
    ):
        train = df.loc[tr_idx]
        test = df.loc[te_idx]
        if len(train) < getattr(cfg.data, "min_observations", 500):
            continue

        ya_tr = log_prices(train.iloc[:, 0])
        xb_tr = log_prices(train.iloc[:, 1])
        beta = estimate_hedge_ratio(ya_tr, xb_tr)
        coint_tr = run_cointegration_test(ya_tr, xb_tr)
        sp_tr = calculate_spread(ya_tr, xb_tr, beta)
        adf_tr = run_adf_test(sp_tr)

        warm_len = min(warmup_need, len(train))
        warm = train.iloc[-warm_len:]
        chunk = pd.concat([warm, test])
        ya_c = log_prices(chunk.iloc[:, 0])
        yb_c = log_prices(chunk.iloc[:, 1])

        spread, _, z = build_spread_and_zscore(
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
        )

        ic = effective_initial_capital(cfg)
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

        eq_all = bt.equity.reindex(chunk.index)
        ret_all = bt.returns.reindex(chunk.index)

        eq_test = eq_all.loc[test.index]
        ret_test = ret_all.loc[test.index]
        summ = summarize_backtest(eq_test, ret_test, timeframe=timeframe)

        rt = build_round_trip_trades(
            chunk.iloc[:, 0],
            chunk.iloc[:, 1],
            bt,
            exit_reasons,
            transaction_fee=st.transaction_fee,
            slippage=st.slippage,
        )
        if not rt.empty and "exit_time" in rt.columns:
            ex = pd.to_datetime(rt["exit_time"], utc=True)
            te_idx = pd.DatetimeIndex(test.index)
            rt_test = rt[ex.isin(te_idx)]
            n_tr = int(len(rt_test))
            wr = win_rate_from_trades_df(rt_test)
        else:
            n_tr = 0
            wr = float("nan")

        rows.append(
            {
                "fold_id": len(rows),
                "train_start": train.index[0].isoformat(),
                "train_end": train.index[-1].isoformat(),
                "test_start": test.index[0].isoformat(),
                "test_end": test.index[-1].isoformat(),
                "asset_a": "",
                "asset_b": "",
                "hedge_ratio_train": beta,
                "coint_pvalue_train": float(coint_tr["pvalue"]),
                "adf_pvalue_train": float(adf_tr["pvalue"]),
                "test_total_return": summ["total_return"],
                "test_sharpe": summ["sharpe"],
                "test_max_drawdown": summ["max_drawdown"],
                "test_cumulative_return": summ["cumulative_return"],
                "test_final_equity": summ["final_equity"],
                "test_trades": n_tr,
                "test_win_rate": wr,
            }
        )

    return pd.DataFrame(rows)


def walk_forward_screen_pairs_calendar(closes: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ``screen_pairs`` on each training window only (no lookahead).

    Returns ``(all_fold_results, selected_pairs_per_fold)``.
    """
    wf = getattr(cfg, "walk_forward")
    data_cfg = getattr(cfg, "data")

    folds_all: list[pd.DataFrame] = []
    folds_sel: list[pd.DataFrame] = []

    for tr_idx, te_idx in iter_calendar_folds(
        pd.DatetimeIndex(closes.index),
        train_months=wf.train_months,
        test_months=wf.test_months,
        step_months=wf.step_months,
    ):
        train_df = closes.loc[tr_idx].dropna(how="any")
        if len(train_df) < data_cfg.min_observations:
            continue
        out = screen_pairs(train_df, cfg)
        out.insert(0, "fold_test_end", str(te_idx.max()))
        out.insert(0, "fold_test_start", str(te_idx.min()))
        out.insert(0, "fold_train_end", str(tr_idx.max()))
        out.insert(0, "fold_train_start", str(tr_idx.min()))
        folds_all.append(out)

        sel = out[out["selected"]].sort_values("correlation", ascending=False)
        sel = sel.head(wf.max_pairs_per_window)
        if len(sel) >= wf.min_pairs_per_window:
            folds_sel.append(sel)

    all_df = pd.concat(folds_all, ignore_index=True) if folds_all else pd.DataFrame()
    sel_df = pd.concat(folds_sel, ignore_index=True) if folds_sel else pd.DataFrame()
    return all_df, sel_df


def walk_forward_portfolio_screening(
    closes: pd.DataFrame,
    cfg: Any,
    *,
    top_k: int | None = None,
    timeframe: str = "1h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mode B (PLAN §11.2): re-screen pairs on each train window, backtest top-``k`` on test.

    Returns ``(walk_forward_results, walk_forward_selected_pairs, walk_forward_equity)``.
    """
    wf = getattr(cfg, "walk_forward")
    st = getattr(cfg, "strategy")
    data_cfg = getattr(cfg, "data")
    k = int(top_k) if top_k is not None else int(wf.max_pairs_per_window)
    ic = effective_initial_capital(cfg)
    warmup_need = int(max(st.rolling_window, st.min_train_periods))

    rows: list[dict[str, Any]] = []
    sel_parts: list[pd.DataFrame] = []
    eq_parts: list[pd.DataFrame] = []

    for fold_id, (tr_idx, te_idx) in enumerate(
        iter_calendar_folds(
            pd.DatetimeIndex(closes.index),
            train_months=wf.train_months,
            test_months=wf.test_months,
            step_months=wf.step_months,
        )
    ):
        train_df = closes.loc[tr_idx].dropna(how="any")
        if len(train_df) < data_cfg.min_observations:
            continue
        out = screen_pairs(train_df, cfg)
        sel = out[out["selected"]].sort_values("correlation", ascending=False).head(k)
        if len(sel) < wf.min_pairs_per_window:
            rows.append(
                {
                    "fold_id": fold_id,
                    "asset_a": "",
                    "asset_b": "",
                    "train_start": train_df.index[0].isoformat(),
                    "train_end": train_df.index[-1].isoformat(),
                    "test_start": te_idx.min().isoformat(),
                    "test_end": te_idx.max().isoformat(),
                    "test_total_return": float("nan"),
                    "test_sharpe": float("nan"),
                    "test_sortino": float("nan"),
                    "test_max_drawdown": float("nan"),
                    "test_trades": 0,
                    "test_win_rate": float("nan"),
                    "test_profit_factor": float("nan"),
                    "fees_paid": float("nan"),
                    "capital_depleted": False,
                    "fold_status": "no_trade_due_to_no_selected_pairs",
                }
            )
            continue

        sel_fold = sel.reset_index(drop=True).copy()
        sel_fold.insert(0, "fold_id", fold_id)
        sel_fold.insert(1, "train_start", train_df.index[0].isoformat())
        sel_fold.insert(2, "train_end", train_df.index[-1].isoformat())
        sel_fold.insert(3, "test_start", te_idx.min().isoformat())
        sel_fold.insert(4, "test_end", te_idx.max().isoformat())
        sel_fold.insert(5, "selected_rank", np.arange(1, len(sel_fold) + 1, dtype=int))
        sel_parts.append(sel_fold)

        for _, srow in sel.iterrows():
            a = str(srow["asset_a"])
            b = str(srow["asset_b"])
            try:
                train = train_df[[a, b]].dropna()
                test = closes.loc[te_idx, [a, b]].dropna()
            except KeyError:
                continue
            if len(train) < data_cfg.min_observations or test.empty:
                continue

            ya_tr = log_prices(train.iloc[:, 0])
            xb_tr = log_prices(train.iloc[:, 1])
            beta = estimate_hedge_ratio(ya_tr, xb_tr)

            warm_len = min(warmup_need, len(train))
            warm = train.iloc[-warm_len:]
            chunk = pd.concat([warm, test])
            ya_c = log_prices(chunk.iloc[:, 0])
            yb_c = log_prices(chunk.iloc[:, 1])

            spread_c, _, z = build_spread_and_zscore(
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
                spread=spread_c,
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
            eq_all = bt.equity.reindex(chunk.index)
            ret_all = bt.returns.reindex(chunk.index)
            eq_test = eq_all.loc[test.index]
            ret_test = ret_all.loc[test.index]

            rt = build_round_trip_trades(
                chunk.iloc[:, 0],
                chunk.iloc[:, 1],
                bt,
                exit_reasons,
                transaction_fee=st.transaction_fee,
                slippage=st.slippage,
            )
            rt_test = pd.DataFrame()
            if not rt.empty and "exit_time" in rt.columns:
                ex = pd.to_datetime(rt["exit_time"], utc=True)
                te_ix = pd.DatetimeIndex(test.index)
                rt_test = rt[ex.isin(te_ix)]
                n_tr = int(len(rt_test))
                wr = win_rate_from_trades_df(rt_test)
            else:
                n_tr = 0
                wr = float("nan")

            summ = summarize_backtest(eq_test, ret_test, timeframe=timeframe)
            det = detailed_performance(
                eq_test,
                ret_test,
                bt.positions.reindex(eq_test.index).fillna(0),
                bt.costs.reindex(eq_test.index).fillna(0.0),
                bt.trades,
                timeframe=timeframe,
                initial_capital=ic,
                round_trips=rt_test if not rt_test.empty else None,
                bt=bt,
            )
            pf_test = profit_factor_from_trades_df(rt_test) if not rt_test.empty else float("nan")
            fees_test = float(bt.costs.reindex(test.index).fillna(0.0).sum())

            rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train.index[0].isoformat(),
                    "train_end": train.index[-1].isoformat(),
                    "test_start": test.index[0].isoformat(),
                    "test_end": test.index[-1].isoformat(),
                    "asset_a": a,
                    "asset_b": b,
                    "hedge_ratio_train": float(srow["hedge_ratio"]),
                    "coint_pvalue_train": float(srow["coint_pvalue"]),
                    "adf_pvalue_train": float(srow["adf_pvalue"]),
                    "test_total_return": float(det.get("total_return_capped", summ["total_return"])),
                    "test_sharpe": float(det.get("sharpe", float("nan"))),
                    "test_sortino": float(det.get("sortino_until_depletion", det.get("sortino", float("nan")))),
                    "test_max_drawdown": float(det.get("max_drawdown", summ["max_drawdown"])),
                    "test_trades": n_tr,
                    "test_win_rate": wr,
                    "test_profit_factor": pf_test,
                    "fees_paid": fees_test,
                    "capital_depleted": bool(bt.capital_depleted),
                    "fold_status": "ok",
                }
            )
            eq_parts.append(
                pd.DataFrame(
                    {
                        "fold_id": fold_id,
                        "asset_a": a,
                        "asset_b": b,
                        "timestamp": eq_test.index,
                        "equity": eq_test.astype(float).values,
                    }
                )
            )

    res = pd.DataFrame(rows)
    sels = pd.concat(sel_parts, ignore_index=True) if sel_parts else pd.DataFrame()
    eq_df = pd.concat(eq_parts, ignore_index=True) if eq_parts else pd.DataFrame()
    return res, sels, eq_df

