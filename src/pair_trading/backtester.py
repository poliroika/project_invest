"""Two-leg synchronous backtester (long/short per leg)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    mark_to_market_pnl: pd.Series
    costs: pd.Series
    positions: pd.Series
    quantities_a: pd.Series
    quantities_b: pd.Series
    trades: pd.DataFrame
    capital_depleted: bool = False
    capital_depletion_time: pd.Timestamp | None = None
    min_equity: float = float("nan")
    stopped_early: bool = False
    max_drawdown_raw: float = float("nan")
    final_equity_raw: float = float("nan")
    equity_accounting: pd.Series | None = None


def _leg_notional_capital(
    initial_capital: float,
    *,
    leg_capital_fraction: float,
    max_position_size_pct: float | None,
) -> float:
    """Dollar allocation per leg (``K`` in :func:`run_two_leg_backtest`)."""
    ic = float(initial_capital)
    leg = float(leg_capital_fraction)
    if max_position_size_pct is None:
        return ic * leg
    cap = min(leg, float(max_position_size_pct))
    return ic * cap


def run_two_leg_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    position: pd.Series,
    hedge_ratio: float,
    *,
    initial_capital: float = 10_000.0,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0002,
    leg_capital_fraction: float = 0.5,
    max_position_size_pct: float | None = None,
    stop_on_capital_depletion: bool = False,
    capital_depletion_threshold: float = 0.0,
) -> BacktestResult:
    """
    Dollar-neutral style legs: each side allocates ``initial_capital * leg_capital_fraction``.

    Long spread (position=1): long A, short B with sizes ``K/Pa`` and ``-beta*K/Pb``.
    Short spread (position=-1): opposite. Costs apply to dollar turnover when the
    discrete position changes.

    When ``stop_on_capital_depletion`` is True and equity falls to
    ``capital_depletion_threshold`` or below, open positions are closed, trading stops,
    and equity is floored at zero from that bar onward (see PLAN §3).
    """
    pa = price_a.astype(float).reindex(position.index)
    pb = price_b.astype(float).reindex(position.index)
    pos_in = position.astype(np.int8).reindex(position.index).fillna(0).astype(np.int8)

    cost_rate = float(transaction_fee + slippage)
    K = _leg_notional_capital(
        initial_capital,
        leg_capital_fraction=leg_capital_fraction,
        max_position_size_pct=max_position_size_pct,
    )
    beta = float(hedge_ratio)

    index = pos_in.index
    n = len(index)
    qa = pd.Series(0.0, index=index, dtype=float)
    qb = pd.Series(0.0, index=index, dtype=float)
    mtm = pd.Series(0.0, index=index, dtype=float)
    costs = pd.Series(0.0, index=index, dtype=float)
    pos_out = pd.Series(np.int8(0), index=index, dtype=np.int8)

    trade_rows: list[dict[str, Any]] = []

    if n == 0:
        empty = pd.Series(dtype=float)
        return BacktestResult(empty, empty, empty, empty, pos_out, qa, qb, pd.DataFrame())

    capital_depleted = False
    depletion_time: pd.Timestamp | None = None
    thresh = float(capital_depletion_threshold)
    cum_equity = float(initial_capital)
    min_eq = cum_equity

    def _apply_close_at_bar(
        i: int,
        pr_a: float,
        pr_b: float,
        qa_c: float,
        qb_c: float,
        *,
        reason: str,
    ) -> tuple[float, float, float]:
        """Return (new_qa, new_qb, close_cost)."""
        nonlocal cum_equity, trade_rows
        if qa_c == 0.0 and qb_c == 0.0:
            return 0.0, 0.0, 0.0
        da_old = qa_c * pr_a
        db_old = qb_c * pr_b
        turnover = abs(0.0 - da_old) + abs(0.0 - db_old)
        c_close = cost_rate * turnover
        costs.iloc[i] = float(costs.iloc[i]) + c_close
        cum_equity -= c_close
        trade_rows.append(
            {
                "timestamp": index[i],
                "side": reason,
                "position_prev": int(np.sign(qa_c)) if qa_c != 0 else 0,
                "position": 0,
                "price_a": pr_a,
                "price_b": pr_b,
                "qty_a": 0.0,
                "qty_b": 0.0,
                "cost": c_close,
            }
        )
        return 0.0, 0.0, c_close

    def _check_depletion(i: int, pr_a: float, pr_b: float) -> None:
        nonlocal capital_depleted, depletion_time, cum_equity, min_eq
        if not stop_on_capital_depletion or capital_depleted:
            return
        if cum_equity > thresh:
            return
        capital_depleted = True
        depletion_time = index[i]
        qa.iloc[i], qb.iloc[i], _ = _apply_close_at_bar(
            i, pr_a, pr_b, float(qa.iloc[i]), float(qb.iloc[i]), reason="capital_depletion_close"
        )
        min_eq = min(min_eq, cum_equity)
        cum_equity = max(cum_equity, 0.0)
        pos_out.iloc[i] = 0

    # Bar 0
    sig0 = int(pos_in.iloc[0]) if not capital_depleted else 0
    pos_out.iloc[0] = 0 if capital_depleted else sig0
    if sig0 != 0:
        pa0 = float(pa.iloc[0])
        pb0 = float(pb.iloc[0])
        qa.iloc[0] = sig0 * (K / pa0)
        qb.iloc[0] = -sig0 * beta * (K / pb0)
        notional = abs(qa.iloc[0] * pa0) + abs(qb.iloc[0] * pb0)
        c0 = cost_rate * notional
        costs.iloc[0] = c0
        cum_equity -= c0
        min_eq = min(min_eq, cum_equity)
        trade_rows.append(
            {
                "timestamp": index[0],
                "side": "open",
                "position": sig0,
                "price_a": pa0,
                "price_b": pb0,
                "qty_a": qa.iloc[0],
                "qty_b": qb.iloc[0],
                "cost": c0,
            }
        )
        _check_depletion(0, pa0, pb0)
        if capital_depleted:
            pos_out.iloc[0] = 0

    for i in range(1, n):
        pr_a = float(pa.iloc[i])
        pr_b = float(pb.iloc[i])
        pr_a_prev = float(pa.iloc[i - 1])
        pr_b_prev = float(pb.iloc[i - 1])

        if capital_depleted:
            qa.iloc[i] = 0.0
            qb.iloc[i] = 0.0
            mtm.iloc[i] = 0.0
            costs.iloc[i] = 0.0
            pos_out.iloc[i] = 0
            continue

        qa_prev = float(qa.iloc[i - 1])
        qb_prev = float(qb.iloc[i - 1])
        mtm.iloc[i] = qa_prev * (pr_a - pr_a_prev) + qb_prev * (pr_b - pr_b_prev)
        cum_equity += float(mtm.iloc[i])
        min_eq = min(min_eq, cum_equity)

        sig = int(pos_in.iloc[i])
        sig_prev = int(pos_out.iloc[i - 1])

        qa_i = qa_prev
        qb_i = qb_prev
        c_i = 0.0

        if sig != sig_prev:
            da_old = qa_prev * pr_a
            db_old = qb_prev * pr_b
            if sig == 0:
                qa_i = 0.0
                qb_i = 0.0
            else:
                qa_i = sig * (K / pr_a)
                qb_i = -sig * beta * (K / pr_b)
            turnover = abs(qa_i * pr_a - da_old) + abs(qb_i * pr_b - db_old)
            c_i = cost_rate * turnover
            costs.iloc[i] = c_i
            cum_equity -= c_i
            min_eq = min(min_eq, cum_equity)
            trade_rows.append(
                {
                    "timestamp": index[i],
                    "side": "rebalance",
                    "position_prev": sig_prev,
                    "position": sig,
                    "price_a": pr_a,
                    "price_b": pr_b,
                    "qty_a": qa_i,
                    "qty_b": qb_i,
                    "cost": c_i,
                }
            )

        qa.iloc[i] = qa_i
        qb.iloc[i] = qb_i
        pos_out.iloc[i] = sig
        _check_depletion(i, pr_a, pr_b)
        if capital_depleted:
            pos_out.iloc[i] = 0

        if capital_depleted:
            for j in range(i + 1, n):
                pos_out.iloc[j] = 0
                qa.iloc[j] = 0.0
                qb.iloc[j] = 0.0
                mtm.iloc[j] = 0.0
                costs.iloc[j] = 0.0
            break

    equity_s = (float(initial_capital) + (mtm - costs).cumsum()).rename("equity")
    equity_accounting_s = equity_s.copy()
    final_equity_raw_val = float(equity_s.iloc[-1]) if len(equity_s) else float("nan")
    eq_pre = equity_s.astype(float)
    peak = eq_pre.cummax()
    dd = (eq_pre / peak) - 1.0
    max_dd_raw = float(dd.min()) if len(eq_pre) else float("nan")

    depletion_di: int | None = None
    if capital_depleted and depletion_time is not None:
        loc = index.get_loc(depletion_time)
        depletion_di = int(loc) if isinstance(loc, (int, np.integer)) else int(loc.start)
        floor = max(float(equity_s.iloc[depletion_di]), 0.0)
        eqv = equity_s.astype(float).values.copy()
        eqv[depletion_di:] = floor
        equity_s = pd.Series(eqv, index=index, name="equity")

    ret = equity_s.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("return")
    if depletion_di is not None:
        ret.iloc[depletion_di + 1 :] = 0.0

    trades_df = pd.DataFrame(trade_rows)
    return BacktestResult(
        equity=equity_s,
        returns=ret,
        mark_to_market_pnl=mtm.rename("mtm"),
        costs=costs.rename("cost"),
        positions=pos_out.rename("position"),
        quantities_a=qa.rename("qty_a"),
        quantities_b=qb.rename("qty_b"),
        trades=trades_df,
        capital_depleted=capital_depleted,
        capital_depletion_time=depletion_time,
        min_equity=float(min_eq),
        stopped_early=capital_depleted,
        max_drawdown_raw=max_dd_raw,
        final_equity_raw=final_equity_raw_val,
        equity_accounting=equity_accounting_s,
    )


def build_equity_curve_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Hourly/bars equity table for ``results/equity_curve.csv`` (PLAN §7.2)."""
    idx = result.equity.index
    out = pd.DataFrame(
        {
            "timestamp": idx,
            "equity": result.equity.astype(float).values,
            "return": result.returns.astype(float).values,
            "position": result.positions.reindex(idx).fillna(0).astype(int).values,
            "qty_a": result.quantities_a.reindex(idx).fillna(0.0).astype(float).values,
            "qty_b": result.quantities_b.reindex(idx).fillna(0.0).astype(float).values,
            "mtm_pnl": result.mark_to_market_pnl.reindex(idx).fillna(0.0).astype(float).values,
            "costs": result.costs.reindex(idx).fillna(0.0).astype(float).values,
        }
    )
    return out


def build_round_trip_trades(
    price_a: pd.Series,
    price_b: pd.Series,
    result: BacktestResult,
    exit_reasons: pd.Series | None = None,
    *,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0002,
) -> pd.DataFrame:
    """
    Aggregate rebalance events into round-trip trades (PLAN §7.1).

    ``exit_reasons`` should be aligned to ``result.positions`` (same index); when
    absent, reasons default to ``mean_reversion_exit`` / ``end_of_backtest_exit``.
    """
    pa = price_a.astype(float).reindex(result.positions.index)
    pb = price_b.astype(float).reindex(result.positions.index)
    pos = result.positions.reindex(result.positions.index).fillna(0).astype(np.int8)
    qa = result.quantities_a.reindex(result.positions.index).fillna(0.0)
    qb = result.quantities_b.reindex(result.positions.index).fillna(0.0)
    mtm = result.mark_to_market_pnl.reindex(result.positions.index).fillna(0.0)
    costs = result.costs.reindex(result.positions.index).fillna(0.0)
    reasons = (
        exit_reasons.reindex(result.positions.index).fillna("")
        if exit_reasons is not None
        else pd.Series("", index=result.positions.index, dtype=object)
    )

    fee_rate = float(transaction_fee)
    slip_rate = float(slippage)
    denom = fee_rate + slip_rate if (fee_rate + slip_rate) > 0 else 1.0

    rows: list[dict[str, Any]] = []
    n = len(pos)
    if n == 0:
        return pd.DataFrame()

    in_trade = False
    entry_i: int | None = None
    direction = 0
    trade_id = 0

    def _close_trade(exit_i: int, reason: str) -> None:
        nonlocal trade_id, in_trade, entry_i, direction
        assert entry_i is not None
        ei = int(entry_i)
        xi = int(exit_i)
        gross = float(mtm.iloc[ei + 1 : xi + 1].sum()) if ei + 1 <= xi else 0.0
        fee_total = float(costs.iloc[ei : xi + 1].sum())
        slip_part = fee_total * (slip_rate / denom)
        fee_part = fee_total - slip_part
        net = gross - fee_total
        pa_e = float(pa.iloc[ei])
        pb_e = float(pb.iloc[ei])
        pa_x = float(pa.iloc[xi])
        pb_x = float(pb.iloc[xi])
        qa_e = float(qa.iloc[ei])
        qb_e = float(qb.iloc[ei])
        notional = abs(qa_e * pa_e) + abs(qb_e * pb_e)
        ret_pct = float(net / notional) if notional > 0 else float("nan")
        dir_label = "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD"
        trade_id += 1
        rows.append(
            {
                "trade_id": trade_id,
                "entry_time": pos.index[ei],
                "exit_time": pos.index[xi],
                "direction": dir_label,
                "entry_price_a": pa_e,
                "entry_price_b": pb_e,
                "exit_price_a": pa_x,
                "exit_price_b": pb_x,
                "qty_a": qa_e,
                "qty_b": qb_e,
                "gross_pnl": gross,
                "fees": fee_part,
                "slippage_cost": slip_part,
                "net_pnl": net,
                "return_pct": ret_pct,
                "duration_bars": int(xi - ei),
                "exit_reason": reason,
            }
        )
        in_trade = False
        entry_i = None
        direction = 0

    for i in range(n):
        p = int(pos.iloc[i])
        p_prev = int(pos.iloc[i - 1]) if i > 0 else 0

        if not in_trade and p != 0:
            in_trade = True
            entry_i = i
            direction = p
        elif in_trade and p == 0 and p_prev != 0:
            r = str(reasons.iloc[i]) if i < len(reasons) else ""
            cap_t = getattr(result, "capital_depletion_time", None)
            if getattr(result, "capital_depleted", False) and cap_t is not None:
                if pd.Timestamp(pos.index[i]) == pd.Timestamp(cap_t):
                    r = "capital_depletion_exit"
            if not r:
                r = "mean_reversion_exit"
            if r == "capital_depletion_close" or "capital_depletion" in r:
                r = "capital_depletion_exit"
            _close_trade(i, r)

    if in_trade and entry_i is not None:
        _close_trade(n - 1, "end_of_backtest_exit")

    return pd.DataFrame(rows)
