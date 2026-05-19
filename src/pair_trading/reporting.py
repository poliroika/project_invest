"""Aggregate tables and embed summaries into reports/report.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return paths[0]


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(v):
        return "n/a"
    return f"{v:.2%}"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _append_selected_pairs_summary(lines: list[str], results_dir: Path) -> None:
    selected = read_optional_csv(results_dir / "selected_pairs.csv")
    if selected is None or selected.empty:
        return
    pairs = ", ".join(f"{r.asset_a} / {r.asset_b}" for r in selected.itertuples())
    lines.append(f"- Selected pairs: {pairs}.")


def _append_multi_pair_summary(lines: list[str], results_dir: Path) -> None:
    multi = read_optional_csv(results_dir / "multi_pair_summary.csv")
    if multi is None or multi.empty:
        return
    rows: list[str] = []
    for r in multi.itertuples():
        label = f"{r.asset_a} / {r.asset_b}"
        rows.append(
            f"{label}: return {_fmt_pct(r.total_return)}, "
            f"Sharpe {_fmt_num(r.sharpe)}, max DD {_fmt_pct(r.max_drawdown)}"
        )
    lines.append(f"- Multi-pair backtest: {'; '.join(rows)}.")


def _append_holdout_summary(lines: list[str], results_dir: Path) -> None:
    holdout = read_optional_csv(results_dir / "holdout" / "backtest_summary_test.csv")
    if holdout is None or holdout.empty or "total_return" not in holdout.columns:
        return
    total = holdout["total_return"].astype(float)
    profitable = float((total > 0).mean())
    best = holdout.loc[total.idxmax()]
    worst = holdout.loc[total.idxmin()]
    lines.append(
        "- Holdout test: "
        f"{len(holdout)} pairs, profitable {_fmt_pct(profitable)}, "
        f"best {best['pair']} ({_fmt_pct(best['total_return'])}), "
        f"worst {worst['pair']} ({_fmt_pct(worst['total_return'])})."
    )


def _append_sensitivity_summary(lines: list[str], results_dir: Path) -> None:
    sens = read_optional_csv(results_dir / "sensitivity" / "sensitivity_summary.csv")
    if sens is None or sens.empty:
        return
    row = sens.iloc[0]
    lines.append(
        "- Sensitivity grid: "
        f"{int(row['n_configs'])} configs, median return {_fmt_pct(row['median_total_return'])}, "
        f"profitable configs {_fmt_pct(row['pct_profitable_configs'])}, "
        f"worst/best {_fmt_pct(row['worst_total_return'])}/{_fmt_pct(row['best_total_return'])}."
    )


def build_metrics_paragraph(results_dir: Path) -> str:
    """Markdown summary of key CSV artifacts (PLAN §16)."""
    lines: list[str] = ["## Автоматически сгенерированная сводка", ""]
    wf_root = results_dir / "walk_forward_screening"
    pairs = [
        ("pair_screening", results_dir / "pair_screening_results.csv"),
        ("selected_pairs", results_dir / "selected_pairs.csv"),
        ("data_availability", results_dir / "data_availability.csv"),
        ("backtest_summary", results_dir / "backtest_summary.csv"),
        ("benchmark_comparison", results_dir / "benchmark_comparison.csv"),
        ("trades", results_dir / "trades.csv"),
        ("equity_curve", results_dir / "equity_curve.csv"),
        ("spread_zscore", results_dir / "spread_zscore.csv"),
        (
            "walk_forward_results",
            _first_existing(wf_root / "walk_forward_results.csv", results_dir / "walk_forward_results.csv"),
        ),
        (
            "walk_forward_selected_pairs",
            _first_existing(wf_root / "walk_forward_selected_pairs.csv", results_dir / "walk_forward_selected_pairs.csv"),
        ),
        (
            "walk_forward_equity",
            _first_existing(wf_root / "walk_forward_equity.csv", results_dir / "walk_forward_equity.csv"),
        ),
        ("multi_pair_summary", results_dir / "multi_pair_summary.csv"),
        ("holdout_backtest_test", results_dir / "holdout" / "backtest_summary_test.csv"),
        ("sensitivity_summary", results_dir / "sensitivity" / "sensitivity_summary.csv"),
    ]
    for label, p in pairs:
        df = read_optional_csv(p)
        if df is None or df.empty:
            lines.append(f"- **{label}**: файл отсутствует или пуст (`{p.as_posix()}`).")
            continue
        lines.append(f"- **{label}**: `{p.as_posix()}` — строк: {len(df)}, столбцов: {len(df.columns)}.")
    lines.append("")
    _append_selected_pairs_summary(lines, results_dir)
    _append_multi_pair_summary(lines, results_dir)
    _append_holdout_summary(lines, results_dir)
    _append_sensitivity_summary(lines, results_dir)
    lines.append("")
    lines.append("Графики: `results/figures/` (`equity_curve.png`, `drawdown.png`, `spread_zscore.png`, и др.).")
    lines.append("")
    return "\n".join(lines)


def append_or_write_report(report_path: Path, paragraph: str, *, marker: str = "<!-- AUTO_REPORT_SUMMARY -->") -> None:
    text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""

    if marker in text:
        before, _, _ = text.partition(marker)
        body = f"{before.rstrip()}\n\n{marker}\n\n{paragraph.strip()}\n"
        report_path.write_text(body, encoding="utf-8")
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text.rstrip() + f"\n\n{marker}\n\n{paragraph.strip()}\n", encoding="utf-8")


def summarize_dict_as_md_table(data: dict[str, Any]) -> str:
    rows = ["| metric | value |", "| --- | --- |"]
    for k, v in data.items():
        rows.append(f"| {k} | {v} |")
    return "\n".join(rows)
