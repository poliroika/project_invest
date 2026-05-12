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
