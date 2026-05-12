#!/usr/bin/env python3
"""CLI: refresh reports/report.md summary section."""

from __future__ import annotations

from pathlib import Path

import typer

from pair_trading.config import load_project_config
from pair_trading.reporting import append_or_write_report, build_metrics_paragraph

app = typer.Typer(help="Embed results summaries into reports/report.md")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
    report: Path = typer.Option(Path("reports/report.md"), "--report"),
) -> None:
    cfg = load_project_config(config)
    paragraph = build_metrics_paragraph(Path(cfg.paths.results_dir))
    append_or_write_report(report, paragraph)
    typer.echo(f"Updated {report}")


if __name__ == "__main__":
    app()
