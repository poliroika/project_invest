#!/usr/bin/env python3
"""Check OHLCV availability for pairs listed in config (PLAN §5.3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from pair_trading.config import load_pairs_list, load_project_config
from pair_trading.data_loader import discover_ohlcv_path, load_pair_close

app = typer.Typer(help="Verify OHLCV files and write results/data_availability.csv.")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/project_config.yaml"), "--config"),
) -> None:
    cfg = load_project_config(config)
    pairs = load_pairs_list(cfg.paths.pairs_file)
    rows: list[dict[str, object]] = []

    for pair in pairs:
        path = discover_ohlcv_path(
            cfg.data.ohlcv_root,
            cfg.exchange,
            pair,
            cfg.timeframe,
            cfg.trading_mode,
        )
        if path is None:
            rows.append(
                {
                    "pair": pair,
                    "file_found": False,
                    "path": "",
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    "missing_values": "",
                }
            )
            continue
        try:
            s = load_pair_close(
                cfg.data.ohlcv_root,
                cfg.exchange,
                pair,
                cfg.timeframe,
                cfg.trading_mode,
                auto_download=False,
                timerange_start=cfg.timerange.start,
                timerange_end=cfg.timerange.end,
            )
            nan_ct = int(s.isna().sum())
            rows.append(
                {
                    "pair": pair,
                    "file_found": True,
                    "path": str(path),
                    "rows": int(len(s)),
                    "start_date": s.index.min().isoformat() if len(s) else "",
                    "end_date": s.index.max().isoformat() if len(s) else "",
                    "missing_values": nan_ct,
                }
            )
        except Exception as e:  # noqa: BLE001 — diagnostic script
            rows.append(
                {
                    "pair": pair,
                    "file_found": True,
                    "path": str(path),
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    "missing_values": f"read_error: {e}",
                }
            )

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_availability.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    typer.echo(f"Wrote {out_path} ({len(rows)} pairs).")


if __name__ == "__main__":
    app()
