"""Command line entry: `python -m retention.features {build,docs,diagnostics}`.

build: features.parquet for the sample made by `make data` (all snapshots in
labels.parquet, or --snapshots). docs: regenerate the dictionary in
docs/features.md from the registry. diagnostics: the leak-catching AUC sweeps
(feature_auc.json); nothing is selected or tuned on them.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from retention.data.clean import write_validation
from retention.data.sample import resolve_fraction, sample_dir
from retention.features.build import BuildError, FeaturePaths, build
from retention.features.diagnostics import sweep, sweep_rows
from retention.features.spec import render_docs
from retention.labels.config import parse_label_config

DOCS = Path("docs/features.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.features")
    parser.add_argument("command", choices=["build", "docs", "diagnostics"])
    parser.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--snapshots", nargs="*", type=date.fromisoformat, default=None)
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args(argv)

    if args.command == "docs":
        DOCS.write_text(render_docs(DOCS.read_text()))
        print(f"wrote {DOCS}")
        return

    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    fraction = resolve_fraction(args.fraction, config["data"])
    interim = sample_dir(Path(config["data"]["interim_dir"]), fraction)
    paths = FeaturePaths(
        processed=sample_dir(cfg.processed_dir, fraction),
        logs=(interim / "user_logs.parquet", interim / "user_logs_v2.parquet"),
    )
    if args.command == "diagnostics":
        run_diagnostics(paths)
        return
    try:
        report = build(paths, cfg, args.snapshots, n_chunks=args.chunks)
    except BuildError as err:
        print(f"FAILED: {err}")
        raise SystemExit(1) from err
    if args.snapshots is None:
        write_validation(paths.processed, "features", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_snapshot"}, indent=1))
    print(f"wrote {paths.out}")


def run_diagnostics(paths: FeaturePaths) -> None:
    df, used, held = sweep_rows(paths.out, paths.processed / "labels.parquet")
    table, counts = sweep(df)
    report = {
        "note": "diagnostic only; nothing is selected, dropped or tuned on these numbers",
        "population": "P2 (in_exp30) rows with label_reliable_30, churned_30 known",
        "snapshots_used": used,
        "snapshots_not_read": held,
        "per_snapshot": counts.to_dict(orient="records"),
        "columns": table.to_dict(orient="records"),
    }
    out = paths.processed / "feature_auc.json"
    out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"snapshots used: {used}")
    print(f"not read (held back for Phase 3): {held}")
    print(counts.to_string(index=False))
    show = table.assign(strength=(table.auc - 0.5).abs()).sort_values("strength", ascending=False)
    cols = [
        "column",
        "auc",
        "auc_2015",
        "auc_2016",
        "auc_pooled",
        "auc_range",
        "non_null_share",
        "null_auc",
        "flag_single",
        "flag_null",
        "flag_range",
        "scoring",
    ]
    print(show[cols].round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
