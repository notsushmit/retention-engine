"""Check that labels built as of D from the full sample equal labels built from
the sample's raw transactions cut at D and cleaned again.

Writes the cut copy to a temporary folder and deletes it. Exits with status 1
if any label column differs.

    uv run python scripts/check_as_of.py --as-of 2016-12-31 --last-snapshot 2016-10-01
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from retention.data.clean import clean
from retention.data.sample import resolve_fraction, sample_dir
from retention.labels.config import parse_label_config
from retention.labels.snapshots import build_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2016, 12, 31))
    ap.add_argument("--last-snapshot", type=date.fromisoformat, default=date(2016, 10, 1))
    ap.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    ap.add_argument("--fraction", type=float, default=None)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    fraction = resolve_fraction(args.fraction, config["data"])
    interim = sample_dir(Path(config["data"]["interim_dir"]), fraction)
    processed = sample_dir(cfg.processed_dir, fraction)
    snaps = tuple(t for t in cfg.snapshots if t <= args.last_snapshot)
    cfg = replace(cfg, as_of=args.as_of, snapshots=snaps, allow_partial=True).validate()
    day = int(args.as_of.strftime("%Y%m%d"))
    with tempfile.TemporaryDirectory() as tmp:
        cut_in, cut_out = Path(tmp) / "interim", Path(tmp) / "processed"
        cut_in.mkdir()
        for t in ("transactions", "transactions_v2"):
            duckdb.sql(
                f"copy (select * from '{interim / t}.parquet' where transaction_date <= {day}) "
                f"to '{cut_in / t}.parquet' (format parquet)"
            )
        for t in ("members_v3", "train", "train_v2"):
            shutil.copy(interim / f"{t}.parquet", cut_in / f"{t}.parquet")
        clean(cut_in, cut_out, cfg)
        full = build_labels(pd.read_parquet(processed / "transactions.parquet"), cfg)
        cut = build_labels(pd.read_parquet(cut_out / "transactions.parquet"), cfg)
    key = ["snapshot_date", "msno"]
    full = full.sort_values(key).reset_index(drop=True)
    cut = cut.sort_values(key).reset_index(drop=True)
    same_rows = len(full) == len(cut) and bool((full[key] == cut[key]).all().all())
    diff = {}
    if same_rows:
        for c in full.columns:
            d = ~((full[c] == cut[c]) | (full[c].isna() & cut[c].isna()))
            diff[c] = int(d.sum())
    out = {
        "as_of": str(args.as_of),
        "snapshots": len(snaps),
        "rows": [len(full), len(cut)],
        "same_rows": same_rows,
        "columns_compared": len(diff),
        "columns_with_differences": {c: n for c, n in diff.items() if n},
    }
    print(json.dumps(out, indent=1))
    if not same_rows or any(diff.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
