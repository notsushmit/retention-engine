"""Check the fitted_to_train rules on a customer slice disjoint from the sample.

Reads the raw transactions and label files for md5 buckets [lo, hi), writes
them to a temporary folder, runs the clean and compare stages, prints the
comparison and the rule tables as JSON, and deletes the folder. Nothing is
refitted. Pass criteria are the ones fixed before the first run.

    uv run python scripts/check_holdout.py --buckets 20 40
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from retention.data.clean import clean
from retention.data.sample import bucket_sql, resolve_fraction, sample_dir, to_percent
from retention.labels.compare import compare
from retention.labels.config import parse_label_config

FILES = {
    "transactions": "transactions.csv",
    "transactions_v2": "transactions_v2.csv",
    "members_v3": "members_v3.csv",  # read only because clean needs it
    "train": "train.csv",
    "train_v2": "train_v2.csv",  # read only because clean needs it
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", nargs=2, type=int, default=[20, 40], metavar=("LO", "HI"))
    ap.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    ap.add_argument("--kappa-coverage", type=float, default=0.979)
    ap.add_argument("--kappa-paid", type=float, default=0.994)
    ap.add_argument("--kappa-tolerance", type=float, default=0.01)
    ap.add_argument("--max-other-pct", type=float, default=20.0)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    lo, hi = args.buckets
    sample_pct = to_percent(resolve_fraction(None, config["data"]))
    if lo < sample_pct:
        raise SystemExit(f"buckets [{lo}, {hi}) overlap the sample (buckets below {sample_pct})")
    raw = Path(config["data"]["raw_dir"]) / "csv"
    sample = sample_dir(Path(config["data"]["interim_dir"]), sample_pct / 100)
    con = duckdb.connect()
    con.sql("set enable_progress_bar = false")
    with tempfile.TemporaryDirectory() as tmp:
        interim, processed = Path(tmp) / "interim", Path(tmp) / "processed"
        interim.mkdir()
        where = f"{bucket_sql()} >= {lo} and {bucket_sql()} < {hi}"
        for table, name in FILES.items():
            con.execute(
                f"copy (select * from read_csv(?, header = true, types = {{'msno': 'VARCHAR'}}) "
                f"where {where}) to '{interim / table}.parquet' (format parquet)",
                [str(raw / name)],
            )
        overlap = {
            t: con.sql(
                f"select count(*) from (select distinct msno from '{interim / t}.parquet') "
                f"semi join (select distinct msno from '{sample / t}.parquet') using (msno)"
            ).fetchone()[0]
            for t in ("transactions", "transactions_v2", "train")
            if (sample / f"{t}.parquet").exists()
        }
        if any(overlap.values()):
            raise SystemExit(f"slice overlaps the sample: {overlap}")
        clean(interim, processed, cfg)
        tx = pd.read_parquet(processed / "transactions.parquet")
        train = pd.read_parquet(processed / "train.parquet").set_index("msno").is_churn
        _, report = compare(tx, train, cfg)
    cov, paid = report["coverage"]["fitted_to_train"], report["paid"]["fitted_to_train"]
    rt = report["rule_tables"]
    gap = rt["by_scala_gap_days"]
    criteria = {
        "kappa_coverage": abs(cov["kappa"] - args.kappa_coverage) <= args.kappa_tolerance,
        "kappa_paid": abs(paid["kappa"] - args.kappa_paid) <= args.kappa_tolerance,
        "gap_30_retained_31_churn": gap["30"]["train_churn_pct"] < 50
        and gap["31"]["train_churn_pct"] > 50,
        "feb_cancel_counts_march_does_not": rt["cancel_after_renewal_by_month"]["2017-02"][
            "train_churn_pct"
        ]
        > 50
        and rt["cancel_after_renewal_by_month"]["2017-03"]["train_churn_pct"] < 50,
        "early_renewal_does_not_count": rt["early_first_renewal"]["False"]["train_churn_pct"] > 50
        and rt["early_first_renewal"]["True"]["train_churn_pct"] < 50,
        "other_share_coverage": cov["other_share_pct"] < args.max_other_pct,
    }
    out = {
        "buckets": [lo, hi],
        "overlap_with_sample": overlap,
        "criteria_passed": f"{sum(criteria.values())} of {len(criteria)}",
        "criteria": {k: bool(v) for k, v in criteria.items()},
        "report": report,
    }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
