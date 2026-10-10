"""Check train_v2.csv against the published labelling rule moved one month.

Uses the exact port of the published labeller with February 2017 history and
March 2017 expiries. Under that rule a customer who renews within 29 days is
never churn; the share labelled churn in train_v2 shows how far the file
departs from it.

    uv run python scripts/check_train_v2.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from retention.data.sample import resolve_fraction, sample_dir
from retention.labels.config import parse_label_config
from retention.labels.official import NO_RENEWAL, scala_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    ap.add_argument("--fraction", type=float, default=None)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    processed = sample_dir(cfg.processed_dir, resolve_fraction(args.fraction, config["data"]))
    tx = pd.read_parquet(processed / "transactions.parquet")
    tv2 = pd.read_parquet(processed / "train_v2.parquet").set_index("msno").is_churn
    tx = tx[tx.msno.isin(tv2.index)]
    o = scala_labels(tx, date(2017, 2, 1), date(2017, 2, 28), date(2017, 3, 1), date(2017, 3, 31))
    c = o[o.candidate].join(tv2.rename("train_v2"), how="inner")
    r = c[(c.gap != NO_RENEWAL) & (c.gap <= 29)]
    timing = pd.cut(
        r.gap.astype(int),
        [-(10**4), -1, 0, 10, 29],
        labels=["before expiry", "on expiry day", "1-10 days", "11-29 days"],
    )
    by = r.groupby(timing, observed=True).train_v2.agg(
        n="size", churn_pct=lambda s: round(100 * s.mean(), 2)
    )
    out = {
        "train_v2_customers": len(tv2),
        "candidates": len(c),
        "renewed_within_29_days": len(r),
        "labelled_churn": int(r.train_v2.sum()),
        "labelled_churn_pct": round(100 * float(r.train_v2.mean()), 2),
        "by_timing": {
            str(k): {"n": int(v.n), "churn_pct": float(v.churn_pct)} for k, v in by.iterrows()
        },
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
