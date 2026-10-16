"""Command line entry: `python -m retention.uplift <command>`.

check-public: checks both randomized datasets against configs/uplift.yaml (hash, rows,
columns, arms, treatment share), stops on the first mismatch, and writes Parquet to
data/interim/public/ only after every check of both files has passed. Refuses to run with
uncommitted changes to tracked files. Writes check_public.json beside the Parquet files.

recount: rebuilds the KKBox renewal units (pre-T label columns and transactions dated on
or before 2016-12-31 only; no outcome column) and compares every registered count, and the
2016 figures quoted in docs/causal_assumptions.md, with the recount. Stops (exit code 1)
if any differs. Writes recount.json under data/processed/kkbox/sample_020/uplift/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from retention.models.protocol import load_protocol
from retention.models.resources import peak_rss_gb
from retention.uplift import config, kkbox_units, public_data

OUT = Path("data/interim/public")
KKBOX = Path("data/processed/kkbox/sample_020")
KKBOX_OUT = KKBOX / "uplift"
# Figures quoted in the first paragraph of docs/causal_assumptions.md (from the counts run):
# full-price or discounted first renewals, 2016-01 to 2016-11, all plan lengths.
DOC_FIGURES_2016 = {"units": 1582100, "method_23": 3457, "customers": 233945, "m23_customers": 399}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _require_clean() -> None:
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("tracked files have uncommitted changes; commit first")


def check_public() -> dict:
    _require_clean()
    start = time.perf_counter()
    v = config.load()["validation"]["datasets"]
    h, c = v["hillstrom"], v["criteo"]
    log: dict = {
        "commit": _git("rev-parse", "--short", "HEAD").strip(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        log["hillstrom"] = public_data.check_hillstrom(
            Path(h["local_file"]),
            h["sha256"],
            h["expected_rows"],
            h["expected_columns"],
            [
                h["treatment_arms"]["mens_vs_none"],
                h["treatment_arms"]["womens_vs_none"],
                h["control_arm"],
            ],
            h["arm_share_tolerance"],
        )
        log["criteo"] = public_data.check_criteo(
            Path(c["local_file"]),
            c["expected_sha256"],
            c["expected_rows"],
            c["expected_columns"],
            c["treatment_share"],
            c["treatment_share_tolerance"],
        )
        log["status"] = "all checks passed"
        log["hillstrom"]["parquet"] = public_data.to_parquet(
            Path(h["local_file"]), OUT / "hillstrom.parquet"
        )
        log["criteo"]["parquet"] = public_data.to_parquet(
            Path(c["local_file"]), OUT / "criteo_uplift_v2.1.parquet"
        )
    except public_data.CheckFailed as err:
        log["status"] = "stopped"
        log["failure"] = str(err)
    log["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    log["seconds"] = round(time.perf_counter() - start, 1)
    log["peak_rss_gb"] = peak_rss_gb()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_public.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def _months(first: date, last: date) -> list[date]:
    return [d.date() for d in pd.date_range(first, last, freq="MS")]


def _units(snapshots: list[date], tx: pd.DataFrame) -> pd.DataFrame:
    """Renewal units, reading at most four snapshots of labels at a time."""
    protocol = load_protocol()
    parts = []
    for i in range(0, len(snapshots), 4):
        members = kkbox_units.read_members(KKBOX / "labels.parquet", snapshots[i : i + 4], protocol)
        parts.append(kkbox_units.renewal_units(members, tx))
    return pd.concat(parts, ignore_index=True)


def recount() -> dict:
    _require_clean()
    start = time.perf_counter()
    registered = config.load()["kkbox"]["counts_before_outcome"]
    tx = kkbox_units.read_transactions(KKBOX / "transactions.parquet")
    units = _units(kkbox_units.SNAPSHOTS, tx)
    reg = kkbox_units.registered_set(units)
    analysed = kkbox_units.analysed_set(reg)
    counts = kkbox_units.recount(reg, analysed)
    year = _units(_months(date(2016, 1, 1), date(2016, 11, 1)), tx)
    fd = year[year.kind.isin(["full", "discounted"])]
    m23 = fd.payment_method_id.eq(kkbox_units.PAYMENT_METHOD)
    doc = {
        "units": len(fd),
        "method_23": int(m23.sum()),
        "customers": int(fd.msno.nunique()),
        "m23_customers": int(fd.msno[m23].nunique()),
    }
    log: dict = {
        "commit": _git("rev-parse", "--short", "HEAD").strip(),
        "registered": {k: registered[k] for k in asdict(counts)},
        "recounted": asdict(counts),
        "share_with_zero_chance": {
            "registered": registered["share_with_zero_chance"],
            "recounted": round(1 - counts.method_23 / counts.registered_set_all_methods, 3),
        },
        "doc_figures_2016": {"quoted": DOC_FIGURES_2016, "recounted": doc},
        "kinds_2016": year.kind.value_counts().to_dict(),
    }
    try:
        kkbox_units.check_counts(counts, registered)
        if doc != DOC_FIGURES_2016:
            raise kkbox_units.CountMismatch(f"2016 doc figures differ: {doc}")
        zero = log["share_with_zero_chance"]
        if zero["registered"] != zero["recounted"]:
            raise kkbox_units.CountMismatch("share_with_zero_chance differs")
        log["status"] = "every count matches"
    except kkbox_units.CountMismatch as err:
        log["status"] = "stopped"
        log["failure"] = str(err)
    log["seconds"] = round(time.perf_counter() - start, 1)
    log["peak_rss_gb"] = peak_rss_gb()
    KKBOX_OUT.mkdir(parents=True, exist_ok=True)
    (KKBOX_OUT / "recount.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.uplift")
    parser.add_argument("command", choices=["check-public", "recount"])
    args = parser.parse_args()
    run = check_public if args.command == "check-public" else recount
    log = run()
    print(json.dumps(log, indent=2, default=str))
    if log.get("status") == "stopped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
