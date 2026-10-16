"""Command line entry: `python -m retention.uplift <command>`.

check-public: checks both randomized datasets against configs/uplift.yaml (hash, rows,
columns, arms, treatment share), stops on the first mismatch, and writes Parquet to
data/interim/public/ only after every check of both files has passed. Refuses to run with
uncommitted changes to tracked files. Writes check_public.json beside the Parquet files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from retention.uplift import config, public_data

OUT = Path("data/interim/public")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def check_public() -> dict:
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("tracked files have uncommitted changes; commit first")
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_public.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.uplift")
    parser.add_argument("command", choices=["check-public"])
    args = parser.parse_args()
    if args.command == "check-public":
        print(json.dumps(check_public(), indent=2, default=str))


if __name__ == "__main__":
    main()
