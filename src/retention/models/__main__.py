"""Command line entry: `python -m retention.models protocol-check`.

protocol-check loads configs/evaluation.yaml (which runs the purge-rule and
locked-month checks), then counts snapshot membership for the fold, fit,
calibration and check months: P2 main rows, the October customer halves,
customers in both halves of a month (must be 0), and each half's overlap with
the fit months and the check month. It reads no outcome column and no test month.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from retention.data.sample import resolve_fraction, sample_dir
from retention.models.protocol import Protocol, load_protocol, membership
from retention.models.resources import peak_rss_gb


def processed_dir(fraction: float | None) -> Path:
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())
    base = Path(raw["labels"]["processed_dir"])
    return sample_dir(base, resolve_fraction(fraction, raw["data"]))


def protocol_check(labels_path: Path, protocol: Protocol) -> dict:
    fit = list(protocol.test_fit.fit)
    oct_ = protocol.halves_snapshot
    check = protocol.check_month
    months = sorted({*fit, oct_, check})
    mem = membership(labels_path, months, protocol)
    p2 = mem[mem.in_exp30 & mem.segment.isin(protocol.main_segments)]

    # The half depends only on the customer ID, so both counts must be 0: a
    # customer in both halves within one month, or in different halves in two months.
    within = mem.groupby(["snapshot_date", "msno"]).half.nunique()
    across = mem.groupby("msno").half.nunique()
    oct_p2 = p2[p2.snapshot_date.dt.date == oct_]
    in_fit = set(mem.msno[mem.snapshot_date.dt.date.isin(fit)])
    in_check = set(mem.msno[mem.snapshot_date.dt.date == check])
    halves = {}
    for half, g in oct_p2.groupby("half"):
        ids = set(g.msno)
        halves[half] = {
            "p2_main_rows": len(g),
            "customers": len(ids),
            "also_in_fit_months": len(ids & in_fit),
            "also_in_check_month": len(ids & in_check),
        }
    return {
        "p2_main_rows": {str(d): int((p2.snapshot_date.dt.date == d).sum()) for d in months},
        "customers_in_both_halves_within_a_month": int((within > 1).sum()),
        "customers_in_different_halves_across_months": int((across > 1).sum()),
        "months_checked_for_both_halves": [str(d) for d in months],
        "october_halves": halves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.models")
    sub = parser.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("protocol-check", help="validate the protocol and count membership")
    pc.add_argument("--fraction", type=float, default=None)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = processed_dir(args.fraction)
    protocol = load_protocol()
    report = protocol_check(out_dir / "labels.parquet", protocol)
    report["seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = peak_rss_gb()
    (out_dir / "protocol_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
