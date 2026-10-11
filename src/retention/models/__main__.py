"""Command line entry: `python -m retention.models {protocol-check,label-check}`.

protocol-check loads configs/evaluation.yaml (which runs the purge-rule and
locked-month checks), then counts snapshot membership for the fold, fit,
calibration and check months: P2 main rows, the October customer halves,
customers in both halves of a month (must be 0), and each half's overlap with
the fit months and the check month. It reads no outcome column and no test month.

label-check builds labels at the earliest fully observed as_of (T + h + 30) for
each 2016 snapshot and horizon in `label_as_of_check`, and the fit snapshots'
survival labels at the test pipeline's as_of, and compares them with the final
labels.parquet. Any differing row is reported and the command exits with an
error. --first-only runs the first build alone, to measure its memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

from retention.data.sample import resolve_fraction, sample_dir
from retention.models.data import labels_as_of, load_labels
from retention.models.protocol import EVAL_CONFIG, Protocol, load_protocol, membership
from retention.models.resources import peak_rss_gb


def processed_dir(fraction: float | None) -> Path:
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())
    base = Path(raw["labels"]["processed_dir"])
    return sample_dir(base, resolve_fraction(fraction, raw["data"]))


def interim_dir(fraction: float | None) -> Path:
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())
    return sample_dir(Path(raw["data"]["interim_dir"]), resolve_fraction(fraction, raw["data"]))


PRE_T_COMPARED = ["expiry_date", "days_until_expiry", "in_exp30", "segment", "paid_state"]


def label_check_builds(protocol: Protocol) -> list[tuple[date, tuple[date, ...], list[str]]]:
    """(as_of, snapshots, outcome columns to compare) for every build of the check."""
    spec = yaml.safe_load(EVAL_CONFIG.read_text())["label_as_of_check"]
    builds = []
    for t in spec["snapshots"]:
        for h in spec["horizons"]:
            as_of = t + timedelta(days=h + protocol.purge_days)
            builds.append((as_of, (t,), [f"churned_{h}" if h in (7, 30) else f"lapse_by_{h}"]))
    tf = protocol.test_fit
    builds.append((tf.as_of, tf.fit, ["event_observed", "days_to_event"]))
    return builds


def _derive(df, columns: list[str]):
    for c in columns:
        if c.startswith("lapse_by_"):
            h = int(c.removeprefix("lapse_by_"))
            df[c] = ((df.event_observed == 1) & (df.days_to_event <= h)).astype("int8")
    return df


def label_check(
    processed: Path, cache: Path, protocol: Protocol, first_only: bool
) -> tuple[dict, int]:
    final = processed / "labels.parquet"
    rows, differing = [], 0
    for as_of, snaps, outcomes in label_check_builds(protocol):
        t0 = time.time()
        built = labels_as_of(processed, cache, as_of, snaps, protocol)
        raw_cols = [c for c in outcomes if not c.startswith("lapse_by_")]
        if len(raw_cols) < len(outcomes):
            raw_cols += ["event_observed", "days_to_event"]
        cols = list(dict.fromkeys([*PRE_T_COMPARED, *raw_cols]))
        early = _derive(load_labels(built, snaps, cols, protocol), outcomes)
        late = _derive(load_labels(final, snaps, cols, protocol), outcomes)
        keys = ["msno", "snapshot_date"]
        m = early.merge(late, on=keys, how="outer", suffixes=("_early", "_final"), indicator=True)
        diff = {"rows_only_in_one": int((m._merge != "both").sum())}
        both = m[m._merge == "both"]
        for c in [*PRE_T_COMPARED, *outcomes]:
            a, b = both[f"{c}_early"], both[f"{c}_final"]
            diff[c] = int((~((a == b).fillna(False) | (a.isna() & b.isna()))).sum())
        n_diff = sum(diff.values())
        differing += n_diff
        rows.append(
            {
                "as_of": str(as_of),
                "snapshots": [str(t) for t in snaps],
                "rows": len(early),
                "events": {c: int(early[c].sum()) for c in outcomes if c != "days_to_event"},
                "differing": diff,
                "seconds": round(time.time() - t0, 1),
                "peak_rss_gb_so_far": peak_rss_gb(),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        if first_only:
            break
    return {"builds": rows, "differing_rows_total": differing}, differing


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
    lc = sub.add_parser("label-check", help="labels as of each split date vs the final labels")
    lc.add_argument("--fraction", type=float, default=None)
    lc.add_argument("--first-only", action="store_true", help="run only the first build")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = processed_dir(args.fraction)
    protocol = load_protocol()
    status = 0
    if args.command == "protocol-check":
        name = "protocol_check.json"
        report = protocol_check(out_dir / "labels.parquet", protocol)
    else:
        name = "label_check_first.json" if args.first_only else "label_check.json"
        cache = interim_dir(args.fraction) / "labels_asof"
        report, differing = label_check(out_dir, cache, protocol, args.first_only)
        status = 1 if differing else 0
    report["seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = peak_rss_gb()
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "builds"}, indent=2))
    sys.exit(status)


if __name__ == "__main__":
    main()
