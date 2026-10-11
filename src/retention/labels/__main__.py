"""Command line entry: `python -m retention.labels {clean,timeline,quality,labels,compare,all}`.

`all` runs the five stages in order on one customer sample.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from retention.data.clean import ValidationError, clean, write_validation
from retention.data.quality import INDICATORS, run_quality, snapshot_status
from retention.data.sample import resolve_fraction, sample_dir
from retention.labels.compare import compare
from retention.labels.config import LabelConfig, parse_label_config
from retention.labels.snapshots import build_labels
from retention.labels.timeline import build_timeline


def run_timeline(out_dir: Path, cfg: LabelConfig) -> None:
    t0 = time.time()
    tx = duckdb.sql(f"select * from '{out_dir / 'transactions.parquet'}'").df()
    tl, spells = build_timeline(tx, cfg.renewal_window_days)
    prev = tl.groupby("msno").eff_expiry_after.shift()
    live = tl.is_cancel.eq(1) & ~tl.sameday_sameplan_cancel & tl.expiry_date.notna() & prev.notna()
    late = live & ((tl.transaction_date - prev).dt.days > cfg.renewal_window_days)
    report = {
        "rows": len(tl),
        "customers": int(tl.msno.nunique()),
        "spells": len(spells),
        "closed_spells": int(spells.next_sub_date.notna().sum()),
        "rows_affecting_coverage": int(tl.affects_coverage.sum()),
        "cancels_cutting_coverage": int((tl.is_cancel.eq(1) & tl.affects_coverage).sum()),
        "cancels_after_lapse_ignored": int(late.sum()),
        "cancels_after_lapse_that_would_have_cut": int((late & (tl.expiry_date < prev)).sum()),
        "gap_days_quantiles": {
            str(q): float(spells.gap_days.dropna().quantile(q)) for q in (0.5, 0.9, 0.99)
        },
    }
    for name, df in (("timeline", tl), ("spells", spells)):
        tmp = out_dir / f"{name}.parquet.tmp"
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out_dir / f"{name}.parquet")
    report["output_bytes"] = {
        n: (out_dir / f"{n}.parquet").stat().st_size for n in ("timeline", "spells")
    }
    report["seconds"] = round(time.time() - t0, 1)
    write_validation(out_dir, "timeline", report)
    print(json.dumps(report, indent=2))


def run_quality_cmd(out_dir: Path, logs: str, cfg: LabelConfig) -> None:
    t0 = time.time()
    table, snaps, th = run_quality(out_dir, logs, cfg)
    q = cfg.data_quality
    table.reset_index().to_parquet(out_dir / "month_quality.parquet", index=False)
    snaps.to_parquet(out_dir / "snapshot_reliability.parquet", index=False)
    counts = snaps.groupby(["horizon", "status"]).size().unstack(fill_value=0)
    report = {
        "windows": {
            "listening_days": list(q.listening_days),
            "listening_min_active_days": q.listening_min_active_days,
            "returns_days": list(q.returns_days),
            "batch_baseline_days": q.batch_baseline_days,
        },
        "rule": {
            "mad_k": q.mad_k,
            "sensitivity": q.sensitivity,
            "min_assessed": q.min_assessed,
            "unreliable_failures": q.unreliable_failures,
            "label_window": "[T, T + h]",
        },
        "thresholds": th,
        "sorted_values": {i: sorted(round(v, 4) for v in table[i].dropna()) for i in INDICATORS},
        "months": table.reset_index().to_dict(orient="records"),
        "month_status_counts": table.status.value_counts().to_dict(),
        "snapshot_status_counts": {int(h): r.to_dict() for h, r in counts.iterrows()},
        "seconds": round(time.time() - t0, 1),
    }
    write_validation(out_dir, "quality", report)
    pd.set_option("display.width", 250)
    print(table.drop(columns=["fails_low_k", "fails_high_k"]).round(4).to_string())
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("thresholds", "month_status_counts", "snapshot_status_counts", "seconds")
            },
            indent=1,
            default=str,
        )
    )


def label_summary(labels: pd.DataFrame, cfg: LabelConfig) -> dict:
    """Counts of churned / retained / unobservable per horizon for both families."""
    out: dict = {}
    for fam in ("", "paid_"):
        for h in cfg.horizons:
            col = labels[f"{fam}churned_{h}"]
            out[f"{fam}churned_{h}"] = {
                "churned": int((col == 1).sum()),
                "retained": int((col == 0).sum()),
                "unobservable": int(col.isna().sum()),
            }
        for c in (f"{fam}event_observed", f"{fam}churned_classic"):
            col = labels[c]
            out[c] = {
                "1": int((col == 1).sum()),
                "0": int((col == 0).sum()),
                "missing": int(col.isna().sum()),
            }
    return out


def run_labels(out_dir: Path, cfg: LabelConfig) -> None:
    t0 = time.time()
    tx = pd.read_parquet(out_dir / "transactions.parquet")
    months = pd.read_parquet(out_dir / "month_quality.parquet").set_index("month").status
    labels = build_labels(tx, cfg, snapshot_status(months, cfg))
    tmp = out_dir / "labels.parquet.tmp"
    labels.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_dir / "labels.parquet")
    both = labels[labels.paid_state.eq("active")]
    per_snapshot = []
    for t, g in labels.groupby("snapshot_date"):
        b = both[both.snapshot_date == t]
        row = {"snapshot": str(t), "rows": len(g), "paid_active": len(b)}
        for h in cfg.horizons:
            x, y = b[f"churned_{h}"], b[f"paid_churned_{h}"]
            row[f"differ_{h}"] = int(((x != y) & x.notna() & y.notna()).sum())
        row["differ_days_to_event"] = int((b.days_to_event != b.paid_days_to_event).sum())
        per_snapshot.append(row)
    report = {
        "as_of": str(cfg.as_of),
        "rows": len(labels),
        "customers": int(labels.msno.nunique()),
        "snapshots": len(cfg.snapshots),
        "counts": label_summary(labels, cfg),
        "paid_state": labels.paid_state.value_counts().to_dict(),
        "per_snapshot": per_snapshot,
        "output_bytes": (out_dir / "labels.parquet").stat().st_size,
        "seconds": round(time.time() - t0, 1),
    }
    write_validation(out_dir, "labels", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_snapshot"}, indent=1))
    print(pd.DataFrame(per_snapshot).to_string(index=False))


def run_compare(out_dir: Path, cfg: LabelConfig) -> None:
    t0 = time.time()
    tx = pd.read_parquet(out_dir / "transactions.parquet")
    train = pd.read_parquet(out_dir / "train.parquet").set_index("msno").is_churn
    table, report = compare(tx, train, cfg)
    table.to_parquet(out_dir / "label_comparison.parquet", index=False)
    report["seconds"] = round(time.time() - t0, 1)
    write_validation(out_dir, "compare", report)
    print(json.dumps(report, indent=1))


STAGES = ["clean", "timeline", "quality", "labels", "compare"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.labels")
    parser.add_argument("command", choices=[*STAGES, "all"])
    parser.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    parser.add_argument("--fraction", type=float, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    fraction = resolve_fraction(args.fraction, config["data"])
    interim = sample_dir(Path(config["data"]["interim_dir"]), fraction)
    out_dir = sample_dir(cfg.processed_dir, fraction)
    logs = " union all ".join(
        f"select msno, date from '{interim / f}'"
        for f in ("user_logs.parquet", "user_logs_v2.parquet")
    )
    for stage in STAGES if args.command == "all" else [args.command]:
        print(f"== {stage}")
        if stage == "clean":
            run_clean(interim, out_dir, cfg)
        elif stage == "timeline":
            run_timeline(out_dir, cfg)
        elif stage == "quality":
            run_quality_cmd(out_dir, f"({logs})", cfg)
        elif stage == "labels":
            run_labels(out_dir, cfg)
        else:
            run_compare(out_dir, cfg)


def run_clean(interim: Path, out_dir: Path, cfg: LabelConfig) -> None:
    try:
        report = clean(interim, out_dir, cfg)
    except ValidationError as err:
        print(f"FAILED: {err}\nsee {out_dir / 'validation.json'}")
        raise SystemExit(1) from err
    keys = ("data_end", "duplicates", "quality_flags", "output_bytes", "seconds")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
