"""Leakage check on real data: change every row dated on or after T, rebuild T, compare.

For each snapshot T and each variant, the sample's transactions and listening
logs are changed only on or after T, then clean, timeline, the labels state and
the feature build run again for snapshot T alone. The rebuilt snapshot must have
the same keys and the same values as features.parquet in every column except
the excluded metadata.

Variants:
  delete   every transaction and log row dated on or after T is removed
  extreme  those rows get extreme values; every customer gets extra rows dated T
           (a renewal expiring in 2036, a 1970 placeholder, and for every second
           customer a cancel), and every customer gets a log row dated T
  shuffle  those rows are reassigned to other customers (a fixed hash), which
           changes post-T row counts and order per customer
  reverse  rows dated before T in reverse file order, everything else unchanged
           (same-day ties must not depend on file position)

In every variant, rows dated before T keep their file order unless the
variant is reverse.

--inject-leak makes days_since_last_txn read rows dated T (and switches off the
build's own cross-check, so only this comparison can catch it): the script must
then fail and name that column.

    uv run python scripts/check_feature_leakage.py [--snapshots 2016-10-01 ...]

Writes feature_leakage.json next to features.parquet; exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import yaml

from retention.data.clean import clean
from retention.data.quality import snapshot_status
from retention.data.sample import resolve_fraction, sample_dir
from retention.features import build as build_mod
from retention.features.build import FeaturePaths, build, peak_rss_bytes, read_features
from retention.features.spec import COLUMNS
from retention.labels.config import parse_label_config
from retention.labels.snapshots import build_labels
from retention.labels.timeline import build_timeline

DEFAULT_SNAPSHOTS = [
    "2015-06-01",
    "2016-01-01",
    "2016-07-01",
    "2016-10-01",
    "2017-01-01",
    "2017-02-01",
]
VARIANTS = ["delete", "extreme", "shuffle", "reverse"]
EXCLUDED = {
    "pre_t_status": "uses the month-quality check, whose churn indicator for month T-1 "
    "is measured up to 30 days after T; offline metadata, never a model input",
    **{
        f"snapshot_usable_{h}": f"label_reliable_{h} and pre_t_status: uses month quality "
        "measured after T; offline metadata, never a model input"
        for h in (7, 30, 60)
    },
}
KEY = ["snapshot_date", "msno"]
LOG_COUNTS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]
SENTINEL = -9.223372036854776e15


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class DiskPeak:
    """Largest size of a folder, sampled every second in the background."""

    def __init__(self, path: Path):
        self.path, self.peak, self._stop = path, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, dir_bytes(self.path))
            except FileNotFoundError:
                pass
            self._stop.wait(1.0)

    def __enter__(self) -> DiskPeak:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def inject_leak() -> None:
    """days_since_last_txn reads rows dated on or before T (a deliberate leak)."""
    original = build_mod.TransactionSource.features

    def leaky(self, t, population, **kw):
        out = original(self, t, population, **kw)
        last = (
            self.con.sql(
                f"select msno, max(d) as d from tx_rows where is_cancel = 0 "
                f"and d <= date '{t}' group by msno"
            )
            .df()
            .set_index("msno")
            .d
        )
        days = (pd.Timestamp(t) - pd.to_datetime(out.msno.map(last))).dt.days
        out["days_since_last_txn"] = days.astype("Int32").array
        return out

    build_mod.TransactionSource.features = leaky
    build_mod.cross_check = lambda df: []


def index_customers(con, customers: str) -> int:
    """Number the customers 0..n-1 once, so the shuffle is a plain equality join."""
    con.sql(
        "create or replace temp table shuffle_customers as "
        f"select msno, (row_number() over (order by msno) - 1)::ubigint as i from {customers}"
    )
    return int(con.sql("select count(*) from shuffle_customers").fetchone()[0])


TX_COLS = (
    "msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, "
    "is_auto_renew, transaction_date, membership_expire_date, is_cancel"
)


def write_transactions(con, interim: Path, out: Path, t: date, variant: str) -> None:
    """Interim transaction files with rows dated on or after t changed by `variant`.

    Rows dated before t keep their original file order (so their source row
    numbers are unchanged), except in the reverse variant, which reverses only
    them. Changed and added rows come after them.
    """
    d = int(t.strftime("%Y%m%d"))
    customers = (
        f"(select distinct msno from read_parquet(['{interim}/transactions.parquet', "
        f"'{interim}/transactions_v2.parquet']))"
    )
    for name in ("transactions", "transactions_v2"):
        src = f"read_parquet('{interim / name}.parquet', file_row_number = true)"
        order = "-file_row_number" if variant == "reverse" else "file_row_number"
        before = (
            f"select {TX_COLS}, 0 as _part, {order} as _pos from {src} where transaction_date < {d}"
        )
        late = f"select *, file_row_number as _pos from {src} where transaction_date >= {d}"
        if variant == "delete":
            body = before
        elif variant == "reverse":
            body = f"{before} union all select {TX_COLS}, 1, _pos from ({late})"
        elif variant == "extreme":
            placeholder = "membership_expire_date < 20040101"
            body = f"""{before} union all
              select msno,
                case when {placeholder} then payment_method_id
                     when payment_method_id = 13 then 39 else 13 end,
                case when {placeholder} then payment_plan_days else 999 end,
                99999, 99999, 1 - is_auto_renew, transaction_date, membership_expire_date,
                is_cancel, 1, _pos
              from ({late})"""
            if name == "transactions":
                body += f""" union all
                  select msno, 13, 999, 99999, 99999, 0, {d}, 20360101, 0, 2, 0 from {customers}
                  union all
                  select msno, 41, 30, 149, 149, 1, {d}, 19700101, 0, 3, 0 from {customers}
                  union all
                  select msno, 41, 30, 149, 149, 1, {d}, {d}, 1, 4, 0 from {customers}
                  where hash(msno) % 2 = 1"""
        else:  # shuffle
            n = index_customers(con, customers)
            body = f"""{before} union all
              select c.msno, l.payment_method_id, l.payment_plan_days, l.plan_list_price,
                l.actual_amount_paid, l.is_auto_renew, l.transaction_date,
                l.membership_expire_date, l.is_cancel, 1, l._pos
              from (select *, hash(msno || transaction_date::varchar || 'shuffle') % {n} as j
                    from ({late})) l
              join shuffle_customers c on c.i = l.j"""
        con.sql(
            f"copy (select {TX_COLS} from ({body}) order by _part, _pos, msno) "
            f"to '{out / name}.parquet' (format parquet)"
        )


def write_logs(con, logs: list[Path], out: Path, t: date, variant: str) -> list[Path]:
    """Pre-T logs unchanged in one file; the variant's rows on or after t in another."""
    if variant == "reverse":
        return logs  # usage aggregates by day, so log row order cannot matter
    d = int(t.strftime("%Y%m%d"))
    src = "read_parquet([" + ", ".join(f"'{p}'" for p in logs) + "])"
    pre = out / "logs_pre.parquet"
    if not pre.exists():
        con.sql(f"copy (select * from {src} where date < {d}) to '{pre}' (format parquet)")
    if variant == "delete":
        return [pre]
    late = f"select * from {src} where date >= {d}"
    post = out / f"logs_{variant}.parquet"
    if variant == "extreme":
        counts = ", ".join(f"1000000 as {c}" for c in LOG_COUNTS)
        secs = f"case when hash(msno, date) % 2 = 0 then 1e12 else {SENTINEL} end"
        customers = f"(select distinct msno from {src})"
        body = f"""select msno, date, {counts}, {secs} as total_secs from ({late})
          union all
          select msno, {d} as date, {counts}, {secs} as total_secs from {customers}"""
    else:
        n = index_customers(con, f"(select distinct msno from {src})")
        body = f"""select c.msno, l.* exclude (msno, j)
          from (select *, hash(msno || date::varchar || 'shuffle') % {n} as j from ({late})) l
          join shuffle_customers c on c.i = l.j"""
    con.sql(f"copy ({body}) to '{post}' (format parquet)")
    return [pre, post]


def prepare(paths, interim: Path, root: Path, cfg, mq, t: date, variant: str) -> dict:
    """Changed inputs for snapshot t: clean, timeline and labels state (in a child process)."""
    con = duckdb.connect()
    con.sql("set enable_progress_bar = false")
    tmp_interim = root / f"interim_{variant}"
    tmp_interim.mkdir(exist_ok=True)
    for name in ("members_v3", "train", "train_v2"):
        shutil.copy(interim / f"{name}.parquet", tmp_interim / f"{name}.parquet")
    write_transactions(con, interim, tmp_interim, t, variant)
    logs = write_logs(con, list(paths.logs), root, t, variant)
    con.close()

    processed = root / f"processed_{variant}"
    clean(tmp_interim, processed, cfg)
    shutil.rmtree(tmp_interim)
    tx = pd.read_parquet(processed / "transactions.parquet")
    tl, _ = build_timeline(tx, cfg.renewal_window_days)
    tl.to_parquet(processed / "timeline.parquet", index=False)
    one = replace(cfg, snapshots=(t,)).validate()
    build_labels(tx, one, snapshot_status(mq.set_index("month").status, one)).to_parquet(
        processed / "labels.parquet", index=False
    )
    shutil.copy(paths.processed / "month_quality.parquet", processed / "month_quality.parquet")
    return {"processed": processed, "logs": tuple(logs), "peak_rss_bytes": peak_rss_bytes()}


def build_and_compare(paths, cfg, t: date, prepared: dict, leak: bool) -> dict:
    """Build snapshot t from the prepared inputs and compare (in a fresh child process)."""
    if leak:
        inject_leak()
    processed = prepared["processed"]
    out = processed / "features_rebuilt.parquet"
    build(FeaturePaths(processed, prepared["logs"]), cfg, [t], out=out)
    rebuilt = read_features(out).sort_values(KEY).reset_index(drop=True)
    shutil.rmtree(processed)
    original = pq.read_table(paths.out, filters=[("snapshot_date", "=", t)])
    original = (
        original.to_pandas(types_mapper=build_mod.NULLABLE.get, date_as_object=False)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    r = compare(original, rebuilt)
    r["peak_rss_bytes"] = peak_rss_bytes()
    return r


def in_child(fn, *args):
    """Run fn in a fresh process, so its memory goes back to the OS when it exits."""
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        return pool.submit(fn, *args).result()


def compare(original: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    keys_equal = set(zip(original.msno, original.snapshot_date, strict=True)) == set(
        zip(rebuilt.msno, rebuilt.snapshot_date, strict=True)
    )
    differing = []
    if keys_equal:
        for c in [c.name for c in COLUMNS if c.name not in EXCLUDED]:
            try:
                pd.testing.assert_series_equal(
                    original[c], rebuilt[c], check_names=False, check_exact=True
                )
            except AssertionError:
                differing.append(c)
    return {"rows": len(rebuilt), "keys_equal": keys_equal, "differing_columns": differing}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--snapshots", nargs="*", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--inject-leak", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-snapshot-minutes", type=float, default=15.0)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    cfg = parse_label_config(config["labels"])
    fraction = resolve_fraction(args.fraction, config["data"])
    interim = sample_dir(Path(config["data"]["interim_dir"]), fraction)
    paths = FeaturePaths(
        processed=sample_dir(cfg.processed_dir, fraction),
        logs=(interim / "user_logs.parquet", interim / "user_logs_v2.parquet"),
    )
    name = "feature_leakage_injected.json" if args.inject_leak else "feature_leakage.json"
    report_path = args.out or paths.processed / name
    print("Excluded from the comparison (offline metadata):")
    for col, why in EXCLUDED.items():
        print(f"  {col}: {why}")

    mq = pd.read_parquet(paths.processed / "month_quality.parquet")
    root = paths.processed / "_leakage_tmp"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    t0 = time.time()
    peak_rss = 0
    stopped = None
    results: dict = {}
    failed = []
    try:
        with DiskPeak(root) as disk:
            for s in args.snapshots:
                t = date.fromisoformat(s)
                results[s] = {}
                t_snap = time.time()
                for v in args.variants:
                    t1 = time.time()
                    prepared = in_child(prepare, paths, interim, root, cfg, mq, t, v)
                    r = in_child(build_and_compare, paths, cfg, t, prepared, args.inject_leak)
                    r["peak_rss_prepare_bytes"] = prepared["peak_rss_bytes"]
                    peak_rss = max(peak_rss, r["peak_rss_bytes"], prepared["peak_rss_bytes"])
                    r["seconds"] = round(time.time() - t1, 1)
                    results[s][v] = r
                    ok = r["keys_equal"] and not r["differing_columns"]
                    if not ok:
                        failed.append((s, v, r["keys_equal"], r["differing_columns"]))
                    print(
                        f"{s} {v:8s} rows {r['rows']:>7} keys_equal {r['keys_equal']} "
                        f"differing {r['differing_columns']} ({r['seconds']}s)",
                        flush=True,
                    )
                for f in root.glob("logs_*.parquet"):
                    f.unlink()
                minutes = (time.time() - t_snap) / 60
                results[s]["minutes"] = round(minutes, 1)
                if minutes > args.max_snapshot_minutes:
                    stopped = f"{s} took {minutes:.1f} min (limit {args.max_snapshot_minutes})"
                    print(f"STOPPED: {stopped}")
                    break
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report = {
        "inject_leak": args.inject_leak,
        "excluded": EXCLUDED,
        "snapshots": results,
        "failures": [
            {"snapshot": s, "variant": v, "keys_equal": k, "differing_columns": c}
            for s, v, k, c in failed
        ],
        "seconds": round(time.time() - t0, 1),
        "peak_disk_bytes": disk.peak,
        "peak_rss_bytes_per_child_process": peak_rss,
        "stopped": stopped,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"total {report['seconds']}s, peak temporary disk {disk.peak / 1e9:.2f} GB, "
        f"peak RSS of a child process {peak_rss / 1e9:.2f} GB"
    )
    print(f"wrote {report_path}")
    if stopped:
        raise SystemExit(2)
    if failed:
        cols = sorted({c for *_, cs in failed for c in cs})
        print(f"FAILED: leakage detected in {cols or 'the key set'}")
        raise SystemExit(1)
    print("PASSED: no differences")


if __name__ == "__main__":
    main()
