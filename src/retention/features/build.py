"""features.parquet: build every snapshot, check, then publish atomically.

The table is written to a temporary path, every check runs on that file, and
only then is it renamed to features.parquet. Any failed check deletes the
temporary file and raises BuildError, so a half-built table never sits where
a good one is expected.
"""

from __future__ import annotations

import resource
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retention.data.quality import completeness_status, pre_t_status
from retention.features import spec
from retention.features.checks import cross_check
from retention.features.members import MEMBER_COLUMNS, member_features
from retention.features.transactions import TransactionSource
from retention.features.usage import UsageSource
from retention.labels.config import LabelConfig

HORIZONS = (7, 30, 60)
LABEL_INPUTS = [
    "msno",
    "snapshot_date",
    "days_until_expiry",
    "segment",
    "paid_state",
    "paid_expiry_date",
    *(f"label_reliable_{h}" for h in HORIZONS),
]
DUCKDB_THREADS = 4  # fewer threads, lower peak memory (about 0.3 GB) for a few seconds
NULL_REPORT_LIMIT = 0.01  # a column above this null share in any snapshot needs a reason
# Label-free report of categories in the likely test months that never appear in training.
TRAIN_MONTHS = [date(2016, m, 1) for m in (7, 8, 9, 10)]
TEST_MONTHS = [date(2017, 1, 1), date(2017, 2, 1)]


# Arrow type -> pandas nullable dtype, so strings and integers with nulls are not object.
NULLABLE = {
    pa.string(): pd.StringDtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.float64(): pd.Float64Dtype(),
}


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeaturePaths:
    processed: Path  # labels, timeline, members and month_quality Parquet files
    logs: tuple[Path, ...]  # raw daily listening logs (typed interim Parquet)

    @property
    def out(self) -> Path:
        return self.processed / "features.parquet"

    @property
    def null_report(self) -> Path:
        return self.processed / "features_null_report.parquet"


def read_labels(path: Path, snapshots: list[date] | None = None) -> pd.DataFrame:
    lab = duckdb.sql(
        f"select {', '.join(LABEL_INPUTS)} from read_parquet('{path}') order by snapshot_date, msno"
    ).df()
    lab["snapshot_date"] = pd.to_datetime(lab.snapshot_date).dt.date
    if snapshots is not None:
        lab = lab[lab.snapshot_date.isin(set(snapshots))]
    return lab.reset_index(drop=True)


def copy_label_columns(frame: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Columns taken unchanged from labels.parquet."""
    for c in spec.LABEL_SOURCED:
        frame[c] = labels[c].to_numpy()


def assemble(
    t: date,
    g: pd.DataFrame,
    usage: UsageSource,
    tx: TransactionSource,
    members: pd.DataFrame,
    pre_t: str,
    data_start: date,
    n_chunks: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """All columns for snapshot t, in registry order, and the failed checks."""
    g = g.reset_index(drop=True)
    msno = g.msno.to_numpy()
    u = usage.features(t, msno, n_chunks=n_chunks)
    x = tx.features(t, msno, check_columns=True)
    m = member_features(members, u[["msno", "has_any_log_before_T"]], t, data_start)
    errors = []
    for name, part in (("usage", u), ("transactions", x), ("members", m)):
        if len(part) != len(g) or not (part.msno.to_numpy() == msno).all():
            errors.append(f"{name}: rows do not match the labels rows for this snapshot")
    if errors:
        return pd.DataFrame(), errors

    f = pd.DataFrame({"msno": msno, "snapshot_date": g.snapshot_date.to_numpy()})
    copy_label_columns(f, g)
    paid = pd.to_datetime(g.paid_expiry_date)
    f["days_until_paid_expiry"] = (paid - pd.Timestamp(t)).dt.days.astype("Int32").array
    for part in (u, x, m):
        for c in part.columns.drop("msno"):
            f[c] = part[c].array
    f["pre_t_status"] = pre_t
    for h in HORIZONS:
        f[f"snapshot_usable_{h}"] = g[f"label_reliable_{h}"].to_numpy() & (pre_t == "reliable")

    errors += cross_check(f)
    quiet = f.has_any_log_before_T == 0
    if quiet.any() and not f.loc[quiet, MEMBER_COLUMNS].isna().all().all():
        errors.append("member columns not null for rows with no log before T")
    for c in spec.LABEL_SOURCED:
        if not f[c].equals(pd.Series(g[c].to_numpy(), name=c)):
            errors.append(f"{c} differs from labels.parquet")
    if f.msno.duplicated().any():
        errors.append("duplicate msno within the snapshot")
    return f[[c.name for c in spec.COLUMNS]], errors


def to_arrow(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame, schema=spec.arrow_schema(), preserve_index=False)


def null_report(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """Null share of every column in every snapshot (long format)."""
    names = [c.name for c in spec.COLUMNS]
    counts = ", ".join(f'count("{n}") as "nn__{n}"' for n in names)
    src = f"read_parquet('{path}')"
    wide = con.sql(
        f"select snapshot_date, count(*) as n_rows, {counts} from {src} group by 1 order by 1"
    ).df()
    long = wide.melt(id_vars=["snapshot_date", "n_rows"], var_name="column", value_name="non_null")
    long["column"] = long.column.str.removeprefix("nn__")
    long["null_share"] = 1 - long.non_null / long.n_rows
    return long[["snapshot_date", "column", "n_rows", "null_share"]]


def check_null_report(report: pd.DataFrame) -> list[str]:
    """Columns above the limit need a documented reason; 'never null' must hold exactly."""
    errors = []
    worst = report.groupby("column").null_share.max()
    for name, share in worst.items():
        reason = spec.BY_NAME[name].null_reason
        if share > NULL_REPORT_LIMIT and not reason:
            errors.append(f"{name}: up to {share:.2%} null with no documented reason")
        if share > 0 and not reason:
            errors.append(f"{name}: documented as never null but {share:.4%} null")
    return errors


def check_output(
    con: duckdb.DuckDBPyConnection, path: Path, labels: pd.DataFrame
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    schema = pq.read_schema(path)
    expected = spec.arrow_schema()
    if schema.names != expected.names:
        errors.append(f"columns differ from the registry: {schema.names} != {expected.names}")
    else:
        for field in expected:
            if schema.field(field.name).type != field.type:
                errors.append(f"{field.name}: type {schema.field(field.name).type} != {field.type}")
    first = next(pq.ParquetFile(path).iter_batches(batch_size=1000))
    sample = pa.Table.from_batches([first]).to_pandas(
        types_mapper=NULLABLE.get, date_as_object=False
    )
    objects = [c for c in sample.columns if sample[c].dtype == object]
    if objects:
        errors.append(f"object columns: {objects}")

    src = f"read_parquet('{path}')"
    # Keys compared one snapshot at a time, so no hash table holds all 4M+ keys.
    n_file = con.sql(f"select count(*) from {src}").fetchone()[0]
    if n_file != len(labels):
        errors.append(f"keys: {n_file} rows in the file, {len(labels)} in labels")
    for t, g in labels.groupby("snapshot_date", sort=True):
        con.register("_label_keys", g[["msno"]])
        n, n_keys, missing, extra = con.sql(
            f"""
            with f as (select msno from {src} where snapshot_date = date '{t}')
            select (select count(*) from f), (select count(distinct msno) from f),
              (select count(*) from _label_keys k anti join f using (msno)),
              (select count(*) from f anti join _label_keys k using (msno))
            """
        ).fetchone()
        con.unregister("_label_keys")
        if not (n == n_keys == len(g)) or missing or extra:
            errors.append(
                f"keys at {t}: {n} rows, {n_keys} distinct, labels {len(g)}, "
                f"missing {missing}, extra {extra}"
            )

    for col, allowed in spec.VOCAB.items():
        seen = [
            r[0]
            for r in con.sql(f"select distinct \"{col}\" from read_parquet('{path}')").fetchall()
        ]
        bad = sorted({v for v in seen if v is not None} - set(allowed), key=str)
        if bad:
            errors.append(f"{col}: values outside the vocabulary: {bad[:20]}")

    nulls = null_report(con, path)
    errors += check_null_report(nulls)
    return errors, {"null_report": nulls, "unseen_categories": unseen_categories(con, path)}


def unseen_categories(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, Any]:
    """Values in the likely test months that never appear in the training months."""
    train = ", ".join(f"date '{d}'" for d in TRAIN_MONTHS)
    test = ", ".join(f"date '{d}'" for d in TEST_MONTHS)
    out: dict[str, Any] = {}
    for col in [c for c in spec.VOCAB if c not in ("pre_t_status", "last_txn_span_status")]:
        rows = con.sql(
            f"""
            select snapshot_date, "{col}" as v, count(*) as n from read_parquet('{path}')
            where snapshot_date in ({test}) and "{col}" is not null
              and "{col}" not in (select "{col}" from read_parquet('{path}')
                                  where snapshot_date in ({train}) and "{col}" is not null)
            group by all order by all
            """
        ).fetchall()
        if rows:
            out[col] = [{"snapshot": str(d), "value": v, "rows": int(k)} for d, v, k in rows]
    return out


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def build(
    paths: FeaturePaths,
    cfg: LabelConfig,
    snapshots: list[date] | None = None,
    n_chunks: int = 1,
    memory_limit: str = "2GB",
    out: Path | None = None,
) -> dict[str, Any]:
    """Build, check and publish features.parquet (or `out`). Raises BuildError on any failure."""
    t0 = time.time()
    out = out or paths.out
    tmp_out = out.with_name(out.name + ".tmp")
    work = Path(tempfile.mkdtemp(prefix="features_work_", dir=paths.processed))
    report: dict[str, Any] = {}
    try:
        con = duckdb.connect(str(work / "work.duckdb"))
        con.sql(f"set memory_limit = '{memory_limit}'")
        con.sql(f"set temp_directory = '{work / 'tmp'}'")
        con.sql("set preserve_insertion_order = false")
        con.sql(f"set threads = {DUCKDB_THREADS}")
        con.sql("set enable_progress_bar = false")

        labels = read_labels(paths.processed / "labels.parquet", snapshots)
        dates = sorted(labels.snapshot_date.unique())
        if not dates:
            raise BuildError("no snapshots to build")
        mq = pd.read_parquet(paths.processed / "month_quality.parquet").set_index("month")
        months = completeness_status(mq, cfg.data_quality)
        if cfg.lookback_days is None:
            raise BuildError("labels.lookback_days is not set in the config")
        pre_t = pre_t_status(months, dates, cfg.lookback_days)

        logs = "read_parquet([" + ", ".join(f"'{p}'" for p in paths.logs) + "])"
        usage = UsageSource(con, logs, labels.msno.unique())
        tx = TransactionSource(
            con,
            f"read_parquet('{paths.processed / 'timeline.parquet'}')",
            months,
            cfg.renewal_window_days,
            cfg.data_start,
        )
        members = duckdb.sql(
            "select msno, city, bd, gender, registered_via, registration_init_time "
            f"from '{paths.processed / 'members.parquet'}'"
        ).df()
        members = members[members.msno.isin(set(labels.msno))]
        report["load_seconds"] = round(time.time() - t0, 1)
        peaks = {"load": peak_rss_bytes()}

        per_snapshot = []
        with pq.ParquetWriter(tmp_out, spec.arrow_schema(), compression="zstd") as writer:
            for t, g in labels.groupby("snapshot_date", sort=True):
                t1 = time.time()
                frame, errors = assemble(
                    t, g, usage, tx, members, pre_t[t], cfg.data_start, n_chunks
                )
                if errors:
                    raise BuildError(f"snapshot {t}: " + "; ".join(errors))
                writer.write_table(to_arrow(frame))
                per_snapshot.append(
                    {"snapshot": str(t), "rows": len(frame), "seconds": round(time.time() - t1, 2)}
                )

        peaks["snapshots"] = peak_rss_bytes()
        # Free the work database (logs, timeline) before checking the written file.
        con.close()
        del usage, tx, members
        con = duckdb.connect()
        con.sql(f"set memory_limit = '{memory_limit}'")
        con.sql(f"set threads = {DUCKDB_THREADS}")
        con.sql("set enable_progress_bar = false")
        errors, checked = check_output(con, tmp_out, labels)
        peaks["checks"] = peak_rss_bytes()
        if errors:
            raise BuildError("output checks failed: " + "; ".join(errors))
        con.close()
        tmp_out.replace(out)
        checked["null_report"].to_parquet(
            out.with_name(out.stem + "_null_report.parquet"), index=False
        )
        worst = checked["null_report"].groupby("column").null_share.max()
        report |= {
            "rows": int(sum(s["rows"] for s in per_snapshot)),
            "snapshots": len(per_snapshot),
            "columns": len(spec.COLUMNS),
            "per_snapshot": per_snapshot,
            "columns_over_1pct_null": {
                k: round(float(v), 4) for k, v in worst.items() if v > NULL_REPORT_LIMIT
            },
            "unseen_categories": checked["unseen_categories"],
            "output_bytes": out.stat().st_size,
            "seconds": round(time.time() - t0, 1),
            "peak_rss_bytes": peak_rss_bytes(),
            # Running maximum after each stage: a rise means that stage set a new peak.
            "peak_rss_bytes_after_stage": peaks,
        }
        return report
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def read_features(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read features.parquet with nullable pandas dtypes (no object columns)."""
    return pq.read_table(path, columns=columns).to_pandas(
        types_mapper=NULLABLE.get, date_as_object=False
    )
