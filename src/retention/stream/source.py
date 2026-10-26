"""The replay's source reader: real 2016 rows in the registered order, stopped at the source.

configs/serving.yaml, streaming and step6. Three files are read (streaming.sources):
transactions, transactions_v2 and user_logs of the sample's interim folder. Every
read names its date bounds, and the reader refuses before any query:
- an upper bound after 2016-11-01 (the replay's store date): no row dated
  2016-11-01 or later is ever requested, so none can be returned;
- for the bootstrap, an upper bound after 2016-10-01.
The column list of each file is fixed here and holds no outcome column (no
train file is a source). TestUnlock is never constructed and score-test never
run; nothing here touches the label files.

Order (streaming.tie_rule): event date; source rank transactions,
transactions_v2, user_logs; msno; then every column of the row in file order,
ascending, nulls first. Events are numbered from 1 in that order.

Each transaction row also carries source_row, its 0-based row number in its
interim file (DuckDB's file_row_number), as Phase 1's cleaning numbered it, so
the move can drop identical rows and break the timeline's last ties exactly as
Phase 1 did. source_row comes last in the sort, so it only orders rows that are
equal in every column: the registered order is unchanged (D-215, found while
planning the move, before any result).

The bootstrap reads user_logs without ORDER BY (D-225, a fix found after a result: the
bucket-2 dry run measured DuckDB holding about 182 bytes per row to sort the logs). The
listening state is order-independent: day totals are integer sums, first and last days a
minimum and a maximum, the invalid flag a set. Both transaction files (appended to lists)
and every October event keep the registered order.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from retention.api.config import load_serving
from retention.data.sample import bucket_sql
from retention.data.stream import USER_LOGS_TYPES
from retention.models.resources import current_rss_gb, peak_rss_gb

CFG = load_serving()
STORE_DATE = date.fromisoformat(CFG["streaming"]["replay"]["store_date_after"])  # 2016-11-01
FIRST_DAY = date.fromisoformat(CFG["streaming"]["replay"]["first_day"])  # 2016-10-01
BOOTSTRAP = date.fromisoformat(CFG["streaming"]["bootstrap"]["snapshot"])  # 2016-10-01

TRANSACTION_COLUMNS = (
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
)
LOG_COLUMNS = tuple(USER_LOGS_TYPES)  # the interim user_logs columns, in file order
UNSORTED_BOOTSTRAP = frozenset({"user_logs"})  # read without ORDER BY at the bootstrap (D-225)


@dataclass(frozen=True)
class Source:
    name: str
    rank: int  # the tie rule's source rank
    path: Path
    date_column: str
    columns: tuple[str, ...]  # file order
    row_number: bool = False  # add source_row (the transaction files)


def sources(folder: Path, cfg: dict[str, Any] | None = None) -> list[Source]:
    """The registered sources, in rank order, with their fixed column lists."""
    names = [Path(p).stem for p in (cfg or CFG)["streaming"]["sources"]]
    if names != ["transactions", "transactions_v2", "user_logs"]:
        raise ValueError(f"the registered sources changed: {names}")
    return [
        Source(
            "transactions",
            0,
            folder / "transactions.parquet",
            "transaction_date",
            TRANSACTION_COLUMNS,
            row_number=True,
        ),
        Source(
            "transactions_v2",
            1,
            folder / "transactions_v2.parquet",
            "transaction_date",
            TRANSACTION_COLUMNS,
            row_number=True,
        ),
        Source("user_logs", 2, folder / "user_logs.parquet", "date", LOG_COLUMNS),
    ]


def day_number(d: date) -> int:
    """A date as the files store it: YYYYMMDD."""
    return d.year * 10_000 + d.month * 100 + d.day


def check_bounds(lo: date | None, hi: date, ceiling: date = STORE_DATE) -> None:
    if hi > ceiling:
        raise PermissionError(f"no row dated {ceiling} or later may be read (asked up to {hi})")
    if lo is not None and lo >= hi:
        raise ValueError(f"empty date range [{lo}, {hi})")


def fields(src: Source) -> tuple[str, ...]:
    """The names of a read row: the file's columns, then source_row for transactions."""
    return (*src.columns, "source_row") if src.row_number else src.columns


Buckets = int | tuple[int, ...] | None  # one md5 bucket, or the buckets of a dry run (D-227)


def bucket_where(bucket: Buckets, column: str = "msno") -> str:
    """The SQL test for one bucket (= k) or several (in (j, k)), by the sample's bucket rule."""
    ks = (bucket,) if isinstance(bucket, int) else tuple(bucket)
    if len(ks) == 1:
        return f"{bucket_sql(column)} = {int(ks[0])}"
    return f"{bucket_sql(column)} in ({', '.join(str(int(k)) for k in ks)})"


def query(
    src: Source, lo: date | None, hi: date, bucket: Buckets = None, ordered: bool = True
) -> str:
    """The one SQL statement a read runs: the window [lo, hi) of one file, in tie order
    (unordered only for the bootstrap's user_logs, D-225); with `bucket`, only that md5
    bucket's customers (the sample's own bucket rule)."""
    cols = ", ".join(src.columns)
    where = f"{src.date_column} < {day_number(hi)}"
    if lo is not None:
        where = f"{src.date_column} >= {day_number(lo)} and {where}"
    if bucket is not None:
        where = f"{where} and {bucket_where(bucket)}"
    order = [src.date_column, "msno", *(f"{c} asc nulls first" for c in src.columns if c != "msno")]
    reader = f"read_parquet('{src.path}')"
    if src.row_number:
        cols += ", file_row_number as source_row"
        reader = f"read_parquet('{src.path}', file_row_number = true)"
        order.append("source_row")
    sql = f"select {cols} from {reader} where {where}"
    return f"{sql} order by {', '.join(order)}" if ordered else sql


def duckdb_memory(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """DuckDB's own memory in use and temporary storage, in bytes (duckdb_memory())."""
    used, temp = (
        con.cursor()
        .execute(
            "select coalesce(sum(memory_usage_bytes), 0), "
            "coalesce(sum(temporary_storage_bytes), 0) from duckdb_memory()"
        )
        .fetchone()
    )
    return int(used), int(temp)


def run(sql: str, batch_rows: int, stats: dict | None = None) -> Iterator[list[tuple]]:
    """Run one read and yield its rows in batches (tests wrap this as the spy).

    With `stats` (a measurement only, D-223): seconds spent fetching; after the first batch
    (for a sorted read, DuckDB has sorted every row by then) the seconds, DuckDB's memory and
    temporary storage, the process's running peak RSS and its current RSS (D-227); DuckDB's
    largest memory and temporary storage."""
    con = duckdb.connect()
    try:
        t0 = time.perf_counter()
        cur = con.execute(sql)
        first = True
        while True:
            t1 = time.perf_counter()
            rows = cur.fetchmany(batch_rows)
            if stats is not None:
                stats["fetch_seconds"] = stats.get("fetch_seconds", 0.0) + time.perf_counter() - t1
                used, temp = duckdb_memory(con)
                stats["duckdb_memory_bytes_max"] = max(
                    stats.get("duckdb_memory_bytes_max", 0), used
                )
                stats["duckdb_temporary_bytes_max"] = max(
                    stats.get("duckdb_temporary_bytes_max", 0), temp
                )
                if first:
                    stats["first_batch_seconds"] = round(time.perf_counter() - t0, 3)
                    stats["duckdb_memory_bytes_after_first_batch"] = used
                    stats["peak_rss_gb_after_first_batch"] = peak_rss_gb()
                    stats["rss_gb_after_first_batch"] = current_rss_gb()
                    first = False
            if not rows:
                return
            yield rows
    finally:
        con.close()


def read(
    src: Source,
    lo: date | None,
    hi: date,
    ceiling: date = STORE_DATE,
    batch_rows: int = 50_000,
    bucket: Buckets = None,
    stats: dict | None = None,
    ordered: bool = True,
) -> Iterator[dict[str, Any]]:
    """One file's rows in [lo, hi), in tie order unless `ordered` is False; refused before
    any query if hi > ceiling."""
    check_bounds(lo, hi, ceiling)
    date_at = src.columns.index(src.date_column)
    sql = query(src, lo, hi, bucket, ordered)
    batches = run(sql, batch_rows) if stats is None else run(sql, batch_rows, stats)
    for rows in batches:
        for row in rows:
            if row[date_at] >= day_number(ceiling):  # a second guard on what came back
                raise PermissionError(f"a row dated {row[date_at]} came back")
            yield dict(zip(fields(src), row, strict=True))


def sort_key(src: Source, row: dict[str, Any]) -> tuple:
    """The tie rule: date, source rank, msno, then every column in file order, nulls first."""
    rest = tuple((v is not None, v) for c, v in row.items() if c != "msno")
    return (row[src.date_column], src.rank, row["msno"], rest)


def keyed(
    src: Source, lo: date, hi: date, ceiling: date, bucket: Buckets = None
) -> Iterator[tuple]:
    """One file's rows with their tie-rule keys (its own function, so each stream keeps its
    own source)."""
    for row in read(src, lo, hi, ceiling, bucket=bucket):
        yield sort_key(src, row), src.name, row


LISTENING_COUNTS = ("num_25", "num_50", "num_75", "num_985", "num_100", "num_unq")


def halve_listening(row: dict[str, Any]) -> dict[str, Any]:
    """The registered drift shift (configs/ops.yaml, drift.shift.rule): total_secs times 0.5,
    and each count halved and rounded down. A missing value stays missing; nothing else
    changes."""
    out = dict(row)
    if out.get("total_secs") is not None:
        out["total_secs"] = out["total_secs"] * 0.5
    for c in LISTENING_COUNTS:
        if out.get(c) is not None:
            out[c] = int(out[c]) // 2
    return out


def events(
    folder: Path,
    lo: date = FIRST_DAY,
    hi: date = STORE_DATE,
    ceiling: date = STORE_DATE,
    bucket: Buckets = None,
    shift: bool = False,
) -> Iterator[dict[str, Any]]:
    """The replay's events, numbered from 1 in the registered order.

    With `shift`, the listening events (the user_logs rows) are halved after the order and the
    numbers are fixed, so the event count, the order and the numbering do not change; transaction
    events are never touched (configs/ops.yaml, drift.shift)."""
    streams = [keyed(s, lo, hi, ceiling, bucket) for s in sources(folder)]
    for seq, (_, name, row) in enumerate(heapq.merge(*streams, key=lambda x: x[0]), start=1):
        if shift and name == "user_logs":
            row = halve_listening(row)
        yield {"seq": seq, "source": name, **row}


def bootstrap_rows(
    folder: Path, source: str, bucket: Buckets = None, stats: dict | None = None
) -> Iterator[dict[str, Any]]:
    """One file's rows dated before the bootstrap snapshot (2016-10-01): the transaction
    files in tie order, user_logs in DuckDB's own order (D-225)."""
    src = next(s for s in sources(folder) if s.name == source)
    ordered = source not in UNSORTED_BOOTSTRAP
    return read(src, None, BOOTSTRAP, BOOTSTRAP, bucket=bucket, stats=stats, ordered=ordered)
