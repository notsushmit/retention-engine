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
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from retention.api.config import load_serving
from retention.data.stream import USER_LOGS_TYPES

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


@dataclass(frozen=True)
class Source:
    name: str
    rank: int  # the tie rule's source rank
    path: Path
    date_column: str
    columns: tuple[str, ...]  # file order


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
        ),
        Source(
            "transactions_v2",
            1,
            folder / "transactions_v2.parquet",
            "transaction_date",
            TRANSACTION_COLUMNS,
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


def query(src: Source, lo: date | None, hi: date) -> str:
    """The one SQL statement a read runs: the window [lo, hi) of one file, in tie order."""
    cols = ", ".join(src.columns)
    where = f"{src.date_column} < {day_number(hi)}"
    if lo is not None:
        where = f"{src.date_column} >= {day_number(lo)} and {where}"
    order = ", ".join(
        [src.date_column, "msno", *(f"{c} asc nulls first" for c in src.columns if c != "msno")]
    )
    return f"select {cols} from read_parquet('{src.path}') where {where} order by {order}"


def run(sql: str, batch_rows: int) -> Iterator[list[tuple]]:
    """Run one read and yield its rows in batches (tests wrap this as the spy)."""
    con = duckdb.connect()
    try:
        cur = con.execute(sql)
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                return
            yield rows
    finally:
        con.close()


def read(
    src: Source, lo: date | None, hi: date, ceiling: date = STORE_DATE, batch_rows: int = 50_000
) -> Iterator[dict[str, Any]]:
    """One file's rows in [lo, hi), in tie order; refused before any query if hi > ceiling."""
    check_bounds(lo, hi, ceiling)
    date_at = src.columns.index(src.date_column)
    for rows in run(query(src, lo, hi), batch_rows):
        for row in rows:
            if row[date_at] >= day_number(ceiling):  # a second guard on what came back
                raise PermissionError(f"a row dated {row[date_at]} came back")
            yield dict(zip(src.columns, row, strict=True))


def sort_key(src: Source, row: dict[str, Any]) -> tuple:
    """The tie rule: date, source rank, msno, then every column in file order, nulls first."""
    rest = tuple((v is not None, v) for c, v in row.items() if c != "msno")
    return (row[src.date_column], src.rank, row["msno"], rest)


def keyed(src: Source, lo: date, hi: date, ceiling: date) -> Iterator[tuple]:
    """One file's rows with their tie-rule keys (its own function, so each stream keeps its
    own source)."""
    for row in read(src, lo, hi, ceiling):
        yield sort_key(src, row), src.name, row


def events(
    folder: Path, lo: date = FIRST_DAY, hi: date = STORE_DATE, ceiling: date = STORE_DATE
) -> Iterator[dict[str, Any]]:
    """The replay's events, numbered from 1 in the registered order."""
    streams = [keyed(s, lo, hi, ceiling) for s in sources(folder)]
    for seq, (_, name, row) in enumerate(heapq.merge(*streams, key=lambda x: x[0]), start=1):
        yield {"seq": seq, "source": name, **row}


def bootstrap_rows(folder: Path, source: str) -> Iterator[dict[str, Any]]:
    """One file's rows dated before the bootstrap snapshot (2016-10-01), in tie order."""
    src = next(s for s in sources(folder) if s.name == source)
    return read(src, None, BOOTSTRAP, ceiling=BOOTSTRAP)
