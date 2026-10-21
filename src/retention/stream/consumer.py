"""The replay's consumer: per-customer running state in the replay database.

configs/serving.yaml, streaming.consumer and step6. Per event:
- a listening row adds to that day's totals in the customer's 28-day ring
  (milliseconds of listening, an invalid-seconds flag, plays, full plays, unique
  songs), and sets the first and last listening day;
- a transaction row is appended to the customer's transaction rows, as TX_STORED
  (the transaction columns without msno, then source v1 or v2 and source_row);
- the customer joins the touched set (customers with an event since the last
  store-date move);
- a cancel row (is_cancel = 1) from a customer scored at the store's date is
  appended to the cancel events as {msno, date} (step6.cancel_events).

Each row's seconds are cleaned exactly as Phase 2's UsageSource cleans them
(retention.features.usage): negative or sentinel values (|v| >= 1e15) count as 0
seconds and flag the day; values above 86,400 are capped at it and flag the day;
the value is then rounded to whole milliseconds half away from zero, as DuckDB's
round does, and the day's totals are integer sums, so the state does not depend
on the order of rows within a day. A missing or non-finite value in a float
column is refused: Phase 2's SQL would read NaN as larger than every number and
NULL as a full capped day, so neither can be reproduced row by row on purpose.

The ring keeps only days that can still fall in a 28-day window: days at least
28 days before the store's date are never written, and the move (a later
commit) drops the days that leave the window. Events must arrive in date order
(the source's order); an older date, or a date on or after 2016-11-01, is
refused.

The bootstrap reads rows dated before 2016-10-01 only through
source.bootstrap_rows, and writes the same state (no touched set, no cancel
events): the first and last listening day for everyone, the ring days from 28
days before the bootstrap snapshot, and every transaction row.

Every write goes to the replay database: the consumer asks the server which
database it is on (producer.require_database) and refuses database 0.

Key names (rp:...) are chosen here, not registered (docs plan, D-213).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from retention.api import store
from retention.features.spec import SECS_PER_DAY_MAX, SECS_SENTINEL_ABS, WINDOWS
from retention.models.resources import current_rss_gb, peak_rss_gb
from retention.stream import producer, source

RING_DAYS = WINDOWS["28"][1]  # 28
FIRST = "rp:first"  # hash msno -> first listening day (yyyymmdd)
LAST = "rp:last"  # hash msno -> last listening day
TOUCHED = "rp:touched"  # set of msno with an event since the last move
CURSOR = "rp:cursor"  # the last stream entry consumed
DAY_FIELDS = ("ms", "x", "p", "f", "u")  # milliseconds, invalid flag, plays, full plays, unique
LOG_FLOATS = ("total_secs",)
LOG_COUNTS = ("num_25", "num_50", "num_75", "num_985", "num_100", "num_unq")


# A stored transaction row: the transaction columns without msno, then Phase 1's source
# tag (v1 or v2) and the row's number in its interim file (D-215).
TX_STORED = (*(c for c in source.TRANSACTION_COLUMNS if c != "msno"), "source", "source_row")
SOURCE_TAG = {"transactions": "v1", "transactions_v2": "v2"}


def ring_key(msno: str) -> str:
    return f"rp:day:{msno}"


def tx_key(msno: str) -> str:
    return f"rp:tx:{msno}"


def as_date(day: int) -> date:
    return date(day // 10_000, day // 100 % 100, day % 100)


def round_half_away(v: float) -> int:
    """DuckDB's round for a non-negative double: halves go up. Exact (no v + 0.5)."""
    floor = math.floor(v)
    return floor + (1 if v - floor >= 0.5 else 0)


def clean_secs(total_secs: Any) -> tuple[int, bool]:
    """(milliseconds, invalid) of one row, as Phase 2 cleans it; refuses missing or NaN."""
    if total_secs is None or not math.isfinite(total_secs):
        raise ValueError(f"a source float column holds {total_secs!r}; refused (D-213)")
    invalid = total_secs < 0 or total_secs > SECS_PER_DAY_MAX
    if total_secs < 0 or abs(total_secs) >= SECS_SENTINEL_ABS:
        return 0, invalid
    return round_half_away(min(total_secs, SECS_PER_DAY_MAX) * 1000), invalid


def day_totals(row: dict[str, Any]) -> dict[str, int]:
    for c in LOG_COUNTS:
        if row[c] is None:
            raise ValueError(f"a source count column {c} is missing; refused (D-213)")
    ms, invalid = clean_secs(row["total_secs"])
    return {
        "ms": ms,
        "x": int(invalid),
        "p": row["num_25"] + row["num_50"] + row["num_75"] + row["num_985"] + row["num_100"],
        "f": row["num_100"],
        "u": row["num_unq"],
    }


class Consumer:
    """The running state of one replay, in one Valkey database (never 0)."""

    def __init__(
        self,
        r,
        database: int,
        store_date: date,
        is_scored: Callable[[str], bool],
        keys: store.Keys = store.KEYS,
    ):
        producer.refuse_loaded_store(r)
        producer.require_database(r, database)
        self.r, self.store_date, self.is_scored, self.keys = r, store_date, is_scored, keys
        self.ring_floor = source.day_number(store_date - timedelta(days=RING_DAYS))
        self.latest = 0

    # --- one event ---------------------------------------------------------------------------

    def _check_date(self, day: int) -> None:
        if day >= source.day_number(source.STORE_DATE):
            raise PermissionError(f"an event dated {day} reached the consumer")
        if day < self.latest:
            raise ValueError(f"an event dated {day} after one dated {self.latest}")
        self.latest = day

    def _log(self, pipe, row: dict[str, Any]) -> None:
        m, day = row["msno"], row["date"]
        totals = day_totals(row)
        pipe.hsetnx(FIRST, m, day)
        pipe.hset(LAST, m, day)
        if day >= self.ring_floor:
            for f in ("ms", "p", "f", "u"):
                pipe.hincrby(ring_key(m), f"{day}:{f}", totals[f])
            if totals["x"]:
                pipe.hset(ring_key(m), f"{day}:x", 1)

    def _tx(self, pipe, row: dict[str, Any], name: str) -> None:
        values = [row[c] for c in source.TRANSACTION_COLUMNS if c != "msno"]
        values += [SOURCE_TAG[name], row["source_row"]]
        pipe.rpush(tx_key(row["msno"]), json.dumps(values, separators=(",", ":")))

    def apply(self, events: Iterable[dict[str, Any]], batch: int = 1_000) -> int:
        """Apply replay events in order; returns how many were applied."""
        pipe = self.r.pipeline(transaction=False)
        n = 0
        for e in events:
            is_log = e["source"] == "user_logs"
            day = e["date"] if is_log else e["transaction_date"]
            self._check_date(day)
            if is_log:
                self._log(pipe, e)
            else:
                self._tx(pipe, e, e["source"])
                if e["is_cancel"] == 1 and self.is_scored(e["msno"]):
                    event = {"msno": e["msno"], "date": as_date(day).isoformat()}
                    pipe.rpush(self.keys.events, json.dumps(event, separators=(",", ":")))
            pipe.sadd(TOUCHED, e["msno"])
            n += 1
            if n % batch == 0:
                pipe.incrby(self.keys.replay[1], batch)
                pipe.execute()
        pipe.incrby(self.keys.replay[1], n % batch)
        pipe.execute()
        return n

    def consume(self, stream: str = producer.STREAM, count: int = 1_000) -> int:
        """Read the stream after the cursor and apply every entry, in stream order."""
        done = 0
        while True:
            after = self.r.get(CURSOR)
            start = "-" if after is None else "(" + after.decode()
            entries = self.r.xrange(stream, min=start, count=count)
            if not entries:
                return done
            events = [json.loads(fields[b"e"]) for _, fields in entries]
            if done == 0 and after is None and events[0]["seq"] != 1:
                raise ValueError("the stream does not start at event 1")
            done += self.apply(events)
            self.r.set(CURSOR, entries[-1][0])

    # --- the bootstrap -----------------------------------------------------------------------

    def bootstrap(
        self,
        folder: Path,
        batch: int = 5_000,
        bucket: source.Buckets = None,
        measure: dict | None = None,
    ) -> dict[str, int]:
        """State from rows dated before 2016-10-01, read only through source.bootstrap_rows.

        With `measure` (a measurement only, D-223), per file: rows; seconds reading (DuckDB's
        fetches), updating the state (everything else in the loop) and writing (pipeline
        executions); the read stats of source.run (DuckDB's memory, the first batch, which
        ends DuckDB's sort on a sorted read); and the process's running peak RSS and current
        RSS (D-227) after the file. user_logs is read unsorted (D-225); the state does not
        depend on its order. Then the first and last listening days' writes, with their
        entries and the peak and current RSS after them."""
        counts = {"user_logs": 0, "transactions": 0, "transactions_v2": 0}
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        pipe = self.r.pipeline(transaction=False)
        write = {"user_logs": 0.0, "transactions": 0.0, "transactions_v2": 0.0}
        stats = {name: {} for name in counts} if measure is not None else None

        def rows_of(name: str):
            if stats is None:
                return source.bootstrap_rows(folder, name, bucket)
            return source.bootstrap_rows(folder, name, bucket, stats=stats[name])

        def flush(name: str) -> None:
            t = time.perf_counter()
            pipe.execute()
            write[name] += time.perf_counter() - t

        loop: dict[str, float] = {}
        t0 = time.perf_counter()
        for row in rows_of("user_logs"):
            m, day = row["msno"], row["date"]
            totals = day_totals(row)  # refuses a missing or NaN value here too
            first[m] = min(first.get(m, day), day)
            last[m] = max(last.get(m, day), day)
            if day >= self.ring_floor:
                for f in ("ms", "p", "f", "u"):
                    pipe.hincrby(ring_key(m), f"{day}:{f}", totals[f])
                if totals["x"]:
                    pipe.hset(ring_key(m), f"{day}:x", 1)
            counts["user_logs"] += 1
            if counts["user_logs"] % batch == 0:
                flush("user_logs")
        flush("user_logs")
        loop["user_logs"] = time.perf_counter() - t0
        peak_after = {"user_logs": peak_rss_gb()}
        rss_after = {"user_logs": current_rss_gb()} if measure is not None else {}
        for name in ("transactions", "transactions_v2"):
            t0 = time.perf_counter()
            for row in rows_of(name):
                self._tx(pipe, row, name)
                counts[name] += 1
                if counts[name] % batch == 0:
                    flush(name)
            flush(name)
            loop[name] = time.perf_counter() - t0
            peak_after[name] = peak_rss_gb()
            if measure is not None:
                rss_after[name] = current_rss_gb()
        t0 = time.perf_counter()
        items = sorted(first)
        for i in range(0, len(items), batch):
            part = items[i : i + batch]
            pipe.hset(FIRST, mapping={m: first[m] for m in part})
            pipe.hset(LAST, mapping={m: last[m] for m in part})
            pipe.execute()
        pipe.execute()
        if measure is not None:
            for name in counts:
                read = stats[name].get("fetch_seconds", 0.0)
                measure[name] = {
                    "rows": counts[name],
                    "read_seconds": round(read, 3),
                    "state_seconds": round(loop[name] - read - write[name], 3),
                    "write_seconds": round(write[name], 3),
                    **{k: v for k, v in stats[name].items() if k != "fetch_seconds"},
                    "peak_rss_gb_after": peak_after[name],
                    "rss_gb_after": rss_after[name],
                }
            measure["first_last_writes"] = {
                "entries": len(items),
                "seconds": round(time.perf_counter() - t0, 3),
                "peak_rss_gb_after": peak_rss_gb(),
                "rss_gb_after": current_rss_gb(),
            }
        return counts

    # --- reading the state (the move and the tests) ------------------------------------------

    def window(self, msno: str, t: date) -> dict[int, dict[str, int]]:
        """The ring's days with offset 1 to 28 before t: {offset: totals}."""
        raw = self.r.hgetall(ring_key(msno))
        out: dict[int, dict[str, int]] = {}
        for k, v in raw.items():
            day, f = k.decode().split(":")
            offset = (t - as_date(int(day))).days
            if 1 <= offset <= RING_DAYS:
                out.setdefault(offset, dict.fromkeys(DAY_FIELDS, 0))[f] = int(v)
        return out

    def first_last(self, msno: str) -> tuple[int | None, int | None]:
        first, last = self.r.hget(FIRST, msno), self.r.hget(LAST, msno)
        return (None if first is None else int(first)), (None if last is None else int(last))

    def transactions(self, msno: str) -> list[list[Any]]:
        return [json.loads(v) for v in self.r.lrange(tx_key(msno), 0, -1)]
