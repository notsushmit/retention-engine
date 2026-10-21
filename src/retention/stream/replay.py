"""The replay's steps: the recount, the cancel-ID count, one replay, and the dry run's checks.

configs/serving.yaml, streaming and step6.

- recount: October's source rows, counted with the reader's own query and bounds
  (event_count: 3,727,472; any other number is not met).
- cancel_ids: counts only: October cancel rows, those from customers scored at
  2016-10-01, their distinct IDs, and how many IDs hold a never-used word as a whole
  word (the response scan's own function). No ID is returned.
- run: one replay into one Valkey database (never 0): the database's replay and store
  keys cleared; the bootstrap; the bootstrap timelines; the October events produced
  (full speed or throttled) and consumed; the move (touched-only splice); the ring
  pruned and the touched set cleared; the stream table written; the replayed store
  written through move.write_store. Every step timed; memory read before and after.
- project: the dry run's projection to the full sample, with the user's stop rules
  (D-210, D-212): a projected process peak above 10 GB stops; projected Valkey memory
  above 7,598,239,392 bytes (7,958,871,888 minus database 0's 360,632,496) stops;
  above 90% of it (6,838,415,452) is stop-and-ask. Valkey memory is judged for both
  replays together (databases 1 and 2 are kept until the report is read).
- link_check: counts only: the cleaned table's rows dated before T for the dry run's
  bucket (or two buckets, D-227) against the stream's cleaned rows, on (source,
  source_row) and content.
- Memory: after every step, the process's running peak (step_peak_rss_gb) and its current
  resident memory (step_rss_gb, D-227), which falls when a step gives memory back.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from retention.api import store
from retention.api.config import load_serving
from retention.api.schemas import never_used_hits
from retention.labels.config import LabelConfig
from retention.models.resources import current_rss_gb, peak_rss_gb
from retention.stream import consumer as cs
from retention.stream import move, producer, source

SERVING = load_serving()
EXPECTED_EVENTS = int(SERVING["streaming"]["pass_marks"]["event_count"]["expected"])
PEAK_GB_MAX = float(
    SERVING["step6"]["memory"]["stop_before_full_replay"]["projected_process_peak_gb_above"]
)
VALKEY_REGISTERED = int(
    SERVING["step6"]["memory"]["stop_before_full_replay"][
        "projected_valkey_used_memory_bytes_above"
    ]
)
DATABASE_0_BYTES = (
    360_632_496  # Valkey used memory with the 5f store (D-209), the user's rule (D-212)
)
VALKEY_LIMIT = VALKEY_REGISTERED - DATABASE_0_BYTES  # 7,598,239,392
VALKEY_ASK = int(VALKEY_LIMIT * 0.9)  # 6,838,415,452
LINK_COLUMNS = move.LINK_COLUMNS


# --- the recount ---------------------------------------------------------------------------------


def recount(folder: Path, expected: int = EXPECTED_EVENTS) -> dict[str, Any]:
    """October's rows per source, with the reader's own query (bounds, columns, order)."""
    counts = {}
    for s in source.sources(folder):
        source.check_bounds(source.FIRST_DAY, source.STORE_DATE)
        sql = source.query(s, source.FIRST_DAY, source.STORE_DATE)
        counts[s.name] = int(duckdb.sql(f"select count(*) from ({sql})").fetchone()[0])
    total = sum(counts.values())
    return {"counts": counts, "total": total, "expected": expected, "met": total == expected}


# --- the cancel-ID count -------------------------------------------------------------------------


def cancel_ids(folder: Path, scored: set[str]) -> dict[str, Any]:
    """Counts only: October cancel rows and the never-used words in their customers' IDs."""
    rows, ids = 0, set()
    for s in source.sources(folder)[:2]:
        for row in source.read(s, source.FIRST_DAY, source.STORE_DATE):
            if row["is_cancel"] == 1:
                rows += 1
                if row["msno"] in scored:
                    ids.add(row["msno"])
    words: dict[str, int] = {}
    with_hit = 0
    for m in ids:
        hits = {h.lower() for h in never_used_hits(m)}
        with_hit += bool(hits)
        for h in hits:
            words[h] = words.get(h, 0) + 1
    return {
        "october_cancel_rows": rows,
        "distinct_ids_scored_at_2016_10_01": len(ids),
        "ids_with_a_whole_word_hit": with_hit,
        "by_word": dict(sorted(words.items())),
    }


# --- one replay ----------------------------------------------------------------------------------


def clear(r) -> int:
    """Delete the replay's own keys (rp:...) and the store's (rs:...) in this database."""
    n = 0
    for pattern in ("rp:*", "rs:*"):
        batch = list(r.scan_iter(pattern, 10_000))
        for i in range(0, len(batch), 5_000):
            n += r.delete(*batch[i : i + 5_000])
    return n


def content_hash(table: pa.Table) -> str:
    """SHA-256 of the table's content in msno order (Arrow IPC), for the determinism mark."""
    ordered = table.sort_by("msno")
    sink = io.BytesIO()
    with ipc.new_stream(sink, ordered.schema) as w:
        w.write_table(ordered)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def run(
    r,
    database: int,
    folder: Path,
    cfg: LabelConfig,
    members_path: Path,
    is_scored: Callable[[str], bool],
    c01: dict,
    seed: int,
    columns: list[str],
    main: tuple[str, ...],
    table_path: Path,
    throttled: bool = False,
    bucket: source.Buckets = None,
    t: date = source.STORE_DATE,
    n_chunks: int = move.WORK_CHUNKS,
    keep_link: bool = False,
) -> tuple[dict[str, Any], move.MoveResult]:
    """One replay into `database`: its report (counts, times, memory; no customer ID) and
    the move's result. Steps are timed under step_seconds and the process's running peak
    memory after each step under step_peak_rss_gb, and its current memory under step_rss_gb
    (keys api.__main__.stamp never writes)."""
    producer.refuse_loaded_store(r)
    producer.require_database(r, database)
    rss_start = peak_rss_gb()
    rss_now_start = current_rss_gb()
    step_seconds: dict[str, float] = {}
    step_peak: dict[str, float] = {}
    step_rss: dict[str, float] = {}
    clock = time.time()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.time()
        step_seconds[name] = round(now - clock, 2)
        step_peak[name] = peak_rss_gb()
        step_rss[name] = current_rss_gb()
        clock = now

    work = table_path.with_name(table_path.stem + "_work")
    shutil.rmtree(work, ignore_errors=True)
    removed = clear(r)
    valkey_before = store.memory_used(r)
    lap("clear")
    c = cs.Consumer(r, database, source.BOOTSTRAP, is_scored)
    bootstrap_detail: dict[str, Any] = {}
    boot = c.bootstrap(folder, bucket=bucket, measure=bootstrap_detail)
    lap("bootstrap")
    move.write_bootstrap_timelines(r, cfg, work, n_chunks)
    lap("bootstrap_timelines")
    produced = producer.produce(r, source.events(folder, bucket=bucket), throttled=throttled)
    lap("produce")
    consumed = c.consume()
    lap("consume")
    touched = r.scard(cs.TOUCHED)
    result = move.chunked_move(r, t, cfg, members_path, work, main, n_chunks, keep_link=keep_link)
    lap("move")
    pruned = move.prune_ring(r, t)
    r.delete(cs.TOUCHED)
    move.write_table(result.table, table_path)
    lap("move_write_table")
    written = move.write_store(
        r, database, t, result.table, result.g, result.prices, result.known, c01, seed, columns,
        main,
    )  # fmt: skip
    lap("store")
    shutil.rmtree(work, ignore_errors=True)
    valkey_after = store.memory_used(r)
    known = written["known"]
    return {
        "database": database,
        "speed": "throttled" if throttled else "full",
        "bucket": list(bucket) if isinstance(bucket, tuple) else bucket,
        "chunks": n_chunks,
        "removed_keys_before": int(removed),
        "bootstrap_rows": boot,
        "produced": int(produced),
        "consumed": int(consumed),
        "touched_at_move": int(touched),
        "ring_days_pruned": int(pruned),
        "feature_rows": result.table.num_rows,
        "feature_columns": result.table.num_columns,
        "table_sha256": content_hash(result.table),
        "store": written,
        "step_seconds": step_seconds,
        "step_peak_rss_gb": step_peak,
        "step_rss_gb": step_rss,  # current memory after each step (D-227)
        "bootstrap_detail": bootstrap_detail,  # a measurement (D-223); stamp never writes it
        "process_peak_rss_gb_start": rss_start,
        "process_rss_gb_start": rss_now_start,
        "process_peak_rss_gb": peak_rss_gb(),
        "valkey_used_memory_bytes_before": int(valkey_before),
        "valkey_used_memory_bytes_after": int(valkey_after),
        "valkey_bytes_per_known_customer": (valkey_after - valkey_before) / known
        if known
        else None,
        "valkey_keys": int(r.dbsize()),
        "cleaned_rows": int(result.cleaned_rows),
    }, result


# --- the dry run's checks ------------------------------------------------------------------------


def project(dry: dict[str, Any], factor: float, replays: int = 2) -> dict[str, Any]:
    """The full-sample projection of a dry run, and the stop rules' verdict. The factor is
    the sample's buckets over the buckets replayed: 20 for one bucket, 10 for two (D-227)."""
    base = dry["process_peak_rss_gb_start"]
    peak = base + (dry["process_peak_rss_gb"] - base) * factor
    one = (dry["valkey_used_memory_bytes_after"] - dry["valkey_used_memory_bytes_before"]) * factor
    total = one * replays
    if peak > PEAK_GB_MAX or total > VALKEY_LIMIT:
        verdict = "stop"
    elif total > VALKEY_ASK:
        verdict = "stop_and_ask"
    else:
        verdict = "proceed"
    return {
        "factor": factor,
        "projected_process_peak_gb": round(peak, 3),
        "process_peak_gb_limit": PEAK_GB_MAX,
        "projected_valkey_bytes_one_replay": int(one),
        "projected_valkey_bytes_all_replays": int(total),
        "replays_kept_together": replays,
        "valkey_bytes_limit": VALKEY_LIMIT,
        "valkey_bytes_ask_above": VALKEY_ASK,
        "verdict": verdict,
    }


def link_check(
    cleaned: Path, stream_tx: pd.DataFrame, t: date, bucket: int | tuple[int, ...]
) -> dict[str, Any]:
    """Counts only: Phase 1's cleaned rows before t of the dry run's bucket or buckets
    against the stream's."""
    batch = duckdb.sql(
        f"select {', '.join(LINK_COLUMNS)} from read_parquet('{cleaned}') "
        f"where transaction_date < date '{t.isoformat()}' and {source.bucket_where(bucket)}"
    ).df()

    def rows(df: pd.DataFrame) -> set[tuple]:
        out = df[LINK_COLUMNS].copy()
        for c in ("transaction_date", "expiry_date"):
            out[c] = pd.to_datetime(out[c]).dt.date.astype(str)
        return set(map(tuple, out.astype(str).itertuples(index=False)))

    b, s = rows(batch), rows(stream_tx)
    return {
        "batch_rows": len(batch),
        "stream_rows": len(stream_tx),
        "equal_rows": len(b & s),
        "only_in_batch": len(b - s),
        "only_in_stream": len(s - b),
        "met": b == s,
    }


def json_ready(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str) + "\n"
