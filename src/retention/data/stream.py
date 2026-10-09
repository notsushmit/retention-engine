"""Stream a CSV out of a .7z archive straight into a sampled Parquet file.

Used for user_logs.csv, which is about 30 GB unpacked and must never be written
to disk as CSV. py7zr decompresses into an OS pipe; a reader thread parses the
pipe with pyarrow's streaming CSV reader, keeps sampled customers with the same
md5 bucket rule as sample.py, and appends each block to a Parquet file.

A stream cannot be scanned twice to guess types, so columns get explicit types.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import py7zr
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from py7zr.io import Py7zIO, WriterFactory

from retention.data.sample import bucket_sql, to_percent

USER_LOGS_TYPES: dict[str, pa.DataType] = {
    "msno": pa.string(),
    "date": pa.int32(),
    "num_25": pa.int64(),
    "num_50": pa.int64(),
    "num_75": pa.int64(),
    "num_985": pa.int64(),
    "num_100": pa.int64(),
    "num_unq": pa.int64(),
    "total_secs": pa.float64(),
}

BLOCK_SIZE = 64 << 20


@dataclass
class StreamStats:
    """Counts over the whole source file (before sampling)."""

    source_rows: int = 0
    source_customers: int = 0
    kept_rows: int = 0
    date_min: int | None = None
    date_max: int | None = None
    uncompressed_bytes: int = 0
    seconds: float = 0.0


class _PipeWriter(Py7zIO):
    """Write-only sink: hands decompressed bytes to the pipe, ignores everything else."""

    def __init__(self, fd: int, stats: StreamStats) -> None:
        self._fd = fd
        self._stats = stats

    def write(self, s: bytes | bytearray) -> int:
        view = memoryview(s)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self._stats.uncompressed_bytes += len(s)
        return len(s)

    def read(self, size: int | None = None) -> bytes:
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self._stats.uncompressed_bytes


class _NullWriter(_PipeWriter):
    def __init__(self) -> None:
        pass

    def write(self, s: bytes | bytearray) -> int:
        return len(s)

    def size(self) -> int:
        return 0


class _Factory(WriterFactory):
    def __init__(self, member: str, fd: int, stats: StreamStats) -> None:
        self._member, self._fd, self._stats = member, fd, stats

    def create(self, filename: str) -> Py7zIO:
        if Path(filename).name == self._member:
            return _PipeWriter(self._fd, self._stats)
        return _NullWriter()


def _consume(
    read_fd: int,
    out_tmp: Path,
    percent: int,
    types: dict[str, pa.DataType],
    stats: StreamStats,
    errors: list[BaseException],
) -> None:
    try:
        with os.fdopen(read_fd, "rb") as source:
            reader = pacsv.open_csv(
                source,
                read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
                convert_options=pacsv.ConvertOptions(column_types=types),
            )
            schema = pa.schema([(name, types[name]) for name in reader.schema.names])
            con = duckdb.connect()
            customers: set[str] = set()  # a few million IDs, a few hundred MB at most
            keep = f"SELECT * FROM batch WHERE {bucket_sql()} < {percent}"
            with pq.ParquetWriter(out_tmp, schema, compression="zstd") as writer:
                for batch in reader:
                    stats.source_rows += batch.num_rows
                    customers.update(pc.unique(batch.column("msno")).to_pylist())
                    if "date" in batch.schema.names and batch.num_rows:
                        mm = pc.min_max(batch.column("date")).as_py()
                        lo, hi = mm["min"], mm["max"]
                        stats.date_min = lo if stats.date_min is None else min(stats.date_min, lo)
                        stats.date_max = hi if stats.date_max is None else max(stats.date_max, hi)
                    con.register("batch", pa.Table.from_batches([batch]))
                    kept = con.execute(keep).to_arrow_table().cast(schema)
                    con.unregister("batch")
                    stats.kept_rows += kept.num_rows
                    writer.write_table(kept)
            stats.source_customers = len(customers)
    except BaseException as exc:  # surfaced to the caller after join
        errors.append(exc)


def stream_sample(
    archive: Path,
    member: str,
    out_path: Path,
    fraction: float,
    types: dict[str, pa.DataType] = USER_LOGS_TYPES,
) -> StreamStats:
    """Decompress `member` from `archive` and write its sampled rows to `out_path`.

    Nothing but the Parquet output touches disk. The output appears only on success.
    """
    percent = to_percent(fraction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = out_path.with_suffix(".parquet.tmp")
    stats = StreamStats()
    errors: list[BaseException] = []
    read_fd, write_fd = os.pipe()
    consumer = threading.Thread(
        target=_consume, args=(read_fd, out_tmp, percent, types, stats, errors), daemon=True
    )
    start = time.monotonic()
    consumer.start()
    producer_error: BaseException | None = None
    try:
        with py7zr.SevenZipFile(archive, mode="r") as z:
            names = [Path(n).name for n in z.getnames()]
            if member not in names:
                raise FileNotFoundError(f"{member} not in {archive.name}: {names}")
            z.extractall(factory=_Factory(member, write_fd, stats))
    except BaseException as exc:  # includes BrokenPipeError when the consumer stops early
        producer_error = exc
    finally:
        os.close(write_fd)
        consumer.join()
    stats.seconds = time.monotonic() - start
    if errors or producer_error is not None:
        out_tmp.unlink(missing_ok=True)
        cause = errors[0] if errors else producer_error
        raise RuntimeError(f"streaming {member} from {archive.name} failed") from cause
    out_tmp.replace(out_path)
    return stats


def compare_logs(streamed: Path, extracted: Path) -> dict[str, object]:
    """Compare two log Parquet files on row count and per-customer num_100 and total_secs sums."""
    con = duckdb.connect()
    counts = con.execute(
        "SELECT (SELECT count(*) FROM read_parquet(?)), (SELECT count(*) FROM read_parquet(?))",
        [str(streamed), str(extracted)],
    ).fetchone()
    assert counts is not None
    diff = con.execute(
        """
        WITH s AS (SELECT msno, sum(num_100) n100, sum(total_secs) secs
                   FROM read_parquet(?) GROUP BY msno),
             e AS (SELECT msno, sum(num_100) n100, sum(total_secs) secs
                   FROM read_parquet(?) GROUP BY msno)
        SELECT count(*) FILTER (s.msno IS NULL OR e.msno IS NULL) AS customers_only_in_one,
               count(*) FILTER (s.n100 IS DISTINCT FROM e.n100) AS num_100_mismatches,
               max(abs(s.secs - e.secs) / greatest(abs(e.secs), 1)) AS total_secs_max_rel_diff,
               count(*) AS customers
        FROM s FULL OUTER JOIN e USING (msno)
        """,
        [str(streamed), str(extracted)],
    ).fetchone()
    assert diff is not None
    return {
        "rows_streamed": counts[0],
        "rows_extracted": counts[1],
        "customers": diff[3],
        "customers_only_in_one": diff[0],
        "num_100_mismatches": diff[1],
        "total_secs_max_rel_diff": diff[2],
    }
