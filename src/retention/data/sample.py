"""Hash-based customer sample of the raw CSVs, written to Parquet with DuckDB.

A customer is kept when bucket(msno) < percent, where bucket is the first
8 hex digits of md5(msno) mod 100. md5 is a fixed standard, so the same
customers are chosen in every table, on every run and in every DuckDB version,
and smaller samples are subsets of larger ones.

Column types are DuckDB's guesses, except msno, which is always VARCHAR.
Explicit dtypes and validation belong to Phase 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

N_BUCKETS = 100


def to_percent(fraction: float) -> int:
    """Turn a fraction like 0.2 into a whole percent; reject anything finer than 1%."""
    percent = round(fraction * N_BUCKETS)
    if not 0 < fraction <= 1 or abs(percent - fraction * N_BUCKETS) > 1e-9:
        raise ValueError(f"sample fraction must be a multiple of 0.01 in (0, 1], got {fraction}")
    return percent


def bucket(msno: str) -> int:
    """Python twin of the SQL bucket expression, used by tests and spot checks."""
    return int(hashlib.md5(msno.encode()).hexdigest()[:8], 16) % N_BUCKETS


def bucket_sql(column: str = "msno") -> str:
    return f"(('0x' || left(md5({column}), 8))::UBIGINT % {N_BUCKETS})"


def sample_dir(interim_dir: Path, fraction: float) -> Path:
    return interim_dir / f"sample_{to_percent(fraction):03d}"


def sample_table(
    con: duckdb.DuckDBPyConnection, csv_path: Path, out_path: Path, percent: int
) -> None:
    """Write the sampled rows of one CSV to Parquet via a temp file, so no partial output."""
    tmp_path = out_path.with_suffix(".parquet.tmp")
    source = "read_csv(?, header = true, types = {'msno': 'VARCHAR'})"
    con.execute(
        f"COPY (SELECT * FROM {source} WHERE {bucket_sql()} < {percent}) "
        f"TO '{tmp_path}' (FORMAT parquet, COMPRESSION zstd)",
        [str(csv_path)],
    )
    tmp_path.replace(out_path)


def describe(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> dict[str, object]:
    """Row count, customer count and min/max of date-like columns for the manifest."""
    rel = con.read_parquet(str(parquet_path))
    date_cols = [c for c in rel.columns if "date" in c or c.endswith("_time")]
    exprs = ["count(*)", "count(DISTINCT msno)"]
    for c in date_cols:
        exprs += [f"min({c})", f"max({c})"]
    row = rel.aggregate(", ".join(exprs)).fetchone()
    assert row is not None
    info: dict[str, object] = {"rows": row[0], "customers": row[1]}
    for i, c in enumerate(date_cols):
        info[c] = {"min": row[2 + 2 * i], "max": row[3 + 2 * i]}
    return info


def sample_all(
    csv_dir: Path,
    tables: dict[str, str],
    interim_dir: Path,
    fraction: float,
    streamed: dict[str, tuple[Path, str]] | None = None,
) -> Path:
    """Sample every available table and write manifest.json. Returns the sample directory.

    `tables` maps table name to a CSV in csv_dir. `streamed` maps table name to
    (archive, member) for files read straight out of a .7z without unpacking.
    Existing Parquet files are kept; missing sources are reported and skipped.
    Stream statistics from earlier runs are kept in the manifest.
    """
    from retention.data.stream import stream_sample  # avoids a circular import

    percent = to_percent(fraction)
    out_dir = sample_dir(interim_dir, fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    old = json.loads(manifest_path.read_text())["tables"] if manifest_path.exists() else {}
    tables_info: dict[str, dict[str, object]] = {}
    con = duckdb.connect()
    for table, csv_name in tables.items():
        csv_path = csv_dir / csv_name
        out_path = out_dir / f"{table}.parquet"
        if not csv_path.exists():
            print(f"skip {table}: {csv_path} not found")
            continue
        if out_path.exists():
            print(f"keep existing {out_path}")
        else:
            print(f"sampling {csv_name} at {percent}%")
            sample_table(con, csv_path, out_path, percent)
        tables_info[table] = describe(con, out_path)
    for table, (archive, member) in (streamed or {}).items():
        out_path = out_dir / f"{table}.parquet"
        source = old.get(table, {}).get("source")
        if out_path.exists():
            print(f"keep existing {out_path}")
        elif not archive.exists():
            print(f"skip {table}: {archive} not found")
            continue
        else:
            print(f"streaming {member} from {archive.name} at {percent}%")
            stats = stream_sample(archive, member, out_path, fraction)
            source = {
                "rows": stats.source_rows,
                "customers": stats.source_customers,
                "date": {"min": stats.date_min, "max": stats.date_max},
                "uncompressed_bytes": stats.uncompressed_bytes,
                "seconds": round(stats.seconds, 1),
            }
            print(f"streamed {stats.source_rows} rows in {stats.seconds:.0f}s")
        tables_info[table] = describe(con, out_path) | ({"source": source} if source else {})
    manifest = {"fraction": percent / N_BUCKETS, "tables": tables_info}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir
