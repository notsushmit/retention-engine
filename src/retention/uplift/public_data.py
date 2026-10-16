"""Checks on the two randomized datasets before any model (configs/uplift.yaml, validation).

Each check stops on the first mismatch. The file's SHA-256 is checked before it is
parsed. The checks write nothing; Parquet is written by `to_parquet`, which the
command calls only after every check of both files has passed. Neither raw
file is changed or unpacked to disk: the Criteo file is read from its gzip stream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb


class CheckFailed(RuntimeError):
    """A public dataset does not match its registered description; nothing is written."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("set threads = 4; set memory_limit = '8GB'")
    return con


def _source(path: Path) -> str:
    return f"read_csv('{path}', header = true)"


def _check_hash(path: Path, expected: str) -> str:
    got = sha256(path)
    if got != expected:
        raise CheckFailed(f"{path.name}: SHA-256 {got} differs from the registered {expected}")
    return got


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [r[0] for r in con.sql(f"describe select * from {_source(path)}").fetchall()]


def check_hillstrom(
    path: Path,
    sha: str,
    rows: int,
    columns: list[str],
    arms: list[str],
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"file": str(path), "sha256": _check_hash(path, sha)}
    con = _connect()
    cols = _columns(con, path)
    result["columns"] = cols
    if sorted(c.lower() for c in cols) != sorted(columns):
        raise CheckFailed(f"hillstrom: columns {cols} differ from the registered {columns}")
    seg = next(c for c in cols if c.lower() == "segment")
    n = con.sql(f"select count(*) from {_source(path)}").fetchone()[0]
    result["rows"] = n
    if n != rows:
        raise CheckFailed(f"hillstrom: {n} rows, registered {rows}")
    counts = dict(con.sql(f'select "{seg}", count(*) from {_source(path)} group by 1').fetchall())
    result["arms"] = counts
    if sorted(counts) != sorted(arms):
        raise CheckFailed(f"hillstrom: arms {sorted(counts)} differ from the registered {arms}")
    shares = {k: v / n for k, v in counts.items()}
    result["arm_shares"] = shares
    off = {k: s for k, s in shares.items() if abs(s - 1 / 3) > tolerance}
    if off:
        raise CheckFailed(f"hillstrom: arm shares {off} outside 1/3 +/- {tolerance}")
    return result


def check_criteo(
    path: Path,
    sha: str,
    rows: int,
    columns: list[str],
    share: float,
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"file": str(path), "sha256": _check_hash(path, sha)}
    con = _connect()
    cols = _columns(con, path)
    result["columns"] = cols
    if sorted(cols) != sorted(columns):
        raise CheckFailed(f"criteo: columns {cols} differ from the registered {columns}")
    n, treated = con.sql(f"select count(*), avg(treatment) from {_source(path)}").fetchone()
    result["rows"] = n
    result["treatment_share"] = treated
    if n != rows:
        raise CheckFailed(f"criteo: {n} rows, registered {rows}")
    if abs(treated - share) > tolerance:
        raise CheckFailed(f"criteo: treatment share {treated:.5f} outside {share} +/- {tolerance}")
    result["row_count_note"] = (
        f"the downloaded file has {n:,} rows; the mirror card's '25M rows' does not describe it"
    )
    return result


def to_parquet(path: Path, out: Path) -> str:
    """Copy a checked CSV (plain or gzip) to Parquet."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _connect().sql(f"copy (select * from {_source(path)}) to '{out}' (format parquet)")
    return str(out)
