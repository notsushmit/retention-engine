"""Tests for the hash-based customer sample. Synthetic CSVs only, no Kaggle needed."""

from pathlib import Path

import duckdb
import pytest

from retention.data.sample import bucket, sample_all, to_percent

N_CUSTOMERS = 10_000


def _ids(n: int = N_CUSTOMERS) -> list[str]:
    return [f"user{i:05d}=" for i in range(n)]


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    d = tmp_path / "csv"
    d.mkdir()
    ids = _ids()
    members = ["msno,city"] + [f"{m},{i % 5}" for i, m in enumerate(ids)]
    # Two log rows per customer, so tables differ in shape.
    logs = ["msno,date,total_secs"] + [
        f"{m},2017030{day},{i * 1.5}" for i, m in enumerate(ids) for day in (1, 2)
    ]
    (d / "members.csv").write_text("\n".join(members) + "\n")
    (d / "logs.csv").write_text("\n".join(logs) + "\n")
    return d


TABLES = {"members": "members.csv", "logs": "logs.csv"}


def _customers(path: Path) -> set[str]:
    rows = duckdb.sql(f"SELECT DISTINCT msno FROM '{path}'").fetchall()
    return {r[0] for r in rows}


def test_same_customers_in_every_table(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    assert _customers(out / "members.parquet") == _customers(out / "logs.parquet")


def test_sql_bucket_matches_python_md5(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    expected = {m for m in _ids() if bucket(m) < 20}
    assert _customers(out / "members.parquet") == expected


def test_deterministic(csv_dir: Path, tmp_path: Path) -> None:
    a = sample_all(csv_dir, TABLES, tmp_path / "a", 0.2)
    b = sample_all(csv_dir, TABLES, tmp_path / "b", 0.2)
    assert _customers(a / "members.parquet") == _customers(b / "members.parquet")


def test_samples_are_nested(csv_dir: Path, tmp_path: Path) -> None:
    small = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    large = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.5)
    assert _customers(small / "members.parquet") < _customers(large / "members.parquet")


def test_full_fraction_keeps_every_row(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 1.0)
    n = duckdb.sql(f"SELECT count(*) FROM '{out / 'logs.parquet'}'").fetchone()
    assert n == (2 * N_CUSTOMERS,)


def test_kept_share_close_to_target(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    share = len(_customers(out / "members.parquet")) / N_CUSTOMERS
    # Binomial sd at n=10k, p=0.2 is 0.004; 0.02 is five sd.
    assert abs(share - 0.2) < 0.02


def test_msno_kept_as_text(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    types = duckdb.sql(f"DESCRIBE SELECT * FROM '{out / 'members.parquet'}'").fetchall()
    assert dict((r[0], r[1]) for r in types)["msno"] == "VARCHAR"


def test_manifest_written(csv_dir: Path, tmp_path: Path) -> None:
    out = sample_all(csv_dir, TABLES, tmp_path / "interim", 0.2)
    assert (out / "manifest.json").exists()
    assert out.name == "sample_020"


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 0.205])
def test_rejects_bad_fraction(bad: float) -> None:
    with pytest.raises(ValueError):
        to_percent(bad)
