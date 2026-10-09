"""Streaming a CSV out of a .7z into sampled Parquet, with no CSV written to disk."""

from pathlib import Path

import duckdb
import py7zr
import pytest

from retention.data import stream
from retention.data.sample import sample_all
from retention.data.stream import compare_logs, stream_sample

N_CUSTOMERS = 3_000
HEADER = "msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs"


def _logs_csv(path: Path, bad_row: bool = False) -> None:
    rows = [HEADER]
    for i in range(N_CUSTOMERS):
        for day in (1, 2, 3):
            rows.append(
                f"user{i:05d}=,2017030{day},{i % 7},1,0,2,{i % 11 + day},5,{i * 1.25 + day}"
            )
    if bad_row:
        rows.append("userX=,20170304,1,1,1,1,not_a_number,1,1.0")
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, Path]:
    """A 7z with the log inside a folder plus an unrelated member, like the KKBox archives."""
    src = tmp_path / "src" / "data" / "churn_comp_refresh"
    src.mkdir(parents=True)
    csv_path = src / "logs.csv"
    _logs_csv(csv_path)
    (src / "readme.txt").write_text("not this one\n")
    arc = tmp_path / "logs.csv.7z"
    with py7zr.SevenZipFile(arc, "w") as z:
        z.writeall(tmp_path / "src" / "data", arcname="data")
    return arc, csv_path


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force many CSV blocks so batch boundaries are exercised.
    monkeypatch.setattr(stream, "BLOCK_SIZE", 1 << 14)


def _per_customer(path: Path) -> dict[str, tuple[int, float]]:
    rows = duckdb.sql(
        f"SELECT msno, sum(num_100), sum(total_secs) FROM '{path}' GROUP BY msno"
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_stream_matches_extract_then_sample(archive: tuple[Path, Path], tmp_path: Path) -> None:
    arc, csv_path = archive
    out = tmp_path / "out" / "logs.parquet"
    stats = stream_sample(arc, "logs.csv", out, 0.2)
    ref_dir = sample_all(csv_path.parent, {"logs": "logs.csv"}, tmp_path / "ref", 0.2)
    streamed, extracted = _per_customer(out), _per_customer(ref_dir / "logs.parquet")
    assert streamed.keys() == extracted.keys()
    for msno, (num_100, secs) in extracted.items():
        assert streamed[msno][0] == num_100
        assert streamed[msno][1] == pytest.approx(secs, rel=1e-12)
    assert stats.kept_rows == duckdb.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]


def test_stats_cover_whole_source(archive: tuple[Path, Path], tmp_path: Path) -> None:
    arc, _ = archive
    stats = stream_sample(arc, "logs.csv", tmp_path / "logs.parquet", 0.2)
    assert stats.source_rows == 3 * N_CUSTOMERS
    assert stats.source_customers == N_CUSTOMERS
    assert (stats.date_min, stats.date_max) == (20170301, 20170303)
    assert stats.uncompressed_bytes > 0


def test_no_csv_written(archive: tuple[Path, Path], tmp_path: Path) -> None:
    arc, _ = archive
    out_dir = tmp_path / "out"
    stream_sample(arc, "logs.csv", out_dir / "logs.parquet", 0.2)
    assert [p.name for p in out_dir.iterdir()] == ["logs.parquet"]


def test_missing_member_raises_and_leaves_nothing(
    archive: tuple[Path, Path], tmp_path: Path
) -> None:
    arc, _ = archive
    out = tmp_path / "out" / "logs.parquet"
    with pytest.raises(RuntimeError):
        stream_sample(arc, "user_logs.csv", out, 0.2)
    assert list(out.parent.iterdir()) == []


def test_parse_error_fails_cleanly(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs.csv"
    _logs_csv(csv_path, bad_row=True)
    arc = tmp_path / "logs.csv.7z"
    with py7zr.SevenZipFile(arc, "w") as z:
        z.write(csv_path, arcname="logs.csv")
    out = tmp_path / "out" / "logs.parquet"
    with pytest.raises(RuntimeError, match="streaming logs.csv"):
        stream_sample(arc, "logs.csv", out, 0.2)
    assert list(out.parent.iterdir()) == []


def test_compare_logs_reports_match_and_mismatch(
    archive: tuple[Path, Path], tmp_path: Path
) -> None:
    arc, csv_path = archive
    out = tmp_path / "out" / "logs.parquet"
    stream_sample(arc, "logs.csv", out, 0.2)
    ref = sample_all(csv_path.parent, {"logs": "logs.csv"}, tmp_path / "ref", 0.2) / "logs.parquet"
    same = compare_logs(out, ref)
    assert same["rows_streamed"] == same["rows_extracted"]
    assert same["customers_only_in_one"] == 0 and same["num_100_mismatches"] == 0
    assert same["total_secs_max_rel_diff"] == 0
    broken = tmp_path / "broken.parquet"
    duckdb.sql(f"COPY (SELECT * REPLACE (num_100 + 1 AS num_100) FROM '{out}') TO '{broken}'")
    assert compare_logs(broken, ref)["num_100_mismatches"] == same["customers"]
