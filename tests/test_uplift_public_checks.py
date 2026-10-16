"""Public dataset checks on small synthetic files: pass, and stop on each kind of mismatch."""

import gzip
from pathlib import Path

import pytest

from retention.uplift import public_data as pdc

H_COLS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]
ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
C_COLS = [*(f"f{i}" for i in range(12)), "treatment", "conversion", "visit", "exposure"]


def hillstrom_file(tmp: Path, per_arm=(100, 100, 100), header=None) -> Path:
    header = header or [c.capitalize() if c != "segment" else "segment" for c in H_COLS]
    lines = [",".join(header)]
    for arm, n in zip(ARMS, per_arm, strict=True):
        lines += [f"1,1) $0 - $100,10.0,1,0,Urban,1,Web,{arm},0,0,0.0"] * n
    path = tmp / "hillstrom.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def criteo_file(tmp: Path, treated=85, control=15, cols=C_COLS) -> Path:
    lines = [",".join(cols)]
    feats = ",".join(["0.5"] * 12)
    lines += [f"{feats},1,0,1,1"] * treated + [f"{feats},0,0,0,0"] * control
    path = tmp / "criteo.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def h_check(path, sha=None, rows=300, tol=0.01):
    return pdc.check_hillstrom(path, sha or pdc.sha256(path), rows, H_COLS, ARMS, tol)


def c_check(path, sha=None, rows=100, share=0.85, tol=0.005, cols=C_COLS):
    return pdc.check_criteo(path, sha or pdc.sha256(path), rows, cols, share, tol)


def test_hillstrom_passes_and_writes_nothing(tmp_path):
    path = hillstrom_file(tmp_path)
    out = h_check(path)
    assert out["rows"] == 300 and out["arms"] == dict.fromkeys(ARMS, 100)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hillstrom.csv"]


def test_hillstrom_stops_on_each_mismatch(tmp_path):
    path = hillstrom_file(tmp_path)
    with pytest.raises(pdc.CheckFailed, match="SHA-256"):
        h_check(path, sha="0" * 64)
    with pytest.raises(pdc.CheckFailed, match="rows"):
        h_check(path, rows=64000)
    with pytest.raises(pdc.CheckFailed, match="arm shares"):
        h_check(hillstrom_file(tmp_path, per_arm=(110, 100, 90)))
    bad = list(H_COLS)
    bad[0] = "recency_days"
    with pytest.raises(pdc.CheckFailed, match="columns"):
        h_check(hillstrom_file(tmp_path, header=bad))


def test_criteo_passes_with_the_row_count_note(tmp_path):
    out = c_check(criteo_file(tmp_path))
    assert out["rows"] == 100 and out["treatment_share"] == pytest.approx(0.85)
    assert "25M" in out["row_count_note"]


def test_criteo_stops_on_each_mismatch(tmp_path):
    path = criteo_file(tmp_path)
    with pytest.raises(pdc.CheckFailed, match="SHA-256"):
        c_check(path, sha="0" * 64)
    with pytest.raises(pdc.CheckFailed, match="rows"):
        c_check(path, rows=13979592)
    with pytest.raises(pdc.CheckFailed, match="treatment share"):
        c_check(criteo_file(tmp_path, treated=80, control=20))
    with pytest.raises(pdc.CheckFailed, match="columns"):
        c_check(path, cols=[c for c in C_COLS if c != "exposure"])


def test_to_parquet_copies_every_row(tmp_path):
    import duckdb

    out = tmp_path / "public" / "c.parquet"
    pdc.to_parquet(criteo_file(tmp_path), out)
    assert duckdb.sql(f"select count(*) from '{out}'").fetchone()[0] == 100


def test_registered_values_are_wired_to_the_command():
    from retention.uplift import config

    v = config.load()["validation"]["datasets"]
    assert v["hillstrom"]["expected_columns"] == H_COLS
    assert sorted(v["criteo"]["expected_columns"]) == sorted(C_COLS)
    assert v["criteo"]["expected_sha256"] == (
        "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
    )
    assert v["criteo"]["expected_rows"] == 13979592
    assert v["hillstrom"]["expected_rows"] == 64000
