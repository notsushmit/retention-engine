"""The build on a synthetic dataset: schema, keys, properties, and refusing bad output."""

from datetime import date

import pandas as pd
import pyarrow.parquet as pq
import pytest

from feature_fixtures import SNAPSHOTS, dataset, label_cfg
from retention.features import build as build_mod
from retention.features import spec
from retention.features.build import BuildError, build, check_null_report, read_features


def run(paths, **kw) -> pd.DataFrame:
    build(paths, label_cfg(), memory_limit="512MB", **kw)
    return read_features(kw.get("out") or paths.out).sort_values(["snapshot_date", "msno"])


@pytest.fixture
def paths(tmp_path, clean_tx):
    return dataset(tmp_path / "data", clean_tx)


def test_schema_keys_and_no_leftovers(paths):
    f = run(paths)
    assert list(f.columns) == [c.name for c in spec.COLUMNS]
    assert pq.read_schema(paths.out).equals(spec.arrow_schema(), check_metadata=False)
    assert not [c for c in f.columns if f[c].dtype == object]
    lab = pd.read_parquet(paths.processed / "labels.parquet")
    assert len(f) == len(lab) and not f.duplicated(["msno", "snapshot_date"]).any()
    assert sorted(f.snapshot_date.dt.date.unique()) == SNAPSHOTS
    left = [p.name for p in paths.processed.iterdir()]
    assert not [n for n in left if n.endswith(".tmp") or n.startswith("features_work_")]
    assert (paths.processed / "features_null_report.parquet").exists()
    for c in spec.LABEL_SOURCED:
        merged = f.merge(
            lab[["msno", "snapshot_date", c]].assign(
                snapshot_date=lambda d: pd.to_datetime(d.snapshot_date)
            ),
            on=["msno", "snapshot_date"],
        )
        assert (merged[f"{c}_x"].astype(object) == merged[f"{c}_y"].astype(object)).all(), c


def test_build_twice_gives_the_same_table(paths, tmp_path):
    a = run(paths)
    b = run(paths, out=tmp_path / "again.parquet")
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True
    )


def test_one_snapshot_alone_equals_full_build(paths, tmp_path):
    full = run(paths)
    for t in (SNAPSHOTS[2], SNAPSHOTS[-1]):
        one = run(paths, snapshots=[t], out=tmp_path / f"one_{t}.parquet")
        part = full[full.snapshot_date.dt.date == t]
        pd.testing.assert_frame_equal(
            part.reset_index(drop=True), one.reset_index(drop=True), check_exact=True
        )
        assert part.ratio_secs_vs_cohort.notna().any()


def test_chunk_size_does_not_change_the_table(paths, tmp_path):
    a = run(paths)
    b = run(paths, n_chunks=3, out=tmp_path / "chunks.parquet")
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True
    )


def test_failed_check_keeps_the_previous_table_and_leaves_nothing(paths, monkeypatch):
    run(paths)
    before = paths.out.read_bytes()

    def corrupt(frame, labels):
        for c in spec.LABEL_SOURCED:
            frame[c] = labels[c].to_numpy()
        frame["days_until_expiry"] = frame.days_until_expiry + 1

    monkeypatch.setattr(build_mod, "copy_label_columns", corrupt)
    with pytest.raises(BuildError, match="days_until_expiry differs"):
        build(paths, label_cfg(), memory_limit="512MB")
    assert paths.out.read_bytes() == before
    left = [p.name for p in paths.processed.iterdir()]
    assert not [n for n in left if n.endswith(".tmp") or n.startswith("features_work_")]


def test_row_mismatch_and_vocabulary_are_refused(paths, monkeypatch):
    original = build_mod.UsageSource.features

    def duplicate(self, t, population, **kw):
        out = original(self, t, population, **kw)
        return pd.concat([out, out.iloc[:1]], ignore_index=True)

    monkeypatch.setattr(build_mod.UsageSource, "features", duplicate)
    with pytest.raises(BuildError, match="rows do not match"):
        build(paths, label_cfg(), memory_limit="512MB")
    assert not paths.out.exists()
    monkeypatch.undo()

    original_members = build_mod.member_features

    def bad_city(*args):
        out = original_members(*args)
        out.loc[out.city.notna(), "city"] = 999
        return out

    monkeypatch.setattr(build_mod, "member_features", bad_city)
    with pytest.raises(BuildError, match="city: values outside the vocabulary"):
        build(paths, label_cfg(), memory_limit="512MB")
    assert not paths.out.exists()


def test_check_null_report():
    def report(column, share):
        return pd.DataFrame(
            {
                "snapshot_date": [date(2016, 8, 1)],
                "column": [column],
                "n_rows": [100],
                "null_share": [share],
            }
        )

    assert check_null_report(report("days_since_last_cancel", 0.9)) == []  # documented
    assert check_null_report(report("n_txn_365", 0.0)) == []
    errors = check_null_report(report("n_txn_365", 0.05))  # documented as never null
    assert any("no documented reason" in e for e in errors)
    assert any("never null" in e for e in check_null_report(report("n_txn_365", 0.001)))
