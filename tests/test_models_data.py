from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import CFG
from retention.labels.snapshots import build_labels
from retention.models import data
from retention.models.data import (
    column_kind,
    fit_label_requests,
    labels_as_of,
    load_features,
    load_labels,
    require_as_of,
)
from retention.models.protocol import fully_observed, load_protocol
from test_timeline import row

REAL_LABELS = Path("data/processed/kkbox/sample_020/labels.parquet")
P = load_protocol()
SNAPSHOTS_24 = list(pd.date_range("2015-03-01", "2017-02-01", freq="MS").date)


def synthetic_tx(clean_tx):
    return clean_tx(
        [
            row(20160101, 20160210, m="lapses"),
            row(20160101, 20160610, m="long"),
            row(20160105, 20160204, m="renews"),
            row(20160203, 20160305, m="renews"),
        ]
    )


# --- column classification ------------------------------------------------------------


def test_every_column_of_a_label_build_is_classified(clean_tx):
    cfg = replace(CFG, snapshots=(date(2016, 2, 1),), allow_partial=True).validate()
    labels = build_labels(synthetic_tx(clean_tx), cfg)
    assert {c: column_kind(c) for c in labels.columns}  # raises on an unclassified column


@pytest.mark.skipif(not REAL_LABELS.exists(), reason="needs the processed 20% sample")
def test_every_column_of_the_real_labels_file_is_classified():
    kinds = {c: column_kind(c) for c in pq.read_schema(REAL_LABELS).names}
    assert kinds["churned_7"] == "outcome" and kinds["label_reliable_30"] == "outcome"
    assert kinds["days_until_expiry"] == "pre_t" and kinds["segment"] == "pre_t"


def test_an_unclassified_column_fails():
    with pytest.raises(KeyError, match="not classified"):
        column_kind("churned")  # no horizon suffix: not one of the known names
    with pytest.raises(KeyError):
        column_kind("new_column")


# --- filtered loading -----------------------------------------------------------------


@pytest.fixture
def features_24(tmp_path):
    df = pd.DataFrame(
        [(f"c{i}", t, float(i)) for t in SNAPSHOTS_24 for i in range(3)],
        columns=["msno", "snapshot_date", "x"],
    )
    df["y"] = 1.0
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    return path


def test_reads_return_only_the_named_snapshots_and_columns(features_24, monkeypatch):
    calls = []
    real = data.ds.dataset

    class Spy:
        def __init__(self, path):
            self.d = real(path)

        def to_table(self, **kw):
            calls.append(kw)
            return self.d.to_table(**kw)

    monkeypatch.setattr(data.ds, "dataset", Spy)
    out = load_features(features_24, SNAPSHOTS_24[:2], ["x"])
    assert sorted(out.snapshot_date.dt.date.unique()) == SNAPSHOTS_24[:2]
    assert list(out.columns) == ["msno", "snapshot_date", "x"]
    assert calls[0]["filter"] is not None and calls[0]["columns"] == ["msno", "snapshot_date", "x"]


@pytest.mark.parametrize("n", [5, 24])
def test_no_read_may_load_many_or_all_snapshots(features_24, n):
    with pytest.raises(ValueError, match="at most"):
        load_features(features_24, SNAPSHOTS_24[:n], ["x"])


def test_a_read_must_name_snapshots_and_columns(features_24):
    with pytest.raises(ValueError):
        load_features(features_24, [], ["x"])
    with pytest.raises(ValueError):
        data._read(features_24, SNAPSHOTS_24[:1], [])


# --- the 2017 outcome lock -------------------------------------------------------------


@pytest.fixture
def labels_file(tmp_path):
    df = pd.DataFrame(
        {
            "msno": ["a", "b", "a", "b"],
            "snapshot_date": [date(2016, 10, 1)] * 2 + [date(2017, 1, 1)] * 2,
            "in_exp30": [True] * 4,
            "segment": ["paid"] * 4,
            "churned_7": [0, 1, 1, 0],
            "days_to_event": [90, 3, 2, 59],
        }
    )
    path = tmp_path / "labels.parquet"
    df.to_parquet(path)
    return path


def test_outcomes_of_locked_months_raise_without_an_unlock(labels_file):
    for cols in (["churned_7"], ["days_to_event"], ["segment", "churned_7"]):
        with pytest.raises(PermissionError, match="locked"):
            load_labels(labels_file, [date(2017, 1, 1)], cols, P)
    with pytest.raises(PermissionError):
        load_labels(labels_file, [date(2016, 10, 1), date(2017, 1, 1)], ["churned_7"], P)
    with pytest.raises(PermissionError):
        load_labels(labels_file, [date(2017, 1, 1)], ["churned_7"], P, unlock=object())


def test_pre_t_columns_of_locked_months_and_2016_outcomes_are_readable(labels_file):
    assert len(load_labels(labels_file, [date(2017, 1, 1)], ["in_exp30", "segment"], P)) == 2
    out = load_labels(labels_file, [date(2016, 10, 1)], ["churned_7"], P)
    assert out.churned_7.tolist() == [0, 1]


def test_labels_as_of_are_never_built_for_locked_months(tmp_path):
    with pytest.raises(PermissionError):
        labels_as_of(tmp_path, tmp_path, date(2017, 3, 1), (date(2017, 1, 1),), P)


# --- as_of of fit and calibration labels -------------------------------------------------


def test_label_builds_carry_their_as_of_and_later_ones_are_refused(clean_tx, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    synthetic_tx(clean_tx).to_parquet(processed / "transactions.parquet")
    snaps, split = (date(2016, 2, 1),), date(2016, 3, 15)
    base = replace(CFG, allow_partial=True)
    for as_of in (split, split + timedelta(days=1)):
        path = labels_as_of(processed, tmp_path / "cache", as_of, snaps, P, base=base)
        got = load_labels(path, snaps, ["churned_7"], P)
        assert got.attrs["as_of"] == as_of
        built = pq.read_table(path, columns=["snapshot_date"]).column(0).unique().to_pylist()
        assert built == list(snaps)  # only the snapshots this as_of needs
        if as_of <= split:
            require_as_of(got, split)
        else:
            with pytest.raises(ValueError, match="later than the split date"):
                require_as_of(got, split)
    final = tmp_path / "final.parquet"
    pd.DataFrame({"msno": ["x"], "snapshot_date": [snaps[0]], "churned_7": [0]}).to_parquet(final)
    with pytest.raises(ValueError, match="no as_of"):
        require_as_of(load_labels(final, snaps, ["churned_7"], P), split)


def test_no_split_uses_labels_built_with_a_later_as_of():
    requests = fit_label_requests(P)
    assert {name for name, _, _ in requests} == {*P.folds, "test", "test_sensitivity"}
    for name, as_of, snaps in requests:
        split = P.test_fit if name.startswith("test") else P.fold(name)
        assert as_of == split.as_of
        horizons = (
            P.horizons
            if name == "test"
            else (7,)
            if name == "test_sensitivity"
            else (P.fold(name).horizon,)
        )
        for t in snaps:
            assert t < P.locked_from
            assert all(fully_observed(t, h, as_of, P.purge_days) for h in horizons)


@pytest.mark.skipif(not REAL_LABELS.exists(), reason="needs the processed 20% sample")
def test_real_label_check_report_shows_no_differences():
    import json

    report = json.loads((REAL_LABELS.parent / "label_check.json").read_text())
    assert report["differing_rows_total"] == 0 and len(report["builds"]) == 13
