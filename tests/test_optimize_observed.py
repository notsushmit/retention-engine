from datetime import date

import numpy as np
import pandas as pd
import pytest

import retention.models.protocol as protocol_module
import retention.optimize.observed as observed
from retention.models.protocol import load_protocol
from retention.optimize.config import load_actions_config
from retention.optimize.solve import NONE

NOV = date(2016, 11, 1)


@pytest.fixture
def labels_file(tmp_path):
    df = pd.DataFrame(
        {
            "msno": ["a", "b", "c", "d"],
            "snapshot_date": pd.to_datetime([NOV] * 4),
            "churned_30": pd.array([1, 0, None, 1], dtype="Int8"),
            "in_exp30": [1, 1, 1, 1],
        }
    )
    path = tmp_path / "labels.parquet"
    df.to_parquet(path)
    return path


def test_the_label_is_the_registered_one():
    assert observed.LABEL == load_actions_config().observed_label


def test_read_once_then_refused(labels_file, tmp_path):
    protocol = load_protocol()
    out = observed.read_once(labels_file, NOV, protocol, tmp_path / "o", "abc1234")
    assert out.churned_30.tolist()[:2] == [1, 0]
    assert observed.marker_path(tmp_path / "o", NOV).exists()
    with pytest.raises(PermissionError, match="already read"):
        observed.read_once(labels_file, NOV, protocol, tmp_path / "o", "abc1234")


def test_2017_and_unregistered_snapshots_are_refused(labels_file, tmp_path):
    protocol = load_protocol()
    for t in (date(2017, 1, 1), date(2016, 12, 1), date(2016, 9, 1)):
        with pytest.raises(PermissionError):
            observed.read_once(labels_file, t, protocol, tmp_path, "x")
    assert not list(tmp_path.glob("observed_read_*"))


def test_only_churned_30_is_requested_and_no_unlock(monkeypatch, labels_file, tmp_path):
    calls = []

    def spy(path, snapshots, columns, protocol, unlock=None):
        calls.append((list(snapshots), list(columns), unlock))
        return pd.DataFrame({"msno": [], "snapshot_date": [], "churned_30": []})

    built = []
    real = protocol_module.TestUnlock.__init__
    monkeypatch.setattr(
        protocol_module.TestUnlock, "__init__", lambda self: built.append(1) or real(self)
    )
    monkeypatch.setattr(observed, "load_labels", spy)
    observed.read_once(labels_file, NOV, load_protocol(), tmp_path, "x")
    assert calls == [([NOV], ["churned_30"], None)]
    assert built == []


def test_shares_counts_only_customers_with_an_outcome(labels_file, tmp_path):
    y = pd.read_parquet(labels_file)[["msno", "churned_30"]]
    msno = np.array(["a", "b", "c", "d", "e"])
    choices = {
        "optimizer": np.array([0, 0, 2, NONE, NONE]),
        "no_action": np.full(5, NONE),
    }
    s = observed.shares(msno, choices, y)
    assert s["optimizer"] == {
        "targeted": 3,
        "with_outcome": 2,
        "lapsed": 1,
        "share_lapsed": 0.5,
        "label": "says nothing about whether any action works",
    }
    assert s["no_action"]["share_lapsed"] is None
