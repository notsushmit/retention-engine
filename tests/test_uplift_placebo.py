"""Placebo window, placebo outcome and the registered leak count."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from retention.features import spec
from retention.uplift import kkbox_units, placebo
from retention.uplift.config import load


def test_placebo_window_is_disjoint_from_and_adjacent_to_every_feature_window():
    usage = [c for c in spec.COLUMNS if c.name in kkbox_units.fs1_columns() and c.group == "usage"]
    windowed = [c for c in usage if c.window != "all history"]
    longest = max(hi for _, hi in spec.WINDOWS.values())
    assert longest == 28
    assert all(c.window_days is not None and c.window_days <= longest for c in windowed)
    assert placebo.PLACEBO_OFFSETS[0] > longest  # disjoint
    assert placebo.PLACEBO_OFFSETS[0] == longest + 1  # adjacent
    assert placebo.PLACEBO_OFFSETS[1] - placebo.PLACEBO_OFFSETS[0] + 1 == 28
    # known exceptions: covariates that read all listening history
    assert {c.name for c in usage if c.window == "all history"} == placebo.ALL_HISTORY


def test_registered_placebo_values_match_the_config():
    p = load()["kkbox"]["outcome_step"]["placebo"]
    assert "T-56 to T-29, inclusive" in p["outcome"]
    assert all(name in p["window_test"] for name in placebo.ALL_HISTORY)
    assert "17 of the 113 kept units (7 full price, 10 discounted)" in " ".join(p["leak"].split())
    assert placebo.REGISTERED_LEAK == {"untreated": 7, "treated": 10}


def write_logs(tmp, rows):
    path = tmp / "user_logs.parquet"
    pd.DataFrame(
        {
            "msno": [r[0] for r in rows],
            "date": np.array([int(r[1].strftime("%Y%m%d")) for r in rows], dtype="int32"),
            "total_secs": 1.0,
        }
    ).to_parquet(path, index=False)
    return path


def test_placebo_days_counts_only_the_window(tmp_path):
    t = pd.Timestamp("2016-09-01")
    day = lambda k: (t - pd.Timedelta(days=k)).date()  # noqa: E731
    rows = [
        ("A", day(57)),  # one day too early
        ("A", day(56)),  # first day of the window
        ("A", day(40)),
        ("A", day(40)),  # a second row on the same day counts once
        ("A", day(29)),  # last day of the window
        ("A", day(28)),  # inside the feature window, not the placebo window
        ("A", date(2016, 9, 4)),  # after T: never counted
        ("B", day(30)),
    ]
    units = pd.DataFrame({"msno": ["A", "B", "C"], "snapshot_date": [t, t, t]})
    got = placebo.placebo_days(write_logs(tmp_path, rows), units)
    assert got.tolist() == [3.0, 1.0, 0.0]


def test_leak_categories_and_count():
    cov = pd.DataFrame(
        {
            "has_any_log_before_T": [0, 1, 1, 1, 1, 1],
            "first_log_in_window": [0, 1, 0, 0, 0, 0],
            "days_since_last_listen": [None, 3, 60, 40, 5, 28],
        }
    )
    cats = placebo.leak_categories(cov)
    assert cats.tolist() == [
        "never_listened",
        "first_listen_last_28",
        "last_listen_over_56",
        "last_listen_29_56",
        "not_fixed",
        "not_fixed",
    ]
    t = np.array([1, 0, 1, 0, 1, 0])
    kept = np.array([True, True, True, True, True, False])
    assert placebo.leak_count(cats, t, kept) == {"untreated": 2, "treated": 2}
    with pytest.raises(placebo.LeakCountMismatch):
        placebo.check_leak({"untreated": 2, "treated": 2})
    placebo.check_leak({"untreated": 7, "treated": 10})
