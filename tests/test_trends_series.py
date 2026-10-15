"""Weekly churn-rate construction on a hand-made case."""

import pandas as pd
import pytest

from retention.trends.series import weekly_rates


def frame() -> pd.DataFrame:
    rows = [
        # id, snapshot, expiry, in_expiry_month, segment, churned
        ("a", "2016-02-01", "2016-02-29", True, "paid", 0),
        ("b", "2016-02-01", "2016-02-29", True, "paid", 1),
        ("c", "2016-03-01", "2016-03-02", True, "paid", 0),
        ("d", "2016-03-01", "2016-03-05", True, "paid", 1),
        ("f", "2016-02-01", "2016-03-10", False, "paid", None),  # not an event in February
        ("f", "2016-03-01", "2016-03-10", True, "paid", 0),
        ("g", "2016-03-01", "2016-03-03", True, "discount100_only", 1),
        ("h", "2016-03-01", "2016-03-04", True, "paid", None),  # outcome not observed
        ("i", "2016-03-01", "2016-03-31", True, "paid", 1),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "msno",
            "snapshot_date",
            "expiry_date",
            "in_expiry_month",
            "segment",
            "churned_classic",
        ],
    )
    df["snapshot_date"] = pd.to_datetime(df.snapshot_date)
    df["expiry_date"] = pd.to_datetime(df.expiry_date)
    df["churned_classic"] = df.churned_classic.astype("Int8")
    return df


def test_week_spanning_two_months_uses_both_snapshots() -> None:
    w = weekly_rates(frame(), "paid", data_end="2016-04-30")
    r = w.loc["2016-02-29"]
    assert (r.events, r.churned, r.unobserved) == (4, 2, 1)  # a, b, c, d; h unobserved
    assert r.rate == 0.5
    assert bool(r.complete)  # Sunday 03-06 + 30 days = 04-05 <= 04-30


def test_segments_are_separate_and_incomplete_week_is_marked() -> None:
    d = weekly_rates(frame(), "discount100_only", data_end="2016-03-20")
    assert d.loc["2016-02-29"].events == 1 and d.loc["2016-02-29"].rate == 1.0
    assert not bool(d.loc["2016-02-29"].complete)  # 03-06 + 30 > 03-20


def test_expiry_counted_once_and_not_in_other_snapshot() -> None:
    w = weekly_rates(frame(), "paid", data_end="2016-04-30")
    assert w.loc["2016-03-07"].events == 1  # f once, from the March snapshot only


def test_duplicate_event_raises() -> None:
    df = pd.concat([frame(), frame().iloc[[0]]])
    with pytest.raises(ValueError):
        weekly_rates(df, "paid")
