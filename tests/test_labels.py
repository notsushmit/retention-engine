import random
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from conftest import CFG
from retention.labels.snapshots import build_labels
from test_timeline import random_rows, row


def cfg(snapshots, as_of=date(2016, 6, 30), **kw):
    return replace(
        CFG, snapshots=tuple(snapshots), as_of=as_of, allow_partial=True, **kw
    ).validate()


def one(labels, msno, snapshot):
    return labels[(labels.msno == msno) & (labels.snapshot_date == snapshot)].iloc[0]


def test_trial_after_paid_plan_paid_lapse_coverage_renewal(clean_tx):
    rows = [
        row(20151101, 20151231, m="a"),
        row(20151231, 20160107, m="a", pm=35, days=7, price=0, paid=0, ar=0),  # free trial
        row(20160203, 20160304, m="a"),  # paid again 27 days after the trial ends
    ]
    lab = build_labels(clean_tx(rows), cfg([date(2015, 12, 1)]))
    r = one(lab, "a", date(2015, 12, 1))
    assert r.churned_classic == 0  # the trial counts as a renewal
    assert r.paid_churned_classic == 1  # paid coverage ended 2015-12-31, next payment 34 days later
    assert r.paid_days_to_event == 30 and r.paid_event_observed == 1
    assert r.event_observed == 0  # coverage continues past the 90-day cap


def test_censoring_and_event_time(clean_tx):
    rows = [row(20160101, 20160210, m="lapses"), row(20160101, 20160610, m="long")]
    lab = build_labels(clean_tx(rows), cfg([date(2016, 2, 1)]))
    a = one(lab, "lapses", date(2016, 2, 1))
    assert a.event_observed == 1 and a.days_to_event == 9 and a.churned_30 == 1
    b = one(lab, "long", date(2016, 2, 1))  # censored at min(T + 90, as_of - 30) = 2016-05-01
    assert b.event_observed == 0 and b.days_to_event == 90 and b.churned_60 == 0


def test_unobservable_horizon_is_empty(clean_tx):
    lab = build_labels(
        clean_tx([row(20160101, 20160210)]), cfg([date(2016, 5, 15)], as_of=date(2016, 6, 30))
    )
    assert lab.empty  # expired before the snapshot: not in the population
    lab = build_labels(
        clean_tx([row(20160501, 20160610)]), cfg([date(2016, 5, 15)], as_of=date(2016, 6, 30))
    )
    r = lab.iloc[0]  # last labelable expiry 2016-05-31
    assert r.churned_7 == 0 and pd.isna(r.churned_30) and pd.isna(r.churned_60)
    assert r.event_observed == 0 and r.days_to_event == 16


def test_grace_period_customers_are_excluded(clean_tx):
    rows = [row(20160101, 20160125, m="grace"), row(20160101, 20160215, m="active")]
    lab = build_labels(clean_tx(rows), cfg([date(2016, 2, 1)]))
    assert lab.msno.tolist() == ["active"]


def test_segment_and_state_use_rows_before_snapshot(clean_tx):
    rows = [
        row(20160101, 20160108, m="t", pm=35, days=7, price=0, paid=0, ar=0),
        row(20160107, 20160210, m="t"),
        row(20160201, 20160310, m="t"),  # dated on the snapshot: not before it
    ]
    lab = build_labels(clean_tx(rows), cfg([date(2016, 1, 7), date(2016, 2, 1)]))
    assert one(lab, "t", date(2016, 1, 7)).segment == "trial_only"
    r = one(lab, "t", date(2016, 2, 1))
    assert r.segment == "paid" and str(r.expiry_date.date()) == "2016-02-10"


def test_left_truncation_flag(clean_tx):
    rows = [row(20160101, 20160301), row(20160301, 20160401)]
    lab = build_labels(clean_tx(rows), cfg([date(2016, 2, 14), date(2016, 3, 15)]))
    assert lab.set_index("snapshot_date").left_truncation_risk.to_dict() == {
        date(2016, 2, 14): True,
        date(2016, 3, 15): False,
    }


@pytest.mark.parametrize("seed", range(6))
def test_as_of_equivalence(clean_tx, seed):
    """Labels from full data as of D equal labels from data truncated at D, for both families."""
    rng = random.Random(seed)
    v1, v2 = random_rows(rng)
    full = clean_tx(v1, v2)
    for d in (date(2016, 3, 15), date(2016, 4, 30)):
        snaps = [date(2016, 1, 15), date(2016, 2, 1), date(2016, 3, 1)]
        snaps = [t for t in snaps if t <= d]
        c = cfg(snaps, as_of=d)
        cut = [r for r in v1 if int(r.split(",")[6]) <= int(d.strftime("%Y%m%d"))]
        cut2 = [r for r in v2 if int(r.split(",")[6]) <= int(d.strftime("%Y%m%d"))]
        a = build_labels(full, c).sort_values(["snapshot_date", "msno"]).reset_index(drop=True)
        b = (
            build_labels(clean_tx(cut, cut2), c)
            .sort_values(["snapshot_date", "msno"])
            .reset_index(drop=True)
        )
        assert len(a) > 0
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
