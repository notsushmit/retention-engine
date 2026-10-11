"""Transaction features from the timeline: last row, recency, 365-day counts, lapses."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from retention.features.transactions import TransactionSource
from retention.labels.timeline import build_timeline
from test_timeline import row

DATA_START = date(2015, 1, 1)


def build(clean_tx, rows, t, months=None, **kw) -> pd.DataFrame:
    tl, _ = build_timeline(clean_tx(rows), 30)
    con = duckdb.connect()
    con.register("tl", tl)
    status = pd.Series(months or {}, dtype=object)
    src = TransactionSource(con, "tl", status, 30, DATA_START)
    return src.features(t, sorted(tl.msno.unique()), **kw).set_index("msno")


def test_renewal_on_t_is_ignored_for_active_grace_and_lapsed(clean_tx):
    t = date(2016, 2, 1)
    base = [
        row(20160101, 20160301, m="active"),
        row(20151201, 20160120, m="grace"),
        row(20151001, 20151101, m="lapsed"),
    ]
    later = [
        row(20160201, 20160401, m="active", pm=38, price=99, paid=0),
        row(20160201, 20160301, m="grace", pm=12, ar=0),
        row(20160201, 20160301, m="lapsed", price=180, paid=180),
        row(20160203, 20160303, m="lapsed", cancel=1),
    ]
    a = build(clean_tx, base, t)
    b = build(clean_tx, base + later, t)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["grace", "days_since_last_txn"] == 62
    assert a.loc["lapsed", "n_prior_lapses_365"] == 0


def test_last_row_values_and_discount_ratio(clean_tx):
    t = date(2016, 6, 1)
    rows = [
        row(20160401, 20160501, m="partial", price=149, paid=99, ar=0, pm=36),
        row(20160501, 20160531, m="over", price=149, paid=180),
        row(20160501, 20160531, m="blank", days=0, price=0, paid=149),
        row(20160510, 20160517, m="trial", pm=35, days=7, price=0, paid=0, ar=0),
        row(20160501, 20160531, m="full_discount", price=149, paid=0),
    ]
    f = build(clean_tx, rows, t)
    assert f.loc["partial", "last_discount_ratio"] == pytest.approx(1 - 99 / 149)
    assert f.loc["partial", "last_is_auto_renew"] == 0
    assert f.loc["partial", "last_payment_method_id"] == 36
    assert f.loc["over", "last_discount_ratio"] == pytest.approx(1 - 180 / 149)
    assert pd.isna(f.loc["blank", "last_discount_ratio"])
    assert f.loc["blank", "last_plan_days"] == 0 and f.loc["blank", "last_plan_days_missing"] == 1
    assert f.loc["over", "last_plan_days_missing"] == 0
    assert pd.isna(f.loc["trial", "last_discount_ratio"])
    assert f.loc["trial", "last_row_kind"] == "trial"
    assert pd.isna(f.loc["trial", "days_since_last_payment"])
    assert f.loc["full_discount", "last_row_kind"] == "discount100"
    assert f.loc["full_discount", "n_discounted_365"] == 1
    assert f.loc["partial", "n_discounted_365"] == 1
    assert f.loc["over", "n_discounted_365"] == 0
    assert f.loc["full_discount", "n_paid_365"] == 0


def test_365_day_window_edges(clean_tx):
    t = date(2016, 6, 1)  # T - 365 = 2015-06-02
    rows = [
        row(20150601, 20150701, m="edge"),  # T - 366: outside
        row(20150602, 20150702, m="edge"),  # T - 365: inside
        row(20160531, 20160630, m="edge"),  # T - 1: inside
    ]
    f = build(clean_tx, rows, t)
    assert f.loc["edge", "n_txn_365"] == 2
    assert f.loc["edge", "days_since_last_txn"] == 1
    assert f.loc["edge", "counts_365_days_observed"] == 365
    f15 = build(clean_tx, rows, date(2015, 6, 1))
    assert f15.loc["edge", "counts_365_days_observed"] == 151  # data start 2015-01-01


def test_cancels(clean_tx):
    t = date(2016, 6, 1)
    rows = [
        row(20160401, 20160601, m="c"),
        row(20160415, 20160501, m="c", cancel=1),
        row(20150301, 20150401, m="old"),
        row(20150310, 20150320, m="old", cancel=1),  # outside 365 days
        row(20160401, 20160601, m="old"),
    ]
    f = build(clean_tx, rows, t)
    assert f.loc["c", "n_cancels_365"] == 1
    assert f.loc["c", "days_since_last_cancel"] == 47
    assert f.loc["c", "last_row_is_cancel"] == 1
    assert f.loc["c", "days_since_last_txn"] == 61  # last non-cancel row
    assert f.loc["old", "n_cancels_365"] == 0 and pd.isna(f.loc["old", "days_since_last_cancel"])
    assert f.loc["old", "last_row_is_cancel"] == 0


def test_prior_lapse_needs_gap_over_30_days(clean_tx):
    t = date(2015, 6, 1)
    rows = [
        row(20150101, 20150131, m="gap30"),
        row(20150302, 20150401, m="gap30"),  # 30 days after 2015-01-31: no lapse
        row(20150101, 20150131, m="gap31"),
        row(20150303, 20150402, m="gap31"),  # 31 days: a return after a lapse
    ]
    f = build(clean_tx, rows, t)
    assert f.loc["gap30", "n_prior_lapses_365"] == 0
    assert f.loc["gap31", "n_prior_lapses_365"] == 1


def test_span_status_and_exposure(clean_tx):
    t = date(2016, 8, 1)
    months = {f"2016-{m:02d}": "reliable" for m in range(1, 13)}
    months["2016-05"] = "uncertain"
    rows = [
        row(20160410, 20160810, m="long", days=120, price=500, paid=500),  # spans May
        row(20160401, 20160501, m="monthly"),
        row(20160501, 20160601, m="monthly"),
        row(20160601, 20160701, m="monthly"),
        row(20160701, 20160810, m="monthly"),
    ]
    f = build(clean_tx, rows, t, months)
    assert f.loc["long", "last_txn_span_status"] == "uncertain"
    assert f.loc["monthly", "last_txn_span_status"] == "reliable"
    assert f.loc["monthly", "n_txn_365_in_nonreliable_months"] == 1
    assert f.loc["monthly", "share_txn_365_in_nonreliable_months"] == pytest.approx(0.25)
    early = build(clean_tx, [row(20150101, 20160101, m="x", days=365)], t, months)
    assert early.loc["x", "n_txn_365"] == 0
    assert pd.isna(early.loc["x", "share_txn_365_in_nonreliable_months"])
    assert early.loc["x", "last_txn_span_status"] == "uncertain"  # 2015 months not given


def test_check_columns(clean_tx):
    t = date(2016, 2, 1)
    rows = [row(20160101, 20160301, m="a"), row(20160110, 20160115, m="a", cancel=1)]
    f = build(clean_tx, rows, t, check_columns=True)
    assert f.loc["a", "_last_eff_days"] == -17  # coverage cut to 2016-01-15
    assert f.loc["a", "_last_expiry_days"] == -17


def test_same_day_twins_pick_by_content_in_either_file_order(clean_tx):
    t = date(2016, 6, 1)
    paid = row("20160501", "20160531", m="t", price=99, paid=99, pm=38, ar=0)
    free = row("20160501", "20160531", m="t", price=149, paid=0, pm=41, ar=1)
    a = build(clean_tx, [paid, free], t)
    b = build(clean_tx, [free, paid], t)
    pd.testing.assert_frame_equal(a, b)
    r = a.loc["t"]
    assert (r.last_amount_paid, r.last_list_price, r.last_row_kind) == (99, 99, "paid")
    assert (r.last_payment_method_id, r.last_is_auto_renew) == (38, 0)


def test_rows_equal_in_every_compared_field_are_interchangeable(clean_tx):
    """Identical rows (also one in v1 and one in v2) give the same features as one row."""
    t = date(2016, 6, 1)
    r = row("20160501", "20160531", m="t", price=99, paid=99)
    one = build(clean_tx, [r], t)
    pd.testing.assert_frame_equal(one, build(clean_tx, [r, r], t))
    tl_both = clean_tx([r], [r])
    assert len(tl_both) == 1  # cleaning keeps one copy
