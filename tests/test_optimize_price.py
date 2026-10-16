from datetime import date

import pandas as pd
import pytest

from retention.optimize.price import price_per_day

T = date(2016, 11, 1)


def tx(msno, d, paid, days, *, cancel=0, expiry=None, list_price=None, method=41):
    return {
        "msno": msno,
        "transaction_date": pd.Timestamp(d),
        "is_cancel": cancel,
        "actual_amount_paid": paid,
        "payment_plan_days": days,
        "plan_list_price": paid if list_price is None else list_price,
        "payment_method_id": method,
        "expiry_date": pd.Timestamp(expiry) if expiry else pd.NaT,
    }


def run(rows, msnos=("a",), t=T):
    pop = pd.DataFrame({"msno": list(msnos), "snapshot_date": [pd.Timestamp(t)] * len(msnos)})
    return price_per_day(pd.DataFrame(rows), pop).set_index("msno")


def test_zero_paid_last_row_uses_the_earlier_paid_row():
    # paid 149 for 30 days, then a free 100%-discount month: the price is 149 / 30
    out = run(
        [
            tx("a", "2016-09-01", 149, 30, expiry="2016-10-01"),
            tx("a", "2016-10-01", 0, 30, expiry="2016-10-31", list_price=149),
        ]
    )
    assert out.loc["a", "price_per_day"] == pytest.approx(149 / 30)
    assert out.loc["a", "price_row_date"] == pd.Timestamp("2016-09-01")


def test_no_paid_row_gives_zero_and_no_price():
    out = run([tx("a", "2016-10-20", 0, 7, list_price=0)])
    assert out.loc["a", "price_per_day"] == 0.0
    assert not out.loc["a", "has_price"]


def test_customer_with_no_rows_gives_zero():
    out = run([tx("b", "2016-10-01", 99, 30)], msnos=("a", "b"))
    assert out.loc["a", "price_per_day"] == 0.0 and not out.loc["a", "has_price"]
    assert out.loc["b", "price_per_day"] == pytest.approx(99 / 30)


def test_rows_on_or_after_t_are_never_used():
    out = run(
        [
            tx("a", "2016-10-01", 99, 30),
            tx("a", "2016-11-01", 180, 30),  # dated T itself
            tx("a", "2016-11-15", 149, 30),
        ]
    )
    assert out.loc["a", "price_per_day"] == pytest.approx(99 / 30)


def test_cancel_rows_and_zero_plan_days_are_skipped():
    out = run(
        [
            tx("a", "2016-09-01", 129, 30),
            tx("a", "2016-10-01", 149, 30, cancel=1),
            tx("a", "2016-10-02", 150, 0),
        ]
    )
    assert out.loc["a", "price_per_day"] == pytest.approx(129 / 30)


def test_same_day_ties_latest_expiry_then_highest_amount():
    rows = [
        tx("a", "2016-10-01", 149, 30, expiry="2016-10-31"),
        tx("a", "2016-10-01", 99, 30, expiry="2016-11-30"),
    ]
    assert run(rows).loc["a", "price_per_day"] == pytest.approx(99 / 30)
    rows = [
        tx("a", "2016-10-01", 99, 30, expiry="2016-10-31"),
        tx("a", "2016-10-01", 149, 30, expiry="2016-10-31"),
    ]
    assert run(rows).loc["a", "price_per_day"] == pytest.approx(149 / 30)
    assert run(rows[::-1]).loc["a", "price_per_day"] == pytest.approx(149 / 30)


def test_long_plan_price_per_day():
    out = run([tx("a", "2016-03-01", 1788, 410)])
    assert out.loc["a", "price_per_day"] == pytest.approx(1788 / 410)


def test_population_order_and_snapshot_are_kept():
    rows = [tx("a", "2016-09-15", 149, 30), tx("a", "2016-10-15", 99, 30)]
    pop = pd.DataFrame(
        {
            "msno": ["a", "a"],
            "snapshot_date": [pd.Timestamp("2016-11-01"), pd.Timestamp("2016-10-01")],
        }
    )
    out = price_per_day(pd.DataFrame(rows), pop)
    assert list(out.snapshot_date) == [pd.Timestamp("2016-11-01"), pd.Timestamp("2016-10-01")]
    assert out.price_per_day.tolist() == pytest.approx([99 / 30, 149 / 30])
