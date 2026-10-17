"""Readers through a Parquet round trip: dates come back as timestamps, never as dates.

The first value run stopped because a probability read from Parquet carried a
datetime64 snapshot and was compared with a datetime.date (no row matched). Every
reader the value and plan steps use is checked here after writing and reading back.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.optimize.config import load_actions_config
from retention.optimize.effects import build_options, missing_deadline_count
from retention.optimize.price import price_per_day, read_transactions
from retention.optimize.value import (
    INTERVAL_ENDS_ALL,
    as_date,
    customer_table,
    first_run_sha_failure,
    read_p30,
)

OCT, NOV = date(2016, 10, 1), date(2016, 11, 1)
CFG = load_actions_config()


def write(df: pd.DataFrame, path, date_columns=()) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    for c in date_columns:  # stored as Parquet date32, as the project's tables are
        i = table.schema.get_field_index(c)
        table = table.set_column(i, c, table.column(c).cast(pa.date32()))
    pq.write_table(table, path)


@pytest.mark.parametrize("stored", ["date32", "timestamp"])
def test_read_p30_returns_dates_that_match_snapshot_filters(tmp_path, stored):
    reasons = pd.DataFrame(
        {
            "msno": ["a", "b", "c", "a"],
            "snapshot_date": pd.to_datetime(
                ["2016-10-01", "2016-10-01", "2016-11-01", "2016-11-01"]
            ),
            "horizon": [30, 30, 30, 7],
            "probability": [0.1, 0.2, 0.3, 0.9],
        }
    )
    path = tmp_path / "reasons.parquet"
    write(reasons, path, ["snapshot_date"] if stored == "date32" else [])
    p = read_p30(path, [OCT, NOV])
    assert all(isinstance(v, date) for v in p.snapshot_date)
    assert (p.snapshot_date == OCT).sum() == 2 and (p.snapshot_date == NOV).sum() == 1


def test_customer_table_joins_every_snapshot_from_a_parquet_p30(tmp_path):
    rows = pd.DataFrame(
        {
            "msno": ["a", "b", "a"],
            "snapshot_date": pd.to_datetime(["2016-10-01", "2016-10-01", "2016-11-01"]),
            "last_is_auto_renew": [0, 1, 0],
            "days_until_paid_expiry": pd.array([5, None, 20], dtype="Int32"),
        }
    )
    p30 = pd.DataFrame(
        {
            "msno": ["a", "b", "a"],
            "snapshot_date": pd.to_datetime(["2016-10-01", "2016-10-01", "2016-11-01"]),
            "probability": [0.4, 0.1, 0.3],
            "horizon": [30, 30, 30],
        }
    )
    path = tmp_path / "reasons.parquet"
    write(p30, path, ["snapshot_date"])
    p = read_p30(path, [OCT, NOV])
    surv = np.tile(np.linspace(1.0, 0.5, len(INTERVAL_ENDS_ALL)), (3, 1))
    prices = pd.DataFrame(
        {"msno": ["a", "b"], "price_per_day": [149 / 30, 0.0], "has_price": [True, False]}
    )
    t = customer_table(rows[rows.snapshot_date.dt.date == OCT], surv[:2], p, prices)
    assert t.p30.tolist() == [
        0.4,
        0.1,
    ]  # the whole p30 table is passed; the join picks the snapshot

    # value.parquet round trip: survival, deadlines and snapshot filters survive
    write(t, tmp_path / "value.parquet", ["snapshot_date"])
    back = pd.read_parquet(tmp_path / "value.parquet")
    back["snapshot_date"] = as_date(back.snapshot_date)
    assert (back.snapshot_date == OCT).sum() == 2
    assert back.days_30_90.tolist() == pytest.approx(t.days_30_90.tolist())
    assert [back[f"s_{e}"].iloc[0] for e in INTERVAL_ENDS_ALL] == pytest.approx(surv[0].tolist())
    assert missing_deadline_count(back) == 1
    o = build_options(back, CFG, CFG.base)
    assert not o.allowed[1].any()  # the missing deadline survives the round trip
    assert o.value[1] == 0.0  # no usable price


def test_price_reader_round_trip_and_snapshot_types(tmp_path):
    tx = pd.DataFrame(
        {
            "msno": ["a", "a", "b"],
            "transaction_date": pd.to_datetime(["2016-09-01", "2016-10-15", "2016-10-20"]),
            "is_cancel": [0, 0, 0],
            "actual_amount_paid": [149, 0, 99],
            "payment_plan_days": [30, 30, 30],
            "plan_list_price": [149, 149, 99],
            "payment_method_id": [41, 41, 41],
            "expiry_date": pd.to_datetime(["2016-10-01", "2016-11-14", "2016-11-19"]),
        }
    )
    path = tmp_path / "tx.parquet"
    write(tx, path, ["transaction_date", "expiry_date"])
    back = read_transactions([OCT, NOV], path)
    for snap in (pd.to_datetime([NOV, NOV]), [NOV, NOV]):  # timestamps or dates
        pop = pd.DataFrame({"msno": ["a", "b"], "snapshot_date": snap})
        out = price_per_day(back, pop)
        assert out.price_per_day.tolist() == pytest.approx([149 / 30, 99 / 30])
    pop = pd.DataFrame({"msno": ["b"], "snapshot_date": [OCT]})
    assert not price_per_day(back, pop).has_price.iloc[0]  # b's only row is after 2016-10-01


def test_first_run_sha_check(tmp_path):
    assert first_run_sha_failure(tmp_path, "abc") is None  # no first run kept
    (tmp_path / "dt01_refit_first_run.json").write_text(json.dumps({"booster_sha256": "abc"}))
    assert first_run_sha_failure(tmp_path, "abc") is None
    assert first_run_sha_failure(tmp_path, "abd") == (
        "booster SHA-256 abd differs from the first run's abc"
    )
