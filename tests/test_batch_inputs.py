"""Batch inputs of one snapshot: status rows, training rows in order, facts and prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.api import batch
from retention.models.data import OUTCOME_LABEL_PREFIXES, column_kind
from retention.models.protocol import load_protocol

P = load_protocol()
T = date(2016, 11, 1)

# msno, segment, in_exp30, days_until_expiry, x (a model column), f (a fact column)
CUSTOMERS = [
    ("m03", "paid", True, 5, 0.3, 1.0),
    ("m01", "paid", True, 12, 0.1, 2.0),
    ("t01", "trial_only", True, 3, 0.9, 3.0),
    ("p01", "promo_only", True, 8, 0.5, 4.0),
    ("m02", "discount100_only", True, 20, 0.2, 5.0),
    ("b01", "paid", False, 45, 0.4, 6.0),
]


def write(root: Path) -> None:
    feats = pd.DataFrame(
        [
            {"msno": m, "snapshot_date": T, "segment": s, "days_until_expiry": d, "x": x, "f": f}
            for m, s, _, d, x, f in CUSTOMERS
        ]
    )
    labels = pd.DataFrame(
        [
            {"msno": m, "snapshot_date": T, "in_exp30": w, "segment": s, "churned_7": 1}
            for m, s, w, _, _, _ in sorted(CUSTOMERS)
        ]
    )
    pq.write_table(pa.Table.from_pandas(feats, preserve_index=False), root / "features.parquet")
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), root / "labels.parquet")


def tx(rows: list[tuple]) -> pd.DataFrame:
    cols = [
        "msno",
        "transaction_date",
        "is_cancel",
        "actual_amount_paid",
        "payment_plan_days",
        "plan_list_price",
        "payment_method_id",
        "expiry_date",
    ]
    return pd.DataFrame(rows, columns=cols)


TX = tx(
    [
        ("m01", date(2016, 10, 1), 0, 149, 30, 149, 41, date(2016, 11, 1)),
        ("m01", date(2016, 11, 1), 0, 300, 30, 300, 41, date(2016, 12, 1)),  # dated T: ignored
        ("m03", date(2016, 9, 15), 0, 99, 30, 99, 41, date(2016, 10, 15)),
        ("m03", date(2016, 10, 15), 0, 0, 30, 99, 41, date(2016, 11, 15)),  # free row after
    ]
)


def test_status_rows_cover_every_customer_with_coverage(tmp_path):
    write(tmp_path)
    s = batch.status_rows(tmp_path, T, P)
    assert sorted(s.msno) == sorted(c[0] for c in CUSTOMERS)
    assert list(s.columns) == ["msno", "in_exp30", "segment", "days_until_expiry"]
    b01 = s.set_index("msno").loc["b01"]
    assert (bool(b01.in_exp30), int(b01.days_until_expiry)) == (False, 45)


def test_training_rows_keep_the_registered_order_and_columns(tmp_path):
    write(tmp_path)
    inputs = batch.snapshot_inputs(tmp_path, T, ["x"], ["f"], P, transactions=TX)
    assert inputs.rows.msno.tolist() == ["m03", "m01", "m02"]  # features-file order, main only
    assert {"x", "f"} <= set(inputs.rows.columns)
    assert sorted(inputs.facts.msno) == ["m01", "m02", "m03"]
    assert list(inputs.facts.columns) == ["msno", "f"]


def test_prices_use_the_last_paid_row_before_t(tmp_path):
    write(tmp_path)
    inputs = batch.snapshot_inputs(tmp_path, T, ["x"], ["f"], P, transactions=TX)
    p = inputs.prices.set_index("msno")
    assert p.loc["m01", "price_per_day"] == pytest.approx(149 / 30)  # the row dated T is ignored
    assert p.loc["m03", "price_per_day"] == pytest.approx(
        99 / 30
    )  # a later free row does not hide it
    assert (bool(p.loc["m02", "has_price"]), p.loc["m02", "price_per_day"]) == (False, 0.0)
    assert inputs.prices.msno.tolist() == inputs.rows.msno.tolist()


def test_no_outcome_column_is_requested(tmp_path, monkeypatch):
    write(tmp_path)
    asked: list[list[str]] = []
    real = batch.load_labels

    def spy(path, snapshots, columns, protocol, unlock=None):
        asked.append(list(columns))
        assert unlock is None
        return real(path, snapshots, columns, protocol, unlock)

    monkeypatch.setattr(batch, "load_labels", spy)
    batch.status_rows(tmp_path, T, P)
    assert asked and all(column_kind(c) == "pre_t" for cols in asked for c in cols)
    assert not [c for cols in asked for c in cols if c.startswith(OUTCOME_LABEL_PREFIXES)]


def test_features_and_labels_with_different_customers_are_refused(tmp_path):
    write(tmp_path)
    lab = pq.read_table(tmp_path / "labels.parquet").to_pandas()
    lab = lab[lab.msno != "b01"]
    pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), tmp_path / "labels.parquet")
    with pytest.raises(ValueError, match="different customers"):
        batch.status_rows(tmp_path, T, P)
