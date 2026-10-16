"""KKBox unit reader: the unit rule, the treatment, and the read guards.

No outcome column is read; no transaction after 2016-12-31 is read; covariates are pre-T.
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.models.data import OUTCOME_LABEL_PREFIXES, column_kind
from retention.models.protocol import load_protocol
from retention.uplift import kkbox_units as ku
from retention.uplift.config import load

T_SEP = date(2016, 9, 1)
T_DEC = date(2016, 12, 1)


def tx_row(row_id, msno, tdate, expiry, list_price, paid, method=23, plan=30, cancel=0):
    kind = "cancel" if cancel else ("paid" if paid > 0 else "discount100")
    return {
        "row_id": row_id,
        "msno": msno,
        "transaction_date": date.fromisoformat(tdate),
        "expiry_date": date.fromisoformat(expiry),
        "is_cancel": cancel,
        "row_kind": kind,
        "plan_list_price": list_price,
        "actual_amount_paid": paid,
        "payment_plan_days": plan,
        "payment_method_id": method,
    }


TX = [
    # A: a cancel after T is skipped; two same-day renewals, the lower row_id (full) wins
    tx_row("v1:1", "A", "2016-09-02", "2016-09-15", 149, 149, cancel=1),
    tx_row("v1:2", "A", "2016-09-10", "2016-10-10", 149, 149),
    tx_row("v1:3", "A", "2016-09-10", "2016-10-10", 149, 129),
    # B: the only renewal falls after the paid expiry plus 30 days: no unit
    tx_row("v1:4", "B", "2016-10-25", "2016-11-25", 149, 149),
    # D: discounted renewal, depth 1 - 129 / 149 = 0.134
    tx_row("v1:5", "D", "2016-09-03", "2016-10-05", 149, 129),
    # D's earlier discount, 100 days before T, and a later one after T (post-treatment)
    tx_row("v1:6", "D", "2016-05-24", "2016-06-23", 149, 129),
    tx_row("v1:7", "D", "2016-09-20", "2016-11-04", 149, 129),
    # F: the planted 2017 row. Its first renewal after T = 2016-12-01 would be this
    # discounted row, making F a treated unit if the date filter were wrong.
    tx_row("v1:8", "F", "2017-01-05", "2017-02-04", 149, 99),
    # G: below the margin (150 -> 149, depth 0.007)
    tx_row("v1:9", "G", "2016-09-08", "2016-10-08", 150, 149),
]


def write_tx(tmp: Path) -> Path:
    path = tmp / "transactions.parquet"
    pd.DataFrame(TX).to_parquet(path, index=False)
    return path


def write_labels(tmp: Path) -> Path:
    rows = [
        ("A", T_SEP, "2016-09-15"),
        ("B", T_SEP, "2016-09-20"),
        ("C", T_SEP, "2016-10-12"),  # expiry not in T's month: not a member
        ("D", T_SEP, "2016-09-05"),
        ("G", T_SEP, "2016-09-10"),
        ("F", T_DEC, "2016-12-20"),
    ]
    table = pa.table(
        {
            "msno": [r[0] for r in rows],
            "snapshot_date": pa.array([r[1] for r in rows], pa.date32()),
            "paid_state": ["active"] * len(rows),
            "paid_expiry_date": pa.array([pd.Timestamp(r[2]) for r in rows], pa.timestamp("ns")),
            # an outcome column that must never be requested
            "paid_churned_classic": pa.array([1] * len(rows), pa.int8()),
        }
    )
    path = tmp / "labels.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def spy(monkeypatch):
    seen: list[tuple[list, list]] = []
    real = ku.load_labels

    def recording(path, snapshots, columns, protocol, unlock=None):
        seen.append((list(snapshots), list(columns)))
        return real(path, snapshots, columns, protocol, unlock)

    monkeypatch.setattr(ku, "load_labels", recording)
    return seen


def units_for(tmp: Path, snapshots) -> pd.DataFrame:
    members = ku.read_members(write_labels(tmp), snapshots, load_protocol())
    return ku.renewal_units(members, ku.read_transactions(write_tx(tmp)))


def test_unit_rule_and_treatment(tmp_path, spy):
    u = units_for(tmp_path, [T_SEP]).set_index("msno")
    assert sorted(u.index) == ["A", "D", "G"]
    assert u.loc["A", "row_id"] == "v1:2" and u.loc["A", "kind"] == "full"
    assert u.loc["D", "kind"] == "discounted" and u.loc["D", "treated"] == 1
    assert u.loc["D", "depth"] == pytest.approx(1 - 129 / 149)
    assert u.loc["G", "kind"] == "below_margin" and u.loc["G", "treated"] == 0


def test_label_reads_are_pre_t_only(tmp_path, spy):
    units_for(tmp_path, [T_SEP, T_DEC])
    assert spy, "the loader was not used"
    for snapshots, columns in spy:
        assert all(column_kind(c) == "pre_t" for c in columns), columns
        assert all(s < date(2017, 1, 1) for s in snapshots)


def test_planted_2017_row_is_never_read(tmp_path):
    tx = ku.read_transactions(write_tx(tmp_path))
    assert pd.to_datetime(tx.transaction_date).max() <= pd.Timestamp("2016-12-31")
    assert "v1:8" not in set(tx.row_id)
    u = units_for(tmp_path, [T_DEC])
    assert "F" not in set(u.msno)  # with the 2017 row, F would be a treated unit


def test_each_renewal_row_belongs_to_one_unit(tmp_path):
    u = units_for(tmp_path, [T_SEP, T_DEC])
    assert u.row_id.is_unique


def test_covariate_history_ignores_rows_on_or_after_t(tmp_path):
    u = units_for(tmp_path, [T_SEP])
    tx = ku.read_transactions(write_tx(tmp_path))
    days = ku.discount_history(u, tx)
    by = dict(zip(u.msno, days, strict=True))
    # D: the renewal on 2016-09-03 and the row on 2016-09-20 are after T; only 2016-05-24 counts
    assert by["D"] == (pd.Timestamp("2016-09-01") - pd.Timestamp("2016-05-24")).days
    assert np.isnan(by["A"])


def test_covariates_carry_no_renewal_row_column(tmp_path):
    u = units_for(tmp_path, [T_SEP])
    cols = [*ku.fs1_columns(), "n_discounted_365"]
    feats = pd.DataFrame({c: [0] * len(u) for c in cols})
    feats["msno"] = u.msno.to_numpy()
    feats["snapshot_date"] = pa.array([T_SEP] * len(u), pa.date32()).to_pandas()
    feats["n_discounted_365"] = [0, 1, 0]
    path = tmp_path / "features.parquet"
    pq.write_table(
        pa.Table.from_pandas(feats, preserve_index=False).set_column(
            feats.columns.get_loc("snapshot_date"),
            "snapshot_date",
            pa.array([T_SEP] * len(u), pa.date32()),
        ),
        path,
    )
    tx = ku.read_transactions(write_tx(tmp_path))
    cov = ku.covariates(u, path, tx)
    renewal_cols = {
        "row_id",
        "transaction_date",
        "expiry_date",
        "plan_list_price",
        "actual_amount_paid",
        "payment_plan_days",
        "payment_method_id",
        "kind",
        "depth",
        "treated",
        "paid_expiry_date",
    }
    assert not renewal_cols & set(cov.columns)
    assert set(cov.columns) == {
        "msno",
        "snapshot_date",
        *ku.fs1_columns(),
        "month_2016_09",
        "any_discount_365",
        "days_since_last_discount",
    }
    assert cov.any_discount_365.tolist() == [0.0, 1.0, 0.0]


def test_no_outcome_column_name_appears_in_the_package():
    pattern = re.compile("|".join(re.escape(p) for p in OUTCOME_LABEL_PREFIXES))
    for path in Path("src/retention/uplift").glob("*.py"):
        code = path.read_text()
        assert not pattern.search(code), f"{path} names an outcome column"


def test_count_check_stops_on_any_difference():
    reg = load()["kkbox"]["counts_before_outcome"]
    good = ku.Counts(
        registered_set_all_methods=reg["registered_set_all_methods"],
        method_23=reg["method_23"],
        method_23_by_month={
            k: {"treated": v["treated"], "untreated": v["untreated"]}
            for k, v in reg["method_23_by_month"].items()
        },
        analysed=dict(reg["analysed"]),
    )
    ku.check_counts(good, reg)
    bad = ku.Counts(**{**good.__dict__, "analysed": {"treated": 70, "untreated": 549}})
    with pytest.raises(ku.CountMismatch, match="analysed"):
        ku.check_counts(bad, reg)


def test_constants_match_the_registered_config():
    k = load()["kkbox"]
    assert k["treatment"]["margin"] == ku.MARGIN
    assert k["treatment"]["plan_days"] == ku.PLAN_DAYS
    assert k["treatment"]["payment_method"] == ku.PAYMENT_METHOD
    assert k["months"]["snapshots"] == ku.SNAPSHOTS
    assert ku.E_BEFORE.isoformat() in k["outcome"]["window_rule"]
    assert k["propensity"]["covariates"]["base"].startswith("feature set fs1")
    assert len(ku.fs1_columns()) == 40
