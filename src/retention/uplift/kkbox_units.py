"""KKBox renewal units, treatment and pre-treatment covariates (configs/uplift.yaml, kkbox).

Reads, and nothing else:
- labels.parquet: only the pre-T columns paid_state and paid_expiry_date, through the
  lock-aware loader, for the named snapshots;
- transactions.parquet: rows dated on or before 2016-12-31 only;
- features.parquet: fs1 columns and n_discounted_365 at T.
No outcome column of any snapshot is read here, and no function in this module reads
one. Covariates come from features at T or from transactions dated before T; the
renewal row after T supplies only the treatment.

Unit: for customers whose paid coverage is active at T and whose paid expiry falls in
T's calendar month, the first non-cancel transaction dated on or after T and on or
before that expiry plus 30 days; same-day ties go to the lowest row_id in string order
(row ids look like "v1:2381392", so this is text order, as in the counts run that set
the registered counts; no tie mixes a discounted row with a full-price one).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retention.models.classifier import feature_set_columns
from retention.models.data import load_features, load_labels
from retention.models.protocol import Protocol

TX_CUTOFF = date(2016, 12, 31)
LABEL_COLUMNS = ["paid_state", "paid_expiry_date"]
TX_COLUMNS = [
    "row_id",
    "msno",
    "transaction_date",
    "expiry_date",
    "is_cancel",
    "row_kind",
    "plan_list_price",
    "actual_amount_paid",
    "payment_plan_days",
    "payment_method_id",
]
SNAPSHOTS = [date(2016, 7, 1), date(2016, 8, 1), date(2016, 9, 1), date(2016, 10, 1)]
MARGIN = 0.05
PLAN_DAYS = 30
PAYMENT_METHOD = 23
E_BEFORE = date(2016, 11, 5)
RENEWAL_WINDOW_DAYS = 30
FS1_DROP = ["counts_365"]
CATEGORICAL = ["last_payment_method_id", "registered_via", "segment", "paid_state", "last_row_kind"]


class CountMismatch(RuntimeError):
    """A recount differs from a registered count; the build stops."""


def read_transactions(path: Path) -> pd.DataFrame:
    """Transaction columns needed here, rows dated on or before the cutoff only."""
    cols = ", ".join(TX_COLUMNS)
    return duckdb.sql(
        f"select {cols} from read_parquet('{path}') "
        f"where transaction_date <= DATE '{TX_CUTOFF.isoformat()}'"
    ).df()


def read_members(labels: Path, snapshots: list[date], protocol: Protocol) -> pd.DataFrame:
    """Customers with active paid coverage at T and a paid expiry in T's month (pre-T only)."""
    lab = load_labels(labels, snapshots, LABEL_COLUMNS, protocol)
    t = pd.to_datetime(lab.snapshot_date)
    exp = pd.to_datetime(lab.paid_expiry_date)
    keep = lab.paid_state.eq("active") & (exp < t + pd.offsets.MonthBegin())
    return pd.DataFrame(
        {"msno": lab.msno[keep], "snapshot_date": t[keep], "paid_expiry_date": exp[keep]}
    ).reset_index(drop=True)


def renewal_units(members: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """The first non-cancel transaction on or after T and by the paid expiry plus 30 days."""
    con = duckdb.connect()
    con.register("m", members)
    con.register("tx", tx)
    units = con.sql(
        f"""
        select * exclude (rn) from (
          select m.snapshot_date, m.paid_expiry_date, tx.*,
            row_number() over (partition by m.msno, m.snapshot_date
                               order by tx.transaction_date, tx.row_id) as rn
          from m join tx on tx.msno = m.msno
          where tx.is_cancel = 0
            and tx.transaction_date >= m.snapshot_date
            and tx.transaction_date <= m.paid_expiry_date
                + interval {RENEWAL_WINDOW_DAYS} day)
        where rn = 1
        """
    ).df()
    return classify(units)


def classify(units: pd.DataFrame) -> pd.DataFrame:
    """Kind of the renewal row and the treatment flag (margin on depth = 1 - paid / list)."""
    out = units.copy()
    lp, ap = out.plan_list_price.astype(float), out.actual_amount_paid.astype(float)
    depth = np.where(lp > 0, 1 - ap / lp.where(lp > 0, 1.0), np.nan)
    kind = np.select(
        [
            out.row_kind.eq("discount100"),
            out.row_kind.isin(["trial", "promo"]),
            (lp == 0) & (ap > 0),
            ap > lp,
            ap == lp,
            depth >= MARGIN,
        ],
        ["pct100", out.row_kind, "blank_plan", "above_list", "full", "discounted"],
        "below_margin",
    )
    out["kind"] = kind
    out["depth"] = depth
    out["treated"] = (kind == "discounted").astype(int)
    return out


def registered_set(units: pd.DataFrame) -> pd.DataFrame:
    """Full-price or discounted 30-day renewals whose expiry E is before the window limit."""
    e = pd.to_datetime(units.expiry_date)
    keep = (
        units.kind.isin(["full", "discounted"])
        & units.payment_plan_days.eq(PLAN_DAYS)
        & (e < pd.Timestamp(E_BEFORE))
        & pd.to_datetime(units.snapshot_date).isin(pd.to_datetime(SNAPSHOTS))
    )
    return units[keep].reset_index(drop=True)


def analysed_set(reg: pd.DataFrame) -> pd.DataFrame:
    """Method 23, then drop months without both arms (structural overlap rule)."""
    m = reg[reg.payment_method_id.eq(PAYMENT_METHOD)]
    arms = m.groupby("snapshot_date").treated.agg(["min", "max"])
    both = arms.index[(arms["min"] == 0) & (arms["max"] == 1)]
    return m[m.snapshot_date.isin(both)].reset_index(drop=True)


@dataclass(frozen=True)
class Counts:
    registered_set_all_methods: int
    method_23: int
    method_23_by_month: dict[str, dict[str, int]]
    analysed: dict[str, int]


def recount(reg: pd.DataFrame, analysed: pd.DataFrame) -> Counts:
    m = reg[reg.payment_method_id.eq(PAYMENT_METHOD)]
    by_month = {}
    for t in SNAPSHOTS:
        g = m[pd.to_datetime(m.snapshot_date) == pd.Timestamp(t)]
        by_month[t.strftime("%Y-%m")] = {
            "treated": int(g.treated.sum()),
            "untreated": int((g.treated == 0).sum()),
        }
    return Counts(
        registered_set_all_methods=len(reg),
        method_23=len(m),
        method_23_by_month=by_month,
        analysed={
            "treated": int(analysed.treated.sum()),
            "untreated": int((analysed.treated == 0).sum()),
        },
    )


def check_counts(counts: Counts, registered: dict) -> None:
    """Raise CountMismatch naming every count that differs from the registered one."""
    expected = {
        "registered_set_all_methods": registered["registered_set_all_methods"],
        "method_23": registered["method_23"],
        "method_23_by_month": {
            str(k): {"treated": v["treated"], "untreated": v["untreated"]}
            for k, v in registered["method_23_by_month"].items()
        },
        "analysed": dict(registered["analysed"]),
    }
    got = {
        "registered_set_all_methods": counts.registered_set_all_methods,
        "method_23": counts.method_23,
        "method_23_by_month": counts.method_23_by_month,
        "analysed": counts.analysed,
    }
    diff = {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]}
    if diff:
        raise CountMismatch(f"recount differs from the registered counts: {diff}")


def fs1_columns() -> list[str]:
    return feature_set_columns(FS1_DROP)


def discount_history(analysed: pd.DataFrame, tx: pd.DataFrame) -> pd.Series:
    """Days from the latest discounted row in the 365 days before T to T; NaN if none.

    Uses rows dated before T only (a diagnostic, not a model input).
    """
    con = duckdb.connect()
    con.register("a", analysed[["msno", "snapshot_date"]])
    con.register("tx", tx)
    hist = con.sql(
        """
        select msno, snapshot_date, date_diff('day', last_discount, snapshot_date) as days
        from (
          select a.msno, a.snapshot_date, max(tx.transaction_date) as last_discount
          from a join tx on tx.msno = a.msno
          where tx.is_cancel = 0 and tx.plan_list_price > 0
            and tx.actual_amount_paid < tx.plan_list_price
            and tx.transaction_date < a.snapshot_date
            and tx.transaction_date >= a.snapshot_date - interval 365 day
          group by a.msno, a.snapshot_date)
        """
    ).df()
    merged = analysed[["msno", "snapshot_date"]].merge(
        hist, on=["msno", "snapshot_date"], how="left"
    )
    return merged.days.astype(float)


def covariates(analysed: pd.DataFrame, features: Path, tx: pd.DataFrame) -> pd.DataFrame:
    """Pre-T covariates per analysed unit, in the analysed set's row order."""
    snaps = sorted({d.date() for d in pd.to_datetime(analysed.snapshot_date)})
    cols = [*fs1_columns(), "n_discounted_365"]
    f = load_features(features, snaps, cols)
    f["snapshot_date"] = pd.to_datetime(f.snapshot_date)
    keys = analysed[["msno", "snapshot_date"]].assign(
        snapshot_date=pd.to_datetime(analysed.snapshot_date)
    )
    out = keys.merge(f, on=["msno", "snapshot_date"], how="left", validate="one_to_one")
    out["month_2016_09"] = (out.snapshot_date == pd.Timestamp("2016-09-01")).astype(float)
    out["any_discount_365"] = (out.n_discounted_365.fillna(0) > 0).astype(float)
    out["days_since_last_discount"] = discount_history(analysed, tx).to_numpy()
    return out.drop(columns=["n_discounted_365"])
