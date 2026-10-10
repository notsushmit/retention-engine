"""Exact Python port of WSDMChurnLabeller.scala, the competition's labeller.

Used only to compare with the competition labels; our own labels come from
snapshots.py. Everything here copies the Scala on purpose, including choices we
do not use elsewhere:

- history is the rows dated in [history_start, history_end] (the Scala uses
  2017-01-01 to 2017-01-31); future rows are those dated after history_end;
- rows are sorted by transaction date, then by a string glued from
  plan_list_price, payment_plan_days and payment_method_id in DESCENDING text
  order, then subscribe before cancel, later expiry last for two renewals and
  earlier expiry last for two cancels (Scala lines 102-131);
- the customer's expiry is the expiry on the last history row (line 132), and
  only customers whose expiry lies in [candidate_start, candidate_end] are labelled;
- in the future rows, cancels move the expiry earlier until the first non-cancel
  row, whose transaction date gives the gap (lines 167-190);
- churn means gap >= 30 or no non-cancel future row (lines 80-83).

Raw expiry dates are used, including 1970 placeholders, as the Scala would.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

NO_RENEWAL = 9999


def scala_order(tx: pd.DataFrame) -> pd.DataFrame:
    """Full Scala order: date, glued string descending, cancel flag, expiry key."""
    sig = (
        tx.plan_list_price.astype(str)
        + tx.payment_plan_days.astype(str)
        + tx.payment_method_id.astype(str)
    )
    exp = (pd.to_datetime(tx.membership_expire_date) - pd.Timestamp("1970-01-01")).dt.days
    out = tx.assign(_sig=sig, _exp_key=np.where(tx.is_cancel == 0, exp, -exp))
    return out.sort_values(
        ["msno", "transaction_date", "_sig", "is_cancel", "_exp_key"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )


def scala_labels(
    tx: pd.DataFrame,
    history_start: date,
    history_end: date,
    candidate_start: date,
    candidate_end: date,
) -> pd.DataFrame:
    """Per customer with a history row: last_expire, candidate, gap, is_churn (candidates only)."""
    tx = tx.assign(
        transaction_date=pd.to_datetime(tx.transaction_date),
        membership_expire_date=pd.to_datetime(tx.membership_expire_date),
    )
    tx = scala_order(tx)
    hs, he = pd.Timestamp(history_start), pd.Timestamp(history_end)
    hist = tx[(tx.transaction_date >= hs) & (tx.transaction_date <= he)]
    last = hist.groupby("msno", sort=False).membership_expire_date.last()
    out = pd.DataFrame({"last_expire": last})
    out["candidate"] = (last >= pd.Timestamp(candidate_start)) & (
        last <= pd.Timestamp(candidate_end)
    )

    fut = tx[(tx.transaction_date > he) & tx.msno.isin(out.index[out.candidate])].copy()
    renewal = fut.is_cancel.eq(0)
    fut["n_before"] = renewal.astype(int).groupby(fut.msno).cumsum() - renewal.astype(int)
    pre = fut[fut.n_before == 0]
    cut = pre[pre.is_cancel == 1].groupby("msno").membership_expire_date.min()
    first = pre[pre.is_cancel == 0].groupby("msno").transaction_date.first()
    le = out.last_expire[out.candidate]
    le = pd.concat([le, cut.reindex(le.index)], axis=1).min(axis=1)
    gap = (first.reindex(le.index) - le).dt.days
    out["gap"] = pd.array(gap.fillna(NO_RENEWAL).reindex(out.index), dtype="Int32")
    churn = (out.gap.fillna(0) >= 30).astype(int)
    out["is_churn"] = pd.array(np.where(out.candidate, churn, pd.NA), dtype="Int8")
    return out.rename_axis("msno")
