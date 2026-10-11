"""Consistency checks between independently computed column groups.

The build runs them on every snapshot and stops if any fails.
"""

from __future__ import annotations

import pandas as pd

LAST_COLUMNS = [
    "last_plan_days",
    "last_plan_days_missing",
    "last_list_price",
    "last_amount_paid",
    "last_is_auto_renew",
    "last_payment_method_id",
    "last_row_kind",
    "last_row_is_cancel",
    "days_since_last_txn",
]


def cross_check(df: pd.DataFrame) -> list[str]:
    """Messages for every failed check; empty when all hold.

    Expects transaction columns (with the _last_eff_days / _last_expiry_days
    check columns) joined to days_until_expiry, paid_state and
    days_until_paid_expiry.
    """
    errors = []
    for c in LAST_COLUMNS:
        n = int(df[c].isna().sum())
        if n:
            errors.append(f"{c}: {n} nulls (every row has a non-cancel row before T)")
    n = int((df.days_since_last_txn < 1).sum())
    if n:
        errors.append(f"last non-cancel row dated on or after T: {n} rows")
    never = df.paid_state.eq("never_paid")
    n = int((never != df.days_since_last_payment.isna()).sum())
    if n:
        errors.append(f"paid_state never_paid vs days_since_last_payment null: {n} rows differ")
    n = int((df.paid_state.eq("active") & ~(df.days_until_paid_expiry >= 0)).sum())
    if n:
        errors.append(f"paid_state active with days_until_paid_expiry < 0: {n} rows")
    sub = df[df.last_row_is_cancel == 0]
    n = int((sub.days_until_expiry != sub._last_eff_days).sum())
    if n:
        errors.append(f"days_until_expiry != last row's effective expiry: {n} rows")
    n = int((sub.days_until_expiry < sub._last_expiry_days).sum())
    if n:
        errors.append(f"days_until_expiry below the last row's own expiry: {n} rows")
    return errors
