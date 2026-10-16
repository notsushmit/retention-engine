"""Price per day at snapshot T, from the last paid transaction dated before T.

The row: the last non-cancel transaction dated strictly before T with
actual_amount_paid > 0 and payment_plan_days > 0. Ties: latest transaction
date, then latest expiry, then highest amount paid (registered); then list
price, plan days and payment method, descending, only so that same-day twins
give one answer whatever their file order. Price per day = amount paid / plan
days. A customer with no such row has no usable price; their value is 0.

A free row (trial, promotion, 100% discount) after a paid one does not hide the
paid row: the feature `last_amount_paid` describes the last row of any price,
so it is not used here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

CLEAN_TRANSACTIONS = Path("data/processed/kkbox/sample_020/transactions.parquet")
PRICE_COLUMNS = [
    "msno",
    "transaction_date",
    "is_cancel",
    "actual_amount_paid",
    "payment_plan_days",
    "plan_list_price",
    "payment_method_id",
    "expiry_date",
]
PAID_ROW_ORDER = (
    "transaction_date desc, expiry_date desc nulls last, actual_amount_paid desc, "
    "plan_list_price desc, payment_plan_days desc, payment_method_id desc"
)


def price_per_day(transactions: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """One row per population row (msno, snapshot_date), in population order.

    Columns added: price_per_day (0.0 when there is no usable price),
    has_price, price_row_date (the date of the row used; missing when none).
    """
    con = duckdb.connect()
    try:
        con.register("tx", transactions[PRICE_COLUMNS])
        pop = population[["msno", "snapshot_date"]].reset_index(drop=True).copy()
        pop["_pos"] = range(len(pop))
        con.register("pop", pop)
        out = con.sql(
            f"""
            with paid as (
              select p._pos, t.actual_amount_paid, t.payment_plan_days,
                t.transaction_date::date as d
              from pop p join tx t
                on t.msno = p.msno and t.transaction_date::date < p.snapshot_date::date
              where t.is_cancel = 0 and t.actual_amount_paid > 0 and t.payment_plan_days > 0
              qualify row_number() over (partition by p._pos order by {PAID_ROW_ORDER}) = 1
            )
            select p._pos, p.msno, p.snapshot_date::date as snapshot_date,
              coalesce(paid.actual_amount_paid / paid.payment_plan_days, 0.0) as price_per_day,
              paid.d is not null as has_price,
              paid.d as price_row_date
            from pop p left join paid using (_pos)
            order by p._pos
            """
        ).df()
    finally:
        con.close()
    return out.drop(columns="_pos")


def read_transactions(snapshots: list[date], path: Path = CLEAN_TRANSACTIONS) -> pd.DataFrame:
    """The price columns of rows dated before the latest snapshot (nothing later is read)."""
    latest = max(snapshots)
    con = duckdb.connect()
    try:
        return con.sql(
            f"select {', '.join(PRICE_COLUMNS)} from read_parquet('{path}') "
            f"where transaction_date < date '{latest.isoformat()}'"
        ).df()
    finally:
        con.close()
