"""Transaction features from the subscription timeline.

Reads these timeline columns only: msno, seq (timeline order), transaction_date,
is_cancel, row_kind, payment_plan_days, plan_list_price, actual_amount_paid,
is_auto_renew, payment_method_id and eff_expiry_after (effective expiry after
the row). Every timeline value of a row depends only on that row and earlier
rows, and every query below keeps rows dated strictly before T, so nothing
about the renewal at or after T is read. spells.parquet is not used.

A prior lapse is a non-cancel row dated more than `window` days after the
effective expiry left by the row before it (a return after a lapse), counted
when the returning row is dated in [T - 365, T - 1].
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import duckdb
import pandas as pd

from retention.data.quality import STATUS_RANK
from retention.features.spec import COUNT_WINDOW, columns_in

TX_FEATURES = columns_in("transactions") + columns_in("counts_365")
# The last non-cancel row before T: latest date, then latest expiry (as in the
# timeline order), then content, so same-day twins with the same expiry give the
# same answer whatever their position in the raw files: the highest amount paid,
# then list price, plan days, payment method, auto-renew and row kind, all
# descending. Rows equal in all of these are interchangeable.
LAST_ROW_ORDER = (
    "d desc, expiry desc nulls last, actual_amount_paid desc, plan_list_price desc, "
    "payment_plan_days desc, payment_method_id desc, is_auto_renew desc, row_kind desc"
)
TX_METADATA = [
    "last_txn_span_status",
    "n_txn_365_in_nonreliable_months",
    "share_txn_365_in_nonreliable_months",
    "counts_365_days_observed",
]


class TransactionSource:
    """Timeline rows in DuckDB, with each row's gap to the previous effective expiry."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        timeline: str,
        month_status: pd.Series,
        window: int,
        data_start: date,
    ):
        self.con = con
        self.month_status = month_status
        self.data_start = data_start
        con.sql(
            f"""
            create or replace table tx_rows as
            select msno, seq, transaction_date::date as d, is_cancel, row_kind,
              payment_plan_days, plan_list_price, actual_amount_paid, is_auto_renew,
              payment_method_id, eff_expiry_after::date as eff, expiry_date::date as expiry,
              source, source_row,
              coalesce(
                is_cancel = 0
                and (transaction_date::date
                     - (lag(eff_expiry_after) over w)::date) > {int(window)},
                false) as lapse_return,
              strftime(transaction_date, '%Y-%m') as month
            from {timeline}
            window w as (partition by msno order by seq)
            """
        )
        bad = pd.DataFrame(
            {"month": month_status.index.astype(str), "bad": month_status.ne("reliable").values}
        )
        con.register("_tx_months", bad)
        con.sql("create or replace table tx_months as select * from _tx_months")
        con.unregister("_tx_months")

    def features(
        self, t: date, population: Sequence[str], check_columns: bool = False
    ) -> pd.DataFrame:
        """Transaction columns for every msno in `population` at snapshot t.

        check_columns adds _last_eff_days and _last_expiry_days (effective and
        own expiry of the last row of any kind, in days from T) for cross-checks.
        """
        pop = pd.DataFrame({"msno": pd.unique(pd.Series(population, dtype=object))})
        self.con.register("_tx_pop", pop)
        ts = f"date '{t.isoformat()}'"
        lo = f"{ts} - {COUNT_WINDOW}"
        df = self.con.sql(
            f"""
            with b as (
              select r.* from tx_rows r join _tx_pop p using (msno) where r.d < {ts}
            ),
            lastnc as (
              select * from b where is_cancel = 0
              qualify row_number() over (partition by msno order by {LAST_ROW_ORDER}) = 1
            ),
            lastany as (
              select msno, is_cancel, eff, expiry from b
              qualify row_number() over (partition by msno order by seq desc) = 1
            ),
            agg as (
              select b.msno,
                max(d) filter (where is_cancel = 0 and actual_amount_paid > 0) as last_pay,
                count(*) filter (where is_cancel = 0 and d >= {lo}) as n_txn_365,
                count(*) filter (where is_cancel = 0 and actual_amount_paid > 0 and d >= {lo})
                  as n_paid_365,
                count(*) filter (where is_cancel = 0 and d >= {lo} and (
                  (actual_amount_paid > 0 and actual_amount_paid < plan_list_price)
                  or row_kind = 'discount100')) as n_discounted_365,
                count(*) filter (where is_cancel = 1 and d >= {lo}) as n_cancels_365,
                max(d) filter (where is_cancel = 1 and d >= {lo}) as last_cancel,
                count(*) filter (where lapse_return and d >= {lo}) as n_prior_lapses_365,
                count(*) filter (where is_cancel = 0 and d >= {lo} and coalesce(m.bad, true))
                  as n_txn_365_in_nonreliable_months
              from b left join tx_months m using (month)
              group by b.msno
            )
            select p.msno,
              n.payment_plan_days as last_plan_days,
              (n.payment_plan_days = 0)::int8 as last_plan_days_missing,
              n.plan_list_price as last_list_price,
              n.actual_amount_paid as last_amount_paid,
              case when n.plan_list_price > 0
                   then 1 - n.actual_amount_paid / n.plan_list_price end as last_discount_ratio,
              n.is_auto_renew as last_is_auto_renew,
              n.payment_method_id as last_payment_method_id,
              n.row_kind as last_row_kind,
              la.is_cancel::int8 as last_row_is_cancel,
              {ts} - n.d as days_since_last_txn,
              {ts} - a.last_pay as days_since_last_payment,
              a.n_txn_365, a.n_paid_365, a.n_discounted_365, a.n_cancels_365,
              {ts} - a.last_cancel as days_since_last_cancel,
              a.n_prior_lapses_365,
              a.n_txn_365_in_nonreliable_months,
              n.d as _last_txn_date,
              la.eff - {ts} as _last_eff_days,
              la.expiry - {ts} as _last_expiry_days
            from _tx_pop p
            left join lastnc n using (msno)
            left join lastany la using (msno)
            left join agg a using (msno)
            """
        ).df()
        self.con.unregister("_tx_pop")
        df = df.set_index("msno").loc[pop.msno].reset_index()  # population order
        out = self._finish(df, t)
        if check_columns:
            out["_last_eff_days"] = df._last_eff_days.astype("Int32")
            out["_last_expiry_days"] = df._last_expiry_days.astype("Int32")
        return out

    def _finish(self, df: pd.DataFrame, t: date) -> pd.DataFrame:
        counts = [
            "n_txn_365",
            "n_paid_365",
            "n_discounted_365",
            "n_cancels_365",
            "n_prior_lapses_365",
            "n_txn_365_in_nonreliable_months",
        ]
        df[counts] = df[counts].fillna(0).astype("int32")
        for c in ("days_since_last_txn", "days_since_last_payment", "days_since_last_cancel"):
            df[c] = df[c].astype("Int32")
        for c in ("last_plan_days", "last_list_price", "last_amount_paid"):
            df[c] = df[c].astype("Int32")
        for c in ("last_plan_days_missing", "last_is_auto_renew", "last_row_is_cancel"):
            df[c] = df[c].astype("Int8")
        df["last_payment_method_id"] = df.last_payment_method_id.astype("Int16")
        df["last_discount_ratio"] = df.last_discount_ratio.astype("float64")
        has = df.n_txn_365 > 0
        df["share_txn_365_in_nonreliable_months"] = (
            df.n_txn_365_in_nonreliable_months / df.n_txn_365.where(has)
        ).astype("float64")
        df["counts_365_days_observed"] = min(COUNT_WINDOW, (t - self.data_start).days)
        df["counts_365_days_observed"] = df.counts_365_days_observed.astype("int16")
        df["last_txn_span_status"] = self._span_status(df.pop("_last_txn_date"), t)
        return df[["msno", *TX_FEATURES, *TX_METADATA]]

    def _span_status(self, last: pd.Series, t: date) -> pd.Series:
        """Worst completeness status of the months from the last row's month to T - 1."""
        names = {v: k for k, v in STATUS_RANK.items()}
        end = pd.Period(t - timedelta(days=1), freq="M")
        cache: dict[str, str] = {}

        def worst(start: pd.Period) -> str:
            key = str(start)
            if key not in cache:
                span = pd.period_range(start, end, freq="M")
                cache[key] = names[
                    max(STATUS_RANK[self.month_status.get(str(p), "uncertain")] for p in span)
                ]
            return cache[key]

        months = pd.to_datetime(last).dt.to_period("M")
        return months.map(lambda p: worst(p) if pd.notna(p) else None).astype(object)
