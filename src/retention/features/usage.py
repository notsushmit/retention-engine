"""Usage features from the daily listening logs.

Each log row is one customer-day. total_secs is cleaned row by row before any
sum: negative values and overflow sentinels (|value| >= SECS_SENTINEL_ABS) are
missing and count as 0 seconds (the day still counts as active: every row has
at least one play); values above SECS_PER_DAY_MAX are capped at it. Each day's
seconds are then rounded to whole milliseconds and summed as integers, so every
sum is exact and does not depend on the order DuckDB adds rows in. The play
counts num_25 ... num_100 and num_unq are used as they are.

For snapshot T a day belongs to a window by its offset T - day (spec.WINDOWS);
day T itself is in no window. Recency and "first listen" read every day before T.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from retention.features import spec
from retention.features.spec import (
    BASELINE_MIN_ACTIVE_DAYS,
    SECS_PER_DAY_MAX,
    SECS_SENTINEL_ABS,
    WINDOWS,
)

# Weights of the least-squares slope over four evenly spaced points, oldest to
# newest, centred at -1.5, -0.5, 0.5, 1.5 (sum of squares 5).
SLOPE_WEIGHTS = np.array([-1.5, -0.5, 0.5, 1.5]) / 5.0

USAGE_COLUMNS = spec.columns_in("usage") + ["invalid_secs_days_28"]


def slope_4w(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slope and relative slope of rows of four weekly points (oldest to newest).

    The relative slope is the slope over the mean of the four points; it is NaN
    when the mean is 0, and lies in [-1.2, 1.2] for non-negative points.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 4)
    slope = pts @ SLOPE_WEIGHTS
    mean = pts.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean > 0, slope / np.where(mean > 0, mean, 1.0), np.nan)
    return slope, rel


def _slope_sql(prefix: str) -> str:
    w = SLOPE_WEIGHTS
    return " + ".join(f"({float(w[i])!r}) * {prefix}{wk}" for i, wk in enumerate(spec.WEEKS))


def _in(window: str) -> str:
    lo, hi = WINDOWS[window]
    return f"o between {lo} and {hi}"


class UsageSource:
    """Cleaned daily logs for a fixed set of customers, held in DuckDB.

    `logs` is a DuckDB relation or SQL table expression with the raw columns
    msno, date (int yyyymmdd), num_25, num_50, num_75, num_985, num_100,
    num_unq, total_secs. Customers are mapped to integer ids to keep memory low.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, logs: str, customers: Sequence[str]):
        self.con = con
        ids = pd.DataFrame({"msno": sorted(set(customers))})
        ids["cid"] = np.arange(len(ids), dtype="int32")
        con.register("_usage_ids", ids)
        con.sql("create or replace table usage_ids as select * from _usage_ids")
        con.unregister("_usage_ids")
        con.sql(
            f"""
            create or replace table usage_daily as
            select i.cid,
              make_date(l.date // 10000, (l.date // 100) % 100, l.date % 100) as day,
              sum(case when l.total_secs < 0 or abs(l.total_secs) >= {SECS_SENTINEL_ABS}
                       then 0
                       else round(least(l.total_secs, {SECS_PER_DAY_MAX}) * 1000) end
                 )::bigint as secs_ms,
              bool_or(l.total_secs < 0 or l.total_secs > {SECS_PER_DAY_MAX}) as secs_invalid,
              sum(l.num_25 + l.num_50 + l.num_75 + l.num_985 + l.num_100)::int32 as plays,
              sum(l.num_100)::int32 as plays_full,
              sum(l.num_unq)::int32 as unq
            from {logs} l join usage_ids i using (msno)
            group by all
            order by day
            """
        )
        con.sql(
            "create or replace table usage_first as "
            "select cid, min(day) as first_day from usage_daily group by cid"
        )

    def features(
        self, t: date, population: Sequence[str], n_chunks: int = 1, keep_weeks: bool = False
    ) -> pd.DataFrame:
        """Usage columns for every msno in `population` at snapshot t.

        The cohort median for ratio_secs_vs_cohort is taken over the whole
        population after all chunks are built, so the result does not depend on
        n_chunks.
        """
        pop = pd.DataFrame({"msno": pd.unique(pd.Series(population, dtype=object))})
        self.con.register("_pop", pop)
        self.con.sql(
            "create or replace temp table usage_pop as "
            "select p.msno, i.cid from _pop p left join usage_ids i using (msno)"
        )
        self.con.unregister("_pop")
        parts = [self._chunk(t, k, n_chunks) for k in range(n_chunks)]
        out = pd.concat(parts, ignore_index=True)
        valid = out.ratio_secs_7_vs_base.dropna()
        median = float(valid.median()) if len(valid) else np.nan
        out["ratio_secs_vs_cohort"] = out.ratio_secs_7_vs_base / median
        out = out.set_index("msno").loc[pop.msno].reset_index()
        weeks = [f"{p}{wk}" for p in ("s_", "a_") for wk in spec.WEEKS] if keep_weeks else []
        return out[["msno", *USAGE_COLUMNS, *weeks]]

    def _chunk(self, t: date, k: int, n: int) -> pd.DataFrame:
        lo28, hi28 = WINDOWS["28"]
        weeks = ",\n".join(
            f"coalesce(sum(secs_ms) filter (where {_in(wk)}), 0) / 7000.0 as s_{wk},\n"
            f"count(*) filter (where {_in(wk)}) as a_{wk}"
            for wk in spec.WEEKS
        )
        t_sql = f"date '{t.isoformat()}'"
        chunk = f"(p.cid is null and {k} = 0) or p.cid % {n} = {k}"
        df = self.con.sql(
            f"""
            with w as (
              select d.*, ({t_sql} - d.day) as o from usage_daily d
              where d.day < {t_sql} and d.day >= {t_sql} - {hi28}
                and d.cid in (select cid from usage_pop p where {chunk})
            ),
            agg as (
              select cid,
                count(*) filter (where {_in("last7")}) as active_days_last7,
                count(*) filter (where {_in("base21")}) as active_days_base21,
                count(*) as active_days_28,
                coalesce(sum(secs_ms) filter (where {_in("last7")}), 0) / 1000.0 as secs_last7,
                coalesce(sum(secs_ms) filter (where {_in("base21")}), 0) / 1000.0
                  as secs_base21,
                coalesce(sum(secs_ms), 0) / 1000.0 as secs_28,
                coalesce(sum(plays) filter (where {_in("last7")}), 0) as plays_last7,
                coalesce(sum(plays), 0) as plays_28,
                coalesce(sum(plays_full) filter (where {_in("last7")}), 0) as full_last7,
                coalesce(sum(plays_full), 0) as full_28,
                coalesce(sum(unq) filter (where {_in("last7")}), 0) as unq_last7,
                coalesce(sum(unq), 0) as unq_28,
                count(*) filter (where secs_invalid) as invalid_secs_days_28,
                {weeks}
              from w group by cid
            ),
            last as (
              select cid, max(day) as last_day from usage_daily
              where day < {t_sql} and cid in (select cid from usage_pop p where {chunk})
              group by cid
            )
            select p.msno,
              (f.first_day < {t_sql})::int8 as has_any_log_before_T,
              {t_sql} - l.last_day as days_since_last_listen,
              (f.first_day < {t_sql} and f.first_day >= {t_sql} - {hi28})::int8
                as first_log_in_window,
              a.*,
              case when coalesce(a.active_days_28, 0) = 0
                     or (f.first_day >= {t_sql} - {hi28} and f.first_day < {t_sql})
                   then null else {_slope_sql("a.s_")} end as slope_secs_4w,
              case when coalesce(a.active_days_28, 0) = 0
                     or (f.first_day >= {t_sql} - {hi28} and f.first_day < {t_sql})
                   then null else {_slope_sql("a.a_")} end as slope_active_4w,
              ({" + ".join(f"a.s_{wk}" for wk in spec.WEEKS)}) / 4.0 as mean_secs_4w,
              ({" + ".join(f"a.a_{wk}" for wk in spec.WEEKS)}) / 4.0 as mean_active_4w
            from usage_pop p
            left join agg a on a.cid = p.cid
            left join last l on l.cid = p.cid
            left join usage_first f on f.cid = p.cid
            where {chunk}
            """
        ).df()
        return _derive(df)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day rates, ratios and slopes from the window sums (pandas, vectorised)."""
    df = df.drop(columns=["cid"])
    count_cols = [c for c in df.columns if c.startswith(("active_", "plays_", "full_", "unq_"))]
    count_cols += [f"a_{wk}" for wk in spec.WEEKS] + ["invalid_secs_days_28"]
    sum_cols = ["secs_last7", "secs_base21", "secs_28"] + [f"s_{wk}" for wk in spec.WEEKS]
    df[count_cols] = df[count_cols].fillna(0).astype("int32")
    df[sum_cols] = df[sum_cols].fillna(0.0).astype("float64")
    df["has_any_log_before_T"] = df.has_any_log_before_T.fillna(0).astype("int8")
    df["first_log_in_window"] = df.first_log_in_window.fillna(0).astype("int8")
    df["days_since_last_listen"] = df.days_since_last_listen.astype("Int32")

    lo, hi = WINDOWS["last7"]
    blo, bhi = WINDOWS["base21"]
    last_days, base_days = hi - lo + 1, bhi - blo + 1
    df["secs_per_day_last7"] = df.secs_last7 / last_days
    df["secs_per_day_base21"] = df.secs_base21 / base_days
    df["secs_per_day_28"] = df.secs_28 / WINDOWS["28"][1]
    active = df.active_days_28 > 0
    df["secs_per_active_day_28"] = np.where(
        active, df.secs_28 / df.active_days_28.where(active, 1), np.nan
    )
    for w in ("last7", "28"):
        plays = df[f"plays_{w}"]
        df[f"complete_share_{w}"] = np.where(
            plays > 0, df[f"full_{w}"] / plays.where(plays > 0, 1), np.nan
        )
    enough = df.active_days_base21 >= BASELINE_MIN_ACTIVE_DAYS
    has_secs = enough & (df.secs_per_day_base21 > 0)
    df["ratio_secs_7_vs_base"] = np.where(
        has_secs, df.secs_per_day_last7 / df.secs_per_day_base21.where(has_secs, 1), np.nan
    )
    df["ratio_active_7_vs_base"] = np.where(
        enough,
        (df.active_days_last7 / last_days) / (df.active_days_base21.where(enough, 1) / base_days),
        np.nan,
    )
    for kind in ("secs", "active"):
        slope = df[f"slope_{kind}_4w"].astype("float64")
        mean = df.pop(f"mean_{kind}_4w").astype("float64")
        ok = slope.notna() & (mean > 0)
        df[f"slope_{kind}_4w"] = slope
        df[f"rel_slope_{kind}_4w"] = np.where(ok, slope / mean.where(ok, 1.0), np.nan)
    return df
