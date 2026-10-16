"""Placebo outcome for the KKBox estimate (configs/uplift.yaml, kkbox.outcome_step.placebo).

The placebo outcome is the number of distinct days with a listening row in days
T-56 to T-29, inclusive: the 28 days just before the 28-day window of the windowed
usage features. Only log rows dated in those days are read. The real discount is dated
after T, so it cannot have changed this outcome; a clear difference is evidence that
the adjustment leaves the groups unlike.

Three covariates read all listening history (ALL_HISTORY). For a unit that did not
listen in the last 28 days they fix or bound the placebo value; leak_categories names
the case per unit, and the placebo step checks the registered count.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PLACEBO_OFFSETS = (29, 56)  # days before T, inclusive
ALL_HISTORY = frozenset({"days_since_last_listen", "has_any_log_before_T", "first_log_in_window"})
# Registered: kept units (primary trimming) whose placebo is fixed or bounded, by arm.
REGISTERED_LEAK = {"untreated": 7, "treated": 10}
FIXED_OR_BOUNDED = (
    "never_listened",
    "first_listen_last_28",
    "last_listen_over_56",
    "last_listen_29_56",
)


class LeakCountMismatch(RuntimeError):
    """The leak count differs from the registered one; the placebo step stops."""


def _yyyymmdd(d: pd.Series) -> pd.Series:
    return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype("int32")


def placebo_days(logs: Path, units: pd.DataFrame) -> np.ndarray:
    """Distinct log days in T-56 to T-29 per unit, in the units' row order (0 if none)."""
    t = pd.to_datetime(units.snapshot_date)
    u = pd.DataFrame(
        {
            "i": np.arange(len(units)),
            "msno": units.msno.to_numpy(),
            "lo": _yyyymmdd(t - pd.Timedelta(days=PLACEBO_OFFSETS[1])).to_numpy(),
            "hi": _yyyymmdd(t - pd.Timedelta(days=PLACEBO_OFFSETS[0])).to_numpy(),
        }
    )
    con = duckdb.connect()
    con.sql("set threads = 4")
    con.register("u", u)
    lo, hi = int(u.lo.min()), int(u.hi.max())
    counts = con.sql(
        f"""
        with l as (
          select distinct msno, date from read_parquet('{logs}')
          where date between {lo} and {hi} and msno in (select msno from u))
        select u.i, count(l.date) as days
        from u left join l on l.msno = u.msno and l.date between u.lo and u.hi
        group by u.i order by u.i
        """
    ).df()
    return counts.days.to_numpy(dtype=float)


def leak_categories(cov: pd.DataFrame) -> np.ndarray:
    """Which all-history covariate fixes or bounds the placebo, per unit."""
    dsll = pd.to_numeric(cov.days_since_last_listen, errors="coerce").to_numpy(dtype=float)
    never = pd.to_numeric(cov.has_any_log_before_T).to_numpy(dtype=float) == 0
    first = pd.to_numeric(cov.first_log_in_window).to_numpy(dtype=float) == 1
    d = np.nan_to_num(dsll, nan=-1.0)
    return np.select(
        [never, first, d > 56, (d >= 29) & (d <= 56)],
        list(FIXED_OR_BOUNDED),
        "not_fixed",
    )


def leak_count(categories: np.ndarray, t: np.ndarray, kept: np.ndarray) -> dict[str, int]:
    hit = np.isin(categories, FIXED_OR_BOUNDED) & kept
    return {"untreated": int((hit & (t == 0)).sum()), "treated": int((hit & (t == 1)).sum())}


def check_leak(count: dict[str, int]) -> None:
    if count != REGISTERED_LEAK:
        raise LeakCountMismatch(f"leak count {count} differs from the registered {REGISTERED_LEAK}")
