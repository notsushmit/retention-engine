"""Monthly data-quality check and label reliability.

For each expiry month M, paid customers whose effective expiry at the start of
M falls in M are followed to their classic outcome. Four indicators flag months
with missing transactions:

- churn: share of those customers who lapsed (bad if high);
- listening: share of churners active on at least N days in a fixed window
  after the lapse (bad if high: they kept using the service);
- returns: share of churners with a paid transaction in a fixed window after
  the lapse (bad if high: they came back soon);
- batch: non-cancel auto-renew rows on the month's last day over the mean per
  day early in the month (bad if low: the month-end billing batch is missing).

Customers whose window runs past as_of are left out of that indicator (for
churn the window is expiry + renewal window; customers inside it whose outcome is
still not visible, such as a cancel that cut coverage near as_of, are counted
as unknown). An
indicator that cannot be computed for a month is "not assessed", never a pass
or a failure. Churn, listening and returns fail above median + k * MAD; the
batch ratio fails below the midpoint of the largest gap in its sorted values.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from retention.labels.config import LabelConfig, QualityConfig
from retention.labels.timeline import outcome_at

INDICATORS = ["batch", "churn", "listening", "returns"]
HIGH_IS_BAD = ["churn", "listening", "returns"]
MAD_SCALE = 1.4826


def months(q: QualityConfig) -> list[pd.Period]:
    return list(pd.period_range(q.first_month, q.last_month, freq="M"))


def batch_ratio(tx: pd.DataFrame, q: QualityConfig) -> pd.Series:
    """Per month: last-day non-cancel auto-renew rows / mean per day on days 1..baseline."""
    ar = tx[(tx.is_cancel == 0) & (tx.is_auto_renew == 1)]
    day = ar.transaction_date
    month = day.dt.to_period("M")
    last = ar[day.dt.is_month_end].groupby(month[day.dt.is_month_end]).size()
    early = ar[day.dt.day <= q.batch_baseline_days]
    base = early.groupby(early.transaction_date.dt.to_period("M")).size() / q.batch_baseline_days
    return (last / base).rename("batch")


def _active_days(logs: str, churners: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    con = duckdb.connect()
    con.sql("set enable_progress_bar = false")
    con.register(
        "ch", churners[["msno", "lapse_date"]].assign(lapse_date=lambda d: d.lapse_date.dt.date)
    )
    days = con.sql(
        f"""select ch.msno, count(distinct l.date) n from ch
            left join {logs} l on l.msno = ch.msno
             and strptime(l.date::varchar, '%Y%m%d')::DATE
                 between ch.lapse_date + {lo} and ch.lapse_date + {hi}
            group by 1"""
    ).df()
    return days.set_index("msno").n


def indicator_table(
    timeline: pd.DataFrame, spells: pd.DataFrame, logs: str, cfg: LabelConfig
) -> pd.DataFrame:
    """One row per expiry month: indicator values and how many customers each used."""
    q = cfg.data_quality
    as_of = pd.Timestamp(cfg.as_of)
    w = cfg.renewal_window_days
    batch = batch_ratio(timeline, q)
    paid_rows = timeline[timeline.row_kind.eq("paid")][["msno", "transaction_date"]]
    rows = []
    for m in months(q):
        start, nxt = m.start_time, (m + 1).start_time
        r: dict[str, Any] = {"month": str(m), "batch": batch.get(m, np.nan)}
        if start >= as_of:
            rows.append(r)
            continue
        o = outcome_at(
            timeline, spells, start.date(), cfg.as_of, w, cfg.snapshot_includes_same_day_rows
        )
        c = o[o.ever_paid & (o.expiry >= start) & (o.expiry < nxt)]
        # Only customers whose renewal window fits before as_of; renewals become known
        # early and lapses late, so using every known outcome would bias churn down.
        fits = c[c.expiry + pd.Timedelta(days=w) <= as_of]
        known = fits[fits.classic.notna()]
        r.update(
            n_cohort=len(c),
            churn_n_used=len(known),
            churn_n_left_out=len(c) - len(fits),
            churn_n_unknown=len(fits) - len(known),
        )
        r["churn"] = known.classic.mean() if len(known) else np.nan
        ch = known[known.classic == 1].reset_index(names="msno")
        for name, (lo, hi) in (("listening", q.listening_days), ("returns", q.returns_days)):
            fits = ch[ch.lapse_date + pd.Timedelta(days=hi) <= as_of]
            r[f"{name}_n_used"], r[f"{name}_n_left_out"] = len(fits), len(ch) - len(fits)
            if fits.empty:
                r[name] = np.nan
                continue
            if name == "listening":
                hit = _active_days(logs, fits, lo, hi) >= q.listening_min_active_days
            else:
                j = fits.merge(paid_rows, on="msno")
                gap = (j.transaction_date - j.lapse_date).dt.days
                hit = pd.Series(True, index=j[(gap >= lo) & (gap <= hi)].msno.unique())
            r[name] = fits.msno.isin(hit[hit].index).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("month")


def thresholds(table: pd.DataFrame, k: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in HIGH_IS_BAD:
        v = table[name].dropna()
        med = float(v.median())
        mad = float(MAD_SCALE * (v - med).abs().median())
        out[name] = {"median": med, "mad": mad, "k": k, "fail_above": med + k * mad}
    v = np.sort(table.batch.dropna().to_numpy())
    i = int(np.argmax(np.diff(v)))
    out["batch"] = {
        "gap_low": float(v[i]),
        "gap_high": float(v[i + 1]),
        "fail_below": float((v[i] + v[i + 1]) / 2),
    }
    return out


def failures(table: pd.DataFrame, th: dict[str, dict[str, float]]) -> pd.DataFrame:
    f = pd.DataFrame(index=table.index)
    for name in HIGH_IS_BAD:
        f[name] = table[name] > th[name]["fail_above"]
    f["batch"] = table.batch < th["batch"]["fail_below"]
    return f[INDICATORS]


def classify(table: pd.DataFrame, q: QualityConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add assessed/failing indicators, flips under k +/- sensitivity, and a status per month."""
    th = {
        s: thresholds(table, q.mad_k * f)
        for s, f in (("base", 1.0), ("low_k", 1 - q.sensitivity), ("high_k", 1 + q.sensitivity))
    }
    assessed = table[INDICATORS].notna()
    fails = {s: failures(table, t) for s, t in th.items()}
    out = table.copy()
    out["assessed"] = assessed.apply(lambda r: ",".join(i for i in INDICATORS if r[i]), axis=1)
    out["n_assessed"] = assessed.sum(axis=1)
    out["fails"] = fails["base"].apply(lambda r: ",".join(i for i in INDICATORS if r[i]), axis=1)
    out["n_fail"] = fails["base"].sum(axis=1)
    out["fails_low_k"] = fails["low_k"].apply(
        lambda r: ",".join(i for i in INDICATORS if r[i]), axis=1
    )
    out["fails_high_k"] = fails["high_k"].apply(
        lambda r: ",".join(i for i in INDICATORS if r[i]), axis=1
    )
    out["flips"] = (out.fails_low_k != out.fails) | (out.fails_high_k != out.fails)
    out["status"] = np.select(
        [
            out.n_fail >= q.unreliable_failures,
            (out.n_fail == 1) | out.flips | (out.n_assessed < q.min_assessed),
        ],
        ["unreliable", "uncertain"],
        "reliable",
    )
    return out, th


def snapshot_status(month_status: pd.Series, cfg: LabelConfig) -> pd.DataFrame:
    """Status per (snapshot, horizon) from every month inside [T - lookback, T + h].

    unknown: T + h is past the last labelable expiry. Otherwise the worst month
    status in the window (unreliable > uncertain > reliable); a month outside
    the checked range counts as uncertain.
    """
    rank = {"reliable": 0, "uncertain": 1, "unreliable": 2}
    names = {v: k for k, v in rank.items()}
    lookback = cfg.lookback_days or 0
    rows = []
    for t in cfg.snapshots:
        for h in (*cfg.horizons, cfg.followup_cap_days):
            if not cfg.fully_observable(t, h):
                status = "unknown"
            else:
                span = pd.period_range(
                    t - timedelta(days=lookback), t + timedelta(days=h), freq="M"
                )
                worst = max(rank[month_status.get(str(p), "uncertain")] for p in span)
                status = names[worst]
            rows.append({"snapshot": t, "horizon": h, "status": status})
    return pd.DataFrame(rows)


def run_quality(
    processed: Path, logs: str, cfg: LabelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timeline = pd.read_parquet(processed / "timeline.parquet")
    spells = pd.read_parquet(processed / "spells.parquet")
    table = indicator_table(timeline, spells, logs, cfg)
    table, th = classify(table, cfg.data_quality)
    snaps = snapshot_status(table.status, cfg)
    return table, snaps, th
