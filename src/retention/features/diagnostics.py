"""Leak-catching sweeps: single-feature AUC and "value is null" AUC against churned_30.

Diagnostic only: nothing is selected, dropped or tuned on these numbers. They run
on the P2 rows (in_exp30) of snapshots whose 30-day label is reliable, up to and
including SWEEP_LAST; later snapshots are not read until Phase 3 fixes the split.

The headline is the mean of the per-snapshot AUCs; the pooled AUC is secondary.
A column is inspected if its headline or any per-snapshot AUC is above 0.8 or
below 0.2, if its null AUC is outside 0.4-0.6, or if its per-snapshot AUCs span
more than 0.10.

AUC is the Mann-Whitney statistic with average ranks for ties, on non-null rows
only. String and code columns are scored by each category's churn rate in the
other snapshots only (leave one snapshot out), so the AUC is not inflated by
fitting the categories on the rows being scored.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from retention.features import spec

SWEEP_LAST = date(2016, 11, 1)
SINGLE_LIMIT = 0.8
NULL_LOW, NULL_HIGH = 0.4, 0.6
RANGE_LIMIT = 0.10
CATEGORICAL = [
    "segment",
    "paid_state",
    "last_row_kind",
    "gender",
    "age_bucket",
    "city",
    "registered_via",
    "last_payment_method_id",
]


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability that a random churner scores above a random non-churner (ties count half)."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _category_scores(col: pd.Series, y: np.ndarray, snap: np.ndarray) -> np.ndarray:
    """Each row's category scored by that category's churn rate in the other snapshots."""
    key = col.astype("string").fillna("<null>").to_numpy()
    frame = pd.DataFrame({"k": key, "y": y, "s": snap})
    out = np.full(len(frame), np.nan)
    for s in np.unique(snap):
        here = snap == s
        rest = frame[~here]
        rate = rest.groupby("k").y.mean()
        out[here] = pd.Series(key[here]).map(rate).fillna(rest.y.mean()).to_numpy()
    return out


def sweep_rows(features: Path, labels: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """P2 rows of reliable 30-day snapshots up to SWEEP_LAST, with churned_30."""
    cols = ", ".join(f'f."{c}"' for c in spec.FEATURE_COLUMNS)
    df = duckdb.sql(
        f"""
        select f.snapshot_date, l.churned_30, {cols}
        from read_parquet('{features}') f
        join read_parquet('{labels}') l using (msno, snapshot_date)
        where l.in_exp30 and l.label_reliable_30 and l.churned_30 is not null
          and f.snapshot_date <= date '{SWEEP_LAST}'
        """
    ).df()
    used = sorted({str(d)[:10] for d in df.snapshot_date})
    held = duckdb.sql(
        f"select distinct snapshot_date from read_parquet('{labels}') "
        f"where snapshot_date > date '{SWEEP_LAST}' order by 1"
    ).fetchall()
    return df, used, [str(r[0]) for r in held]


def _group(snaps: list[str], year: str) -> list[str]:
    return [s for s in snaps if s.startswith(year)]


def _mean(values) -> float:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def sweep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature AUC summary, and per-snapshot n and events."""
    y = df.churned_30.to_numpy(dtype=bool)
    snap = df.snapshot_date.astype(str).str[:10].to_numpy()
    snaps = sorted(set(snap))
    counts = pd.DataFrame(
        [
            {"snapshot": s, "n": int((snap == s).sum()), "events": int(y[snap == s].sum())}
            for s in snaps
        ]
    )
    rows = []
    for c in spec.FEATURE_COLUMNS:
        col = df[c]
        null = col.isna().to_numpy()
        if c in CATEGORICAL:
            score = _category_scores(col, y, snap)
            keep = np.ones(len(col), dtype=bool)  # null is its own category here
        else:
            score = col.astype("float64").to_numpy()
            keep = ~null
        r: dict[str, Any] = {"column": c, "group": spec.FEATURE_COLUMNS[c]}
        r["scoring"] = "category rate, other snapshots" if c in CATEGORICAL else "value"
        per, per_null, per_share = {}, {}, {}
        for s in snaps:
            m = snap == s
            k = m & keep
            per[s] = auc(score[k], y[k]) if k.any() else np.nan
            per_null[s] = auc(null[m].astype(float), y[m]) if null[m].any() else np.nan
            per_share[s] = float(1 - null[m].mean())
        vals = np.array([per[s] for s in snaps], dtype=float)
        r["auc"] = _mean(vals)
        r["auc_2015"] = _mean(per[s] for s in _group(snaps, "2015"))
        r["auc_2016"] = _mean(per[s] for s in _group(snaps, "2016"))
        ok = vals[~np.isnan(vals)]
        r["auc_range"] = float(ok.max() - ok.min()) if len(ok) else float("nan")
        r["auc_pooled"] = auc(score[keep], y[keep]) if keep.any() else np.nan
        r["non_null_share"] = float(1 - null.mean())
        nulls = np.array([per_null[s] for s in snaps], dtype=float)
        r["null_auc"] = _mean(nulls)
        r["null_auc_pooled"] = auc(null.astype(float), y) if null.any() else np.nan
        r["auc_by_snapshot"] = per
        r["null_auc_by_snapshot"] = per_null
        r["non_null_share_by_snapshot"] = per_share
        singles = [a for a in [r["auc"], *vals] if not np.isnan(a)]
        r["flag_single"] = any(a > SINGLE_LIMIT or a < 1 - SINGLE_LIMIT for a in singles)
        null_vals = [a for a in [r["null_auc"], *nulls] if not np.isnan(a)]
        r["flag_null"] = any(a < NULL_LOW or a > NULL_HIGH for a in null_vals)
        r["flag_range"] = r["auc_range"] > RANGE_LIMIT
        rows.append(r)
    return pd.DataFrame(rows), counts
