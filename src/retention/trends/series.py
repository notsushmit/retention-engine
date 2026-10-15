"""Weekly churn-rate series by expiry week, built from snapshot label rows.

One event is one (customer, expiry date) among snapshot rows with
`in_expiry_month` (the expiry stands at the snapshot's first day and falls in
that calendar month), so each expiry date appears in exactly one snapshot and a
week that spans two months is filled from two snapshots. The week is the ISO
week (Monday start) of `expiry_date`. Churn is `churned_classic` (coverage
family): 1 = no new subscription within 30 days (inclusive) after the expiry.
Rows whose outcome is not observed (NA) are counted separately and are not in
the rate. The segment is the one at the snapshot's first day.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["msno", "snapshot_date", "expiry_date", "in_expiry_month", "segment", "churned_classic"]


def weekly_rates(labels: pd.DataFrame, segment: str, data_end: str = "2017-03-31") -> pd.DataFrame:
    """Per expiry week: events, churned, unobserved, rate and whether the week is complete."""
    missing = [c for c in COLUMNS if c not in labels.columns]
    if missing:
        raise KeyError(f"labels frame lacks {missing}")
    d = labels[labels.in_expiry_month.astype(bool) & (labels.segment == segment)].copy()
    d["week"] = d.expiry_date.dt.to_period("W-SUN").dt.start_time
    if d.duplicated(["msno", "expiry_date"]).any():
        raise ValueError("a customer has the same expiry date in two snapshots")
    g = d.groupby("week")
    out = pd.DataFrame(
        {
            "events": g.churned_classic.count(),
            "churned": g.churned_classic.sum(min_count=0).astype(int),
            "unobserved": g.churned_classic.apply(lambda s: int(s.isna().sum())),
        }
    )
    out["rate"] = out.churned / out.events.where(out.events > 0)
    week_end = out.index + pd.Timedelta(days=6)
    out["complete"] = (week_end + pd.Timedelta(days=30)) <= pd.Timestamp(data_end)
    return out
