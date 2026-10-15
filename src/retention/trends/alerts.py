"""Alerts for detected change points: flags, drill-down and templated sentence.

Everything here is a pure function of a weekly table, the detector's change
points, gap dates, month statuses and (for the drill-down) segment counts. The
registered rules live in configs/changepoints.yaml (section `alerts`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def segments_from_cps(n: int, cps: list[int]) -> list[tuple[int, int]]:
    """Half-open [start, end) week indexes of the detector's segments."""
    edges = [0, *sorted(cps), n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def week_flags(
    week: pd.Timestamp,
    gap_dates: list[pd.Timestamp],
    month_status: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Flags of one week by the registered rules."""
    end = week + pd.Timedelta(days=6)
    gaps = sorted(d for d in gap_dates if week <= d <= end + pd.Timedelta(days=30))
    months = {str(week.to_period("M")), str(end.to_period("M"))}
    unreliable = sorted(m for m in months if month_status.get(m) == "unreliable")
    uncertain = sorted(m for m in months if month_status.get(m) == "uncertain")
    known = [
        k["label"]
        for k in cfg["flags"]["known_period"]
        if pd.Timestamp(k["from"]) <= week <= pd.Timestamp(k["to"])
    ]
    return {
        "renewal_gap_dates": gaps,
        "unreliable_months": unreliable,
        "uncertain_months": uncertain,
        "left_truncation_risk": bool(week < pd.Timestamp("2016-02-15")),
        "known_periods": known,
    }


def merge_flags(per_week: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({d for f in per_week for d in f["renewal_gap_dates"]})
    return {
        "renewal_gap_dates": dates,
        "unreliable_months": sorted({m for f in per_week for m in f["unreliable_months"]}),
        "uncertain_months": sorted({m for f in per_week for m in f["uncertain_months"]}),
        "left_truncation_risk": any(f["left_truncation_risk"] for f in per_week),
        "known_periods": sorted({k for f in per_week for k in f["known_periods"]}),
    }


def flag_names(f: dict[str, Any]) -> list[str]:
    names = []
    if f["renewal_gap_dates"]:
        names.append("renewal_gap")
    if f["unreliable_months"]:
        names.append("unreliable_month")
    if f["uncertain_months"]:
        names.append("uncertain_month")
    if f["left_truncation_risk"]:
        names.append("left_truncation_risk")
    if f["known_periods"]:
        names.append("known_period")
    return names


def quiet_week_sd(
    rates: np.ndarray, segments: list[tuple[int, int]], flagged: np.ndarray, min_len: int = 8
) -> dict[str, float | int]:
    """Pooled within-segment SD over unflagged weeks in segments of at least `min_len` weeks."""
    ss, n, k = 0.0, 0, 0
    for a, b in segments:
        if b - a < min_len:
            continue
        idx = np.arange(a, b)
        idx = idx[~flagged[a:b]]
        if len(idx) < 2:
            continue
        x = rates[idx]
        ss += float(((x - x.mean()) ** 2).sum())
        n += len(x)
        k += 1
    sd = float(np.sqrt(ss / (n - k))) if n > k else float("nan")
    return {"sd": sd, "weeks": n, "segments": k}


def returned_to_level(
    before_mean: float, quiet_sd: float, next_mean: float | None
) -> tuple[list[float | None], bool | None]:
    """Band and verdict. No quiet-week SD or no later segment: [None, None] and None."""
    if quiet_sd is None or not np.isfinite(quiet_sd):
        return [None, None], None
    band = [before_mean - 2 * quiet_sd, before_mean + 2 * quiet_sd]
    if next_mean is None:
        return band, None
    return band, bool(band[0] <= next_mean <= band[1])


def contributions(
    before: pd.DataFrame, after: pd.DataFrame, min_events: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Per dimension and value: rate and mix contributions (midpoint decomposition).

    Frames have columns dimension, value, events, churned. Returns (ranked rows,
    too-small rows, per-dimension sum of rate and mix contributions)."""
    b = before.set_index(["dimension", "value"])
    a = after.set_index(["dimension", "value"])
    rows: list[dict[str, Any]] = []
    small: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    for dim in sorted(set(b.index.get_level_values(0)) | set(a.index.get_level_values(0))):
        bd = b.loc[dim] if dim in b.index.get_level_values(0) else b.iloc[0:0].droplevel(0)
        ad = a.loc[dim] if dim in a.index.get_level_values(0) else a.iloc[0:0].droplevel(0)
        nb, na = float(bd.events.sum()), float(ad.events.sum())
        total = 0.0
        for v in sorted(set(bd.index) | set(ad.index), key=str):
            eb = float(bd.events.get(v, 0))
            ea = float(ad.events.get(v, 0))
            if eb < min_events or ea < min_events:
                small.append(
                    {
                        "dimension": dim,
                        "value": str(v),
                        "events_before": int(eb),
                        "events_after": int(ea),
                    }
                )
                # a too-small segment still sits in the dimension's total change
                rb = float(bd.churned.get(v, 0)) / eb if eb else 0.0
                ra = float(ad.churned.get(v, 0)) / ea if ea else 0.0
                total += (ea / na) * ra - (eb / nb) * rb
                continue
            rb = float(bd.churned[v]) / eb
            ra = float(ad.churned[v]) / ea
            wb, wa = eb / nb, ea / na
            rate_c = (ra - rb) * (wb + wa) / 2
            mix_c = (wa - wb) * (rb + ra) / 2
            total += rate_c + mix_c
            rows.append(
                {
                    "dimension": dim,
                    "value": str(v),
                    "rate_before": rb,
                    "rate_after": ra,
                    "share_before": wb,
                    "share_after": wa,
                    "rate_contribution": rate_c,
                    "mix_contribution": mix_c,
                    "events_before": int(eb),
                    "events_after": int(ea),
                }
            )
        totals[dim] = total
    return rows, small, totals


def _bucket(x: pd.Series, edges: list[tuple[str, float, float]]) -> pd.Series:
    out = pd.Series("missing", index=x.index, dtype="object")
    for name, lo, hi in edges:
        out[(x >= lo) & (x <= hi)] = name
    return out


def _label(x: pd.Series) -> pd.Series:
    """Integer codes as text; a missing value is the value "missing" (never dropped)."""
    return x.astype("Int64").astype(object).where(x.notna(), "missing").astype(str)


def segment_columns(d: pd.DataFrame) -> dict[str, pd.Series]:
    """The six registered dimensions from feature columns at the snapshot's first day."""
    return {
        "plan_length": _bucket(
            d.last_plan_days,
            [("0-7", 0, 7), ("8-30", 8, 30), ("31-99", 31, 99), ("100+", 100, 1e9)],
        ),
        "payment_method": _label(d.last_payment_method_id),
        "auto_renew": _label(d.last_is_auto_renew),
        "city": _label(d.city),
        "registration_channel": _label(d.registered_via),
        "tenure": _bucket(
            d.tenure_days,
            [("0-89", 0, 89), ("90-364", 90, 364), ("365-729", 365, 729), ("730+", 730, 1e9)],
        ),
    }


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _join(items: list[str]) -> str:
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def segment_text(dimension: str, value: str) -> str:
    names = {
        "plan_length": "plan length {} days",
        "payment_method": "payment method {}",
        "auto_renew": "auto-renew {}",
        "city": "city {}",
        "registration_channel": "registration channel {}",
        "tenure": "tenure {} days",
    }
    v = (
        {"0.0": "off", "1.0": "on", "0": "off", "1": "on"}.get(value, value)
        if dimension == "auto_renew"
        else value
    )
    return names[dimension].format(v)


def primary_template(flags: dict[str, Any], small: bool) -> str:
    if flags["renewal_gap_dates"]:
        return "renewal_gap"
    if flags["unreliable_months"]:
        return "unreliable_month"
    if flags["left_truncation_risk"]:
        return "left_truncation_risk"
    if flags["known_periods"]:
        return "known_period"
    return "small" if small else "normal"


def render(alert: dict[str, Any], cfg: dict[str, Any]) -> str:
    """The one-sentence summary from the registered templates."""
    t = cfg["sentence"]["templates"]
    pc = cfg["sentence"]["pieces"]
    flags = alert["flags"]
    which = primary_template(flags, alert["size_class"] == "small")
    ranked = [segment_text(s["dimension"], s["value"]) for s in alert["top_segments"]]
    segs = pc["segments"].format(list=_join(ranked)) if ranked else pc["segments_none"]
    ret = alert["series_returned_to_level"]
    returned = pc["returned"].format(return_week=alert["return_week"]) if ret else ""
    status_month = None
    if which != "unreliable_month" and flags["unreliable_months"]:
        status_month = (flags["unreliable_months"][0], "unreliable")
    elif flags["uncertain_months"]:
        status_month = (flags["uncertain_months"][0], "uncertain")
    also = pc["also"].format(month=status_month[0], status=status_month[1]) if status_month else ""
    fields = {
        "dates": _join([str(pd.Timestamp(d).date()) for d in flags["renewal_gap_dates"][:3]]),
        "series": alert["series"],
        "before": _fmt_pct(alert["before_rate"]),
        "after": _fmt_pct(alert["after_rate"]),
        "week": alert["week"],
        "moved": "rose" if alert["direction"] == "up" else "fell",
        "month": flags["unreliable_months"][0] if flags["unreliable_months"] else "",
        "status": "unreliable",
        "label": flags["known_periods"][0] if flags["known_periods"] else "",
        "segments": segs,
        "returned": returned,
        "also": also,
    }
    return t[which].format(**fields)
