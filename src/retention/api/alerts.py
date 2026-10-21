"""GET /alerts: the Phase 5 trend alerts knowable at the store's date, and cancel events.

Trend alerts come from the pinned alerts file (configs/serving.yaml, step5.pins),
read once at startup and never written. An alert is knowable at the store's date
T when its week start plus the registered number of days (37) is on or before T.
Before it is served, every segment and dimension total in a hidden dimension
(city, registration channel, tenure; api.alerts and amendment A7) is removed from
top_segments, too_small_segments and dimension_totals, and the number removed is
shown; every other stored field, the data-quality flags included, is served as
stored, with the three registered labels.

Event alerts are the cancel rows the replay (step 6) appends to the store's event
list, each labelled "a cancel row, not a change in risk"; the list is empty
before any replay.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from retention.api import schemas as sch
from retention.api import store
from retention.api.labels import LABELS, Labels

SEGMENT_LISTS = ("top_segments", "too_small_segments")
TOTALS = "dimension_totals"


def knowable_from(week: date, labels: Labels = LABELS) -> date:
    return week + timedelta(days=labels.knowable_days)


def is_knowable(week: date, store_date: date, labels: Labels = LABELS) -> bool:
    """The registered rule: week start + 37 days on or before the store's date."""
    return knowable_from(week, labels) <= store_date


def served_alert(alert: dict[str, Any], labels: Labels = LABELS) -> sch.TrendAlert:
    """One stored alert with the hidden dimensions removed and counted."""
    hidden = set(labels.hidden_dimensions)
    fields = {k: v for k, v in alert.items() if k not in (*SEGMENT_LISTS, TOTALS)}
    kept = {k: [s for s in alert[k] if s["dimension"] not in hidden] for k in SEGMENT_LISTS}
    removed = {k: len(alert[k]) - len(kept[k]) for k in SEGMENT_LISTS}
    totals = {k: v for k, v in alert[TOTALS].items() if k not in hidden}
    week = date.fromisoformat(alert["week"])
    return sch.TrendAlert(
        kind="trend",
        knowable_from=knowable_from(week, labels),
        labels=labels.trend_labels,
        hidden_by_rule=sch.HiddenByRule(
            top_segments=removed["top_segments"],
            too_small_segments=removed["too_small_segments"],
            segments=sum(removed.values()),
            dimension_totals=len(alert[TOTALS]) - len(totals),
        ),
        **kept,
        dimension_totals=totals,
        **fields,
    )


def check_run(run: dict[str, Any]) -> None:
    """Every stored alert can be served (refused at startup otherwise)."""
    for alert in run["alerts"]:
        served_alert(alert)


def knowable_count(run: dict[str, Any], store_date: date) -> int:
    return sum(is_knowable(date.fromisoformat(a["week"]), store_date) for a in run["alerts"])


def event_alerts(r, keys: store.Keys = store.KEYS) -> list[sch.EventAlert]:
    """The cancel rows the replay appended, oldest first (none before any replay)."""
    out = []
    for raw in r.lrange(keys.events, 0, -1):
        e = json.loads(raw)
        out.append(sch.EventAlert(kind="event", label=LABELS.event_label, **e))
    return out


def alerts_body(run: dict[str, Any], r, store_date: date) -> dict[str, Any]:
    trend = [
        served_alert(a)
        for a in run["alerts"]
        if is_knowable(date.fromisoformat(a["week"]), store_date)
    ]
    body = sch.Alerts(
        store_date=store_date,
        trend_in_file=len(run["alerts"]),
        trend_knowable=len(trend),
        trend=tuple(trend),
        event=tuple(event_alerts(r)),
    )
    return body.model_dump(mode="json")
