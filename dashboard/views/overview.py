"""Overview: the big number, the three buckets with the customers outside the population, one
chart with two panels (C1 bars), then the tabs "Alerts" and "Top 20 at risk" (C2).

Capture mode (?capture=1) omits the top-20 table and the cancel events; nothing else changes.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()
COMPUTING = "Computing the summary: the first call after a store-date move takes several seconds."
TOP = 20


def takeaway(summary: dict[str, Any]) -> str:
    return (
        S.page("overview")["takeaway"]
        .replace("<expected_1_7>", cp.count(summary["buckets"]["days_1_7"]))
        .replace("<scored>", cp.count(summary["counts"]["scored"]))
    )


def buckets(summary: dict[str, Any]) -> None:
    b, c = summary["buckets"], summary["counts"]
    cols = st.columns(4)
    with cols[0]:
        cp.number("Days 1 to 7", cp.count(b["days_1_7"]), tip("days_1_7"))
    with cols[1]:
        cp.number("Days 8 to 30", cp.count(b["days_8_30"]), tip("days_8_30"))
    with cols[2]:
        cp.demo59("Days 31 to 59", cp.count(b["days_31_59"]["value"]), tip("days_31_59"))
    with cols[3]:
        share = c["outside_share_of_active"]
        cp.number(
            "Expiring more than 30 days ahead",
            "-" if share is None else cp.share(share),
            tip("outside"),
        )
    st.caption(f"{b['label']}. {c['outside_label']}")


def reasons(summary: dict[str, Any]) -> None:
    if summary["reasons"] is None:
        st.caption(f"{summary['reasons_withheld']}. {summary['data_warning']}")
        return
    bars = S.c1["overview_bars"]
    st.altair_chart(cp.reason_chart(summary), theme=None)
    for r in summary["reasons"]:
        text = cp.sentence(r)
        if text:
            st.markdown(text, help=tip("reason_share"))
    st.caption(
        bars["wording"].replace("<group>", "the group on each bar")
        + f", {S.raw['labels']['reason_shares'].replace('<horizon>', '7 or 30')}. "
        + f"{summary['reasons'][0]['explains']} {bars['attribution_line']}"
    )


def alerts_tab(client: Client, capture: bool) -> None:
    a = client.alerts()
    st.caption(
        f"{a['trend_knowable']} of {a['trend_in_file']} trend alerts are knowable "
        "at the store's date."
    )
    for t in a["trend"]:
        st.markdown(f"**{cp.esc(t['sentence'])}**", unsafe_allow_html=True, help=tip("alert_rates"))
        st.caption(
            " · ".join(t["labels"])
            + f" · {t['hidden_by_rule']['segments']} segments and "
            + f"{t['hidden_by_rule']['dimension_totals']} dimension totals hidden by rule"
        )
    if not capture:
        for e in a["event"]:
            st.markdown(
                f'<span class="re-id">{cp.esc(e["msno"])}</span> '
                f"{cp.esc(e['date'])}: {cp.esc(e['label'])}",
                unsafe_allow_html=True,
            )


def top_tab(client: Client) -> None:
    page = client.at_risk(horizon=30, limit=TOP, offset=0)
    rows = pd.DataFrame(
        [
            {
                "rank": r["rank"],
                "customer": r["msno"],
                "probability": r["text"],
                "auto_renew": r["auto_renew"],
                "plan action": (r["plan"] or {}).get("action") or "",
            }
            for r in page["rows"]
        ]
    )
    st.dataframe(
        rows,
        hide_index=True,
        column_config={"probability": st.column_config.TextColumn(help=tip("probability"))},
    )
    st.caption(f"{page['header_note']} Plan fields: {page['plan_status']}.")


def render(client: Client, health: dict[str, Any], capture: bool) -> None:
    holder = st.empty()
    holder.info(COMPUTING)
    try:
        summary = client.summary()
    except httpx.TimeoutException:
        return  # the computing state stays on the page
    holder.empty()
    st.header(takeaway(summary), help=tip("days_1_7"))
    cp.big(
        cp.count(summary["buckets"]["days_1_7"]), "expected lapses in days 1 to 7", tip("days_1_7")
    )
    buckets(summary)
    reasons(summary)
    names = ["Alerts"] if capture else ["Alerts", f"Top {TOP} at risk"]
    tabs = st.tabs(names)
    with tabs[0]:
        alerts_tab(client, capture)
    if not capture:
        with tabs[1]:
            top_tab(client)
