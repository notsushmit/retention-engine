"""At-risk list: GET /at-risk a page at a time, in the API's order, with the registered
filters (auto_renew, in_plan, horizon) and a CSV of the displayed rows."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()
F = S.raw["filters"]
SIZES = [50, 100, 500, 1000]  # at most the registered limit of 1,000
ANY = "any"


def horizon_label(h: int) -> str:
    return f"{h} days ({cp.DEMO})" if h == 59 else f"{h} days"


def frame_of(page: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in page["rows"]:
        plan = r["plan"]
        row = {
            "rank": r["rank"],
            "customer": r["msno"],
            "probability": r["probability"],
            "text": r["text"],
            "auto_renew": r["auto_renew"],
            "in_plan": None if plan is None else plan["in_plan"],
            "plan action": None if plan is None else plan["action"],
            "plan label": None if plan is None else plan["label"],
        }
        if page["horizon"] == 59:
            row["grade"] = r["grade"]
        rows.append(row)
    return pd.DataFrame(rows)


def csv_bytes(page: dict[str, Any]) -> bytes:
    """The displayed rows only, with store_date and horizon added."""
    df = frame_of(page)
    df.insert(0, "horizon", page["horizon"])
    df.insert(0, "store_date", page["store_date"])
    return df.to_csv(index=False).encode()


def render(client: Client, health: dict[str, Any], capture: bool) -> None:
    side = st.sidebar
    horizon = side.selectbox("Horizon", F["horizon"], index=1, format_func=horizon_label)
    auto = side.selectbox("Auto-renew", [ANY, "on", "off"])
    first = client.at_risk(horizon=horizon, limit=1, offset=0)
    no_plan = first["plan_status"] != S.serving["api"]["at_risk"]["plan_label"]
    in_plan = side.selectbox(
        "In the plan",
        [ANY, "true", "false"],
        disabled=no_plan,
        help=first["plan_status"] if no_plan else None,
    )
    size = side.selectbox("Rows per page", SIZES)
    number = side.number_input("Page", min_value=1, value=1, step=1)
    page = client.at_risk(
        horizon=horizon,
        limit=size,
        offset=(int(number) - 1) * size,
        auto_renew=None if auto == ANY else auto,
        in_plan=None if in_plan == ANY or no_plan else in_plan,
    )
    st.header(
        S.page("at_risk_list")["takeaway"]
        .replace("<matching>", cp.count(page["matching"]))
        .replace("<horizon>", str(horizon)),
        help=tip("matching"),
    )
    cp.big(cp.count(page["matching"]), "customers match the filters", tip("matching"))
    if horizon == 59:
        cp.demo59("59-day probabilities (demonstration)", "every value in this list", tip("p59"))
    df = frame_of(page)
    st.dataframe(
        df.drop(columns=["probability"]) if not df.empty else df,
        hide_index=True,
        column_config={"text": st.column_config.TextColumn("probability", help=tip("probability"))},
    )
    st.caption(f"{page['header_note']} {page['plan_status']}.")
    c = page["counts"]
    st.caption(
        f"Scored {cp.count(c['scored'])}; outside the population "
        + ", ".join(f"{k} {cp.count(v)}" for k, v in c["outside_population"].items())
        + f"; not active {cp.count(c['not_active'])}.",
        help=tip("scored"),
    )
    st.download_button(
        "Download the displayed rows (CSV)",
        csv_bytes(page),
        file_name=f"at_risk_{page['store_date']}_{horizon}d.csv",
        mime="text/csv",
    )
