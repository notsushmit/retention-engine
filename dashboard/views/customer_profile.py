"""Customer profile: POST /score for one customer, every label as served."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import components as cp
from dashboard.client import ApiError, Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()
T = S.tokens


def points(risk: dict[str, Any]) -> alt.LayerChart:
    """The three probabilities as three points; the 59-day point hollow, grey and badged."""
    df = pd.DataFrame(
        [
            {"days": h, "p": risk[f"days_{h}"]["probability"], "label": risk[f"days_{h}"]["text"]}
            for h in (7, 30, 59)
        ]
    )
    df.loc[df.days == 59, "label"] = df.loc[df.days == 59, "label"] + f" ({cp.DEMO})"
    base = alt.Chart(df).encode(
        x=alt.X("days:O", title="days"),
        y=alt.Y("p:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1])),
    )
    solid = base.transform_filter("datum.days != 59").mark_point(
        filled=True, size=120, color=T["ink"]
    )
    demo = base.transform_filter("datum.days == 59").mark_point(
        filled=False, size=120, color=T["label_grey"], strokeDash=[3, 2]
    )
    labels = base.mark_text(dy=-14, color=T["ink"], font="JetBrains Mono").encode(text="label:N")
    return (
        (solid + demo + labels)
        .properties(width=360, height=180)
        .configure(background=T["background"], font="Space Grotesk")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=T["body_grey"],
            titleColor=T["body_grey"],
            domainColor=T["label_grey"],
            tickColor=T["label_grey"],
        )
    )


def scored(body: dict[str, Any]) -> None:
    risk = body["risk"]
    st.header(
        S.page("customer_profile")["takeaway"].replace(
            "<text of the 30-day probability>", risk["days_30"]["text"]
        ),
        help=tip("p30"),
    )
    cp.big(risk["days_30"]["text"], "chance of lapsing within 30 days", tip("p30"))
    st.altair_chart(points(risk), theme=None)
    cols = st.columns(3)
    with cols[0]:
        cp.number("7 days", risk["days_7"]["text"], tip("p7"))
    with cols[1]:
        cp.number("30 days", risk["days_30"]["text"], tip("p30"))
    with cols[2]:
        cp.demo59("59 days", risk["days_59"]["text"], tip("p59"))
    if risk["moved_by_ordering_fix"]:
        st.caption("The ordering fix moved this customer's probabilities.")
    if body["reasons"] is None:
        st.caption(f"{body['reasons_withheld']}. {body['data_warning']}")
    else:
        for r in body["reasons"]:
            st.markdown(
                f"**{r['horizon']} days** ({r['probability_before_fix']:.0%} before the fix): "
                + cp.esc(r["text"]).replace("\n", "  \n"),
                help=tip("reason_probability"),
            )
        st.caption(body["reasons"][0]["explains"])
    st.caption(S.raw["handoff"]["what_if"]["line"])
    pd_ = body["paid_days"]
    cp.number(pd_["name"], f"{pd_['value']:.1f}", tip("paid_days"))
    st.caption("; ".join(x for x in pd_["labels"] if x != pd_["name"]))  # the name is the label
    cp.action_block(body["action"])
    st.caption(f"{body['archetype']['status']}.")


def render(client: Client, health: dict[str, Any], capture: bool) -> None:
    msno = st.text_input("Customer ID").strip()
    if not msno:
        st.header("Enter a customer ID from the At-risk list")
        return
    try:
        body = client.score(msno)
    except ApiError as e:
        st.warning(f"{e.status} {e.detail}")
        return
    st.markdown(f'<span class="re-id">{cp.esc(body["msno"])}</span>', unsafe_allow_html=True)
    if body["status"] == "scored":
        scored(body)
    elif body["status"] == "outside_population":
        st.header(f"Outside the population: {body['code']}")
        days = body["next_expiry_in_days"]
        if days is not None:
            st.caption(f"Next expiry in {days} days.")
        st.caption(S.disclaimers[1])
    else:
        st.header("Not active at the store's date")
        st.caption(S.serving["step5"]["known_customers"]["not_active"])
