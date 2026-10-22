"""Action planner: GET /plans, Phase 7's precomputed plans for the store's date; nothing is
re-planned and nothing calls an optimizer."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()
T = S.tokens
HEADLINE = ("optimizer", "5,000")
# One literal sentence, so the fixed text starts with the registered effect prefix
# (configs/serving.yaml dashboard_endpoints.plans.labels.effect_prefix; tested).
HEADLINE_CAPTION = (
    "Under the assumed relative base effect, the optimizer's net value at the 5,000 budget"
)
DECIMALS = {
    "expected_retained_customers": 1,
    "expected_retained_value": 2,
    "expected_cost": 2,
    "net_value": 2,
    "contacted": 1,
}
COLUMNS = [
    "expected_retained_customers",
    "expected_retained_value",
    "expected_cost",
    "net_value",
    "contacted",
]


def budget_order(rows: list[dict[str, Any]]) -> list[str]:
    def key(label: str) -> float:
        return float("inf") if label == "no limit" else float(label.replace(",", ""))

    return sorted({r["budget_label"] for r in rows}, key=key)


LABEL_GAP = 0.07  # share of the net-value range between two end labels


def label_positions(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """The line ends with label_y: each label by its value, spread to a minimum gap."""
    ends = df[df.budget == order[-1]].copy()
    span = float(df.net.max() - df.net.min()) or 1.0
    ends["label_y"] = cp.spread(ends.net.astype(float).tolist(), LABEL_GAP * span)
    return ends


def chart(rows: list[dict[str, Any]]) -> alt.LayerChart:
    """Net value by budget level, one grey line per strategy, labelled at its end."""
    order = budget_order(rows)
    df = pd.DataFrame(
        [
            {"strategy": r["strategy"], "budget": r["budget_label"], "net": r["net_value"]}
            for r in rows
        ]
    )
    df["colour"] = [T["ink"] if s == HEADLINE[0] else T["bar_grey"] for s in df.strategy]
    base = alt.Chart(df).encode(
        x=alt.X("budget:O", sort=order, title=f"budget ({S.money_unit})"),
        y=alt.Y("net:Q", title=f"net value ({S.money_unit})"),
    )
    lines = base.mark_line().encode(
        detail="strategy:N", color=alt.Color("colour:N", scale=None, legend=None)
    )
    ends = label_positions(df, order)
    labels = (
        alt.Chart(ends)
        .mark_text(align="left", dx=6, color=T["ink"], font="JetBrains Mono")
        .encode(x=alt.X("budget:O", sort=order), y="label_y:Q", text="strategy:N")
    )
    return (
        (lines + labels)
        .properties(width=560, height=280)
        .configure(background=T["background"], font="Space Grotesk")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=T["body_grey"],
            titleColor=T["body_grey"],
            domainColor=T["label_grey"],
            tickColor=T["label_grey"],
            gridColor=T["background"],
        )
    )


def render(client: Client, health: dict[str, Any], capture: bool) -> None:
    p = client.plans()
    if not p["rows"]:
        st.header(p["plan_status"])
        st.caption(p["dates"])
        return
    head = next((r for r in p["rows"] if (r["strategy"], r["budget_label"]) == HEADLINE), None)
    if head is not None:
        st.header(
            S.page("action_planner")["takeaway"].replace(
                "<contacted>", cp.count(head["contacted"])
            ),
            help=tip("plan_values"),
        )
        cp.big(
            cp.money(head["net_value"]),
            HEADLINE_CAPTION,
            tip("plan_values"),
        )
    st.markdown(
        f'<span class="re-tag" title="{cp.esc(p["instability_note"])}">assumed effect</span> '
        f"{cp.esc(p['plan_status'])}",
        unsafe_allow_html=True,
        help=p["instability_note"],
    )
    st.altair_chart(chart(p["rows"]), theme=None)
    table = pd.DataFrame(
        [
            {"strategy": r["strategy"], "budget": r["budget_label"], **{c: r[c] for c in COLUMNS}}
            for r in p["rows"]
        ]
    )
    st.dataframe(
        cp.as_text(table, DECIMALS),
        hide_index=True,
        column_config={c: st.column_config.TextColumn(help=tip("plan_values")) for c in COLUMNS},
    )
    st.caption(f"{p['effect_prefix']}. Comparison caption: {p['caption']}.")
    spread = [r for r in p["rows"] if r["range"]]
    if spread:
        st.caption(
            f"{p['effect_prefix']}, random targeting's net value ranges "
            f"({spread[0]['range']['label']}) "
            + "; ".join(
                f"{r['budget_label']}: {cp.money(r['range']['p2_5']['net_value'])} to "
                f"{cp.money(r['range']['p97_5']['net_value'])}"
                for r in spread
            ),
            help=tip("plan_range"),
        )
    st.caption(
        f"effect_basis: {p['effect_basis']}. {p['instability_note']} {p['no_estimate_line']} "
        f"{p['dates']} Values in {p['money_unit']}."
    )
