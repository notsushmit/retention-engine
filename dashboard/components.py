"""Shared page parts: the frame of every page, numbers with tooltips, the 59-day block,
the action block and the reason bars (configs/dashboard.yaml, C1, C2, design_tokens).

Every colour is a registered token; the accent is a fill only (C2), and every bar
carries its value as a direct label in ink. Any text that comes from the API is
escaped before it is placed in HTML.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.client import ApiError, Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()
T = S.tokens
STYLE = Path(__file__).parent / "static" / "style.css"
DEMO = "demonstration"


def esc(x: Any) -> str:
    return html.escape(str(x))


def css() -> None:
    st.markdown(f"<style>{STYLE.read_text()}</style>", unsafe_allow_html=True)


def line(text: str, cls: str = "re-label", help: str | None = None) -> None:
    st.markdown(f'<p class="{cls}">{esc(text)}</p>', unsafe_allow_html=True, help=help)


# --- numbers --------------------------------------------------------------------------------------


def count(x: float) -> str:
    return f"{x:,.0f}"


def share(x: float) -> str:
    return f"{x:.0%}"


def money(x: float | None) -> str:
    return "-" if x is None else f"{x:,.2f} {S.money_unit}"


def fixed(x: float | None, decimals: int) -> str:
    """A table value: thousands separators and a fixed number of decimals."""
    return "-" if x is None else f"{x:,.{decimals}f}"


def as_text(df: pd.DataFrame, decimals: dict[str, int]) -> pd.DataFrame:
    """A copy for display with each named numeric column formatted by `fixed`."""
    out = df.copy()
    for col, d in decimals.items():
        if col in out:
            out[col] = [fixed(None if pd.isna(v) else float(v), d) for v in out[col]]
    return out


def spread(values: list[float], gap: float) -> list[float]:
    """Label positions for line ends: each label at its value unless that is closer than
    `gap` to the label above it; then it moves down to keep the gap. Order is kept."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    above = None
    for i in order:
        pos = values[i] if above is None else min(values[i], above - gap)
        out[i] = pos
        above = pos
    return out


def big(value: str, caption: str, help: str) -> None:
    """The one big number per page, in the italic serif."""
    st.markdown(
        f'<div class="re-big">{esc(value)}</div>'
        f'<p class="re-label re-big-label">{esc(caption)}</p>',
        unsafe_allow_html=True,
        help=help,
    )


def number(label: str, value: str, help: str) -> None:
    st.metric(label, value, help=help)


def demo59(label: str, value: str, help: str) -> None:
    """A 59-day figure: muted, dashed outline, the demonstration badge; never st.metric."""
    st.markdown(
        f'<div class="re-demo59"><span class="re-label">{esc(label)}</span>'
        f'<span class="re-badge">{DEMO}</span><br>{esc(value)}</div>',
        unsafe_allow_html=True,
        help=help,
    )


# --- the action block -----------------------------------------------------------------------------


def action_block(a: dict[str, Any]) -> None:
    """The API's action block with every registered label (handoff item 5)."""
    note = a["instability_note"]
    prefix = S.serving["dashboard_endpoints"]["plans"]["labels"]["effect_prefix"]
    sc = a["scenario"]
    reading = a["reads_as"] or ("pays" if a["pays"] else "")
    st.markdown(
        f'<span class="re-tag" title="{esc(note)}">assumed effect</span> '
        f"{esc(prefix)}: highest net value action <b>{esc(a['highest_net_value_action'])}</b>"
        f" ({esc(reading)})",
        unsafe_allow_html=True,
        help=note,
    )
    number(
        "Net value",
        money(a["net_value"]),
        help=tip("action_values"),
    )
    st.caption(
        f"{prefix}: expected retained value {money(a['expected_retained_value'])}; "
        f"cost {money(a['cost'])}.",
        help=tip("action_values"),
    )
    st.caption(f"effect_basis: {a['effect_basis']}. {note} {a['no_estimate_line']}")
    st.caption(
        "Scenario: "
        + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in sc.items())
        + f". Action block: {a['probability_basis']}."
    )
    if a.get("price"):
        st.caption(a["price"])


# --- the reason bars (C1, C2) ---------------------------------------------------------------------


def all_cells(reasons: dict[str, Any]) -> list[dict[str, Any]]:
    seg = next(x for x in reasons["segments"] if x["segment"] == "all")
    return [] if seg["suppressed"] else list(seg["cells"])


def leading(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    shown = [c for c in cells if not c["suppressed"]]
    return min(shown, key=lambda c: (-c["customers"], c["group"])) if shown else None


def sentence(reasons: dict[str, Any]) -> str | None:
    """Generated from the body (C1, C2); never typed."""
    cells, h, pop = all_cells(reasons), reasons["horizon"], reasons["population"]
    lead = leading(cells)
    if lead is None:
        return None
    t = S.c1["overview_bars"]["sentence"]
    if len(cells) == 1 and lead["customers"] == pop:
        out = t["one_group"]
    else:
        out = S.c2["several_groups_sentence"]
    return (
        out.replace("<horizon>", str(h))
        .replace("<population>", f"{pop:,}")
        .replace("<group>", lead["group"])
        .replace("<customers>", f"{lead['customers']:,}")
    )


def bars_frame(reasons: dict[str, Any]) -> pd.DataFrame:
    cells = all_cells(reasons)
    lead = leading(cells)
    rows = []
    for c in cells:
        rows.append(
            {
                "horizon": f"{reasons['horizon']} days",
                "group": c["group"],
                "share": c["share"] if not c["suppressed"] else 0.0,
                "label": share(c["share"]) if not c["suppressed"] else f"fewer than {S.min_cell}",
                "fill": T["accent"] if lead is not None and c is lead else T["bar_grey"],
            }
        )
    return pd.DataFrame(rows)


def panel(reasons: dict[str, Any]) -> alt.LayerChart:
    df = bars_frame(reasons)
    base = alt.Chart(df).encode(
        y=alt.Y("group:N", sort=None, title=None),
        x=alt.X("share:Q", axis=None, scale=alt.Scale(domain=[0, 1.25])),
    )
    bars = base.mark_bar().encode(color=alt.Color("fill:N", scale=None, legend=None))
    labels = base.mark_text(align="left", dx=4, color=T["ink"], font="JetBrains Mono").encode(
        text="label:N"
    )
    return (bars + labels).properties(
        title=alt.TitleParams(f"{reasons['horizon']} DAYS", color=T["label_grey"], anchor="start"),
        width=260,
        height=alt.Step(26),
    )


def reason_chart(summary: dict[str, Any]) -> alt.HConcatChart:
    """One chart with two panels, 7 and 30 days (C2)."""
    chart = alt.hconcat(*[panel(r) for r in summary["reasons"]])
    return (
        chart.configure(background=T["background"], font="Space Grotesk")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=T["body_grey"],
            titleColor=T["body_grey"],
            domainColor=T["label_grey"],
            tickColor=T["label_grey"],
            gridColor=T["background"],
            labelFont="Space Grotesk",
        )
    )


# --- the frame of every page ----------------------------------------------------------------------


def replay_line(h: dict[str, Any]) -> str:
    p = h.get("replay_progress")
    rp = S.raw["every_page"]["replay_progress"]
    if not p or (p["produced"] == 0 and p["consumed"] == 0):
        return rp["none"]
    return (
        rp["running"]
        .replace("<consumed>", f"{p['consumed']:,}")
        .replace("<expected_events>", f"{p['expected_events']:,}")
    )


def frame(
    client: Client, key: str, render: Callable[[Client, dict[str, Any], bool], None], capture: bool
) -> None:
    css()
    page = next(p for p in S.pages if p.key == key)
    line(S.raw["app_name"]["text"], "re-app")
    line(S.banner, "re-banner")
    line(
        S.raw["every_page"]["breadcrumb"]
        .replace("<group>", page.group)
        .replace("<page>", page.title),
        "re-breadcrumb",
    )
    try:
        h = client.health()
    except (ApiError, OSError) as e:
        h = {"ready": False, "reason": str(e), "store_date": None, "replay_progress": None}
    store_date = h.get("store_date") or "none loaded"
    line(
        S.raw["every_page"]["store_date_line"].replace("<store_date>", store_date),
        "re-store-date",
    )
    line(replay_line(h), "re-label", help=tip("replay"))
    if not h["ready"]:
        st.warning(f"Not ready: {h['reason']}")
    else:
        try:
            render(client, h, capture)
        except ApiError as e:
            st.warning(f"{e.status} {e.detail}" + (f": {e.reason}" if e.reason else ""))
    st.markdown(
        '<div class="re-footer">' + "".join(f"<p>{esc(d)}</p>" for d in S.disclaimers) + "</div>",
        unsafe_allow_html=True,
    )
