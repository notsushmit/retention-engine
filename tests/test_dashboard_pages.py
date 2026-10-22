"""Every page through Streamlit's AppTest, against the fake API (no Docker, no store).

Checks the frame on every page (banner, breadcrumb, store date, the four disclaimers), the
not-ready and no-replay states, capture mode, the 59-day style, the action labels, tooltips
on every metric, the registered filters and no banned control.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard_fake as f
from dashboard.settings import load
from dashboard.views import at_risk_list

S = load()
PAGES = [p.key for p in S.pages]


def script():
    import streamlit as st

    from dashboard.entry import run

    run(st.session_state["page_key"])


def render(key: str, api: f.FakeApi | None = None, capture: bool = False, **state) -> AppTest:
    at = AppTest.from_function(script, default_timeout=30)
    at.session_state["client"] = f.client(api)
    at.session_state["page_key"] = key
    for k, v in state.items():
        at.session_state[k] = v
    if capture:
        at.query_params["capture"] = "1"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def protos(node, out):
    p = getattr(node, "proto", None)
    if p is not None:
        out.append(str(p))
    for c in getattr(node, "children", {}).values():
        protos(c, out)
    return out


def all_text(at: AppTest) -> str:
    """Every rendered element's content, the sidebar, tabs and table cells included."""
    parts = protos(at._tree, [])
    parts += [df.value.to_csv() for df in at.dataframe]
    return "\n".join(parts)


def md(at: AppTest) -> list:
    """Markdown elements, without the injected style block."""
    return [m for m in at.markdown if not m.value.startswith("<style>")]


def visible(at: AppTest) -> str:
    parts = [m.value for m in md(at)]
    parts += [c.value for c in at.caption]
    parts += (
        [h.value for h in at.header] + [w.value for w in at.warning] + [i.value for i in at.info]
    )
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


# --- the frame of every page -------------------------------------------------------------------


def test_five_pages_in_three_groups():
    assert PAGES == [
        "overview",
        "at_risk_list",
        "customer_profile",
        "action_planner",
        "model_health",
    ]
    assert [p.group for p in S.pages] == ["Who to act on"] * 3 + ["What to do", "Can I trust it"]


@pytest.mark.parametrize("key", PAGES)
def test_every_page_carries_the_frame(key):
    text = visible(render(key))
    assert "Replay of 2016 data, not real-time." in text
    page = next(p for p in S.pages if p.key == key)
    assert f"{page.group} / {page.title}" in text
    assert "Store date 2016-11-01 (one date per loaded snapshot; no daily dates)" in text
    for d in S.disclaimers:
        assert d in text


@pytest.mark.parametrize("key", PAGES)
def test_a_service_that_is_not_ready_shows_the_reason_and_no_number(key):
    at = render(key, f.FakeApi(f.not_ready()))
    assert "no snapshot is loaded (the store has no date)" in visible(at)
    assert list(at.metric) == [] and list(at.dataframe) == []
    assert "re-big" not in visible(at)


def test_no_replay_running_never_reads_0_of_0():
    text = visible(render("overview"))
    assert "no replay running" in text and "0 of 0" not in text


def test_a_running_replay_reads_its_progress():
    api = f.FakeApi({"/health": (200, f.health(produced=10, consumed=7))})
    assert "7 of 3,727,472 events replayed" in visible(render("overview", api))


@pytest.mark.parametrize("key", PAGES)
def test_every_metric_has_a_tooltip(key):
    state = {"page_key": key}
    at = render(key)
    if key == "customer_profile":
        at.text_input[0].input(f.SCORED_ID).run()
    assert all(m.proto.help for m in at.metric), state


# --- the overview ------------------------------------------------------------------------------


def test_overview_order_big_number_buckets_chart_then_two_tabs():
    at = render("overview")
    text = visible(at)
    assert (
        "About 3,160 of 169,673 scored customers are expected to lapse in the next 7 days" in text
    )
    assert [m.label for m in at.metric] == [
        "Days 1 to 7",
        "Days 8 to 30",
        "Expiring more than 30 days ahead",
    ]
    assert [t.label for t in at.tabs] == ["Alerts", "Top 20 at risk"]
    assert "At 7 days all 1,000 share renewal_timing as their first group" in text
    assert "At 30 days renewal_timing leads with 823 of the top 1,000 at risk" in text
    assert "Reasons are model attributions, not causes." in text
    assert "Share of at-risk customers whose top raising reason is" in text
    for label in S.trend_labels:
        assert label in text


def test_the_59_day_bucket_is_a_badged_demonstration_never_a_metric():
    at = render("overview")
    assert not any("59" in m.label for m in at.metric)
    demo = [m.value for m in md(at) if "re-demo59" in m.value]
    assert len(demo) == 1 and "demonstration" in demo[0] and "3,408" in demo[0]


def test_capture_mode_shows_no_customer_id_anywhere_on_the_overview():
    normal = all_text(render("overview"))
    assert f.IDS[0] in normal and f.CANCEL_ID in normal  # the test can fail
    at = render("overview", capture=True)
    text = all_text(at)
    leaked = [m for m in [*f.IDS, f.CANCEL_ID] if m in text]
    assert leaked == []
    assert [t.label for t in at.tabs] == ["Alerts"]


def test_capture_mode_changes_nothing_else_on_the_overview():
    a, b = render("overview"), render("overview", capture=True)
    assert [m.value for m in a.metric] == [m.value for m in b.metric]
    assert [h.value for h in a.header] == [h.value for h in b.header]


def test_a_summary_timeout_leaves_the_computing_state():
    class Slow(f.FakeApi):
        def __call__(self, request):
            if request.url.path == "/summary":
                raise httpx.ReadTimeout("slow", request=request)
            return super().__call__(request)

    at = render("overview", Slow())
    assert [i.value for i in at.info] == [
        "Computing the summary: the first call after a store-date move takes several seconds."
    ]


def test_december_withholds_the_reason_bars():
    body = f.summary()
    body |= {
        "store_date": "2016-12-01",
        "reasons": None,
        "reasons_withheld": "2016-12 not explained",
        "data_warning": "data warning about the 2016-12-05 missing-renewal day",
    }
    text = visible(render("overview", f.FakeApi({"/summary": (200, body)})))
    assert "2016-12 not explained" in text and "first group" not in text


# --- the at-risk list --------------------------------------------------------------------------


def test_at_risk_list_reads_the_api_order_and_the_registered_filters():
    at = render("at_risk_list")
    labels = [s.label for s in at.sidebar.selectbox]
    assert labels == ["Horizon", "Auto-renew", "In the plan", "Rows per page"]
    assert at.sidebar.selectbox[0].options == ["7 days", "30 days", "59 days (demonstration)"]
    assert at.sidebar.selectbox[3].options == ["50", "100", "500", "1000"]
    df = at.dataframe[0].value
    assert df.customer.tolist() == f.IDS[:25] and df["rank"].tolist() == [
        str(i) for i in range(1, 26)
    ]


def test_at_59_days_the_list_carries_the_demonstration_badge():
    at = render("at_risk_list")
    at.sidebar.selectbox[0].select(59).run()
    assert any("re-demo59" in m.value and "demonstration" in m.value for m in md(at))
    assert set(at.dataframe[0].value.grade) == {"demonstration"}


def test_in_plan_is_disabled_on_a_date_without_a_plan():
    page = f.at_risk()
    page |= {"plan_status": "no plan for this date", "rows": []}
    at = render("at_risk_list", f.FakeApi({"/at-risk": (200, page)}))
    assert at.sidebar.selectbox[2].disabled


def test_the_csv_holds_the_displayed_rows_with_store_date_and_horizon():
    page = f.at_risk(horizon=30, limit=5)
    df = pd.read_csv(io.BytesIO(at_risk_list.csv_bytes(page)))
    assert df.columns[:2].tolist() == ["store_date", "horizon"]
    assert df.customer.tolist() == f.IDS[:5]
    assert set(df.store_date) == {"2016-11-01"} and set(df.horizon) == {30}
    assert set(df["plan label"]) == {S.serving["api"]["at_risk"]["plan_label"]}


# --- the customer profile ----------------------------------------------------------------------


def profile(msno: str) -> AppTest:
    at = render("customer_profile")
    at.text_input[0].input(msno).run()
    assert not at.exception
    return at


def test_a_scored_customer_shows_three_points_two_reasons_and_the_labelled_action():
    at = profile(f.SCORED_ID)
    text = visible(at)
    assert [m.label for m in at.metric][:2] == ["7 days", "30 days"]
    assert not any("59" in m.label for m in at.metric)
    assert any("re-demo59" in m.value and "61%" in m.value for m in md(at))
    assert "Reasons explain the probability before the ordering fix." in text
    for needed in (
        "assumed effect",
        "effect_basis: assumed",
        "The plan is unstable under 5 of 11 variants.",
        "No KKBox effect estimate exists; every effect is an assumption.",
        "Under the assumed relative base effect",
        "Counterfactuals cover 2.7% of explained rows and are model scores, not effects.",
    ):
        assert needed in text, needed
    tag = next(m for m in md(at) if "re-tag" in m.value)
    assert 'title="The plan is unstable under 5 of 11 variants."' in tag.value


def test_outside_not_active_and_unknown_answers_are_shown_as_served():
    assert "expiry_beyond_30_days" in visible(profile(f.OUT_ID))
    assert "Not active at the store's date" in visible(profile(f.INACTIVE_ID))
    assert "404 unknown customer" in visible(profile("nobody-here"))


# --- the action planner and model health ---------------------------------------------------------


def test_the_planner_shows_precomputed_plans_with_every_label():
    text = visible(render("action_planner"))
    for needed in (
        S.serving["dashboard_endpoints"]["plans"]["labels"]["plan"],
        S.caption,
        "spread of random targeting, not a confidence interval",
        "The plan is unstable under 5 of 11 variants.",
        "No KKBox effect estimate exists; every effect is an assumption.",
        S.disclaimers[2],
        "Under the assumed relative base effect, the 5,000 budget plan contacts 4,887 customers",
        "5,312.56 price units",
    ):
        assert needed in text, needed


def test_the_planner_on_a_date_without_a_plan():
    api = f.FakeApi({"/plans": (200, f.plans(f.date(2016, 12, 1)))})
    text = visible(render("action_planner", api))
    assert "no plan for this date" in text and "re-big" not in text


def test_model_health_shows_files_calibration_and_drift():
    at = render("model_health")
    text = visible(at)
    assert "Model version 1; every file matches its registered hash" in text
    assert "1.0005" in text  # the 30-day slope after the fix, as the big number
    assert "drift: not monitored (Phase 10)" in text
    assert "no 59-day row" in text
    assert len(at.dataframe[1].value) == 8


# --- controls ----------------------------------------------------------------------------------


BANNED = re.compile(
    r"\b("
    + "|".join(
        map(re.escape, [*S.serving["wording"]["never_used"], *S.raw["banned_controls"]["words"]])
    )
    + r")\b",
    re.I,
)


@pytest.mark.parametrize("key", PAGES)
def test_no_banned_control_on_any_page(key):
    at = render(key)
    widgets = [
        *at.selectbox,
        *at.sidebar.selectbox,
        *at.text_input,
        *at.number_input,
        *at.sidebar.number_input,
    ]
    widgets += [*at.date_input, *at.slider, *at.radio, *at.multiselect, *at.toggle, *at.checkbox]
    texts = []
    for w in widgets:
        texts.append(str(w.label))
        texts += [str(o) for o in getattr(w, "options", [])]
    assert [t for t in texts if BANNED.search(t.replace("_", " "))] == []
    assert list(at.date_input) == []  # no store date picker


def test_the_app_entry_point_opens_on_the_overview():
    at = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=30)
    at.session_state["client"] = f.client()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Who to act on / Overview" in visible(at)


# --- found after a run (D-258) -------------------------------------------------------------------


@pytest.mark.parametrize("key", PAGES)
def test_every_page_opens_with_the_application_name(key):
    at = render(key)
    assert md(at)[0].value == '<p class="re-app">Retention Decision Engine</p>'


@pytest.mark.parametrize("key", PAGES)
def test_no_table_shows_a_raw_float(key):
    at = render(key)
    for df in at.dataframe:
        floats = [c for c in df.value.columns if pd.api.types.is_float_dtype(df.value[c])]
        assert floats == [], (key, floats)


NUM = r"-?\d{1,3}(,\d{3})*"


def test_the_planner_table_has_separators_and_fixed_decimals():
    df = render("action_planner").dataframe[0].value
    pattern = {"expected_retained_customers": 1, "expected_retained_value": 2, "net_value": 2}
    for col, d in pattern.items():
        assert df[col].str.fullmatch(NUM + rf"\.\d{{{d}}}").all(), col
    assert "5,312.56" in df.net_value.tolist()


def test_the_model_table_has_fixed_decimals():
    df = render("model_health").dataframe[0].value
    assert df.events.tolist() == ["937", "3,388"]
    assert df.slope_after_fix.tolist() == ["0.9807", "1.0005"]


def test_the_overview_buckets_sit_in_one_row_and_the_30_day_ahead_tile_below():
    at = render("overview")
    assert len(at.columns) == 5  # three bucket columns, then the tile and its note
    assert at.metric[-1].label == "Expiring more than 30 days ahead"
