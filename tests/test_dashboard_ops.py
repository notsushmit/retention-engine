"""Model health: the drift panel and the last retraining comparison (configs/dashboard.yaml, C8)."""

from __future__ import annotations

import re

import dashboard_fake as f
from dashboard.settings import load
from retention.api.labels import DASH, OPS
from test_dashboard_pages import render, visible
from test_dashboard_wording import check

S = load()


def api(monitored=True, retrain=None, drift=None) -> f.FakeApi:
    return f.FakeApi(
        {
            "/model": (200, f.model(monitored=monitored, last_retrain=retrain)),
            "/drift": (200, drift if drift is not None else f.drift()),
        }
    )


def test_the_call_to_drift_is_registered_and_the_client_allows_it():
    assert ("GET", "/drift") in S.calls
    client = f.client(api())
    assert client.drift()["status"] == "monitored"


def test_without_drift_the_page_shows_the_registered_line_and_no_drift_number():
    fake = f.FakeApi()
    text = visible(render("model_health", fake))
    assert DASH.drift in text
    assert "Drifted columns" not in text and "Dataset alert" not in text
    assert "/drift" not in [r.url.path for r in fake.requests]  # not even asked for


def test_the_drift_panel_shows_the_shares_the_alert_and_the_label():
    at = render("model_health", api())
    text = visible(at)
    assert "14 of 47 (30%)" in text and "Dataset alert" in text and "no" in text
    assert "0.26 (required 0.20)" in text
    assert OPS.drift_label in text and OPS.synthetic_sentence in text
    assert OPS.drift_see in text
    assert "Days outside the event-count band" in text
    names = [df.value for df in at.dataframe]
    assert any("col_0" in df.to_csv() for df in names)


def test_a_monitored_non_synthetic_file_carries_only_the_registered_label():
    text = visible(render("model_health", api(drift=f.drift(synthetic=False))))
    assert OPS.drift_label in text and OPS.synthetic_sentence not in text


def test_the_panel_shows_the_sentence_when_the_drift_answer_is_not_monitored():
    text = visible(render("model_health", api(drift=f.not_monitored())))
    assert "Drifted columns" not in text and DASH.drift in text


def test_the_last_retraining_comparison_is_shown_only_when_there_is_one():
    assert "Last retraining comparison" not in visible(render("model_health", api()))
    text = visible(render("model_health", api(retrain=f.retrain())))
    assert "Last retraining comparison" in text and OPS.decisions[1] in text
    assert "0.8123" in text and "0.8101" in text and "-0.0011 to 0.0055" in text
    assert OPS.retrain_label in text


def test_every_number_on_the_new_panels_has_a_tooltip():
    at = render("model_health", api(retrain=f.retrain()))
    metrics = [m for m in at.metric if m.label in {
        "Drifted columns", "Dataset alert", "Share over the unshifted store",
        "Margin and alert rules met", "Days outside the event-count band", "Decision",
        "30-day AUC, candidate", "30-day AUC, current model", "Difference and its interval",
    }]  # fmt: skip
    assert len(metrics) == 9
    assert all(m.help for m in metrics)
    for m in metrics:
        assert re.search(r"/(drift|model)\b", m.help), m.label


def test_the_new_panels_pass_the_wording_rules_and_name_no_never_used_attribute():
    text = visible(render("model_health", api(retrain=f.retrain())))
    assert check(text) == []
