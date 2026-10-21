"""GET /alerts: the knowable rule, hidden dimensions (api.alerts and A7), labels, events, pins.

The hand-made alerts file of test_service has four alerts; at 2016-11-01 two are
knowable, one of them on the boundary day. The real pinned file is not in git: the
tests that read it skip in CI, the registered knowable count (7 of 10 at
2016-11-01) among them.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from retention.api import alerts as al
from retention.api import schemas as sch
from retention.api import service as svc
from retention.api import store as st
from retention.api.config import file_sha256, load_serving
from retention.api.labels import LABELS
from retention.api.schemas import never_used_hits, response_hits
from test_app import error_of, flip_last_byte, make_client, score  # noqa: F401 (fixture)
from test_service import (
    HIDDEN,
    NOV,
    alerts_run,
    bundle_labels,  # noqa: F401 (fixture)
    load,
    pins,  # noqa: F401 (fixture)
    registered,  # noqa: F401 (fixture)
    snapshot,
    trend_alert,
    write_pins,
)

SERVING = load_serving()
REAL_PIN = SERVING["step5"]["pins"]["alerts"]
REAL = Path(REAL_PIN["path"])
needs_real = pytest.mark.skipif(not REAL.exists(), reason="needs the pinned alerts file")


# --- the knowable rule --------------------------------------------------------------------------


def test_the_registered_rule_reads_37_days():
    assert LABELS.knowable_days == 37


# Written out by hand for the store date 2016-11-01, not derived from the code under test:
# week start, its knowable-from date (week start + 37 days), knowable at 2016-11-01.
BOUNDARY = [
    ("d38", "2016-09-24", date(2016, 10, 31), True),  # 38 days before
    ("d37", "2016-09-25", date(2016, 11, 1), True),  # 37 days before: the boundary day
    ("d36", "2016-09-26", date(2016, 11, 2), False),  # 36 days before: one day late
]


@pytest.mark.parametrize("alert_id, week, knowable_from, knowable", BOUNDARY)
def test_made_up_alerts_at_38_37_and_36_days_before_the_store_date(
    alert_id, week, knowable_from, knowable
):
    served = al.served_alert(trend_alert(alert_id, week))
    assert served.knowable_from == knowable_from
    assert al.is_knowable(date.fromisoformat(week), date(2016, 11, 1)) is knowable


def test_of_the_three_made_up_alerts_only_38_and_37_days_before_are_knowable():
    run = {"alerts": [trend_alert(i, w) for i, w, _, _ in BOUNDARY]}
    assert al.knowable_count(run, date(2016, 11, 1)) == 2
    shown = [
        a["alert_id"]
        for a in run["alerts"]
        if al.is_knowable(date.fromisoformat(a["week"]), date(2016, 11, 1))
    ]
    assert shown == ["d38", "d37"]


def test_the_hand_made_file_has_two_knowable_alerts_at_2016_11_01():
    assert al.knowable_count(alerts_run(), NOV) == 2


# --- hiding (api.alerts and A7) -----------------------------------------------------------------


def test_hidden_dimensions_are_removed_from_every_segment_list_and_counted():
    stored = trend_alert("x", "2016-03-07")
    served = al.served_alert(stored).model_dump(mode="json")
    assert [s["dimension"] for s in served["top_segments"]] == ["auto_renew"]
    assert [s["dimension"] for s in served["too_small_segments"]] == ["payment_method"]
    assert sorted(served["dimension_totals"]) == ["auto_renew", "plan_length"]
    assert served["hidden_by_rule"] == {
        "top_segments": 2,
        "too_small_segments": 2,
        "segments": 4,
        "dimension_totals": 3,
    }
    assert served["labels"] == list(LABELS.trend_labels)
    assert served["data_quality_flags"] == stored["data_quality_flags"]
    for k in ("sentence", "flags", "windows", "noise_assumption", "band", "series", "note"):
        assert served[k] == stored[k], k
    text = json.dumps(served)
    assert never_used_hits(text) == []
    assert not any(d in text for d in HIDDEN)


def test_a_served_alert_that_still_holds_a_hidden_dimension_cannot_be_built():
    served = al.served_alert(trend_alert("x", "2016-03-07")).model_dump()
    served["too_small_segments"].append({"dimension": "tenure", "value": "730+"})
    with pytest.raises(ValidationError, match="hidden dimension"):
        sch.TrendAlert(**served)


def test_a_trend_alert_needs_exactly_the_three_labels():
    served = al.served_alert(trend_alert("x", "2016-03-07")).model_dump()
    with pytest.raises(ValidationError, match="three registered labels"):
        sch.TrendAlert(**{**served, "labels": LABELS.trend_labels[:2]})


def test_a_stored_alert_with_an_unknown_field_is_refused():
    run = alerts_run()
    run["alerts"][0]["extra"] = 1
    with pytest.raises(ValidationError):
        al.check_run(run)


# --- the scan reaches sentences and free text -----------------------------------------------------


@pytest.mark.parametrize(
    "place",
    ["sentence", "note", "series", "windows", "flags", "noise_assumption", "segment value"],
)
def test_a_never_used_word_in_sentence_or_free_text_is_caught_in_the_served_body(place):
    stored = trend_alert("x", "2016-03-07")
    planted = "the city of the customer"
    if place in ("windows", "flags", "noise_assumption"):
        stored[place]["text"] = planted
    elif place == "segment value":
        stored["top_segments"][1]["value"] = planted
    else:
        stored[place] = planted
    served = json.dumps(al.served_alert(stored).model_dump(mode="json"))
    assert response_hits(served, 200, None) == ["city"]


# --- over HTTP ------------------------------------------------------------------------------------


@pytest.mark.valkey
def test_alerts_serves_the_knowable_trend_alerts_and_no_events_before_replay(make_client, valkey):  # noqa: F811
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx))
    r = c.get("/alerts")
    assert r.status_code == 200
    body = r.json()
    assert [a["alert_id"] for a in body["trend"]] == ["cp1", "cp2"]
    assert (body["trend_in_file"], body["trend_knowable"], body["event"]) == (4, 2, [])
    assert body["store_date"] == "2016-11-01"
    for a in body["trend"]:
        assert a["labels"] == list(LABELS.trend_labels) and a["hidden_by_rule"]["segments"] == 4
    assert response_hits(r.text, 200, None) == []
    assert not any(d in r.text for d in HIDDEN)


@pytest.mark.valkey
def test_cancel_events_carry_their_label(make_client, valkey):  # noqa: F811
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx))
    valkey.rpush(st.KEYS.events, json.dumps({"msno": "c003", "date": "2016-11-03"}))
    events = c.get("/alerts").json()["event"]
    assert events == [
        {"kind": "event", "msno": "c003", "date": "2016-11-03", "label": LABELS.event_label}
    ]


@pytest.mark.valkey
def test_alerts_takes_no_parameters_and_is_503_when_not_ready(make_client, valkey):  # noqa: F811
    c = make_client()
    r = c.get("/alerts")
    assert r.status_code == 503
    assert r.json() == error_of(503, "no snapshot is loaded (the store has no date)")
    load(valkey, snapshot(c.app.state.ctx))
    for q in ("?kind=trend", "?city=1", "?"):
        r = c.get("/alerts" + q)
        if q == "?":
            assert r.status_code == 200
        else:
            assert r.status_code == 422 and r.json() == error_of(422)


# --- the pins ------------------------------------------------------------------------------------


@pytest.mark.valkey
@pytest.mark.parametrize("name, file", [("alerts", "alerts_run.json"), ("plans", "plans.parquet")])
def test_startup_is_refused_on_a_changed_pinned_file_naming_it_and_both_hashes(
    make_client,  # noqa: F811
    tmp_path,
    name,
    file,  # noqa: F811
):
    copy = write_pins(tmp_path / "pins")
    expected = copy[name][1]
    flip_last_byte(copy[name][0])
    found = file_sha256(copy[name][0])
    with pytest.raises(svc.PinRefused) as e:
        make_client(pinned=copy)
    assert file in str(e.value) and expected in str(e.value) and found in str(e.value)


@pytest.mark.valkey
@pytest.mark.parametrize("name, file", [("alerts", "alerts_run.json"), ("plans", "plans.parquet")])
def test_a_pinned_file_changed_after_startup_makes_every_endpoint_503(
    make_client,  # noqa: F811
    valkey,
    tmp_path,
    name,
    file,  # noqa: F811
):
    copy = write_pins(tmp_path / "pins")
    c = make_client(pinned=copy)
    load(valkey, snapshot(c.app.state.ctx))
    assert c.get("/health").status_code == 200
    flip_last_byte(copy[name][0])  # a temporary copy, never the real file
    h = c.get("/health")
    assert h.status_code == 503 and h.json()["ready"] is False
    reason = h.json()["reason"]
    assert file in reason and copy[name][1] in reason
    for r in (score(c, msno="c000"), c.get("/at-risk"), c.get("/alerts")):
        assert r.status_code == 503 and r.json() == error_of(503, reason)


# --- the real pinned file (skips in CI: the file is not in git) -----------------------------------


@needs_real
def test_the_real_alerts_file_has_its_pinned_hash():
    assert file_sha256(REAL) == REAL_PIN["sha256"]


@needs_real
def test_the_real_knowable_count_at_2016_11_01_is_amendment_a8_s():
    """A8: 6 under the unchanged rule; the registered 7 stays in the config, labelled wrong."""
    a8 = SERVING["alerts_knowable_count"]
    assert a8["rule"] == SERVING["api"]["alerts"]["trend"]["knowable_rule"]
    assert SERVING["step5"]["alerts"]["trend_knowable_at_2016_11_01"] == a8["registered_value"]
    run = json.loads(REAL.read_text())
    assert len(run["alerts"]) == 10
    assert al.knowable_count(run, date(2016, 11, 1)) == a8["knowable_at_2016_11_01"] == 6
    late = next(a for a in run["alerts"] if a["alert_id"] == a8["first_not_knowable"]["alert_id"])
    assert late["week"] == a8["first_not_knowable"]["week"] == "2016-09-26"
    assert al.served_alert(late).knowable_from == date(2016, 11, 2)


@needs_real
def test_the_real_alerts_served_at_2016_11_01_hide_every_hidden_dimension_and_pass_the_scan():
    run = json.loads(REAL.read_text())
    al.check_run(run)
    for a in run["alerts"]:
        text = json.dumps(al.served_alert(a).model_dump(mode="json"))
        assert response_hits(text, 200, None) == [], a["alert_id"]
        assert '"registration_channel"' not in text
