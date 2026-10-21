"""GET /at-risk: pages of the stored ranking, filters, plan fields, errors, the index, the scan.

Every expectation is read from the store itself (the stored ranking and documents)
or from the hand-made plan of test_service, never from the code under test.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from retention.api import at_risk as ar
from retention.api import store as st
from retention.api.labels import LABELS
from retention.api.schemas import never_used_hits, response_hits
from retention.explain.reasons import probability_text
from test_app import error_of, make_client, with_id  # noqa: F401 (fixture)
from test_service import (
    N_SCORED,
    NOV_PLAN,
    OCT_PLAN,
    bundle_labels,  # noqa: F401 (fixture)
    load,
    pins,  # noqa: F401 (fixture)
    plan_rows,
    registered,  # noqa: F401 (fixture)
    snapshot,
)

DEC = date(2016, 12, 1)


def get(c, **params):
    return c.get("/at-risk", params=params)


def stored(valkey, h: int) -> list[tuple[str, float]]:
    return st.ranking(valkey, h, 0, N_SCORED)


def auto_renew(valkey, m: str) -> str:
    return "on" if st.customer(valkey, m)["row"].get("last_is_auto_renew") == 1 else "off"


@pytest.fixture()
def loaded(make_client, valkey):  # noqa: F811
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx))
    return c


# --- the query ------------------------------------------------------------------------------------


def test_defaults_are_the_registered_ones():
    assert ar.parse_query([]) == ar.Query(30, 50, 0, None, None)


def test_a_full_query_is_read_exactly():
    q = ar.parse_query(
        [
            ("horizon", "59"),
            ("limit", "1000"),
            ("offset", "7"),
            ("auto_renew", "off"),
            ("in_plan", "true"),
        ]
    )
    assert q == ar.Query(59, 1000, 7, "off", True)


MALFORMED = [
    [("horizon", "60")],
    [("horizon", "0")],
    [("horizon", "30.0")],
    [("horizon", "+30")],
    [("horizon", "")],
    [("limit", "0")],
    [("limit", "1001")],
    [("limit", "abc")],
    [("offset", "-1")],
    [("auto_renew", "yes")],
    [("auto_renew", "true")],
    [("in_plan", "1")],
    [("in_plan", "on")],
    [("horizon", "7"), ("horizon", "30")],
    [("city", "1")],
    [("tenure", "730+")],
    [("segment", "paid")],
]


@pytest.mark.parametrize("items", MALFORMED)
def test_anything_else_is_malformed(items):
    with pytest.raises(ar.Malformed):
        ar.parse_query(items)


def test_the_plan_cell_is_the_registered_one_and_nothing_else():
    cells = ar.plan_cell(pd.DataFrame(plan_rows()))
    assert cells == {date(2016, 11, 1): NOV_PLAN, date(2016, 10, 1): OCT_PLAN}


# --- pages of the stored ranking ------------------------------------------------------------------


@pytest.mark.valkey
@pytest.mark.parametrize("h", [7, 30, 59])
@pytest.mark.parametrize("offset, limit", [(0, 50), (0, 1000), (10, 7), (55, 10), (60, 5)])
def test_an_unfiltered_page_equals_the_stored_ranking(loaded, valkey, h, offset, limit):
    r = get(loaded, horizon=h, offset=offset, limit=limit)
    assert r.status_code == 200
    body = r.json()
    want = stored(valkey, h)[offset : offset + limit]
    assert [(x["msno"], x["probability"]) for x in body["rows"]] == want
    assert [x["rank"] for x in body["rows"]] == list(range(offset + 1, offset + 1 + len(want)))
    assert body["matching"] == N_SCORED
    for x in body["rows"]:
        assert x["text"] == probability_text(x["probability"])
        assert x["grade"] == ("demonstration" if h == 59 else None)
        assert x["auto_renew"] == auto_renew(valkey, x["msno"])


@pytest.mark.valkey
def test_the_default_page(loaded):
    body = loaded.get("/at-risk").json()
    assert (body["horizon"], body["limit"], body["offset"], len(body["rows"])) == (30, 50, 0, 50)


def filtered(valkey, h: int, filters: dict) -> list[tuple[str, float]]:
    """The filter applied to the full stored ranking, read from the store and the plan."""

    def keep(m: str) -> bool:
        if "auto_renew" in filters and auto_renew(valkey, m) != filters["auto_renew"]:
            return False
        return "in_plan" not in filters or (m in NOV_PLAN) == (filters["in_plan"] == "true")

    return [(m, p) for m, p in stored(valkey, h) if keep(m)]


FILTERS = [
    {"auto_renew": "on"},
    {"auto_renew": "off"},
    {"in_plan": "true"},
    {"in_plan": "false"},
    {"auto_renew": "on", "in_plan": "true"},
    {"auto_renew": "off", "in_plan": "false"},
]


@pytest.mark.valkey
@pytest.mark.parametrize("h", [7, 30, 59])
@pytest.mark.parametrize("filters", FILTERS)
def test_a_filtered_page_equals_the_filter_on_the_full_stored_ranking(loaded, valkey, h, filters):
    full = filtered(valkey, h, filters)
    assert full  # every filter matches someone in the hand-made snapshot
    for offset, limit in ((0, 1000), (0, 3), (2, 4), (len(full), 5)):
        body = get(loaded, horizon=h, offset=offset, limit=limit, **filters).json()
        assert [(x["msno"], x["probability"]) for x in body["rows"]] == full[
            offset : offset + limit
        ]
        assert body["matching"] == len(full)
        assert body["filters"] == {
            "auto_renew": filters.get("auto_renew"),
            "in_plan": {"true": True, "false": False}.get(filters.get("in_plan")),
        }


# --- plan fields and the header -------------------------------------------------------------------


@pytest.mark.valkey
def test_every_plan_field_carries_the_label_and_the_header_its_note(loaded, valkey):
    body = get(loaded, limit=1000).json()
    assert body["header_note"] == LABELS.at_risk_header_note
    assert body["plan_status"] == LABELS.plan_label
    for x in body["rows"]:
        assert x["plan"] == {
            "in_plan": x["msno"] in NOV_PLAN,
            "action": NOV_PLAN.get(x["msno"]),
            "label": LABELS.plan_label,
        }
    counts = st.counts(valkey)
    assert body["counts"] == {
        "scored": counts["scored"],
        "outside_population": {
            "segment_trial_only": counts["outside_population_segment_trial_only"],
            "segment_promo_only": counts["outside_population_segment_promo_only"],
            "expiry_beyond_30_days": counts["outside_population_expiry_beyond_30_days"],
        },
        "not_active": counts["not_active"],
    }


@pytest.mark.valkey
def test_a_date_without_a_plan_says_so_and_refuses_the_plan_filter(make_client, valkey):  # noqa: F811
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx, DEC))
    body = get(c, limit=1000).json()
    assert body["plan_status"] == "no plan for this date"
    assert all(x["plan"] is None for x in body["rows"])
    r = get(c, in_plan="true")
    assert r.status_code == 422 and r.json() == error_of(422)
    assert get(c, auto_renew="on").status_code == 200


# --- errors ---------------------------------------------------------------------------------------


@pytest.mark.valkey
@pytest.mark.parametrize("items", MALFORMED)
def test_a_malformed_query_is_422_with_the_registered_body_only(loaded, items):
    r = loaded.get("/at-risk", params=items)
    assert r.status_code == 422 and r.json() == error_of(422)


@pytest.mark.valkey
def test_not_ready_is_503(make_client):  # noqa: F811
    r = make_client().get("/at-risk")
    assert r.status_code == 503
    assert r.json() == error_of(503, "no snapshot is loaded (the store has no date)")


# --- the index, once per store date ---------------------------------------------------------------


@pytest.mark.valkey
def test_the_index_is_built_once_per_store_date(loaded, valkey):
    builds = []

    def counting(r, t):
        builds.append(t)
        return ar.build_index(r, t)

    loaded.app.state.ctx.at_risk_index = ar.IndexCache(counting)
    get(loaded)
    assert builds == []  # an unfiltered page reads the stored ranking directly
    get(loaded, auto_renew="on")
    get(loaded, in_plan="false", horizon=7)
    assert builds == [date(2016, 11, 1)]
    load(valkey, snapshot(loaded.app.state.ctx, DEC))
    body = get(loaded, auto_renew="on").json()
    assert builds == [date(2016, 11, 1), DEC] and body["store_date"] == "2016-12-01"


# --- the scan: header and errors in full, rows one by one with the listed ID (A6) ----------------


def scan_page(body: dict, listed: list[str]) -> list[str]:
    """The header in full; each row with the ID listed at its rank in the stored ranking."""
    header = {k: v for k, v in body.items() if k != "rows"}
    hits = never_used_hits(json.dumps(header))
    for row, own in zip(body["rows"], listed, strict=True):
        hits += response_hits(json.dumps(row), 200, own)
    return hits


@pytest.mark.valkey
def test_every_served_page_passes_the_scan_with_rows_checked_against_their_listed_id(
    make_client,  # noqa: F811
    valkey,
):
    c = make_client()
    s = with_id(snapshot(c.app.state.ctx), "c001", "x/age+01")  # an ID holding "age"
    load(valkey, s)
    seen_age = False
    for h in (7, 30, 59):
        for filters in ({}, *FILTERS[:3]):
            r = get(c, horizon=h, limit=1000, **filters)
            body = r.json()
            listed = [m for m, _ in filtered(valkey, h, filters)]
            assert scan_page(body, listed) == []
            seen_age |= "x/age+01" in listed
            if "x/age+01" in listed:
                assert never_used_hits(r.text) == ["age"]  # only the row's own ID holds it
                i = listed.index("x/age+01")
                other = listed[i - 1] if i else listed[i + 1]
                assert response_hits(json.dumps(body["rows"][i]), 200, other) == ["age"]
    assert seen_age
    for e in (get(c, horizon=60), get(c, city="1"), get(c, tenure="1")):
        assert e.status_code == 422 and never_used_hits(e.text) == []
