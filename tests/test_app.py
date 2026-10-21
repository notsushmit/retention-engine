"""The scoring API over HTTP: startup refusal, GET /health, POST /score, every status and error.

The app runs on the hand-made models of test_service (registered in a temporary
MLflow store) and on Valkey database 15. Every file a test changes is a temporary
copy: the hand-made bundle in the temporary store, or the app's own download.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from retention.api import app as api
from retention.api import schemas as sch
from retention.api import service as svc
from retention.api import store as st
from retention.api.labels import LABELS
from retention.api.saved import ModelRefused
from retention.api.schemas import never_used_hits, response_hits
from test_registry import store_folder
from test_service import (
    MALFORMED,
    MODEL_SETTINGS,
    N_RANDOM,
    N_SCORED,
    NOV,
    RANDOM_SEED,
    UNKNOWN,
    P,
    bundle_labels,  # noqa: F401 (fixture)
    load,
    registered,  # noqa: F401 (fixture)
    snapshot,
    unreachable,
)

pytestmark = pytest.mark.valkey


def flip_last_byte(path) -> None:
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))


@pytest.fixture()
def make_client(registered, bundle_labels, valkey, tmp_path):  # noqa: F811
    """make_client(connect=...) -> a started TestClient on the hand-made models."""
    clients = []

    def start(connect=None, settings=None):
        app = api.create_app(
            load=lambda: svc.load_context(
                settings or registered, MODEL_SETTINGS, P, tmp_path / f"work{len(clients)}"
            ),
            connect=connect or (lambda: valkey),
        )
        c = TestClient(app, raise_server_exceptions=False)
        c.__enter__()
        clients.append(c)
        return c

    yield start
    for c in clients:
        c.__exit__(None, None, None)


def score(client, **body):
    return client.post("/score", json=body)


def error_of(status: int, reason: str | None = None) -> dict:
    return sch.error_body(status, reason).model_dump(mode="json")


# --- startup ------------------------------------------------------------------------------------


def test_startup_is_refused_when_a_registered_file_changed(registered, bundle_labels, tmp_path):  # noqa: F811
    folder = store_folder(registered)
    keep = tmp_path / "keep"
    shutil.copytree(folder, keep)
    try:
        flip_last_byte(folder / "booster_30d.txt")
        app = api.create_app(
            load=lambda: svc.load_context(registered, MODEL_SETTINGS, P, tmp_path / "work"),
            connect=unreachable,
        )
        with pytest.raises(ModelRefused, match="booster_30d"):
            with TestClient(app):
                pass
    finally:
        shutil.copyfile(keep / "booster_30d.txt", folder / "booster_30d.txt")


def test_startup_is_refused_without_the_alias(registered, bundle_labels, tmp_path):  # noqa: F811
    other = dataclasses.replace(registered, name="not-registered")
    app = api.create_app(
        load=lambda: svc.load_context(other, MODEL_SETTINGS, P, tmp_path / "work"),
        connect=unreachable,
    )
    with pytest.raises(ModelRefused, match="not registered"):
        with TestClient(app):
            pass


def test_valkey_down_does_not_stop_startup(make_client):
    c = make_client(connect=unreachable)
    assert c.get("/health").status_code == 503


# --- /health ------------------------------------------------------------------------------------


def test_health_is_200_with_every_registered_field_once_loaded(make_client, valkey):
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx))
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "ready",
        "reason",
        "store_date",
        "replay_progress",
        "model_checksums_match",
        "config_sha256",
        "valkey_reachable",
    }
    assert body["ready"] is True and body["reason"] is None
    assert body["store_date"] == "2016-11-01"
    assert body["replay_progress"]["expected_events"] == 3727472


def test_not_ready_without_a_store_date(make_client):
    c = make_client()
    h = c.get("/health")
    assert (
        h.status_code == 503
        and h.json()["reason"] == "no snapshot is loaded (the store has no date)"
    )
    r = score(c, msno="c000")
    assert r.status_code == 503
    assert r.json() == error_of(503, "no snapshot is loaded (the store has no date)")


def test_a_changed_model_file_makes_health_and_score_503(make_client, valkey):
    c = make_client()
    load(valkey, snapshot(c.app.state.ctx))
    flip_last_byte(c.app.state.ctx.bundle_dir / "dt01.txt")  # the app's own downloaded copy
    h = c.get("/health")
    assert h.status_code == 503
    body = h.json()
    assert body["ready"] is False and body["model_checksums_match"] is False
    assert "dt01.txt" in body["reason"] and body["reason"].startswith("model files differ")
    r = score(c, msno="c000")
    assert r.status_code == 503 and r.json()["reason"] == body["reason"]


def test_valkey_down_makes_health_and_score_503(make_client):
    c = make_client(connect=unreachable)
    h = c.get("/health")
    assert h.status_code == 503
    assert h.json()["valkey_reachable"] is False
    assert h.json()["reason"] == "Valkey cannot be reached"
    assert score(c, msno="c000").json() == error_of(503, "Valkey cannot be reached")


# --- /score -------------------------------------------------------------------------------------


@pytest.fixture()
def loaded(make_client, valkey):
    c = make_client()
    s = snapshot(c.app.state.ctx)
    load(valkey, s)
    return c, s


def score_each(c, valkey, s, n: int) -> tuple[list[str], list[str]]:
    """Every scored customer through /score, one at a time; the customers whose body
    differs from the batch output, and those whose probabilities differ from the
    stored ranking (every customer is checked for both before anything fails)."""
    batch = svc.scored_bodies(c.app.state.ctx, s.t, s.rows, s.prices)
    stored = {h: dict(st.ranking(valkey, h, 0, n)) for h in (7, 30, 59)}
    assert all(len(v) == n for v in stored.values())
    body_differs, rank_differs = [], []
    for i, m in enumerate(s.rows.msno):
        r = score(c, msno=m)
        got = r.json()
        if r.status_code != 200 or got != batch[i] or got["msno"] != m:
            body_differs.append(m)
        if any(got["risk"][f"days_{h}"]["probability"] != stored[h][m] for h in (7, 30, 59)):
            rank_differs.append(m)
    return body_differs, rank_differs


def test_score_equals_the_batch_and_the_ranking_for_every_customer(loaded, valkey):
    c, s = loaded
    assert score_each(c, valkey, s, N_SCORED) == ([], [])


@pytest.mark.random2000
def test_score_equals_the_batch_and_the_ranking_on_a_random_2000_row_set(make_client, valkey):
    c = make_client()
    s = snapshot(c.app.state.ctx, n=N_RANDOM, seed=RANDOM_SEED)
    load(valkey, s)
    assert score_each(c, valkey, s, N_RANDOM) == ([], [])


def test_outside_and_not_active_answers(loaded):
    c, _ = loaded
    assert score(c, msno="o-trial").json() == {
        "status": "outside_population",
        "msno": "o-trial",
        "features_as_of": "2016-11-01",
        "code": "segment_trial_only",
        "next_expiry_in_days": 12,
    }
    assert score(c, msno="o-unseen").json()["code"] == "unseen_level:last_payment_method_id"
    assert score(c, msno="n-inactive-1").json() == {
        "status": "not_active",
        "msno": "n-inactive-1",
        "features_as_of": "2016-11-01",
    }


def test_unknown_customer_is_404(loaded):
    c, _ = loaded
    r = score(c, msno=UNKNOWN)
    assert r.status_code == 404 and r.json() == error_of(404)


def test_as_of_other_than_the_store_date_is_409_and_equal_is_200(loaded):
    c, _ = loaded
    r = score(c, msno="c000", as_of="2016-10-01")
    assert r.status_code == 409 and r.json() == error_of(409)
    assert score(c, msno="c000", as_of=NOV.isoformat()).status_code == 200


@pytest.mark.parametrize("body", MALFORMED)
def test_malformed_input_is_422_with_the_registered_body_only(loaded, body):
    c, _ = loaded
    r = c.post("/score", json=body)
    assert r.status_code == 422 and r.json() == error_of(422)


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"c000"'])
def test_a_body_that_is_not_a_json_object_is_422(loaded, raw):
    c, _ = loaded
    r = c.post("/score", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json() == error_of(422)


def test_a_response_the_schemas_refuse_is_a_500_not_a_partial_answer(loaded, monkeypatch):
    c, _ = loaded
    monkeypatch.setattr(svc.sc, "paid_days_block", lambda days: {"value": days})
    r = score(c, msno="c000")
    assert r.status_code == 500 and r.text == "Internal Server Error"


# --- the never-used scan over every endpoint, error bodies included --------------------------


def every_body(make_client, valkey, monkeypatch) -> dict[str, str]:
    bodies = {}
    down = make_client(connect=unreachable)
    bodies["health 503 valkey"] = down.get("/health").text
    bodies["score 503 valkey"] = score(down, msno="c000").text
    c = make_client()
    bodies["health 503 no date"] = c.get("/health").text
    bodies["score 503 no date"] = score(c, msno="c000").text
    s = snapshot(c.app.state.ctx)
    load(valkey, s)
    bodies["health 200"] = c.get("/health").text
    for m in [*s.rows.msno, "o-trial", "o-beyond", "o-unseen", "n-inactive-1"]:
        bodies[f"score {m}"] = score(c, msno=m).text
    for word in LABELS.never_in_any_body:  # a caller naming an attribute gets none back
        bodies[f"score 404 {word}"] = score(c, msno=word).text
        bodies[f"score 422 field {word}"] = c.post("/score", json={"msno": "c000", word: 1}).text
        bodies[f"score 422 value {word}"] = score(c, msno="c000", as_of=word).text
    bodies["score 409"] = score(c, msno="c000", as_of="2016-10-01").text
    for i, body in enumerate(MALFORMED):
        bodies[f"score 422 #{i}"] = c.post("/score", json=body).text
    bodies["unknown path"] = c.get("/nowhere").text
    bodies["wrong method"] = c.get("/score").text
    bodies["score 404 placeholder"] = score(c, msno=svc.PAIR_MSNO).text

    def fails(*args):
        raise ValueError(f"scoring failed for {svc.PAIR_MSNO}")

    monkeypatch.setattr(svc, "scored_bodies", fails)
    bodies["score 500"] = score(c, msno="c000").text
    monkeypatch.undo()
    flip_last_byte(c.app.state.ctx.bundle_dir / "model_7d.json")
    bodies["health 503 model"] = c.get("/health").text
    bodies["score 503 model"] = score(c, msno="c000").text
    return bodies


def test_no_body_of_any_endpoint_names_a_never_used_attribute(make_client, valkey, monkeypatch):
    bodies = every_body(make_client, valkey, monkeypatch)
    assert len(bodies) >= N_SCORED + 30
    hits = {k: never_used_hits(v) for k, v in bodies.items() if never_used_hits(v)}
    assert hits == {}
    scored = json.loads(bodies["score c000"])
    assert scored["status"] == "scored" and scored["reasons"] is not None


def test_no_body_carries_the_placeholder_or_the_duplicate_row(make_client, valkey, monkeypatch):
    bodies = every_body(make_client, valkey, monkeypatch)
    assert [k for k, v in bodies.items() if svc.PAIR_MSNO in v] == []
    assert bodies["score 500"] == "Internal Server Error"
    for k, v in bodies.items():
        if k.startswith("score c"):  # a scored customer: one object, that customer's own
            body = json.loads(v)
            assert isinstance(body, dict) and body["msno"] == k.removeprefix("score ")
            assert v.count('"msno"') == 1


# --- amendment A6 over HTTP ----------------------------------------------------------------------

AGE_SCORED, AGE_INACTIVE, AGE_UNKNOWN = "x/age+01", "n/age+02", "y/age+03"


def with_id(s, old: str, new: str):
    """The snapshot with one scored customer's ID replaced in every table."""
    for frame in (s.rows, s.prices, s.status, s.finals):
        frame["msno"] = frame.msno.astype(str).replace(old, new)
    s.known = sorted(new if m == old else m for m in s.known)
    return s


def test_a6_a_stored_customer_whose_id_contains_age(make_client, valkey):
    c = make_client()
    s = with_id(snapshot(c.app.state.ctx), "c000", AGE_SCORED)
    s.known = sorted([*s.known, AGE_INACTIVE])
    load(valkey, s)
    r = score(c, msno=AGE_SCORED)
    assert r.status_code == 200
    got = r.json()
    assert (got["status"], got["msno"]) == ("scored", AGE_SCORED)
    assert never_used_hits(r.text) == ["age"]  # the plain scan flags the ID itself
    assert response_hits(r.text, 200, AGE_SCORED) == []  # A6: the own ID, in msno only
    assert response_hits(r.text, 200, "c001") == ["age"]  # another customer's ID in msno
    other = json.dumps({**got, "features_as_of": AGE_SCORED})
    assert response_hits(other, 200, AGE_SCORED) == ["age"]  # the same string elsewhere
    assert response_hits(r.text, 503, AGE_SCORED) == ["age"]  # an error status: in full
    n = score(c, msno=AGE_INACTIVE)
    assert n.json()["status"] == "not_active"
    assert response_hits(n.text, 200, AGE_INACTIVE) == []
    for req in ({"msno": AGE_UNKNOWN}, {"msno": AGE_SCORED, "as_of": "2016-10-01"}):
        e = score(c, **req)
        assert e.status_code in (404, 409)
        assert response_hits(e.text, e.status_code, req["msno"]) == []  # nothing echoed
