"""The 5f smoke check on the hand-made models and the test database; never on real data here."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from retention.api import __main__ as cli
from retention.api import service as svc
from retention.api import smoke
from test_service import (
    MODEL_SETTINGS,
    NOV,
    P,
    bundle_labels,  # noqa: F401 (fixture)
    load,
    pins,  # noqa: F401 (fixture)
    registered,  # noqa: F401 (fixture)
    snapshot,
)

pytestmark = pytest.mark.valkey
KNOWABLE_IN_THE_HAND_MADE_FILE = 2  # written out: cp1 and cp2 of test_service's alerts file


@pytest.fixture()
def start(registered, bundle_labels, pins, valkey, tmp_path):  # noqa: F811
    """start() -> the smoke report on a loaded hand-made snapshot."""

    def go(t=NOV, expected=KNOWABLE_IN_THE_HAND_MADE_FILE, loaded=True):
        def load_ctx():
            return svc.load_context(registered, MODEL_SETTINGS, P, tmp_path / "work", pins=pins)

        if loaded:
            ctx = load_ctx()
            s = snapshot(ctx, t)
            ctx.close()
            load(valkey, s)
        return smoke.run(load_ctx, lambda: valkey, NOV, expected)

    return go


def test_every_endpoint_answers_as_expected_on_a_loaded_store(start):
    report = start()
    assert report["unexpected"] == [] and report["answered_as_expected"] is True
    assert report["label"] == "not a mark; its numbers are never quoted as parity or latency"
    assert [c["call"] for c in report["calls"]] == [
        "GET /health",
        "POST /score scored",
        "POST /score outside_population",
        "POST /score not_active",
        "POST /score unknown",
        "GET /at-risk default",
        "GET /at-risk auto_renew=on",
        "GET /alerts",
    ]
    assert all(c["status"] == c["expected_status"] for c in report["calls"])
    assert all(c["scan_hits"] == 0 for c in report["calls"])
    alerts = report["calls"][-1]
    assert (alerts["found"], alerts["expected"]) == (2, 2)


def test_the_index_is_built_once_and_every_timing_is_recorded(start):
    t = start()["timings"]
    assert t["index_builds"] == 1 and t["index_build_seconds"] > 0
    for k in ("plans_parquet_hash", "readiness_hash_check"):
        assert t[k]["repeats"] == 20 and 0 < t[k]["median_ms"] <= t[k]["max_ms"]


def test_the_report_holds_no_customer_id(start, valkey):
    text = json.dumps(start())
    known = [m.decode() for m in valkey.smembers("rs:known")]
    assert known and not any(m in text for m in known)


def test_customers_are_chosen_by_the_fixed_rule(start, valkey):
    start()
    picks = smoke.pick_customers(valkey)
    known = sorted(m.decode() for m in valkey.smembers("rs:known"))
    docs = {m: valkey.get(f"rs:customer:{m}") for m in known}
    outside = [m for m in known if docs[m] and json.loads(docs[m])["status"] != "scored"]
    inactive = [m for m in known if docs[m] is None]
    assert picks["outside_population"] == outside[0] and picks["not_active"] == inactive[0]
    top = valkey.zrange("rs:rank:30", 0, 0)[0].decode()
    assert picks["scored"] == top


def test_a_wrong_answer_is_named_as_unexpected(start):
    report = start(expected=3)
    assert report["unexpected"] == ["GET /alerts"] and report["answered_as_expected"] is False


def test_a_store_without_the_snapshot_s_date_stops_the_check(start):
    report = start(loaded=False)
    assert report["unexpected"] == ["the store's date is None, not 2016-11-01"]
    assert "calls" not in report


# --- the command ---------------------------------------------------------------------------------


def args():
    return SimpleNamespace(fraction=None, snapshot="2016-11-01")


def test_smoke_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.smoke_cmd(args())


def test_smoke_runs_once_and_writes_only_its_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "head_commit", lambda repo: "abc1234")
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    monkeypatch.setattr(smoke, "run", lambda *a: {"label": smoke.NOT_A_MARK, "unexpected": []})
    assert cli.smoke_cmd(args()) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["smoke_2016-11-01.json"]
    with pytest.raises(SystemExit, match="runs once"):
        cli.smoke_cmd(args())


def test_smoke_is_registered_for_2016_11_01_only(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    with pytest.raises(SystemExit, match="2016-11-01 only"):
        cli.smoke_cmd(SimpleNamespace(fraction=None, snapshot="2016-10-01"))
