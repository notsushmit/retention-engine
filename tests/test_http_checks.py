"""Step 8's HTTP checks (retention.api.http_checks and the three commands) on hand-made data.

The service runs in-process on the hand-made models and pins of test_service, with a
store loaded into the test database (15); FastAPI's test client stands in for the
stack, so the functions the real runs use are the ones tested. No report holds an ID.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from retention.api import __main__ as cli
from retention.api import http_checks as hc
from retention.api.config import SERVING_CONFIG, file_sha256
from test_app import loaded, make_client  # noqa: F401
from test_service import bundle_labels, pins, registered  # noqa: F401

pytestmark = pytest.mark.valkey


@pytest.fixture()
def served(loaded):  # noqa: F811
    """The in-process service on a store loaded into the test database (test_app.loaded)."""
    return loaded


def test_the_registered_draws_and_limits():
    assert (hc.SAMPLE_SEED, hc.WARMUP_SEED) == (20161001, 20161002)
    assert (hc.SAMPLE_SIZE, hc.WARMUP_SIZE, hc.P95_MS_MAX) == (1000, 50, 100.0)


# --- the draws ----------------------------------------------------------------------------------

IDS = [f"id{i:05d}" for i in range(3000)]


def test_the_sample_is_fixed_and_does_not_depend_on_the_input_order():
    a = hc.sample(IDS)
    assert a == hc.sample(list(reversed(IDS))) and len(a) == len(set(a)) == 1000
    assert a != sorted(a)  # the draw's order, not sorted


def test_the_warmup_is_drawn_from_outside_the_sample():
    drawn = hc.sample(IDS)
    warm = hc.warmup(IDS, drawn)
    assert len(warm) == 50 and not set(warm) & set(drawn)
    assert warm == hc.warmup(list(reversed(IDS)), drawn)


# --- parity -------------------------------------------------------------------------------------


def test_every_http_body_equals_the_host_answer(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)
    got = hc.parity(c, c.app.state.ctx, valkey, ids)
    assert got["equal"] == got["customers"] == len(ids) > 20 and got["differing"] == 0
    assert got["http_status"] == {"200": len(ids)} and got["differing_fields"] == {}
    assert not got["met"]  # the registered mark is 1,000 customers


class Planted:
    """A client that changes one field of one /score answer."""

    def __init__(self, c, field):
        self.c, self.field, self.done = c, field, False

    def post(self, url, json):
        r = self.c.post(url, json=json)
        if self.done:
            return r
        self.done = True
        body = r.json() | {self.field: "planted"}
        return SimpleNamespace(status_code=r.status_code, json=lambda: body, text="")

    def get(self, url):
        return self.c.get(url)


def test_a_changed_field_is_counted_by_name_with_no_id(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)[:10]
    got = hc.parity(Planted(c, "archetype"), c.app.state.ctx, valkey, ids)
    assert got["differing"] == 1 and got["differing_fields"] == {"archetype": 1}
    assert not any(m in json.dumps(got) for m in ids)


# --- latency ------------------------------------------------------------------------------------


def test_percentiles_and_the_p95_verdict_on_fixed_timings():
    ms = [float(i) for i in range(1, 101)]
    got = hc.summarize(ms)
    assert (got["p50_ms"], got["p95_ms"], got["p99_ms"]) == (50.5, 95.05, 99.01)
    assert (got["mean_ms"], got["max_ms"]) == (50.5, 100.0) and got["met"]
    assert not hc.summarize(ms, limit=95.0)["met"]


def test_a_latency_run_times_every_call_and_reports_statuses(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)
    got = hc.latency(c, ids[:3], ids[3:13])
    assert got["calls"] == 10 and got["warmup_calls"] == 3
    assert got["http_status"] == {"200": 10} and got["warmup_http_status"] == {"200": 3}
    assert 0 < got["p50_ms"] <= got["p95_ms"] <= got["p99_ms"] <= got["max_ms"]
    assert not got["met"]  # 10 calls, not the registered 1,000


# --- concurrency, December, config, registry ----------------------------------------------------


def test_concurrent_answers_equal_the_sequential_ones(served, valkey):
    c, _ = served
    got = hc.concurrency(lambda: c, hc.scored_ids(valkey), threads=8, at_risk_calls=10)
    assert got["met"] and got["server_errors"] == 0
    assert got["score_differing_from_sequential"] == got["at_risk_differing_from_sequential"] == 0


class Flaky:
    """Answers the sequential pass normally, then changes every later /score answer once
    the concurrent pass starts."""

    def __init__(self, c, sequential):
        self.c, self.left = c, sequential

    def post(self, url, json):
        r = self.c.post(url, json=json)
        self.left -= 1
        if self.left < 0:
            return SimpleNamespace(status_code=r.status_code, text=r.text + " ")
        return r

    def get(self, url):
        return self.c.get(url)


def test_a_concurrent_answer_that_differs_is_flagged(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)[:12]
    flaky = Flaky(c, len(ids))
    got = hc.concurrency(lambda: flaky, ids, threads=4, at_risk_calls=4)
    assert not got["met"] and got["score_differing_from_sequential"] == len(ids)


def test_december_2016_over_http_is_409_with_no_reasons_and_a_clean_scan(served, valkey):
    c, _ = served
    got = hc.december(c, hc.scored_ids(valkey))
    assert got == {
        "as_of": "2016-12-01",
        "http_status": 409,
        "reasons_in_body": False,
        "scan_hits": 0,
        "met": True,
    }


def test_the_served_config_is_the_committed_file(served):
    c, _ = served
    assert hc.config_matches(c, file_sha256(SERVING_CONFIG))["met"]
    assert not hc.config_matches(c, "0" * 64)["met"]


def test_a_changed_or_added_file_changes_the_tree_hash(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.db").write_bytes(b"1")
    before = hc.tree_sha256(tmp_path)
    (tmp_path / "a" / "x.db").write_bytes(b"2")
    assert hc.tree_sha256(tmp_path) != before
    (tmp_path / "a" / "x.db").write_bytes(b"1")
    assert hc.tree_sha256(tmp_path) == before
    (tmp_path / "y").write_bytes(b"")
    assert hc.tree_sha256(tmp_path)["files"] == 2


# --- the commands ---------------------------------------------------------------------------------


def args():
    return SimpleNamespace(fraction=None, url=None)


@pytest.fixture()
def quiet(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    (mlruns / "mlflow.db").write_bytes(b"db")
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "head_commit", lambda repo: "abc1234")
    monkeypatch.setattr(cli, "out_dir", lambda fraction: out)
    monkeypatch.setattr(cli, "STORE_DATABASE", 15)
    monkeypatch.setattr(cli, "MLRUNS", mlruns)
    monkeypatch.setattr(hc, "SAMPLE_SIZE", 12)
    monkeypatch.setattr(hc, "WARMUP_SIZE", 3)
    return out, mlruns


def write_load_report(out, mlruns, met=True):
    reload = {
        "reason": "r",
        "mlruns_before": hc.tree_sha256(mlruns),
        "mlflow_db_sha256_before": file_sha256(mlruns / "mlflow.db"),
    }
    (out / cli.LOAD_REPORT).write_text(json.dumps({"reload": reload, "met": met}))


def test_each_command_needs_the_one_before(quiet, valkey):
    out, mlruns = quiet
    with pytest.raises(SystemExit, match="reload of database 0 runs first"):
        cli.http_parity_cmd(args(), client=object(), ctx=object(), r=valkey)
    write_load_report(out, mlruns, met=False)
    with pytest.raises(SystemExit, match="reload of database 0 runs first"):
        cli.http_parity_cmd(args(), client=object(), ctx=object(), r=valkey)
    with pytest.raises(SystemExit, match="http-parity runs first"):
        cli.latency_cmd(args(), client=object(), r=valkey)
    with pytest.raises(SystemExit, match="latency run comes first"):
        cli.http_checks_cmd(args(), client_for=object, r=valkey)


def test_the_commands_run_once_in_order_and_write_no_id(quiet, served, valkey):
    out, mlruns = quiet
    c, _ = served
    write_load_report(out, mlruns)
    assert cli.http_parity_cmd(args(), client=c, ctx=c.app.state.ctx, r=valkey) == 0
    assert cli.latency_cmd(args(), client=c, r=valkey) in (0, 1)  # timing depends on the machine
    assert cli.http_checks_cmd(args(), client_for=lambda: c, r=valkey) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(
        [cli.LOAD_REPORT, cli.HTTP_PARITY_REPORT, cli.LATENCY_REPORT, cli.HTTP_CHECKS_REPORT]
    )
    parity = json.loads((out / cli.HTTP_PARITY_REPORT).read_text())
    assert parity["met"] and parity["equal"] == 12 and parity["health"]["ready"] is True
    checks = json.loads((out / cli.HTTP_CHECKS_REPORT).read_text())
    assert checks["met"] and checks["mlflow_after_startup"]["met"]
    ids = hc.scored_ids(valkey)
    for name in names[1:]:
        assert not any(m in (out / name).read_text() for m in ids)
    for call in (
        lambda: cli.http_parity_cmd(args(), client=c, ctx=c.app.state.ctx, r=valkey),
        lambda: cli.latency_cmd(args(), client=c, r=valkey),
        lambda: cli.http_checks_cmd(args(), client_for=lambda: c, r=valkey),
    ):
        with pytest.raises(SystemExit, match="runs once"):
            call()


def test_a_changed_registry_fails_the_mlflow_check(quiet, served, valkey):
    out, mlruns = quiet
    c, _ = served
    write_load_report(out, mlruns)
    (out / cli.LATENCY_REPORT).write_text(json.dumps({"met": True}))
    (mlruns / "mlflow.db").write_bytes(b"migrated")
    assert cli.http_checks_cmd(args(), client_for=lambda: c, r=valkey) == 1
    checks = json.loads((out / cli.HTTP_CHECKS_REPORT).read_text())
    assert not checks["mlflow_after_startup"]["met"] and checks["concurrency"]["met"]


def test_a_store_on_another_database_is_refused(quiet, valkey, monkeypatch):
    out, mlruns = quiet
    write_load_report(out, mlruns)
    monkeypatch.setattr(cli, "STORE_DATABASE", 0)
    with pytest.raises(SystemExit, match="database 0, not 15"):
        cli.http_parity_cmd(args(), client=object(), ctx=object(), r=valkey)


# --- the parity diagnosis (D-244) ---------------------------------------------------------------


def test_adjacent_floats_are_one_ulp_apart_on_both_sides_of_zero():
    import math

    for x in (0.1234, 1e-300, 0.5, 1.0):
        assert hc.ulps(x, math.nextafter(x, 2.0)) == 1
    assert hc.ulps(-0.25, math.nextafter(-0.25, 0.0)) == 1
    assert hc.ulps(0.0, -0.0) == 0 and hc.ulps(0.3, 0.3) == 0


def test_gaps_are_kept_by_schema_path_with_no_value():
    import math

    a = {
        "risk": {"days_7": {"probability": 0.2, "text": "low"}},
        "reasons": [{"text": "left-text"}],
    }
    b = {
        "risk": {"days_7": {"probability": math.nextafter(0.2, 1.0), "text": "low"}},
        "reasons": [{"text": "right-text"}],
    }
    got = hc.leaf_gaps([(a, b), (a, a)])
    assert set(got) == {"risk.days_7.probability", "reasons.0.text"}
    p = got["risk.days_7.probability"]
    assert p["differing"] == 1 and p["max_ulps"] == 1 and 0 < p["max_rel_gap"] < 1e-15
    assert got["reasons.0.text"] == {"differing": 1, "kind": "not a float pair"}
    assert "-text" not in json.dumps(got) and "0.2" not in json.dumps(got)


def test_a_diagnosis_of_equal_sides_finds_nothing(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)[:15]
    got = hc.diagnose(c, c.app.state.ctx, valkey, ids)
    assert got["http_differs_from_itself"] == got["host_differs_from_itself"] == 0
    assert got["http_differs_from_host"] == 0 and got["gaps_by_path"] == {}
    assert got["action_changed"] == 0


class Nudged:
    """A client whose answers carry a one-ulp change in the 30-day pre-fix probability and,
    for the first customer, another action."""

    def __init__(self, c):
        self.c, self.first = c, None

    def post(self, url, json):
        import math

        body = self.c.post(url, json=json).json()
        r = body["reasons"][1]
        r["probability_before_fix"] = math.nextafter(r["probability_before_fix"], 2.0)
        self.first = self.first or json["msno"]
        if json["msno"] == self.first:
            body["action"]["highest_net_value_action"] = "planted"
        return SimpleNamespace(status_code=200, json=lambda: body, text="")


def test_a_diagnosis_reports_ulps_reason_positions_and_whether_an_action_follows_a_gap(
    served, valkey
):
    c, _ = served
    ids = hc.scored_ids(valkey)[:10]
    got = hc.diagnose(Nudged(c), c.app.state.ctx, valkey, ids)
    assert got["http_differs_from_itself"] == 0 and got["http_differs_from_host"] == 10
    gap = got["gaps_by_path"]["reasons.1.probability_before_fix"]
    assert gap["differing"] == 10 and gap["max_ulps"] == 1
    assert got["reason_positions_changed"] == {"0": 0, "1": 10}
    assert got["action_changed"] == 1
    assert got["action_changed_with_a_probability_gap"] == 1
    assert got["action_changed_with_a_30_day_pre_fix_gap"] == 1
    assert not any(m in json.dumps(got) for m in ids)


class Unstable:
    """Answers differ between the first and the second pass."""

    def __init__(self, c, n):
        self.c, self.left = c, n

    def post(self, url, json):
        body = self.c.post(url, json=json).json()
        self.left -= 1
        if self.left < 0:
            body["paid_days"]["value"] += 1.0
        return SimpleNamespace(status_code=200, json=lambda: body, text="")


def test_a_side_that_differs_from_itself_is_counted(served, valkey):
    c, _ = served
    ids = hc.scored_ids(valkey)[:6]
    got = hc.diagnose(Unstable(c, len(ids)), c.app.state.ctx, valkey, ids)
    assert got["http_differs_from_itself"] == 6 and got["host_differs_from_itself"] == 0


def test_the_diagnosis_command_reads_only_runs_once_and_writes_no_id(quiet, served, valkey):
    out, _ = quiet
    c, _ = served
    with pytest.raises(SystemExit, match="http-parity runs first"):
        cli.http_diagnose_cmd(args(), client=c, ctx=c.app.state.ctx, r=valkey)
    (out / cli.HTTP_PARITY_REPORT).write_text(json.dumps({"met": False}))
    keys = valkey.dbsize()
    assert cli.http_diagnose_cmd(args(), client=c, ctx=c.app.state.ctx, r=valkey) == 0
    report = json.loads((out / cli.HTTP_DIAGNOSIS_REPORT).read_text())
    assert report["store_keys_before"] == report["store_keys_after"] == keys == valkey.dbsize()
    assert report["customers"] == 12 and report["http_differs_from_host"] == 0
    text = (out / cli.HTTP_DIAGNOSIS_REPORT).read_text()
    assert not any(m in text for m in hc.scored_ids(valkey))
    with pytest.raises(SystemExit, match="runs once"):
        cli.http_diagnose_cmd(args(), client=c, ctx=c.app.state.ctx, r=valkey)
