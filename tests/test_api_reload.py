"""The registered reload of database 0 (configs/serving.yaml, step8.reload_database_0).

Every refusal happens before anything is renamed or written; the earlier report is set
aside under its git head, never overwritten; the marks are judged exactly. Valkey tests
use the test database (15), standing in for database 0 where the check takes a database.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from retention.api import __main__ as cli
from retention.api import store
from retention.api.config import load_serving

MARKS = load_serving()["step8"]["reload_database_0"]["marks"]
COUNTS = {  # the D-209 load's counts, as load-snapshot writes them
    "scored": 169673,
    "outside_population_segment_trial_only": 8890,
    "outside_population_segment_promo_only": 31,
    "outside_population_expiry_beyond_30_days": 51692,
    "total_with_coverage": 230286,
    "not_active": 195234,
}


def test_the_registered_marks_are_the_seven_counts_and_the_key_total():
    assert set(MARKS) == set(COUNTS) | {"known", "keys_database_0"}
    assert MARKS["known"] == 425520 and MARKS["keys_database_0"] == 230294
    assert MARKS["keys_database_0"] == MARKS["total_with_coverage"] + 1 + 3 + 1 + 2 + 1


# --- the marks ----------------------------------------------------------------------------------


def test_equal_counts_meet_every_mark():
    judged = cli.reload_marks(COUNTS, 425520, 230294)
    assert judged["met"] and all(m["met"] for m in judged["marks"].values())
    assert len(judged["marks"]) == 8


@pytest.mark.parametrize(
    "counts, known, keys, wrong",
    [
        (COUNTS | {"scored": 169672}, 425520, 230294, "scored"),
        (COUNTS, 425521, 230294, "known"),
        (COUNTS, 425520, 230295, "keys_database_0"),
        ({k: v for k, v in COUNTS.items() if k != "not_active"}, 425520, 230294, "not_active"),
    ],
)
def test_any_difference_is_not_met_and_named(counts, known, keys, wrong):
    judged = cli.reload_marks(counts, known, keys)
    assert not judged["met"]
    assert [k for k, m in judged["marks"].items() if not m["met"]] == [wrong]


# --- setting the earlier report aside ------------------------------------------------------------


def earlier(tmp_path, head="cd9ca4b"):
    path = tmp_path / cli.LOAD_REPORT
    path.write_text(json.dumps({"git_head": head, "passed": True}))
    return path


def test_the_earlier_report_is_renamed_under_its_git_head_with_the_reason(tmp_path):
    path = earlier(tmp_path)
    got = cli.set_aside_report(path, "store wiped by the test fixture (D-211)")
    assert got == {
        "reason": "store wiped by the test fixture (D-211)",
        "earlier_git_head": "cd9ca4b",
        "earlier_report": "load_snapshot_2016-11-01_cd9ca4b.json",
    }
    assert not path.exists()
    renamed = tmp_path / "load_snapshot_2016-11-01_cd9ca4b.json"
    assert json.loads(renamed.read_text()) == {"git_head": "cd9ca4b", "passed": True}


@pytest.mark.parametrize("reason", ["", "   "])
def test_an_empty_reason_is_refused(tmp_path, reason):
    path = earlier(tmp_path)
    with pytest.raises(SystemExit, match="needs a reason"):
        cli.set_aside_report(path, reason)
    assert path.exists()


def test_a_missing_earlier_report_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        cli.set_aside_report(tmp_path / cli.LOAD_REPORT, "r")


def test_a_report_without_a_git_head_is_refused(tmp_path):
    path = earlier(tmp_path, head="")
    with pytest.raises(SystemExit, match="no git_head"):
        cli.set_aside_report(path, "r")
    assert path.exists()


def test_an_existing_renamed_file_is_never_overwritten(tmp_path):
    path = earlier(tmp_path)
    taken = tmp_path / "load_snapshot_2016-11-01_cd9ca4b.json"
    taken.write_text("{}")
    with pytest.raises(SystemExit, match="exists; nothing renamed"):
        cli.set_aside_report(path, "r")
    assert path.exists() and taken.read_text() == "{}"


# --- the store the reload writes -----------------------------------------------------------------


@pytest.mark.valkey
def test_the_reload_writes_only_an_empty_database_it_is_on(valkey):
    cli.check_reload_store(valkey, 15)  # empty, and the server says 15
    with pytest.raises(SystemExit, match="the client is on 15"):
        cli.check_reload_store(valkey)  # the registered database is 0
    valkey.set("rs:store_date", "2016-11-01")
    with pytest.raises(SystemExit, match="holds 1 keys"):
        cli.check_reload_store(valkey, 15)


# --- the command ---------------------------------------------------------------------------------


def args(**kw):
    return SimpleNamespace(**({"fraction": None, "snapshot": "2016-11-01"} | kw))


@pytest.fixture()
def quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    return tmp_path


def test_without_a_reason_a_second_load_is_refused_by_the_run_once_guard(quiet):
    earlier(quiet)
    with pytest.raises(SystemExit, match="runs once"):
        cli.load_snapshot_cmd(args())


def test_an_empty_reason_is_refused_before_the_store_is_touched(quiet, monkeypatch):
    path = earlier(quiet)
    monkeypatch.setattr(store, "client", lambda: pytest.fail("the store was touched"))
    with pytest.raises(SystemExit, match="needs a reason"):
        cli.load_snapshot_cmd(args(reload_reason="  "))
    assert path.exists()


@pytest.mark.valkey
def test_a_store_on_another_database_refuses_before_anything_is_renamed(quiet, monkeypatch, valkey):
    path = earlier(quiet)
    monkeypatch.setattr(store, "client", lambda: valkey)
    with pytest.raises(SystemExit, match="writes database 0"):
        cli.load_snapshot_cmd(args(reload_reason="store wiped (D-211)"))
    assert path.exists() and sorted(p.name for p in quiet.iterdir()) == [cli.LOAD_REPORT]


def test_the_reload_reason_is_a_load_snapshot_option(monkeypatch):
    got = []
    monkeypatch.setattr(cli, "load_snapshot_cmd", lambda a: got.append(a) or 0)
    cli.main(["load-snapshot", "--snapshot", "2016-11-01", "--reload-reason", "why"])
    assert got[0].reload_reason == "why" and got[0].snapshot == "2016-11-01"
