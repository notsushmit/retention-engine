"""The drift replays: the listening shift and the drift-replay command (configs/ops.yaml, R1/A15).

Hand-made data only; the Valkey test database (15) stands in for databases 1 and 2.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from retention.api import __main__ as api_cli
from retention.features import spec
from retention.stream import __main__ as cli
from retention.stream import consumer as cs
from retention.stream import move, parity, replay, source
from retention.stream.source import halve_listening
from test_stream_move import CFG, T, batch_table, no_test_unlock, raw  # noqa: F401
from test_stream_parity import COLUMNS, MAIN, P, tiny_models  # noqa: F401
from test_stream_replay import GATHER, args, fake_run
from test_stream_replay import quiet as quiet  # noqa: PLC0414 (fixture)
from test_stream_source import folder, log, tx, write  # noqa: F401

# --- the registered shift ----------------------------------------------------------------------


def test_the_shift_halves_seconds_and_rounds_every_count_down():
    row = {
        "msno": "a",
        "date": 20161005,
        "num_25": 5,
        "num_50": 3,
        "num_75": 1,
        "num_985": 0,
        "num_100": 9,
        "num_unq": 7,
        "total_secs": 101.0,
    }
    got = halve_listening(row)
    assert got["total_secs"] == 50.5
    assert [got[c] for c in source.LISTENING_COUNTS] == [2, 1, 0, 0, 4, 3]
    assert got["msno"] == "a" and got["date"] == 20161005
    assert row["num_25"] == 5  # the input is not changed


def test_a_missing_value_stays_missing():
    got = halve_listening(
        {"msno": "a", "date": 1, "num_25": None, "total_secs": None, "num_100": 4}
    )
    assert got["num_25"] is None and got["total_secs"] is None and got["num_100"] == 2


def test_the_shift_changes_listening_events_only_and_keeps_order_count_and_numbers(folder):  # noqa: F811
    plain = list(source.events(folder))
    shifted = list(source.events(folder, shift=True))
    assert len(plain) == len(shifted) > 0
    assert [(e["seq"], e["source"], e["msno"]) for e in plain] == [
        (e["seq"], e["source"], e["msno"]) for e in shifted
    ]
    kinds = set()
    for a, b in zip(plain, shifted, strict=True):
        kinds.add(a["source"] == "user_logs")
        if a["source"] == "user_logs":
            assert b["total_secs"] == a["total_secs"] * 0.5 and b["num_100"] == a["num_100"] // 2
        else:
            assert a == b  # transaction events are never touched
    assert kinds == {True, False}


def test_the_bootstrap_reader_has_no_shift_argument():
    import inspect

    assert "shift" not in inspect.signature(source.bootstrap_rows).parameters


# --- a shifted replay, end to end on hand-made data ---------------------------------------------


def run_to(valkey, raw, path, shift):  # noqa: F811
    data, members_path = raw
    models = tiny_models(batch_table(data, members_path))
    return replay.run(
        valkey, 15, data, CFG, members_path, lambda m: True, models, P.seed, COLUMNS, MAIN,
        path, shift=shift,
    )  # fmt: skip


@pytest.mark.valkey
def test_a_shifted_replay_changes_the_listening_columns_and_nothing_else(valkey, raw, tmp_path):  # noqa: F811
    plain, _ = run_to(valkey, raw, tmp_path / "plain.parquet", shift=False)
    shifted, _ = run_to(valkey, raw, tmp_path / "shifted.parquet", shift=True)
    assert plain["shift"] is None and "listening halved" in shifted["shift"]
    assert plain["produced"] == plain["consumed"] == shifted["produced"] == shifted["consumed"]
    a = move.pq.read_table(tmp_path / "plain.parquet").to_pandas()
    b = move.pq.read_table(tmp_path / "shifted.parquet").to_pandas()
    assert a.msno.tolist() == b.msno.tolist()
    # invalid_secs_days_28 is a listening diagnostic kept in the metadata group, not a model input
    usage = {c.name for c in spec.COLUMNS if c.group == "usage"} | {"invalid_secs_days_28"}
    changed = {c for c in a.columns if not ((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all()}
    assert changed and changed <= usage
    untouched = set(a.columns) - usage
    assert untouched - changed == untouched  # no transaction, state or member column moved
    assert plain["table_sha256"] != shifted["table_sha256"]


@pytest.mark.valkey
def test_the_shifted_store_is_a_store_of_the_same_customers(valkey, raw, tmp_path):  # noqa: F811
    plain, _ = run_to(valkey, raw, tmp_path / "plain.parquet", shift=False)
    shifted, _ = run_to(valkey, raw, tmp_path / "shifted.parquet", shift=True)
    assert plain["store"]["customer_documents"] == shifted["store"]["customer_documents"]
    assert valkey.scard(cs.TOUCHED) == 0


# --- the command ---------------------------------------------------------------------------------


def dargs(**kw):
    return args(**({"db": 1, "shift": False} | kw))


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"db": 0}, "database 0 holds the loaded store"),
        ({"db": 2, "shift": False}, "unshifted drift replay goes to database 1"),
        ({"db": 1, "shift": True}, "shifted drift replay goes to database 2"),
    ],
)
def test_the_command_refuses_database_0_and_the_wrong_database(kw, message):
    with pytest.raises(SystemExit, match=message):
        cli.drift_replay_cmd(dargs(**kw), gather=lambda a: pytest.fail("gathered"))


def test_the_registered_databases_and_names():
    assert cli.DRIFT_DATABASES == {False: 1, True: 2}
    assert cli.drift_report_name(False) == "drift_replay_unshifted.json"
    assert cli.drift_report_name(True) == "drift_replay_shifted.json"
    assert cli.drift_table_path(True).name == "stream_features_2016-11-01_drift_shifted.parquet"
    registered = cli.table_path(args(bucket=None, speed="full"))
    assert cli.drift_table_path(False) != registered  # the parity table is never overwritten


def test_a_database_that_holds_a_key_is_never_overwritten():
    class Db:
        def __init__(self, n):
            self.n = n

        def dbsize(self):
            return self.n

    cli.require_empty_database(Db(0), 1)
    with pytest.raises(SystemExit, match="holds 3 keys; clear it first"):
        cli.require_empty_database(Db(3), 1)


class EmptyDb:
    def dbsize(self):
        return 0


def test_the_command_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.drift_replay_cmd(dargs(), gather=lambda a: pytest.fail("gathered"))


def test_the_command_refuses_a_non_empty_database_before_replaying(quiet, monkeypatch):  # noqa: F811
    class Full(EmptyDb):
        def dbsize(self):
            return 5

    monkeypatch.setattr(replay, "run", lambda *a, **k: pytest.fail("replayed"))
    with pytest.raises(SystemExit, match="clear it first"):
        cli.drift_replay_cmd(dargs(), gather=lambda a: GATHER | {"r": Full()})
    assert not (quiet / "drift_replay_unshifted.json").exists()


@pytest.mark.parametrize(("shift", "db"), [(False, 1), (True, 2)])
def test_the_command_writes_its_report_once_and_checks_the_event_count(
    quiet, monkeypatch, shift, db
):
    seen = {}

    def run(*a, **k):
        seen.update(shift=k["shift"], table=a[10])
        return fake_run(produced=replay.EXPECTED_EVENTS, consumed=replay.EXPECTED_EVENTS)(*a, **k)

    monkeypatch.setattr(replay, "run", run)
    gather = lambda a: GATHER | {"r": EmptyDb()}  # noqa: E731
    assert cli.drift_replay_cmd(dargs(db=db, shift=shift), gather=gather) == 0
    assert seen["shift"] is shift and seen["table"] == cli.drift_table_path(shift)
    report = json.loads((quiet / cli.drift_report_name(shift)).read_text())
    assert report["event_count_check"]["equal"] is True and "not_a_mark" in report
    with pytest.raises(SystemExit, match="runs once"):
        cli.drift_replay_cmd(dargs(db=db, shift=shift), gather=gather)


def test_a_wrong_event_count_is_reported_and_not_clean(quiet, monkeypatch):  # noqa: F811
    monkeypatch.setattr(replay, "run", fake_run(produced=6, consumed=6))
    code = cli.drift_replay_cmd(dargs(), gather=lambda a: GATHER | {"r": EmptyDb()})
    assert code == 1
    assert (
        json.loads((quiet / "drift_replay_unshifted.json").read_text())["event_count_check"][
            "equal"
        ]
        is False
    )


def test_the_parity_report_names_are_not_used_by_the_drift_replays():
    assert cli.drift_report_name(False) != cli.replay_name(args(bucket=None, speed="full"))
    assert parity  # the parity module is imported to show the two paths share no name


# --- drift-clear ---------------------------------------------------------------------------------


class FakeDb:
    def __init__(self, database, keys=7):
        self.database, self.keys, self.flushed = database, keys, False

    def client_info(self):
        return {"db": self.database}

    def dbsize(self):
        return 0 if self.flushed else self.keys

    def flushdb(self):
        self.flushed = True


@pytest.mark.parametrize("db", [1, 2])
def test_drift_clear_flushes_the_drift_database_and_reports_the_count(db, capsys):
    r = FakeDb(db)
    assert cli.drift_clear_cmd(SimpleNamespace(db=db), client=r) == 0
    assert r.flushed
    assert json.loads(capsys.readouterr().out) == {"database": db, "keys_removed": 7, "keys_now": 0}


@pytest.mark.parametrize(
    ("db", "message"),
    [
        (0, "database 0 holds the loaded store"),
        (3, "clears database 1 or 2 only"),
        (15, "1 or 2 only"),
    ],
)
def test_drift_clear_never_touches_database_0_or_another_one(db, message):
    r = FakeDb(db)
    with pytest.raises(SystemExit, match=message):
        cli.drift_clear_cmd(SimpleNamespace(db=db), client=r)
    assert not r.flushed


def test_drift_clear_refuses_a_client_that_is_on_database_0_whatever_was_asked():
    r = FakeDb(0)
    with pytest.raises(PermissionError):
        cli.drift_clear_cmd(SimpleNamespace(db=1), client=r)
    assert not r.flushed
