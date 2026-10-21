"""The replay commands on hand-made data: recount, cancel-ids, one replay, the dry run's checks.

Nothing reads the sample's data: the files are hand-made (test_stream_source, test_stream_move)
and the Valkey test database (15) stands in for the replay databases.
"""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import duckdb
import pytest

from retention.api import __main__ as api_cli
from retention.api import store
from retention.data.clean import _build_transactions, _load_transactions
from retention.data.sample import bucket
from retention.data.stream import USER_LOGS_TYPES
from retention.stream import __main__ as cli
from retention.stream import consumer as cs
from retention.stream import move, parity, replay
from test_stream_move import CFG, T, batch_table, no_test_unlock, raw  # noqa: F401
from test_stream_parity import COLUMNS, MAIN, P, planted, tiny_models
from test_stream_source import TX_TYPES, folder, log, tx, write  # noqa: F401

# --- the recount ---------------------------------------------------------------------------------


def test_the_recount_counts_october_rows_with_the_reader_s_bounds(folder):  # noqa: F811
    """The boundary fixture holds rows on 2016-09-30, 10-01, 10-31, 11-01 and 12-15 per file."""
    r = replay.recount(folder)
    assert r["counts"] == {"transactions": 2, "transactions_v2": 2, "user_logs": 2}
    assert (r["total"], r["expected"], r["met"]) == (6, 3_727_472, False)
    assert replay.recount(folder, expected=6)["met"] is True


# --- the cancel-ID count -------------------------------------------------------------------------


def test_cancel_ids_counts_only(tmp_path):
    f = tmp_path / "cancels"
    write(
        f,
        "transactions",
        [
            tx("x/age+01", 20161005, cancel=1),
            tx("c1", 20161006, cancel=1),
            tx("c1", 20161007, cancel=1),
            tx("n/city+02", 20161008, cancel=1),  # not scored
            tx("x/age+01", 20160920, cancel=1),  # September
            tx("c2", 20161010),  # not a cancel
        ],
        TX_TYPES,
    )
    write(f, "transactions_v2", [tx("c3", 20161015, cancel=1)], TX_TYPES)
    write(f, "user_logs", [log("c1", 20161005)], USER_LOGS_TYPES)
    report = replay.cancel_ids(f, scored={"x/age+01", "c1", "c2", "c3"})
    assert report == {
        "october_cancel_rows": 5,
        "distinct_ids_scored_at_2016_10_01": 3,
        "ids_with_a_whole_word_hit": 1,
        "by_word": {"age": 1},
    }
    text = json.dumps(report)
    assert not any(m in text for m in ("x/age+01", "c1", "c2", "c3", "n/city+02"))


# --- one replay ----------------------------------------------------------------------------------


def cleaned(data):
    con = duckdb.connect()
    _load_transactions(con, data / "transactions.parquet", data / "transactions_v2.parquet")
    _build_transactions(con, CFG)
    return con.sql("select * from tx order by msno, transaction_date, source, source_row").df()


def one(valkey, raw, tmp_path, **kw):  # noqa: F811
    data, members_path = raw
    models = tiny_models(batch_table(data, members_path))
    return replay.run(
        valkey, 15, data, CFG, members_path, lambda m: True, models, P.seed, COLUMNS, MAIN,
        tmp_path / "stream.parquet", **kw,
    )  # fmt: skip


@pytest.mark.valkey
def test_a_full_replay_equals_batch_and_counts_every_october_event(valkey, raw, tmp_path):  # noqa: F811
    data, members_path = raw
    report, _ = one(valkey, raw, tmp_path)
    assert report["produced"] == report["consumed"] == replay.recount(data)["total"] > 100
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    assert parity.compare_tables(stream, batch_table(data, members_path))["met"]
    assert store.store_date(valkey) == T and valkey.scard(cs.TOUCHED) == 0
    assert report["feature_rows"] == stream.num_rows and report["feature_columns"] == 61
    assert report["store"]["customer_documents"] == stream.num_rows
    assert set(report["seconds"]) == {
        "clear", "bootstrap", "bootstrap_timelines", "produce", "consume",
        "move_timelines", "move_features", "move_write_table", "store",
    }  # fmt: skip


@pytest.mark.valkey
def test_a_throttled_replay_gives_the_same_table_hash(valkey, raw, tmp_path):  # noqa: F811
    full, _ = one(valkey, raw, tmp_path)
    throttled, _ = one(valkey, raw, tmp_path, throttled=True)
    assert throttled["removed_keys_before"] > 0  # the second replay cleared the first one's keys
    assert full["table_sha256"] == throttled["table_sha256"]
    assert full["produced"] == throttled["produced"]


@pytest.mark.valkey
def test_a_replay_refuses_database_0_and_a_database_other_than_the_one_meant(valkey, raw, tmp_path):  # noqa: F811
    from test_stream_source import Untouchable

    data, members_path = raw
    with pytest.raises(PermissionError, match="database 0 holds the loaded store"):
        replay.run(
            Untouchable(db=0, settings_db=1), 1, data, CFG, members_path, bool, {}, 0, [], (),
            tmp_path / "x.parquet",
        )  # fmt: skip
    with pytest.raises(PermissionError, match="connected to database 15, expected 1"):
        replay.run(valkey, 1, data, CFG, members_path, bool, {}, 0, [], (), tmp_path / "x.parquet")


# --- the dry run ---------------------------------------------------------------------------------


@pytest.mark.valkey
def test_a_bucket_replay_holds_that_bucket_only_and_links_to_phase_1(valkey, raw, tmp_path):  # noqa: F811
    data, members_path = raw
    population = batch_table(data, members_path).column("msno").to_pylist()
    k, n = Counter(bucket(m) for m in population).most_common(1)[0]
    report, tl = one(valkey, raw, tmp_path, bucket=k)
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    assert report["bucket"] == k and report["feature_rows"] == n
    assert {bucket(m) for m in stream.column("msno").to_pylist()} == {k}
    clean_path = tmp_path / "cleaned.parquet"
    cleaned(data).to_parquet(clean_path)
    link = replay.link_check(clean_path, tl.tx, T, k)
    assert link["met"] and link["batch_rows"] == link["stream_rows"] == link["equal_rows"] > 0
    shifted = cleaned(data)
    i = shifted.index[shifted.msno.map(bucket).eq(k) & (shifted.transaction_date < str(T))][0]
    shifted.loc[i, "source_row"] += 1_000_000
    shifted.to_parquet(clean_path)
    bad = replay.link_check(clean_path, tl.tx, T, k)
    assert not bad["met"] and (bad["only_in_batch"], bad["only_in_stream"]) == (1, 1)


def dry(peak: float, valkey_after: int) -> dict:
    return {
        "process_peak_rss_gb_start": 0.5,
        "process_peak_rss_gb": peak,
        "valkey_used_memory_bytes_before": 1_000_000,
        "valkey_used_memory_bytes_after": valkey_after,
    }


def test_the_registered_limits_and_the_user_s_rule():
    assert replay.PEAK_GB_MAX == 10
    assert replay.VALKEY_LIMIT == 7_598_239_392
    assert replay.VALKEY_ASK == 6_838_415_452


@pytest.mark.parametrize(
    "peak, valkey_after, verdict",
    [
        (0.9, 101_000_000, "proceed"),  # 8.5 GB; 2 x 20 x 100 MB = 4.0 GB of Valkey
        (0.97, 171_960_385, "proceed"),  # 9.9 GB; Valkey 6,838,415,400 (just under 90%)
        (0.9, 171_960_387, "stop_and_ask"),  # Valkey 6,838,415,480 (just over 90%)
        (0.9, 190_955_985, "stop"),  # Valkey 7,598,239,400 (just over the limit)
        (0.98, 101_000_000, "stop"),  # 10.1 GB process peak
    ],
)
def test_the_projection_s_verdict_follows_the_stop_rules(peak, valkey_after, verdict):
    p = replay.project(dry(peak, valkey_after), factor=20)
    assert p["verdict"] == verdict


# --- the commands --------------------------------------------------------------------------------


def args(**kw):
    return SimpleNamespace(**({"fraction": None, "db": 1, "speed": "full", "bucket": None} | kw))


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"db": 0}, "database 0 holds the loaded store"),
        ({"db": 2}, "goes to database 1"),
        ({"db": 1, "speed": "throttled"}, "goes to database 2"),
        ({"db": 2, "speed": "throttled", "bucket": 3}, "one bucket at full speed"),
    ],
)
def test_the_replay_command_refuses_the_wrong_database_or_mode(kw, message):
    with pytest.raises(SystemExit, match=message):
        cli.replay_cmd(args(**kw), gather=lambda a: pytest.fail("gathered"))


@pytest.fixture()
def quiet(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(api_cli, "head_commit", lambda repo: "abc1234")
    monkeypatch.setattr(api_cli, "out_dir", lambda fraction: out)
    return out


def test_every_command_refuses_a_dirty_tree(monkeypatch, folder):  # noqa: F811
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: True)
    for call in (
        lambda: cli.recount_cmd(args(), folder=folder),
        lambda: cli.cancel_ids_cmd(args(), folder=folder, scored=set()),
        lambda: cli.replay_cmd(args(), gather=lambda a: pytest.fail("gathered")),
    ):
        with pytest.raises(SystemExit, match="commit first"):
            call()


def test_recount_and_cancel_ids_write_only_their_report_once(quiet, folder):  # noqa: F811
    assert cli.recount_cmd(args(), folder=folder) == 1  # 6 is not the registered 3,727,472
    assert cli.cancel_ids_cmd(args(), folder=folder, scored={"a", "b"}) == 0
    assert sorted(p.name for p in quiet.iterdir()) == [
        "cancel_ids_2016-10.json",
        "recount_2016-10.json",
    ]
    for call in (
        lambda: cli.recount_cmd(args(), folder=folder),
        lambda: cli.cancel_ids_cmd(args(), folder=folder, scored=set()),
    ):
        with pytest.raises(SystemExit, match="runs once"):
            call()


def fake_run(table_sha="a" * 64, produced=6, consumed=6, peak=0.9, valkey=101_000_000):
    def run(*a, **k):
        report = {
            "table_sha256": table_sha,
            "produced": produced,
            "consumed": consumed,
            **dry(peak, valkey),
        }
        return report, SimpleNamespace(tx=None)

    return run


GATHER = {
    "r": None, "folder": None, "cfg": None, "members": None, "cleaned": None, "is_scored": None,
    "c01": None, "models_version": 1, "seed": 0, "columns": [], "main": (), "sample_buckets": 20,
}  # fmt: skip


def test_the_full_replay_judges_the_event_count_against_the_recount(quiet, monkeypatch):
    with pytest.raises(SystemExit, match="recount runs before"):
        monkeypatch.setattr(replay, "run", fake_run())
        cli.replay_cmd(args(), gather=lambda a: GATHER)
    (quiet / "replay_full.json").unlink(missing_ok=True)
    (quiet / "recount_2016-10.json").write_text(json.dumps({"total": 3_727_472}))
    monkeypatch.setattr(replay, "run", fake_run(produced=3_727_472, consumed=3_727_472))
    assert cli.replay_cmd(args(), gather=lambda a: GATHER) == 0
    event = json.loads((quiet / "replay_full.json").read_text())["event_count"]
    assert event == {
        "recount": 3_727_472,
        "registered": 3_727_472,
        "produced": 3_727_472,
        "consumed": 3_727_472,
        "met": True,
    }
    with pytest.raises(SystemExit, match="runs once"):
        cli.replay_cmd(args(), gather=lambda a: GATHER)


def test_the_throttled_replay_runs_after_the_full_one_and_judges_determinism(quiet, monkeypatch):
    t = args(db=2, speed="throttled")
    with pytest.raises(SystemExit, match="runs after the full-speed one"):
        cli.replay_cmd(t, gather=lambda a: GATHER)
    (quiet / "replay_full.json").write_text(json.dumps({"table_sha256": "a" * 64}))
    monkeypatch.setattr(replay, "run", fake_run(table_sha="b" * 64))
    assert cli.replay_cmd(t, gather=lambda a: GATHER) == 1
    d = json.loads((quiet / "replay_throttled.json").read_text())["determinism"]
    assert d == {"full_speed_sha256": "a" * 64, "throttled_sha256": "b" * 64, "met": False}


@pytest.mark.parametrize("valkey, code", [(101_000_000, 0), (171_960_387, 2), (190_955_985, 1)])
def test_the_dry_run_returns_the_projection_s_verdict(quiet, monkeypatch, valkey, code):
    monkeypatch.setattr(replay, "run", fake_run(valkey=valkey))
    monkeypatch.setattr(replay, "link_check", lambda *a: {"met": True})
    assert cli.replay_cmd(args(bucket=3), gather=lambda a: GATHER) == code
    report = json.loads((quiet / "replay_dry_bucket3.json").read_text())
    assert report["not_a_mark"] == "dry run on one md5 bucket"
    assert report["projection"]["factor"] == 20


def test_a_dry_run_with_a_broken_link_is_not_clean(quiet, monkeypatch):
    monkeypatch.setattr(replay, "run", fake_run())
    monkeypatch.setattr(replay, "link_check", lambda *a: {"met": False})
    assert cli.replay_cmd(args(bucket=3), gather=lambda a: GATHER) == 1


def test_table_paths_follow_the_registered_one():
    assert str(cli.table_path(args())).endswith("serving/stream_features_2016-11-01.parquet")
    assert cli.table_path(args(db=2, speed="throttled")).name == (
        "stream_features_2016-11-01_throttled.parquet"
    )
    assert cli.table_path(args(bucket=7)).name == "stream_features_2016-11-01_bucket7.parquet"


# --- the parity report: ties and labels state counted apart --------------------------------------


@pytest.mark.valkey
def test_tied_customers_with_a_labels_state_mismatch_are_counted_apart(raw, valkey, tmp_path):  # noqa: F811
    data, members_path = raw
    report, tl = one(valkey, raw, tmp_path)
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    batch = planted(batch_table(data, members_path), "s-tie", "segment", "discount100_only")
    full = tl.coverage
    d = parity.diagnostics(
        stream, batch, lambda ids: full[full.msno.isin(ids)], lambda ids: full[full.msno.isin(ids)],
        data / "user_logs.parquet", T,
    )  # fmt: skip
    assert d["with_same_day_ties"] == 1
    assert d["tied_with_labels_state_mismatch"] == 1
    assert d["tied_without_labels_state_mismatch"] == 0
    assert d["labels_state_mismatch_without_ties"] == 0
    assert d["classes"] == {"ordering_cause": 1, "phase2_finding": 0, "stream_difference": 0}
