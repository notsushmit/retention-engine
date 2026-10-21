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
from retention.stream import move, parity, replay, source
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
    steps = {
        "clear", "bootstrap", "bootstrap_timelines", "produce", "consume", "move",
        "move_write_table", "store",
    }  # fmt: skip
    assert set(report["step_seconds"]) == set(report["step_peak_rss_gb"]) == steps
    assert set(report["step_rss_gb"]) == steps  # current memory after each step (D-227)
    assert all(0 < v <= report["step_peak_rss_gb"][k] for k, v in report["step_rss_gb"].items())
    assert 0 < report["process_rss_gb_start"] <= report["process_peak_rss_gb_start"]
    assert "seconds" not in report  # stamp's key, written only by the command
    peaks = list(report["step_peak_rss_gb"].values())
    assert peaks == sorted(peaks) and peaks[0] >= report["process_peak_rss_gb_start"]
    assert not (tmp_path / "stream_work").exists()  # the chunk work folder is removed
    detail = report["bootstrap_detail"]
    assert set(detail) == {"user_logs", "transactions", "transactions_v2", "first_last_writes"}
    assert {k: detail[k]["rows"] for k in report["bootstrap_rows"]} == report["bootstrap_rows"]
    for k, d in detail.items():
        assert 0 < d["rss_gb_after"] <= d["peak_rss_gb_after"]
        if k != "first_last_writes":
            assert 0 < d["rss_gb_after_first_batch"] <= d["peak_rss_gb_after_first_batch"]


STEPS = [
    "clear", "bootstrap", "bootstrap_timelines", "produce", "consume", "move",
    "move_write_table", "store",
]  # fmt: skip


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.mark.valkey
def test_every_step_appends_a_progress_line_and_runs_never_truncate(valkey, raw, tmp_path):  # noqa: F811
    progress = tmp_path / "replay_full_progress.jsonl"
    first, _ = one(valkey, raw, tmp_path, progress=progress)
    got = lines(progress)
    assert [x["step"] for x in got] == ["start", *STEPS]
    assert {x["run_started"] for x in got} == {got[0]["run_started"]}
    for x in got[1:]:
        assert x["seconds"] == first["step_seconds"][x["step"]]
        assert x["peak_rss_gb"] == first["step_peak_rss_gb"][x["step"]]
        assert x["rss_gb"] == first["step_rss_gb"][x["step"]] and x["database"] == 15
    one(valkey, raw, tmp_path, progress=progress)
    again = lines(progress)
    assert again[: len(got)] == got and len(again) == 2 * len(got)
    assert again[len(got)]["run_started"] != got[0]["run_started"]


@pytest.mark.valkey
def test_a_run_stopped_in_the_move_leaves_the_steps_before_it(valkey, raw, tmp_path, monkeypatch):  # noqa: F811
    def stopped(*a, **k):
        raise KeyboardInterrupt("stopped in the move")

    monkeypatch.setattr(move, "chunked_move", stopped)
    progress = tmp_path / "replay_throttled_progress.jsonl"
    with pytest.raises(KeyboardInterrupt):
        one(valkey, raw, tmp_path, throttled=True, progress=progress)
    got = lines(progress)
    assert [x["step"] for x in got] == ["start", *STEPS[:5]]
    assert all(x["peak_rss_gb"] >= x["rss_gb"] > 0 for x in got)


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
    report, result = one(valkey, raw, tmp_path, bucket=k, keep_link=True)
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    assert report["bucket"] == k and report["feature_rows"] == n
    assert {bucket(m) for m in stream.column("msno").to_pylist()} == {k}
    clean_path = tmp_path / "cleaned.parquet"
    cleaned(data).to_parquet(clean_path)
    link = replay.link_check(clean_path, result.link_rows, T, k)
    assert link["met"] and link["batch_rows"] == link["stream_rows"] == link["equal_rows"] > 0
    shifted = cleaned(data)
    i = shifted.index[shifted.msno.map(bucket).eq(k) & (shifted.transaction_date < str(T))][0]
    shifted.loc[i, "source_row"] += 1_000_000
    shifted.to_parquet(clean_path)
    bad = replay.link_check(clean_path, result.link_rows, T, k)
    assert not bad["met"] and (bad["only_in_batch"], bad["only_in_stream"]) == (1, 1)


@pytest.mark.valkey
def test_a_two_bucket_replay_holds_both_buckets_only_and_links_to_phase_1(valkey, raw, tmp_path):  # noqa: F811
    data, members_path = raw
    population = batch_table(data, members_path).column("msno").to_pylist()
    (k, n), (j, m) = Counter(bucket(x) for x in population).most_common(2)
    report, result = one(valkey, raw, tmp_path, bucket=(k, j), keep_link=True)
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    assert report["bucket"] == [k, j] and report["feature_rows"] == n + m
    assert {bucket(x) for x in stream.column("msno").to_pylist()} == {k, j}
    clean_path = tmp_path / "cleaned.parquet"
    cleaned(data).to_parquet(clean_path)
    link = replay.link_check(clean_path, result.link_rows, T, (k, j))
    assert link["met"] and link["batch_rows"] == link["stream_rows"] == link["equal_rows"] > 0
    single = replay.link_check(clean_path, result.link_rows, T, k)
    assert not single["met"] and single["only_in_batch"] == 0 < single["only_in_stream"]


def test_the_bucket_filter_takes_one_bucket_or_several():
    assert source.bucket_where(3).endswith(" = 3")
    assert source.bucket_where((3,)).endswith(" = 3")
    assert source.bucket_where((2, 3)).endswith(" in (2, 3)")


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
        ({"bucket": [1, 2, 3]}, "one bucket or two different buckets"),
        ({"bucket": [2, 2]}, "one bucket or two different buckets"),
        ({"rerun_reason": "again"}, "only a dry run is rerun"),
        ({"bucket": [2], "rerun_reason": "  "}, "needs a reason"),
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
        report |= {
            "step_seconds": {"bootstrap": 1.5},
            "step_peak_rss_gb": {"bootstrap": 0.7},
            "bootstrap_detail": {"user_logs": {"read_seconds": 1.0, "peak_rss_gb_after": 0.7}},
        }
        return report, SimpleNamespace(link_rows=None)

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
    # the written report keeps the per-step keys next to stamp's own total
    assert report["step_seconds"] == {"bootstrap": 1.5}
    assert report["step_peak_rss_gb"] == {"bootstrap": 0.7}
    assert report["bootstrap_detail"] == {
        "user_logs": {"read_seconds": 1.0, "peak_rss_gb_after": 0.7}
    }
    assert isinstance(report["seconds"], float)


def test_a_dry_run_with_a_broken_link_is_not_clean(quiet, monkeypatch):
    monkeypatch.setattr(replay, "run", fake_run())
    monkeypatch.setattr(replay, "link_check", lambda *a: {"met": False})
    assert cli.replay_cmd(args(bucket=3), gather=lambda a: GATHER) == 1


def test_the_command_writes_progress_next_to_its_report(quiet, monkeypatch):
    seen = []

    def run(*a, **k):
        seen.append(k["progress"])
        return fake_run()(*a, **k)

    monkeypatch.setattr(replay, "run", run)
    monkeypatch.setattr(replay, "link_check", lambda *a: {"met": True})
    cli.replay_cmd(args(bucket=3), gather=lambda a: GATHER)
    assert seen == [quiet / "replay_dry_bucket3_progress.jsonl"]


def test_a_two_bucket_dry_run_projects_by_a_factor_of_10(quiet, monkeypatch):
    monkeypatch.setattr(replay, "run", fake_run())
    seen = []
    monkeypatch.setattr(replay, "link_check", lambda *a: seen.append(a[3]) or {"met": True})
    assert cli.replay_cmd(args(bucket=[2, 3]), gather=lambda a: GATHER) == 0
    report = json.loads((quiet / "replay_dry_buckets2_3.json").read_text())
    assert report["not_a_mark"] == "dry run on 2 md5 buckets"
    assert report["projection"]["factor"] == 10 and seen == [(2, 3)]
    assert report["projection"]["process_peak_gb_limit"] == 10  # the registered limit, unchanged
    one_bucket = replay.project(dry(0.9, 101_000_000), factor=20)
    assert one_bucket["factor"] == 20


def test_bucket_flags_append_and_one_bucket_reads_as_before(monkeypatch):
    got = []
    monkeypatch.setattr(cli, "replay_cmd", lambda a: got.append(a) or 0)
    cli.main(["replay", "--db", "1", "--speed", "full", "--bucket", "2", "--bucket", "3"])
    cli.main(["replay", "--db", "1", "--speed", "full", "--bucket", "2", "--rerun-reason", "x"])
    assert cli.bucket_arg(got[0]) == (2, 3) and got[0].rerun_reason is None
    assert cli.bucket_arg(got[1]) == 2 and got[1].rerun_reason == "x"
    assert cli.replay_name(got[0]) == "replay_dry_buckets2_3.json"
    assert cli.table_path(got[0]).name == "stream_features_2016-11-01_buckets2_3.parquet"


@pytest.fixture()
def dry_outputs(quiet, monkeypatch, tmp_path):
    """An earlier bucket-3 dry run's report and table, the table kept in a temporary folder."""
    table = tmp_path / "stream_features_2016-11-01_bucket3.parquet"
    monkeypatch.setattr(cli, "table_path", lambda a: table)
    monkeypatch.setattr(replay, "run", fake_run())
    monkeypatch.setattr(replay, "link_check", lambda *a: {"met": True})
    (quiet / "replay_dry_bucket3.json").write_text(json.dumps({"git_head": "0da8916"}))
    table.write_bytes(b"earlier table")
    return quiet, table


def test_a_dry_run_is_rerun_only_with_a_reason_and_keeps_the_earlier_outputs(dry_outputs):
    out, table = dry_outputs
    with pytest.raises(SystemExit, match="runs once"):
        cli.replay_cmd(args(bucket=3), gather=lambda a: GATHER)
    reason = "measured after a fix (D-225)"
    assert cli.replay_cmd(args(bucket=3, rerun_reason=reason), gather=lambda a: GATHER) == 0
    assert json.loads((out / "replay_dry_bucket3_0da8916.json").read_text()) == {
        "git_head": "0da8916"
    }
    assert (table.parent / "stream_features_2016-11-01_bucket3_0da8916.parquet").read_bytes() == (
        b"earlier table"
    )
    report = json.loads((out / "replay_dry_bucket3.json").read_text())
    assert report["rerun"] == {
        "reason": reason,
        "earlier_git_head": "0da8916",
        "earlier_report": "replay_dry_bucket3_0da8916.json",
        "earlier_table": "stream_features_2016-11-01_bucket3_0da8916.parquet",
    }


def test_a_rerun_never_overwrites_and_needs_an_earlier_report(dry_outputs):
    out, table = dry_outputs
    (out / "replay_dry_bucket3_0da8916.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists; nothing renamed"):
        cli.replay_cmd(args(bucket=3, rerun_reason="r"), gather=lambda a: pytest.fail("gathered"))
    assert (out / "replay_dry_bucket3.json").exists() and table.exists()
    with pytest.raises(SystemExit, match="does not exist"):
        cli.replay_cmd(args(bucket=4, rerun_reason="r"), gather=lambda a: pytest.fail("gathered"))


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
    one(valkey, raw, tmp_path)
    stream = move.pq.read_table(tmp_path / "stream.parquet")
    batch = planted(batch_table(data, members_path), "s-tie", "segment", "discount100_only")
    full = move.rebuild(valkey, CFG, None).coverage  # the stored rows outlive the move
    d = parity.diagnostics(
        stream, batch, lambda ids: full[full.msno.isin(ids)], lambda ids: full[full.msno.isin(ids)],
        data / "user_logs.parquet", T,
    )  # fmt: skip
    assert d["with_same_day_ties"] == 1
    assert d["tied_with_labels_state_mismatch"] == 1
    assert d["tied_without_labels_state_mismatch"] == 0
    assert d["labels_state_mismatch_without_ties"] == 0
    assert d["classes"] == {"ordering_cause": 1, "phase2_finding": 0, "stream_difference": 0}
