"""The replay's consumer on hand-made rows and the test database (15); nothing reads real data.

Every expected value is written out by hand, or (for rounding) taken from DuckDB running
Phase 2's own cleaning expression on the same doubles.
"""

from __future__ import annotations

import json
import math
from datetime import date

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.data.stream import USER_LOGS_TYPES
from retention.features.spec import SECS_PER_DAY_MAX, SECS_SENTINEL_ABS
from retention.models import protocol
from retention.stream import consumer as cs
from retention.stream import producer, source
from test_stream_source import TX_TYPES, Untouchable, log, numbered, tx, write

REPLAY_STORE_DATE = date(2016, 10, 1)
MOVE = date(2016, 11, 1)
SCORED = {"s1"}


@pytest.fixture(autouse=True)
def no_test_unlock(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("TestUnlock was constructed")

    monkeypatch.setattr(protocol.TestUnlock, "__init__", refuse)


@pytest.fixture()
def consumer(valkey):
    return cs.Consumer(valkey, 15, REPLAY_STORE_DATE, is_scored=lambda m: m in SCORED)


def logs(*rows):
    return [{"seq": i, "source": "user_logs", **r} for i, r in enumerate(rows, start=1)]


def txs(*rows, name="transactions"):
    """Transaction events with a source_row each (their position, as the reader gives it)."""
    return [
        {"seq": i, "source": name, **r, "source_row": i - 1} for i, r in enumerate(rows, start=1)
    ]


def with_row(row: dict, n: int) -> dict:
    return row | {"source_row": n} if "transaction_date" in row else row


# --- cleaning one row ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secs, expected",
    [
        (-5.0, (0, True)),
        (0.0, (0, False)),
        (1.5, (1500, False)),
        (86400.0, (86_400_000, False)),
        (90000.0, (86_400_000, True)),
        (1e15, (0, True)),
    ],
)
def test_one_row_s_seconds_are_cleaned_as_phase_2_cleans_them(secs, expected):
    assert cs.clean_secs(secs) == expected


@pytest.mark.parametrize(
    "v, expected", [(2.5, 3), (0.5, 1), (1.4999999999999998, 1), (12345.499999999998, 12345)]
)
def test_rounding_sends_halves_away_from_zero_and_nothing_else(v, expected):
    assert cs.round_half_away(v) == expected


def test_milliseconds_equal_duckdb_running_phase_2_s_expression_on_the_same_doubles(tmp_path):
    values = [12.3455, 0.0004999, 86399.9995, 3.0005, 7.25e-4, 100.0, 0.0015, 1234.5675]
    path = tmp_path / "secs.parquet"
    pq.write_table(pa.table({"total_secs": pa.array(values, pa.float64())}), path)
    expr = (
        f"case when total_secs < 0 or abs(total_secs) >= {SECS_SENTINEL_ABS} then 0 "
        f"else round(least(total_secs, {SECS_PER_DAY_MAX}) * 1000) end"
    )
    ref = [int(x) for (x,) in duckdb.sql(f"select ({expr})::bigint from '{path}'").fetchall()]
    assert [cs.clean_secs(v)[0] for v in values] == ref


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None])
def test_a_missing_or_non_finite_float_is_refused(bad):
    with pytest.raises(ValueError, match="refused"):
        cs.clean_secs(bad)


@pytest.mark.valkey
def test_a_nan_in_an_event_is_refused_and_nothing_is_written(consumer, valkey):
    with pytest.raises(ValueError, match="refused"):
        consumer.apply(logs(log("a", 20161002), log("a", 20161003, math.nan)))
    assert valkey.dbsize() == 0


@pytest.mark.valkey
def test_a_missing_count_is_refused(consumer):
    row = log("a", 20161002) | {"num_unq": None}
    with pytest.raises(ValueError, match="num_unq is missing"):
        consumer.apply(logs(row))


# --- the ring ------------------------------------------------------------------------------------


@pytest.mark.valkey
def test_the_ring_at_days_1_28_and_29_before_the_move(consumer):
    consumer.apply(
        logs(log("a", 20161003, 10.0), log("a", 20161004, 20.0), log("a", 20161031, 30.0))
    )
    # Offsets from 2016-11-01: 10-31 is day 1, 10-04 is day 28, 10-03 is day 29 (outside).
    assert consumer.window("a", MOVE) == {
        1: {"ms": 30_000, "x": 0, "p": 3, "f": 2, "u": 3},
        28: {"ms": 20_000, "x": 0, "p": 3, "f": 2, "u": 3},
    }


@pytest.mark.valkey
def test_a_day_s_totals_are_integer_sums_of_its_rows(consumer):
    consumer.apply(
        logs(log("b", 20161010, -5.0), log("b", 20161010, 90000.0), log("b", 20161010, 1.5))
    )
    assert consumer.window("b", MOVE) == {22: {"ms": 86_401_500, "x": 1, "p": 9, "f": 6, "u": 9}}


# --- first and last listening day ----------------------------------------------------------------


@pytest.fixture()
def folder(tmp_path):
    """Bootstrap rows on both sides of 2016-10-01."""
    f = tmp_path / "interim"
    write(
        f,
        "user_logs",
        [log("a", 20160815), log("a", 20160930), log("a", 20161001), log("c", 20160902)],
        USER_LOGS_TYPES,
    )
    write(f, "transactions", [tx("a", 20160801), tx("a", 20161001)], TX_TYPES)
    write(f, "transactions_v2", [tx("a", 20160920, pm=36)], TX_TYPES)
    return f


@pytest.mark.valkey
def test_first_and_last_listening_day_across_the_bootstrap_and_the_replay(consumer, folder):
    consumer.bootstrap(folder)
    assert consumer.first_last("a") == (20160815, 20160930)
    consumer.apply(logs(log("a", 20161002), log("a", 20161020)))
    assert consumer.first_last("a") == (20160815, 20161020)
    assert consumer.first_last("c") == (20160902, 20160902)
    assert consumer.first_last("nobody") == (None, None)


@pytest.mark.valkey
def test_the_bootstrap_never_sees_a_row_dated_2016_10_01_or_later(consumer, folder, monkeypatch):
    seen = []
    real = source.bootstrap_rows

    def spying(f, name, bucket=None):
        for row in real(f, name, bucket):
            seen.append(row.get("date", row.get("transaction_date")))
            yield row

    monkeypatch.setattr(source, "bootstrap_rows", spying)
    counts = consumer.bootstrap(folder)
    assert sorted(seen) == [20160801, 20160815, 20160902, 20160920, 20160930]
    assert counts == {"user_logs": 3, "transactions": 1, "transactions_v2": 1}
    # The ring keeps only days from 28 days before the bootstrap's store date (2016-09-03 on).
    days = sorted({k.decode().split(":")[0] for k in consumer.r.hkeys(cs.ring_key("a"))})
    assert days == ["20160930"]
    assert consumer.transactions("a") == [
        [41, 30, 149, 149, 1, 20160801, 20160831, 0, "v1", 0],
        [36, 30, 149, 149, 1, 20160920, 20160950, 0, "v2", 0],
    ]
    assert consumer.r.scard(cs.TOUCHED) == 0  # the bootstrap marks no one as touched


# --- order within a day --------------------------------------------------------------------------


def canonical(r) -> dict:
    """Every key's value, lists sorted, so two states can be compared whatever the row order."""
    out = {}
    for k in sorted(r.keys("*")):
        t = r.type(k).decode()
        if t == "hash":
            out[k] = dict(sorted(r.hgetall(k).items()))
        elif t == "list":
            out[k] = sorted(r.lrange(k, 0, -1))
        elif t == "set":
            out[k] = sorted(r.smembers(k))
        elif t == "string":
            out[k] = r.get(k)
    return out


DAY = [
    ("user_logs", log("a", 20161012, 12.5)),
    ("transactions", tx("s1", 20161012, cancel=1)),
    ("user_logs", log("a", 20161012, 7.25)),
    ("user_logs", log("b", 20161012, 3.0)),
    ("transactions_v2", tx("s1", 20161012, pm=36)),
    ("user_logs", log("a", 20161012, 90000.0)),
]


@pytest.mark.valkey
def test_the_state_does_not_depend_on_the_order_of_rows_within_a_day(valkey):
    states = []
    for rows in (DAY, list(reversed(DAY)), DAY[3:] + DAY[:3]):
        valkey.flushdb()
        c = cs.Consumer(valkey, 15, REPLAY_STORE_DATE, is_scored=lambda m: m in SCORED)
        c.apply(
            {"seq": i, "source": s, **with_row(row, k)}
            for i, (k, (s, row)) in enumerate(((DAY.index(x), x) for x in rows), start=1)
        )
        states.append(canonical(valkey))
    assert states[0] == states[1] == states[2]
    assert (
        len(states[0]) == 8
    )  # first, last, touched, a's ring, b's ring, s1's rows, events, consumed


# --- cancel events -------------------------------------------------------------------------------


@pytest.mark.valkey
def test_a_cancel_event_is_written_only_for_a_customer_scored_at_the_store_s_date(consumer, valkey):
    consumer.apply(
        txs(
            tx("s1", 20161005, cancel=1), tx("n1", 20161006, cancel=1), tx("s1", 20161007, cancel=0)
        )
    )
    assert [json.loads(e) for e in valkey.lrange("rs:events:cancel", 0, -1)] == [
        {"msno": "s1", "date": "2016-10-05"}
    ]
    assert sorted(valkey.smembers(cs.TOUCHED)) == [b"n1", b"s1"]


# --- dates and databases -------------------------------------------------------------------------


@pytest.mark.valkey
def test_an_older_date_or_one_on_2016_11_01_is_refused(consumer):
    consumer.apply(logs(log("a", 20161010)))
    with pytest.raises(ValueError, match="after one dated 20161010"):
        consumer.apply(logs(log("a", 20161009)))
    with pytest.raises(PermissionError, match="dated 20161101 reached the consumer"):
        consumer.apply(logs(log("a", 20161101)))


@pytest.mark.parametrize("settings_db", [0, 1])
def test_the_consumer_refuses_database_0_before_anything_is_written(settings_db):
    r = Untouchable(db=0, settings_db=settings_db)
    with pytest.raises(PermissionError, match="database 0 holds the loaded store"):
        cs.Consumer(r, 1, REPLAY_STORE_DATE, is_scored=lambda m: True)
    assert r.calls == []


@pytest.mark.valkey
def test_the_consumer_refuses_a_database_other_than_the_one_it_means(valkey):
    with pytest.raises(PermissionError, match="connected to database 15, expected 1"):
        cs.Consumer(valkey, 1, REPLAY_STORE_DATE, is_scored=lambda m: True)


# --- the stream ----------------------------------------------------------------------------------


@pytest.mark.valkey
def test_every_produced_event_is_consumed_once_in_stream_order(consumer, valkey):
    assert producer.produce(valkey, numbered(2_345)) == 2_345
    assert consumer.consume() == 2_345
    assert int(valkey.get("rs:replay:consumed")) == 2_345
    assert consumer.consume() == 0  # nothing new after the cursor
    assert valkey.scard(cs.TOUCHED) == 2_345


def test_a_stored_transaction_row_is_tx_stored_in_order(consumer):
    assert cs.TX_STORED == (
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
        "source",
        "source_row",
    )
    consumer.apply(txs(tx("s1", 20161005), tx("s1", 20161006, pm=36), name="transactions_v2"))
    assert consumer.transactions("s1") == [
        [41, 30, 149, 149, 1, 20161005, 20161035, 0, "v2", 0],
        [36, 30, 149, 149, 1, 20161006, 20161036, 0, "v2", 1],
    ]


# --- the bootstrap measurement (D-223) -----------------------------------------------------------


@pytest.mark.valkey
def test_a_measured_bootstrap_writes_the_same_state_and_splits_its_time(valkey, folder):
    cs.Consumer(valkey, 15, REPLAY_STORE_DATE, is_scored=lambda m: False).bootstrap(folder)
    plain = canonical(valkey)
    valkey.flushdb()
    measure: dict = {}
    c = cs.Consumer(valkey, 15, REPLAY_STORE_DATE, is_scored=lambda m: False)
    counts = c.bootstrap(folder, measure=measure)
    assert canonical(valkey) == plain
    assert counts == {"user_logs": 3, "transactions": 1, "transactions_v2": 1}
    for name, rows in counts.items():
        m = measure[name]
        assert m["rows"] == rows
        assert min(m["read_seconds"], m["write_seconds"]) >= 0
        assert m["peak_rss_gb_after"] >= m["peak_rss_gb_after_first_batch"] > 0
        assert m["duckdb_memory_bytes_max"] >= 0
    assert measure["first_last_writes"]["entries"] == 2  # a and c listened before 2016-10-01
