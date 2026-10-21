"""The replay's source and producer on hand-made files: the stop, the order, no outcome, no db 0.

Every file here is written to a temporary folder with the interim files' column
types; nothing reads the sample's data.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.data.stream import USER_LOGS_TYPES
from retention.models import protocol
from retention.models.data import OUTCOME_LABEL_PREFIXES, PRE_T_LABEL_COLUMNS
from retention.stream import producer, source

TX_TYPES = {c: (pa.string() if c == "msno" else pa.int64()) for c in source.TRANSACTION_COLUMNS}


def tx(msno, day, pm=41, cancel=0):
    return {
        "msno": msno,
        "payment_method_id": pm,
        "payment_plan_days": 30,
        "plan_list_price": 149,
        "actual_amount_paid": 149,
        "is_auto_renew": 1,
        "transaction_date": day,
        "membership_expire_date": day + 30,
        "is_cancel": cancel,
    }


def log(msno, day, secs=100.0):
    return {
        "msno": msno,
        "date": day,
        "num_25": 1,
        "num_50": 0,
        "num_75": 0,
        "num_985": 0,
        "num_100": 2,
        "num_unq": 3,
        "total_secs": secs,
    }


def write(folder, name, rows, types):
    folder.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(c, t) for c, t in types.items()])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / f"{name}.parquet")


@pytest.fixture()
def folder(tmp_path):
    """Rows on both sides of every bound: 2016-09-30, 10-01, 10-31, 11-01, 12-15."""
    f = tmp_path / "interim"
    days = (20160930, 20161001, 20161031, 20161101, 20161215)
    write(f, "transactions", [tx("a", d) for d in days], TX_TYPES)
    write(f, "transactions_v2", [tx("b", d) for d in days], TX_TYPES)
    write(f, "user_logs", [log("c", d) for d in days], USER_LOGS_TYPES)
    return f


@pytest.fixture()
def spy(monkeypatch):
    """Record every SQL statement the reader runs and every row it gets back."""
    seen = SimpleNamespace(sql=[], rows=[])
    real = source.run

    def recording(sql, batch_rows):
        seen.sql.append(sql)
        for rows in real(sql, batch_rows):
            seen.rows += rows
            yield rows

    monkeypatch.setattr(source, "run", recording)
    return seen


@pytest.fixture(autouse=True)
def no_test_unlock(monkeypatch):
    """TestUnlock raises if anything here constructs it."""

    def refuse(*a, **k):
        raise AssertionError("TestUnlock was constructed")

    monkeypatch.setattr(protocol.TestUnlock, "__init__", refuse)


# --- the stop at the source ---------------------------------------------------------------------


def test_no_row_dated_2016_11_01_or_later_is_requested_or_returned(folder, spy):
    got = list(source.events(folder))
    assert {e["transaction_date"] if "transaction_date" in e else e["date"] for e in got} == {
        20161001,
        20161031,
    }
    assert len(spy.sql) == 3
    assert all("< 20161101" in q and "20161102" not in q for q in spy.sql)
    returned = [
        r[source.TRANSACTION_COLUMNS.index("transaction_date")] for r in spy.rows if len(r) == 9
    ]
    assert max(returned) < 20161101


def test_a_read_past_the_store_date_is_refused_before_any_query(folder, spy):
    src = source.sources(folder)[0]
    with pytest.raises(PermissionError, match="no row dated 2016-11-01 or later"):
        list(source.read(src, date(2016, 10, 1), date(2016, 11, 2)))
    assert spy.sql == []


def test_the_bootstrap_reads_only_before_2016_10_01(folder, spy):
    for name in ("transactions", "transactions_v2", "user_logs"):
        rows = list(source.bootstrap_rows(folder, name))
        dates = [r.get("transaction_date", r.get("date")) for r in rows]
        assert dates == [20160930]
    assert len(spy.sql) == 3 and all("< 20161001" in q for q in spy.sql)
    src = source.sources(folder)[2]
    with pytest.raises(PermissionError, match="2016-10-01 or later"):
        list(source.read(src, None, date(2016, 10, 2), ceiling=source.BOOTSTRAP))


def test_a_row_past_the_bound_that_comes_back_anyway_is_refused(folder, monkeypatch):
    def leaky(sql, batch_rows):
        yield [tuple(tx("a", 20161101).values())]

    monkeypatch.setattr(source, "run", leaky)
    with pytest.raises(PermissionError, match="came back"):
        list(source.read(source.sources(folder)[0], date(2016, 10, 1), date(2016, 11, 1)))


# --- no outcome column ---------------------------------------------------------------------------


def test_the_reader_holds_no_outcome_column_and_reads_no_train_file(folder):
    for s in source.sources(folder):
        assert "is_churn" not in s.columns
        assert not any(c.startswith(OUTCOME_LABEL_PREFIXES) for c in s.columns)
        assert not set(s.columns) & set(PRE_T_LABEL_COLUMNS) - {"msno"}
        assert "train" not in s.path.name
    assert [s.name for s in source.sources(folder)] == [
        "transactions",
        "transactions_v2",
        "user_logs",
    ]


def test_the_column_lists_are_the_interim_files_in_file_order():
    from conftest import TX_HEADER

    assert source.TRANSACTION_COLUMNS == tuple(TX_HEADER.split(","))
    assert source.LOG_COLUMNS == tuple(USER_LOGS_TYPES)


# --- the tie rule --------------------------------------------------------------------------------


def test_the_tie_order_is_the_registered_one(tmp_path):
    f = tmp_path / "ties"
    write(
        f,
        "transactions",
        [tx("b", 20161005), tx("a", 20161005, pm=41), tx("a", 20161005, pm=None)],
        TX_TYPES,
    )
    write(f, "transactions_v2", [tx("a", 20161005, pm=36)], TX_TYPES)
    write(
        f,
        "user_logs",
        [log("a", 20161005, 5.0), log("0", 20161005), log("z", 20161004), log("a", 20161005, None)],
        USER_LOGS_TYPES,
    )
    got = [
        (e["seq"], e["source"], e["msno"], e.get("payment_method_id", e.get("total_secs")))
        for e in source.events(f)
    ]
    # Written out by hand: date first; then transactions, transactions_v2, user_logs; then
    # msno; then the other columns in file order with nulls first.
    assert got == [
        (1, "user_logs", "z", 100.0),
        (2, "transactions", "a", None),
        (3, "transactions", "a", 41),
        (4, "transactions", "b", 41),
        (5, "transactions_v2", "a", 36),
        (6, "user_logs", "0", 100.0),
        (7, "user_logs", "a", None),
        (8, "user_logs", "a", 5.0),
    ]


# --- the producer --------------------------------------------------------------------------------


def test_the_registered_throttle_is_1_ms_per_1000_events():
    assert producer.throttle_settings() == (0.001, 1000)


class Untouchable:
    """A client whose server says database `db`; any call but client_info is recorded."""

    def __init__(self, db: int, settings_db: int):
        self.db, self.calls = db, []
        self.connection_pool = SimpleNamespace(connection_kwargs={"db": settings_db})

    def client_info(self):
        return {"db": self.db}

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"{name} called")


@pytest.mark.parametrize("settings_db", [0, 15])
def test_database_0_is_refused_before_anything_is_written(settings_db):
    """The server's answer decides, not the client's own settings (D-211)."""
    r = Untouchable(db=0, settings_db=settings_db)
    with pytest.raises(PermissionError, match="database 0 holds the loaded store"):
        producer.produce(r, iter([{"seq": 1}]))
    assert r.calls == []


def test_a_command_connected_elsewhere_than_it_means_is_refused():
    with pytest.raises(PermissionError, match="connected to database 2, expected 1"):
        producer.require_database(Untouchable(db=2, settings_db=1), 1)
    producer.require_database(Untouchable(db=1, settings_db=0), 1)


def numbered(n):
    return (
        {"seq": i, "source": "user_logs", **log(f"m{i:05d}", 20161001)} for i in range(1, n + 1)
    )


@pytest.mark.valkey
def test_events_reach_the_stream_in_order_and_are_counted(valkey):
    assert producer.produce(valkey, numbered(2_500)) == 2_500
    entries = valkey.xrange(producer.STREAM)
    assert [json.loads(e[b"e"])["seq"] for _, e in entries] == list(range(1, 2_501))
    assert int(valkey.get("rs:replay:produced")) == 2_500


@pytest.mark.valkey
def test_an_event_out_of_order_is_refused(valkey):
    with pytest.raises(ValueError, match="out of order"):
        producer.produce(valkey, iter([{"seq": 1}, {"seq": 3}]))


@pytest.mark.valkey
@pytest.mark.parametrize("throttled, pauses", [(False, []), (True, [0.001, 0.001])])
def test_only_the_throttled_replay_pauses_once_per_1000_events(valkey, throttled, pauses):
    slept = []
    producer.produce(valkey, numbered(2_500), throttled=throttled, sleep=slept.append)
    assert slept == pauses
