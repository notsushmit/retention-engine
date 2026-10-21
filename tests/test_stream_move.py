"""The move on hand-made raw tables: the stream's 2016-11-01 table equals batch's (A9's 61 columns).

Batch runs Phase 1's cleaning, the timelines, build_labels, UsageSource,
TransactionSource and assemble on every row, including rows dated after 2016-11-01.
The stream runs the consumer (bootstrap, then October's events through the source)
and the move on rows dated before 2016-11-01 only. Every file is hand-made in a
temporary folder; the Valkey test database (15) stands in for the replay database.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from retention.data.clean import _build_transactions, _load_transactions
from retention.data.stream import USER_LOGS_TYPES
from retention.features.build import HORIZONS, LABEL_INPUTS, assemble, to_arrow
from retention.features.transactions import TransactionSource
from retention.features.usage import UsageSource
from retention.labels.config import parse_label_config
from retention.labels.snapshots import build_labels
from retention.labels.timeline import build_timeline
from retention.models import protocol
from retention.stream import consumer as cs
from retention.stream import move, parity, source
from test_stream_source import TX_TYPES, Untouchable

T = date(2016, 11, 1)
RAW = yaml.safe_load(Path("configs/kkbox.yaml").read_text())["labels"]
CFG = parse_label_config({**RAW, "snapshots": {"dates": [T.isoformat()]}})
MEMBER_TYPES = {
    "msno": pa.string(),
    "city": pa.int16(),
    "bd": pa.int32(),
    "gender": pa.string(),
    "registered_via": pa.int16(),
    "registration_init_time": pa.date32(),
}


@pytest.fixture(autouse=True)
def no_test_unlock(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("TestUnlock was constructed")

    monkeypatch.setattr(protocol.TestUnlock, "__init__", refuse)


# --- hand-made raw tables ------------------------------------------------------------------------


def day(d: date) -> int:
    return d.year * 10_000 + d.month * 100 + d.day


def tx_row(m, d, expire, pm=36, plan=30, price=149, paid=149, auto=1, cancel=0):
    return {
        "msno": m,
        "payment_method_id": pm,
        "payment_plan_days": plan,
        "plan_list_price": price,
        "actual_amount_paid": paid,
        "is_auto_renew": auto,
        "transaction_date": day(d),
        "membership_expire_date": expire,
        "is_cancel": cancel,
    }


def raw_tables(seed: int, n: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Subscriptions with renewals, lapses and returns, cancels (cutting and same-day),
    placeholder expiries, trials, 100% discounts, identical twins across v1 and v2, rows
    after 2016-11-01; listening with several rows a day, invalid seconds, newcomers and
    customers with no listening; members with gaps, odd ages and late registrations."""
    rng = np.random.default_rng(seed)
    v1, v2, logs, members = [], [], [], []
    for i in range(n):
        m = f"m{i:03d}"
        d = date(2016, 1, 5) + timedelta(days=int(rng.integers(0, 290)))
        pm = int(rng.choice([41, 36, 38, 39]))
        while d <= date(2016, 12, 20):
            kind = rng.random()
            if kind < 0.06:  # a trial
                row = tx_row(m, d, day(d + timedelta(days=7)), pm=pm, plan=7, price=0, paid=0)
                length = 7
            elif kind < 0.1:  # a 100% discount
                row = tx_row(m, d, day(d + timedelta(days=30)), pm=pm, paid=0)
                length = 30
            else:
                placeholder = pm == 41 and rng.random() < 0.3
                expire = 19700101 if placeholder else day(d + timedelta(days=30))
                row = tx_row(m, d, expire, pm=pm, auto=int(rng.random() < 0.7))
                length = 31 if placeholder else 30
            target = v2 if d >= date(2016, 9, 1) and rng.random() < 0.4 else v1
            target.append(row)
            if rng.random() < 0.05:  # an identical twin in the other file
                (v1 if target is v2 else v2).append(dict(row))
            if rng.random() < 0.05 and row["is_cancel"] == 0:  # a same-day same-plan cancel
                v1.append(row | {"is_cancel": 1, "membership_expire_date": row["transaction_date"]})
            expiry = d + timedelta(days=length)
            if rng.random() < 0.08:  # a cancel that cuts coverage
                c = d + timedelta(days=int(rng.integers(3, 20)))
                v1.append(tx_row(m, c, day(c), pm=pm, cancel=1))
                expiry = c
            if rng.random() < 0.15:  # a lapse, then a return
                d = expiry + timedelta(days=int(rng.integers(31, 70)))
            else:
                d = expiry - timedelta(days=int(rng.integers(0, 3)))
        if rng.random() < 0.85:
            first = date(2016, 7, 1) + timedelta(days=int(rng.integers(0, 140)))
            for _day in range(int(rng.integers(0, 60))):
                ld = first + timedelta(days=int(rng.integers(0, 120)))
                for _ in range(int(rng.integers(1, 3))):
                    secs = float(rng.choice([rng.uniform(10, 9000), -1.0, 90000.0, 1e15, 0.0005]))
                    logs.append(
                        {
                            "msno": m,
                            "date": day(ld),
                            "num_25": int(rng.integers(0, 5)),
                            "num_50": int(rng.integers(0, 3)),
                            "num_75": int(rng.integers(0, 3)),
                            "num_985": int(rng.integers(0, 2)),
                            "num_100": int(rng.integers(1, 20)),
                            "num_unq": int(rng.integers(1, 15)),
                            "total_secs": secs,
                        }
                    )
        if rng.random() < 0.85:
            reg = date(2013, 1, 1) + timedelta(days=int(rng.integers(0, 1500)))
            members.append(
                {
                    "msno": m,
                    "city": int(rng.choice([1, 4, 13])),
                    "bd": int(rng.choice([0, 25, 34, 150])),
                    "gender": rng.choice(["male", "female", None]),
                    "registered_via": int(rng.choice([3, 7, 9])),
                    "registration_init_time": reg,
                }
            )
    v1 += special_customers()
    return v1, v2, logs, members


def special_customers() -> list[dict]:
    """Customers whose state the random ones rarely reach, written out by hand."""
    return [
        # trial only: segment trial_only, never paid
        tx_row("s-trial", date(2016, 10, 28), 20161104, plan=7, price=0, paid=0),
        # 100% discount only: discount100_only, never paid
        tx_row("s-disc", date(2016, 10, 20), 20161119, paid=0),
        # zero price, 30 days: promo_only, never paid
        tx_row("s-promo", date(2016, 10, 15), 20161114, price=0, paid=0),
        # a 90-day paid plan: expiry more than 30 days after T (not in_exp30)
        tx_row("s-long", date(2016, 9, 15), 20161214, plan=90, price=447, paid=447),
        # paid until 2016-10-20, then a trial: coverage by the trial, paid state grace
        tx_row("s-grace", date(2016, 9, 20), 20161020),
        tx_row("s-grace", date(2016, 10, 28), 20161104, plan=7, price=0, paid=0),
        # two same-day rows tied on every timeline key: placeholder expiries whose raw dates
        # differ (so cleaning keeps both) and impute to the same expiry (2016-11-20)
        tx_row("s-tie", date(2016, 10, 20), 19700101, pm=41),
        tx_row("s-tie", date(2016, 10, 20), 19700102, pm=41),
        # a row after T for s-long, which batch reads and the stream never does
        tx_row("s-long", date(2016, 12, 10), 20170310, plan=90, price=447, paid=447),
    ]


def write(folder: Path, name: str, rows: list[dict], types: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(c, t) for c, t in types.items()])
    path = folder / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


@pytest.fixture(scope="module", params=[2016, 31])
def raw(tmp_path_factory, request):
    """Two seeds of hand-made raw tables, each in its own folder."""
    folder = tmp_path_factory.mktemp(f"interim{request.param}")
    v1, v2, logs, members = raw_tables(seed=request.param, n=160)
    write(folder, "transactions", v1, TX_TYPES)
    write(folder, "transactions_v2", v2, TX_TYPES)
    write(folder, "user_logs", logs, USER_LOGS_TYPES)
    members_path = write(folder, "members", members, MEMBER_TYPES)
    return folder, members_path


# --- batch: Phase 1 and Phase 2's own functions on every row -------------------------------------


def batch_table(folder: Path, members_path: Path, plant=None) -> pa.Table:
    con = duckdb.connect()
    _load_transactions(con, folder / "transactions.parquet", folder / "transactions_v2.parquet")
    _build_transactions(con, CFG)
    tx = con.sql("select * from tx order by msno, transaction_date, source, source_row").df()
    timeline = build_timeline(tx, CFG.renewal_window_days)[0]
    labels = build_labels(tx, CFG)
    g = labels[labels.snapshot_date == T].sort_values("msno").reset_index(drop=True)
    for h in HORIZONS:
        g[f"label_reliable_{h}"] = False  # placeholders, as the stream (A9 drops the column)
    g = g[LABEL_INPUTS].copy()
    if plant is not None:
        plant(g)
    usage = UsageSource(con, f"read_parquet('{folder / 'user_logs.parquet'}')", g.msno.unique())
    con.register("batch_timeline", timeline)
    txs = TransactionSource(
        con, "batch_timeline", pd.Series(dtype=object), CFG.renewal_window_days, CFG.data_start
    )
    members = duckdb.sql(f"select * from '{members_path}'").df()
    members = members[members.msno.isin(set(g.msno))]
    frame, errors = assemble(T, g, usage, txs, members, "uncertain", CFG.data_start)
    con.close()
    assert errors == []
    return to_arrow(frame).select(list(move.JUDGED))


# --- the stream: bootstrap, October, the move ----------------------------------------------------


@pytest.fixture()
def replayed(raw, valkey):
    folder, members_path = raw
    c = cs.Consumer(valkey, 15, date(2016, 10, 1), is_scored=lambda m: True)
    c.bootstrap(folder)
    base = move.rebuild(valkey, CFG, None)  # the timelines after the bootstrap
    c.apply(source.events(folder))
    return valkey, base, members_path, folder


@pytest.mark.valkey
def test_the_stream_s_table_equals_batch_s_on_the_61_judged_columns(replayed):
    r, _, members_path, folder = replayed
    tl = move.rebuild(r, CFG, None)
    stream, g = move.feature_table(r, T, tl, CFG, members_path)
    result = parity.compare_tables(stream, batch_table(folder, members_path))
    assert result["met"], result
    assert result["rows_compared"] == stream.num_rows == len(g) > 50
    assert stream.column_names == list(move.JUDGED) and len(move.JUDGED) == 61
    assert not set(move.EXCLUDED) & set(stream.column_names)


@pytest.mark.valkey
def test_a_planted_labels_state_difference_is_reported_as_such(replayed):
    r, _, members_path, folder = replayed
    stream, _ = move.feature_table(r, T, move.rebuild(r, CFG, None), CFG, members_path)

    def plant(g):
        i = g.index[g.segment.eq("paid")][0]
        g.loc[i, "segment"] = "discount100_only"

    result = parity.compare_tables(stream, batch_table(folder, members_path, plant))
    assert not result["met"]
    assert result["labels_state_differences"] == ["segment"]
    assert result["stream_differences"] == []
    assert result["columns_differing"] == {"segment": {"differing": 1, "largest_relative": None}}


@pytest.mark.valkey
def test_rebuilding_touched_customers_only_equals_a_full_rebuild(replayed):
    r, base, _, _ = replayed
    touched = r.smembers(cs.TOUCHED)
    assert 0 < len(touched) < len(move.stored_rows(r).msno.unique())
    full, spliced = move.rebuild(r, CFG, None), move.rebuild(r, CFG, base)
    for name in ("tx", "coverage", "paid"):
        a, b = getattr(full, name), getattr(spliced, name)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@pytest.mark.valkey
def test_tx_raw_matches_phase_1_s_schema_and_rows_before_t(replayed):
    r, _, _, folder = replayed
    stream_con, phase1 = duckdb.connect(), duckdb.connect()
    move.tx_raw(stream_con, move.stored_rows(r))
    _load_transactions(phase1, folder / "transactions.parquet", folder / "transactions_v2.parquet")
    assert stream_con.sql("describe tx_raw").fetchall() == phase1.sql("describe tx_raw").fetchall()
    order = "order by source, source_row"
    ours = stream_con.sql(f"select * from tx_raw {order}").fetchall()
    theirs = phase1.sql(
        f"select * from tx_raw where transaction_date < date '{T}' {order}"
    ).fetchall()
    assert ours == theirs and len(ours) > 100


@pytest.mark.valkey
def test_a_missing_integer_stays_null_through_the_cast(valkey):
    c = cs.Consumer(valkey, 15, date(2016, 10, 1), is_scored=lambda m: False)
    row = tx_row("x", date(2016, 10, 3), 20161102, pm=None) | {"source_row": 0}
    c.apply([{"seq": 1, "source": "transactions", **row}])
    con = duckdb.connect()
    move.tx_raw(con, move.stored_rows(valkey))
    assert con.sql("select payment_method_id, _raw_payment_method_id from tx_raw").fetchall() == [
        (None, None)
    ]


@pytest.mark.valkey
def test_the_move_drops_ring_days_that_left_the_window(replayed):
    r, _, _, _ = replayed
    move.prune_ring(r, T)
    days = {int(k.decode().split(":")[0]) for key in r.scan_iter("rp:day:*") for k in r.hkeys(key)}
    assert days and min(days) >= 20161004  # 28 days before 2016-11-01


# --- the replayed store --------------------------------------------------------------------------


@pytest.mark.valkey
def test_the_replayed_store_is_written_to_the_replay_database_only(replayed, tmp_path):
    from retention.api import store
    from retention.explain.contributions import SavedModel
    from retention.explain.model import Settings, booster_sha256, fit_models, load_settings
    from retention.models.protocol import load_protocol

    r, _, members_path, _ = replayed
    tl = move.rebuild(r, CFG, None)
    table, g = move.feature_table(r, T, tl, CFG, members_path)
    real = load_settings(horizons=(7, 30, 59))
    settings = Settings(
        columns=list(real.columns),
        categorical=list(real.categorical),
        fixed=yaml.safe_load(Path("configs/evaluation.yaml").read_text())["classifier"][
            "fixed_params"
        ],
        candidates={
            h: {"num_leaves": 3, "n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 5}
            for h in (7, 30, 59)
        },
    )
    feats = table.to_pandas(types_mapper=move.NULLABLE.get, date_as_object=False)
    rng = np.random.default_rng(1)
    p = load_protocol()
    c01 = {}
    for h in (7, 30, 59):
        fit = feats.assign(y=rng.integers(0, 2, len(feats)))
        model, platt = fit_models(fit, fit, settings, h, p)
        c01[h] = SavedModel(h, model, platt.a, platt.b, booster_sha256(model))
    columns = list(real.columns)
    written = move.write_store(
        r, 15, T, table, g, tl.tx, c01, p.seed, columns, tuple(p.main_segments)
    )
    assert store.store_date(r) == T
    assert written["customer_documents"] == len(g)
    assert written["ranked"] == r.zcard("rs:rank:30") > 0
    with pytest.raises(PermissionError, match="database 0 holds the loaded store"):
        move.write_store(
            Untouchable(db=0, settings_db=1), 1, T, table, g, tl.tx, c01, p.seed, columns, ()
        )
    with pytest.raises(PermissionError, match="connected to database 15, expected 1"):
        move.write_store(r, 1, T, table, g, tl.tx, c01, p.seed, columns, ())


def test_a9_s_judged_columns_are_the_registry_without_the_seven():
    assert len(move.JUDGED) == 61 and len(move.EXCLUDED) == 7
    assert move.JUDGED[:2] == ("msno", "snapshot_date")
    assert dataclasses.is_dataclass(move.Timelines)
