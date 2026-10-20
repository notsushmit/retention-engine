"""The Valkey store: exact round trips, one store date, known customers, ranking, the hook."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import valkey_guard
from retention.api import __main__ as cli
from retention.api import store as st
from retention.api.batch import known_customers
from retention.api.config import load_serving
from retention.features.spec import arrow_schema
from retention.models.data import load_features

OCT, NOV = date(2016, 10, 1), date(2016, 11, 1)


def typed_columns() -> list[str]:
    """One feature column of every type the features schema uses (its only date column is
    snapshot_date, a key, so no feature is a date)."""
    s = arrow_schema()
    want = {"string", "int8", "int16", "int32", "double", "bool"}
    out, seen = [], set()
    for f in s:
        t = str(f.type)
        if t in want and t not in seen and f.name not in ("msno", "snapshot_date"):
            out.append(f.name)
            seen.add(t)
    assert seen == want, seen
    return out


COLS = typed_columns()
TRICKY = [0.1 + 0.2, 1e-300, np.nextafter(0.5, 1.0), -0.0, 123456.78901234567, None]


def records(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    s = st.schema_for(COLS)
    out = []
    for i in range(n):
        r = {}
        for c in COLS:
            t = s.field(c).type
            if i % 5 == 4:
                r[c] = None
            elif pa.types.is_floating(t):
                r[c] = TRICKY[i % len(TRICKY)] if i < len(TRICKY) else float(rng.normal())
            elif pa.types.is_integer(t):
                r[c] = int(rng.integers(0, 100))
            elif pa.types.is_boolean(t):
                r[c] = bool(rng.integers(0, 2))
            else:
                r[c] = str(rng.choice(["paid", "trial", "x"]))
        out.append(r)
    return out


def snapshot_parts(ids: list[str], seed: int):
    recs = records(len(ids), seed)
    rows = st.rows_frame(recs, COLS)
    rows.insert(0, "msno", ids)
    status = pd.DataFrame(
        {
            "msno": ids,
            "status": "scored",
            "code": [None] * len(ids),
            "next_expiry_in_days": pd.array(range(len(ids)), "Int32"),
        }
    )
    prices = pd.DataFrame(
        {
            "msno": ids,
            "price_per_day": [4.966666666666667] * len(ids),
            "has_price": True,
            "price_row_date": [date(2016, 10, 3)] * len(ids),
        }
    )
    rng = np.random.default_rng(seed + 1)
    finals = pd.DataFrame({"msno": ids})
    for h in (7, 30, 59):
        finals[f"days_{h}"] = rng.uniform(0, 1, len(ids))
    return status, rows, prices, finals


def load(r, snapshot, ids, seed=0, extra_known=()):
    status, rows, prices, finals = snapshot_parts(ids, seed)
    counts = {"scored": len(ids), "not_active": len(extra_known)}
    return st.load_snapshot(
        r, snapshot, status, rows, prices, COLS, [*ids, *extra_known], finals, counts
    ), (status, rows, prices, finals)


# --- without Valkey -------------------------------------------------------------------------------


def test_key_names_are_the_registered_ones():
    k = load_serving()["step5"]["store"]["keys"]
    assert st.KEYS.prefix == "rs:"
    assert st.KEYS.customer_key("abc") == k["customer"].replace("<msno>", "abc")
    assert st.KEYS.rank_key(30) == "rs:rank:30"
    assert (st.KEYS.store_date, st.KEYS.known, st.KEYS.counts) == (
        k["store_date"],
        k["known"],
        k["counts"],
    )


def test_rows_come_back_with_the_batch_loader_s_types(tmp_path):
    recs = records(12, 3)
    table = st.rows_frame(recs, COLS)
    full = pa.Table.from_pandas(table, preserve_index=False)
    full = full.append_column("msno", pa.array([f"m{i:02d}" for i in range(12)]))
    full = full.append_column("snapshot_date", pa.array([NOV] * 12, pa.date32()))
    pq.write_table(full, tmp_path / "features.parquet")
    batch = load_features(tmp_path / "features.parquet", [NOV], COLS)
    pd.testing.assert_frame_equal(
        batch[COLS].reset_index(drop=True), table, check_exact=True, check_dtype=True
    )


def test_json_values():
    assert st.to_json_value(np.float64(0.1)) == 0.1 and st.to_json_value(np.int32(4)) == 4
    assert st.to_json_value(pd.NA) is None and st.to_json_value(float("nan")) is None
    assert st.to_json_value(pd.Timestamp(NOV)) == "2016-11-01"
    assert st.to_json_value(NOV) == "2016-11-01"


def test_known_customers_read_nothing_dated_on_or_after_t(tmp_path):
    tx = pd.DataFrame(
        {
            "msno": ["a", "b", "b", "c"],
            "transaction_date": [date(2016, 10, 31), date(2016, 11, 1), date(2015, 1, 1), NOV],
        }
    )
    tx.to_parquet(tmp_path / "tx.parquet")
    assert known_customers(NOV, tmp_path / "tx.parquet") == ["a", "b"]


# --- the hook ------------------------------------------------------------------------------------


def run_hook(skipped: bool, marked: bool) -> str:
    report = SimpleNamespace(skipped=skipped, outcome="skipped" if skipped else "passed")
    item = SimpleNamespace(
        nodeid="x::t", get_closest_marker=lambda name: object() if marked else None
    )
    gen = valkey_guard.pytest_runtest_makereport(item, None)
    next(gen)
    try:
        gen.send(SimpleNamespace(get_result=lambda: report))
    except StopIteration:
        pass
    return report.outcome


def test_a_skipped_valkey_test_fails_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert run_hook(skipped=True, marked=True) == "failed"
    assert run_hook(skipped=True, marked=False) == "skipped"  # other skips are left alone
    assert run_hook(skipped=False, marked=True) == "passed"


def test_a_skipped_valkey_test_stays_skipped_outside_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    assert run_hook(skipped=True, marked=True) == "skipped"


def test_the_fixture_skips_with_a_stated_reason(monkeypatch):
    monkeypatch.setenv("VALKEY_URL", "redis://127.0.0.1:1/0")  # nothing listens there
    gen = valkey_guard.valkey.__wrapped__()
    with pytest.raises(pytest.skip.Exception, match="Valkey not reachable at"):
        next(gen)


# --- with Valkey ----------------------------------------------------------------------------------


@pytest.mark.valkey
def test_floats_and_every_type_round_trip_exactly(valkey):
    ids = [f"c{i:03d}" for i in range(30)]
    _, (status, rows, prices, finals) = load(valkey, NOV, ids, seed=5)
    back = st.rows_frame([st.customer(valkey, m)["row"] for m in ids], COLS)
    pd.testing.assert_frame_equal(back, rows[COLS], check_exact=True, check_dtype=True)
    for h in (7, 30, 59):
        got = dict(st.ranking(valkey, h, 0, 100))
        assert all(got[m] == p for m, p in zip(finals.msno, finals[f"days_{h}"], strict=True))
    doc = st.customer(valkey, ids[0])
    assert doc["price"] == {
        "price_per_day": 4.966666666666667,
        "has_price": True,
        "price_row_date": "2016-10-03",
    }


@pytest.mark.valkey
def test_one_store_date_and_a_second_snapshot_replaces_the_first(valkey):
    assert st.store_date(valkey) is None
    load(valkey, OCT, ["a1", "a2", "a3"], extra_known=["k1"])
    valkey.set("other:untouched", "1")
    assert st.store_date(valkey) == OCT
    written, _ = load(valkey, NOV, ["b1", "b2"], seed=1)
    assert st.store_date(valkey) == NOV
    assert st.customer(valkey, "a1") is None and st.customer(valkey, "b1") is not None
    assert not st.is_known(valkey, "a1") and not st.is_known(valkey, "k1")
    assert {m for m, _ in st.ranking(valkey, 30, 0, 10)} == {"b1", "b2"}
    assert valkey.get("other:untouched") == b"1"  # only the store's prefix is cleared
    assert written["removed_keys"] > 0


@pytest.mark.valkey
def test_known_customers_include_those_without_coverage(valkey):
    load(valkey, NOV, ["s1", "s2"], extra_known=["n1"])
    assert st.is_known(valkey, "s1") and st.is_known(valkey, "n1")
    assert not st.is_known(valkey, "zz")
    assert st.customer(valkey, "n1") is None  # known but no coverage: no document


@pytest.mark.valkey
def test_ranking_is_descending_with_ties_by_msno_and_pages_cleanly(valkey):
    ids = ["d", "b", "a", "c", "e"]
    status, rows, prices, finals = snapshot_parts(ids, 2)
    finals["days_30"] = [0.5, 0.9, 0.5, 0.5, 0.1]
    st.load_snapshot(valkey, NOV, status, rows, prices, COLS, ids, finals, {"scored": 5})
    order = [m for m, _ in st.ranking(valkey, 30, 0, 10)]
    assert order == ["b", "a", "c", "d", "e"]
    pages = [m for o in (0, 2, 4) for m, _ in st.ranking(valkey, 30, o, 2)]
    assert pages == order
    with pytest.raises(ValueError):
        st.ranking(valkey, 14, 0, 1)


@pytest.mark.valkey
def test_counts_and_replay_counters(valkey):
    load(valkey, NOV, ["x1"], extra_known=["x2", "x3"])
    assert st.counts(valkey) == {"scored": 1, "not_active": 2}
    for key in st.KEYS.replay:
        assert valkey.get(key) == b"0"
    assert valkey.llen(st.KEYS.events) == 0  # filled by the replay in step 6


@pytest.mark.valkey
def test_inconsistent_inputs_are_refused_before_anything_is_written(valkey):
    status, rows, prices, finals = snapshot_parts(["a", "b"], 0)
    bad = [
        (pd.concat([status, status.iloc[:1]]), rows, ["a", "b"]),
        (status.iloc[:1], rows, ["a", "b"]),
        (status, rows, ["a"]),
    ]
    for s, r_, known in bad:
        with pytest.raises(ValueError):
            st.load_snapshot(valkey, NOV, s, r_, prices, COLS, known, finals, {})
    assert st.store_date(valkey) is None and valkey.dbsize() == 0


@pytest.mark.valkey
def test_documents_are_plain_json(valkey):
    load(valkey, NOV, ["j1"])
    raw = valkey.get(st.KEYS.customer_key("j1"))
    doc = json.loads(raw)
    assert set(doc) == {"status", "code", "next_expiry_in_days", "row", "price"}
    assert b"NaN" not in raw and b"Infinity" not in raw


# --- the command's guards -------------------------------------------------------------------------


def test_load_snapshot_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["load-snapshot", "--snapshot", "2016-11-01"])


def test_load_snapshot_runs_once_per_snapshot(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    (tmp_path / "load_snapshot_2016-11-01.json").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        cli.main(["load-snapshot", "--snapshot", "2016-11-01"])
