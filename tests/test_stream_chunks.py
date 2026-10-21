"""The chunked replay (D-221) on hand-made data: chunks change nothing.

For 1, 3 and 7 chunks on both seeds of test_stream_move's raw tables: the table equals the
unchunked move's and batch's on A9's 61 columns; the timelines (cleaned rows, coverage and
paid, ties included) equal the unchunked touched-only rebuild and a full rebuild; prices
and the labels state are the same; the replayed store holds the same ranking.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd
import pytest

from retention.data.sample import bucket
from retention.stream import consumer as cs
from retention.stream import move, parity, replay, source
from test_stream_move import CFG, T, batch_table, no_test_unlock, raw  # noqa: F401
from test_stream_parity import COLUMNS, MAIN, P, tiny_models

pytestmark = pytest.mark.valkey
COUNTS = (1, 3, 7)


@pytest.fixture()
def state(raw, valkey, tmp_path):  # noqa: F811
    """The consumer after the bootstrap (timelines written for every chunk count) and the
    October events; the unchunked bootstrap timelines kept in memory as the reference."""
    data, members_path = raw
    c = cs.Consumer(valkey, 15, date(2016, 10, 1), is_scored=lambda m: True)
    c.bootstrap(data)
    for n in COUNTS:
        move.write_bootstrap_timelines(valkey, CFG, tmp_path / f"work{n}", n)
    base = move.rebuild(valkey, CFG, None)
    c.apply(source.events(data))
    return valkey, data, members_path, tmp_path, base


def chunked(state, n, **kw):
    r, _, members_path, tmp_path, _ = state
    return move.chunked_move(r, T, CFG, members_path, tmp_path / f"work{n}", MAIN, n, **kw)


@pytest.mark.parametrize("n", COUNTS)
def test_a_chunked_table_equals_the_unchunked_one_and_batch_s(state, n):
    r, data, members_path, _, base = state
    unchunked, _ = move.feature_table(r, T, move.rebuild(r, CFG, base), CFG, members_path)
    result = chunked(state, n)
    assert parity.compare_tables(result.table, unchunked)["met"]
    assert replay.content_hash(result.table) == replay.content_hash(unchunked)
    assert parity.compare_tables(result.table, batch_table(data, members_path))["met"]


@pytest.mark.parametrize("n", COUNTS)
def test_chunk_boundaries_change_neither_ties_nor_the_touched_only_rebuild(state, n):
    r, _, _, _, base = state
    result = chunked(state, n, keep_timelines=True)
    spliced, full = move.rebuild(r, CFG, base), move.rebuild(r, CFG, None)
    for name in ("tx", "coverage", "paid"):
        got = getattr(result.timelines, name)
        for want in (getattr(spliced, name), getattr(full, name)):
            pd.testing.assert_frame_equal(got, want.reset_index(drop=True))
    tie = result.timelines.coverage[result.timelines.coverage.msno == "s-tie"]
    assert tie[["seq", "source", "source_row"]].values.tolist() == (
        full.coverage[full.coverage.msno == "s-tie"][
            ["seq", "source", "source_row"]
        ].values.tolist()
    )


def test_prices_labels_state_and_known_customers_do_not_depend_on_chunks(state):
    results = {n: chunked(state, n) for n in COUNTS}
    one = results[1]
    for n in COUNTS[1:]:
        r = results[n]
        pd.testing.assert_frame_equal(
            r.g.reset_index(drop=True), one.g.reset_index(drop=True), check_like=True
        )
        a = r.prices.sort_values("msno").reset_index(drop=True)
        b = one.prices.sort_values("msno").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        assert r.known == one.known and r.cleaned_rows == one.cleaned_rows


def test_the_store_holds_the_same_ranking_whatever_the_chunks(state):
    r = state[0]
    models = tiny_models(chunked(state, 1).table)
    rankings = {}
    for n in (1, 7):
        result = chunked(state, n)
        move.write_store(
            r, 15, T, result.table, result.g, result.prices, result.known, models, P.seed,
            COLUMNS, MAIN,
        )  # fmt: skip
        rankings[n] = {h: r.zrange(f"rs:rank:{h}", 0, -1, withscores=True) for h in (7, 30, 59)}
    assert rankings[1] == rankings[7] and len(rankings[1][30]) > 0


def test_chunks_split_customers_within_one_bucket(raw):  # noqa: F811
    data, members_path = raw
    ids = batch_table(data, members_path).column("msno").to_pylist()
    k = Counter(bucket(m) for m in ids).most_common(1)[0][0]
    same_bucket = [m for m in ids if bucket(m) == k]
    assert len({move.chunk_of(m, 7) for m in same_bucket}) > 1
