"""The store-date move: the feature table at T from the replay's state, with Phase 1 and 2 code.

configs/serving.yaml, streaming.consumer, step6 and amendment A9 (stream_parity_columns).
Everything here reads only the replay database's state, which holds rows dated before
2016-11-01, and members.parquet (A9's fourth input). Phase 1 and Phase 2 functions are
reused unchanged:

- cleaning: the stored transaction rows become a tx_raw table with Phase 1's layout
  and casts (source, source_row, retention.data.schema.typed_select of TRANSACTIONS),
  then retention.data.clean._build_transactions builds the cleaned rows;
- timelines: retention.labels.timeline.build_timeline, coverage and paid renewals;
  state at T by in_force_at, segments by snapshots._segments and _segment_at, paid
  state by the same rule as snapshots.build_labels;
- transactions: retention.features.transactions.TransactionSource on the stream's
  coverage timeline;
- usage: the frame UsageSource._chunk returns, rebuilt from the ring and the first and
  last listening days with the same integer sums and the same float steps, then
  retention.features.usage._derive and the same cohort median;
- members: retention.features.members.member_features on members.parquet, masked by
  the stream's own has_any_log_before_T;
- assembly: retention.features.build.assemble and its checks, with the stream's usage
  source and Phase 2's TransactionSource.

The timelines can be rebuilt for touched customers only and spliced into the timelines
built at the bootstrap (or the last move); a test shows this equals a full rebuild.

The table keeps A9's 61 columns: the 7 excluded metadata columns depend on month
completeness statuses computed from later data, or on label_reliable_h (outcome);
assemble needs a value for them, so it is given a placeholder ("uncertain", False)
and the 7 columns are dropped before anything is written. No outcome column is read.

The replayed store is written only to the replay database, through
producer.require_database, never database 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retention.api import scoring as sc
from retention.api import store
from retention.api.config import load_serving
from retention.data.clean import _build_transactions
from retention.data.schema import TRANSACTIONS, typed_select
from retention.features import spec
from retention.features.build import HORIZONS, NULLABLE, assemble, to_arrow
from retention.features.transactions import TransactionSource
from retention.features.usage import SLOPE_WEIGHTS, USAGE_COLUMNS, _derive
from retention.labels.config import LabelConfig
from retention.labels.snapshots import _segment_at, _segments
from retention.labels.timeline import build_timeline, in_force_at
from retention.optimize.price import PRICE_COLUMNS, price_per_day
from retention.stream import consumer as cs
from retention.stream import producer

A9 = load_serving()["stream_parity_columns"]
EXCLUDED = tuple(A9["excluded"])
JUDGED = tuple(c.name for c in spec.COLUMNS if c.name not in EXCLUDED)
MEMBER_INPUT = tuple(A9["members_input"]["columns"])
LABELS_STATE = ("days_until_expiry", "days_until_paid_expiry", "segment", "paid_state")
PLACEHOLDER_PRE_T = "uncertain"  # assemble needs a value; the column is dropped (A9)
CHUNK = 5_000
if len(JUDGED) != int(A9["columns_judged"]):
    raise ValueError("A9's judged columns and the feature registry disagree")


# --- the stored rows -----------------------------------------------------------------------------


def stored_rows(r, msnos: list[str] | None = None) -> pd.DataFrame:
    """Stored transaction rows (msno and consumer.TX_STORED); every customer, or `msnos`."""
    if msnos is None:
        msnos = sorted(k.decode().removeprefix("rp:tx:") for k in r.scan_iter("rp:tx:*", 10_000))
    records = []
    for i in range(0, len(msnos), CHUNK):
        part = msnos[i : i + CHUNK]
        pipe = r.pipeline(transaction=False)
        for m in part:
            pipe.lrange(cs.tx_key(m), 0, -1)
        for m, rows in zip(part, pipe.execute(), strict=True):
            records += [[m, *json.loads(v)] for v in rows]
    frame = pd.DataFrame(records, columns=["msno", *cs.TX_STORED], dtype=object)
    for c in frame.columns:
        if c not in ("msno", "source"):
            frame[c] = pd.array(frame[c].tolist(), dtype="Int64")  # nulls stay nulls, never floats
    return frame


def tx_raw(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> None:
    """Phase 1's tx_raw table (clean._load_transactions' layout and casts) from stored rows."""
    con.register("_stored", rows)
    con.execute(
        "create or replace table tx_raw as select s.source, s.source_row, "
        f"{typed_select(TRANSACTIONS, prefix='s.')} from _stored s"
    )
    con.unregister("_stored")


def clean_rows(rows: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Phase 1's cleaned transactions for the stored rows, ordered as Phase 1 writes them."""
    con = duckdb.connect()
    try:
        tx_raw(con, rows)
        _build_transactions(con, cfg)
        bad = con.sql("select count(*) from tx where expiry_imputed and expiry_date is null")
        if bad.fetchone()[0]:
            raise ValueError("a placeholder expiry with no registered coverage; Phase 1 stops here")
        return con.sql("select * from tx order by msno, transaction_date, source, source_row").df()
    finally:
        con.close()


# --- the timelines -------------------------------------------------------------------------------


@dataclass
class Timelines:
    tx: pd.DataFrame  # cleaned rows
    coverage: pd.DataFrame
    paid: pd.DataFrame

    @classmethod
    def build(cls, rows: pd.DataFrame, cfg: LabelConfig) -> Timelines:
        tx = clean_rows(rows, cfg)
        w = cfg.renewal_window_days
        return cls(tx, build_timeline(tx, w)[0], build_timeline(tx, w, "paid")[0])

    def splice(self, touched: Timelines, msnos: set[str]) -> Timelines:
        """These timelines with the given customers' rows replaced by `touched`."""

        def merge(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
            kept = old[~old.msno.isin(msnos)]
            out = pd.concat([kept, new], ignore_index=True)
            return out.sort_values(keys, kind="stable").reset_index(drop=True)

        order = ["msno", "transaction_date", "source", "source_row"]
        return Timelines(
            merge(self.tx, touched.tx, order),
            merge(self.coverage, touched.coverage, ["msno", "seq"]),
            merge(self.paid, touched.paid, ["msno", "seq"]),
        )


def rebuild(r, cfg: LabelConfig, base: Timelines | None) -> Timelines:
    """Timelines at a move: a full build, or the touched customers spliced into `base`."""
    if base is None:
        return Timelines.build(stored_rows(r), cfg)
    touched = sorted(m.decode() for m in r.smembers(cs.TOUCHED))
    return base.splice(Timelines.build(stored_rows(r, touched), cfg), set(touched))


# --- labels state at T (no outcome) ---------------------------------------------------------------


def labels_state(tl: Timelines, t: date, cfg: LabelConfig) -> pd.DataFrame:
    """Coverage population at T with its state, as snapshots.build_labels computes it."""
    w, same = cfg.renewal_window_days, cfg.snapshot_includes_same_day_rows
    ts = pd.Timestamp(t)
    st = in_force_at(tl.coverage, t, same).rename(columns={"eff_expiry_after": "expiry"})
    st = st[st.expiry >= ts]
    out = pd.DataFrame(index=st.index)
    out["snapshot_date"] = ts
    out["expiry_date"] = st.expiry
    out["days_until_expiry"] = (st.expiry - ts).dt.days.astype("int32")
    out["in_exp30"] = out.days_until_expiry < w
    out = out.join(_segment_at(tl.coverage, _segments(tl.coverage), ts, same))
    pst = in_force_at(tl.paid, t, same).rename(columns={"eff_expiry_after": "expiry"})
    pst = pst.reindex(out.index)
    out["paid_expiry_date"] = pst.expiry
    out["paid_state"] = np.select(
        [pst.expiry.isna(), pst.expiry >= ts, pst.expiry >= ts - pd.Timedelta(days=w)],
        ["never_paid", "active", "grace"],
        "lapsed",
    )
    out = out.rename_axis("msno").reset_index().sort_values("msno", kind="stable")
    out["snapshot_date"] = out.snapshot_date.dt.date
    return out.reset_index(drop=True)


# --- usage from the ring -------------------------------------------------------------------------


def decimal_weights() -> tuple[tuple[int, ...], float]:
    """The slope weights as DuckDB reads their SQL literals: integer units of one decimal
    scale, and 10^scale as a float (usage._slope_sql writes each weight with repr)."""
    parts = [Decimal(repr(float(x))) for x in SLOPE_WEIGHTS]
    scale = max(-d.as_tuple().exponent for d in parts)
    return tuple(int(d.scaleb(scale)) for d in parts), float(10**scale)


DECIMAL_WEIGHTS = decimal_weights()


class StreamUsage:
    """UsageSource's features, from the replay's ring and listening days (duck-typed)."""

    def __init__(self, r):
        self.r = r

    def frame(self, t: date, population: list[str]) -> pd.DataFrame:
        """The frame UsageSource._chunk returns, column for column."""
        tday = cs.source.day_number(t)
        lo = cs.source.day_number(t - timedelta(days=spec.WINDOW_LONG))
        w = [float(x) for x in SLOPE_WEIGHTS]
        rows = []
        for i in range(0, len(population), CHUNK):
            part = population[i : i + CHUNK]
            pipe = self.r.pipeline(transaction=False)
            pipe.hmget(cs.FIRST, part)
            pipe.hmget(cs.LAST, part)
            for m in part:
                pipe.hgetall(cs.ring_key(m))
            got = pipe.execute()
            firsts, lasts, rings = got[0], got[1], got[2:]
            for m, f, la, ring in zip(part, firsts, lasts, rings, strict=True):
                rows.append(self._row(t, tday, lo, w, m, f, la, ring))
        return pd.DataFrame(rows)

    @staticmethod
    def _row(t, tday, lo, w, m, f, la, ring) -> dict[str, Any]:
        first = None if f is None else int(f)
        last = None if la is None else int(la)
        days: dict[int, dict[str, int]] = {}
        for k, v in ring.items():
            day, field = k.decode().split(":")
            if lo <= int(day) < tday:
                days.setdefault(int(day), dict.fromkeys(cs.DAY_FIELDS, 0))[field] = int(v)
        nan = float("nan")
        row: dict[str, Any] = {
            "msno": m,
            "has_any_log_before_T": nan if first is None else float(first < tday),
            "days_since_last_listen": nan if last is None else float((t - cs.as_date(last)).days),
            "first_log_in_window": nan if first is None else float(lo <= first < tday),
            "cid": 0,
        }
        names = (
            "active_days_last7 active_days_base21 active_days_28 secs_last7 secs_base21 "
            "secs_28 plays_last7 plays_28 full_last7 full_28 unq_last7 unq_28 "
            "invalid_secs_days_28"
        ).split()
        weeks = [f"{p}_{wk}" for wk in spec.WEEKS for p in ("s", "a")]
        if not days:
            row |= dict.fromkeys(names + weeks, nan)
            row |= dict.fromkeys(["slope_secs_4w", "slope_active_4w"], nan)
            row |= dict.fromkeys(["mean_secs_4w", "mean_active_4w"], nan)
            return row

        def offset(day: int) -> int:
            return (t - cs.as_date(day)).days

        def window(name: str) -> list[dict[str, int]]:
            a, b = spec.WINDOWS[name]
            return [v for d, v in days.items() if a <= offset(d) <= b]

        last7, base21, all28 = window("last7"), window("base21"), window("28")
        row |= {
            "active_days_last7": len(last7),
            "active_days_base21": len(base21),
            "active_days_28": len(all28),
            "secs_last7": sum(v["ms"] for v in last7) / 1000.0,
            "secs_base21": sum(v["ms"] for v in base21) / 1000.0,
            "secs_28": sum(v["ms"] for v in all28) / 1000.0,
            "plays_last7": sum(v["p"] for v in last7),
            "plays_28": sum(v["p"] for v in all28),
            "full_last7": sum(v["f"] for v in last7),
            "full_28": sum(v["f"] for v in all28),
            "unq_last7": sum(v["u"] for v in last7),
            "unq_28": sum(v["u"] for v in all28),
            "invalid_secs_days_28": sum(1 for v in all28 if v["x"]),
        }
        for wk in spec.WEEKS:
            vals = window(wk)
            row[f"s_{wk}"] = sum(v["ms"] for v in vals) / 7000.0
            row[f"a_{wk}"] = len(vals)
        s = [row[f"s_{wk}"] for wk in spec.WEEKS]
        a = [row[f"a_{wk}"] for wk in spec.WEEKS]
        newcomer = first is not None and lo <= first < tday
        no_slope = row["active_days_28"] == 0 or newcomer
        # The same expressions, in the same order, as usage._slope_sql and the mean in _chunk.
        # DuckDB reads the weight literals as DECIMAL: times a DOUBLE (seconds) the product is
        # float arithmetic; times a BIGINT (active days) it stays an exact decimal, cast to
        # DOUBLE once at the end (value / 10^scale). Both are reproduced as DuckDB does them.
        row["slope_secs_4w"] = (
            nan if no_slope else w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * s[3]
        )
        units, scale = DECIMAL_WEIGHTS
        exact = units[0] * a[0] + units[1] * a[1] + units[2] * a[2] + units[3] * a[3]
        row["slope_active_4w"] = nan if no_slope else exact / scale
        row["mean_secs_4w"] = (s[0] + s[1] + s[2] + s[3]) / 4.0
        row["mean_active_4w"] = (a[0] + a[1] + a[2] + a[3]) / 4.0
        return row

    def features(self, t: date, population, n_chunks: int = 1, keep_weeks: bool = False):
        """As UsageSource.features: _derive, then the cohort median over the population."""
        pop = list(pd.unique(pd.Series(population, dtype=object)))
        out = _derive(self.frame(t, pop))
        valid = out.ratio_secs_7_vs_base.dropna()
        median = float(valid.median()) if len(valid) else np.nan
        out["ratio_secs_vs_cohort"] = out.ratio_secs_7_vs_base / median
        out = out.set_index("msno").loc[pop].reset_index()
        return out[["msno", *USAGE_COLUMNS]]


# --- the table ----------------------------------------------------------------------------------


def read_members(path: Path, population: list[str]) -> pd.DataFrame:
    """members.parquet (A9's fourth input), the columns Phase 2 reads, as Phase 2 reads them."""
    m = duckdb.sql(f"select {', '.join(MEMBER_INPUT)} from '{path}'").df()
    return m[m.msno.isin(set(population))]


def feature_table(
    r, t: date, tl: Timelines, cfg: LabelConfig, members_path: Path
) -> tuple[pa.Table, pd.DataFrame]:
    """A9's 61 columns for every customer with coverage at t, and the labels state."""
    g = labels_state(tl, t, cfg)
    for h in HORIZONS:  # placeholders for assemble; snapshot_usable_h is dropped (A9)
        g[f"label_reliable_{h}"] = False
    con = duckdb.connect()
    try:
        con.register("stream_timeline", tl.coverage)
        tx = TransactionSource(
            con,
            "stream_timeline",
            pd.Series(dtype=object),
            cfg.renewal_window_days,
            cfg.data_start,
        )
        members = read_members(members_path, g.msno.tolist())
        frame, errors = assemble(
            t, g, StreamUsage(r), tx, members, PLACEHOLDER_PRE_T, cfg.data_start
        )
    finally:
        con.close()
    if errors:
        raise ValueError("the stream's feature checks failed: " + "; ".join(errors))
    return to_arrow(frame).select(list(JUDGED)), g


def write_table(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def prune_ring(r, t: date) -> int:
    """Drop ring days that can no longer fall in a 28-day window at or after t."""
    floor = cs.source.day_number(t - timedelta(days=spec.WINDOW_LONG))
    dropped = 0
    for key in r.scan_iter("rp:day:*", 10_000):
        old = [k for k in r.hkeys(key) if int(k.decode().split(":")[0]) < floor]
        if old:
            dropped += r.hdel(key, *old)
    return dropped


# --- the replayed store ---------------------------------------------------------------------------


def write_store(
    r,
    database: int,
    t: date,
    table: pa.Table,
    g: pd.DataFrame,
    tx_clean: pd.DataFrame,
    c01: dict,
    seed: int,
    columns: list[str],
    main_segments: tuple[str, ...],
) -> dict[str, int]:
    """The replayed store for t, written only to the replay database (never database 0)."""
    producer.refuse_loaded_store(r)
    producer.require_database(r, database)
    feats = table.to_pandas(types_mapper=NULLABLE.get, date_as_object=False)
    status_rows = g[["msno", "in_exp30", "segment"]].merge(
        feats[["msno", "days_until_expiry"]], on="msno", validate="one_to_one"
    )
    status = sc.population_status(status_rows, main_segments)
    scored = set(status.msno[status.status == sc.SCORED])
    rows = feats[feats.msno.isin(scored)].reset_index(drop=True)
    status, keep = sc.refuse_unseen(status, rows, [m.model for m in c01.values()])
    rows = rows[keep].reset_index(drop=True)
    prices = price_per_day(tx_clean[PRICE_COLUMNS], rows[["msno", "snapshot_date"]])
    pr = sc.probabilities(c01, rows, seed)
    finals = pd.DataFrame({"msno": rows.msno.astype(str)})
    for k, h in enumerate(sc.HORIZONS):
        finals[f"days_{h}"] = pr.final[:, k]
    known = sorted(tx_clean.msno.astype(str).unique())
    counts = sc.population_counts(status) | {"not_active": len(known) - len(status)}
    return store.load_snapshot(r, t, status, rows, prices, columns, known, finals, counts)
