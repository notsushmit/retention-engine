"""Batch inputs of one snapshot for the scoring function, read from the processed tables.

Every reader here is an existing one: Phase 3's join (p2_main_rows, the row order
the rehearsal and every later reproduction used), Phase 4's feature loader for the
reason facts, and Phase 7's price reader. labels.parquet is read for two columns
known at T only (in_exp30, segment); no outcome column is read.

- status rows: every customer with coverage at T (P1, all segments) with in_exp30,
  segment and days_until_expiry, for the status of each customer;
- rows: the training population (P2 main) with every model, value and fact
  column, in the registered order;
- facts: the reason fact columns of those rows, as Phase 4 read them;
- prices: price per day from the last paid transaction dated before T (Phase 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from retention.explain.model import LABEL_COLUMNS, p2_main_rows
from retention.models.data import column_kind, load_features, load_labels
from retention.models.protocol import Protocol
from retention.optimize.price import CLEAN_TRANSACTIONS, price_per_day, read_transactions

STATUS_FEATURES = ("days_until_expiry",)


@dataclass
class SnapshotInputs:
    snapshot: date
    status_rows: pd.DataFrame  # msno, in_exp30, segment, days_until_expiry; every P1 row
    rows: pd.DataFrame  # P2 main, every requested column, registered order
    facts: pd.DataFrame  # msno plus the fact columns, one row per P2 main customer
    prices: pd.DataFrame  # msno, snapshot_date, price_per_day, has_price, price_row_date


def status_rows(processed: Path, snapshot: date, protocol: Protocol) -> pd.DataFrame:
    """Every customer with coverage at T, with the two label columns known at T."""
    for c in LABEL_COLUMNS:
        if column_kind(c) != "pre_t":
            raise PermissionError(f"labels column {c!r} is not known at T")
    labels = load_labels(processed / "labels.parquet", [snapshot], list(LABEL_COLUMNS), protocol)
    feats = load_features(processed / "features.parquet", [snapshot], list(STATUS_FEATURES))
    df = feats.drop(columns="snapshot_date").merge(
        labels.drop(columns="snapshot_date"), on="msno", how="inner", validate="one_to_one"
    )
    if len(df) != len(feats) or len(df) != len(labels):
        raise ValueError(f"{snapshot}: features and labels hold different customers")
    return df[["msno", "in_exp30", "segment", *STATUS_FEATURES]]


def snapshot_inputs(
    processed: Path,
    snapshot: date,
    columns: list[str],
    fact_columns: list[str],
    protocol: Protocol,
    transactions: pd.DataFrame | None = None,
) -> SnapshotInputs:
    """Everything the scoring function needs for one snapshot, from the batch tables."""
    cols = list(dict.fromkeys([*columns, *fact_columns]))
    rows = p2_main_rows(processed, snapshot, cols, protocol)
    facts = load_features(processed / "features.parquet", [snapshot], list(fact_columns))
    facts = facts.drop(columns="snapshot_date")
    facts = facts[facts.msno.isin(set(rows.msno))].reset_index(drop=True)
    tx = read_transactions([snapshot]) if transactions is None else transactions
    prices = price_per_day(tx, rows[["msno", "snapshot_date"]])
    return SnapshotInputs(snapshot, status_rows(processed, snapshot, protocol), rows, facts, prices)


def known_customers(snapshot: date, path: Path = CLEAN_TRANSACTIONS) -> list[str]:
    """Customers with a transaction dated before the snapshot (step 5's known customers);
    nothing dated on or after it is read."""
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.sql(
            f"select distinct msno from read_parquet('{path}') "
            f"where transaction_date < date '{snapshot.isoformat()}' order by msno"
        ).fetchall()
    finally:
        con.close()
    return [m for (m,) in rows]
