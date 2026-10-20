"""The service's store in Valkey (through the redis client), one store date at a time.

Key names come from configs/serving.yaml, `step5.store.keys`:
- the store's date;
- one JSON document per customer with coverage: status, outside code,
  next_expiry_in_days and, for a scored customer, the model, value and fact
  columns and the price;
- the known customers (a transaction dated before the store's date);
- one ranking per horizon (7, 30, 59) of scored customers by final probability;
- the /at-risk header counts;
- the cancel-event list and the replay counters (filled in step 6).

Loading a snapshot deletes every key under the store's prefix first and writes the
store's date last, so a half-written store has no date and the service is not
ready. A float survives the JSON round trip exactly (Python writes the shortest
text that reads back as the same float); rows are rebuilt with the same pandas
types the batch loader gives them, from the features schema, so no data file is
needed to read them.

Rankings are sorted sets scored by minus the final probability: the set's own
order (score ascending, then member ascending) is final probability descending,
ties by msno ascending, as registered.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from retention.api.config import load_serving
from retention.features.build import NULLABLE
from retention.features.spec import arrow_schema

HORIZONS = (7, 30, 59)
CHUNK = 5_000
DEFAULT_URL = "redis://localhost:6379/0"


@dataclass(frozen=True)
class Keys:
    prefix: str
    store_date: str
    customer: str  # with "<msno>"
    known: str
    rank: str  # with "<horizon>"
    counts: str
    events: str
    replay: tuple[str, ...]

    def customer_key(self, msno: str) -> str:
        return self.customer.replace("<msno>", msno)

    def rank_key(self, horizon: int) -> str:
        return self.rank.replace("<horizon>", str(horizon))


def load_keys(cfg: dict[str, Any] | None = None) -> Keys:
    k = (cfg if cfg is not None else load_serving())["step5"]["store"]["keys"]
    names = [k["store_date"], k["customer"], k["known"], k["rank"], k["counts"], k["events"]]
    names += list(k["replay"])
    prefixes = {n.split(":")[0] for n in names}
    if len(prefixes) != 1:
        raise ValueError("every registered store key must share one prefix")
    return Keys(
        prefix=prefixes.pop() + ":",
        store_date=k["store_date"],
        customer=k["customer"],
        known=k["known"],
        rank=k["rank"],
        counts=k["counts"],
        events=k["events"],
        replay=tuple(k["replay"]),
    )


KEYS = load_keys()


def client(url: str | None = None):
    import redis

    return redis.Redis.from_url(url or os.environ.get("VALKEY_URL") or DEFAULT_URL)


# --- values in and out of JSON ---------------------------------------------------------------


def to_json_value(v: Any) -> Any:
    """A value JSON can hold exactly: missing as null, dates as ISO text, numpy as Python."""
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    return v


def schema_for(columns: list[str]) -> pa.Schema:
    s = arrow_schema()
    return pa.schema([s.field(c) for c in columns])


def rows_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Rows rebuilt with the batch loader's pandas types (features schema, nullable types)."""
    schema = schema_for(columns)
    data: dict[str, list] = {c: [] for c in columns}
    for r in records:
        for c in columns:
            v = r.get(c)
            if v is not None and pa.types.is_date32(schema.field(c).type):
                v = date.fromisoformat(v)
            data[c].append(v)
    table = pa.table(data, schema=schema)
    return table.to_pandas(types_mapper=NULLABLE.get, date_as_object=False)


# --- writing one snapshot ----------------------------------------------------------------------


def clear(r, keys: Keys = KEYS) -> int:
    """Delete every key under the store's prefix; returns how many."""
    n = 0
    batch = []
    for k in r.scan_iter(match=keys.prefix + "*", count=CHUNK):
        batch.append(k)
        if len(batch) >= CHUNK:
            n += r.delete(*batch)
            batch = []
    if batch:
        n += r.delete(*batch)
    return n


def load_snapshot(
    r,
    snapshot: date,
    status: pd.DataFrame,
    rows: pd.DataFrame,
    prices: pd.DataFrame,
    columns: list[str],
    known: list[str],
    finals: pd.DataFrame,
    counts: dict[str, int],
    keys: Keys = KEYS,
) -> dict[str, int]:
    """Replace the store with one snapshot; the store's date is written last.

    status: msno, status, code, next_expiry_in_days for every customer with coverage;
    rows and prices: the scored customers (same msno order as finals);
    finals: msno and the final probability at 7, 30 and 59 days of scored customers.
    """
    if status.msno.duplicated().any() or rows.msno.duplicated().any():
        raise ValueError("one document per customer")
    if not set(rows.msno) <= set(status.msno):
        raise ValueError("every scored row needs a status row")
    if not set(status.msno) <= set(known):
        raise ValueError("every customer with coverage must be known")
    removed = clear(r, keys)
    row_of = {m: i for i, m in enumerate(rows.msno.astype(str))}
    price_of = prices.set_index(prices.msno.astype(str))
    pipe = r.pipeline(transaction=False)
    written = 0
    for rec in status.to_dict("records"):
        m = str(rec["msno"])
        doc: dict[str, Any] = {
            "status": rec["status"],
            "code": rec["code"],
            "next_expiry_in_days": to_json_value(rec["next_expiry_in_days"]),
        }
        if m in row_of:
            i = row_of[m]
            doc["row"] = {c: to_json_value(rows[c].iloc[i]) for c in columns}
            p = price_of.loc[m]
            doc["price"] = {
                "price_per_day": to_json_value(p.price_per_day),
                "has_price": bool(p.has_price),
                "price_row_date": to_json_value(p.price_row_date),
            }
        pipe.set(keys.customer_key(m), json.dumps(doc, allow_nan=False))
        written += 1
        if written % CHUNK == 0:
            pipe.execute()
    pipe.execute()
    for i in range(0, len(known), CHUNK):
        r.sadd(keys.known, *known[i : i + CHUNK])
    for h in HORIZONS:
        col = f"days_{h}"
        scores = {str(m): -float(p) for m, p in zip(finals.msno, finals[col], strict=True)}
        items = list(scores.items())
        for i in range(0, len(items), CHUNK):
            r.zadd(keys.rank_key(h), dict(items[i : i + CHUNK]))
    r.set(keys.counts, json.dumps(counts, allow_nan=False))
    for key in keys.replay:
        r.set(key, 0)
    r.set(keys.store_date, snapshot.isoformat())
    return {
        "removed_keys": int(removed),
        "customer_documents": written,
        "known": len(known),
        "ranked": len(finals),
    }


# --- reading -------------------------------------------------------------------------------------


def store_date(r, keys: Keys = KEYS) -> date | None:
    v = r.get(keys.store_date)
    return None if v is None else date.fromisoformat(v.decode())


def is_known(r, msno: str, keys: Keys = KEYS) -> bool:
    return bool(r.sismember(keys.known, msno))


def customer(r, msno: str, keys: Keys = KEYS) -> dict[str, Any] | None:
    v = r.get(keys.customer_key(msno))
    return None if v is None else json.loads(v)


def ranking(r, horizon: int, offset: int, limit: int, keys: Keys = KEYS) -> list[tuple[str, float]]:
    """(msno, final probability) by final probability descending, ties by msno ascending."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    got = r.zrange(keys.rank_key(horizon), offset, offset + limit - 1, withscores=True)
    return [(m.decode(), -s) for m, s in got]


def counts(r, keys: Keys = KEYS) -> dict[str, int] | None:
    v = r.get(keys.counts)
    return None if v is None else json.loads(v)


def memory_used(r) -> int:
    return int(r.info("memory")["used_memory"])
