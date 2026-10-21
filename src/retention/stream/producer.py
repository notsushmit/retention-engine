"""The replay's producer: numbered events into a Valkey stream in the replay database.

configs/serving.yaml, step6. Every step 6 command refuses database 0 (the store
loaded in step 5f); the producer asks the server which database the client is on
(client_info), not the client's own settings (D-211), before it writes anything.
Events are appended in the source's order, one stream entry each, with the event
as one JSON field; the produced count is kept in the store's
rs:replay:produced counter (step5.store.keys) of the same database. The
throttled replay pauses 1 ms after every 1,000 events (step6.throttle); full
speed never pauses.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from retention.api import store
from retention.api.config import load_serving

STEP6 = load_serving()["step6"]
LOADED_STORE_DB = int(STEP6["databases"]["loaded_store"])  # 0
STREAM = "rp:events"  # the replay's event stream, in the replay database
THROTTLE = re.compile(r"^(?P<ms>\d+) ms pause per (?P<every>[\d,]+) events$")


def throttle_settings(cfg: dict[str, Any] | None = None) -> tuple[float, int]:
    """(pause in seconds, events between pauses) from the registered text."""
    m = THROTTLE.match((cfg or STEP6)["throttle"]["throttled"])
    if m is None:
        raise ValueError("the registered throttle no longer fits its pattern")
    return int(m["ms"]) / 1000, int(m["every"].replace(",", ""))


def client_for(database: int, url: str | None = None):
    """A client on `database` of VALKEY_URL's server: the URL's path is rewritten, because a
    database in the URL overrides a db argument in redis-py (D-211); then checked with the
    server."""
    import os
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    import redis

    parts = urlsplit(url or os.environ.get("VALKEY_URL") or store.DEFAULT_URL)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k != "db"])
    r = redis.Redis.from_url(urlunsplit(parts._replace(path=f"/{database}", query=query)))
    require_database(r, database)
    return r


def database_of(r) -> int:
    """The database the server says this client is on."""
    return int(r.client_info()["db"])


def require_database(r, expected: int) -> None:
    """A step 6 command's check that it is connected where it means to write."""
    found = database_of(r)
    if found != expected:
        raise PermissionError(f"connected to database {found}, expected {expected}")


def refuse_loaded_store(r) -> None:
    """Every step 6 command refuses database 0 (step6.databases.refuse)."""
    if database_of(r) == LOADED_STORE_DB:
        raise PermissionError(
            f"database {LOADED_STORE_DB} holds the loaded store; step 6 never writes it"
        )


def encode(event: dict[str, Any]) -> str:
    return json.dumps(event, allow_nan=False, separators=(",", ":"))


def produce(
    r,
    events: Iterable[dict[str, Any]],
    throttled: bool = False,
    stream: str = STREAM,
    keys: store.Keys = store.KEYS,
    sleep: Callable[[float], None] = time.sleep,
    batch: int = 1_000,
) -> int:
    """Append every event to the stream; returns the number produced."""
    refuse_loaded_store(r)
    pause, every = throttle_settings()
    produced_key = keys.replay[0]
    n = 0
    pipe = r.pipeline(transaction=False)
    for event in events:
        if event["seq"] != n + 1:
            raise ValueError(f"event {event['seq']} out of order after {n}")
        pipe.xadd(stream, {"e": encode(event)})
        n += 1
        if n % batch == 0:
            pipe.set(produced_key, n)
            pipe.execute()
        if throttled and n % every == 0:
            sleep(pause)
    pipe.set(produced_key, n)
    pipe.execute()
    return n
