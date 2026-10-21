"""Valkey-backed tests: a fixture, and a hook that fails a skipped one in CI.

configs/serving.yaml, step5.tests: tests marked `valkey` need a reachable Valkey
(VALKEY_URL). Locally they skip only with a stated reason when it cannot be
reached; in CI (CI=true, as GitHub Actions sets it) a skipped Valkey-backed test
fails the run. Tests use database 15, flushed before and after each test, so the
store's own database is never touched.

The database is forced by rewriting the URL's path (and dropping any db query
parameter): redis-py lets a database in the URL override a db argument, which is
how every Valkey-backed test ran on database 0 until this fix (D-211). After
connecting, the fixture asks the server which database it is on and fails the
test unless it is 15.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

MARK = "valkey"
TEST_DB = 15
URL_ENV = "VALKEY_URL"
DEFAULT_URL = "redis://localhost:6379/0"


def in_ci() -> bool:
    return os.environ.get("CI", "").lower() == "true"


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", f"{MARK}: needs a reachable Valkey ({URL_ENV})")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker(MARK) is not None and in_ci():
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid}: a Valkey-backed test was skipped in CI; "
            "CI must run every Valkey-backed test"
        )


def test_url(url: str, db: int = TEST_DB) -> str:
    """The same server with the database forced: path /<db>, no db query parameter."""
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k != "db"])
    return urlunsplit(parts._replace(path=f"/{db}", query=query))


test_url.__test__ = False  # a helper, not a test


def check_database(r, db: int = TEST_DB) -> None:
    """Fail the test unless the server says this client is on `db`."""
    found = int(r.client_info()["db"])
    if found != db:
        pytest.fail(f"the Valkey test client is on database {found}, not {db}")


@pytest.fixture()
def valkey():
    """A client on the test database, flushed before and after; skips with a reason if
    Valkey cannot be reached (which fails the run in CI)."""
    import redis

    url = test_url(os.environ.get(URL_ENV) or DEFAULT_URL)
    r = redis.Redis.from_url(url)
    try:
        r.ping()
    except Exception as e:  # noqa: BLE001 (any connection failure is a reason to skip)
        pytest.skip(f"Valkey not reachable at {url} ({type(e).__name__}); start it to run")
    check_database(r)
    r.flushdb()
    yield r
    check_database(r)
    r.flushdb()
