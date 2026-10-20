"""Valkey-backed tests: a fixture, and a hook that fails a skipped one in CI.

configs/serving.yaml, step5.tests: tests marked `valkey` need a reachable Valkey
(VALKEY_URL). Locally they skip only with a stated reason when it cannot be
reached; in CI (CI=true, as GitHub Actions sets it) a skipped Valkey-backed test
fails the run. Tests use database 15, flushed before and after each test, so the
store's own database is never touched.
"""

from __future__ import annotations

import os

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


@pytest.fixture()
def valkey():
    """A client on the test database, flushed before and after; skips with a reason if
    Valkey cannot be reached (which fails the run in CI)."""
    import redis

    url = os.environ.get(URL_ENV) or DEFAULT_URL
    r = redis.Redis.from_url(url, db=TEST_DB)
    try:
        r.ping()
    except Exception as e:  # noqa: BLE001 (any connection failure is a reason to skip)
        pytest.skip(f"Valkey not reachable at {url} ({type(e).__name__}); start it to run")
    r.flushdb()
    yield r
    r.flushdb()
