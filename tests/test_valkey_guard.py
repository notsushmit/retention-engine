"""The Valkey test fixture really uses database 15 (D-211); nothing here writes database 0."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import redis

import valkey_guard
from retention.api import store


@pytest.mark.parametrize(
    "url, forced",
    [
        ("redis://localhost:6379/0", "redis://localhost:6379/15"),
        ("redis://localhost:6379", "redis://localhost:6379/15"),
        ("redis://127.0.0.1:6379/3?db=0", "redis://127.0.0.1:6379/15"),
        ("redis://h:1/0?socket_timeout=2&db=4", "redis://h:1/15?socket_timeout=2"),
    ],
)
def test_the_test_url_forces_database_15(url, forced):
    assert valkey_guard.test_url(url) == forced
    assert redis.Redis.from_url(forced).connection_pool.connection_kwargs["db"] == 15


def test_a_client_built_the_old_way_is_on_database_0():
    """The cause of D-211: the URL's database wins over the db argument. No connection is made."""
    old = redis.Redis.from_url("redis://localhost:6379/0", db=15)
    assert old.connection_pool.connection_kwargs["db"] == 0


def test_a_client_on_another_database_fails_the_test():
    other = SimpleNamespace(client_info=lambda: {"db": 0})
    with pytest.raises(pytest.fail.Exception, match="on database 0, not 15"):
        valkey_guard.check_database(other)


@pytest.mark.valkey
def test_the_fixture_is_on_database_15_by_the_server_s_own_account(valkey):
    assert int(valkey.client_info()["db"]) == 15
    assert valkey.connection_pool.connection_kwargs["db"] == 15


# --- the audit: which database each URL-built client uses (no connection made) ---------------


def test_the_service_store_client_uses_the_url_s_database(monkeypatch):
    """store.client (the service, load-snapshot and smoke) takes VALKEY_URL as given: database 0
    unless the URL names another; it is never pointed at database 15 by the tests' fixture."""
    monkeypatch.delenv("VALKEY_URL", raising=False)
    assert store.client().connection_pool.connection_kwargs["db"] == 0
    monkeypatch.setenv("VALKEY_URL", "redis://localhost:6379/1")
    assert store.client().connection_pool.connection_kwargs["db"] == 1
