"""The dashboard's API client: loopback only, registered calls only, registered timeouts."""

from __future__ import annotations

import httpx
import pytest

import dashboard.client as dc
from dashboard.client import ApiError, Client, Refused, check_url
from dashboard.settings import load
from dashboard_fake import FakeApi, client, not_ready

S = load()


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000", "http://localhost:8000/", "http://[::1]:8000"]
)
def test_loopback_urls_are_accepted(url):
    assert check_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    ["http://10.0.0.5:8000", "http://example.com", "ftp://127.0.0.1", "http://127.0.0.1.evil.io"],
)
def test_any_other_host_is_refused(url):
    with pytest.raises(Refused):
        check_url(url)


def test_the_env_url_is_checked_too(monkeypatch):
    monkeypatch.setenv(S.api_env, "http://192.168.1.4:8000")
    with pytest.raises(Refused):
        Client(transport=httpx.MockTransport(FakeApi()))


def test_the_default_is_the_registered_loopback_url(monkeypatch):
    monkeypatch.delenv(S.api_env, raising=False)
    assert Client(transport=httpx.MockTransport(FakeApi())).base_url == "http://127.0.0.1:8000"


def test_only_the_registered_calls_are_allowed():
    """The seven of source.calls and C8's GET /drift (ops_dashboard.calls_added)."""
    assert sorted(S.calls) == sorted(
        [
            ("GET", "/health"),
            ("GET", "/at-risk"),
            ("GET", "/alerts"),
            ("GET", "/summary"),
            ("GET", "/plans"),
            ("GET", "/model"),
            ("POST", "/score"),
            ("GET", "/drift"),
        ]
    )
    c = client()
    for method, path in [("GET", "/score"), ("POST", "/health"), ("GET", "/nowhere")]:
        with pytest.raises(Refused):
            c.call(method, path)


def test_summary_waits_30_seconds_and_every_other_call_10():
    api = FakeApi()
    c = client(api)
    for name in ("health", "alerts", "summary", "plans", "model"):
        getattr(c, name)()
    c.at_risk(horizon=30)
    c.score("fakeid00AbCdEfGh")
    timeouts = {r.url.path: r.extensions["timeout"]["read"] for r in api.requests}
    assert timeouts["/summary"] == 30.0
    assert {p: t for p, t in timeouts.items() if p != "/summary"} == {
        p: 10.0 for p in ("/health", "/alerts", "/plans", "/model", "/at-risk", "/score")
    }


def test_an_error_carries_the_apis_detail_and_reason():
    c = client(FakeApi(not_ready()))
    with pytest.raises(ApiError) as e:
        c.summary()
    assert e.value.status == 503
    assert e.value.reason == "no snapshot is loaded (the store has no date)"
    assert c.health()["ready"] is False  # /health returns its 503 body


def test_a_404_is_an_error_with_the_registered_detail():
    with pytest.raises(ApiError) as e:
        client().score("nobody")
    assert (e.value.status, e.value.detail) == (404, "unknown customer")


def test_at_risk_drops_unset_filters():
    api = FakeApi()
    client(api).at_risk(horizon=7, limit=50, offset=0, auto_renew=None, in_plan="true")
    assert dict(api.requests[0].url.params) == {
        "horizon": "7",
        "limit": "50",
        "offset": "0",
        "in_plan": "true",
    }


# --- the compose service host (C8, reinstated by A16) ---


def test_loopback_is_the_default_and_the_service_host_is_refused_without_the_variable(monkeypatch):
    monkeypatch.delenv("RETENTION_API_ALLOW_HOST", raising=False)
    assert dc.allowed_hosts() == dc.LOOPBACK
    assert dc.check_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    with pytest.raises(Refused):
        dc.check_url("http://api:8000", dc.allowed_hosts())


def test_the_service_host_is_accepted_only_when_the_variable_is_exactly_api(monkeypatch):
    monkeypatch.setenv("RETENTION_API_ALLOW_HOST", "api")
    hosts = dc.allowed_hosts()
    assert hosts == dc.LOOPBACK | {"api"}
    assert dc.check_url("http://api:8000", hosts) == "http://api:8000"
    assert dc.check_url("http://localhost:8000", hosts)  # loopback stays allowed
    for third in ("http://example.com:8000", "http://api.example.com:8000", "http://10.0.0.5:8000"):
        with pytest.raises(Refused):
            dc.check_url(third, hosts)


@pytest.mark.parametrize(
    "value", ["", "API", "api.example.com", "api,localhost", "10.0.0.5", " api"]
)
def test_any_other_value_of_the_variable_stops_startup(monkeypatch, value):
    monkeypatch.setenv("RETENTION_API_ALLOW_HOST", value)
    with pytest.raises(Refused, match="must be unset or exactly 'api'"):
        Client("http://127.0.0.1:8000", transport=httpx.MockTransport(FakeApi()))


def test_a_third_host_is_refused_with_and_without_the_variable(monkeypatch):
    for value in (None, "api"):
        if value is None:
            monkeypatch.delenv("RETENTION_API_ALLOW_HOST", raising=False)
        else:
            monkeypatch.setenv("RETENTION_API_ALLOW_HOST", value)
        with pytest.raises(Refused):
            Client("http://example.com:8000", transport=httpx.MockTransport(FakeApi()))


def test_the_client_built_for_the_compose_service_reaches_the_api_host(monkeypatch):
    monkeypatch.setenv("RETENTION_API_ALLOW_HOST", "api")
    monkeypatch.setenv("RETENTION_API_URL", "http://api:8000")
    c = Client(transport=httpx.MockTransport(FakeApi()))
    assert c.base_url == "http://api:8000"
    assert c.health()["ready"] is True
