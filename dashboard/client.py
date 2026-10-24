"""The dashboard's only data source: the scoring API over HTTP (configs/dashboard.yaml source).

The client refuses any host other than loopback, allows only the registered calls,
and uses the registered timeouts (10 s; 30 s for GET /summary, C1). An answer other
than 200 becomes ApiError with the API's own detail and reason; GET /health also
returns its 503 body, which says why the service is not ready.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from dashboard.settings import Settings, load

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str, reason: str | None = None):
        super().__init__(f"{status} {detail}" + (f": {reason}" if reason else ""))
        self.status, self.detail, self.reason = status, detail, reason


class Refused(ValueError):
    """A call or a host the registered source does not allow."""


def check_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in LOOPBACK:
        raise Refused(f"the API URL must be on loopback: {url!r}")
    return url.rstrip("/")


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or load()
        s = self.settings
        self.base_url = check_url(base_url or os.environ.get(s.api_env) or s.api_default)
        self._http = httpx.Client(base_url=self.base_url, transport=transport)

    def timeout_for(self, path: str) -> float:
        return self.settings.summary_timeout if path == "/summary" else self.settings.timeout

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if (method, path) not in self.settings.calls:
            raise Refused(f"{method} {path} is not a registered call")
        r = self._http.request(method, path, timeout=self.timeout_for(path), **kwargs)
        body = r.json()
        if r.status_code == 200 or (path == "/health" and r.status_code == 503 and "ready" in body):
            return body
        raise ApiError(r.status_code, str(body.get("detail", "")), body.get("reason"))

    def health(self) -> dict[str, Any]:
        return self.call("GET", "/health")

    def at_risk(self, **params: Any) -> dict[str, Any]:
        return self.call(
            "GET", "/at-risk", params={k: v for k, v in params.items() if v is not None}
        )

    def alerts(self) -> dict[str, Any]:
        return self.call("GET", "/alerts")

    def summary(self) -> dict[str, Any]:
        return self.call("GET", "/summary")

    def plans(self) -> dict[str, Any]:
        return self.call("GET", "/plans")

    def model(self) -> dict[str, Any]:
        return self.call("GET", "/model")

    def drift(self) -> dict[str, Any]:
        return self.call("GET", "/drift")

    def score(self, msno: str) -> dict[str, Any]:
        return self.call("POST", "/score", json={"msno": msno})
