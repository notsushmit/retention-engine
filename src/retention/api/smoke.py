"""smoke --snapshot: the step 5f smoke check of the loaded store. Not a mark.

configs/serving.yaml, step5.smoke_5f: not a mark; its numbers are never quoted as
parity or latency. The service is started in-process (the app and its lifespan,
with no network port) on the loaded store, and each endpoint is called once:

- GET /health;
- POST /score for one customer of each kind, chosen by a fixed rule so a rerun
  would choose the same ones: the top of the 30-day ranking (scored), the first
  outside_population and the first not_active customer in sorted ID order, and
  an ID that cannot exist (404);
- GET /at-risk, the default page and then with auto_renew=on (the first
  filtered call builds the index, which is timed);
- GET /alerts.

Each answer is compared with what it should be (status code, answer kind, row
count, the knowable alert count). The never-used scan runs on every body with
amendment A6's rule, recorded as an observation (the step 8 sample is the mark).
Also recorded: the index build time; the time to hash plans.parquet again and the
time of the whole readiness hash check (both pinned files and the model files),
median and maximum over REPEATS calls; Valkey's used memory, swap and free memory
before and after. No customer ID is written to the report, and no request is
timed (latency is step 8's).
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from retention.api import at_risk, store
from retention.api import service as svc
from retention.api.config import file_sha256, load_serving
from retention.api.schemas import never_used_hits, response_hits

REPEATS = 20
UNKNOWN = "smoke-check:not-a-customer"  # ":" and "-" are never in a KKBox ID
NOT_A_MARK = load_serving()["step5"]["smoke_5f"]["status"]


# --- the machine ---------------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def machine(r) -> dict[str, Any]:
    """Swap in use (MB), free memory (%), Valkey's used memory and the Docker VM's memory."""
    swap = _run(["sysctl", "vm.swapusage"])
    used = re.search(r"used = ([\d.]+)M", swap or "")
    pressure = _run(["memory_pressure"])
    free = re.search(r"free percentage: (\d+)%", pressure or "")
    vm = _run(["docker", "info", "--format", "{{.MemTotal}}"])
    return {
        "swap_used_mb": float(used.group(1)) if used else None,
        "free_memory_percent": int(free.group(1)) if free else None,
        "valkey_used_memory_bytes": store.memory_used(r),
        "docker_vm_memory_bytes": int(vm) if vm and vm.strip().isdigit() else None,
    }


def timed(fn: Callable[[], Any], repeats: int = REPEATS) -> dict[str, Any]:
    ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return {"repeats": repeats, "median_ms": statistics.median(ms), "max_ms": max(ms)}


# --- the customers -------------------------------------------------------------------------------


def pick_customers(r, keys: store.Keys = store.KEYS) -> dict[str, str | None]:
    """One customer of each kind, by a fixed rule (see the module note)."""
    top = store.ranking(r, 30, 0, 1)
    known = sorted(m.decode() for m in r.smembers(keys.known))
    picks: dict[str, str | None] = {
        "scored": top[0][0] if top else None,
        "outside_population": None,
        "not_active": None,
    }
    for i in range(0, len(known), store.CHUNK):
        part = known[i : i + store.CHUNK]
        for m, raw in zip(part, r.mget([keys.customer_key(m) for m in part]), strict=True):
            if raw is None:
                picks["not_active"] = picks["not_active"] or m
            elif json.loads(raw)["status"] == "outside_population":
                picks["outside_population"] = picks["outside_population"] or m
        if picks["outside_population"] and picks["not_active"]:
            break
    if r.sismember(keys.known, UNKNOWN):
        raise ValueError("the made-up ID is a known customer")
    return picks


# --- the check -----------------------------------------------------------------------------------


def run(
    load: Callable[[], svc.Context],
    connect: Callable[[], Any],
    snapshot: date,
    expected_knowable: int,
) -> dict[str, Any]:
    """Start the app on the loaded store, call each endpoint once, record and compare."""
    from fastapi.testclient import TestClient  # the test client needs httpx (dev group)

    from retention.api.app import create_app

    app = create_app(load=load, connect=connect)
    calls, unexpected = [], []
    with TestClient(app, raise_server_exceptions=False) as c:
        ctx, r = app.state.ctx, app.state.valkey
        before = machine(r)
        stored_date = store.store_date(r)
        if stored_date != snapshot:
            unexpected.append(f"the store's date is {stored_date}, not {snapshot}")
            return {"label": NOT_A_MARK, "unexpected": unexpected, "machine_before": before}
        builds: list[float] = []

        def timed_build(rr, t):
            t0 = time.perf_counter()
            idx = at_risk.build_index(rr, t)
            builds.append(time.perf_counter() - t0)
            return idx

        ctx.at_risk_index = at_risk.IndexCache(timed_build)
        picks = pick_customers(r)

        def record(name, resp, expected_status, own_id=None, check=None):
            entry = {
                "call": name,
                "status": resp.status_code,
                "expected_status": expected_status,
                "scan_hits": len(response_hits(resp.text, resp.status_code, own_id)),
                "body_bytes": len(resp.content),
            }
            ok = resp.status_code == expected_status
            if ok and check is not None:
                found, want = check(resp.json())
                entry |= {"found": found, "expected": want}
                ok = found == want
            if not ok:
                unexpected.append(name)
            calls.append(entry)
            return resp

        record("GET /health", c.get("/health"), 200, check=lambda b: (b["ready"], True))
        for kind in ("scored", "outside_population", "not_active"):
            m = picks[kind]
            if m is None:
                unexpected.append(f"no {kind} customer in the store")
                continue
            resp = c.post("/score", json={"msno": m})
            record(f"POST /score {kind}", resp, 200, m, lambda b, k=kind: (b["status"], k))
        record("POST /score unknown", c.post("/score", json={"msno": UNKNOWN}), 404)
        page = record(
            "GET /at-risk default",
            c.get("/at-risk"),
            200,
            check=lambda b: (len(b["rows"]), min(50, b["matching"])),
        )
        filtered = record("GET /at-risk auto_renew=on", c.get("/at-risk?auto_renew=on"), 200)
        record(
            "GET /alerts",
            c.get("/alerts"),
            200,
            check=lambda b: (b["trend_knowable"], expected_knowable),
        )
        # /at-risk rows, one by one, against the ID listed at their rank (A6)
        listed = {
            "GET /at-risk default": [m for m, _ in store.ranking(r, 30, 0, 50)],
            "GET /at-risk auto_renew=on": [
                m
                for m, _ in ctx.at_risk_index.get(r, snapshot).rankings[30]
                if ctx.at_risk_index.get(r, snapshot).auto_renew[m] == "on"
            ][:50],
        }
        for name, resp in (
            ("GET /at-risk default", page),
            ("GET /at-risk auto_renew=on", filtered),
        ):
            body = resp.json()
            header = {k: v for k, v in body.items() if k != "rows"}
            hits = len(never_used_hits(json.dumps(header)))
            for row, own in zip(body["rows"], listed[name], strict=True):
                hits += len(response_hits(json.dumps(row), 200, own))
            next(x for x in calls if x["call"] == name)["scan_hits"] = hits
        timings = {
            "index_build_seconds": builds[0] if builds else None,
            "index_builds": len(builds),
            "plans_parquet_hash": timed(lambda: file_sha256(ctx.pinned.pins["plans"][0])),
            "readiness_hash_check": timed(
                lambda: (svc.model_problem(ctx), svc.pin_problems(ctx.pinned.pins))
            ),
        }
        after = machine(r)
    unexpected += [x["call"] for x in calls if x["scan_hits"]]
    return {
        "label": NOT_A_MARK,
        "snapshot": snapshot.isoformat(),
        "customer_rule": "top of the 30-day ranking; first outside and not_active by sorted ID",
        "calls": calls,
        "timings": timings,
        "machine_before": before,
        "machine_after": after,
        "unexpected": sorted(set(unexpected)),
        "answered_as_expected": not unexpected,
    }
