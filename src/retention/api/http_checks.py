"""Step 8's checks over HTTP against the running stack (configs/serving.yaml, step8).

Every function takes the client it calls (httpx.Client against the stack, or FastAPI's
test client in tests) so the code the tests run is the code the real runs use. Reports
hold counts, timings and field names only, never a customer ID.

- sample / warmup: the registered seeded draws from the sorted scored IDs.
- parity: each POST /score body over HTTP against svc.answer on the host, both as
  canonical JSON (sorted keys); the strings must be equal (1,000 of 1,000).
- latency: warm-up calls, then the sample, sequential, timed around sending the request
  and reading the whole body; p50, p95, p99, mean and max in milliseconds.
- concurrency: the sample's sequential answers, then the same requests from several
  threads at once, every answer equal and no 5xx; GET /at-risk alike.
- december: POST /score with an as_of in December 2016 answers 409 with no reasons and
  passes the never-used-word scan.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from retention.api import schemas as sch
from retention.api import service as svc
from retention.api import store
from retention.api.config import load_serving

STEP8 = load_serving()["step8"]
LATENCY = load_serving()["streaming"]["pass_marks"]["latency"]
HTTP_SAMPLE = load_serving()["streaming"]["pass_marks"]["score_parity"]["http_sample"]
DECEMBER_AS_OF = "2016-12-01"


def registered_seed(text: str) -> int:
    """The seed written in a registered draw, numpy.random.default_rng(<seed>)."""
    found = re.findall(r"default_rng\((\d+)\)", text)
    if len(found) != 1:
        raise ValueError(f"expected one registered seed in {text!r}")
    return int(found[0])


SAMPLE_SEED = registered_seed(STEP8["http_score_parity"]["sample"])
WARMUP_SEED = registered_seed(STEP8["latency"]["warmup"])
SAMPLE_SIZE = int(HTTP_SAMPLE["customers"])
WARMUP_SIZE = int(LATENCY["warmup_calls"])
P95_MS_MAX = float(LATENCY["p95_ms_max"])


# --- the draws ----------------------------------------------------------------------------------


def scored_ids(r) -> list[str]:
    """Every scored customer in the store (the ranked ones), sorted."""
    return sorted(m.decode() for m in r.zrange(store.KEYS.rank_key(30), 0, -1))


def sample(ids: list[str], n: int | None = None, seed: int = SAMPLE_SEED) -> list[str]:
    """n IDs (the registered 1,000 by default) drawn without replacement from the sorted
    IDs, in the draw's order."""
    n = SAMPLE_SIZE if n is None else n
    pool = sorted(ids)
    picked = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in picked]


def warmup(
    ids: list[str], drawn: list[str], n: int | None = None, seed: int = WARMUP_SEED
) -> list[str]:
    """n IDs (the registered 50 by default) from the scored IDs outside the sample, drawn
    with their own seed."""
    n = WARMUP_SIZE if n is None else n
    return sample(sorted(set(ids) - set(drawn)), n, seed)


# --- parity -------------------------------------------------------------------------------------


def canonical(body: Any) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def differing_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """The top-level fields whose canonical JSON differs (names only)."""
    keys = sorted(set(a) | set(b))
    return [k for k in keys if canonical(a.get(k)) != canonical(b.get(k))]


def parity(client, ctx, r, ids: list[str]) -> dict[str, Any]:
    """Each POST /score body against svc.answer on the host; counts and field names only."""
    statuses: Counter = Counter()
    fields: Counter = Counter()
    equal = 0
    for m in ids:
        resp = client.post("/score", json={"msno": m})
        statuses[resp.status_code] += 1
        host = svc.answer(ctx, r, svc.ScoreRequest(msno=m))
        got = resp.json()
        if resp.status_code == 200 and canonical(got) == canonical(host):
            equal += 1
        else:
            fields.update(differing_fields(got, host) if isinstance(got, dict) else ["body"])
    return {
        "customers": len(ids),
        "equal": equal,
        "differing": len(ids) - equal,
        "http_status": {str(k): v for k, v in sorted(statuses.items())},
        "differing_fields": dict(sorted(fields.items())),
        "met": equal == len(ids) == SAMPLE_SIZE,
    }


# --- latency ------------------------------------------------------------------------------------


def timed_post(client, msno: str) -> tuple[float, int]:
    """Milliseconds around sending POST /score and reading the whole body; the status."""
    t0 = time.perf_counter()
    resp = client.post("/score", json={"msno": msno})
    _ = resp.content
    return (time.perf_counter() - t0) * 1000.0, resp.status_code


def summarize(ms: list[float], limit: float | None = None) -> dict[str, Any]:
    limit = P95_MS_MAX if limit is None else limit
    a = np.asarray(ms, dtype=float)
    p50, p95, p99 = (float(x) for x in np.percentile(a, [50, 95, 99]))
    return {
        "calls": len(ms),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(float(a.mean()), 3),
        "max_ms": round(float(a.max()), 3),
        "p95_ms_max": limit,
        "met": p95 <= limit,
    }


def latency(client, warm: list[str], calls: list[str]) -> dict[str, Any]:
    """Warm-up calls (not timed in the result), then the timed sequential calls."""
    warm_status = Counter(timed_post(client, m)[1] for m in warm)
    timed = [timed_post(client, m) for m in calls]
    status = Counter(s for _, s in timed)
    out = summarize([t for t, _ in timed])
    out |= {
        "warmup_calls": len(warm),
        "warmup_http_status": {str(k): v for k, v in sorted(warm_status.items())},
        "http_status": {str(k): v for k, v in sorted(status.items())},
    }
    out["met"] = out["met"] and set(status) == {200} and len(calls) == SAMPLE_SIZE
    return out


# --- concurrency --------------------------------------------------------------------------------


def concurrency(
    client_for: Callable[[], Any], ids: list[str], threads: int, at_risk_calls: int
) -> dict[str, Any]:
    """Sequential answers first, then the same requests from `threads` threads at once."""
    seq_client = client_for()
    expected = {}
    for m in ids:
        r = seq_client.post("/score", json={"msno": m})
        expected[m] = (r.status_code, r.text)
    at_risk_ref = seq_client.get("/at-risk")
    at_risk_expected = (at_risk_ref.status_code, at_risk_ref.text)

    def score(m: str) -> tuple[str, int, str]:
        r = client_for().post("/score", json={"msno": m})
        return m, r.status_code, r.text

    def at_risk(_: int) -> tuple[int, str]:
        r = client_for().get("/at-risk")
        return r.status_code, r.text

    with ThreadPoolExecutor(max_workers=threads) as pool:
        scored = list(pool.map(score, ids))
        pages = list(pool.map(at_risk, range(at_risk_calls)))
    differing = sum((s, t) != expected[m] for m, s, t in scored)
    server_errors = sum(s >= 500 for _, s, _ in scored) + sum(s >= 500 for s, _ in pages)
    at_risk_differing = sum(p != at_risk_expected for p in pages)
    return {
        "threads": threads,
        "score_calls": len(scored),
        "score_differing_from_sequential": differing,
        "at_risk_calls": len(pages),
        "at_risk_differing_from_sequential": at_risk_differing,
        "server_errors": server_errors,
        "met": differing == 0 and at_risk_differing == 0 and server_errors == 0,
    }


# --- December 2016 over HTTP ------------------------------------------------------------------


def december(client, ids: list[str]) -> dict[str, Any]:
    """POST /score with as_of in December 2016 for the first sample ID whose own text holds
    no never-used word: 409, no reasons anywhere in the body, and a clean scan."""
    m = next(x for x in ids if not sch.never_used_hits(x))
    resp = client.post("/score", json={"msno": m, "as_of": DECEMBER_AS_OF})
    text = resp.text
    hits = sch.response_hits(text, resp.status_code, m)
    reasons = '"reasons' in text
    return {
        "as_of": DECEMBER_AS_OF,
        "http_status": resp.status_code,
        "reasons_in_body": reasons,
        "scan_hits": len(hits),
        "met": resp.status_code == 409 and not reasons and not hits,
    }


# --- the registry and the config after startup -------------------------------------------------


def tree_sha256(folder: Path) -> dict[str, Any]:
    """One SHA-256 over every file's relative path and SHA-256 under `folder`, sorted."""
    h = hashlib.sha256()
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(folder)).encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return {"files": len(files), "sha256": h.hexdigest()}


def config_matches(client, committed_sha256: str) -> dict[str, Any]:
    body = client.get("/health").json()
    return {
        "served_config_sha256": body.get("config_sha256"),
        "committed_config_sha256": committed_sha256,
        "met": body.get("config_sha256") == committed_sha256,
    }
