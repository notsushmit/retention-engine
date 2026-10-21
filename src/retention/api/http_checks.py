"""Step 8's checks over HTTP against the running stack (configs/serving.yaml, step8).

Every function takes the client it calls (httpx.Client against the stack, or FastAPI's
test client in tests) so the code the tests run is the code the real runs use. Reports
hold counts, timings and field names only, never a customer ID.

- sample / warmup: the registered seeded draws from the sorted scored IDs.
- parity: each POST /score body over HTTP against svc.answer on the host. Judged under
  amendment A10 (configs/serving.yaml, http_parity_a10; found after a result): every
  non-float field exact, every float within relative 1e-12; the count of bodies equal as
  canonical JSON strings (the registered wording) is reported beside it.
- latency: warm-up calls, then the sample, sequential, timed around sending the request
  and reading the whole body; p50, p95, p99, mean and max in milliseconds.
- concurrency: the sample's sequential answers, then the same requests from several
  threads at once, every answer equal and no 5xx; GET /at-risk alike.
- december: POST /score with an as_of in December 2016 answers 409 with no reasons and
  passes the never-used-word scan.
- diagnose (D-244, a diagnostic found after a result, not a mark): the container and the
  host twice each on the same sample; each side against itself, and per schema path the
  largest gaps (absolute, relative, ulps) where they differ; counts only.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
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
A10 = load_serving()["http_parity_a10"]
A10_TOLERANCE = float(A10["judged"]["float_relative_tolerance"])
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


def a10_failures(a: Any, b: Any, tolerance: float | None = None) -> list[str]:
    """The schema paths failing A10: a leaf on one side only, a non-float leaf that differs
    (texts, the action name, flags, codes, statuses), or two floats further apart than
    the relative tolerance."""
    tolerance = A10_TOLERANCE if tolerance is None else tolerance
    la, lb = leaves(a), leaves(b)
    bad = []
    for path in sorted(set(la) | set(lb)):
        if path not in la or path not in lb:
            bad.append(path)
            continue
        x, y = la[path], lb[path]
        if is_float(x) and is_float(y):
            if abs(x - y) > tolerance * max(abs(x), abs(y)):
                bad.append(path)
        elif canonical(x) != canonical(y):
            bad.append(path)
    return bad


def parity(client, ctx, r, ids: list[str]) -> dict[str, Any]:
    """Each POST /score body against svc.answer on the host, judged under A10, with the
    count of bodies equal as strings beside it; counts, sizes and field names only."""
    statuses: Counter = Counter()
    fields: Counter = Counter()
    failing: Counter = Counter()
    equal = passed = float_gap = 0
    worst_ulps, worst_rel = 0, 0.0
    for m in ids:
        resp = client.post("/score", json={"msno": m})
        statuses[resp.status_code] += 1
        host = svc.answer(ctx, r, svc.ScoreRequest(msno=m))
        got = resp.json()
        if resp.status_code == 200 and canonical(got) == canonical(host):
            equal += 1
            passed += 1
            continue
        fields.update(differing_fields(got, host) if isinstance(got, dict) else ["body"])
        bad = a10_failures(got, host) if resp.status_code == 200 else ["http_status"]
        failing.update(bad)
        passed += not bad
        gaps = leaf_gaps([(got, host)]) if isinstance(got, dict) else {}
        floats = [g for g in gaps.values() if "max_ulps" in g]
        float_gap += bool(floats)
        worst_ulps = max([worst_ulps, *(g["max_ulps"] for g in floats)])
        worst_rel = max([worst_rel, *(g["max_rel_gap"] for g in floats)])
    return {
        "customers": len(ids),
        "equal": equal,
        "differing": len(ids) - equal,
        "http_status": {str(k): v for k, v in sorted(statuses.items())},
        "differing_fields": dict(sorted(fields.items())),
        "registered_wording": A10["registered_wording"],
        "a10_float_relative_tolerance": A10_TOLERANCE,
        "a10_passed": passed,
        "a10_failing_paths": dict(sorted(failing.items())),
        "customers_with_a_float_gap": float_gap,
        "largest_ulps": worst_ulps,
        "largest_relative_gap": worst_rel,
        "met": passed == len(ids) == SAMPLE_SIZE,
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


# --- the parity diagnosis (D-244, found after a result: HTTP parity 996 of 1,000) -------------


def leaves(body: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a JSON body by its schema path (list items by position)."""
    if isinstance(body, dict):
        out: dict[str, Any] = {}
        for k, v in body.items():
            out |= leaves(v, f"{prefix}{k}.")
        return out
    if isinstance(body, list | tuple):
        out = {}
        for i, v in enumerate(body):
            out |= leaves(v, f"{prefix}{i}.")
        return out
    return {prefix.rstrip("."): body}


def ordered_bits(x: float) -> int:
    """The float's bits as an integer that orders like the float (for distances in ulps)."""
    i = struct.unpack("<q", struct.pack("<d", x))[0]
    return -(i & 0x7FFFFFFFFFFFFFFF) if i < 0 else i


def ulps(a: float, b: float) -> int:
    return abs(ordered_bits(a) - ordered_bits(b))


def is_float(v: Any) -> bool:
    return isinstance(v, float) and not isinstance(v, bool)


def leaf_gaps(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per schema path, over every (a, b) pair: how many differ; for floats the largest
    absolute and relative gap and the largest distance in ulps. Values are never kept."""
    out: dict[str, dict[str, Any]] = {}
    for a, b in pairs:
        la, lb = leaves(a), leaves(b)
        for path in sorted(set(la) | set(lb)):
            x, y = la.get(path), lb.get(path)
            if canonical(x) == canonical(y):
                continue
            row = out.setdefault(path, {"differing": 0})
            row["differing"] += 1
            if is_float(x) and is_float(y):
                gap = abs(x - y)
                rel = gap / max(abs(x), abs(y)) if max(abs(x), abs(y)) else 0.0
                row["max_abs_gap"] = max(row.get("max_abs_gap", 0.0), gap)
                row["max_rel_gap"] = max(row.get("max_rel_gap", 0.0), rel)
                row["max_ulps"] = max(row.get("max_ulps", 0), ulps(x, y))
            else:
                row["kind"] = "not a float pair"
    return dict(sorted(out.items()))


PROBABILITY_PATHS = (
    "risk.days_7.probability",
    "risk.days_30.probability",
    "risk.days_59.probability",
    "reasons.0.probability_before_fix",
    "reasons.1.probability_before_fix",
)
ACTION_BASIS = "reasons.1.probability_before_fix"  # the 30-day probability before the fix


def diagnose(client, ctx, r, ids: list[str]) -> dict[str, Any]:
    """The container (HTTP) and the host (svc.answer), twice each, on `ids`: counts and gap
    sizes by schema path only; no ID and no customer's value."""
    http, host = [], []
    for _ in range(2):
        http.append({m: client.post("/score", json={"msno": m}).json() for m in ids})
        host.append({m: svc.answer(ctx, r, svc.ScoreRequest(msno=m)) for m in ids})
    http_self = sum(canonical(http[0][m]) != canonical(http[1][m]) for m in ids)
    host_self = sum(canonical(host[0][m]) != canonical(host[1][m]) for m in ids)
    differ = [m for m in ids if canonical(http[0][m]) != canonical(host[0][m])]
    pairs = [(http[0][m], host[0][m]) for m in differ]
    gaps = leaf_gaps(pairs)
    reason_positions = {
        str(i): sum(any(p.startswith(f"reasons.{i}.") for p in leaf_gaps([pair])) for pair in pairs)
        for i in (0, 1)
    }
    action_changed = [
        pair
        for pair in pairs
        if canonical(pair[0]["action"]["highest_net_value_action"])
        != canonical(pair[1]["action"]["highest_net_value_action"])
    ]
    follows = [any(p in leaf_gaps([pair]) for p in PROBABILITY_PATHS) for pair in action_changed]
    basis = [ACTION_BASIS in leaf_gaps([pair]) for pair in action_changed]
    return {
        "customers": len(ids),
        "http_differs_from_itself": http_self,
        "host_differs_from_itself": host_self,
        "http_differs_from_host": len(differ),
        "gaps_by_path": gaps,
        "reason_positions_changed": reason_positions,
        "action_changed": len(action_changed),
        "action_changed_with_a_probability_gap": sum(follows),
        "action_changed_with_a_30_day_pre_fix_gap": sum(basis),
    }
