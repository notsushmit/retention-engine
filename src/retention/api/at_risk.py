"""GET /at-risk: scored customers by final probability at one horizon, a page at a time.

The ranking is the one stored at snapshot load (configs/serving.yaml, step5: final
probability descending, ties by msno ascending); this module never ranks. Without
a filter a page is read straight from the stored ranking. With a filter
(auto_renew, in_plan; nothing else), the filter is applied to the full stored
ranking before the page is cut, from an in-memory index built once per store
date: the three stored rankings and each scored customer's auto_renew.

Plan membership is a batch fact: the Phase 7 plan for the store's date, cell
optimizer, base scenario, budget 5,000, read from the pinned plans file. Every
plan field carries "plan built from the 30-day probability before the ordering
fix"; a date without a plan reads "no plan for this date". auto_renew reads "on"
when the last transaction had auto-renew on, as Phase 7's customer table reads it.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from retention.api import schemas as sch
from retention.api import store
from retention.api.config import load_serving
from retention.api.labels import LABELS
from retention.explain.reasons import probability_text

CFG = load_serving()["api"]["at_risk"]
HORIZONS = tuple(CFG["horizon"]["allowed"])
FILTERS = tuple(CFG["filters"])
PARAMS = ("horizon", "limit", "offset", *FILTERS)
DIGITS = re.compile(r"^\d{1,9}$")
AUTO_RENEW = ("on", "off")
BOOLEAN = {"true": True, "false": False}
CHUNK = 5_000


class Malformed(ValueError):
    """422: a query the registered /at-risk parameters do not allow."""


@dataclass(frozen=True)
class Query:
    horizon: int
    limit: int
    offset: int
    auto_renew: str | None = None
    in_plan: bool | None = None

    @property
    def filtered(self) -> bool:
        return self.auto_renew is not None or self.in_plan is not None


def parse_query(items: Iterable[tuple[str, str]]) -> Query:
    """Only the registered parameters, each at most once, in their exact forms."""
    seen: dict[str, str] = {}
    for k, v in items:
        if k not in PARAMS or k in seen:
            raise Malformed(k)
        seen[k] = v

    def number(name: str, default: int) -> int:
        if name not in seen:
            return default
        if not DIGITS.match(seen[name]):
            raise Malformed(name)
        return int(seen[name])

    horizon = number("horizon", int(CFG["horizon"]["default"]))
    limit = number("limit", int(CFG["limit"]["default"]))
    offset = number("offset", 0)
    if horizon not in HORIZONS or not 1 <= limit <= int(CFG["limit"]["max"]):
        raise Malformed("horizon or limit")
    auto_renew = seen.get("auto_renew")
    if auto_renew is not None and auto_renew not in AUTO_RENEW:
        raise Malformed("auto_renew")
    in_plan = seen.get("in_plan")
    if in_plan is not None and in_plan not in BOOLEAN:
        raise Malformed("in_plan")
    return Query(horizon, limit, offset, auto_renew, None if in_plan is None else BOOLEAN[in_plan])


# --- the plan -------------------------------------------------------------------------------------


def plan_cell(plans: pd.DataFrame, cell: dict[str, Any] | None = None) -> dict[date, dict]:
    """{snapshot: {msno: action}} of the registered plan cell."""
    cell = CFG["plan_cell"] if cell is None else cell
    keep = (
        (plans.strategy == cell["strategy"])
        & (plans.scenario == cell["scenario"])
        & (plans.budget_label == f"{int(cell['budget']):,}")
    )
    sel = plans[keep]
    if sel.duplicated(["snapshot", "msno"]).any():
        raise ValueError("a customer appears twice in one plan")
    return {
        date.fromisoformat(t): dict(zip(g.msno.astype(str), g.action.astype(str), strict=True))
        for t, g in sel.groupby("snapshot")
    }


# --- the index, once per store date ---------------------------------------------------------------


def auto_renew_of(docs: Iterable[bytes | None]) -> list[str]:
    out = []
    for raw in docs:
        if raw is None:
            raise ValueError("a ranked customer has no stored document")
        row = json.loads(raw)["row"]
        out.append("on" if row.get("last_is_auto_renew") == 1 else "off")
    return out


def stored_auto_renew(r, msnos: list[str], keys: store.Keys = store.KEYS) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(msnos), CHUNK):
        part = msnos[i : i + CHUNK]
        docs = r.mget([keys.customer_key(m) for m in part])
        out |= dict(zip(part, auto_renew_of(docs), strict=True))
    return out


@dataclass
class Index:
    store_date: date
    rankings: dict[int, list[tuple[str, float]]]  # the full stored ranking per horizon
    auto_renew: dict[str, str]


def build_index(r, store_date: date, keys: store.Keys = store.KEYS) -> Index:
    rankings = {h: store.ranking(r, h, 0, r.zcard(keys.rank_key(h)), keys) for h in HORIZONS}
    msnos = [m for m, _ in rankings[HORIZONS[0]]]
    return Index(store_date, rankings, stored_auto_renew(r, msnos, keys))


class IndexCache:
    """The index of the current store date; rebuilt only when the store date moves."""

    def __init__(self, build: Callable[[Any, date], Index] = build_index):
        self._build = build
        self._lock = threading.Lock()
        self._index: Index | None = None

    def get(self, r, store_date: date) -> Index:
        with self._lock:
            if self._index is None or self._index.store_date != store_date:
                self._index = self._build(r, store_date)
            return self._index


# --- one page -------------------------------------------------------------------------------------


def header_counts(stored: dict[str, int]) -> sch.AtRiskCounts:
    prefix = "outside_population_"
    return sch.AtRiskCounts(
        scored=stored["scored"],
        outside_population={
            k.removeprefix(prefix): v for k, v in stored.items() if k.startswith(prefix)
        },
        not_active=stored["not_active"],
    )


def page(
    r, store_date: date, q: Query, plan: dict[str, str] | None, cache: IndexCache
) -> dict[str, Any]:
    """One /at-risk page; filters are applied to the full stored ranking first."""
    if q.in_plan is not None and plan is None:
        raise Malformed("in_plan: no plan for this date")
    if q.filtered:
        idx = cache.get(r, store_date)

        def keep(m: str) -> bool:
            if q.auto_renew is not None and idx.auto_renew[m] != q.auto_renew:
                return False
            return q.in_plan is None or (m in plan) == q.in_plan

        matched = [(m, p) for m, p in idx.rankings[q.horizon] if keep(m)]
        total, rows = len(matched), matched[q.offset : q.offset + q.limit]
    else:
        total = int(r.zcard(store.KEYS.rank_key(q.horizon)))
        rows = store.ranking(r, q.horizon, q.offset, q.limit)
    auto = stored_auto_renew(r, [m for m, _ in rows])
    out_rows = []
    for i, (m, p) in enumerate(rows):
        plan_field = None
        if plan is not None:
            plan_field = sch.PlanField(
                in_plan=m in plan, action=plan.get(m), label=LABELS.plan_label
            )
        out_rows.append(
            sch.AtRiskRow(
                rank=q.offset + i + 1,
                msno=m,
                probability=p,
                text=probability_text(p),
                grade=LABELS.grade_59 if q.horizon == 59 else None,
                auto_renew=auto[m],
                plan=plan_field,
            )
        )
    body = sch.AtRiskPage(
        store_date=store_date,
        horizon=q.horizon,
        offset=q.offset,
        limit=q.limit,
        filters=sch.AtRiskFilters(auto_renew=q.auto_renew, in_plan=q.in_plan),
        matching=total,
        header_note=LABELS.at_risk_header_note,
        plan_status=LABELS.no_plan_status if plan is None else LABELS.plan_label,
        counts=header_counts(store.counts(r)),
        rows=tuple(out_rows),
    )
    return body.model_dump(mode="json")
