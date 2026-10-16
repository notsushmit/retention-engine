"""The comparison strategies, scored by the same assumed effects as the optimizer.

Every figure here is arithmetic on assumed effects (configs/actions.yaml): the
optimizer maximizes this quantity by design, so the comparison is not evidence
that any strategy or action works.

- highest_risk: customers ranked by calibrated 30-day probability.
- highest_assumed_uplift_x_value: ranked by the largest assumed effect x value
  over the customer's allowed actions (the uplift is assumed, not estimated).
- random: seeded shuffles (seed + draw number).
- best-action variants give each targeted customer their allowed action with
  the highest net value, even when negative (cheaper, then action order, on ties).
- fixed_action: the reminder to every customer it is allowed for, ranked by risk.
- no_action: nobody.
Fill: walk the ranking; take a customer when their action fits the remaining
budget (and the contact limit); otherwise skip and continue. With no budget limit,
each baseline targets as many customers as the optimizer chose. Ranking ties:
score descending, then msno ascending.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from retention.optimize.effects import Options
from retention.optimize.solve import NONE, Plan

METRICS = (
    "expected_retained_customers",
    "expected_retained_value",
    "expected_cost",
    "net_value",
    "contacted",
)


def best_action(opts: Options) -> np.ndarray:
    """Per customer, the allowed action with the highest net value (NONE if none allowed)."""
    net = np.where(opts.allowed, opts.net, -np.inf)
    out = np.full(opts.n, NONE, dtype=np.int64)
    for i in range(opts.n):
        if not opts.allowed[i].any():
            continue
        js = [j for j in range(len(opts.actions)) if opts.allowed[i, j]]
        out[i] = min(js, key=lambda j: (-net[i, j], opts.cost[i, j], j))
    return out


def ranking(score: np.ndarray, msno: np.ndarray) -> np.ndarray:
    """Indices by score descending, then msno ascending."""
    return np.lexsort((msno.astype(str), -score))


def fill(
    order: np.ndarray,
    action: np.ndarray,
    cost: np.ndarray,
    budget: float | None,
    max_count: int | None,
) -> np.ndarray:
    """Skip-and-continue fill along `order`; customers with action NONE are passed over."""
    order = order[action[order] != NONE]
    c = cost[order, action[order]]
    take = np.zeros(len(order), dtype=bool)
    remaining = math.inf if budget is None else float(budget)
    limit = len(order) if max_count is None else int(max_count)
    if budget is None:
        take[: min(limit, len(order))] = True
    else:
        suffix_min = np.minimum.accumulate(c[::-1])[::-1] if len(c) else c
        start, count = 0, 0
        while start < len(c) and count < limit and remaining >= suffix_min[start]:
            cs = np.cumsum(c[start:])
            fit = int(np.searchsorted(cs, remaining, side="right"))  # items start..start+fit-1 fit
            fit = min(fit, limit - count)
            take[start : start + fit] = True
            if fit:
                remaining -= float(cs[fit - 1])
            count += fit
            start += fit + 1  # the next item did not fit (or the limit was reached): skip it
    choice = np.full(len(action), NONE, dtype=np.int64)
    choice[order[take]] = action[order[take]]
    return choice


def summarize(opts: Options, choice: np.ndarray) -> dict[str, Any]:
    idx = np.flatnonzero(choice != NONE)
    j = choice[idx]
    eff = opts.effect[idx, j]
    out = {
        "expected_retained_customers": math.fsum(eff.tolist()),
        "expected_retained_value": math.fsum((opts.value[idx] * eff).tolist()),
        "expected_cost": math.fsum(opts.cost[idx, j].tolist()),
        "net_value": math.fsum(opts.net[idx, j].tolist()),
        "contacted": len(idx),
    }
    for k, a in enumerate(opts.actions):
        out[f"contacted_{a}"] = int((j == k).sum())
    return out


def baseline_choices(
    opts: Options,
    budget: float | None,
    contact_limit: int | None,
    optimizer: Plan,
    action_index: dict[str, int],
) -> dict[str, np.ndarray]:
    """The deterministic baselines (random is separate)."""
    target = optimizer.contacts if budget is None else None
    count = _min_count(target, contact_limit)
    best = best_action(opts)
    uplift_value = np.where(opts.allowed, opts.effect * opts.value[:, None], -np.inf).max(axis=1)
    risk_order = ranking(opts.p, opts.msno)
    reminder = action_index["reminder"]
    fixed = np.where(opts.allowed[:, reminder], reminder, NONE)
    return {
        "optimizer": optimizer.choice,
        "highest_risk": fill(risk_order, best, opts.cost, budget, count),
        "highest_assumed_uplift_x_value": fill(
            ranking(uplift_value, opts.msno), best, opts.cost, budget, count
        ),
        "fixed_action_reminder": fill(risk_order, fixed, opts.cost, budget, count),
        "no_action": np.full(opts.n, NONE, dtype=np.int64),
    }


def random_draws(
    opts: Options,
    budget: float | None,
    contact_limit: int | None,
    optimizer: Plan,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Mean and 2.5-97.5% range of each metric over seeded random targeting."""
    target = optimizer.contacts if budget is None else None
    count = _min_count(target, contact_limit)
    best = best_action(opts)
    rows = []
    for d in range(draws):
        order = np.random.default_rng(seed + d).permutation(opts.n)
        rows.append(summarize(opts, fill(order, best, opts.cost, budget, count)))
    out = {}
    for m in rows[0]:
        v = np.array([r[m] for r in rows], dtype=float)
        out[m] = {
            "mean": float(v.mean()),
            "p2_5": float(np.percentile(v, 2.5)),
            "p97_5": float(np.percentile(v, 97.5)),
        }
    return out


def time_random(opts: Options, budget: float | None, optimizer: Plan, draws: int) -> float:
    """Seconds for `draws` random draws on the given options."""
    t = time.perf_counter()
    random_draws(opts, budget, None, optimizer, draws, seed=0)
    return time.perf_counter() - t


def projected_hours(seconds_per_cell: float, cells: int) -> float:
    return seconds_per_cell * cells / 3600.0


def _min_count(a: int | None, b: int | None) -> int | None:
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None
