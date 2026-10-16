"""Customer-clustered bootstrap for the KKBox estimates (configs/uplift.yaml,
kkbox.outcome_step).

Each resample draws customers with replacement from all analysed units; a drawn
customer brings every one of their units, and all copies keep the customer as their
group, so they share a fold. For the effect estimates, each resample refits the
propensity (C fixed) on new customer-grouped folds, keeps the trimming membership of
the original run, and refits the outcome models on the kept units.

The interval covers sampling variation given the trimmed set, not the uncertainty of
the trimming itself.

If any resample is not computable, the bootstrap stops there and the interval is
"not computable" with the reason; no interval is formed from the other resamples.

Time per resample is measured on the first 10 resamples; if the projected total for the
whole step exceeds the budget, the run stops (OverBudget).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from retention.uplift import learners, propensity

TIMING_RESAMPLES = 10
BUDGET_SECONDS = 3 * 3600


class OverBudget(RuntimeError):
    """The projected time of the step exceeds the registered budget."""


@dataclass
class Budget:
    """Projects the whole step from the measured time per resample."""

    step_started: float
    later_work: float = 0.0  # resample-equivalents still to run after the current bootstrap
    limit: float = BUDGET_SECONDS
    clock: Callable[[], float] = time.perf_counter
    projections: list[dict] = field(default_factory=list)

    def check(self, name: str, per_resample: float, remaining: int) -> None:
        elapsed = self.clock() - self.step_started
        projected = elapsed + per_resample * (remaining + self.later_work)
        self.projections.append(
            {
                "bootstrap": name,
                "seconds_per_resample": round(per_resample, 4),
                "projected_total_seconds": round(projected, 1),
            }
        )
        if projected > self.limit:
            hours, limit = projected / 3600, self.limit / 3600
            raise OverBudget(f"{name}: projected total {hours:.2f} h exceeds {limit:.1f} h")


@dataclass
class Result:
    estimates: dict[str, float]
    status: str  # "computed" or "not computable"
    reason: str | None = None
    intervals: dict[str, tuple[float, float]] | None = None
    replicates: dict[str, np.ndarray] | None = None
    resamples_done: int = 0
    seconds_per_resample: float | None = None


def resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one customer resample (a drawn customer brings all their units)."""
    customers, inverse = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == i) for i in range(len(customers))]
    drawn = rng.integers(0, len(customers), size=len(customers))
    return np.concatenate([rows_of[i] for i in drawn])


def bootstrap(
    name: str,
    statistic: Callable[[np.ndarray, np.random.Generator], dict[str, float]],
    estimates: dict[str, float],
    groups: np.ndarray,
    b: int = 1000,
    seed: int = 0,
    budget: Budget | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Result:
    """Run `statistic` on b customer resamples; stop at the first not-computable one."""
    rng = np.random.default_rng(seed)
    reps: dict[str, list[float]] = {k: [] for k in estimates}
    start = clock()
    per = None
    for r in range(b):
        idx = resample(groups, rng)
        try:
            values = statistic(idx, rng)
        except learners.NotComputable as err:
            return Result(estimates, "not computable", f"resample {r + 1}: {err}", resamples_done=r)
        for k in reps:
            reps[k].append(values[k])
        if r + 1 == TIMING_RESAMPLES:
            per = (clock() - start) / TIMING_RESAMPLES
            if budget is not None:
                budget.check(name, per, b - TIMING_RESAMPLES)
    arrays = {k: np.array(v) for k, v in reps.items()}
    intervals = {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in arrays.items()
    }
    return Result(estimates, "computed", None, intervals, arrays, b, per)


def effect_statistic(
    x: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kept: np.ndarray,
    c: float,
    kind: str,
    params: dict,
) -> Callable[[np.ndarray, np.random.Generator], dict[str, float]]:
    """Refit the propensity on the resample (C fixed, new grouped folds), keep the original
    trimming membership, refit the outcome models on the kept units."""

    def statistic(idx: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        xr, tr, gr, yr, kr = (
            x.iloc[idx].reset_index(drop=True),
            t[idx],
            groups[idx],
            y[idx],
            kept[idx],
        )
        fold_r = propensity.folds(tr, gr, 5, int(rng.integers(0, 2**31 - 1)))
        e_r = propensity.oof_propensity(xr, tr, fold_r, c)
        if np.any((e_r[kr] <= 0) | (e_r[kr] >= 1)):
            raise learners.NotComputable("a kept unit has a propensity of exactly 0 or 1")
        est = learners.estimate(
            xr[kr].reset_index(drop=True), yr[kr], tr[kr], fold_r[kr], e_r[kr], kind, params
        )
        return {"aipw": est.aipw, "t_learner": est.t_learner, "x_learner": est.x_learner}

    return statistic


def naive_statistic(
    t: np.ndarray, y: np.ndarray, rows: np.ndarray
) -> Callable[[np.ndarray, np.random.Generator], dict[str, float]]:
    """Naive difference on the resampled units that belong to `rows` (a boolean mask)."""

    def statistic(idx: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        keep = idx[rows[idx]]
        return {"naive": learners.naive_difference(y[keep], t[keep])}

    return statistic
