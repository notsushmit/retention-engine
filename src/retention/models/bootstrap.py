"""Customer-clustered bootstrap, paired AUC differences and the registered verdicts.

One replicate draws customers with replacement from the union of customers
across the folds (or months) being combined. A drawn customer brings all
their rows in every fold, so a row's weight is how many times its customer was
drawn. The statistic is computed per fold on those weights and averaged across
folds. The interval is the 2.5th and 97.5th percentiles of the replicates; the
point estimate is the statistic on the original data.

The interval covers which customers happened to be in the month, not
month-to-month change.

Replicates can run on several worker processes (`use_workers`): each worker
replays the one random stream up to the start of its block of replicates and
computes only that block, so the replicates, and the interval, are identical
for any number of workers.

If the statistic is not computable on the original data, or on any replicate
(for example a resample with no lapses), the interval is labelled "not
computable" with the reason, and no interval is formed from the other
replicates; nothing raises.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from retention.models.metrics import NOT_COMPUTABLE, NotComputable, SortedScores

VERDICT_THRESHOLDS = (0.005, 0.01, 0.02)
_WORKERS = 1
_JOB: dict = {}


def use_workers(n: int) -> None:
    """Number of worker processes for every bootstrap in this process (1 = no workers)."""
    global _WORKERS
    _WORKERS = max(1, int(n))


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    replicates: np.ndarray
    status: str = "ok"
    reason: str = ""


@dataclass(frozen=True)
class FoldScores:
    """One fold (or month): customer IDs, outcomes and the two scores compared."""

    customer: np.ndarray
    y: np.ndarray
    score_a: np.ndarray
    score_b: np.ndarray


class Draws:
    """Bootstrap weights for rows of several folds, by customer or (for comparison) by row."""

    def __init__(self, customers: Sequence[np.ndarray], seed: int, cluster: str = "customer"):
        if cluster not in ("customer", "row"):
            raise ValueError("cluster must be 'customer' or 'row'")
        self.cluster = cluster
        self.rng = np.random.default_rng(seed)
        union, inverse = np.unique(np.concatenate(customers), return_inverse=True)
        self.n_customers = len(union)
        bounds = np.cumsum([0, *[len(c) for c in customers]])
        self.index = [inverse[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]

    def skip(self, replicates: int) -> None:
        """Advance the random stream past `replicates` draws without computing weights."""
        for _ in range(replicates):
            if self.cluster == "row":
                for i in self.index:
                    self.rng.integers(0, len(i), len(i))
            else:
                self.rng.integers(0, self.n_customers, self.n_customers)

    def next(self) -> list[np.ndarray]:
        if self.cluster == "row":
            return [
                np.bincount(self.rng.integers(0, len(i), len(i)), minlength=len(i)).astype(float)
                for i in self.index
            ]
        n = self.n_customers
        counts = np.bincount(self.rng.integers(0, n, n), minlength=n).astype(float)
        return [counts[i] for i in self.index]


def bootstrap(
    customers: Sequence[np.ndarray],
    statistic: Callable[[int, np.ndarray | None], float],
    replicates: int,
    seed: int,
    level: float = 0.95,
    cluster: str = "customer",
) -> Interval:
    """`statistic(fold, weights)` per fold, averaged over folds, with its percentile interval."""
    k = len(customers)
    nan = float("nan")
    try:
        point = float(np.mean([statistic(f, None) for f in range(k)]))
    except NotComputable as err:
        return Interval(nan, nan, nan, np.array([]), NOT_COMPUTABLE, str(err))
    workers = min(_WORKERS, replicates)
    if workers == 1:
        reps, failed, reason = _block((0, replicates), customers, statistic, seed, cluster)
    else:
        _JOB["args"] = (customers, statistic, seed, cluster)
        edges = np.linspace(0, replicates, workers + 1).astype(int)
        blocks = list(zip(edges[:-1], edges[1:], strict=True))
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_job_block, blocks)
        _JOB.clear()
        reps = np.concatenate([p[0] for p in parts])
        failed = sum(p[1] for p in parts)
        reason = next((p[2] for p in parts if p[2]), "")
    if failed:
        why = f"{failed} of {replicates} replicates not computable ({reason})"
        return Interval(point, nan, nan, reps, NOT_COMPUTABLE, why)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(reps, [tail, 100 - tail])
    return Interval(point, float(lo), float(hi), reps)


def _block(bounds, customers, statistic, seed, cluster):
    a, b = bounds
    draws = Draws(customers, seed, cluster)
    draws.skip(a)
    k = len(customers)
    out = np.full(b - a, np.nan)
    failed, reason = 0, ""
    for i in range(b - a):
        weights = draws.next()
        try:
            out[i] = np.mean([statistic(f, weights[f]) for f in range(k)])
        except NotComputable as err:
            failed, reason = failed + 1, str(err)
    return out, failed, reason


def _job_block(bounds):
    customers, statistic, seed, cluster = _JOB["args"]
    return _block(bounds, customers, statistic, seed, cluster)


def paired_auc_difference(
    folds: Sequence[FoldScores],
    replicates: int,
    seed: int,
    level: float = 0.95,
    cluster: str = "customer",
) -> Interval:
    """AUC(score_a) - AUC(score_b), averaged over folds, on the same resampled customers."""
    sorted_a = [SortedScores(f.score_a) for f in folds]
    sorted_b = [SortedScores(f.score_b) for f in folds]

    def stat(f: int, w: np.ndarray | None) -> float:
        return sorted_a[f].auc(folds[f].y, w) - sorted_b[f].auc(folds[f].y, w)

    return bootstrap([f.customer for f in folds], stat, replicates, seed, level, cluster)


# --- verdicts -----------------------------------------------------------------------


def verdict(d: float, lo: float, hi: float, threshold: float = 0.01) -> str:
    """The registered verdict for a difference d with paired interval [lo, hi]."""
    if np.isnan([d, lo, hi]).any():
        return "not_computable"
    if lo > 0:
        return "win" if d >= threshold else "below_threshold"
    if hi < 0:
        return "loss"
    return "no_difference"


def verdicts(d: float, lo: float, hi: float) -> dict[float, str]:
    """The verdict at each reported threshold; the 0.01 one stands."""
    return {t: verdict(d, lo, hi, t) for t in VERDICT_THRESHOLDS}


def combine_months(per_month: Sequence[str]) -> str:
    """A cell over several test months wins only if every month wins; differing months are mixed."""
    return per_month[0] if len(set(per_month)) == 1 else "mixed"


def ece_interpretable(value: float, lo: float, hi: float) -> bool:
    """An ECE is read only if its interval stays within 0.5x to 1.5x its value."""
    return value > 0 and lo >= 0.5 * value and hi <= 1.5 * value
