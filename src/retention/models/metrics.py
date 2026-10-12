"""Evaluation metrics, with the project's censoring convention.

Convention (as in the labeller): `time` is days from the snapshot to the lapse
for an observed lapse (`observed` = 1), or to the stop day C for a censored row
(`observed` = 0). A row censored at C is known to be lapse-free through day C,
so a lapse on day C is observed and censoring takes effect just after C.

Every metric takes optional per-row `weight`s; the customer bootstrap passes
how many times each row's customer was drawn. Equal weights give the plain
metric.

A metric that the data cannot support (no lapses or no non-lapses, no
comparable pairs, outcomes separated perfectly by p) raises NotComputable;
`cell` turns that into a labelled "not computable" result, so a run reports the
cell and carries on. Asking for a day beyond the stop day is a different
error (ValueError): it means the code asked for something the protocol forbids.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

CLIP = 1e-6
NOT_COMPUTABLE = "not computable"


class NotComputable(ValueError):
    """The data in this cell cannot support the metric (for example, no lapses)."""


@dataclass(frozen=True)
class Cell:
    """A metric's value, or a labelled "not computable" with the reason."""

    value: Any
    status: str = "ok"
    reason: str = ""


def cell(metric: Callable[..., Any], *args: Any, **kwargs: Any) -> Cell:
    try:
        return Cell(metric(*args, **kwargs))
    except NotComputable as err:
        return Cell(float("nan"), NOT_COMPUTABLE, str(err))


def _w(n: int, weight: np.ndarray | None) -> np.ndarray:
    return np.ones(n) if weight is None else np.asarray(weight, dtype=float)


# --- discrimination ---------------------------------------------------------------


class SortedScores:
    """A score vector sorted once, so the AUC of many weightings is O(n) each."""

    def __init__(self, score: np.ndarray) -> None:
        score = np.asarray(score, dtype=float)
        self.order = np.argsort(score, kind="stable")
        _, self.group = np.unique(score[self.order], return_inverse=True)

    def auc(self, y: np.ndarray, weight: np.ndarray | None = None) -> float:
        """ROC AUC: share of (lapser, non-lapser) pairs ranked right; a tie counts 0.5."""
        y = np.asarray(y, dtype=float)[self.order]
        w = _w(len(y), weight)[self.order] if weight is not None else np.ones(len(y))
        pos = np.bincount(self.group, w * y)
        neg = np.bincount(self.group, w * (1 - y))
        below = np.cumsum(neg) - neg
        total = pos.sum() * neg.sum()
        if total == 0:
            raise NotComputable("AUC needs at least one lapse and one non-lapse")
        return float((pos * (below + 0.5 * neg)).sum() / total)


def auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    return SortedScores(score).auc(y, weight)


def status_at(time: np.ndarray, observed: np.ndarray, t: float) -> np.ndarray:
    """1 = lapsed by day t, 0 = known lapse-free through day t; raises if unknown."""
    time, observed = np.asarray(time), np.asarray(observed).astype(bool)
    case = observed & (time <= t)
    control = time > t
    control |= ~observed & (time >= t)
    if not (case | control).all():
        raise ValueError(f"day {t} is beyond the stop day of some rows; their status is unknown")
    return case.astype(int)


def td_auc(
    time: np.ndarray,
    observed: np.ndarray,
    score: np.ndarray,
    t: float,
    weight: np.ndarray | None = None,
) -> float:
    """Time-dependent (cumulative/dynamic) AUC at day t: lapsed by t vs lapse-free through t."""
    return auc(status_at(time, observed, t), score, weight)


def harrell_c(
    time: np.ndarray,
    observed: np.ndarray,
    risk: np.ndarray,
    tau: float | None = None,
    weight: np.ndarray | None = None,
) -> float:
    """Harrell's concordance, cut at day tau.

    A pair counts if the earlier customer had an observed lapse on or before tau
    and the other is known to last longer: a later time, or censored on the same
    day. Two lapses on the same day are not a pair. Concordant if the earlier
    lapser has the higher risk; a tied risk counts 0.5.
    """
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed).astype(bool)
    w = _w(len(time), weight)
    tau = np.inf if tau is None else tau
    _, rank = np.unique(np.asarray(risk, dtype=float), return_inverse=True)
    tree = np.zeros(rank.max() + 2)

    def add(i: int, v: float) -> None:
        i += 1
        while i < len(tree):
            tree[i] += v
            i += i & -i

    def below(i: int) -> float:  # total weight of ranks < i
        s = 0.0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    num = den = total = 0.0
    order = np.lexsort((observed, -time))  # time descending; censored first within a day
    k = 0
    while k < len(order):
        day = time[order[k]]
        group = []
        while k < len(order) and time[order[k]] == day:
            group.append(order[k])
            k += 1
        for i in group:
            if not observed[i]:
                add(rank[i], w[i])
                total += w[i]
        for i in group:
            if observed[i] and day <= tau:
                less, upto = below(rank[i]), below(rank[i] + 1)
                num += w[i] * (less + 0.5 * (upto - less))
                den += w[i] * total
        for i in group:
            if observed[i]:
                add(rank[i], w[i])
                total += w[i]
    if den == 0:
        raise NotComputable("no comparable pairs")
    return num / den


# --- censoring weights and Brier score ----------------------------------------------


def censoring_curve(time: np.ndarray, observed: np.ndarray, weight: np.ndarray | None = None):
    """Reverse Kaplan-Meier G(t): the share still followed after day t.

    A row censored at c drops out just after c; a row that lapsed on day c is no
    longer at risk of being censored. So the censorings at c lower G(t) only for
    t > c, and G is the same just before a whole day as on it. Returns G(t).
    """
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed).astype(bool)
    w = _w(len(time), weight)
    days = np.unique(time[~observed])
    at_risk = np.array([w[(time > d) | ((time == d) & ~observed)].sum() for d in days])
    dropped = np.array([w[(time == d) & ~observed].sum() for d in days])
    steps = np.cumprod(1 - dropped / at_risk)

    def g(t: float) -> float:
        n = np.searchsorted(days, t, side="left")  # censoring days strictly before t
        return float(steps[n - 1]) if n > 0 else 1.0

    return g


def brier_at(
    time: np.ndarray,
    observed: np.ndarray,
    surv: np.ndarray,
    t: float,
    weight: np.ndarray | None = None,
) -> float:
    """Brier score at day t with inverse-probability-of-censoring weights (Graf).

    Lapsed by t: (0 - S(t))^2 / G(lapse day, just before). Lapse-free through t:
    (1 - S(t))^2 / G(t). Censored before t: weight 0. Raises if G(t) is 0
    (nobody is followed beyond t).
    """
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed).astype(bool)
    surv = np.asarray(surv, dtype=float)
    w = _w(len(time), weight)
    g = censoring_curve(time, observed, weight)
    gt = g(t)
    if gt <= 0:
        raise ValueError(f"day {t} is beyond the stop day: no row is followed past it")
    case = observed & (time <= t)
    alive = (time > t) | (~observed & (time >= t))
    gi = np.array([g(x) for x in time[case]])
    contrib = np.zeros(len(time))
    contrib[case] = surv[case] ** 2 / gi
    contrib[alive] = (1 - surv[alive]) ** 2 / gt
    return float((w * contrib).sum() / w.sum())


def ibs(
    time: np.ndarray,
    observed: np.ndarray,
    surv: np.ndarray,
    grid: list[int] | np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """Integrated Brier score: the mean of the Brier score over the grid days.

    `surv` has one column per grid day. Each grid day has equal weight (the
    interval-end grid is spaced 7 to 9 days apart).
    """
    surv = np.asarray(surv, dtype=float)
    return float(
        np.mean([brier_at(time, observed, surv[:, k], t, weight) for k, t in enumerate(grid)])
    )


def brier(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.average((y - p) ** 2, weights=weight))


def log_loss(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weight))


def average_precision(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Area under the precision-recall curve, as a step sum over score thresholds."""
    y = np.asarray(y, dtype=float)
    w = _w(len(y), weight)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    s = np.asarray(score, dtype=float)[order]
    tp = np.cumsum((w * y)[order])
    fp = np.cumsum((w * (1 - y))[order])
    last = np.r_[np.diff(s) != 0, True]  # one point per distinct threshold
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# --- Kaplan-Meier references -----------------------------------------------------------


def kaplan_meier(
    time: np.ndarray, observed: np.ndarray, grid: list[int] | np.ndarray, weight=None
) -> np.ndarray:
    """Share lapse-free through each grid day (rows censored at c are at risk through c)."""
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed).astype(bool)
    w = _w(len(time), weight)
    days = np.unique(time[observed])
    at_risk = np.array([w[time >= d].sum() for d in days])
    lapsed = np.array([w[(time == d) & observed].sum() for d in days])
    steps = np.cumprod(1 - lapsed / at_risk)
    idx = np.searchsorted(days, np.asarray(grid), side="right")
    return np.where(idx > 0, steps[np.maximum(idx - 1, 0)], 1.0)


def kaplan_meier_by_band(
    time: np.ndarray,
    observed: np.ndarray,
    band: np.ndarray,
    grid: list[int] | np.ndarray,
) -> dict[object, np.ndarray]:
    """One Kaplan-Meier curve per band (for example a days_until_expiry band)."""
    band = np.asarray(band)
    time, observed = np.asarray(time), np.asarray(observed)
    return {b: kaplan_meier(time[band == b], observed[band == b], grid) for b in np.unique(band)}


def expiry_band(days_until_expiry: np.ndarray, edges: list[int]) -> np.ndarray:
    """Band index: 0 for days <= edges[0], ..., len(edges) for days > edges[-1]."""
    return np.searchsorted(np.asarray(edges), np.asarray(days_until_expiry), side="left")


# --- calibration -------------------------------------------------------------------


def logit(p: np.ndarray, clip: float = CLIP) -> tuple[np.ndarray, int]:
    """Logit of p clipped to [clip, 1 - clip], and how many values were clipped."""
    p = np.asarray(p, dtype=float)
    clipped = int(((p < clip) | (p > 1 - clip)).sum())
    q = np.clip(p, clip, 1 - clip)
    return np.log(q / (1 - q)), clipped


def _newton_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        mu = expit(x @ beta + offset)
        grad = x.T @ (w * (y - mu))
        hess = (x * (w * mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-12:
            return beta
    raise NotComputable("calibration regression did not converge (outcomes separated by p)")


def calibration_slope(
    y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None, clip: float = CLIP
) -> dict[str, float]:
    """Slope and intercept of y ~ a + b logit(p), and the level with the slope fixed at 1.

    `slope` b: below 1, predictions too extreme; above 1, too timid (undefined,
    NaN, when every p is equal). `intercept` (calibration in the large):
    y ~ a + 1 * logit(p); positive means predicted risk too low overall.
    `slope_model_intercept` is a from the two-parameter fit.
    """
    y = np.asarray(y, dtype=float)
    w = _w(len(y), weight)
    if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
        raise NotComputable("calibration needs at least one lapse and one non-lapse")
    z, clipped = logit(p, clip)
    (level,) = _newton_logistic(np.ones((len(z), 1)), y, w, z)
    if np.ptp(z) == 0:
        a = b = float("nan")
    else:
        a, b = _newton_logistic(np.c_[np.ones_like(z), z], y, w, np.zeros_like(z))
    return {
        "slope": float(b),
        "slope_model_intercept": float(a),
        "intercept": float(level),
        "clipped": clipped,
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10, weight: np.ndarray | None = None) -> float:
    """Expected calibration error over equal-count bins of p (descriptive only).

    Rows sorted by p are cut into `bins` groups of equal total weight; ECE is the
    weight-share-weighted mean of |mean p - event rate| per bin.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    w = _w(len(y), weight)
    order = np.argsort(p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    before = np.cumsum(w) - w
    b = np.minimum((before / w.sum() * bins).astype(int), bins - 1)
    out = 0.0
    for k in range(bins):
        m = b == k
        if w[m].sum() > 0:
            out += (
                w[m].sum()
                / w.sum()
                * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
            )
    return float(out)
