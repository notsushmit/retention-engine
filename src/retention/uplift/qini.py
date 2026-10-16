"""Qini curve and area, Poisson bootstrap intervals, and the naive difference.

Definition (configs/uplift.yaml, section validation.qini):

- Rank customers by score, highest first; ties are broken by a random key drawn
  with a fixed seed, so the ranking is fully determined by (score, seed).
- Cut points: the top k_i = floor(i * N / n_cuts) customers, i = 1 .. n_cuts.
- Q(phi_i) = (treated outcomes in the top k_i
              - control outcomes in the top k_i * N_T(k_i) / N_C(k_i)) / N.
  When N_C(k_i) = 0 the control term is taken as 0.
- Area A = mean over the cut points of Q(phi_i) - phi_i * Q(1). A random
  ranking scores 0 in expectation.

The bootstrap gives every customer a Poisson(1) weight. Weights change the sums,
never the order: the ranking is computed once, from the scores and the tie seed,
so the interval covers which customers happened to be evaluated, not
model-training variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CUTS = 100


def rank_order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Positions sorted by score descending; ties broken by a seeded random key."""
    score = np.asarray(score, dtype=float)
    key = np.random.default_rng(seed).random(len(score))
    return np.lexsort((key, -score))


def cut_positions(n: int, n_cuts: int = N_CUTS) -> np.ndarray:
    """k_i = floor(i * n / n_cuts) for i = 1 .. n_cuts."""
    return (np.arange(1, n_cuts + 1) * n) // n_cuts


def _curve_sorted(y: np.ndarray, t: np.ndarray, w: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Q at the cut positions, for arrays already in ranking order."""
    yt = np.concatenate([[0.0], np.cumsum(w * y * t)])
    yc = np.concatenate([[0.0], np.cumsum(w * y * (1 - t))])
    nt = np.concatenate([[0.0], np.cumsum(w * t)])
    nc = np.concatenate([[0.0], np.cumsum(w * (1 - t))])
    ratio = np.divide(nt[cuts], nc[cuts], out=np.zeros(len(cuts)), where=nc[cuts] != 0)
    return (yt[cuts] - yc[cuts] * ratio) / w.sum()


def _area_from_curve(q: np.ndarray) -> float:
    phi = np.arange(1, len(q) + 1) / len(q)
    return float(np.mean(q - phi * q[-1]))


def _check(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> None:
    if not (len(y) == len(t) == len(score)):
        raise ValueError("y, t and score must have the same length")
    if len(y) < N_CUTS:
        raise ValueError(f"need at least {N_CUTS} customers for {N_CUTS} cut points")
    if not np.isin(t, (0, 1)).all():
        raise ValueError("treatment must be 0 or 1")


def qini_curve(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, seed: int = 0, n_cuts: int = N_CUTS
) -> tuple[np.ndarray, np.ndarray]:
    """(phi, Q(phi)) at the n_cuts cut points."""
    y, t, score = (np.asarray(a, dtype=float) for a in (y, t, score))
    _check(y, t, score)
    order = rank_order(score, seed)
    cuts = cut_positions(len(y), n_cuts)
    q = _curve_sorted(y[order], t[order], np.ones(len(y)), cuts)
    return np.arange(1, n_cuts + 1) / n_cuts, q


def qini_area(y: np.ndarray, t: np.ndarray, score: np.ndarray, seed: int = 0) -> float:
    """A = mean over the cut points of Q(phi) - phi * Q(1)."""
    return _area_from_curve(qini_curve(y, t, score, seed)[1])


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    replicates: np.ndarray


def _percentiles(estimate: float, reps: np.ndarray) -> Interval:
    low, high = np.percentile(reps, [2.5, 97.5])
    return Interval(float(estimate), float(low), float(high), reps)


def replicate_weights(n: int, b: int, seed: int):
    """Yield the Poisson(1) weights of each replicate in turn, from one seeded stream."""
    rng = np.random.default_rng(seed)
    for _ in range(b):
        yield rng.poisson(1.0, size=n).astype(float)


def bootstrap_area(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    b: int = 1000,
    seed: int = 0,
    tie_seed: int = 0,
) -> Interval:
    """Qini area with a Poisson(1) customer bootstrap; the ranking is fixed."""
    return bootstrap_area_difference(y, t, score, None, b, seed, tie_seed)


def bootstrap_area_difference(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    other: np.ndarray | None,
    b: int = 1000,
    seed: int = 0,
    tie_seed: int = 0,
) -> Interval:
    """Area of `score`, or the paired difference A(score) - A(other) on the same weights."""
    y, t, score = (np.asarray(a, dtype=float) for a in (y, t, score))
    _check(y, t, score)
    cuts = cut_positions(len(y))
    rankings = [rank_order(score, tie_seed)]
    if other is not None:
        other = np.asarray(other, dtype=float)
        _check(y, t, other)
        rankings.append(rank_order(other, tie_seed))
    sorted_ = [(y[o], t[o], o) for o in rankings]

    def stat(w: np.ndarray) -> float:
        areas = [_area_from_curve(_curve_sorted(ys, ts, w[o], cuts)) for ys, ts, o in sorted_]
        return areas[0] - areas[1] if other is not None else areas[0]

    estimate = stat(np.ones(len(y)))
    reps = np.array([stat(w) for w in replicate_weights(len(y), b, seed)])
    return _percentiles(estimate, reps)


def bootstrap_naive_difference(
    y: np.ndarray, t: np.ndarray, b: int = 1000, seed: int = 0
) -> Interval:
    """Outcome rate of treated minus control, with the same Poisson(1) bootstrap."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)

    def stat(w: np.ndarray) -> float:
        treated = (w * y * t).sum() / (w * t).sum()
        control = (w * y * (1 - t)).sum() / (w * (1 - t)).sum()
        return float(treated - control)

    estimate = stat(np.ones(len(y)))
    reps = np.array([stat(w) for w in replicate_weights(len(y), b, seed)])
    return _percentiles(estimate, reps)
