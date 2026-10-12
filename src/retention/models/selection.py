"""Choosing between candidates on the folds: one total simplicity order and the ladder.

Simplicity order (configs/evaluation.yaml, `selection`), compared axis by axis;
the first axis on which two candidates differ decides:
  1. family: cox < xgboost_aft < discrete_time
  2. fewer feature groups, then fewer columns
  3. library default over tuned hyperparameters
  4. Platt over isotonic
  5. candidate id order (ids are listed simplest first)

Ladder: sort a slot's candidates by that order; the current pick starts as the
simplest. Walk the list; a candidate replaces the current pick only if BOTH
  (a) the 95% bootstrap interval of the fold-mean difference (candidate minus
      current pick, higher is better) lies above 0, and
  (b) the candidate is better than the current pick on every fold of the slot.
Otherwise the current pick stays. The last current pick is chosen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retention.models.bootstrap import FoldScores, paired_auc_difference
from retention.models.metrics import SortedScores

FAMILY_ORDER = ("cox", "xgboost_aft", "discrete_time")
CALIBRATION_ORDER = ("platt", "isotonic")


@dataclass(frozen=True)
class Candidate:
    id: str
    family: str
    feature_groups: int
    columns: int
    tuned: bool = False
    calibration: str | None = None


def simplicity_key(c: Candidate) -> tuple:
    """Sort key: smaller is simpler. Families outside FAMILY_ORDER (a one-family slot) rank 0."""
    family = FAMILY_ORDER.index(c.family) if c.family in FAMILY_ORDER else 0
    calibration = CALIBRATION_ORDER.index(c.calibration) if c.calibration else 0
    return (family, c.feature_groups, c.columns, c.tuned, calibration, c.id)


def simplest_first(candidates: Sequence[Candidate]) -> list[Candidate]:
    families = {c.family for c in candidates}
    if len(families) > 1 and not families <= set(FAMILY_ORDER):
        raise ValueError(f"no registered order between families {sorted(families)}")
    return sorted(candidates, key=simplicity_key)


@dataclass(frozen=True)
class Comparison:
    """Candidate minus current pick: fold-mean difference, its interval, per-fold values."""

    d: float
    lo: float
    hi: float
    candidate_by_fold: tuple[float, ...]
    current_by_fold: tuple[float, ...]


@dataclass(frozen=True)
class Step:
    candidate: str
    current: str
    comparison: Comparison
    interval_above_zero: bool
    better_on_every_fold: bool
    replaced: bool


def ladder(
    candidates: Sequence[Candidate],
    compare: Callable[[Candidate, Candidate], Comparison],
) -> tuple[Candidate, list[Step]]:
    """Walk the candidates simplest first; return the chosen one and every step taken."""
    order = simplest_first(candidates)
    current, steps = order[0], []
    for cand in order[1:]:
        c = compare(cand, current)
        above = c.lo > 0
        every = all(a > b for a, b in zip(c.candidate_by_fold, c.current_by_fold, strict=True))
        replaced = above and every
        steps.append(Step(cand.id, current.id, c, above, every, replaced))
        if replaced:
            current = cand
    return current, steps


def auc_comparison(
    scores: dict[str, Sequence[FoldScores]], replicates: int, seed: int
) -> Callable[[Candidate, Candidate], Comparison]:
    """compare() for the ladder from each candidate's per-fold scores.

    `scores[id]` holds one FoldScores per fold, with the candidate's score in
    score_a (score_b is ignored); every candidate is scored on the same rows.
    """

    def compare(cand: Candidate, current: Candidate) -> Comparison:
        folds = [
            FoldScores(a.customer, a.y, a.score_a, b.score_a)
            for a, b in zip(scores[cand.id], scores[current.id], strict=True)
        ]
        iv = paired_auc_difference(folds, replicates, seed)
        by = [(SortedScores(f.score_a).auc(f.y), SortedScores(f.score_b).auc(f.y)) for f in folds]
        return Comparison(iv.point, iv.lo, iv.hi, tuple(a for a, _ in by), tuple(b for _, b in by))

    return compare
