"""T-learner, X-learner and AIPW on LightGBM bases (configs/uplift.yaml, learners).

Every prediction for a unit comes from models fitted on the other folds. The
X-learner's imputed effects use out-of-fold stage-1 predictions, and its stage-2
models are cross-fitted on the same folds. A fold whose training rows leave an arm
with fewer than 2 units raises NotComputable.

Constant-fit rule (configs/uplift.yaml, kkbox.outcome_step.constant_fits): if every
training outcome in an arm and fold is the same value, LightGBM cannot fit a classifier;
the arm's prediction in that fold is the value itself (0 or 1), and the fit is logged as
degenerate with its fold and arm. Only stage-1 outcome models count as fits here; the
X-learner's stage-2 regressors are counted apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

KKBOX_FIXED = {"n_estimators": 100, "num_leaves": 4, "min_child_samples": 10, "learning_rate": 0.05}
COMMON = {"seed": 0, "deterministic": True, "n_jobs": 4, "verbose": -1}


class NotComputable(RuntimeError):
    """A statistic cannot be computed on these rows; the reason is the message."""


@dataclass
class FitLog:
    outcome_fits: int = 0
    degenerate: list[tuple[int, int]] = field(default_factory=list)  # (fold, arm)
    stage2_fits: int = 0

    @property
    def constant_fits(self) -> int:
        return len(self.degenerate)

    @property
    def share(self) -> float:
        return self.constant_fits / self.outcome_fits if self.outcome_fits else 0.0

    def summary(self) -> str:
        return f"outcome models were degenerate in {self.constant_fits} of {self.outcome_fits} fits"


def _fit_predict(
    kind: str,
    params: dict,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    log: FitLog,
    fold: int,
    arm: int,
) -> np.ndarray:
    log.outcome_fits += 1
    if kind == "classifier":
        values = np.unique(y_train)
        if len(values) == 1:
            log.degenerate.append((fold, arm))
            return np.full(len(x_pred), float(values[0]))
        model = lgb.LGBMClassifier(**params, **COMMON).fit(x_train, y_train)
        return model.predict_proba(x_pred)[:, 1]
    model = lgb.LGBMRegressor(**params, **COMMON).fit(x_train, y_train)
    return model.predict(x_pred)


def _folds(fold_id: np.ndarray) -> list[int]:
    return sorted(int(k) for k in np.unique(fold_id))


def _check_arms(t_train: np.ndarray, k: int) -> None:
    for arm in (0, 1):
        if (t_train == arm).sum() < 2:
            raise NotComputable(f"fold {k} leaves arm {arm} with fewer than 2 training units")


def t_learner(
    x: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    fold_id: np.ndarray,
    kind: str,
    params: dict,
    log: FitLog | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions of the outcome without (mu0) and with (mu1) treatment."""
    log = log or FitLog()
    mu0, mu1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in _folds(fold_id):
        test = fold_id == k
        train = ~test
        _check_arms(t[train], k)
        for arm, out in ((0, mu0), (1, mu1)):
            rows = train & (t == arm)
            out[test] = _fit_predict(kind, params, x[rows], y[rows], x[test], log, k, arm)
    return mu0, mu1


def x_learner(
    x: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    fold_id: np.ndarray,
    kind: str,
    params: dict,
    e: np.ndarray,
    log: FitLog | None = None,
    mu: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-unit effect tau = e * tau0 + (1 - e) * tau1, cross-fitted at both stages.

    `mu` passes stage-1 predictions already made on the same folds, so they are not refit.
    """
    log = log or FitLog()
    mu0, mu1 = mu if mu is not None else t_learner(x, y, t, fold_id, kind, params, log)
    imputed = np.where(t == 1, y - mu0, mu1 - y)
    tau0, tau1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in _folds(fold_id):
        test = fold_id == k
        train = ~test
        _check_arms(t[train], k)
        for arm, out in ((0, tau0), (1, tau1)):
            rows = train & (t == arm)
            stage2 = lgb.LGBMRegressor(**params, **COMMON).fit(x[rows], imputed[rows])
            log.stage2_fits += 1
            out[test] = stage2.predict(x[test])
    return e * tau0 + (1 - e) * tau1


def aipw(y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray) -> float:
    """Mean of mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)."""
    if np.any((e <= 0) | (e >= 1)):
        raise NotComputable("a propensity of exactly 0 or 1")
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return float(np.mean(psi))


def naive_difference(y: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted outcome mean of treated minus untreated (weights default to 1)."""
    w = np.ones(len(y)) if w is None else w
    tr, un = t == 1, t == 0
    if w[tr].sum() == 0 or w[un].sum() == 0:
        raise NotComputable("an arm has no weight")
    return float((w[tr] * y[tr]).sum() / w[tr].sum() - (w[un] * y[un]).sum() / w[un].sum())


@dataclass(frozen=True)
class Estimates:
    aipw: float
    t_learner: float
    x_learner: float


def estimate(
    x: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    fold_id: np.ndarray,
    e: np.ndarray,
    kind: str,
    params: dict,
    log: FitLog | None = None,
) -> Estimates:
    """AIPW and the T- and X-learner average effects on the same folds and propensities."""
    log = log or FitLog()
    mu0, mu1 = t_learner(x, y, t, fold_id, kind, params, log)
    tau_x = x_learner(x, y, t, fold_id, kind, params, e, log, mu=(mu0, mu1))
    return Estimates(aipw(y, t, mu0, mu1, e), float(np.mean(mu1 - mu0)), float(np.mean(tau_x)))
