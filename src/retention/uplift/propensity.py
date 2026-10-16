"""Propensity design for the KKBox uplift estimate (configs/uplift.yaml, kkbox.propensity).

Column rules use the analysed units' feature values only (no outcome):
constant columns are dropped; a binary column, or a categorical level, held by
fewer than `min_count` units is dropped; a categorical column is one-hot coded
with nulls as the level "missing"; a numeric column with nulls gets a 0/1 flag;
after coding, a column identical to an earlier one is dropped.

Inside the folds: numeric nulls are filled with the training-fold median and
inputs are standardised with training-fold statistics, then an L2-regularised
logistic regression is fitted. Folds are StratifiedGroupKFold, grouped by
customer and stratified by treatment. The penalty C is chosen by mean
out-of-fold log loss of treatment; ties go to the smaller C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Design:
    """The coded inputs and why each removed column was removed."""

    x: pd.DataFrame
    dropped: dict[str, str] = field(default_factory=dict)


def build_design(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], min_count: int = 5
) -> Design:
    dropped: dict[str, str] = {}
    cols: dict[str, pd.Series] = {}
    for c in numeric:
        s = pd.to_numeric(df[c], errors="raise").astype(float)
        values = s.dropna().unique()
        if len(values) + int(s.isna().any()) <= 1:
            dropped[c] = "constant"
            continue
        if len(values) == 2 and not s.isna().any() and s.value_counts().min() < min_count:
            dropped[c] = f"binary value held by fewer than {min_count} units"
            continue
        cols[c] = s
        if s.isna().any():
            cols[f"{c}_missing"] = s.isna().astype(float)
    for c in categorical:
        s = df[c].astype(object).where(df[c].notna(), "missing").astype(str)
        if s.nunique() <= 1:
            dropped[c] = "constant"
            continue
        for level, n in s.value_counts().sort_index().items():
            name = f"{c}={level}"
            if n < min_count:
                dropped[name] = f"level held by fewer than {min_count} units"
                continue
            cols[name] = (s == level).astype(float)
    x = pd.DataFrame(cols, index=df.index)
    seen: dict[bytes, str] = {}
    for c in list(x.columns):
        key = np.nan_to_num(x[c].to_numpy(), nan=-1e300).tobytes()
        if key in seen:
            dropped[c] = f"identical to {seen[key]}"
            x = x.drop(columns=c)
        else:
            seen[key] = c
    return Design(x, dropped)


def folds(t: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per unit; every unit of a customer lands in the same fold."""
    out = np.full(len(t), -1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(t)), t, groups)):
        out[test] = k
    return out


def _model(c: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000),
    )


def oof_propensity(x: pd.DataFrame, t: np.ndarray, fold_id: np.ndarray, c: float) -> np.ndarray:
    """Out-of-fold probability of treatment."""
    e = np.full(len(t), np.nan)
    for k in np.unique(fold_id):
        test = fold_id == k
        model = _model(c).fit(x[~test], t[~test])
        e[test] = model.predict_proba(x[test])[:, 1]
    return e


TIE_TOL = 1e-9


def pick_c(losses: dict[float, float], tol: float = TIE_TOL) -> float:
    """Smallest mean loss; losses within tol of the smallest are ties, which go to the
    smaller C (the stronger penalty)."""
    best = min(losses.values())
    return min(c for c, v in losses.items() if v <= best + tol)


def choose_c(
    x: pd.DataFrame, t: np.ndarray, fold_id: np.ndarray, grid: list[float]
) -> tuple[float, dict[float, float]]:
    """Mean over folds of the out-of-fold log loss of treatment, per C."""
    losses: dict[float, float] = {}
    for c in grid:
        e = oof_propensity(x, t, fold_id, c)
        per_fold = [
            log_loss(t[fold_id == k], e[fold_id == k], labels=[0, 1]) for k in np.unique(fold_id)
        ]
        losses[c] = float(np.mean(per_fold))
    return pick_c(losses), losses


def trim_mask(e: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep units with low <= e <= high."""
    return (e >= low) & (e <= high)


def stabilised_weights(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Treated p / e, untreated (1 - p) / (1 - e), with p the treated share of the given units."""
    t = np.asarray(t, dtype=float)
    p = t.mean()
    return np.where(t == 1, p / e, (1 - p) / (1 - e))


def kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())
