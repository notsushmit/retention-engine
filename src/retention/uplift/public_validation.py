"""Validation of the uplift learners on randomized public data (configs/uplift.yaml, validation).

Hillstrom: one 50/50 split of all 64,000 customers, stratified by the three arms (seed 0);
the cells men's against none and women's against none both use it, so they share the
No E-Mail control rows. Criteo: 50/50 stratified by treatment (seed 0); grid selection on
a fixed 1,000,000-row subsample of the training half, stratified by treatment (seed 0).
The evaluation half is never used for any choice.

Base models (stage-1 per arm, stage-2 regressors per arm, the pooled outcome-probability
baseline) each pick a point of the public grid by 5-fold cross-validated loss on the
training data (log loss for classifiers, squared error for regressors). The folds are one
StratifiedKFold by treatment (5 splits, shuffle, seed 0), also used for the out-of-fold
stage-1 predictions behind the X-learner's imputed effects. Ties go to fewer trees, then
fewer leaves, then larger min_child_samples.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from retention.uplift import qini
from retention.uplift.learners import COMMON

SEED = 0
FOLDS = 5
GRID_SUBSAMPLE = 1_000_000
LEARNING_RATE = 0.05
GRID = [
    {
        "n_estimators": n,
        "num_leaves": leaves,
        "min_child_samples": mcs,
        "learning_rate": LEARNING_RATE,
    }
    for n, leaves, mcs in itertools.product([100, 300], [4, 15], [20, 100])
]
TIE_TOL = 1e-9

HILLSTROM_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
HILLSTROM_CATEGORICAL = ["zip_code", "channel"]
HILLSTROM_ARMS = {"mens_vs_none": "Mens E-Mail", "womens_vs_none": "Womens E-Mail"}
HILLSTROM_CONTROL = "No E-Mail"
CRITEO_FEATURES = [f"f{i}" for i in range(12)]
NEVER_FEATURES = frozenset({"exposure", "segment", "treatment", "visit", "conversion", "spend"})


# --- coding -----------------------------------------------------------------------


def hillstrom_features(df: pd.DataFrame, train_rows: np.ndarray) -> pd.DataFrame:
    """Numbers as they are; zip_code and channel one-hot on the levels of the training half."""
    out = df[HILLSTROM_NUMERIC].astype(float).copy()
    for c in HILLSTROM_CATEGORICAL:
        for level in sorted(df.loc[train_rows, c].astype(str).unique()):
            out[f"{c}={level}"] = (df[c].astype(str) == level).astype(float)
    return out


def criteo_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[CRITEO_FEATURES].astype("float32")


def check_features(x: pd.DataFrame) -> None:
    bad = sorted(set(x.columns) & NEVER_FEATURES)
    if bad:
        raise ValueError(f"never a feature: {bad}")


# --- splits -----------------------------------------------------------------------


def evaluation_half(strata: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Boolean mask of the evaluation half of a 50/50 split stratified by `strata`."""
    idx = np.arange(len(strata))
    _, ev = train_test_split(idx, test_size=0.5, stratify=strata, random_state=seed)
    mask = np.zeros(len(strata), dtype=bool)
    mask[ev] = True
    return mask


def grid_subsample(train_idx: np.ndarray, t: np.ndarray, size: int = GRID_SUBSAMPLE) -> np.ndarray:
    """Row indices of the fixed grid subsample of the training half, stratified by treatment."""
    if size >= len(train_idx):
        return np.sort(train_idx)
    sub, _ = train_test_split(train_idx, train_size=size, stratify=t[train_idx], random_state=SEED)
    return np.sort(sub)


@dataclass(frozen=True)
class Cell:
    dataset: str
    name: str
    outcome: str
    rows: np.ndarray  # row positions in the dataset frame
    t: np.ndarray
    y: np.ndarray
    evaluation: np.ndarray  # mask over `rows`


def hillstrom_cells(df: pd.DataFrame, ev: np.ndarray, outcome: str) -> list[Cell]:
    """Men's and women's email against no email; both use the same control rows."""
    cells = []
    for name, arm in HILLSTROM_ARMS.items():
        rows = np.flatnonzero(df.segment.isin([arm, HILLSTROM_CONTROL]).to_numpy())
        t = (df.segment.to_numpy()[rows] == arm).astype(int)
        y = df[outcome].to_numpy(dtype=float)[rows]
        cells.append(Cell("hillstrom", name, outcome, rows, t, y, ev[rows]))
    return cells


# --- grid selection ---------------------------------------------------------------


def tie_key(point: dict) -> tuple:
    return (point["n_estimators"], point["num_leaves"], -point["min_child_samples"])


def pick_point(losses: list[tuple[dict, float]], tol: float = TIE_TOL) -> dict:
    """Lowest loss; losses within tol of it are ties, which go to fewer trees, then fewer
    leaves, then larger min_child_samples."""
    best = min(v for _, v in losses)
    tied = [p for p, v in losses if v <= best + tol]
    return min(tied, key=tie_key)


def folds(t: np.ndarray, seed: int = SEED) -> np.ndarray:
    out = np.full(len(t), -1)
    split = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(split.split(np.zeros(len(t)), t)):
        out[test] = k
    return out


def _model(kind: str, point: dict):
    cls = lgb.LGBMClassifier if kind == "classifier" else lgb.LGBMRegressor
    return cls(**point, **COMMON)


def fit_predict(kind: str, point: dict, x_fit, y_fit, x_pred) -> np.ndarray:
    model = _model(kind, point).fit(x_fit, y_fit)
    return model.predict_proba(x_pred)[:, 1] if kind == "classifier" else model.predict(x_pred)


def cv_loss(kind: str, point: dict, x: pd.DataFrame, y: np.ndarray, fold_id: np.ndarray) -> float:
    losses = []
    for k in range(FOLDS):
        test = fold_id == k
        pred = fit_predict(kind, point, x[~test], y[~test], x[test])
        if kind == "classifier":
            losses.append(log_loss(y[test], np.clip(pred, 1e-15, 1 - 1e-15), labels=[0, 1]))
        else:
            losses.append(mean_squared_error(y[test], pred))
    return float(np.mean(losses))


def select(kind: str, x: pd.DataFrame, y: np.ndarray, fold_id: np.ndarray) -> tuple[dict, list]:
    """The registered grid point for one base model, with every point's loss."""
    losses = [(p, cv_loss(kind, p, x, y, fold_id)) for p in GRID]
    return pick_point(losses), [{**p, "loss": round(v, 6)} for p, v in losses]


def oof_stage1(
    x: pd.DataFrame, y: np.ndarray, t: np.ndarray, fold_id: np.ndarray, p0: dict, p1: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold mu0 and mu1 over every row, from the chosen stage-1 settings."""
    mu0, mu1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(FOLDS):
        test = fold_id == k
        for arm, point, out in ((0, p0, mu0), (1, p1, mu1)):
            rows = ~test & (t == arm)
            out[test] = fit_predict("classifier", point, x[rows], y[rows], x[test])
    return mu0, mu1


# --- fitting a cell ---------------------------------------------------------------


@dataclass
class Scores:
    uplift_t: np.ndarray
    uplift_x: np.ndarray
    outcome_probability: np.ndarray
    chosen: dict
    losses: dict


def fit_cell(x: pd.DataFrame, cell: Cell, selection: np.ndarray | None = None) -> Scores:
    """Choose every base model on the selection rows, refit on the whole training half,
    and score the evaluation half.

    `x` holds the features of the cell's rows, in the cell's row order. `selection` is a
    boolean mask over those rows (inside the training half) used for grid selection; by
    default the whole training half.
    """
    check_features(x)
    t, y = cell.t, cell.y
    train = ~cell.evaluation
    sel = train if selection is None else selection & train
    xs, ys, ts = x[sel].reset_index(drop=True), y[sel], t[sel]
    fs = folds(ts)
    chosen, losses = {}, {}
    for arm, name in ((0, "mu0"), (1, "mu1")):
        m = ts == arm
        chosen[name], losses[name] = select("classifier", xs[m], ys[m], fs[m])
    mu0, mu1 = oof_stage1(xs, ys, ts, fs, chosen["mu0"], chosen["mu1"])
    imputed = np.where(ts == 1, ys - mu0, mu1 - ys)
    for arm, name in ((0, "tau0"), (1, "tau1")):
        m = ts == arm
        chosen[name], losses[name] = select("regressor", xs[m], imputed[m], fs[m])
    chosen["outcome"], losses["outcome"] = select("classifier", xs, ys, fs)

    xt, yt, tt = x[train].reset_index(drop=True), y[train], t[train]
    ft = folds(tt)
    mu0t, mu1t = oof_stage1(xt, yt, tt, ft, chosen["mu0"], chosen["mu1"])
    imputed_t = np.where(tt == 1, yt - mu0t, mu1t - yt)
    xe = x[cell.evaluation].reset_index(drop=True)
    pred = {}
    for arm, name in ((0, "mu0"), (1, "mu1")):
        m = tt == arm
        pred[name] = fit_predict("classifier", chosen[name], xt[m], yt[m], xe)
    for arm, name in ((0, "tau0"), (1, "tau1")):
        m = tt == arm
        pred[name] = fit_predict("regressor", chosen[name], xt[m], imputed_t[m], xe)
    pred["outcome"] = fit_predict("classifier", chosen["outcome"], xt, yt, xe)
    e = float(tt.mean())  # randomized: the treated share of the training half
    return Scores(
        uplift_t=pred["mu1"] - pred["mu0"],
        uplift_x=e * pred["tau0"] + (1 - e) * pred["tau1"],
        outcome_probability=pred["outcome"],
        chosen=chosen,
        losses=losses,
    )


# --- pass marks -------------------------------------------------------------------

MIN_SIZE = 0.10
PASSED_ON_A_ALONE = "passed on (a) alone; average effect not distinguishable from zero"
PRIMARY = ["hillstrom_mens_vs_none_visit", "hillstrom_womens_vs_none_visit", "criteo_visit"]


def judge(area: qini.Interval, naive: qini.Interval, q1: float) -> dict:
    """(a) the lower end of A's interval above 0; (b) A >= 0.10 * Q(1) / 2, judged only
    when the naive difference's interval lies entirely above 0. A cell passing (a) alone
    is reported as such and counts as not met."""
    a = area.low > 0
    b_judged = naive.low > 0
    b = bool(area.estimate >= MIN_SIZE * q1 / 2) if b_judged else None
    if b_judged:
        met = a and b
        status = "met" if met else "not met"
    else:
        met = False
        status = PASSED_ON_A_ALONE if a else "not met"
    return {
        "a": a,
        "b_judged": b_judged,
        "b": b,
        "b_threshold": MIN_SIZE * q1 / 2,
        "met": met,
        "status": status,
    }


def done_when(verdicts: dict[str, dict]) -> dict:
    """Met only if every primary cell passes both (a) and (b)."""
    missing = [c for c in PRIMARY if c not in verdicts]
    if missing:
        raise ValueError(f"primary cells missing: {missing}")
    failing = [c for c in PRIMARY if not verdicts[c]["met"]]
    return {"met": not failing, "failing_primary_cells": failing}


def sanity(average_effect: float, naive: qini.Interval) -> bool:
    """A learner's average predicted effect lies inside the naive difference's 95% interval."""
    return bool(naive.low <= average_effect <= naive.high)


# --- outputs ----------------------------------------------------------------------

PUBLIC_DATASETS = frozenset({"hillstrom", "criteo"})
SCORE_COLUMNS = [
    "dataset",
    "cell",
    "outcome",
    "row",
    "uplift_t_learner",
    "uplift_x_learner",
    "outcome_probability",
]


def write_uplift_scores(scores: pd.DataFrame, path) -> None:
    """Public evaluation rows only; a KKBox row or any customer field is refused."""
    if list(scores.columns) != SCORE_COLUMNS:
        raise ValueError(f"uplift scores take exactly {SCORE_COLUMNS}, got {list(scores.columns)}")
    other = sorted(set(scores.dataset.astype(str)) - PUBLIC_DATASETS)
    if other:
        raise ValueError(f"uplift scores hold public datasets only; refused {other}")
    scores.to_parquet(path, index=False)
