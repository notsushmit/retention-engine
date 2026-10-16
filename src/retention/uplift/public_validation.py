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
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from retention.uplift import kkbox_bootstrap, qini
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


def select(
    kind: str, x: pd.DataFrame, y: np.ndarray, fold_id: np.ndarray, grid: list[dict] = GRID
) -> tuple[dict, list]:
    """The registered grid point for one base model, with every point's loss."""
    losses = [(p, cv_loss(kind, p, x, y, fold_id)) for p in grid]
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


def fit_cell(
    x: pd.DataFrame, cell: Cell, selection: np.ndarray | None = None, grid: list[dict] = GRID
) -> Scores:
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
        chosen[name], losses[name] = select("classifier", xs[m], ys[m], fs[m], grid)
    mu0, mu1 = oof_stage1(xs, ys, ts, fs, chosen["mu0"], chosen["mu1"])
    imputed = np.where(ts == 1, ys - mu0, mu1 - ys)
    for arm, name in ((0, "tau0"), (1, "tau1")):
        m = ts == arm
        chosen[name], losses[name] = select("regressor", xs[m], imputed[m], fs[m], grid)
    chosen["outcome"], losses["outcome"] = select("classifier", xs, ys, fs, grid)

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


# --- evaluating a cell -------------------------------------------------------------

B = 1000
TIMING_RESAMPLES = 10
BUDGET_SECONDS = 6 * 3600
BOOTSTRAPS_PER_CELL = 6  # four areas, the paired difference, the naive difference
NOTHING_OVER_RISK = (
    "uplift modelling added nothing over targeting by risk on this dataset "
    "(the X-learner beats random but not the outcome-probability ranking)"
)


def cell_key(cell: Cell) -> str:
    if cell.dataset == "criteo":
        return f"criteo_{cell.outcome}"
    return f"{cell.dataset}_{cell.name}_{cell.outcome}"


def _iv(i: qini.Interval) -> dict:
    """Interval before estimate."""
    return {"ci_low": i.low, "ci_high": i.high, "estimate": i.estimate}


def evaluate_cell(
    cell: Cell,
    scores: Scores,
    b: int = B,
    budget: kkbox_bootstrap.Budget | None = None,
    later_work: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    """Qini areas with Poisson bootstrap intervals, the pass marks and the sanity marks."""
    ev = cell.evaluation
    y, t = cell.y[ev], cell.t[ev]
    rankings = {
        "x_learner": scores.uplift_x,
        "t_learner": scores.uplift_t,
        "outcome_probability": scores.outcome_probability,
        "random": np.random.default_rng(SEED).random(len(y)),
    }
    key = cell_key(cell)
    curves = []
    for name, s in rankings.items():
        phi, q = qini.qini_curve(y, t, s)
        curves.append(pd.DataFrame({"cell": key, "ranking": name, "phi": phi, "q": q}))
    q1 = float(curves[0].q.iloc[-1])
    start = time.perf_counter()
    timing: dict = {}

    def progress(r: int) -> None:
        if r == TIMING_RESAMPLES:
            per = (time.perf_counter() - start) / TIMING_RESAMPLES
            timing["seconds_per_resample"] = round(per, 4)
            if budget is not None:
                budget.later_work = later_work
                budget.check(key, per, BOOTSTRAPS_PER_CELL * b - TIMING_RESAMPLES)

    areas = {}
    for i, (name, s) in enumerate(rankings.items()):
        areas[name] = qini.bootstrap_area(y, t, s, b, SEED, SEED, progress if i == 0 else None)
    paired = qini.bootstrap_area_difference(
        y, t, scores.uplift_x, scores.outcome_probability, b, SEED, SEED
    )
    naive = qini.bootstrap_naive_difference(y, t, b, SEED)
    verdict = judge(areas["x_learner"], naive, q1)
    beats_outcome = bool(paired.low > 0)
    record = {
        "cell": key,
        "evaluation_rows": int(len(y)),
        "treated_share_evaluation": float(t.mean()),
        "chosen_grid_points": scores.chosen,
        "grid_losses": scores.losses,
        "q1": q1,
        "qini_area": {name: _iv(a) for name, a in areas.items()},
        "naive_difference": _iv(naive),
        "x_minus_outcome_probability": _iv(paired),
        "pass_marks": verdict,
        "beats_outcome_ranking": beats_outcome,
        "note": NOTHING_OVER_RISK if verdict["met"] and not beats_outcome else None,
        "sanity": {
            "t_learner": sanity(float(scores.uplift_t.mean()), naive),
            "x_learner": sanity(float(scores.uplift_x.mean()), naive),
        },
        "average_effect": {
            "t_learner": float(scores.uplift_t.mean()),
            "x_learner": float(scores.uplift_x.mean()),
        },
        "timing": timing,
    }
    return record, pd.concat(curves, ignore_index=True)


# --- the single run ---------------------------------------------------------------

MARKER = "validation_marker.json"
LOG = "validation.json"


class RunOnce(RuntimeError):
    """The validation already ran, or an unfinished attempt needs a stated reason."""


def guard(out_dir: Path, rerun_reason: str | None = None) -> None:
    if (out_dir / LOG).exists():
        raise RunOnce(f"{LOG} exists: the validation ran and was logged; it runs once")
    if (out_dir / MARKER).exists() and not rerun_reason:
        raise RunOnce(
            "an earlier attempt started fitting and did not finish; "
            "repeat it only with a stated reason"
        )


def begin_run(out_dir: Path, commit: str, rerun_reason: str | None = None) -> dict:
    """Write (or extend) the marker before the first model is fitted."""
    guard(out_dir, rerun_reason)
    path = out_dir / MARKER
    stamp = {"commit": commit, "started": datetime.now(UTC).isoformat()}
    if path.exists():
        marker = json.loads(path.read_text())
        marker.setdefault("repeats", []).append({**stamp, "reason": rerun_reason})
    else:
        marker = {"first_attempt": stamp, "repeats": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def build_cells(hill: pd.DataFrame, crit: pd.DataFrame, grid_size: int = GRID_SUBSAMPLE):
    """(cell, features of its rows, selection mask) for every registered cell, Hillstrom first."""
    out = []
    ev_h = evaluation_half(hill.segment.to_numpy())
    x_h = hillstrom_features(hill, ~ev_h)
    for outcome in ("visit", "conversion"):
        for cell in hillstrom_cells(hill, ev_h, outcome):
            out.append((cell, x_h.iloc[cell.rows].reset_index(drop=True), None))
    t_c = crit.treatment.to_numpy().astype(int)
    ev_c = evaluation_half(t_c)
    x_c = criteo_features(crit)
    sub = grid_subsample(np.flatnonzero(~ev_c), t_c, grid_size)
    selection = np.zeros(len(t_c), dtype=bool)
    selection[sub] = True
    rows = np.arange(len(t_c))
    for outcome in ("visit", "conversion"):
        y = crit[outcome].to_numpy(dtype=float)
        cell = Cell("criteo", "treatment_vs_control", outcome, rows, t_c, y, ev_c)
        out.append((cell, x_c, selection))
    return out


def run_validation(
    hill: pd.DataFrame,
    crit: pd.DataFrame,
    out_dir: Path,
    commit: str,
    rerun_reason: str | None = None,
    b: int = B,
    grid: list[dict] = GRID,
    grid_size: int = GRID_SUBSAMPLE,
    budget_seconds: float = BUDGET_SECONDS,
) -> dict:
    """The single validation run; writes the scores, the curves and, last, the log."""
    step_start = time.perf_counter()
    begin_run(out_dir, commit, rerun_reason)
    cells = build_cells(hill, crit, grid_size)
    budget = kkbox_bootstrap.Budget(step_start, limit=budget_seconds)
    sizes = [int(c.evaluation.sum()) for c, _, _ in cells]
    records, curves, scores_rows = {}, [], []
    for i, (cell, x, selection) in enumerate(cells):
        scores = fit_cell(x, cell, selection, grid)
        later = sum(BOOTSTRAPS_PER_CELL * b * n / sizes[i] for n in sizes[i + 1 :])
        record, curve = evaluate_cell(cell, scores, b, budget, later)
        records[record["cell"]] = record
        curves.append(curve)
        scores_rows.append(
            pd.DataFrame(
                {
                    "dataset": cell.dataset,
                    "cell": cell.name,
                    "outcome": cell.outcome,
                    "row": cell.rows[cell.evaluation],
                    "uplift_t_learner": scores.uplift_t,
                    "uplift_x_learner": scores.uplift_x,
                    "outcome_probability": scores.outcome_probability,
                }
            )
        )
    write_uplift_scores(
        pd.concat(scores_rows, ignore_index=True), out_dir / "uplift_scores.parquet"
    )
    pd.concat(curves, ignore_index=True).to_parquet(out_dir / "qini_curves.parquet", index=False)
    log = {
        "commit": commit,
        "rerun_reason": rerun_reason,
        "cells": records,
        "done_when_first_criterion": done_when({k: r["pass_marks"] for k, r in records.items()}),
        "budget": budget.projections,
        "seconds": round(time.perf_counter() - step_start, 1),
    }
    return log


def write_log(out_dir: Path, log: dict) -> None:
    text = json.dumps(log, indent=2, default=str) + "\n"
    (out_dir / "validation_first_run.json").write_text(text)
    (out_dir / LOG).write_text(text)
