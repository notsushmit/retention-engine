"""Customer groups on the randomized public data (configs/uplift.yaml, groups).

Built only for the two primary cells that were met; Hillstrom men's visit is "not
built: the cell did not beat random". No KKBox customer is ever assigned to a group.

For each cell: the T-learner's two models are refit on the training half with the grid
points the validation run logged, and the rebuilt T-learner uplift must equal the logged
one within 1e-9 (the reproduction check), or the step stops. Thresholds come from the
training half only: delta is half the naive treated-minus-control difference in outcome
rate, m is the control arm's outcome rate. Groups are assigned on the evaluation half from
the X-learner uplift tau and the refit control model's prediction mu0:
  persuadable     tau >= delta
  do_not_disturb  tau <= -delta
  stays_anyway    |tau| < delta and mu0 >= m
  lost_cause      |tau| < delta and mu0 < m
Assignment reads no evaluation-half outcome. The one registered use of evaluation-half
outcomes is the check: within each group, the treated-minus-control difference with a
Poisson customer bootstrap interval; the Persuadable group is expected to show the largest
difference, and the result is reported as found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention.uplift import qini
from retention.uplift.public_validation import (
    PUBLIC_DATASETS,
    Cell,
    cell_key,
    check_features,
    fit_predict,
)

CELLS = ["hillstrom_womens_vs_none_visit", "criteo_visit"]
NOT_BUILT = {"hillstrom_mens_vs_none_visit": "not built: the cell did not beat random"}
GROUPS = ["persuadable", "stays_anyway", "lost_cause", "do_not_disturb"]
REPRODUCTION_TOL = 1e-9
B, SEED = 1000, 0


class ReproductionFailed(RuntimeError):
    """The refit T-learner does not reproduce the logged uplift scores."""


def thresholds(y_train: np.ndarray, t_train: np.ndarray) -> dict[str, float]:
    """delta = half the training-half naive effect; m = the training-half control rate."""
    treated = float(y_train[t_train == 1].mean())
    control = float(y_train[t_train == 0].mean())
    return {"delta": 0.5 * (treated - control), "m": control}


def assign(tau: np.ndarray, mu0: np.ndarray, delta: float, m: float) -> np.ndarray:
    return np.select(
        [tau >= delta, tau <= -delta, mu0 >= m],
        ["persuadable", "do_not_disturb", "stays_anyway"],
        "lost_cause",
    )


def refit_t(x: pd.DataFrame, cell: Cell, chosen: dict) -> tuple[np.ndarray, np.ndarray]:
    """mu0 and mu1 for the evaluation half, refit on the training half with the logged points."""
    check_features(x)
    train = ~cell.evaluation
    xt, yt, tt = x[train].reset_index(drop=True), cell.y[train], cell.t[train]
    xe = x[cell.evaluation].reset_index(drop=True)
    out = []
    for arm, name in ((0, "mu0"), (1, "mu1")):
        m = tt == arm
        out.append(fit_predict("classifier", chosen[name], xt[m], yt[m], xe))
    return out[0], out[1]


def reproduce(rebuilt: np.ndarray, logged: np.ndarray) -> float:
    if len(rebuilt) != len(logged):
        raise ReproductionFailed(f"{len(rebuilt)} rebuilt scores against {len(logged)} logged")
    diff = float(np.max(np.abs(rebuilt - logged))) if len(logged) else 0.0
    if diff > REPRODUCTION_TOL:
        raise ReproductionFailed(f"rebuilt T-learner uplift differs by {diff:.2e}")
    return diff


def group_check(
    y_eval: np.ndarray, t_eval: np.ndarray, groups: np.ndarray, b: int = B
) -> dict[str, dict]:
    """The registered check: within each group, treated minus control on the evaluation half."""
    out = {}
    for g in GROUPS:
        rows = groups == g
        y, t = y_eval[rows], t_eval[rows]
        rec = {
            "customers": int(rows.sum()),
            "treated": int(t.sum()),
            "control": int((t == 0).sum()),
        }
        if rec["treated"] == 0 or rec["control"] == 0:
            rec["status"] = "not computable: an arm is empty"
        else:
            iv = qini.bootstrap_naive_difference(y, t, b, SEED)
            if np.isnan(iv.replicates).any():
                rec["status"] = "not computable: a resample left an arm with no weight"
            else:
                rec |= {
                    "status": "computed",
                    "ci_low": iv.low,
                    "ci_high": iv.high,
                    "estimate": iv.estimate,
                }
        out[g] = rec
    return out


def persuadable_largest(check: dict[str, dict]) -> str:
    """'yes', 'no', or 'not computable' when any group's difference could not be computed."""
    if any(r["status"] != "computed" for r in check.values()):
        return "not computable"
    best = max(check, key=lambda g: check[g]["estimate"])
    return "yes" if best == "persuadable" else "no"


def group_cell(
    cell: Cell, x: pd.DataFrame, chosen: dict, logged: pd.DataFrame, b: int = B
) -> tuple[dict, pd.DataFrame]:
    """Groups and the registered check for one cell; `logged` holds that cell's scores in
    evaluation-row order (columns row, uplift_t_learner, uplift_x_learner)."""
    if not np.array_equal(logged.row.to_numpy(), cell.rows[cell.evaluation]):
        raise ReproductionFailed("logged rows are not the cell's evaluation rows in order")
    mu0, mu1 = refit_t(x, cell, chosen)
    diff = reproduce(mu1 - mu0, logged.uplift_t_learner.to_numpy())
    train = ~cell.evaluation
    th = thresholds(cell.y[train], cell.t[train])
    groups = assign(logged.uplift_x_learner.to_numpy(), mu0, th["delta"], th["m"])
    counts = {g: int((groups == g).sum()) for g in GROUPS}
    ev = cell.evaluation
    check = group_check(cell.y[ev], cell.t[ev], groups, b)
    record = {
        "cell": cell_key(cell),
        "reproduction_max_abs_diff": diff,
        "thresholds": th,
        "counts": counts,
        "check": check,
        "persuadable_largest": persuadable_largest(check),
    }
    members = pd.DataFrame(
        {
            "dataset": cell.dataset,
            "cell": cell_key(cell),
            "row": cell.rows[ev],
            "group": groups,
        }
    )
    return record, members


def write_members(members: pd.DataFrame, path) -> None:
    """Public evaluation rows only; a KKBox row or a customer field is refused."""
    if list(members.columns) != ["dataset", "cell", "row", "group"]:
        raise ValueError(
            f"group members take exactly dataset, cell, row, group; got {list(members.columns)}"
        )
    other = sorted(set(members.dataset.astype(str)) - PUBLIC_DATASETS)
    if other:
        raise ValueError(f"groups hold public datasets only; refused {other}")
    members.to_parquet(path, index=False)


def run_groups(
    hill: pd.DataFrame,
    crit: pd.DataFrame,
    scores: pd.DataFrame,
    validation_log: dict,
    out_dir,
    b: int = B,
    grid_size: int | None = None,
) -> dict:
    """Groups for the registered cells; writes group_members.parquet and returns the log."""
    from retention.uplift.public_validation import GRID_SUBSAMPLE, build_cells

    built = build_cells(hill, crit, GRID_SUBSAMPLE if grid_size is None else grid_size)
    cells = {cell_key(c): (c, x) for c, x, _ in built}
    records, members = {}, []
    for key in CELLS:
        cell, x = cells[key]
        chosen = validation_log["cells"][key]["chosen_grid_points"]
        mine = scores[
            (scores.dataset == cell.dataset)
            & (scores.cell == cell.name)
            & (scores.outcome == cell.outcome)
        ].reset_index(drop=True)
        record, m = group_cell(cell, x, chosen, mine, b)
        records[key] = record
        members.append(m)
    write_members(pd.concat(members, ignore_index=True), out_dir / "group_members.parquet")
    return {
        "note": (
            "Groups are built on the randomized public data only and mostly separate customers "
            "by their baseline rate; no KKBox customer is assigned to a group."
        ),
        "validation_commit": validation_log["commit"],
        "cells": records,
        "not_built": NOT_BUILT,
    }
