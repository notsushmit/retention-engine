"""XGBoost accelerated failure time model (objective survival:aft).

log T = f(x) + sigma * Z, with Z normal or logistic. Time is counted from day
1: a lapse on day d is an exact time d + 1; a row censored on day C is known
to be lapse-free through C, so its lapse is on day C + 1 or later, a time of
at least C + 2: the interval [C + 2, inf). The probability
of a lapse by day h is F((log(h + 1) - f(x)) / sigma), F the distribution of Z.
Native missing values; categorical columns as pandas categoricals with levels
frozen from the fit rows (unseen levels become missing and are counted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import logistic, norm

from retention.models.protocol import Threads

CDF = {"normal": norm.cdf, "logistic": logistic.cdf}


def encode(
    rows: pd.DataFrame, columns: list[str], cats: list[str], levels: dict[str, list]
) -> tuple[pd.DataFrame, dict[str, int]]:
    x = rows[columns].copy()
    unseen = {}
    for c in cats:
        values = x[c].astype("object").where(x[c].notna(), None)
        known = values.isin(levels[c]) | values.isna()
        unseen[c] = int((~known).sum())
        x[c] = pd.Categorical(values.where(known, None), categories=levels[c])
    for c in columns:
        if c not in cats:
            x[c] = x[c].astype("float64")
    return x, unseen


def aft_bounds(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds in time counted from day 1: [d + 1, d + 1] for a lapse, [C + 2, inf) if censored."""
    t = np.asarray(time, dtype=float) + 1.0
    e = np.asarray(event).astype(bool)
    return np.where(e, t, t + 1.0), np.where(e, t, np.inf)


@dataclass
class AFTModel:
    distribution: str
    sigma: float
    fixed: dict[str, Any]
    categorical: list[str]
    threads: Threads
    seed: int

    def fit(self, rows: pd.DataFrame, columns: list[str]) -> AFTModel:
        self.columns = columns
        self.cats = [c for c in self.categorical if c in columns]
        self.levels = {
            c: sorted(rows[c].dropna().astype("object").unique().tolist()) for c in self.cats
        }
        x, _ = encode(rows, columns, self.cats, self.levels)
        lower, upper = aft_bounds(rows.time.to_numpy(), rows.event.to_numpy())
        data = xgb.DMatrix(x, enable_categorical=True)
        data.set_float_info("label_lower_bound", lower)
        data.set_float_info("label_upper_bound", upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.distribution,
            "aft_loss_distribution_scale": self.sigma,
            "max_depth": self.fixed["max_depth"],
            "eta": self.fixed["eta"],
            "tree_method": self.fixed["tree_method"],
            "seed": self.seed,
            **self.threads.xgboost(),
        }
        self.booster = xgb.train(params, data, num_boost_round=self.fixed["num_boost_round"])
        return self

    def lapse_probability(self, rows: pd.DataFrame, days: list[int]) -> tuple[np.ndarray, dict]:
        """P(lapse by day h) for each row (rows x days), and unseen levels per column."""
        x, unseen = encode(rows, self.columns, self.cats, self.levels)
        location = np.log(self.booster.predict(xgb.DMatrix(x, enable_categorical=True)))
        z = (np.log(np.asarray(days, dtype=float) + 1.0)[None, :] - location[:, None]) / self.sigma
        return CDF[self.distribution](z), unseen
