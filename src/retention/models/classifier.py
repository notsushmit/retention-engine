"""The LightGBM classifier: one model per horizon, target y (lapse within h days).

Settings come from configs/evaluation.yaml: a candidate's num_leaves,
n_estimators, learning_rate and min_child_samples, the fixed parameters
(deterministic, column-wise, no row or column subsampling, quiet), the thread
count and the seed. Missing values are handled natively. Categorical columns
become pandas categoricals whose levels are frozen from the fit rows; a level
first seen in a scored month becomes missing and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from retention.features.spec import FEATURE_COLUMNS, PRIMARY_FEATURES
from retention.models.protocol import Threads

CANDIDATE_FIELDS = ("num_leaves", "n_estimators", "learning_rate", "min_child_samples")


def feature_set_columns(drop_groups: list[str]) -> list[str]:
    """Primary feature columns without the dropped groups (the profile group is never in)."""
    return [c for c in PRIMARY_FEATURES if FEATURE_COLUMNS[c] not in drop_groups]


def candidate_params(values: list[Any]) -> dict[str, Any]:
    return dict(zip(CANDIDATE_FIELDS, values, strict=True))


@dataclass
class Classifier:
    columns: list[str]
    categorical: list[str]
    candidate: dict[str, Any]
    fixed: dict[str, Any]
    threads: Threads
    seed: int

    def _frame(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        x = rows[self.columns].copy()
        unseen = {}
        for c in self.cats:
            values = x[c].astype("object").where(x[c].notna(), None)
            known = values.isin(self.levels[c]) | values.isna()
            unseen[c] = int((~known).sum())
            x[c] = pd.Categorical(values.where(known, None), categories=self.levels[c])
        for c in self.columns:
            if c not in self.cats:
                x[c] = x[c].astype("float64")
        return x, unseen

    def fit(self, rows: pd.DataFrame) -> Classifier:
        self.cats = [c for c in self.categorical if c in self.columns]
        self.levels = {
            c: sorted(rows[c].dropna().astype("object").unique().tolist()) for c in self.cats
        }
        x, _ = self._frame(rows)
        params = {
            "objective": "binary",
            "num_leaves": self.candidate["num_leaves"],
            "learning_rate": self.candidate["learning_rate"],
            "min_child_samples": self.candidate["min_child_samples"],
            "seed": self.seed,
            **self.fixed,
            **self.threads.lightgbm(),
        }
        data = lgb.Dataset(x, rows.y.to_numpy(), categorical_feature=self.cats, free_raw_data=True)
        self.booster = lgb.train(params, data, num_boost_round=self.candidate["n_estimators"])
        return self

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        """Raw probabilities and, per categorical column, how many values were unseen in fit."""
        x, unseen = self._frame(rows)
        return self.booster.predict(x), unseen

    def importance(self) -> dict[str, float]:
        """Total gain per column (descriptive only; not causal, not used for selection)."""
        gain = self.booster.feature_importance(importance_type="gain")
        return dict(sorted(zip(self.columns, map(float, gain), strict=True), key=lambda kv: -kv[1]))


def score_summary(p: np.ndarray) -> dict[str, Any]:
    """Mean, deciles and number of distinct values of raw scores."""
    return {
        "mean": float(np.mean(p)),
        "deciles": [float(q) for q in np.quantile(p, np.linspace(0.1, 0.9, 9))],
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "distinct": int(np.unique(p).size),
    }
