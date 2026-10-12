"""Simple baselines, fitted on a fold's fit rows and scored on its scored rows.

base_rate              the fit rows' lapse rate for everyone (AUC 0.5 by definition;
                       the reference for Brier score and log loss)
auto_renew_off         1 - last_is_auto_renew
fewer_days_to_expiry   -days_until_expiry
two_feature_logistic   logistic regression on z-scaled days_until_expiry and
                       last_is_auto_renew (no penalty)
banded_logistic        logistic regression on days_until_expiry bands (0-7 the
                       reference; 8-14, 15-21, 22-29, 30+) and last_is_auto_renew

The two rules were chosen after the single-feature leak screen; at 7 days only
customers expiring within 7 days can lapse, which is why fewer days to expiry
is a baseline at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from retention.models.metrics import expiry_band
from retention.models.protocol import Threads

INPUTS = ["days_until_expiry", "last_is_auto_renew"]
NAMES = (
    "base_rate",
    "auto_renew_off",
    "fewer_days_to_expiry",
    "two_feature_logistic",
    "banded_logistic",
)


def _band_matrix(days: pd.Series, edges: list[int]) -> np.ndarray:
    band = expiry_band(days.to_numpy(dtype=float), edges)
    return np.stack([(band == b).astype(float) for b in range(1, len(edges) + 1)], axis=1)


class Baselines:
    """Fit once on fit rows; `score(name, rows)` gives each baseline's risk score."""

    def __init__(self, fit: pd.DataFrame, edges: list[int], threads: Threads, seed: int):
        self.edges = edges
        self.rate = float(fit.y.mean())
        days = fit.days_until_expiry.to_numpy(dtype=float)
        self.mu, self.sd = days.mean(), days.std()
        with threads.limit():
            self.two = LogisticRegression(C=np.inf, max_iter=1000, random_state=seed).fit(
                self._two_x(fit), fit.y
            )
            bands = self._band_x(fit)
            self.kept = bands.std(axis=0) > 0  # a band empty in the fit rows has no coefficient
            self.banded = LogisticRegression(C=np.inf, max_iter=1000, random_state=seed).fit(
                bands[:, self.kept], fit.y
            )
        self.dropped_bands = int((~self.kept).sum())

    def _two_x(self, rows: pd.DataFrame) -> np.ndarray:
        days = (rows.days_until_expiry.to_numpy(dtype=float) - self.mu) / self.sd
        return np.c_[days, rows.last_is_auto_renew.to_numpy(dtype=float)]

    def _band_x(self, rows: pd.DataFrame) -> np.ndarray:
        return np.c_[
            _band_matrix(rows.days_until_expiry, self.edges),
            rows.last_is_auto_renew.to_numpy(dtype=float),
        ]

    def score(self, name: str, rows: pd.DataFrame) -> np.ndarray:
        if name == "base_rate":
            return np.full(len(rows), self.rate)
        if name == "auto_renew_off":
            return 1 - rows.last_is_auto_renew.to_numpy(dtype=float)
        if name == "fewer_days_to_expiry":
            return -rows.days_until_expiry.to_numpy(dtype=float)
        if name == "two_feature_logistic":
            return self.two.predict_proba(self._two_x(rows))[:, 1]
        if name == "banded_logistic":
            return self.banded.predict_proba(self._band_x(rows)[:, self.kept])[:, 1]
        raise KeyError(name)
