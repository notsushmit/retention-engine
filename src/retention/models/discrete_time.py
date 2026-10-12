"""Discrete-time hazard model: LightGBM on one row per customer and interval at risk.

Intervals end on the registered days (7, 14, 21, 30, 37, 44, 51, 59, 67, 74,
81, 90); the first covers days 0-7. A row enters interval k if it had not
lapsed before the interval starts; the target is a lapse inside the interval.
An interval that a censored row does not reach the end of is left out (a
partly observed interval would count an unknown stretch as survived).
S(end of interval k) = product over j <= k of (1 - hazard_j); a lapse by day h
(an interval end) has probability 1 - S(h).

Inputs: the feature set, the interval end day (the time variable), and, in the
dt02 candidate, "interval end minus days_until_expiry", a deliberate
engineering input that tells the model where the current expiry falls
relative to the interval (it makes the model less pure survival modelling).
Hyperparameters are those chosen for the 7-day classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retention.models.classifier import Classifier
from retention.models.protocol import Threads

TIME = "interval_end"
OFFSET = "interval_end_minus_days_until_expiry"


def expand(rows: pd.DataFrame, ends: list[int], for_fit: bool) -> pd.DataFrame:
    """One row per (customer row, interval at risk); `y` is a lapse inside the interval.

    For scoring (for_fit False), every row gets every interval.
    """
    starts = [-1, *ends[:-1]]
    parts = []
    for start, end in zip(starts, ends, strict=True):
        if for_fit:
            t, e = rows.time.to_numpy(), rows.event.to_numpy().astype(bool)
            at_risk = t > start
            observed = (e & (t <= end)) | (t >= end)  # lapsed inside, or followed past the end
            keep = at_risk & observed
            part = rows[keep].copy()
            part["y"] = (e[keep] & (t[keep] <= end)).astype(int)
        else:
            part = rows.copy()
        part[TIME] = end
        part[OFFSET] = end - part.days_until_expiry.astype(float)
        part["source_row"] = np.flatnonzero(keep) if for_fit else np.arange(len(rows))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@dataclass
class DiscreteTimeModel:
    offset_input: bool
    ends: list[int]
    candidate: dict[str, Any]
    fixed: dict[str, Any]
    categorical: list[str]
    threads: Threads
    seed: int

    def fit(self, rows: pd.DataFrame, columns: list[str]) -> DiscreteTimeModel:
        self.columns = [*columns, TIME, *([OFFSET] if self.offset_input else [])]
        long = expand(rows, self.ends, for_fit=True)
        self.expanded_rows = len(long)
        self.model = Classifier(
            self.columns, self.categorical, self.candidate, self.fixed, self.threads, self.seed
        ).fit(long)
        return self

    def lapse_probability(self, rows: pd.DataFrame, days: list[int]) -> tuple[np.ndarray, dict]:
        """P(lapse by day h) for each row (rows x days); every h must be an interval end."""
        if not set(days) <= set(self.ends):
            raise ValueError(f"days {days} must be interval ends {self.ends}")
        needed = [e for e in self.ends if e <= max(days)]
        long = expand(rows, needed, for_fit=False)
        hazard, unseen = self.model.predict(long)
        h = hazard.reshape(len(needed), len(rows)).T  # rows x intervals
        surv = np.cumprod(1 - h, axis=1)
        idx = [needed.index(d) for d in days]
        return 1 - surv[:, idx], unseen
