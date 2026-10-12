"""Calibration of raw probabilities, and the ordering fix across horizons.

Per model and horizon (configs/evaluation.yaml, `calibration_steps`):
  1. apply the horizon's calibrator to the raw probability;
  2. clip to [1e-6, 1 - 1e-6];
  3. ordering fix: for each customer, (p7, p30, p59) is replaced by its
     least-squares non-decreasing fit with equal weights (pool adjacent
     violators; a violating pair becomes its average);
  4. clip again;
  5. evaluate slope and intercept on the result.

Platt: logistic regression of the outcome on logit(raw p), with an intercept.
Isotonic: a non-decreasing step function (scikit-learn), allowed only where the
calibration slice has at least `min_events` events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from retention.models.metrics import CLIP, logit


@dataclass
class Platt:
    seed: int

    def fit(self, p: np.ndarray, y: np.ndarray) -> Platt:
        z, self.clipped = logit(p)
        self.model = LogisticRegression(C=np.inf, max_iter=1000, random_state=self.seed)
        self.model.fit(z[:, None], np.asarray(y))
        self.a = float(self.model.intercept_[0])
        self.b = float(self.model.coef_[0, 0])
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z, _ = logit(p)
        return 1 / (1 + np.exp(-(self.a + self.b * z)))


@dataclass
class Isotonic:
    min_events: int

    def fit(self, p: np.ndarray, y: np.ndarray) -> Isotonic:
        events = int(np.asarray(y).sum())
        if events < self.min_events:
            raise ValueError(
                f"isotonic needs at least {self.min_events} events, the slice has {events}"
            )
        self.model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        self.model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(p, dtype=float))


def order_fix(p: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit of each row (equal weights).

    Uses the closed form of isotonic regression: the fitted value at k is
    max over i <= k of min over j >= k of mean(p[i..j]).
    """
    p = np.asarray(p, dtype=float)
    n, m = p.shape
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(p, axis=1)], axis=1)
    out = np.empty_like(p)
    for k in range(m):
        best = np.full(n, -np.inf)
        for i in range(k + 1):
            worst = np.full(n, np.inf)
            for j in range(k, m):
                mean = (csum[:, j + 1] - csum[:, i]) / (j - i + 1)
                worst = np.minimum(worst, mean)
            best = np.maximum(best, worst)
        out[:, k] = best
    return out


def clip(p: np.ndarray) -> tuple[np.ndarray, int]:
    p = np.asarray(p, dtype=float)
    return np.clip(p, CLIP, 1 - CLIP), int(((p < CLIP) | (p > 1 - CLIP)).sum())


def calibrate(raw: np.ndarray, calibrators: list[Any]) -> dict[str, Any]:
    """Apply one calibrator per column of `raw` (rows x horizons) in the registered order."""
    calibrated = np.column_stack([c.predict(raw[:, k]) for k, c in enumerate(calibrators)])
    first, clipped_1 = clip(calibrated)
    fixed = order_fix(first)
    final, clipped_2 = clip(fixed)
    change = np.abs(final - first)
    changed = change.max(axis=1) > 1e-12
    return {
        "before_fix": first,
        "final": final,
        "clipped": clipped_1 + clipped_2,
        "rows_changed": int(changed.sum()),
        "share_changed": float(changed.mean()),
        "mean_change_changed_rows": float(change[changed].mean()) if changed.any() else 0.0,
        "max_change": float(change.max()),
    }
