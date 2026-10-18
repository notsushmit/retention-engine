"""The ordering fix as the service applies it: the registered Phase 3 code, unchanged.

Raw probabilities of the three classifiers (columns 7, 30 and 59 days) go
through retention.models.calibration.calibrate: Platt per horizon, clip to
[1e-6, 1 - 1e-6], pool adjacent violators with equal weights, clip again. This
module only builds the calibrators from stored parameters and counts rows whose
final probabilities are out of order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from retention.models.calibration import Platt, calibrate

HORIZONS = (7, 30, 59)


def platt_from(a: float, b: float, seed: int) -> Platt:
    """A Platt calibrator from stored parameters (no fit)."""
    platt = Platt(seed)
    platt.a, platt.b = float(a), float(b)
    return platt


def final_probabilities(raw: np.ndarray, calibrators: list[Platt]) -> dict[str, Any]:
    """`raw` is rows x 3 (7, 30, 59 days); returns calibrate()'s result unchanged."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(HORIZONS) or len(calibrators) != len(HORIZONS):
        raise ValueError(f"need rows x {len(HORIZONS)} raw probabilities and one calibrator each")
    return calibrate(raw, calibrators)


def order_violations(final: np.ndarray) -> int:
    """Rows where p7 <= p30 <= p59 does not hold."""
    final = np.asarray(final, dtype=float)
    return int((np.diff(final, axis=1) < 0).any(axis=1).sum())
