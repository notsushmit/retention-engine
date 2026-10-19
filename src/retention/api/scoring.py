"""The scoring function shared by the API and the parity run: status and probabilities.

Status (configs/serving.yaml, contract), for every customer with coverage at T:
- expiry more than 30 days ahead (not in_exp30): outside_population,
  expiry_beyond_30_days, whatever the segment;
- in_exp30 and trial-only or promo-only: outside_population, segment_trial_only
  or segment_promo_only;
- in_exp30 and paid or 100%-discount-only (the training population): scored,
  unless a category value was never seen by the models: outside_population,
  unseen_level:<feature> (refused, never guessed).
Every outside row keeps next_expiry_in_days (days_until_expiry). A null value
inside a feature row stays in population (native missing values).

Probabilities: c01 at 7, 30 and 59 days, Platt per horizon, then the registered
clip, pool-adjacent-violators fix and clip (retention.api.ordering, which calls
Phase 3's calibrate unchanged). A row "moved" when the fix changed any of its
three values by more than 1e-12, the threshold calibrate counts with. Display
text follows Phase 4's rule (one decimal; "under 0.1%" and "over 99.9%").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from retention.api.ordering import HORIZONS, final_probabilities, platt_from
from retention.explain.contributions import SavedModel
from retention.explain.reasons import probability_text
from retention.models.classifier import Classifier

SCORED = "scored"
OUTSIDE = "outside_population"
NOT_ACTIVE = "not_active"
TRIAL = "segment_trial_only"
PROMO = "segment_promo_only"
BEYOND = "expiry_beyond_30_days"
UNSEEN = "unseen_level"
MOVED_THRESHOLD = 1e-12
DECEMBER_2016 = (date(2016, 12, 1), date(2017, 1, 1))  # [start, end)


def population_status(status_rows: pd.DataFrame, main_segments: tuple[str, ...]) -> pd.DataFrame:
    """msno, status, code (None when scored) and next_expiry_in_days for every P1 row."""
    seg = status_rows.segment.astype(str).to_numpy()
    in_window = status_rows.in_exp30.astype(bool).to_numpy()
    known = set(main_segments) | {"trial_only", "promo_only"}
    unknown = sorted(set(seg) - known)
    if unknown:
        raise ValueError(f"segments outside the contract: {unknown}")
    code = np.where(
        ~in_window,
        BEYOND,
        np.where(seg == "trial_only", TRIAL, np.where(seg == "promo_only", PROMO, "")),
    )
    out = pd.DataFrame(
        {
            "msno": status_rows.msno.astype(str).to_numpy(),
            "status": np.where(code == "", SCORED, OUTSIDE),
            "code": pd.Series([c or None for c in code], dtype=object),
            "next_expiry_in_days": status_rows.days_until_expiry.to_numpy(),
        }
    )
    return out


def population_counts(status: pd.DataFrame) -> dict[str, int]:
    """The registered population_counts keys, recomputed from the status table."""
    code = status.code.fillna("")
    return {
        "scored": int((status.status == SCORED).sum()),
        "outside_population_segment_trial_only": int((code == TRIAL).sum()),
        "outside_population_segment_promo_only": int((code == PROMO).sum()),
        "outside_population_expiry_beyond_30_days": int((code == BEYOND).sum()),
        "total_with_coverage": int(len(status)),
    }


def unseen_feature(model: Classifier, rows: pd.DataFrame) -> np.ndarray:
    """Per row, the first category feature whose value the model never saw (None if none)."""
    out = np.full(len(rows), None, dtype=object)
    for c in model.cats:
        values = rows[c].astype("object")
        bad = values.notna().to_numpy() & ~values.isin(model.levels[c]).to_numpy()
        out[bad & pd.isna(out)] = c
    return out


def refuse_unseen(status: pd.DataFrame, rows: pd.DataFrame, models: list[Classifier]):
    """Mark scored rows with an unseen category value as outside_population, unseen_level:<f>.

    Returns the updated status table and the boolean mask of `rows` still scored.
    """
    first = np.full(len(rows), None, dtype=object)
    for m in models:
        f = unseen_feature(m, rows)
        first = np.where(pd.isna(first), f, first)
    refused = pd.Series(first, index=rows.msno.astype(str).to_numpy()).dropna()
    status = status.copy()
    hit = status.msno.isin(refused.index).to_numpy()
    status.loc[hit, "status"] = OUTSIDE
    status.loc[hit, "code"] = [f"{UNSEEN}:{refused[m]}" for m in status.msno[hit]]
    return status, np.array([f is None for f in first])


@dataclass
class Probabilities:
    raw: np.ndarray  # rows x 3 (7, 30, 59 days), the classifiers' raw probabilities
    before_fix: np.ndarray  # Platt per horizon, clipped
    final: np.ndarray  # after the ordering fix, clipped again
    moved: np.ndarray  # bool per row
    summary: dict[str, Any]  # calibrate's counts: rows_changed, max_change, ...

    def text(self) -> np.ndarray:
        """Display text of the final probabilities (rows x 3)."""
        return np.vectorize(probability_text, otypes=[object])(self.final)


def probabilities(c01: dict[int, SavedModel], rows: pd.DataFrame, seed: int) -> Probabilities:
    """The three final probabilities of every row, through the registered order."""
    if sorted(c01) != sorted(HORIZONS):
        raise ValueError(f"need c01 at {HORIZONS} days, got {sorted(c01)}")
    raw = np.column_stack([c01[h].model.predict(rows)[0] for h in HORIZONS])
    cals = [platt_from(c01[h].platt_a, c01[h].platt_b, seed) for h in HORIZONS]
    res = final_probabilities(raw, cals)
    moved = np.abs(res["final"] - res["before_fix"]).max(axis=1) > MOVED_THRESHOLD
    summary = {k: v for k, v in res.items() if k not in ("before_fix", "final")}
    return Probabilities(raw, res["before_fix"], res["final"], moved, summary)


def in_december_2016(snapshot: date) -> bool:
    return DECEMBER_2016[0] <= snapshot < DECEMBER_2016[1]
