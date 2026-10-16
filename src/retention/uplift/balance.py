"""Covariate balance (configs/uplift.yaml, kkbox.propensity.balance).

SMD = (weighted mean treated - weighted mean untreated)
      / sqrt((var treated + var untreated) / 2),
with unweighted sample variances (ddof = 1) of the units passed in (the trimmed
group). If both variances are 0, the SMD is 0 when the means are equal and
infinite (a failure) otherwise. Variance ratios use weighted variances and are
reported, not judged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PASS_ABS_SMD = 0.10
# Values closer than this count as tied wherever a largest value is named.
TIE_TOL = 1e-9


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((w * x).sum() / w.sum())


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(x, w)
    return float((w * (x - m) ** 2).sum() / w.sum())


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    x, t = np.asarray(x, dtype=float), np.asarray(t)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    tr, un = t == 1, t == 0
    diff = _wmean(x[tr], w[tr]) - _wmean(x[un], w[un])
    denom = np.sqrt((x[tr].var(ddof=1) + x[un].var(ddof=1)) / 2)
    if denom == 0:
        return 0.0 if diff == 0 else float("inf")
    return float(diff / denom)


def variance_ratio(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    x, t, w = np.asarray(x, dtype=float), np.asarray(t), np.asarray(w, dtype=float)
    vu = _wvar(x[t == 0], w[t == 0])
    return float("nan") if vu == 0 else _wvar(x[t == 1], w[t == 1]) / vu


def balance_table(
    x: pd.DataFrame, t: np.ndarray, w: np.ndarray, threshold: float = PASS_ABS_SMD
) -> pd.DataFrame:
    """One row per column: SMD before and after weighting, variance ratio, pass flag."""
    rows = []
    for c in x.columns:
        v = x[c].to_numpy(dtype=float)
        after = smd(v, t, w)
        rows.append(
            {
                "column": c,
                "smd_before": smd(v, t),
                "smd_after": after,
                "variance_ratio_after": variance_ratio(v, t, w),
                "passes": bool(abs(after) <= threshold),
            }
        )
    return pd.DataFrame(rows)


def fill_for_balance(x: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric nulls with the pooled median of all rows given (the analysed units).

    The registered balance fill: one median over every analysed unit, before trimming,
    never a training-fold median. The 0/1 null flags are already columns of `x`.
    """
    return x.fillna(x.median(numeric_only=True))


def tied_max(values: pd.Series, labels: pd.Series, tol: float = TIE_TOL) -> list[str]:
    """Every label whose value is within tol of the largest, in sorted order.

    Naming one first maximum lets floating-point noise pick between tied columns;
    returning all of them, sorted, gives the same list in any row order.
    """
    v = values.to_numpy(dtype=float)
    top = np.nanmax(v)
    return sorted(str(lab) for lab, x in zip(labels, v, strict=True) if x >= top - tol)
