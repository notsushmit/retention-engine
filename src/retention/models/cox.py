"""Cox proportional hazards (lifelines) and its preprocessing.

Preprocessing, fitted on the fit rows only and stored with the model
(configs/evaluation.yaml, `cox_preprocessing`):
- one missing flag per distinct null pattern of the numeric columns (1 if any
  column of the pattern is null);
- median imputation; winsorising at the 0.5th and 99.5th percentiles;
- z-scaling of continuous columns (0/1 columns and flags stay unscaled);
- one-hot categoricals: levels with at least 100 fit rows, others "other",
  null "missing", unseen at scoring "other" (counted); the most frequent level
  is the reference and gets no column;
- columns that exactly copy or complement an earlier column on the fit rows
  are dropped and reported.
days_until_paid_expiry is never an input; the stratified variant also leaves
out days_until_expiry and fits one baseline per days_until_expiry band.

Probability of a lapse by day h: 1 - S0(h) ** exp(b.x), with S0 the Breslow
baseline (per stratum when stratified), via lifelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from retention.models.metrics import expiry_band
from retention.models.protocol import Threads

ALWAYS_DROPPED = ("days_until_paid_expiry",)
STRATUM = "expiry_band"


@dataclass
class CoxPreprocessor:
    categorical: list[str]
    min_level_rows: int = 100
    winsor: tuple[float, float] = (0.005, 0.995)
    constants: dict[str, dict[str, float]] = field(default_factory=dict)
    flags: dict[str, list[str]] = field(default_factory=dict)
    levels: dict[str, list[str]] = field(default_factory=dict)
    reference: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    unseen: dict[str, int] = field(default_factory=dict)
    fit_values: dict[str, set] = field(default_factory=dict)

    def fit(self, rows: pd.DataFrame, columns: list[str]) -> CoxPreprocessor:
        self.cats = [c for c in columns if c in self.categorical]
        self.numeric = [c for c in columns if c not in self.categorical]
        num = rows[self.numeric].astype("float64")
        null = num.isna()
        patterns: dict[tuple, list[str]] = {}
        for c in self.numeric:
            if null[c].any():
                patterns.setdefault(tuple(null[c].to_numpy()), []).append(c)
        self.flags = {f"missing:{cols[0]}": cols for cols in patterns.values()}
        for c in self.numeric:
            v = num[c].dropna()
            lo, hi = np.quantile(v, self.winsor)
            self.constants[c] = {"median": float(v.median()), "lo": float(lo), "hi": float(hi)}
        clipped = self._numeric(rows)
        for c in self.numeric:
            binary = set(np.unique(clipped[c])) <= {0.0, 1.0}
            mean, sd = float(clipped[c].mean()), float(clipped[c].std())
            scale = not binary and sd > 0
            self.constants[c] |= {"mean": mean if scale else 0.0, "sd": sd if scale else 1.0}
        for c in self.cats:
            v = rows[c].astype("object").where(rows[c].notna(), "missing").astype(str)
            counts = v.value_counts()
            self.fit_values[c] = set(counts.index)
            kept = sorted(counts[counts >= self.min_level_rows].index)
            mapped = v.where(v.isin(kept), "other")
            self.reference[c] = mapped.value_counts().idxmax()
            self.levels[c] = sorted(set(mapped) - {self.reference[c]})
        self.dropped = []
        design = self._design(rows, count_unseen=False)
        keep: list[str] = []
        for c in design.columns:
            x = design[c].to_numpy()
            if np.ptp(x) == 0 or any(
                np.array_equal(x, design[k].to_numpy())
                or np.array_equal(x, 1 - design[k].to_numpy())
                for k in keep
            ):
                self.dropped.append(c)
            else:
                keep.append(c)
        self.columns = keep
        return self

    def _numeric(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            k = self.constants[c]
            v = rows[c].astype("float64").fillna(k["median"]).clip(k["lo"], k["hi"])
            out[c] = v.to_numpy()
        return pd.DataFrame(out, index=rows.index)

    def _design(self, rows: pd.DataFrame, count_unseen: bool = True) -> pd.DataFrame:
        num = self._numeric(rows)
        for c in self.numeric:
            k = self.constants.get(c, {})
            if "sd" in k:
                num[c] = (num[c] - k["mean"]) / k["sd"]
        parts = [num]
        for name, cols in self.flags.items():
            parts.append(
                pd.DataFrame(
                    {name: rows[cols].isna().any(axis=1).astype(float).to_numpy()}, index=rows.index
                )
            )
        for c in self.cats:
            v = rows[c].astype("object").where(rows[c].notna(), "missing").astype(str)
            known = set(self.levels[c]) | {self.reference[c]}
            if count_unseen:
                self.unseen[c] = int((~v.isin(self.fit_values[c])).sum())
            v = v.where(v.isin(known), "other")
            parts.append(
                pd.DataFrame(
                    {f"{c}={lv}": (v == lv).astype(float).to_numpy() for lv in self.levels[c]},
                    index=rows.index,
                )
            )
        return pd.concat(parts, axis=1)

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self._design(rows)[self.columns]

    def logged_constants(self, names: list[str]) -> dict[str, dict[str, float]]:
        return {c: self.constants[c] for c in names if c in self.constants}


@dataclass
class CoxModel:
    stratified: bool
    penalizer: float
    categorical: list[str]
    strata_edges: list[int]
    threads: Threads

    def fit(self, rows: pd.DataFrame, columns: list[str]) -> CoxModel:
        drop = set(ALWAYS_DROPPED) | ({"days_until_expiry"} if self.stratified else set())
        self.inputs = [c for c in columns if c not in drop]
        self.prep = CoxPreprocessor(self.categorical).fit(rows, self.inputs)
        df = self._frame(rows).assign(time=rows.time.to_numpy(), event=rows.event.to_numpy())
        self.fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=0.0)
        with self.threads.limit():
            self.fitter.fit(
                df,
                duration_col="time",
                event_col="event",
                strata=[STRATUM] if self.stratified else None,
            )
        # A valid fit is never worse than the no-covariate model; a weak penalty on
        # near-separated data can leave the optimiser far below it (a failed fit).
        self.log_likelihood = float(self.fitter.log_likelihood_)
        self.null_log_likelihood = float(self.fitter._ll_null_)
        self.fit_ok = self.log_likelihood >= self.null_log_likelihood
        return self

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = self.prep.transform(rows)
        if self.stratified:
            x[STRATUM] = expiry_band(
                rows.days_until_expiry.to_numpy(dtype=float), self.strata_edges
            )
        return x.reset_index(drop=True)

    def lapse_probability(self, rows: pd.DataFrame, days: list[int]) -> np.ndarray:
        """P(lapse by day h) for each row (rows x days)."""
        x = self._frame(rows)
        with self.threads.limit():
            surv = self.fitter.predict_survival_function(x, times=days)
        # A stratified model returns one column per row grouped by stratum; select
        # the columns by row label so each probability stays with its customer.
        return 1 - surv[x.index].to_numpy().T

    def summary(self) -> dict[str, Any]:
        return {
            "fit_ok": self.fit_ok,
            "log_likelihood": self.log_likelihood,
            "null_log_likelihood": self.null_log_likelihood,
            "max_abs_coefficient": float(self.fitter.params_.abs().max()),
            "inputs": len(self.inputs),
            "design_columns": len(self.prep.columns),
            "flags": self.prep.flags,
            "dropped": self.prep.dropped,
            "unseen_levels": self.prep.unseen,
        }
