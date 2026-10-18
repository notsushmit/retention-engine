"""The explained model: the frozen classifier refit per horizon and checked against the rehearsal.

Phase 3 kept its fitted models in memory only, so the classifier that reasons
explain is refit here exactly as the test stage fits it:

- settings from configs/choices.yaml (feature set, candidate per horizon) and
  configs/evaluation.yaml (categorical columns, fixed parameters, seed, threads);
- fit rows: P2 main rows of the test pipeline's fit months, from the cached label
  build as of the test pipeline's date that also holds the calibration month
  (the same file, read through the same loader, split the same way);
- Platt on the calibration half of the calibration month.

Only the 7-day and 30-day models are refit; the 59-day model is not explained.

The refit is verified against the raw-score summaries the 2016 rehearsal logged
(rehearsal.json, written at the same commit and from the same inputs as the test
run): every field must be equal, floats compared exactly after a JSON round
trip. The test run's own result file is never opened. A mean depends on the
order its values are summed in, so the rows are read and joined exactly as the
rehearsal read them.

The 2016-12-01 cell is a fingerprint only: its 7-day raw scores are summarised
for this check and nothing else. No reason or archetype is ever built on
2016-12-01 (its outcomes are not usable at 7 or 30 days).

Rows for any snapshot are read from features.parquet plus two columns of
labels.parquet known at T (in_exp30, segment). No outcome column is read for a
scored row, and nothing here can open the locked months.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml

from retention.models.calibration import Platt
from retention.models.classifier import (
    Classifier,
    candidate_params,
    feature_set_columns,
    score_summary,
)
from retention.models.data import column_kind, load_labels
from retention.models.frames import _join, fit_rows, half_mask
from retention.models.protocol import EVAL_CONFIG, Protocol

CHOICES = Path("configs/choices.yaml")
HORIZONS = (7, 30)
# Columns of labels.parquet read for scored rows; each must be known at T.
LABEL_COLUMNS = ("in_exp30", "segment")
PLATT_REVIEW_BAND = (0.9, 1.1)
BASE_VALUE_TOLERANCE = 1e-9

# (horizon, snapshot, path of the logged summary inside rehearsal.json).
# 2016-12-01 is a fingerprint only (see the module docstring).
REPRODUCTION_CELLS: tuple[tuple[int, date, tuple[str, ...]], ...] = (
    (7, date(2016, 11, 1), ("primary", "7 days, 2016-11-01", "models", "classifier")),
    (7, date(2016, 12, 1), ("primary", "7 days, 2016-12-01", "models", "classifier")),
    (30, date(2016, 11, 1), ("primary", "30 days, 2016-11-01", "models", "classifier")),
)
FINGERPRINT_ONLY = date(2016, 12, 1)
SUMMARY_KEY = "score_summary_raw"
FLOAT_FIELDS = ("mean", "min", "max")


# --- settings ----------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    columns: list[str]
    categorical: list[str]
    fixed: dict[str, Any]
    candidates: dict[int, dict[str, Any]]


def load_settings(
    choices_path: Path = CHOICES,
    eval_path: Path = EVAL_CONFIG,
    horizons: tuple[int, ...] = HORIZONS,
) -> Settings:
    """The frozen classifier settings, assembled as the test stage assembles them.

    `horizons` defaults to the explained ones; the scoring service also asks for 59
    (the 59-day refit lives in retention.api, never here).
    """
    raw = yaml.safe_load(eval_path.read_text())
    choices = yaml.safe_load(choices_path.read_text())["choices"]
    drop = raw["selection"]["feature_sets"][choices["feature_set"]]["drop"]
    cands = raw["classifier"]["candidates"]
    return Settings(
        columns=feature_set_columns(drop),
        categorical=list(raw["classifier"]["categorical"]),
        fixed=dict(raw["classifier"]["fixed_params"]),
        candidates={
            h: candidate_params(cands[choices["classifier_hyperparameters"][h]]) for h in horizons
        },
    )


# --- rows ----------------------------------------------------------------------------


def p2_main_rows(
    processed: Path, snapshot: date, columns: list[str], protocol: Protocol
) -> pd.DataFrame:
    """P2 main rows of one snapshot: feature columns plus in_exp30 and segment.

    The join and filter are the ones the rehearsal used, so the row order is the
    same (features file order).
    """
    for c in LABEL_COLUMNS:
        if column_kind(c) != "pre_t":
            raise PermissionError(f"labels column {c!r} is not known at T")
    labels = load_labels(processed / "labels.parquet", [snapshot], list(LABEL_COLUMNS), protocol)
    return _join(processed / "features.parquet", labels, [snapshot], columns, protocol, "p2")


# --- fitting -------------------------------------------------------------------------


def fit_models(
    fit: pd.DataFrame,
    cal: pd.DataFrame,
    settings: Settings,
    horizon: int,
    protocol: Protocol,
) -> tuple[Classifier, Platt]:
    """One classifier and its Platt calibrator, as the test stage fits them."""
    model = Classifier(
        settings.columns,
        settings.categorical,
        settings.candidates[horizon],
        settings.fixed,
        protocol.threads,
        protocol.seed,
    ).fit(fit)
    platt = Platt(protocol.seed).fit(model.predict(cal)[0], cal.y.to_numpy())
    return model, platt


def booster_sha256(model: Classifier) -> str:
    return hashlib.sha256(model.booster.model_to_string().encode()).hexdigest()


def expected_value_from_trees(booster: lgb.Booster) -> float:
    """Sum over trees of leaf values weighted by leaf count / root count (no SHAP)."""
    total = 0.0
    for tree in booster.dump_model()["tree_info"]:
        root = tree["tree_structure"]
        n = root.get("internal_count", root.get("leaf_count"))
        stack, value = [root], 0.0
        while stack:
            node = stack.pop()
            if "leaf_value" in node:
                value += node["leaf_value"] * node.get("leaf_count", n) / n
            else:
                stack += [node["left_child"], node["right_child"]]
        total += value
    return total


def mean_raw_score(model: Classifier, rows: pd.DataFrame) -> float:
    x, _ = model._frame(rows)
    return float(np.mean(model.booster.predict(x, raw_score=True)))


def slope_status(b: float, band: tuple[float, float] = PLATT_REVIEW_BAND) -> str:
    """'in review band' or 'outside review band'; a slope that is not positive raises."""
    if not b > 0:
        raise ValueError(f"Platt slope b = {b} is not positive")
    return "in review band" if band[0] <= b <= band[1] else "outside review band"


@dataclass
class Refit:
    horizon: int
    model: Classifier
    platt: Platt
    fit_rows: int
    calibration_rows: int
    # The fit rows, kept only long enough to compute the base value; never written.
    fit_frame: pd.DataFrame | None = None


def refit_horizon(
    protocol: Protocol, processed: Path, cache: Path, settings: Settings, horizon: int
) -> Refit:
    tf = protocol.test_fit
    snaps = (*tf.fit, tf.calibration.snapshot)
    rows = fit_rows(protocol, processed, cache, tf.as_of, snaps, settings.columns, horizon, "p2")
    fit = rows[rows.snapshot_date.dt.date.isin(tf.fit)]
    cal = rows[rows.snapshot_date.dt.date == tf.calibration.snapshot]
    cal = cal[half_mask(cal.msno, "calibration", protocol)]
    model, platt = fit_models(fit, cal, settings, horizon, protocol)
    return Refit(horizon, model, platt, len(fit), len(cal), fit)


# --- the rehearsal comparison -----------------------------------------------------------


def logged_summary(rehearsal: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """The logged score_summary_raw at `path`; a missing key raises."""
    node: Any = rehearsal
    for key in (*path, SUMMARY_KEY):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"rehearsal.json has no {'/'.join((*path, SUMMARY_KEY))}")
        node = node[key]
    return node


def compare_summary(logged: dict[str, Any], refit: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-by-field exact comparison after a JSON round trip of the refit summary.

    Floats must be equal as floats (no tolerance); deciles element by element;
    the count of distinct values as an integer.
    """
    refit = json.loads(json.dumps(refit))
    rows = []
    for f in FLOAT_FIELDS:
        a, b = logged[f], refit[f]
        rows.append({"field": f, "logged": a, "refit": b, "equal": float(a) == float(b)})
    la, ra = logged["deciles"], refit["deciles"]
    if len(la) != len(ra):
        rows.append({"field": "deciles", "logged": la, "refit": ra, "equal": False})
    for k, (a, b) in enumerate(zip(la, ra, strict=False)):
        rows.append({"field": f"decile_{k + 1}", "logged": a, "refit": b, "equal": a == b})
    a, b = logged["distinct"], refit["distinct"]
    same = isinstance(a, int) and isinstance(b, int) and a == b
    rows.append({"field": "distinct", "logged": a, "refit": b, "equal": same})
    return rows


# --- the whole check --------------------------------------------------------------------


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def run_check(
    protocol: Protocol, processed: Path, cache: Path, settings: Settings
) -> tuple[dict[str, Any], dict[int, Refit]]:
    """Refit twice per horizon, compare with the rehearsal, collect every pass mark.

    Returns the report and the first refit of each horizon. Nothing is written.
    """
    rehearsal = json.loads((processed / "rehearsal.json").read_text())
    report: dict[str, Any] = {"horizons": {}, "reproduction": [], "failures": []}
    refits: dict[int, Refit] = {}
    for h in HORIZONS:
        first = refit_horizon(protocol, processed, cache, settings, h)
        second = refit_horizon(protocol, processed, cache, settings, h)
        sha, sha2 = booster_sha256(first.model), booster_sha256(second.model)
        same_platt = (first.platt.a, first.platt.b) == (second.platt.a, second.platt.b)
        base_mean = mean_raw_score(first.model, first.fit_frame)
        base_trees = expected_value_from_trees(first.model.booster)
        try:
            status = slope_status(first.platt.b)
        except ValueError as e:
            status = "not positive"
            report["failures"].append(f"{h} days: {e}")
        if sha != sha2:
            report["failures"].append(f"{h} days: two refits give different boosters")
        if not same_platt:
            report["failures"].append(f"{h} days: two refits give different Platt parameters")
        if abs(base_mean - base_trees) > BASE_VALUE_TOLERANCE:
            report["failures"].append(
                f"{h} days: base values differ by {abs(base_mean - base_trees):.3e}"
            )
        report["horizons"][str(h)] = {
            "fit_rows": first.fit_rows,
            "calibration_rows": first.calibration_rows,
            "booster_sha256": sha,
            "booster_sha256_second_refit": sha2,
            "platt_a": [first.platt.a, second.platt.a],
            "platt_b": [first.platt.b, second.platt.b],
            "platt_b_status": status,
            "base_value_mean_raw_score": base_mean,
            "base_value_from_trees": base_trees,
            "base_value_difference": abs(base_mean - base_trees),
        }
        first.fit_frame = second = None
        refits[h] = first

    for h, snapshot, path in REPRODUCTION_CELLS:
        rows = p2_main_rows(processed, snapshot, settings.columns, protocol)
        p, _ = refits[h].model.predict(rows)
        table = compare_summary(logged_summary(rehearsal, path), score_summary(p))
        report["reproduction"].append(
            {
                "horizon": h,
                "snapshot": snapshot.isoformat(),
                "rows": len(rows),
                "fingerprint_only": snapshot == FINGERPRINT_ONLY,
                "fields": table,
            }
        )
        bad = [r["field"] for r in table if not r["equal"]]
        if bad:
            report["failures"].append(f"{h} days {snapshot}: differs in {bad}")
    report["passed"] = not report["failures"]
    report["library_versions"] = library_versions()
    return report, refits


# --- saving --------------------------------------------------------------------------------


def save_models(out_dir: Path, refits: dict[int, Refit], expected: dict[int, str]) -> list[Path]:
    """Write each booster and its Platt parameters, only if every SHA-256 is the expected one.

    Nothing is written when any horizon is missing or differs.
    """
    if sorted(expected) != sorted(refits):
        raise ValueError(f"expected SHA-256 for {sorted(expected)}, refit {sorted(refits)}")
    for h, r in refits.items():
        if booster_sha256(r.model) != expected[h]:
            raise ValueError(f"{h} days: booster SHA-256 differs from the expected one")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for h, r in refits.items():
        path = out_dir / f"booster_{h}d.txt"
        path.write_text(r.model.booster.model_to_string())
        meta = {
            "horizon": h,
            "booster_sha256": expected[h],
            "columns": r.model.columns,
            "categorical_levels": r.model.levels,
            "platt_a": r.platt.a,
            "platt_b": r.platt.b,
        }
        meta_path = out_dir / f"model_{h}d.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        written += [path, meta_path]
    return written
