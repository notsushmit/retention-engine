"""Drift between a reference and a current table, with Evidently's defaults (configs/ops.yaml, R1).

The mechanism only: it computes the statistics, applies the registered margin rule and builds the
body that GET /drift serves. It reads no outcome and no customer ID. The compared columns are the
44 primary features that are not a never-used attribute plus the three final probabilities
(47 columns); the thresholds are the library's defaults and are read and written to a file before
any report is run (`log_defaults`, `run_report` refuses without that file).

Drift on replayed or synthetic data demonstrates the mechanism, not that monitoring is effective in
operation (the registered label below).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retention.ops.telemetry import OPS_CONFIG, telemetry_off
from retention.synthetic import SYNTHETIC_SENTENCE

NEVER_COMPARED = ("tenure_days", "registered_via")  # never-used attributes (R1)
PREDICTIONS = ["p7", "p30", "p59"]
CATEGORICAL = ["segment", "paid_state", "last_row_kind", "last_payment_method_id"]
DEFAULTS_FILE = "drift_defaults.json"
LABEL = (
    "Drift on replayed 2016 data: it demonstrates the mechanism, "
    "not that monitoring is effective in operation."
)
P_VALUE_TESTS = (
    "K-S p_value",
    "Chi-square p_value",
    "Z-test p_value",
)  # drift when value < threshold


def registered() -> dict[str, Any]:
    return yaml.safe_load(OPS_CONFIG.read_text())["drift"]


def compared_columns() -> list[str]:
    """The 47 compared columns, in feature-registry order then the predictions."""
    from retention.features.spec import PRIMARY_FEATURES

    cols = [c for c in PRIMARY_FEATURES if c not in NEVER_COMPARED] + PREDICTIONS
    expected = int(str(registered()["compared"]["count"]).split()[0])
    if len(cols) != expected:
        raise ValueError(f"{len(cols)} compared columns, registered {expected}")
    return cols


def _import_evidently():
    telemetry_off()  # A13: before the first import
    import evidently
    from evidently import DataDefinition, Dataset, Report
    from evidently.legacy.calculations import stattests
    from evidently.presets import DataDriftPreset

    return evidently, DataDefinition, Dataset, Report, stattests, DataDriftPreset


def library_defaults() -> dict[str, Any]:
    """The installed library's defaults for the report this module runs (read, not chosen)."""
    evidently, _, _, _, st, preset = _import_evidently()
    share = inspect.signature(preset.__init__).parameters["drift_share"].default
    tests = {
        "ks": st.ks_stat_test,
        "wasserstein": st.wasserstein_stat_test,
        "jensenshannon": st.jensenshannon_stat_test,
        "chi_square": st.chi_stat_test,
        "z": st.z_stat_test,
    }
    return {
        "evidently_version": evidently.__version__,
        "dataset_drift_share": share,
        "stattests": {
            k: {"name": t.display_name, "default_threshold": t.default_threshold}
            for k, t in tests.items()
        },
        "selection": (
            "reference rows <= 1000: K-S for numeric columns with more than 5 values, "
            "Chi-square for more than 2 and Z-test for 2 values; reference rows > 1000: "
            "normed Wasserstein for numeric columns with more than 5 values, "
            "Jensen-Shannon otherwise"
        ),
        "drift_rule": "p-value tests drift below the threshold, distance tests at or above it",
    }


def log_defaults(out_dir: Path) -> Path:
    """Write the defaults before any report is run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DEFAULTS_FILE
    path.write_text(json.dumps(library_defaults(), indent=1) + "\n")
    return path


def _category(s: pd.Series) -> pd.Series:
    """Category labels as text, whole numbers without a decimal point.

    A code column that holds missing values is a float column in one table and an integer column
    in another; "38.0" and "38" would be two categories (found after the first synthetic run).
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() == s.notna().sum() and (num.dropna() % 1 == 0).all():
        s = num.astype("Int64")
    return s.astype("object").where(s.notna(), None).map(lambda v: None if v is None else str(v))


def _clean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(frame)))
    for c in columns:
        s = frame[c].reset_index(drop=True)
        if c in CATEGORICAL:
            out[c] = _category(s)
        else:
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
    return out


def _drifted(method: str, value: float, threshold: float) -> bool:
    return value < threshold if "p_value" in method.lower() else value >= threshold


def run_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """One Evidently data-drift report on the 47 columns; refuses before the defaults are logged."""
    if not (out_dir / DEFAULTS_FILE).exists():
        raise RuntimeError(f"the library defaults are not logged yet ({out_dir / DEFAULTS_FILE})")
    cols = compared_columns()
    _, DataDefinition, Dataset, Report, _, preset = _import_evidently()
    cat = [c for c in cols if c in CATEGORICAL]
    definition = DataDefinition(
        numerical_columns=[c for c in cols if c not in cat], categorical_columns=cat
    )
    ref, cur = _clean(reference, cols), _clean(current, cols)
    snapshot = Report([preset()]).run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    metrics = snapshot.dict()["metrics"]
    columns = []
    for m in metrics:
        cfg = m["config"]
        if cfg["type"].endswith("ValueDrift"):
            columns.append(
                {
                    "column": cfg["column"],
                    "method": cfg["method"],
                    "threshold": float(cfg["threshold"]),
                    "statistic": float(m["value"]),
                    "drifted": _drifted(cfg["method"], float(m["value"]), float(cfg["threshold"])),
                }
            )
    count = next(m["value"] for m in metrics if m["config"]["type"].endswith("DriftedColumnsCount"))
    drifted = [c for c in columns if c["drifted"]]
    if len(columns) != len(cols) or len(drifted) != int(count["count"]):
        raise RuntimeError(
            f"the per-column reading ({len(drifted)} of {len(columns)}) differs from the "
            f"library's count ({count['count']} of {len(cols)})"
        )
    share = len(drifted) / len(cols)
    return {
        "columns": sorted(columns, key=lambda c: c["column"]),
        "columns_compared": len(cols),
        "columns_drifted": len(drifted),
        "drift_share": share,
        "library_share": float(count["share"]),
        "dataset_alert": share >= library_defaults()["dataset_drift_share"],
        "reference_rows": len(ref),
        "current_rows": len(cur),
    }


def margin_result(unshifted: dict[str, Any], shifted: dict[str, Any]) -> dict[str, Any]:
    """The registered margin rule and alert rule (R1 pass_marks); `met` needs both."""
    marks = registered()["pass_marks"]
    required = float(marks["margin_rule"]["margin"])
    difference = shifted["drift_share"] - unshifted["drift_share"]
    # the shares are fractions of 47 columns, so nothing real lies within 1e-12 of the margin;
    # the allowance is for float rounding only (0.60 - 0.40 is 0.19999999999999996)
    margin_met = difference >= required - 1e-12
    return {
        "required": required,
        "unshifted_share": unshifted["drift_share"],
        "shifted_share": shifted["drift_share"],
        "difference": difference,
        "margin_met": bool(margin_met),
        "shifted_alert": bool(shifted["dataset_alert"]),
        "unshifted_alert": bool(unshifted["dataset_alert"]),
        "met": bool(margin_met and shifted["dataset_alert"]),
    }


GROUP_LABEL = (
    "Group-level mark: added after a result was seen; unvalidated."  # configs/ops.yaml A15
)


def usage_columns() -> list[str]:
    """The usage-group columns among the 47 compared (23)."""
    from retention.features.spec import FEATURE_COLUMNS

    cols = [c for c in compared_columns()[:-3] if FEATURE_COLUMNS[c] == "usage"]
    if len(cols) != 23:
        raise ValueError(f"{len(cols)} usage columns among the compared, A15 names 23")
    return cols


def group_mark(unshifted: dict[str, Any], shifted: dict[str, Any]) -> dict[str, Any]:
    """The group-level mark of amendment A15 (real replay only, unvalidated): the share of the
    23 usage columns that drift in the shifted replay is at least the library's own default
    dataset share (0.5), and the registered margin rule holds. Neither replaces the dataset-level
    result of `margin_result`."""
    group = usage_columns()
    drifted = [c["column"] for c in shifted["columns"] if c["column"] in group and c["drifted"]]
    share = len(drifted) / len(group)
    rule = library_defaults()["dataset_drift_share"]
    margin = margin_result(unshifted, shifted)
    return {
        "group": "usage",
        "group_columns": len(group),
        "drifted_in_group": len(drifted),
        "drifted_columns": sorted(drifted),
        "share": share,
        "rule_share": rule,
        "group_rule_met": bool(share >= rule),
        "margin_rule_met": margin["margin_met"],
        "met": bool(share >= rule and margin["margin_met"]),
        "unvalidated": True,
        "label": GROUP_LABEL,
    }


def daily_events(transactions: list[Path], logs: list[Path], month: str) -> pd.Series:
    """Transaction rows plus listening rows per day of a month (YYYY-MM), zero-filled; the
    files hold YYYYMMDD integers in transaction_date and date."""
    import duckdb

    y, m = int(month[:4]), int(month[5:7])
    lo, hi = y * 10000 + m * 100 + 1, y * 10000 + m * 100 + 31
    parts = [f"select transaction_date d from '{p}'" for p in transactions]
    parts += [f"select date d from '{p}'" for p in logs]
    sql = (
        f"select d, count(*) n from ({' union all '.join(parts)}) "
        f"where d between {lo} and {hi} group by d order by d"
    )
    counts = duckdb.sql(sql).df().set_index("d").n
    index = [y * 10000 + m * 100 + k for k in range(1, pd.Period(month).days_in_month + 1)]
    return counts.reindex(index, fill_value=0)


def batch_size_indicator(reference_days: pd.Series, current_days: pd.Series) -> dict[str, Any]:
    """Events per day against mean +/- 3 sample standard deviations of the reference days."""
    mean, sd = float(reference_days.mean()), float(reference_days.std(ddof=1))
    lo, hi = mean - 3 * sd, mean + 3 * sd
    out = current_days[(current_days < lo) | (current_days > hi)]
    return {
        "band_low": lo,
        "band_high": hi,
        "reference_days": len(reference_days),
        "current_days": len(current_days),
        "days_outside_band": {str(k): int(v) for k, v in out.items()},
        "mark": "none",
    }


def body(
    *,
    shifted_report: dict[str, Any],
    margin: dict[str, Any],
    shifted: bool,
    batch: dict[str, Any],
    reference: dict[str, Any],
    current: dict[str, Any],
    synthetic: bool,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """The fields of GET /drift (serving.yaml ops_endpoints.drift.fields), without the hash."""
    drifted = [
        {k: c[k] for k in ("column", "method", "threshold", "statistic")}
        for c in shifted_report["columns"]
        if c["drifted"]
    ]
    return {
        "status": "monitored",
        "reference": reference,
        "current": current,
        "columns_compared": {"count": shifted_report["columns_compared"], "hidden_by_rule": 0},
        "columns_drifted": drifted,
        "drift_share": shifted_report["drift_share"],
        "dataset_alert": shifted_report["dataset_alert"],
        "margin": margin,
        "margin_met": margin["met"],
        "shifted": shifted,
        "batch_size": batch,
        "library_defaults": defaults,
        "synthetic": synthetic,
        "label": LABEL + (" " + SYNTHETIC_SENTENCE if synthetic else ""),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(json.dumps(content, indent=1, default=float) + "\n")
    return path
