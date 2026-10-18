"""The 59-day classifier refit and the ordering fix, against registered pass marks.

Phase 3 held the 59-day classifier c01 in memory only. The service needs it for
the ordering fix over (p7, p30, p59), so it is refit here exactly as the test
stage fitted it (retention.explain.model.refit_horizon with the frozen settings
for 59 days) and judged by configs/serving.yaml, `ordering.pass_marks`:

1. inputs: the frozen protocol's fit months, label date, threads and seed equal
   the registered ones; every snapshot is before the locked months;
2. reproduction: the raw-score summary on 2016-11-01 P2 main equals the
   registered 13 fields exactly (rehearsal.json is checked against its
   registered SHA-256 first, and its logged summary against the config);
3. Platt at 59 days equals the registered a and b exactly;
4. the fix: saved 7-day and 30-day boosters plus the refit give the registered
   rows changed exactly and max change within its tolerance; every row then
   satisfies p7 <= p30 <= p59;
5. determinism: two refits in one process give the identical booster string and
   Platt parameters; the booster is saved only by a run given its SHA-256.

The descriptive table (no pass mark) reports calibration slope and intercept at
7 and 30 days before and after the fix on the selection half of 2016-10-01
(2016 outcomes only, through the lock-aware loader).

Nothing here constructs a test unlock or opens the test stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retention.api.config import OrderingMarks, file_sha256
from retention.api.ordering import final_probabilities, order_violations, platt_from
from retention.explain.contributions import EXPECTED_SHA256, SavedModel, load_saved_model
from retention.explain.model import (
    SUMMARY_KEY,
    Refit,
    Settings,
    booster_sha256,
    compare_summary,
    library_versions,
    logged_summary,
    p2_main_rows,
    refit_horizon,
)
from retention.models.classifier import score_summary
from retention.models.frames import fit_rows, half_mask
from retention.models.metrics import CLIP, calibration_slope
from retention.models.protocol import Protocol

HORIZON = 59
SAVED_HORIZONS = (7, 30)
# The selection half of 2016-10-01 (P2 main), as registered in the config text.
SELECTION_ROWS = 72_395


class CheckFailed(RuntimeError):
    """A data guard failed before any pass mark could be judged."""


# --- inputs ------------------------------------------------------------------------------


def check_inputs(marks: OrderingMarks, protocol: Protocol, settings: Settings) -> list[str]:
    """Registered inputs against the frozen protocol and settings; returns failures."""
    tf = protocol.test_fit
    failures = []
    if tuple(tf.fit) != marks.fit_snapshots:
        failures.append(f"fit months {tf.fit} differ from the registered {marks.fit_snapshots}")
    if tf.as_of != marks.labels_as_of:
        failures.append(f"label date {tf.as_of} differs from the registered {marks.labels_as_of}")
    if protocol.threads.n != marks.threads:
        failures.append(f"threads {protocol.threads.n} differ from the registered {marks.threads}")
    if protocol.seed != marks.seed:
        failures.append(f"seed {protocol.seed} differs from the registered {marks.seed}")
    if marks.clip != (CLIP, 1 - CLIP):
        failures.append(f"clip {marks.clip} differs from the code's {CLIP}")
    for t in (*tf.fit, tf.calibration.snapshot, marks.snapshot):
        if t >= protocol.locked_from:
            failures.append(f"{t} is in the locked months")
    if HORIZON not in settings.candidates:
        failures.append("no frozen classifier settings for 59 days")
    return failures


def check_rehearsal(processed: Path, marks: OrderingMarks) -> dict[str, Any]:
    """The rehearsal file has its registered hash and logs the registered summary."""
    path = processed / marks.rehearsal_path.name
    sha = file_sha256(path)
    if sha != marks.rehearsal_sha256:
        raise CheckFailed(f"{path} SHA-256 {sha} differs from the registered one")
    key = marks.rehearsal_key
    key = key[:-1] if key and key[-1] == SUMMARY_KEY else key
    logged = logged_summary(json.loads(path.read_text()), key)
    if logged != marks.expected_summary:
        raise CheckFailed("rehearsal.json's logged 59-day summary differs from the config")
    return logged


# --- the pass marks ----------------------------------------------------------------------


def check_platt(a: float, b: float, marks: OrderingMarks) -> list[str]:
    failures = []
    if a != marks.platt_a:
        failures.append(f"Platt a {a!r} differs from the registered {marks.platt_a!r}")
    if b != marks.platt_b:
        failures.append(f"Platt b {b!r} differs from the registered {marks.platt_b!r}")
    return failures


def check_fix(
    raw: np.ndarray, calibrators: list, marks: OrderingMarks
) -> tuple[dict[str, Any], list[str]]:
    """The registered order on rows x (7, 30, 59) raw probabilities, judged by the marks."""
    res = final_probabilities(raw, calibrators)
    violations = order_violations(res["final"])
    summary = {
        "rows": int(raw.shape[0]),
        "clipped": int(res["clipped"]),
        "rows_changed": res["rows_changed"],
        "share_changed": res["share_changed"],
        "mean_change_changed_rows": res["mean_change_changed_rows"],
        "max_change": res["max_change"],
        "max_change_difference": abs(res["max_change"] - marks.fix_max_change),
        "order_violations_after_fix": violations,
    }
    failures = []
    if summary["rows"] != marks.fix_rows:
        failures.append(f"fix rows {summary['rows']} differ from the registered {marks.fix_rows}")
    if summary["rows_changed"] != marks.fix_rows_changed:
        failures.append(
            f"rows changed {summary['rows_changed']} differ from the registered "
            f"{marks.fix_rows_changed}"
        )
    if not summary["max_change_difference"] <= marks.fix_max_change_tolerance:
        failures.append(
            f"max change {summary['max_change']!r} differs from the registered "
            f"{marks.fix_max_change!r} by {summary['max_change_difference']:.3e}"
        )
    if violations:
        failures.append(f"{violations} rows break p7 <= p30 <= p59 after the fix")
    return summary, failures


def raw_probabilities(models: list, rows: pd.DataFrame) -> np.ndarray:
    """rows x len(models) raw probabilities, in model order, on the same rows."""
    return np.column_stack([m.predict(rows)[0] for m in models])


def saved_models(model_dir: Path, settings: Settings, protocol: Protocol) -> list[SavedModel]:
    """The saved 7-day and 30-day boosters, refused unless their registered hashes match."""
    return [
        load_saved_model(model_dir, h, settings, protocol, EXPECTED_SHA256[h])
        for h in SAVED_HORIZONS
    ]


# --- the whole check ---------------------------------------------------------------------


@dataclass
class CheckResult:
    report: dict[str, Any]
    refit: Refit


def run_check(
    marks: OrderingMarks,
    protocol: Protocol,
    processed: Path,
    cache: Path,
    explain_model_dir: Path,
    settings: Settings,
) -> CheckResult:
    """Every registered mark of the 59-day refit; nothing is written here."""
    report: dict[str, Any] = {"failures": check_inputs(marks, protocol, settings)}
    if report["failures"]:
        raise CheckFailed("; ".join(report["failures"]))
    check_rehearsal(processed, marks)

    first = refit_horizon(protocol, processed, cache, settings, HORIZON)
    second = refit_horizon(protocol, processed, cache, settings, HORIZON)
    same_booster = first.model.booster.model_to_string() == second.model.booster.model_to_string()
    same_platt = (first.platt.a, first.platt.b) == (second.platt.a, second.platt.b)
    if not same_booster:
        report["failures"].append("two refits give different boosters")
    if not same_platt:
        report["failures"].append("two refits give different Platt parameters")
    report["refit"] = {
        "fit_rows": first.fit_rows,
        "calibration_rows": first.calibration_rows,
        "booster_sha256": booster_sha256(first.model),
        "booster_sha256_second_refit": booster_sha256(second.model),
        "platt_a": [first.platt.a, second.platt.a],
        "platt_b": [first.platt.b, second.platt.b],
    }
    first.fit_frame = second = None

    report["failures"] += check_platt(first.platt.a, first.platt.b, marks)

    rows = p2_main_rows(processed, marks.snapshot, settings.columns, protocol)
    saved = saved_models(explain_model_dir, settings, protocol)
    raw = raw_probabilities([s.model for s in saved] + [first.model], rows)
    table = compare_summary(marks.expected_summary, score_summary(raw[:, 2]))
    report["reproduction"] = {"snapshot": marks.snapshot.isoformat(), "rows": len(rows)}
    report["reproduction"]["fields"] = table
    bad = [r["field"] for r in table if not r["equal"]]
    if bad:
        report["failures"].append(f"59 days {marks.snapshot}: summary differs in {bad}")

    calibrators = [platt_from(s.platt_a, s.platt_b, protocol.seed) for s in saved]
    report["fix"], fix_failures = check_fix(raw, [*calibrators, first.platt], marks)
    report["failures"] += fix_failures
    report["saved_boosters"] = {str(s.horizon): s.sha256 for s in saved}
    report["passed"] = not report["failures"]
    report["library_versions"] = library_versions()
    return CheckResult(report, first)


# --- the descriptive table ----------------------------------------------------------------


def selection_rows(
    protocol: Protocol, processed: Path, cache: Path, columns: list[str]
) -> pd.DataFrame:
    """Selection half of the calibration month (P2 main) with y7 and y30 attached by msno."""
    tf = protocol.test_fit
    snaps = (*tf.fit, tf.calibration.snapshot)
    out = None
    for h in SAVED_HORIZONS:
        rows = fit_rows(protocol, processed, cache, tf.as_of, snaps, columns, h, "p2")
        rows = rows[rows.snapshot_date.dt.date == tf.calibration.snapshot]
        rows = rows[half_mask(rows.msno, "selection", protocol)]
        if out is None:
            out = rows.rename(columns={"y": f"y{h}"})
        else:
            y = rows.set_index("msno").y
            if set(y.index) != set(out.msno):
                raise CheckFailed(f"{h}-day selection rows differ from the 7-day ones")
            out[f"y{h}"] = out.msno.map(y).to_numpy()
    if len(out) != SELECTION_ROWS:
        raise CheckFailed(f"selection half has {len(out)} rows, not {SELECTION_ROWS}")
    return out


def calibration_table(rows: pd.DataFrame, models: list, calibrators: list) -> dict[str, Any]:
    """Slope and intercept at 7 and 30 days before and after the fix (descriptive)."""
    raw = raw_probabilities(models, rows)
    res = final_probabilities(raw, calibrators)
    table: dict[str, Any] = {"rows": len(rows), "rows_changed": res["rows_changed"]}
    for k, h in enumerate(SAVED_HORIZONS):
        y = rows[f"y{h}"].to_numpy(dtype=float)
        before = calibration_slope(y, res["before_fix"][:, k])
        after = calibration_slope(y, res["final"][:, k])
        table[str(h)] = {
            "events": int(y.sum()),
            "slope_before_fix": before["slope"],
            "intercept_before_fix": before["intercept"],
            "slope_after_fix": after["slope"],
            "intercept_after_fix": after["intercept"],
        }
    return table
