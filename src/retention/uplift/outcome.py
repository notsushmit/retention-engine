"""The only reader of a KKBox outcome column in the uplift package.

What it reads: paid_event_observed and paid_days_to_event (plus the pre-T keys
paid_state and paid_expiry_date) for the snapshots 2016-08-01 and 2016-09-01 only,
from paid-based labels rebuilt as of 2016-12-31 (no 2017 transaction enters),
through the lock-aware loader.

Outcome: churn at E = 1 if the paid lapse (T + paid_days_to_event, observed) falls on
or before E, the expiry on the unit's own first renewal after T; otherwise 0.

Order and once-only rules:
- Placebo first: the read refuses unless a placebo log exists that was written in this
  run (not before it started), at the same commit, for the same set of units.
- Read once: a marker is written before the read. If the outcome log already exists,
  every further read refuses. If a marker exists without a log (an attempt that began
  reading and did not finish), a repeat needs a stated reason, which is recorded in the
  marker against the earlier attempt and in the new log.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retention.models.data import labels_as_of, load_labels
from retention.models.protocol import Protocol

OUTCOME_COLUMNS = ("paid_event_observed", "paid_days_to_event")
PRE_T_KEYS = ("paid_state", "paid_expiry_date")
AS_OF = date(2016, 12, 31)
SNAPSHOTS = (date(2016, 8, 1), date(2016, 9, 1))
PROCESSED = Path("data/processed/kkbox/sample_020")
LABEL_CACHE = Path("data/interim/kkbox/sample_020/labels_asof")
MARKER = "outcome_read_marker.json"
LOG = "outcome.json"


class PlaceboFirst(RuntimeError):
    """The placebo has not been computed and logged in this run for these units."""


class ReadOnce(RuntimeError):
    """The outcome was already read, or an unfinished attempt needs a stated reason."""


class OutcomeMismatch(RuntimeError):
    """The rebuilt labels do not describe the analysed units as the pre-T data did."""


def now() -> datetime:
    return datetime.now(UTC)


def unit_hash(units: pd.DataFrame) -> str:
    keys = sorted(
        f"{m}|{pd.Timestamp(t).date().isoformat()}"
        for m, t in zip(units.msno, units.snapshot_date, strict=True)
    )
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def write_placebo_log(
    path: Path, payload: dict[str, Any], commit: str, units: pd.DataFrame
) -> Path:
    record = {
        **payload,
        "commit": commit,
        "unit_hash": unit_hash(units),
        "written": now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def require_placebo_log(
    path: Path, commit: str, units: pd.DataFrame, run_started: datetime
) -> None:
    if not path.exists():
        raise PlaceboFirst("no placebo log: the placebo is computed and logged before the outcome")
    log = json.loads(path.read_text())
    if log.get("commit") != commit:
        raise PlaceboFirst(f"the placebo log is from commit {log.get('commit')}, not {commit}")
    if log.get("unit_hash") != unit_hash(units):
        raise PlaceboFirst("the placebo log was computed on a different set of units")
    if datetime.fromisoformat(log["written"]) < run_started:
        raise PlaceboFirst("the placebo log was written before this run started")


def guard_not_read(out_dir: Path, rerun_reason: str | None = None) -> None:
    """Refuse if the outcome was read and logged, or if an unfinished attempt has no reason."""
    if (out_dir / LOG).exists():
        raise ReadOnce(f"{LOG} exists: the KKBox outcome was read and logged; it is read once")
    if (out_dir / MARKER).exists() and not rerun_reason:
        raise ReadOnce(
            "an earlier attempt began reading the outcome and did not finish; "
            "repeat it only with a stated reason"
        )


def begin_read(out_dir: Path, commit: str, rerun_reason: str | None = None) -> dict[str, Any]:
    """Write (or extend) the marker before any outcome column is read."""
    guard_not_read(out_dir, rerun_reason)
    path = out_dir / MARKER
    stamp = {"commit": commit, "started": now().isoformat()}
    if path.exists():
        marker = json.loads(path.read_text())
        marker.setdefault("repeats", []).append({**stamp, "reason": rerun_reason})
    else:
        marker = {"first_attempt": stamp, "repeats": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def churn_at_e(
    snapshot: pd.Series, e: pd.Series, observed: pd.Series, days: pd.Series
) -> np.ndarray:
    """1 if the observed paid lapse (T + days) falls on or before E, else 0."""
    lapse = pd.to_datetime(snapshot) + pd.to_timedelta(days.astype(float), unit="D")
    hit = observed.astype(float).eq(1) & (lapse <= pd.to_datetime(e))
    return hit.to_numpy(dtype=float)


def read_outcome(
    units: pd.DataFrame,
    *,
    out_dir: Path,
    placebo_log: Path,
    commit: str,
    run_started: datetime,
    protocol: Protocol,
    rerun_reason: str | None = None,
    processed: Path = PROCESSED,
    cache_dir: Path = LABEL_CACHE,
) -> np.ndarray:
    """Churn at E for each unit, in the units' row order."""
    require_placebo_log(placebo_log, commit, units, run_started)
    begin_read(out_dir, commit, rerun_reason)
    path = labels_as_of(processed, cache_dir, AS_OF, SNAPSHOTS, protocol)
    lab = load_labels(path, list(SNAPSHOTS), [*OUTCOME_COLUMNS, *PRE_T_KEYS], protocol)
    if lab.attrs.get("as_of") != AS_OF:
        raise OutcomeMismatch(f"labels built as of {lab.attrs.get('as_of')}, not {AS_OF}")
    lab = lab.assign(snapshot_date=pd.to_datetime(lab.snapshot_date))
    keys = units[["msno", "snapshot_date", "paid_expiry_date", "expiry_date"]].assign(
        snapshot_date=pd.to_datetime(units.snapshot_date)
    )
    m = keys.merge(
        lab,
        on=["msno", "snapshot_date"],
        how="left",
        suffixes=("", "_rebuilt"),
        validate="one_to_one",
    )
    bad = ~m.paid_state.eq("active") | (
        pd.to_datetime(m.paid_expiry_date_rebuilt) != pd.to_datetime(m.paid_expiry_date)
    )
    if bad.any():
        raise OutcomeMismatch(f"{int(bad.sum())} units differ in the rebuilt pre-T state")
    return churn_at_e(m.snapshot_date, m.expiry_date, m.paid_event_observed, m.paid_days_to_event)


def placebo_then_outcome(
    compute_placebo: Callable[[], Any],
    write_log: Callable[[Any], Path],
    read: Callable[[Path], np.ndarray],
) -> tuple[Any, Path, np.ndarray]:
    """The only order the command uses: placebo, its log, then the outcome."""
    result = compute_placebo()
    log_path = write_log(result)
    return result, log_path, read(log_path)
