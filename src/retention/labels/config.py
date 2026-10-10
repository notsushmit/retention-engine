"""Typed loader for the `labels` block of configs/kkbox.yaml.

Unknown keys fail loudly, so a typo cannot silently fall back to a default.
A snapshot is fully observable when its whole follow-up window ends on or
before the last labelable expiry (as_of minus the renewal window); other
snapshots are rejected unless `snapshots.allow_partial` is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LABEL_KEYS = {
    "processed_dir",
    "renewal_window_days",
    "as_of",
    "followup_cap_days",
    "horizons",
    "snapshots",
    "lookback_days",
    "snapshot_includes_same_day_rows",
    "data_start",
    "left_truncation_days",
    "data_quality",
    "trial_max_coverage_days",
    "placeholder_expiry_before",
    "far_future_expiry_after",
    "placeholder_coverage_days",
}
SNAPSHOT_KEYS = {"start", "end", "freq", "dates", "allow_partial"}
QUALITY_KEYS = {
    "first_month",
    "last_month",
    "listening_days",
    "listening_min_active_days",
    "returns_days",
    "batch_baseline_days",
    "mad_k",
    "sensitivity",
    "min_assessed",
    "unreliable_failures",
}


def _date(value: Any, key: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as err:
        raise ValueError(f"labels.{key}: not a YYYY-MM-DD date: {value!r}") from err


def _check_keys(given: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = set(given) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    missing = required - set(given)
    if missing:
        raise ValueError(f"missing key(s) in {where}: {sorted(missing)}")


def _snapshot_dates(spec: dict[str, Any]) -> tuple[date, ...]:
    _check_keys(spec, SNAPSHOT_KEYS, set(), "labels.snapshots")
    has_range = any(k in spec for k in ("start", "end", "freq"))
    if ("dates" in spec) == has_range:
        raise ValueError("labels.snapshots: give either `dates` or `start`/`end`/`freq`, not both")
    if "dates" in spec:
        dates = [_date(d, "snapshots.dates") for d in spec["dates"]]
    else:
        if not all(k in spec for k in ("start", "end", "freq")):
            raise ValueError("labels.snapshots: a range needs `start`, `end` and `freq`")
        start, end = _date(spec["start"], "snapshots.start"), _date(spec["end"], "snapshots.end")
        try:
            dates = [ts.date() for ts in pd.date_range(start, end, freq=spec["freq"])]
        except ValueError as err:
            raise ValueError(f"labels.snapshots.freq: {err}") from err
    if not dates:
        raise ValueError("labels.snapshots: no snapshot dates")
    if len(set(dates)) != len(dates):
        raise ValueError("labels.snapshots: duplicate dates")
    return tuple(sorted(dates))


@dataclass(frozen=True)
class QualityConfig:
    first_month: str
    last_month: str
    listening_days: tuple[int, int]
    listening_min_active_days: int
    returns_days: tuple[int, int]
    batch_baseline_days: int
    mad_k: float
    sensitivity: float
    min_assessed: int
    unreliable_failures: int


def _quality(raw: dict[str, Any]) -> QualityConfig:
    _check_keys(raw, QUALITY_KEYS, QUALITY_KEYS, "labels.data_quality")
    q = QualityConfig(
        first_month=str(raw["first_month"]),
        last_month=str(raw["last_month"]),
        listening_days=tuple(int(d) for d in raw["listening_days"]),  # type: ignore[arg-type]
        listening_min_active_days=int(raw["listening_min_active_days"]),
        returns_days=tuple(int(d) for d in raw["returns_days"]),  # type: ignore[arg-type]
        batch_baseline_days=int(raw["batch_baseline_days"]),
        mad_k=float(raw["mad_k"]),
        sensitivity=float(raw["sensitivity"]),
        min_assessed=int(raw["min_assessed"]),
        unreliable_failures=int(raw["unreliable_failures"]),
    )
    for name in ("listening_days", "returns_days"):
        lo, hi = getattr(q, name)
        if not 0 < lo <= hi:
            raise ValueError(
                f"labels.data_quality.{name} must be [start, end] with 0 < start <= end"
            )
    if not 0 < q.sensitivity < 1 or q.mad_k <= 0:
        raise ValueError("labels.data_quality: mad_k > 0 and 0 < sensitivity < 1")
    return q


@dataclass(frozen=True)
class LabelConfig:
    processed_dir: Path
    renewal_window_days: int
    as_of: date
    followup_cap_days: int
    horizons: tuple[int, ...]
    snapshots: tuple[date, ...]
    allow_partial: bool
    lookback_days: int | None
    snapshot_includes_same_day_rows: bool
    data_start: date
    left_truncation_days: int
    data_quality: QualityConfig
    trial_max_coverage_days: int
    placeholder_expiry_before: date
    far_future_expiry_after: date
    # (payment_method_id, payment_plan_days) -> coverage days for placeholder expiries
    placeholder_coverage_days: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def last_labelable_expiry(self) -> date:
        """Latest expiry whose whole renewal window is visible as of `as_of`."""
        return self.as_of - timedelta(days=self.renewal_window_days)

    def fully_observable(self, snapshot: date, days: int | None = None) -> bool:
        """True if every expiry in [snapshot, snapshot + days] can be labeled."""
        days = self.followup_cap_days if days is None else days
        return snapshot + timedelta(days=days) <= self.last_labelable_expiry

    @property
    def partial_snapshots(self) -> tuple[date, ...]:
        return tuple(t for t in self.snapshots if not self.fully_observable(t))

    def validate(self) -> LabelConfig:
        if self.renewal_window_days <= 0 or self.followup_cap_days <= 0:
            raise ValueError("labels: renewal_window_days and followup_cap_days must be positive")
        if not self.horizons or any(h <= 0 or h > self.followup_cap_days for h in self.horizons):
            raise ValueError(
                f"labels.horizons must lie in 1..{self.followup_cap_days}: {self.horizons}"
            )
        late = [t for t in self.snapshots if t > self.as_of]
        if late:
            raise ValueError(f"labels.snapshots after as_of {self.as_of}: {late}")
        if self.partial_snapshots and not self.allow_partial:
            raise ValueError(
                f"labels.snapshots: {len(self.partial_snapshots)} snapshot(s) have a follow-up "
                "window "
                f"past the last labelable expiry {self.last_labelable_expiry} "
                f"(first: {self.partial_snapshots[0]}); "
                "set snapshots.allow_partial: true to keep them"
            )
        if self.lookback_days is not None and self.lookback_days < 0:
            raise ValueError("labels.lookback_days must be null or >= 0")
        return self


def parse_label_config(raw: dict[str, Any], as_of: date | None = None) -> LabelConfig:
    """Build a LabelConfig from the `labels` mapping; `as_of` overrides the configured date."""
    _check_keys(raw, LABEL_KEYS, LABEL_KEYS, "labels")
    snapshots = raw["snapshots"]
    cfg = LabelConfig(
        processed_dir=Path(raw["processed_dir"]),
        renewal_window_days=int(raw["renewal_window_days"]),
        as_of=_date(raw["as_of"], "as_of"),
        followup_cap_days=int(raw["followup_cap_days"]),
        horizons=tuple(int(h) for h in raw["horizons"]),
        snapshots=_snapshot_dates(snapshots),
        allow_partial=bool(snapshots.get("allow_partial", False)),
        lookback_days=None if raw["lookback_days"] is None else int(raw["lookback_days"]),
        snapshot_includes_same_day_rows=bool(raw["snapshot_includes_same_day_rows"]),
        data_start=_date(raw["data_start"], "data_start"),
        left_truncation_days=int(raw["left_truncation_days"]),
        data_quality=_quality(raw["data_quality"]),
        trial_max_coverage_days=int(raw["trial_max_coverage_days"]),
        placeholder_expiry_before=_date(
            raw["placeholder_expiry_before"], "placeholder_expiry_before"
        ),
        far_future_expiry_after=_date(raw["far_future_expiry_after"], "far_future_expiry_after"),
        placeholder_coverage_days={
            (int(method), int(days)): int(cov)
            for method, by_days in raw["placeholder_coverage_days"].items()
            for days, cov in by_days.items()
        },
    )
    if as_of is not None:
        cfg = replace(cfg, as_of=as_of)
    return cfg.validate()


def load_label_config(
    path: Path = Path("configs/kkbox.yaml"), as_of: date | None = None
) -> LabelConfig:
    return parse_label_config(yaml.safe_load(path.read_text())["labels"], as_of=as_of)
