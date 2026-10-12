"""Rows for a fold: fit rows with labels as of the fold date, scored rows with final labels.

Fit rows read the cached label build for (fold as_of, fit snapshots) and
refuse labels built later than the fold date. Scored rows read the final
labels.parquet: a fully observed outcome is the same under any later as_of.
Both join the named feature columns on (customer, snapshot), keep P2 main rows
(in_exp30, main segments), and, for a half of the October snapshot, only that
half's customers. Targets: churned_7 and churned_30 from the labels, lapse
by day 59 derived from event_observed and days_to_event.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from retention.data.sample import bucket
from retention.models.data import cached_labels, load_features, load_labels, require_as_of
from retention.models.protocol import Protocol, Slice

KEYS = ["msno", "snapshot_date"]


def target_columns(horizon: int) -> list[str]:
    return [f"churned_{horizon}"] if horizon in (7, 30) else ["event_observed", "days_to_event"]


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Column `y`: 1 if the customer lapsed within `horizon` days; rows must be fully observed."""
    if horizon in (7, 30):
        y = df[f"churned_{horizon}"]
    else:
        unknown = (df.event_observed == 0) & (df.days_to_event < horizon)
        if unknown.any():
            raise ValueError(f"{int(unknown.sum())} rows are censored before day {horizon}")
        y = ((df.event_observed == 1) & (df.days_to_event <= horizon)).astype("Int8")
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} rows have no {horizon}-day outcome")
    return df.assign(y=y.astype(int).to_numpy())


def _join(
    features: Path,
    labels: pd.DataFrame,
    snapshots: list[date],
    feature_cols: list[str],
    protocol: Protocol,
) -> pd.DataFrame:
    feats = load_features(features, snapshots, feature_cols)
    # segment (and paid_state) are copied unchanged from labels into features in
    # Phase 2, whose build stops if they differ; keep the features' copy.
    labels = labels.drop(
        columns=[c for c in labels.columns if c in feats.columns and c not in KEYS]
    )
    labels = labels.assign(snapshot_date=labels.snapshot_date.astype("datetime64[ns]"))
    feats = feats.assign(snapshot_date=feats.snapshot_date.astype("datetime64[ns]"))
    df = feats.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    keep = df.in_exp30.astype(bool) & df.segment.isin(protocol.main_segments)
    return df[keep].reset_index(drop=True)


def half_mask(msno: pd.Series, half: str, protocol: Protocol) -> np.ndarray:
    parity = protocol.halves[half]
    return np.array([bucket(m) % 2 == parity for m in msno])


def fit_rows(
    protocol: Protocol,
    processed: Path,
    cache: Path,
    as_of: date,
    snapshots: tuple[date, ...],
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    path = cached_labels(cache, as_of, tuple(sorted(snapshots)))
    labels = load_labels(
        path, list(snapshots), ["in_exp30", "segment", *target_columns(horizon)], protocol
    )
    require_as_of(labels, as_of)
    df = _join(processed / "features.parquet", labels, list(snapshots), feature_cols, protocol)
    return add_target(df, horizon)


def survival_fit_rows(
    protocol: Protocol,
    processed: Path,
    cache: Path,
    as_of: date,
    snapshots: tuple[date, ...],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Fit rows for a survival model: `time` (days to the lapse or the stop day) and `event`."""
    path = cached_labels(cache, as_of, tuple(sorted(snapshots)))
    cols = ["in_exp30", "segment", "event_observed", "days_to_event"]
    labels = load_labels(path, list(snapshots), cols, protocol)
    require_as_of(labels, as_of)
    df = _join(processed / "features.parquet", labels, list(snapshots), feature_cols, protocol)
    return df.assign(
        time=df.days_to_event.astype(int).to_numpy(), event=df.event_observed.astype(int).to_numpy()
    )


def score_rows(
    protocol: Protocol,
    processed: Path,
    where: Slice,
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    snaps = [where.snapshot]
    labels = load_labels(
        processed / "labels.parquet",
        snaps,
        ["in_exp30", "segment", *target_columns(horizon)],
        protocol,
    )
    df = _join(processed / "features.parquet", labels, snaps, feature_cols, protocol)
    if where.half is not None:
        df = df[half_mask(df.msno, where.half, protocol)].reset_index(drop=True)
    return add_target(df, horizon)
