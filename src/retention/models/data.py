"""Model data: labels as of a split date, filtered loading, and the 2017 outcome lock.

- Labels for a fit or calibration role are built with as_of = the split date,
  only for the snapshots that role needs, from the clean transactions table,
  and cached as Parquet with the as_of and snapshots stored in the file.
  `require_as_of` refuses labels built with a later as_of than the split's.
- Scored rows read the final labels.parquet (as_of = the last transaction
  date): a fully observed outcome is the same under any later as_of.
- Every read names its snapshots and columns and goes through a Parquet
  filter; no read may name more than MAX_SNAPSHOTS_PER_READ snapshots.
- Every column of labels.parquet is classified as pre-T or outcome-derived.
  Outcome columns of snapshots on or after the locked date are readable only
  with a TestUnlock.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from retention.features.build import NULLABLE
from retention.labels.config import LabelConfig, load_label_config
from retention.models.protocol import Protocol, TestUnlock, half_sql

# The largest single read in the protocol names 3 snapshots (the survival
# sensitivity fit, or the never-seen reference months).
MAX_SNAPSHOTS_PER_READ = 4
META_KEY = b"retention.labels_as_of"
# Pre-T columns of labels.parquet that membership queries read.
MEMBERSHIP_COLUMNS = ("msno", "snapshot_date", "in_exp30", "segment")

# Every column of labels.parquet, by whether it is known at T.
PRE_T_LABEL_COLUMNS = frozenset(
    {
        "msno",
        "snapshot_date",
        "expiry_date",
        "days_until_expiry",
        "in_exp30",
        "in_expiry_month",
        "segment",
        "any_imputed_expiry",
        "stacked_far_future",
        "left_truncation_risk",
        "paid_expiry_date",
        "paid_state",
    }
)
OUTCOME_LABEL_PREFIXES = (
    "event_observed",
    "days_to_event",
    "churned_",
    "paid_event_observed",
    "paid_days_to_event",
    "paid_churned_",
    # Reliability of the label window [T, T + h] reads months after T.
    "reliability_status_",
    "label_reliable_",
)


def column_kind(name: str) -> str:
    """'pre_t' or 'outcome'; an unclassified column raises."""
    if name in PRE_T_LABEL_COLUMNS:
        return "pre_t"
    if name.startswith(OUTCOME_LABEL_PREFIXES):
        return "outcome"
    raise KeyError(f"labels column {name!r} is not classified as pre-T or outcome")


def _check_read(snapshots: list[date] | tuple[date, ...], columns: list[str]) -> None:
    if not snapshots:
        raise ValueError("a read must name its snapshots")
    if len(snapshots) > MAX_SNAPSHOTS_PER_READ:
        raise ValueError(
            f"a read may name at most {MAX_SNAPSHOTS_PER_READ} snapshots, got {len(snapshots)}"
        )
    if not columns:
        raise ValueError("a read must name its columns")


def _read(path: Path, snapshots: list[date] | tuple[date, ...], columns: list[str]) -> pd.DataFrame:
    _check_read(snapshots, columns)
    table = ds.dataset(path).to_table(
        columns=list(columns), filter=pc.field("snapshot_date").isin(list(snapshots))
    )
    return table.to_pandas(types_mapper=NULLABLE.get, date_as_object=False)


def _lock(
    protocol: Protocol,
    snapshots: list[date] | tuple[date, ...],
    columns: list[str],
    unlock: TestUnlock | None,
) -> None:
    outcome = [c for c in columns if column_kind(c) == "outcome"]
    locked = [t for t in snapshots if t >= protocol.locked_from]
    if outcome and locked and not isinstance(unlock, TestUnlock):
        raise PermissionError(
            f"outcome columns {outcome} of {locked} are locked until the test run"
        )


def load_features(
    path: Path, snapshots: list[date] | tuple[date, ...], columns: list[str]
) -> pd.DataFrame:
    """Named feature columns (plus the keys) for the named snapshots only."""
    cols = list(dict.fromkeys(["msno", "snapshot_date", *columns]))
    return _read(path, snapshots, cols)


def load_labels(
    path: Path,
    snapshots: list[date] | tuple[date, ...],
    columns: list[str],
    protocol: Protocol,
    unlock: TestUnlock | None = None,
) -> pd.DataFrame:
    """Named label columns from a labels file, under the outcome lock."""
    cols = list(dict.fromkeys(["msno", "snapshot_date", *columns]))
    _lock(protocol, snapshots, cols, unlock)
    out = _read(path, snapshots, cols)
    meta = pq.read_schema(path).metadata or {}
    if META_KEY in meta:
        out.attrs["as_of"] = date.fromisoformat(json.loads(meta[META_KEY])["as_of"])
    return out


# --- labels as of a split date ------------------------------------------------------


def cache_path(cache_dir: Path, as_of: date, snapshots: tuple[date, ...]) -> Path:
    names = "_".join(t.isoformat() for t in sorted(snapshots))
    return cache_dir / f"labels_asof_{as_of.isoformat()}__{names}.parquet"


def build_labels_as_of(
    tx: pd.DataFrame, as_of: date, snapshots: tuple[date, ...], base: LabelConfig
) -> pd.DataFrame:
    """Labels for exactly `snapshots`, from transactions dated on or before `as_of`.

    The label-building code is imported here, not at module level, so a model
    process (which only reads cached builds) never loads it.
    """
    from retention.labels.snapshots import build_labels

    cfg = replace(base, as_of=as_of, snapshots=tuple(sorted(snapshots))).validate()
    return build_labels(tx, cfg)


def labels_as_of(
    processed: Path,
    cache_dir: Path,
    as_of: date,
    snapshots: tuple[date, ...],
    protocol: Protocol,
    base: LabelConfig | None = None,
) -> Path:
    """Build (or reuse) the labels of `snapshots` as of `as_of`; returns the cached file."""
    if any(t >= protocol.locked_from for t in snapshots):
        raise PermissionError("labels as of a split date are built only for unlocked snapshots")
    path = cache_path(cache_dir, as_of, snapshots)
    if path.exists():
        return path
    base = base or load_label_config()
    tx = pd.read_parquet(processed / "transactions.parquet")
    labels = build_labels_as_of(tx, as_of, snapshots, base)
    meta = {"as_of": as_of.isoformat(), "snapshots": [t.isoformat() for t in sorted(snapshots)]}
    table = pa.Table.from_pandas(labels, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), META_KEY: json.dumps(meta).encode()}
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def cached_labels(cache_dir: Path, as_of: date, snapshots: tuple[date, ...]) -> Path:
    """The cached build for (as_of, snapshots); model code never builds labels itself."""
    path = cache_path(cache_dir, as_of, snapshots)
    if not path.exists():
        raise FileNotFoundError(
            f"no label build as of {as_of} for {[str(t) for t in snapshots]}; "
            "run `python -m retention.models build-labels` first"
        )
    return path


def fit_label_requests(protocol: Protocol) -> list[tuple[str, date, tuple[date, ...]]]:
    """(split, as_of, snapshots) of every label build a fit or calibration role needs."""
    out = []
    for f in protocol.folds.values():
        snaps = {*f.fit, *([f.calibration.snapshot] if f.calibration else [])}
        out.append((f.name, f.as_of, tuple(sorted(snaps))))
    tf = protocol.test_fit
    out.append(("test", tf.as_of, tuple(sorted({*tf.fit, tf.calibration.snapshot}))))
    # Registered sensitivity: the survival model also fitted on the check month
    # (2016-11-01), partly observed as of the test pipeline's date.
    snaps = tuple(sorted({*tf.fit, tf.calibration.snapshot, protocol.check_month}))
    out.append(("test_sensitivity", tf.as_of, snaps))
    return out


def require_as_of(labels: pd.DataFrame, split_as_of: date) -> None:
    """Raise unless `labels` were built with an as_of no later than the split's."""
    as_of = labels.attrs.get("as_of")
    if as_of is None:
        raise ValueError("labels carry no as_of; fit and calibration labels must come from a build")
    if as_of > split_as_of:
        raise ValueError(f"labels built as of {as_of} are later than the split date {split_as_of}")


# --- membership (pre-T columns only) ----------------------------------------------


def membership(
    labels_path: Path,
    snapshots: list[date] | tuple[date, ...],
    protocol: Protocol,
    p2_main_only: bool = False,
) -> pd.DataFrame:
    """Customer, snapshot, in_exp30, segment and half for the given snapshots."""
    _check_read(snapshots, list(MEMBERSHIP_COLUMNS))
    dates = ", ".join(f"DATE '{d}'" for d in snapshots)
    where = f"snapshot_date in ({dates})"
    if p2_main_only:
        segs = ", ".join(f"'{s}'" for s in protocol.main_segments)
        where += f" and in_exp30 and segment in ({segs})"
    cols = ", ".join(MEMBERSHIP_COLUMNS)
    sql = f"select {cols}, {half_sql(protocol)} as half from read_parquet(?) where {where}"
    return duckdb.execute(sql, [str(labels_path)]).df()
