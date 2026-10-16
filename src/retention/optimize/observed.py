"""Observed churn among each strategy's targeted customers: says nothing about whether any
action works.

No KKBox customer was given these actions, so this only shows whether a strategy
aims at customers who went on to lapse. Registered (configs/actions.yaml,
observed_churn_check): churned_30 (coverage-based, the target c01 was trained on),
read through the lock-aware loader with no unlock, 2016-11-01 and 2016-10-01
only, one read per snapshot. The read marker is written before the outcomes are
returned; a second read of the same snapshot is refused.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retention.models.data import load_labels
from retention.models.protocol import Protocol
from retention.optimize.solve import NONE

OUTCOME = "churned_30"
LABEL = "says nothing about whether any action works"
ALLOWED_SNAPSHOTS = (date(2016, 11, 1), date(2016, 10, 1))


def marker_path(out_dir: Path, snapshot: date) -> Path:
    return out_dir / f"observed_read_{snapshot.isoformat()}.json"


def read_once(
    labels_path: Path, snapshot: date, protocol: Protocol, out_dir: Path, git_head: str
) -> pd.DataFrame:
    """msno and churned_30 of one registered 2016 snapshot, read once and logged."""
    if snapshot not in ALLOWED_SNAPSHOTS:
        raise PermissionError(f"{snapshot} is not a registered observed-churn snapshot")
    if snapshot >= protocol.locked_from:
        raise PermissionError(f"{snapshot} is locked")
    marker = marker_path(out_dir, snapshot)
    if marker.exists():
        raise PermissionError(f"{snapshot} outcomes were already read ({marker.name}); never again")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "snapshot": snapshot.isoformat(),
                "column": OUTCOME,
                "read_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "git_head": git_head,
                "label": LABEL,
            },
            indent=2,
        )
        + "\n"
    )
    return load_labels(labels_path, [snapshot], [OUTCOME], protocol)


def shares(
    msno: np.ndarray, choices: dict[str, np.ndarray], outcomes: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Per strategy: targeted, with a defined outcome, lapsed, and the share lapsed."""
    y = outcomes.set_index("msno")[OUTCOME].reindex(msno)
    out = {}
    for name, choice in choices.items():
        targeted = choice != NONE
        known = targeted & y.notna().to_numpy()
        lapsed = int(y[known].astype(int).sum())
        out[name] = {
            "targeted": int(targeted.sum()),
            "with_outcome": int(known.sum()),
            "lapsed": lapsed,
            "share_lapsed": lapsed / int(known.sum()) if known.any() else None,
            "label": LABEL,
        }
    return out
