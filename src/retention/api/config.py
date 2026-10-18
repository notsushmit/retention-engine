"""configs/serving.yaml: the scoring service's registered settings and pass marks.

Only the `ordering` section is read into typed form here (the 59-day refit and
the ordering fix). The file is registered in commits that only ever add keys;
tests/test_serving_config.py holds the registered blob hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SERVING_CONFIG = Path("configs/serving.yaml")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_serving(path: Path = SERVING_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@dataclass(frozen=True)
class OrderingMarks:
    """The registered pass marks of the 59-day refit and the ordering fix."""

    horizons: tuple[int, ...]
    clip: tuple[float, float]
    fit_snapshots: tuple[date, ...]
    labels_as_of: date
    threads: int
    seed: int
    rehearsal_path: Path
    rehearsal_sha256: str
    rehearsal_key: tuple[str, ...]
    snapshot: date
    expected_summary: dict[str, Any]
    platt_a: float
    platt_b: float
    fix_rows: int
    fix_rows_changed: int
    fix_max_change: float
    fix_max_change_tolerance: float


def ordering_marks(cfg: dict[str, Any] | None = None) -> OrderingMarks:
    o = (cfg if cfg is not None else load_serving())["ordering"]
    refit, marks = o["refit_59"], o["pass_marks"]
    rep, platt, fix = marks["reproduction"], marks["platt_59"], marks["fix"]
    return OrderingMarks(
        horizons=tuple(o["horizons"]),
        clip=(float(o["clip"][0]), float(o["clip"][1])),
        fit_snapshots=tuple(date.fromisoformat(s) for s in refit["fit_snapshots"]),
        labels_as_of=date.fromisoformat(refit["labels_as_of"]),
        threads=int(refit["threads"]),
        seed=int(refit["seed"]),
        rehearsal_path=Path(rep["source"]),
        rehearsal_sha256=rep["source_sha256"],
        rehearsal_key=tuple(k.strip() for k in rep["key"].split("/")),
        snapshot=date.fromisoformat(rep["snapshot"]),
        expected_summary=dict(rep["expected"]),
        platt_a=float(platt["a"]),
        platt_b=float(platt["b"]),
        fix_rows=int(fix["rows"]),
        fix_rows_changed=int(fix["rows_changed"]),
        fix_max_change=float(fix["max_change"]),
        fix_max_change_tolerance=float(fix["max_change_tolerance"]),
    )
