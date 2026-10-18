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
    fix_rows_changed: int  # as registered; mis-specified (amendment A1), reported as written
    fix_max_change: float
    fix_max_change_tolerance: float
    # Amendment A1: the training-population count, and the logged figures on all P2 rows.
    amendment: str | None = None
    main_rows_changed: int | None = None
    all_p2: dict[str, Any] | None = None


FIX_AMENDS = "pass_marks.fix.rows_changed"


def fix_amendment(o: dict[str, Any]) -> dict[str, Any] | None:
    """The one amendment of the fix's rows-changed mark, if registered."""
    found = [a for a in o.get("amendments", []) if a["amends"] == FIX_AMENDS]
    if len(found) > 1:
        raise ValueError(f"more than one amendment of {FIX_AMENDS}")
    return found[0] if found else None


def ordering_marks(cfg: dict[str, Any] | None = None) -> OrderingMarks:
    o = (cfg if cfg is not None else load_serving())["ordering"]
    refit, marks = o["refit_59"], o["pass_marks"]
    rep, platt, fix = marks["reproduction"], marks["platt_59"], marks["fix"]
    amend = fix_amendment(o)
    main = amend["mark"]["training_population"] if amend else None
    if main is not None and int(main["rows"]) != int(fix["rows"]):
        raise ValueError("amendment A1 names another training-population row count")
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
        amendment=amend["id"] if amend else None,
        main_rows_changed=int(main["rows_changed"]) if main else None,
        all_p2=dict(amend["mark"]["all_p2_tie"]) if amend else None,
    )
