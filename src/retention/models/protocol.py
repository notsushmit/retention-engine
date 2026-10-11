"""Evaluation protocol: the splits, purge rule, October customer halves and threads.

A typed view of configs/evaluation.yaml, checked when it loads:

- Purge rule: every snapshot that fits or calibrates a model judged at date D
  must be fully observed at D for the horizon h: T + h + 30 <= D.
- Locked months: fold, check-month, fit and calibration roles may not name a
  snapshot on or after `test_guard.locked_outcomes_from` (2017-01-01). Only the
  test pipeline names those months, and it is handed out only against a
  TestUnlock, which the test guard issues after its checks pass.

Membership queries read only pre-T columns of labels.parquet (customer,
snapshot, in_exp30, segment), never an outcome.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from retention.data.sample import bucket_sql

EVAL_CONFIG = Path("configs/evaluation.yaml")
# Pre-T columns of labels.parquet that membership queries may read.
MEMBERSHIP_COLUMNS = ("msno", "snapshot_date", "in_exp30", "segment")
PARITY = {"even": 0, "odd": 1}


class ProtocolError(ValueError):
    """The evaluation config breaks the purge rule or the locked-month rule."""


GUARD_MODULE = "retention.models.guard"
GUARD_FUNCTION = "open_test_session"


class TestUnlock:
    """Permission to name the test months.

    There is no key to import. The constructor checks that its direct caller is
    `open_test_session` in retention/models/guard.py (module name, function name
    and source file), which creates it only after the guard's checks pass. This
    stops accidental use; Python cannot stop deliberate circumvention (code can
    patch this class), which is why the proof of order is the pushed commit
    history, not this check.
    """

    __test__ = False  # not a pytest class

    def __init__(self) -> None:
        caller = sys._getframe(1)
        guard_file = Path(__file__).with_name("guard.py").resolve()
        if (
            caller.f_globals.get("__name__") != GUARD_MODULE
            or caller.f_code.co_name != GUARD_FUNCTION
            or Path(caller.f_code.co_filename).resolve() != guard_file
        ):
            raise PermissionError(
                f"only {GUARD_MODULE}.{GUARD_FUNCTION} can unlock the test months"
            )


@dataclass(frozen=True)
class Slice:
    """A snapshot, or one customer half of it ("selection" or "calibration")."""

    snapshot: date
    half: str | None = None


@dataclass(frozen=True)
class Fold:
    name: str
    horizon: int
    as_of: date
    fit: tuple[date, ...]
    score: Slice
    calibration: Slice | None
    decides: str


@dataclass(frozen=True)
class TestFit:
    """The test pipeline's fit and calibration roles (no test month in it)."""

    __test__ = False
    as_of: date
    fit: tuple[date, ...]
    calibration: Slice


@dataclass(frozen=True)
class TestMonths:
    """The scored test months per horizon, and the as_of of their labels."""

    __test__ = False
    label_as_of: date
    months: dict[int, tuple[date, ...]]


@dataclass(frozen=True)
class Threads:
    """One thread count, applied to every library from here."""

    n: int

    def lightgbm(self) -> dict[str, int]:
        return {"num_threads": self.n}

    def xgboost(self) -> dict[str, int]:
        return {"nthread": self.n}

    @contextmanager
    def limit(self) -> Iterator[None]:
        """Limit every OpenMP and BLAS pool (scikit-learn, numpy, scipy) to n threads."""
        with threadpool_limits(limits=self.n):
            yield


@dataclass(frozen=True)
class Protocol:
    seed: int
    horizons: tuple[int, ...]
    purge_days: int
    main_segments: tuple[str, ...]
    halves_snapshot: date
    halves: dict[str, int]  # half name -> md5 bucket parity
    folds: dict[str, Fold]
    check_month: date
    test_fit: TestFit
    locked_from: date
    threads: Threads
    _test_months: TestMonths

    def fold(self, name: str) -> Fold:
        return self.folds[name]

    def test_months(self, unlock: TestUnlock) -> TestMonths:
        if not isinstance(unlock, TestUnlock):
            raise PermissionError("the test months need a TestUnlock from the test guard")
        return self._test_months


def _date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _slice(raw: dict[str, Any]) -> Slice:
    return Slice(_date(raw["snapshot"]), raw.get("half"))


def fully_observed(snapshot: date, horizon: int, as_of: date, purge_days: int) -> bool:
    """True if every outcome within `horizon` days of `snapshot` is known at `as_of`."""
    return snapshot + timedelta(days=horizon + purge_days) <= as_of


def check_role(
    role: str,
    snapshots: tuple[date, ...],
    horizon: int,
    as_of: date,
    purge_days: int,
    locked_from: date,
) -> tuple[date, ...]:
    """Validate the snapshots of a fit, calibration, fold-score or check-month role."""
    for t in snapshots:
        if t >= locked_from:
            raise ProtocolError(f"{role}: {t} is on or after the locked date {locked_from}")
        if role.startswith(("fit", "calibration")) and not fully_observed(
            t, horizon, as_of, purge_days
        ):
            raise ProtocolError(
                f"{role}: {t} + {horizon} + {purge_days} days is after as_of {as_of} "
                "(purge rule: only fully observed snapshots fit or calibrate)"
            )
    return snapshots


def parse_protocol(raw: dict[str, Any]) -> Protocol:
    purge = int(raw["purge_days_after_horizon"])
    locked = _date(raw["test_guard"]["locked_outcomes_from"])
    halves_raw = raw["customer_halves"]
    halves = {
        "selection": PARITY[halves_raw["selection"]],
        "calibration": PARITY[halves_raw["calibration"]],
    }
    if halves["selection"] == halves["calibration"]:
        raise ProtocolError("customer_halves: selection and calibration must differ")
    halves_snapshot = _date(halves_raw["snapshot"])

    folds = {}
    for name, f in raw["folds"].items():
        h, as_of = int(f["horizon"]), _date(f["as_of"])
        fit = tuple(_date(t) for t in f["fit"])
        score = _slice(f["score"])
        cal = _slice(f["calibration"]) if "calibration" in f else None
        check_role(f"fit of {name}", fit, h, as_of, purge, locked)
        if cal is not None:
            check_role(f"calibration of {name}", (cal.snapshot,), h, as_of, purge, locked)
        check_role(f"score of {name}", (score.snapshot,), h, as_of, purge, locked)
        if any(t >= score.snapshot for t in fit) or (cal and cal.snapshot >= score.snapshot):
            raise ProtocolError(f"{name}: fit and calibration must come before the scored month")
        if score.half is not None and score.snapshot != halves_snapshot:
            raise ProtocolError(f"{name}: halves exist only for {halves_snapshot}")
        folds[name] = Fold(name, h, as_of, fit, score, cal, f["decides"])

    check_month = _date(raw["check_month"])
    check_role("check month", (check_month,), 0, check_month, purge, locked)

    t = raw["test"]
    fit_as_of = _date(t["fit_as_of"])
    fit = tuple(_date(x) for x in t["fit"])
    cal = _slice(t["calibration"])
    months = {int(h): tuple(_date(x) for x in m) for h, m in t["months"].items()}
    for h, d in t["demonstration"].items():
        months[int(h)] = (_date(d["month"]),)
    label_as_of = _date(t["label_as_of"])
    for h in months:
        check_role("fit of the test pipeline", fit, h, fit_as_of, purge, locked)
        check_role("calibration of the test pipeline", (cal.snapshot,), h, fit_as_of, purge, locked)
        for m in months[h]:
            if m < locked:
                raise ProtocolError(f"test month {m} is before the locked date {locked}")
            if not fully_observed(m, h, label_as_of, purge):
                raise ProtocolError(f"test month {m} is not fully observed at {label_as_of}")
    if check_month in fit or check_month == cal.snapshot:
        raise ProtocolError("the check month may not fit or calibrate the test pipeline")
    if cal.half != "calibration" or cal.snapshot != halves_snapshot:
        raise ProtocolError("the test pipeline calibrates on the calibration half")

    return Protocol(
        seed=int(raw["seed"]),
        horizons=tuple(int(h) for h in raw["population"]["horizons"]),
        purge_days=purge,
        main_segments=tuple(raw["population"]["main_segments"]),
        halves_snapshot=halves_snapshot,
        halves=halves,
        folds=folds,
        check_month=check_month,
        test_fit=TestFit(fit_as_of, fit, cal),
        locked_from=locked,
        threads=Threads(int(raw["execution"]["threads"])),
        _test_months=TestMonths(label_as_of, months),
    )


def load_protocol(path: Path = EVAL_CONFIG) -> Protocol:
    return parse_protocol(yaml.safe_load(path.read_text()))


# --- membership (pre-T columns only) ----------------------------------------------


def half_sql(protocol: Protocol, column: str = "msno") -> str:
    """SQL expression naming a customer's half from the md5 bucket parity."""
    sel = protocol.halves["selection"]
    return f"case when {bucket_sql(column)} % 2 = {sel} then 'selection' else 'calibration' end"


def membership(
    labels_path: Path,
    snapshots: list[date] | tuple[date, ...],
    protocol: Protocol,
    p2_main_only: bool = False,
) -> pd.DataFrame:
    """Customer, snapshot, in_exp30, segment and half for the given snapshots."""
    if not snapshots:
        raise ValueError("membership needs at least one snapshot")
    dates = ", ".join(f"DATE '{d}'" for d in snapshots)
    where = f"snapshot_date in ({dates})"
    if p2_main_only:
        segs = ", ".join(f"'{s}'" for s in protocol.main_segments)
        where += f" and in_exp30 and segment in ({segs})"
    cols = ", ".join(MEMBERSHIP_COLUMNS)
    sql = f"select {cols}, {half_sql(protocol)} as half from read_parquet(?) where {where}"
    return duckdb.execute(sql, [str(labels_path)]).df()
