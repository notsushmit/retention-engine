"""Registered cells and their summaries, for reports/model_comparison.md.

Every registered cell is listed with its status, whether or not a result was
logged for it. A summary's denominator is always the number of registered
cells: a "not computable" or missing cell counts as not met, never dropped.
The full report (tables, wording, figures) is built from logged MLflow numbers
in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

MET = {"discrimination": "win", "calibration": "pass"}


@dataclass(frozen=True)
class Cell:
    kind: str  # "discrimination" or "calibration"
    horizon: int
    month: date
    against: str  # comparator (discrimination) or the model judged (calibration)

    @property
    def key(self) -> tuple:
        return (self.kind, self.horizon, self.month, self.against)


def registered_cells(raw: dict[str, Any]) -> list[Cell]:
    """The cells behind the two registered phase goals (the 59-day demonstration is not one)."""
    months = {
        int(h): [m if isinstance(m, date) else date.fromisoformat(m) for m in ms]
        for h, ms in raw["test"]["months"].items()
    }
    cells = [
        Cell("discrimination", h, m, comparator)
        for h, ms in months.items()
        for m in ms
        for comparator in raw["discrimination"]["comparators"]
    ]
    cells += [Cell("calibration", h, m, "survival") for h, ms in months.items() for m in ms]
    return cells


def cell_rows(cells: list[Cell], results: dict[tuple, str]) -> list[dict[str, Any]]:
    """One row per registered cell; a cell without a logged result reads "missing"."""
    return [
        {
            "kind": c.kind,
            "horizon": c.horizon,
            "month": c.month.isoformat(),
            "against": c.against,
            "status": results.get(c.key, "missing"),
            "met": results.get(c.key) == MET[c.kind],
        }
        for c in cells
    ]


def summarize(cells: list[Cell], results: dict[tuple, str]) -> dict[str, dict[str, Any]]:
    """Per goal: met only if every registered cell is met; counts over all registered cells."""
    rows = cell_rows(cells, results)
    out = {}
    for kind in MET:
        mine = [r for r in rows if r["kind"] == kind]
        not_met = [r for r in mine if not r["met"]]
        out[kind] = {
            "met": not not_met,
            "cells": len(mine),
            "cells_met": len(mine) - len(not_met),
            "not_met": [
                f"{r['horizon']} days, {r['month']}, {r['against']}: {r['status']}" for r in not_met
            ],
        }
    return out


# Descriptive breakdowns of the test cells, defined before any 2017 outcome was
# read (added after registration; they change no verdict). For every test cell
# and group: rows, share of rows, lapses, lapse rate, AUC, calibration slope and
# intercept, with customer-bootstrap intervals, or "not computable".
DESCRIPTIVE_BREAKDOWNS = {
    "last_row": {
        "column": "last_row_kind",
        "groups": {"paid": ["paid"], "zero-paid": ["trial", "promo", "discount100"]},
        "statistics": [
            "rows",
            "share_of_rows",
            "lapses",
            "lapse_rate",
            "auc",
            "calibration_slope",
            "calibration_intercept",
        ],
        "decides": "nothing",
    }
}


def breakdown_group(value: object, breakdown: dict[str, Any]) -> str:
    """The group of one row's value; a value in no group (or missing) reads 'missing'."""
    for group, values in breakdown["groups"].items():
        if value in values:
            return group
    return "missing"


# Descriptive model cells, added after the fold results were seen; they change
# no verdict and are never a phase-goal comparator.
DESCRIPTIVE_MODELS = {
    "dt02": {
        "family": "discrete_time",
        "candidate": "dt02",
        "added": "after the fold results were seen",
        "reason": (
            "fold-mean AUC tied the classifier (+0.0003 [-0.0008, +0.0012]) and beat the "
            "registered pick dt01 by +0.019, but it was lower on f30 by 0.0001, so the "
            "registered rule kept dt01"
        ),
        "decides": "nothing",
    }
}

# Statements the report must carry (checked by the report tests).
REQUIRED_NOTES = [
    "Cox candidates cox02 and cox04 (ridge penalty 0.01) failed to fit: the optimiser "
    "diverged on customers whose last row is a trial, ending with a log-likelihood below "
    "the model without inputs. They are excluded, so Cox has two usable candidates "
    "(cox01 and cox03).",
    "The proportional-hazards test ran on a fixed subsample of customers (md5 buckets 0-1 of "
    "the sample), never on all fit rows, because its cost grows faster than quadratically "
    "with rows; it is descriptive.",
]
