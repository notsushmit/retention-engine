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
