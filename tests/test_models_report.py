from datetime import date

import yaml

from retention.models.report import cell_rows, registered_cells, summarize

RAW = yaml.safe_load(open("configs/evaluation.yaml"))
JAN, FEB = date(2017, 1, 1), date(2017, 2, 1)


def test_every_registered_cell_is_listed():
    cells = registered_cells(RAW)
    disc = {(c.horizon, c.month, c.against) for c in cells if c.kind == "discrimination"}
    assert disc == {
        (h, m, comp)
        for h, m in ((7, JAN), (7, FEB), (30, JAN))
        for comp in ("lightgbm_classifier", "strongest_simple_baseline")
    }
    calib = {(c.horizon, c.month) for c in cells if c.kind == "calibration"}
    assert calib == {(7, JAN), (7, FEB), (30, JAN)}  # the 59-day demonstration is not a goal cell


def all_met():
    return {c.key: ("win" if c.kind == "discrimination" else "pass") for c in registered_cells(RAW)}


def test_all_cells_met_meets_both_goals():
    s = summarize(registered_cells(RAW), all_met())
    assert s["discrimination"] == {"met": True, "cells": 6, "cells_met": 6, "not_met": []}
    assert s["calibration"]["met"] and s["calibration"]["cells"] == 3


def test_a_not_computable_cell_counts_as_not_met_and_stays_in_the_denominator():
    cells = registered_cells(RAW)
    results = all_met()
    results[("discrimination", 7, FEB, "lightgbm_classifier")] = "not_computable"
    s = summarize(cells, results)["discrimination"]
    assert not s["met"] and s["cells"] == 6 and s["cells_met"] == 5
    assert s["not_met"] == ["7 days, 2017-02-01, lightgbm_classifier: not_computable"]


def test_a_missing_cell_is_listed_as_missing_and_not_met():
    cells = registered_cells(RAW)
    results = all_met()
    del results[("calibration", 30, JAN, "survival")]
    rows = cell_rows(cells, results)
    assert len(rows) == len(cells)
    assert {r["status"] for r in rows if r["kind"] == "calibration"} == {"pass", "missing"}
    assert summarize(cells, results)["calibration"]["cells_met"] == 2


def test_last_row_breakdown_groups():
    from retention.models.report import DESCRIPTIVE_BREAKDOWNS, breakdown_group

    b = DESCRIPTIVE_BREAKDOWNS["last_row"]
    assert [breakdown_group(v, b) for v in ("paid", "trial", "promo", "discount100", None)] == [
        "paid",
        "zero-paid",
        "zero-paid",
        "zero-paid",
        "missing",
    ]
    assert b["decides"] == "nothing"


def test_descriptive_models_decide_nothing_and_notes_exist():
    from retention.models.report import DESCRIPTIVE_MODELS, REQUIRED_NOTES

    assert DESCRIPTIVE_MODELS["dt02"]["decides"] == "nothing"
    assert "after the fold results were seen" in DESCRIPTIVE_MODELS["dt02"]["added"]
    assert any("cox02 and cox04" in n and "0.01" in n for n in REQUIRED_NOTES)
