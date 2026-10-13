from datetime import date
from pathlib import Path

import pytest
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
    assert any("its own regression" in n and "borderline" in n for n in REQUIRED_NOTES)


# --- the comparison report ---------------------------------------------------------------

NOTE = (
    "This interval covers which customers happened to be in that month, not month-to-month change."
)


def test_the_interval_note_is_the_registered_sentence():
    import yaml

    assert yaml.safe_load(open("configs/evaluation.yaml"))["interval_note"] == NOTE


def test_report_check_finds_a_missing_interval_sentence_and_forbidden_text():
    from retention.models.comparison_report import check_report

    good = f"AUC 0.9054 [0.8971, 0.9136]\n\n{NOTE}\n"
    assert check_report(good, NOTE) == []
    assert check_report("AUC 0.9054 [0.8971, 0.9136]\n\nNext.\n", NOTE)
    assert check_report(f"{NOTE}\n\nsee D-999\n", NOTE) == ["a decision ID"]
    assert check_report(f"{NOTE}\n\nfrom /Users/me/x\n", NOTE) == ["an absolute path"]


def test_verdict_wording_is_the_registered_wording():
    import yaml

    from retention.models.comparison_report import verdict_text

    raw = yaml.safe_load(open("configs/evaluation.yaml"))["discrimination"]["verdicts"]
    registered = {k: v.split(": ", 1)[1] for k, v in raw.items()}
    assert verdict_text("win", "X", 7, "2017-01-01") == registered["win"].replace(
        "h days", "7 days"
    ).replace("month M", "2017-01")
    for k in ("below_threshold", "no_difference"):
        assert verdict_text(k, "X", 7, "2017-01-01") == registered[k]
    assert verdict_text("loss", "X", 7, "2017-01-01") == registered["loss"]
    assert verdict_text("mixed", "X", 7, "both") == "mixed: not shown at 7 days"


def test_the_preregistered_intercept_statement_is_quoted_exactly():
    import re

    from retention.models.comparison_report import PREREGISTERED_INTERCEPT

    comments = " ".join(
        line.strip().lstrip("#").strip()
        for line in open("configs/evaluation.yaml")
        if line.strip().startswith("#")
    )
    assert PREREGISTERED_INTERCEPT in re.sub(r"\s+", " ", comments)


def test_the_report_of_the_logged_test_run_keeps_the_rules():
    """Skipped where the tracking store has no finished test run (for example in CI)."""
    import json
    import sqlite3

    import yaml

    from retention.models.comparison_report import check_report, render_report

    db = Path("mlruns/mlflow.db")
    if not db.exists():
        pytest.skip("no tracking store")
    row = (
        sqlite3.connect(db)
        .execute(
            "select r.run_uuid from runs r join tags t on t.run_uuid = r.run_uuid "
            "where t.key = 'run_kind' and t.value = 'test' and r.status = 'FINISHED'"
        )
        .fetchone()
    )
    if row is None:
        pytest.skip("no finished test run")
    run_id = row[0]
    art = Path("mlruns/artifacts") / run_id / "artifacts" / "test_results.json"
    results = json.loads(art.read_text())
    raw = yaml.safe_load(open("configs/evaluation.yaml"))
    tags = dict(
        sqlite3.connect(db).execute("select key, value from tags where run_uuid = ?", (run_id,))
    )
    frozen = yaml.safe_load(open("configs/choices.yaml"))
    meta = {
        "run_id": run_id,
        "tags": tags,
        "config": raw,
        "interval_note": raw["interval_note"],
        "replicates": raw["bootstrap"]["replicates"],
        "choices": frozen["choices"],
        "history": ["a commit"],
    }
    text = render_report(results, meta)
    assert check_report(text, raw["interval_note"]) == []
    assert "macOS arm64" in text and "59-day demonstration" in text
    assert "Added after the fold results were seen" in text
    from retention.models.comparison_report import EXPLORATORY_LIMITS

    assert all(n in text for n in EXPLORATORY_LIMITS)
    assert all(n.startswith("Exploratory, post hoc") for n in EXPLORATORY_LIMITS[1:])
    assert text == Path("reports/model_comparison.md").read_text().replace(
        "\n".join(f"- {h}" for h in _history(Path("reports/model_comparison.md"))), "- a commit"
    )


def _history(report: Path) -> list[str]:
    lines = report.read_text().split("### History of commits\n\n", 1)[1].strip().splitlines()
    return [line[2:] for line in lines]
