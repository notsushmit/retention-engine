"""The 59-day refit: input checks, exact marks, the fix check, guards and the lock."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.api import __main__ as cli
from retention.api import refit as refit_module
from retention.api.config import load_serving, ordering_marks
from retention.api.ordering import platt_from
from retention.api.refit import (
    CheckFailed,
    calibration_table,
    check_fix,
    check_fix_all_p2,
    check_inputs,
    check_platt,
    check_rehearsal,
    p2_all_rows,
    saved_models,
)
from retention.explain.contributions import EXPECTED_SHA256
from retention.explain.model import load_settings
from retention.models.classifier import candidate_params
from retention.models.protocol import load_protocol

P = load_protocol()
MARKS = ordering_marks()
SETTINGS = load_settings(horizons=(7, 30, 59))
SEED = 20161001
EXPLAIN_MODELS = Path("data/interim/kkbox/sample_020/explain/model")
REHEARSAL = Path("data/processed/kkbox/sample_020/rehearsal.json")


# --- inputs ---------------------------------------------------------------------------------


def test_registered_inputs_equal_the_frozen_protocol():
    assert check_inputs(MARKS, P, SETTINGS) == []


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1},
        {"threads": 8},
        {"fit_snapshots": (date(2016, 8, 1),)},
        {"labels_as_of": date(2016, 12, 1)},
        {"clip": (1e-7, 1 - 1e-7)},
        {"snapshot": date(2017, 1, 1)},
    ],
)
def test_a_changed_input_is_a_failure(change):
    assert check_inputs(dataclasses.replace(MARKS, **change), P, SETTINGS) != []


def test_59_day_settings_are_the_frozen_c01():
    raw = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    choices = yaml.safe_load(Path("configs/choices.yaml").read_text())["choices"]
    assert choices["classifier_hyperparameters"][59] == "c01"
    assert SETTINGS.candidates[59] == candidate_params(raw["classifier"]["candidates"]["c01"])


def test_explained_settings_still_have_two_horizons():
    assert sorted(load_settings().candidates) == [7, 30]
    assert check_inputs(MARKS, P, load_settings()) == ["no frozen classifier settings for 59 days"]


# --- exact marks ------------------------------------------------------------------------------


def test_platt_must_be_exactly_equal():
    assert check_platt(MARKS.platt_a, MARKS.platt_b, MARKS) == []
    assert len(check_platt(np.nextafter(MARKS.platt_a, 1.0), MARKS.platt_b, MARKS)) == 1
    assert len(check_platt(MARKS.platt_a, np.nextafter(MARKS.platt_b, 0.0), MARKS)) == 1


def rehearsal_file(tmp_path: Path, summary: dict) -> Path:
    node: dict = {}
    cur = node
    keys = MARKS.rehearsal_key
    for k in keys[:-1]:
        cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = summary
    path = tmp_path / MARKS.rehearsal_path.name
    path.write_text(json.dumps(node))
    return path


def test_rehearsal_with_the_registered_hash_and_summary_passes(tmp_path):
    path = rehearsal_file(tmp_path, MARKS.expected_summary)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    marks = dataclasses.replace(MARKS, rehearsal_sha256=sha)
    assert check_rehearsal(tmp_path, marks) == MARKS.expected_summary


def test_rehearsal_with_another_hash_is_refused(tmp_path):
    rehearsal_file(tmp_path, MARKS.expected_summary)
    with pytest.raises(CheckFailed, match="SHA-256"):
        check_rehearsal(tmp_path, dataclasses.replace(MARKS, rehearsal_sha256="0" * 64))


def test_rehearsal_logging_another_summary_is_refused(tmp_path):
    other = dict(MARKS.expected_summary, mean=np.nextafter(MARKS.expected_summary["mean"], 1.0))
    path = rehearsal_file(tmp_path, other)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CheckFailed, match="differs from the config"):
        check_rehearsal(tmp_path, dataclasses.replace(MARKS, rehearsal_sha256=sha))


# --- the fix check ----------------------------------------------------------------------------


def violating_raw(n: int, bad: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    raw = np.sort(rng.uniform(0.05, 0.95, size=(n, 3)), axis=1)
    raw[:bad] = raw[:bad, ::-1]  # reversed rows break the order
    return raw


IDENTITY = [platt_from(0.0, 1.0, SEED) for _ in range(3)]


def marks_for(raw: np.ndarray, **change) -> object:
    """MARKS with the fix numbers of `raw` as both the registered and the A1 count."""
    summary, _ = check_fix(raw, IDENTITY, MARKS)
    base = {
        "fix_rows": summary["rows"],
        "fix_rows_changed": summary["rows_changed"],
        "main_rows_changed": summary["rows_changed"],
        "fix_max_change": summary["max_change"],
    }
    return dataclasses.replace(MARKS, **(base | change))


def test_fix_check_passes_on_its_own_numbers():
    raw = violating_raw(1_000, 37)
    summary, failures = check_fix(raw, IDENTITY, marks_for(raw))
    assert failures == []
    assert summary["rows_changed"] == 37
    assert summary["order_violations_after_fix"] == 0


def test_fix_check_fails_on_one_row_more_or_a_moved_max():
    raw = violating_raw(1_000, 37)
    _, failures = check_fix(raw, IDENTITY, marks_for(raw, main_rows_changed=38))
    assert any("rows changed" in f for f in failures)
    m = marks_for(raw)
    _, failures = check_fix(
        raw, IDENTITY, dataclasses.replace(m, fix_max_change=m.fix_max_change + 2e-12)
    )
    assert any("max change" in f for f in failures)
    _, failures = check_fix(raw, IDENTITY, dataclasses.replace(m, fix_rows=999))
    assert any("fix rows" in f for f in failures)


def test_amendment_judges_rows_changed_and_reports_the_registered_mark_as_written():
    raw = violating_raw(1_000, 37)
    m = marks_for(raw, fix_rows_changed=16_444)  # the registered, mis-specified count
    summary, failures = check_fix(raw, IDENTITY, m)
    assert failures == []
    assert summary["rows_changed_judged_by"] == "A1"
    assert (summary["registered_rows_changed"], summary["registered_mark_as_written"]) == (
        16_444,
        "not met",
    )


def test_without_the_amendment_the_registered_count_is_judged():
    raw = violating_raw(1_000, 37)
    m = marks_for(raw, fix_rows_changed=16_444, amendment=None, main_rows_changed=None)
    summary, failures = check_fix(raw, IDENTITY, m)
    assert summary["rows_changed_judged_by"] == "the registered mark"
    assert any("16444" in f for f in failures)


def tie_marks(raw: np.ndarray, segment: np.ndarray, **change) -> object:
    summary, _ = check_fix_all_p2(raw, IDENTITY, MARKS, segment)
    tie = {
        k: summary[k]
        for k in (
            "rows",
            "rows_changed",
            "share_changed",
            "mean_change_changed_rows",
            "max_change",
            "order_violations_after_fix",
        )
    }
    return dataclasses.replace(MARKS, all_p2=tie | change)


def test_all_p2_tie_passes_on_its_own_numbers_and_counts_by_segment():
    raw = violating_raw(1_000, 37)
    segment = np.array(["paid"] * 900 + ["trial_only"] * 100)
    segment[:10] = "trial_only"
    summary, failures = check_fix_all_p2(raw, IDENTITY, tie_marks(raw, segment), segment)
    assert failures == []
    assert summary["by_segment"]["trial_only"] == {"rows": 110, "changed": 10}
    assert summary["by_segment"]["paid"] == {"rows": 890, "changed": 27}


@pytest.mark.parametrize("key", ["share_changed", "mean_change_changed_rows"])
def test_all_p2_tie_is_exact_for_share_and_mean_change(key):
    raw = violating_raw(1_000, 37)
    segment = np.array(["paid"] * 1_000)
    m = tie_marks(raw, segment)
    m = dataclasses.replace(m, all_p2=m.all_p2 | {key: np.nextafter(m.all_p2[key], 1.0)})
    _, failures = check_fix_all_p2(raw, IDENTITY, m, segment)
    assert any(key in f for f in failures)


def test_all_p2_tie_fails_on_another_count():
    raw = violating_raw(1_000, 37)
    segment = np.array(["paid"] * 1_000)
    _, failures = check_fix_all_p2(raw, IDENTITY, tie_marks(raw, segment, rows_changed=36), segment)
    assert any("rows_changed" in f for f in failures)


def test_all_p2_rows_read_only_columns_known_at_t(monkeypatch):
    calls = {}

    def labels_spy(path, snapshots, columns, protocol, unlock=None):
        calls["columns"], calls["unlock"] = list(columns), unlock
        return pd.DataFrame()

    def join_spy(features, labels, snapshots, columns, protocol, population):
        calls["population"] = population
        return pd.DataFrame()

    monkeypatch.setattr(refit_module, "load_labels", labels_spy)
    monkeypatch.setattr(refit_module, "_join", join_spy)
    p2_all_rows(Path("x"), date(2016, 11, 1), ["a"], P)
    assert calls == {"columns": ["in_exp30", "segment"], "unlock": None, "population": "p2_all"}


def test_the_registered_amendment_is_parsed():
    assert MARKS.fix_rows_changed == 16_444  # the registered value is not edited
    assert (MARKS.amendment, MARKS.main_rows_changed) == ("A1", 7_632)
    assert MARKS.all_p2 == {
        "rows": 178_594,
        "rows_changed": 16_444,
        "share_changed": 0.09207476175011478,
        "mean_change_changed_rows": 0.014105477095603738,
        "max_change": 0.300535285645459,
        "order_violations_after_fix": 0,
    }
    a1 = load_serving()["ordering"]["amendments"][0]["reconciliation"]
    assert (
        a1["main_rows_changed"] + a1["out_of_population_rows_changed"] == a1["all_p2_rows_changed"]
    )


def test_two_amendments_of_the_fix_mark_are_refused():
    cfg = load_serving()
    cfg["ordering"]["amendments"] = cfg["ordering"]["amendments"] * 2
    with pytest.raises(ValueError, match="more than one"):
        ordering_marks(cfg)


# --- the descriptive table --------------------------------------------------------------------


class Stub:
    def __init__(self, p: np.ndarray) -> None:
        self.p = p

    def predict(self, rows: pd.DataFrame):
        return self.p, {}


def test_calibration_table_reports_both_horizons_before_and_after():
    rng = np.random.default_rng(SEED)
    n = 5_000
    raw = violating_raw(n, 400)
    rows = pd.DataFrame(
        {
            "y7": (rng.random(n) < raw[:, 0]).astype(int),
            "y30": (rng.random(n) < raw[:, 1]).astype(int),
        }
    )
    table = calibration_table(rows, [Stub(raw[:, k]) for k in range(3)], IDENTITY)
    assert table["rows"] == n and table["rows_changed"] == 400
    for h in ("7", "30"):
        assert set(table[h]) == {
            "events",
            "slope_before_fix",
            "intercept_before_fix",
            "slope_after_fix",
            "intercept_after_fix",
        }
        assert table[h]["slope_before_fix"] != table[h]["slope_after_fix"]


# --- guards ---------------------------------------------------------------------------------


def test_run_once_refuses_an_existing_report(tmp_path):
    (tmp_path / "refit59_report.json").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        cli.run_once(tmp_path / "refit59_report.json")
    cli.run_once(tmp_path / "other.json")


def test_commands_refuse_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["refit59"])
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["calibration-table"])


@pytest.mark.parametrize("values", [[], ["7=abc"], ["59="], ["59=abc", "30=def"]])
def test_expect_sha_takes_exactly_one_59_day_hash(values):
    with pytest.raises(SystemExit):
        cli.parse_expected(values)


def test_expect_sha_parses_the_59_day_hash():
    assert cli.parse_expected(["59= abc "]) == "abc"


def test_save_without_expect_sha_is_refused(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    with pytest.raises(SystemExit):
        cli.main(["refit59", "--save"])


# --- the saved 7-day and 30-day boosters --------------------------------------------------------


@pytest.mark.skipif(not EXPLAIN_MODELS.exists(), reason="needs the saved explain boosters")
def test_saved_boosters_load_with_their_registered_hashes():
    saved = saved_models(EXPLAIN_MODELS, SETTINGS, P)
    assert [s.horizon for s in saved] == [7, 30]
    assert [s.sha256 for s in saved] == [EXPECTED_SHA256[7], EXPECTED_SHA256[30]]


@pytest.mark.skipif(not REHEARSAL.exists(), reason="needs the 20% sample's rehearsal.json")
def test_real_rehearsal_passes_its_check():
    assert check_rehearsal(REHEARSAL.parent, MARKS) == MARKS.expected_summary


# --- nothing under src/retention/api can reach the locked months ---------------------------------

FORBIDDEN = (
    "TestUnlock",
    "open_test_session",
    "test_months",
    "scored_month_rows",
    "score_rows",
    "run_test_stage",
    "models.guard",
    "models import guard",
    "score_test",
    "score-test",
    "test_results",
)


def test_api_package_never_names_the_test_session():
    root = Path("src/retention/api")
    hits = [
        (str(p), name)
        for p in sorted(root.rglob("*.py"))
        for name in FORBIDDEN
        if name in p.read_text()
    ]
    assert {"config.py", "refit.py", "ordering.py", "__main__.py"} <= {
        p.name for p in root.glob("*.py")
    }
    assert hits == []
