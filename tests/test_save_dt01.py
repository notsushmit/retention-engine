"""save-dt01: the saved files rebuild dt01 exactly; hashes and Phase 7's file are enforced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.api import __main__ as cli
from retention.api import dt01 as d
from retention.api.saved import ModelRefused
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import load_protocol
from retention.optimize.value import INTERVAL_ENDS_ALL, dt01_settings

P = load_protocol()
RAW = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
PHASE7_DT01 = Path("data/interim/kkbox/sample_020/optimize/model/dt01.txt")
REGISTERED_DT01 = "6d2edd153b89ba149c46de3f647c0b8900a4e22a66fd365d9c927714c2a7338d"
COLS = ["days_until_expiry", "last_is_auto_renew", "secs_per_day_28", "last_payment_method_id"]


def rows(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 30, n)
    auto = rng.integers(0, 2, n)
    lapse = rng.random(n) < 1 / (1 + np.exp(-(-2 + -1.5 * auto)))
    return pd.DataFrame(
        {
            "days_until_expiry": days,
            "last_is_auto_renew": auto,
            "secs_per_day_28": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(2, 900, n)),
            "last_payment_method_id": pd.array(rng.choice([41, 36, 38], n), dtype="Int16"),
            "time": np.where(lapse, days, 90),
            "event": lapse.astype(int),
        }
    )


class S:
    """The parts of Dt01Settings that meta_of reads."""

    columns = COLS


@pytest.fixture(scope="module")
def fitted() -> DiscreteTimeModel:
    cand = {"num_leaves": 7, "n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 20}
    return DiscreteTimeModel(
        False,
        list(INTERVAL_ENDS_ALL),
        cand,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(rows(), COLS)


def saved(fitted, directory: Path) -> dict[str, str]:
    sha = d.text_sha256(d.booster_text(fitted))
    return d.save(fitted, S, directory, sha)


# --- the round trip -------------------------------------------------------------------------------


def test_saved_files_rebuild_dt01_exactly(fitted, tmp_path):
    hashes = saved(fitted, tmp_path)
    loaded = d.load_dt01(tmp_path, hashes, P)
    r = rows(500, seed=1)
    ends = list(INTERVAL_ENDS_ALL)
    a, _ = fitted.lapse_probability(r, ends)
    b, _ = loaded.lapse_probability(r, ends)
    assert np.array_equal(a, b)
    assert d.text_sha256(d.booster_text(loaded)) == hashes[d.BOOSTER]


def test_model_json_holds_what_scoring_needs(fitted, tmp_path):
    hashes = saved(fitted, tmp_path)
    meta = json.loads((tmp_path / d.META).read_text())
    assert meta["booster_sha256"] == hashes[d.BOOSTER]
    assert meta["columns"] == [*COLS, "interval_end"]  # dt01 has no offset input
    assert meta["ends"] == list(INTERVAL_ENDS_ALL) and meta["offset_input"] is False
    assert meta["categorical_levels"] == {"last_payment_method_id": [36, 38, 41]}


def test_save_refuses_another_hash_and_writes_nothing(fitted, tmp_path):
    with pytest.raises(ValueError, match="differs"):
        d.save(fitted, S, tmp_path / "out", "0" * 64)
    assert not (tmp_path / "out").exists()


# --- refusal at load ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [d.BOOSTER, d.META])
def test_a_changed_file_is_refused(fitted, tmp_path, name):
    hashes = saved(fitted, tmp_path)
    data = bytearray((tmp_path / name).read_bytes())
    data[-2] ^= 1
    (tmp_path / name).write_bytes(bytes(data))
    with pytest.raises(ModelRefused, match=name.replace(".", r"\.")):
        d.load_dt01(tmp_path, hashes, P)


@pytest.mark.parametrize("name", [d.BOOSTER, d.META])
def test_a_missing_file_is_refused(fitted, tmp_path, name):
    hashes = saved(fitted, tmp_path)
    (tmp_path / name).unlink()
    with pytest.raises(ModelRefused, match="missing"):
        d.load_dt01(tmp_path, hashes, P)


def test_a_json_naming_another_booster_is_refused(fitted, tmp_path):
    hashes = saved(fitted, tmp_path)
    meta = json.loads((tmp_path / d.META).read_text())
    meta["booster_sha256"] = "0" * 64
    (tmp_path / d.META).write_text(json.dumps(meta))
    hashes[d.META] = hashlib.sha256((tmp_path / d.META).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="another booster"):
        d.load_dt01(tmp_path, hashes, P)


def test_expected_hashes_must_name_both_files(fitted, tmp_path):
    hashes = saved(fitted, tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        d.load_dt01(tmp_path, {d.BOOSTER: hashes[d.BOOSTER]}, P)


# --- the save check, with the real refit replaced by the fitted fixture ---------------------------


class Cfg:
    reproduction_cells = ((7, pd.Timestamp("2016-11-01").date(), ("x", "score_summary_raw")),)


def check(monkeypatch, tmp_path, fitted, expected, phase7_text, fail=None):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "rehearsal.json").write_text("{}")
    phase7 = tmp_path / "dt01.txt"
    if phase7_text is not None:
        phase7.write_text(phase7_text)
    monkeypatch.setattr(d, "refit_dt01", lambda *a: (fitted, 123))
    monkeypatch.setattr(d, "p2_main_rows", lambda *a: pd.DataFrame())
    monkeypatch.setattr(d, "reproduce", lambda *a: ([{"fields": []}], [fail] if fail else []))
    return d.run_save_check(P, processed, tmp_path, S, Cfg, expected, phase7)[0]


def test_save_check_passes_when_every_hash_and_cell_agrees(monkeypatch, tmp_path, fitted):
    text = d.booster_text(fitted)
    report = check(monkeypatch, tmp_path, fitted, d.text_sha256(text), text)
    assert report["passed"] and report["failures"] == []
    assert report["booster_sha256"] == report["phase7_booster_sha256"] == report["expected_sha256"]


def test_save_check_fails_on_another_expected_hash(monkeypatch, tmp_path, fitted):
    report = check(monkeypatch, tmp_path, fitted, "0" * 64, d.booster_text(fitted))
    assert not report["passed"]
    assert any("--expect-sha" in f for f in report["failures"])


@pytest.mark.parametrize("phase7_text", [None, "another booster\n"])
def test_save_check_fails_unless_phase_7_file_has_the_hash(
    monkeypatch, tmp_path, fitted, phase7_text
):
    sha = d.text_sha256(d.booster_text(fitted))
    report = check(monkeypatch, tmp_path, fitted, sha, phase7_text)
    assert not report["passed"]
    assert any("Phase 7" in f for f in report["failures"])


def test_save_check_fails_on_a_reproduction_difference(monkeypatch, tmp_path, fitted):
    text = d.booster_text(fitted)
    report = check(monkeypatch, tmp_path, fitted, d.text_sha256(text), text, fail="dt01 7 days")
    assert not report["passed"] and report["failures"] == ["dt01 7 days"]


# --- the command's guards -------------------------------------------------------------------------


def test_save_dt01_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["save-dt01", "--expect-sha", REGISTERED_DT01])


def test_save_dt01_needs_an_expected_hash():
    with pytest.raises(SystemExit):
        cli.main(["save-dt01"])


def test_save_dt01_runs_once(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    (tmp_path / cli.DT01_REPORT).write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        cli.main(["save-dt01", "--expect-sha", REGISTERED_DT01])


# --- the frozen settings and Phase 7's file -------------------------------------------------------


def test_frozen_dt01_settings_have_no_offset_and_every_interval_end():
    s = dt01_settings()
    assert s.offset_input is False
    assert tuple(s.ends) == INTERVAL_ENDS_ALL
    assert "days_until_expiry" in s.columns


@pytest.mark.skipif(not PHASE7_DT01.exists(), reason="needs Phase 7's saved dt01.txt")
def test_phase_7_dt01_file_has_the_logged_hash():
    assert hashlib.sha256(PHASE7_DT01.read_bytes()).hexdigest() == REGISTERED_DT01
