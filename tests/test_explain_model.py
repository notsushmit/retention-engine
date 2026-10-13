"""The explained model: exact comparison, determinism, base value, rows, saving, the lock."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from retention.explain import model as em
from retention.explain.model import (
    REPRODUCTION_CELLS,
    Refit,
    Settings,
    compare_summary,
    expected_value_from_trees,
    fit_models,
    load_settings,
    logged_summary,
    mean_raw_score,
    p2_main_rows,
    save_models,
    slope_status,
)
from retention.features.spec import FEATURE_COLUMNS, PRIMARY_FEATURES
from retention.models.classifier import score_summary
from retention.models.data import OUTCOME_LABEL_PREFIXES, column_kind, load_labels
from retention.models.frames import scored_month_rows
from retention.models.protocol import load_protocol

P = load_protocol()
REHEARSAL = Path("data/processed/kkbox/sample_020/rehearsal.json")


# --- the strict comparison ----------------------------------------------------------------


def _summary(p: np.ndarray) -> dict:
    return json.loads(json.dumps(score_summary(p)))


def test_identical_summaries_compare_equal():
    p = np.random.default_rng(0).random(500)
    rows = compare_summary(_summary(p), score_summary(p))
    assert rows and all(r["equal"] for r in rows)
    assert {r["field"] for r in rows} >= {"mean", "min", "max", "distinct", "decile_1", "decile_9"}


def test_one_ulp_nudge_of_the_mean_fails():
    p = np.random.default_rng(1).random(500)
    logged = _summary(p)
    nudged = dict(score_summary(p))
    nudged["mean"] = float(np.nextafter(nudged["mean"], np.inf))
    assert nudged["mean"] != logged["mean"]  # one unit in the last place, nothing more
    rows = {r["field"]: r["equal"] for r in compare_summary(logged, nudged)}
    assert rows["mean"] is False
    assert all(v for f, v in rows.items() if f != "mean")


def test_one_ulp_nudge_of_a_decile_fails():
    p = np.random.default_rng(2).random(500)
    logged = _summary(p)
    nudged = dict(score_summary(p))
    nudged["deciles"] = list(nudged["deciles"])
    nudged["deciles"][4] = float(np.nextafter(nudged["deciles"][4], -np.inf))
    rows = {r["field"]: r["equal"] for r in compare_summary(logged, nudged)}
    assert rows["decile_5"] is False


def test_distinct_count_must_be_an_equal_integer():
    p = np.random.default_rng(3).random(100)
    logged = _summary(p)
    for bad in (logged["distinct"] + 1, float(logged["distinct"])):
        refit = dict(score_summary(p), distinct=bad)
        rows = {r["field"]: r["equal"] for r in compare_summary(logged, refit)}
        assert rows["distinct"] is False


def test_logged_summary_missing_path_raises():
    with pytest.raises(KeyError):
        logged_summary({"primary": {}}, ("primary", "7 days, 2016-11-01", "models", "classifier"))


@pytest.mark.skipif(not REHEARSAL.exists(), reason="needs the 20% sample's rehearsal.json")
def test_every_reproduction_cell_exists_in_the_rehearsal():
    rehearsal = json.loads(REHEARSAL.read_text())
    for _, _, path in REPRODUCTION_CELLS:
        s = logged_summary(rehearsal, path)
        assert set(s) >= {"mean", "deciles", "min", "max", "distinct"}


def test_reproduction_cells_are_the_registered_ones():
    cells = [(h, t) for h, t, _ in REPRODUCTION_CELLS]
    assert cells == [(7, date(2016, 11, 1)), (7, date(2016, 12, 1)), (30, date(2016, 11, 1))]
    assert all(t < P.locked_from for _, t in cells)


# --- settings ---------------------------------------------------------------------------------


def test_settings_match_the_test_stage_assembly():
    s = load_settings()
    raw = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    choices = yaml.safe_load(Path("configs/choices.yaml").read_text())["choices"]
    drop = raw["selection"]["feature_sets"][choices["feature_set"]]["drop"]
    assert s.columns == [c for c in PRIMARY_FEATURES if FEATURE_COLUMNS[c] not in drop]
    assert s.categorical == raw["classifier"]["categorical"]
    assert s.fixed == raw["classifier"]["fixed_params"]
    for h in (7, 30):
        cand = raw["classifier"]["candidates"][choices["classifier_hyperparameters"][h]]
        assert list(s.candidates[h].values()) == cand
    assert sorted(s.candidates) == [7, 30]  # the 59-day model is not refit


# --- fitting on synthetic data ------------------------------------------------------------------


def synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.1] = np.nan
    b = rng.normal(size=n)
    cat = rng.choice(["x", "y", "z"], size=n)
    z = 1.2 * np.nan_to_num(a) - 0.8 * b + np.where(cat == "y", 1.0, 0.0) - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "cat": pd.array(cat, dtype="string"), "y": y})


SETTINGS = Settings(
    columns=["a", "b", "cat"],
    categorical=["cat"],
    fixed=yaml.safe_load(Path("configs/evaluation.yaml").read_text())["classifier"]["fixed_params"],
    candidates={
        7: {"num_leaves": 7, "n_estimators": 30, "learning_rate": 0.1, "min_child_samples": 20}
    },
)


@pytest.fixture(scope="module")
def fitted():
    fit, cal = synthetic(4000, 0), synthetic(1500, 1)
    return fit, cal, fit_models(fit, cal, SETTINGS, 7, P)


def test_two_fits_give_the_identical_booster_and_platt(fitted):
    fit, cal, (m1, p1) = fitted
    m2, p2 = fit_models(fit, cal, SETTINGS, 7, P)
    assert m1.booster.model_to_string() == m2.booster.model_to_string()
    assert em.booster_sha256(m1) == em.booster_sha256(m2)
    assert (p1.a, p1.b) == (p2.a, p2.b)


def test_base_value_from_trees_equals_mean_raw_score_on_fit_rows(fitted):
    fit, _, (m, _) = fitted
    assert abs(expected_value_from_trees(m.booster) - mean_raw_score(m, fit)) <= 1e-9


def test_slope_status():
    assert slope_status(1.0) == "in review band"
    assert slope_status(0.9) == "in review band"
    assert slope_status(0.85) == "outside review band"
    assert slope_status(1.2) == "outside review band"
    for b in (0.0, -0.3, float("nan")):
        with pytest.raises(ValueError):
            slope_status(b)


# --- saving ---------------------------------------------------------------------------------------


def test_save_refuses_a_different_sha_and_writes_nothing(fitted, tmp_path):
    fit, cal, (m, p) = fitted
    refits = {7: Refit(7, m, p, len(fit), len(cal))}
    out = tmp_path / "model"
    with pytest.raises(ValueError, match="differs"):
        save_models(out, refits, {7: "0" * 64})
    with pytest.raises(ValueError):
        save_models(out, refits, {7: em.booster_sha256(m), 30: "0" * 64})
    assert not out.exists()


def test_save_writes_the_booster_with_the_expected_sha(fitted, tmp_path):
    fit, cal, (m, p) = fitted
    sha = em.booster_sha256(m)
    written = save_models(tmp_path / "model", {7: Refit(7, m, p, len(fit), len(cal))}, {7: sha})
    text = (tmp_path / "model" / "booster_7d.txt").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == sha
    meta = json.loads((tmp_path / "model" / "model_7d.json").read_text())
    assert (meta["platt_a"], meta["platt_b"], meta["booster_sha256"]) == (p.a, p.b, sha)
    assert len(written) == 2


# --- rows: order and the outcome lock ------------------------------------------------------------

# Values of very different magnitudes: their floating-point mean depends on the
# order they are summed in, so a row-order bug changes the mean.
X = [1e16, 1.0, 0.1, -1e16, 0.3, 1e-3, 3.0, 1e8, 0.7, -1e8, 1e16, 5.5, -1e16, 0.2]
T16, T17 = date(2016, 11, 1), date(2017, 1, 1)


def write_tables(root: Path) -> list[str]:
    """features.parquet and labels.parquet; returns P2 main customers in features-file order."""
    feats, labels, expected = [], [], []
    for t in (T16, T17):
        # P2 main rows, named in reverse so features order differs from sorted order,
        # with rows outside P2 main in between.
        for i, x in enumerate(X):
            m = f"m{len(X) - i:02d}"
            feats.append({"msno": m, "snapshot_date": t, "segment": "paid", "x": x})
            labels.append({"msno": m, "snapshot_date": t, "in_exp30": True, "segment": "paid"})
            if t == T16:
                expected.append(m)
            if i % 4 == 1:
                # Outside P2 main: a trial-only row inside the 30-day window, or a
                # paid row outside it.
                o = f"o{i:02d}"
                seg = "trial_only" if i % 8 == 1 else "paid"
                feats.append({"msno": o, "snapshot_date": t, "segment": seg, "x": 7e15})
                in_window = seg == "trial_only"
                labels.append(
                    {"msno": o, "snapshot_date": t, "in_exp30": in_window, "segment": seg}
                )
    lab = pd.DataFrame(labels).sort_values(["snapshot_date", "msno"])  # a different order
    lab = lab.assign(churned_7=1, churned_30=1, event_observed=1, days_to_event=3)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(feats), preserve_index=False), root / "features.parquet"
    )
    pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), root / "labels.parquet")
    return expected


def test_fixture_mean_is_order_sensitive():
    x = np.array(X)
    assert np.mean(x) != np.mean(x[::-1])


def test_rows_keep_the_rehearsal_order(tmp_path):
    expected = write_tables(tmp_path)
    got = p2_main_rows(tmp_path, T16, ["x"], P)
    assert got.msno.tolist() == expected
    # The rehearsal's path: every P2 row with outcomes, then the main-segment mask.
    ref = scored_month_rows(P, tmp_path, T16, ["x"], "p2_all")
    ref = ref[ref.segment.isin(P.main_segments)]
    assert ref.msno.tolist() == got.msno.tolist()
    mean = np.mean(got.x.to_numpy(dtype=float))
    assert mean == np.mean(ref.x.to_numpy(dtype=float))
    # The labels-file (sorted) order would give a different mean: an order bug is caught.
    by_msno = got.sort_values("msno").x.to_numpy(dtype=float)
    assert np.mean(by_msno) != mean


def test_rows_never_request_an_outcome_column(tmp_path, monkeypatch):
    write_tables(tmp_path)
    requested: list[list[str]] = []
    real = em.load_labels

    def spy(path, snapshots, columns, protocol, unlock=None):
        requested.append(list(columns))
        assert unlock is None
        return real(path, snapshots, columns, protocol, unlock)

    monkeypatch.setattr(em, "load_labels", spy)
    for t in (T16, T17):
        rows = p2_main_rows(tmp_path, t, ["x"], P)
        assert not [c for c in rows.columns if c.startswith(OUTCOME_LABEL_PREFIXES)]
        assert len(rows) == len(X)
    assert requested and all(column_kind(c) == "pre_t" for cols in requested for c in cols)


def test_outcome_columns_of_2017_stay_locked(tmp_path):
    write_tables(tmp_path)
    with pytest.raises(PermissionError):
        load_labels(tmp_path / "labels.parquet", [T17], ["churned_7"], P)


# --- nothing under src/retention/explain can reach the locked months ------------------------------

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


def forbidden_hits(root: Path) -> list[tuple[str, str]]:
    hits = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            hits += [(str(path), name) for name in FORBIDDEN if name.encode() in data]
    return hits


def test_scan_finds_a_planted_name(tmp_path):
    (tmp_path / "x.py").write_text("from retention.models.protocol import TestUnlock\n")
    assert forbidden_hits(tmp_path) == [(str(tmp_path / "x.py"), "TestUnlock")]


def test_explain_package_never_names_the_test_session():
    root = Path("src/retention/explain")
    assert {"__init__.py", "model.py", "__main__.py"} <= {p.name for p in root.glob("*.py")}
    assert forbidden_hits(root) == []
