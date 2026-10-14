"""Contributions: exact Shapley values, additivity, base value, groups, saved models, the run."""

from __future__ import annotations

import itertools
import math
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from retention.explain import model as explain_model
from retention.explain.__main__ import audit_label_reads
from retention.explain.config import load_explain_config, parse_explain_config
from retention.explain.contributions import (
    EXPECTED_ROWS,
    CheckFailed,
    check_chunk,
    chunk_contributions,
    group_sums,
    load_saved_model,
    run_contributions,
    same_outputs,
)
from retention.explain.describe import facts_check, illustrative_rows, table_b_month
from retention.explain.model import (
    REPRODUCTION_CELLS,
    SUMMARY_KEY,
    Refit,
    Settings,
    booster_sha256,
    expected_value_from_trees,
    fit_models,
    load_settings,
    save_models,
)
from retention.models.protocol import load_protocol
from test_explain_model import T16, T17, write_tables

P = load_protocol()
FIXED = yaml.safe_load(Path("configs/evaluation.yaml").read_text())["classifier"]["fixed_params"]

# --- a hand-made one-tree model -------------------------------------------------------------
#
#            root: a <= 0.5 ?   (100 rows)
#            /               \
#   b <= 0.5 ? (60 rows)      leaf R = 10 (40 rows)
#    /          \
# leaf L1 = 1    leaf L2 = 3
# (20 rows)      (40 rows)
#
# Path-dependent expectation v(S) (features in S known, others follow the row counts):
#   v({})   = (20*1 + 40*3 + 40*10) / 100 = 540/100 = 5.4   (the base value)
#   For x = (a=0, b=1), f(x) = 3:
#     v({a}) = (20*1 + 40*3) / 60 = 140/60 = 7/3
#     v({b}) = 0.6 * 3 + 0.4 * 10 = 5.8   (b known -> L2 on the left; right is R)
#     phi_a = 1/2 [(7/3 - 5.4) + (3 - 5.8)] = 1/2 [-46/15 - 42/15] = -44/15
#     phi_b = 1/2 [(5.8 - 5.4) + (3 - 7/3)] = 1/2 [6/15 + 10/15]   =   8/15
#   For x = (a=0, b=0), f(x) = 1:
#     v({a}) = 7/3;  v({b}) = 0.6 * 1 + 0.4 * 10 = 4.6
#     phi_a = 1/2 [(7/3 - 5.4) + (1 - 4.6)] = 1/2 [-46/15 - 54/15] = -10/3
#     phi_b = 1/2 [(4.6 - 5.4) + (1 - 7/3)] = 1/2 [-12/15 - 20/15] = -16/15
#   For x = (a=1, b=0), f(x) = 10:
#     v({a}) = 10;  v({b}) = 4.6
#     phi_a = 1/2 [(10 - 5.4) + (10 - 4.6)] = 1/2 [4.6 + 5.4]  = 5
#     phi_b = 1/2 [(4.6 - 5.4) + (10 - 10)]  = 1/2 [-0.8]       = -2/5
# Each row: phi_a + phi_b + 5.4 = f(x).
HANDMADE = """tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=1
objective=regression
feature_names=a b
feature_infos=[0:1] [0:1]

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1 1
threshold=0.5 0.5
decision_type=2 2
left_child=1 -1
right_child=-3 -2
leaf_value=1 3 10
leaf_weight=20 40 40
leaf_count=20 40 40
internal_value=5.4 2.3333333333333335
internal_weight=100 60
internal_count=100 60
is_linear=0
shrinkage=1


end of trees

pandas_categorical:null
"""


def test_handmade_tree_gives_the_hand_computed_shapley_values():
    booster = lgb.Booster(model_str=HANDMADE)
    x = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    assert booster.predict(x).tolist() == [3.0, 1.0, 10.0]
    got = booster.predict(x, pred_contrib=True)
    want = np.array(
        [
            [-44 / 15, 8 / 15, 5.4],
            [-10 / 3, -16 / 15, 5.4],
            [5.0, -2 / 5, 5.4],
        ]
    )
    assert np.max(np.abs(got - want)) <= 1e-12
    assert expected_value_from_trees(booster) == pytest.approx(5.4, abs=1e-12)


# --- brute-force reference on a small trained model --------------------------------------------


def _count(node: dict) -> float:
    return node.get("internal_count", node.get("leaf_count"))


def _expectation(node: dict, x: np.ndarray, known: frozenset[int]) -> float:
    """E[f(x) | features in `known`], other splits weighted by the tree's row counts."""
    if "leaf_value" in node:
        return node["leaf_value"]
    assert node["decision_type"] == "<="
    left, right = node["left_child"], node["right_child"]
    f = node["split_feature"]
    if f in known:
        return _expectation(left if x[f] <= node["threshold"] else right, x, known)
    nl, nr = _count(left), _count(right)
    return (nl * _expectation(left, x, known) + nr * _expectation(right, x, known)) / (nl + nr)


def brute_force_shapley(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    trees = [t["tree_structure"] for t in booster.dump_model()["tree_info"]]
    n = len(x)

    def v(known: frozenset[int]) -> float:
        return sum(_expectation(t, x, known) for t in trees)

    phi = np.zeros(n + 1)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for s in itertools.combinations(others, size):
                known = frozenset(s)
                phi[i] += w * (v(known | {i}) - v(known))
    phi[n] = v(frozenset())
    return phi


def test_brute_force_shapley_matches_pred_contrib():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3000, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] * x[:, 2] - 0.3 * x[:, 3] + rng.normal(size=3000) > 0).astype(int)
    params = {
        "objective": "binary",
        "num_leaves": 6,
        "learning_rate": 0.3,
        "min_child_samples": 20,
        "seed": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": 1,
    }
    booster = lgb.train(params, lgb.Dataset(x, y), num_boost_round=8)
    rows = x[:20]
    got = booster.predict(rows, pred_contrib=True)
    want = np.vstack([brute_force_shapley(booster, r) for r in rows])
    assert np.max(np.abs(got - want)) <= 1e-9


# --- chunked, additivity, base value ---------------------------------------------------------


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
    fixed=FIXED,
    candidates={
        7: {"num_leaves": 7, "n_estimators": 30, "learning_rate": 0.1, "min_child_samples": 20}
    },
)


@pytest.fixture(scope="module")
def fitted():
    fit, cal = synthetic(4000, 0), synthetic(1500, 1)
    return fit, cal, fit_models(fit, cal, SETTINGS, 7, P)


def test_chunked_equals_unchunked(fitted):
    fit, _, (m, _) = fitted
    rows = synthetic(3001, 9)
    raw, contrib = chunk_contributions(m, rows)
    parts = [chunk_contributions(m, rows.iloc[i : i + 1000]) for i in range(0, len(rows), 1000)]
    assert np.array_equal(raw, np.concatenate([p[0] for p in parts]))
    assert np.array_equal(contrib, np.vstack([p[1] for p in parts]))


def test_additivity_and_base_value_on_synthetic_rows(fitted):
    fit, _, (m, _) = fitted
    raw, contrib = chunk_contributions(m, synthetic(2000, 3))
    base_ref = expected_value_from_trees(m.booster)
    out = check_chunk(raw, contrib, base_ref, 1e-6, 1e-9)
    assert out["failures"] == []
    assert out["max_additivity_gap"] <= 1e-12
    assert out["base_value_constant"]


def test_check_chunk_catches_each_failure(fitted):
    _, _, (m, _) = fitted
    raw, contrib = chunk_contributions(m, synthetic(500, 4))
    base = float(contrib[0, -1])
    bad_sum = contrib.copy()
    bad_sum[7, 0] += 1e-5
    assert any("additivity" in f for f in check_chunk(raw, bad_sum, base, 1e-6, 1e-9)["failures"])
    bad_base = contrib.copy()
    bad_base[3, -1] = np.nextafter(bad_base[3, -1], np.inf)
    bad_base[3, 0] = np.nextafter(bad_base[3, 0], -np.inf)
    out = check_chunk(raw, bad_base, base, 1e-6, 1e-9)
    assert any("not constant" in f for f in out["failures"])
    out = check_chunk(raw, contrib, base + 1e-8, 1e-6, 1e-9)
    assert any("differs from the refit" in f for f in out["failures"])


# --- groups, the expiry merge, the real config ---------------------------------------------------


def test_group_sums_and_the_expiry_merge():
    columns = [
        "days_until_expiry",
        "days_until_paid_expiry",
        "days_since_last_txn",
        "days_since_last_payment",
        "x",
        "y",
    ]
    groups = {"renewal_timing": columns[:4], "other": ["x", "y"]}
    rng = np.random.default_rng(0)
    contrib = np.column_stack([rng.normal(size=(50, 6)), np.full(50, -2.0)])
    g = group_sums(contrib, columns, groups)
    assert np.array_equal(g[:, 0], contrib[:, 0:4].sum(axis=1))
    assert np.array_equal(g[:, 1], contrib[:, 4:6].sum(axis=1))
    raw = contrib.sum(axis=1)
    assert np.max(np.abs(g.sum(axis=1) + contrib[:, -1] - raw)) <= 1e-12


def test_real_config_groups_cover_the_model_columns_once():
    columns = load_settings().columns
    cfg = load_explain_config(columns)
    flat = [c for cols in cfg.groups.values() for c in cols]
    assert sorted(flat) == sorted(columns) and len(flat) == len(set(flat)) == 40
    assert cfg.groups["renewal_timing"] == [
        "days_until_expiry",
        "days_until_paid_expiry",
        "days_since_last_txn",
        "days_since_last_payment",
    ]
    assert cfg.hidden == ("hidden",) and cfg.groups["hidden"] == ["registered_via"]
    assert cfg.shown_only_when_known == {"tenure": "tenure_days"}
    assert "hidden" not in cfg.facts


def test_config_reproduction_cells_match_the_code():
    cfg = load_explain_config(load_settings().columns)
    code = [(h, t, (*path, SUMMARY_KEY)) for h, t, path in REPRODUCTION_CELLS]
    assert list(cfg.reproduction_cells) == code


def _raw_config(groups, facts, hidden=(), known=None, snapshots=(T16, T17)):
    return {
        "model": {"horizons": [7], "reproduction": {"cells": []}},
        "rows": {"snapshots": [str(t) for t in snapshots]},
        "contributions": {
            "max_additivity_error": 1e-6,
            "max_base_value_error": 1e-9,
            "memory_ceiling_gb": 10.0,
        },
        "groups": groups,
        "hidden_groups": list(hidden),
        "shown_only_when_known": known or {},
        "facts": facts,
        "strength": {"strong": 2.0, "moderate": 1.5, "slight": 1.2},
        "selection": {"top_raising": 3, "top_lowering": 2},
        "position_cutoff": 1.2,
        "other_factors": {"small_below": 1.2},
    }


def test_config_refuses_overlaps_gaps_and_outside_facts():
    with pytest.raises(ValueError, match="groups"):
        parse_explain_config(_raw_config({"g": ["x", "y"], "h": ["y"]}, {}), ["x", "y"])
    with pytest.raises(ValueError, match="cover"):
        parse_explain_config(_raw_config({"g": ["x"]}, {}), ["x", "y"])
    with pytest.raises(ValueError, match="outside"):
        parse_explain_config(_raw_config({"g": ["x"], "h": ["y"]}, {"g": ["y"]}), ["x", "y"])
    with pytest.raises(ValueError, match="not a shown group"):
        parse_explain_config(
            _raw_config({"g": ["x"], "h": ["y"]}, {"h": ["y"]}, hidden=["h"]), ["x", "y"]
        )


# --- saved models -----------------------------------------------------------------------------


def test_saved_model_loads_and_a_tampered_file_is_refused(fitted, tmp_path):
    fit, cal, (m, p) = fitted
    sha = booster_sha256(m)
    save_models(tmp_path, {7: Refit(7, m, p, len(fit), len(cal))}, {7: sha})
    loaded = load_saved_model(tmp_path, 7, SETTINGS, P, sha)
    rows = synthetic(300, 11)
    assert np.array_equal(loaded.model.predict(rows)[0], m.predict(rows)[0])
    assert (loaded.platt_a, loaded.platt_b) == (p.a, p.b)
    with pytest.raises(CheckFailed):
        load_saved_model(tmp_path, 7, SETTINGS, P, "0" * 64)
    path = tmp_path / "booster_7d.txt"
    path.write_text(path.read_text() + "\n")  # one extra byte
    with pytest.raises(CheckFailed):
        load_saved_model(tmp_path, 7, SETTINGS, P, sha)


# --- the whole run on a fixture ------------------------------------------------------------------


@pytest.fixture()
def run_setup(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    write_tables(processed)
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000) * 1e3
    fit = pd.DataFrame({"x": x, "y": (x + rng.normal(size=3000) * 500 > 300).astype(int)})
    settings = Settings(["x"], [], FIXED, {7: SETTINGS.candidates[7]})
    model, platt = fit_models(fit, fit.iloc[:1000], settings, 7, P)
    sha = booster_sha256(model)
    model_dir = tmp_path / "model"
    save_models(model_dir, {7: Refit(7, model, platt, 3000, 1000)}, {7: sha})
    cfg = parse_explain_config(_raw_config({"g": ["x"]}, {"g": ["x"]}), ["x"])
    refit_report = {
        "horizons": {
            "7": {
                "booster_sha256": sha,
                "base_value_mean_raw_score": explain_model.mean_raw_score(model, fit),
            }
        }
    }
    rows = {T16: 14, T17: 14}
    return processed, model_dir, refit_report, cfg, settings, {7: sha}, rows


def test_run_writes_parts_only_after_every_chunk_passes(run_setup, tmp_path):
    processed, model_dir, rep, cfg, settings, sha, rows = run_setup
    out = tmp_path / "out"
    report, results = run_contributions(
        P, processed, model_dir, rep, out, cfg, settings, lambda m: None, sha, rows
    )
    assert report["passed"] and len(report["chunks"]) == 2
    assert sorted(p.name for p in out.iterdir()) == [
        "h7_2016-11-01.parquet",
        "h7_2017-01-01.parquet",
    ]
    assert all(c["max_listed_plus_other_minus_d"] <= 1e-12 for c in report["chunks"])
    # The descriptive tables run on the results.
    assert table_b_month(cfg, results[0])["rows"] == 14
    assert set(facts_check(cfg, results)["7"]["g"]) == {"raises", "lowers"}
    first = illustrative_rows(cfg, results)
    assert first == illustrative_rows(cfg, results)  # fixed seed
    # A second run gives identical outputs.
    out2 = tmp_path / "out2"
    run_contributions(P, processed, model_dir, rep, out2, cfg, settings, lambda m: None, sha, rows)
    assert same_outputs(out, out2)["identical"]


def test_a_failed_check_leaves_no_output(run_setup, tmp_path):
    processed, model_dir, rep, cfg, settings, sha, rows = run_setup
    out = tmp_path / "out"
    with pytest.raises(CheckFailed, match="rows"):
        run_contributions(
            P,
            processed,
            model_dir,
            rep,
            out,
            cfg,
            settings,
            lambda m: None,
            sha,
            {T16: 14, T17: 13},
        )
    assert list(out.glob("*.parquet*")) == []


def test_run_reads_only_columns_known_at_t(run_setup, tmp_path, monkeypatch):
    processed, model_dir, rep, cfg, settings, sha, rows = run_setup
    monkeypatch.setattr(explain_model, "load_labels", explain_model.load_labels)  # restored after
    reads: list[dict] = []
    audit_label_reads(reads)
    run_contributions(
        P, processed, model_dir, rep, tmp_path / "o", cfg, settings, lambda m: None, sha, rows
    )
    assert {tuple(r["columns"]) for r in reads} == {("in_exp30", "segment")}
    assert {s for r in reads for s in r["snapshots"]} == {str(T16), str(T17)}
    with pytest.raises(PermissionError):
        explain_model.load_labels(processed / "labels.parquet", [T17], ["churned_7"], P)


def test_expected_rows_are_the_registered_counts():
    assert EXPECTED_ROWS == {
        date(2016, 10, 1): 144_996,
        date(2016, 11, 1): 169_673,
        date(2017, 1, 1): 150_672,
        date(2017, 2, 1): 180_412,
    }
