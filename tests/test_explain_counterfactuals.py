"""Counterfactuals: the lever rule, the search, the support rule, the wording, the reads."""

from datetime import date
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from retention.explain import counterfactuals as cf
from retention.models.data import column_kind
from retention.models.protocol import load_protocol

LEVER = cf.LEVER


def frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "msno": [f"c{i}" for i in range(n)],
            "snapshot_date": pd.to_datetime(["2016-10-01"] * n),
            LEVER: rng.integers(0, 2, n),
            "x": rng.normal(size=n),
            "last_plan_days": 30,
            "last_list_price": 149,
            "last_payment_method_id": 41,
            "active_days_28": rng.integers(0, 29, n),
            "days_until_expiry": rng.integers(0, 30, n),
            "tenure_days": rng.integers(100, 900, n),
        }
    )


def hand_booster():
    """Known structure: churn is likely only when auto-renew is off and x > 0."""
    rows = frame(4000, seed=1)
    p = np.where((rows[LEVER] == 0) & (rows.x > 0), 0.8, 0.05)
    y = (np.random.default_rng(2).random(len(rows)) < p).astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=50, num_leaves=4, seed=0, deterministic=True, verbose=-1
    )
    model.fit(rows[[LEVER, "x"]], y)
    return lambda r: model.predict_proba(r[[LEVER, "x"]])[:, 1]


def test_hand_booster_known_best_change():
    score = hand_booster()
    rows = pd.DataFrame(
        {
            LEVER: [0, 0, 1, 0],
            "x": [1.0, -1.0, 1.0, 1.0],
            "active_days_28": 10,
            "days_until_expiry": 5,
        }
    )
    support = np.array([True, True, True, False])
    res = cf.search(rows, score, support, a=0.0, b=1.0, min_reduction=0.01)
    assert res.status.tolist() == ["counterfactual", "raises_score", "already_on", "no_support"]
    # row 0: auto-renew on moves the score from about 0.8 down to about 0.05
    assert res.old[0] > 0.6 and res.new[0] < 0.15
    # row 1: when x <= 0 the lever does not lower the score; here it raises it
    assert res.new[1] > res.old[1]
    assert np.isnan(res.new[2]) and np.isnan(res.new[3])


def test_done_when_no_feature_outside_the_lever_changes_and_never_switched_off():
    rows = frame()
    changed = cf.apply_lever(rows, np.ones(len(rows), bool))
    assert cf.changed_columns(rows, changed) == {LEVER}
    assert (changed[LEVER] == 1).all()  # off rows went up; on rows stayed on
    assert (changed.loc[rows[LEVER] == 1, LEVER] == 1).all()
    cf.guard_change(rows, changed)
    tampered = changed.copy()
    tampered["tenure_days"] += 1
    with pytest.raises(cf.LeverBreach, match="outside the lever"):
        cf.guard_change(rows, tampered)
    switched_off = rows.copy()
    switched_off.loc[rows[LEVER] == 1, LEVER] = 0
    with pytest.raises(cf.LeverBreach, match="switched off"):
        cf.guard_change(rows, switched_off)


def test_search_is_deterministic():
    score = hand_booster()
    rows = frame()
    support = np.ones(len(rows), bool)
    a = cf.search(rows, score, support, 0.0, 1.0, 0.01)
    b = cf.search(rows, score, support, 0.0, 1.0, 0.01)
    assert np.array_equal(a.old, b.old) and np.array_equal(a.new, b.new, equal_nan=True)
    assert np.array_equal(a.status, b.status)


def test_support_needs_ten_auto_renew_on_training_rows_in_the_group():
    base = frame(40)
    base["active_days_28"] = 10
    base["days_until_expiry"] = 5
    base[LEVER] = 0
    base.loc[:9, LEVER] = 1  # ten on-rows in this group
    other = base.copy()
    other["last_payment_method_id"] = 38
    other[LEVER] = 0
    other.loc[:8, LEVER] = 1  # nine on-rows in this group
    table = cf.support_table(pd.concat([base, other]))
    explained = pd.concat([base.head(1), other.head(1)])
    assert cf.supported(explained, table).tolist() == [True, False]


def test_calibration_is_platt():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cf.calibrated(p, 0.0, 1.0), p)
    z = np.log(p / (1 - p))
    assert np.allclose(cf.calibrated(p, -0.2, 1.1), 1 / (1 + np.exp(-(-0.2 + 1.1 * z))))


def test_coverage_counts_add_up():
    status = np.array(
        ["counterfactual", "no_support", "already_on", "below_threshold", "no_support"]
    )
    snaps = pd.Series(pd.to_datetime(["2016-10-01"] * 3 + ["2016-11-01"] * 2))
    cov = cf.coverage(snaps, status)
    assert sum(cov["2016-10-01"].values()) == 3 and sum(cov["2016-11-01"].values()) == 2
    assert cov["2016-11-01"]["no_support"] == 1


# --- wording --------------------------------------------------------------------------


def test_the_registered_phrase_file_breaks_no_rule():
    assert cf.phrase_problems(cf.load_phrases()) == []


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("Auto-renew {new} because {old}.", "causal"),
        ("You should look at {new}.", "advice"),
        ("We could offer {new}.", "lever verb"),
        ("This helps {new} and {old}.", "action claim"),
        ("Tenure {new} and {old}.", "never-named"),
        ("The model's 30-day score would be {new}.", "digit outside a fact field"),
    ],
)
def test_each_wording_rule_is_caught(line, rule):
    phrases = {**cf.load_phrases(), "counterfactual_line": line}
    assert any(rule in p for p in cf.phrase_problems(phrases))


def test_line_reads_correctly_when_the_new_score_is_higher_than_the_old():
    lines = cf.render("counterfactual", 0.051, 0.062, cf.load_phrases(), 0.01)
    assert lines[0] == (
        "With auto-renew recorded as on, the model's 30-day churn score would be 6.2% "
        "instead of 5.1%."
    )
    assert "not the effect of any action" in lines[1]


def test_no_counterfactual_lines():
    phrases = cf.load_phrases()
    assert cf.render("no_support", 0.05, np.nan, phrases, 0.01) == [
        "No counterfactual: auto-renew is not recorded as on for enough comparable customers."
    ]
    (line,) = cf.render("below_threshold", 0.05, 0.045, phrases, 0.01)
    assert line.endswith("by at least 1 percentage point.")
    assert cf.render("already_on", 0.05, np.nan, phrases, 0.01) == []


def test_min_reduction_matches_the_config():
    assert cf.registered_min_reduction() == 0.01


# --- reads ----------------------------------------------------------------------------


def test_the_run_reads_rows_only_through_the_explained_row_reader(monkeypatch):
    import retention.explain.contributions as contributions
    import retention.explain.model as explain_model

    score = hand_booster()
    calls = []

    def fake_rows(processed, t, columns, protocol):
        calls.append(t)
        return frame(200, seed=t.day + t.month).assign(snapshot_date=pd.Timestamp(t))

    stub = SimpleNamespace(
        model=SimpleNamespace(predict=lambda r: (score(r), {})),
        platt_a=0.0,
        platt_b=1.0,
        sha256="test",
    )
    monkeypatch.setattr(explain_model, "p2_main_rows", fake_rows)
    monkeypatch.setattr(contributions, "load_saved_model", lambda *a, **k: stub)
    snaps = [date(2016, 10, 1), date(2016, 11, 1)]
    log, table = cf.run(
        None, None, None, SimpleNamespace(columns=[]), snaps, 0.01, cf.load_phrases()
    )
    assert calls == [*cf.TRAINING_SNAPSHOTS, *snaps]  # training rows for support, then explained
    assert set(table.status) <= set(cf.STATUSES)
    assert sum(sum(v.values()) for v in log["coverage_by_snapshot"].values()) == len(table)
    assert set(table.columns) == {
        "msno",
        "snapshot_date",
        "score",
        "score_with_auto_renew_on",
        "status",
    }


def test_row_reader_requests_no_outcome_column(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import retention.explain.model as explain_model

    t = date(2016, 10, 1)
    labels = pa.table(
        {
            "msno": ["a", "b"],
            "snapshot_date": pa.array([t, t], pa.date32()),
            "in_exp30": [True, True],
            "segment": ["paid", "paid"],
            "paid_churned_30": pa.array([1, 0], pa.int8()),
            "churned_30": pa.array([1, 0], pa.int8()),
        }
    )
    pq.write_table(labels, tmp_path / "labels.parquet")
    feats = pa.table(
        {
            "msno": ["a", "b"],
            "snapshot_date": pa.array([t, t], pa.date32()),
            "segment": ["paid", "paid"],
            LEVER: pa.array([0, 1], pa.int8()),
        }
    )
    pq.write_table(feats, tmp_path / "features.parquet")
    seen = []
    real = explain_model.load_labels

    def spy(path, snapshots, columns, protocol, unlock=None):
        seen.append(list(columns))
        return real(path, snapshots, columns, protocol, unlock)

    monkeypatch.setattr(explain_model, "load_labels", spy)
    rows = explain_model.p2_main_rows(tmp_path, t, [LEVER, "segment"], load_protocol())
    assert len(rows) == 2 and seen
    assert all(column_kind(c) == "pre_t" for cols in seen for c in cols)


def test_raised_score_line_and_below_threshold_are_told_apart():
    rows = pd.DataFrame({LEVER: [0, 0, 0], "active_days_28": 10, "days_until_expiry": 5})
    scores = {0: np.array([0.05, 0.05, 0.05]), 1: np.array([0.03, 0.045, 0.07])}

    def score(r):
        return scores[int(r[LEVER].iloc[0])]

    res = cf.search(rows, score, np.ones(3, bool), 0.0, 1.0, 0.01)
    # 0.05 -> 0.03 meets the 1-point minimum; 0.05 -> 0.045 does not; 0.05 -> 0.07 rises
    assert res.status.tolist() == ["counterfactual", "below_threshold", "raises_score"]
    lines = cf.render("raises_score", 0.05, 0.07, cf.load_phrases(), 0.01)
    assert lines[0] == (
        "With auto-renew recorded as on, the model's 30-day churn score would be higher, "
        "7.0% instead of 5.0%."
    )
    assert "not the effect of any action" in lines[1]


@pytest.mark.parametrize("word", ["because", "should", "offer", "helps", "tenure", "30"])
def test_raised_score_line_is_held_to_the_same_rules(word):
    phrases = {**cf.load_phrases(), "raises_line": f"Score {{new}} {word} {{old}}."}
    assert any(p.startswith("raises_line") for p in cf.phrase_problems(phrases))
