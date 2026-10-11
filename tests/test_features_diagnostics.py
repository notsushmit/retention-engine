"""The AUC used by the leak sweeps."""

import numpy as np
import pandas as pd
import pytest

from retention.features.diagnostics import auc, sweep


def brute(score, y):
    pos, neg = score[y], score[~y]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def test_auc_matches_pair_counting_with_ties():
    rng = np.random.default_rng(3)
    score = rng.integers(0, 5, size=300).astype(float)
    y = rng.random(300) < 0.3
    assert auc(score, y) == pytest.approx(brute(score, y))
    assert auc(np.array([1.0, 2, 3, 4]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auc(np.array([4.0, 3, 2, 1]), np.array([0, 0, 1, 1], bool)) == 0.0
    assert auc(np.ones(4), np.array([0, 1, 0, 1], bool)) == 0.5
    assert np.isnan(auc(np.ones(3), np.zeros(3, bool)))


def test_sweep_flags_a_perfect_feature_and_an_informative_null(monkeypatch):
    from retention.features import diagnostics, spec

    monkeypatch.setattr(spec, "FEATURE_COLUMNS", {"leaky": "usage", "nully": "usage"})
    monkeypatch.setattr(diagnostics.spec, "FEATURE_COLUMNS", spec.FEATURE_COLUMNS)
    y = np.array([0, 1] * 50, bool)
    df = pd.DataFrame(
        {
            "snapshot_date": pd.Timestamp("2016-08-01"),
            "churned_30": y.astype(int),
            "leaky": y.astype(float),
            "nully": np.where(y, np.nan, 1.0),
        }
    )
    t, counts = sweep(df)
    t = t.set_index("column")
    assert counts.to_dict(orient="records") == [{"snapshot": "2016-08-01", "n": 100, "events": 50}]
    assert t.loc["leaky", "flag_single"] and t.loc["leaky", "auc"] == 1.0
    assert t.loc["nully", "flag_null"] and t.loc["nully", "null_auc"] == 1.0
    assert t.loc["nully", "non_null_share"] == 0.5


def test_categories_are_scored_from_the_other_snapshots_only(monkeypatch):
    from retention.features import diagnostics, spec

    monkeypatch.setattr(spec, "FEATURE_COLUMNS", {"segment": "segment_state"})
    monkeypatch.setattr(diagnostics.spec, "FEATURE_COLUMNS", spec.FEATURE_COLUMNS)
    # In snapshot A category "x" always churns; in B it never does. Scored from the other
    # snapshot only, "x" ranks the wrong way in both, so the AUC is below 0.5.
    df = pd.DataFrame(
        {
            "snapshot_date": ["2016-07-01"] * 4 + ["2016-08-01"] * 4,
            "churned_30": [1, 1, 0, 0, 0, 0, 1, 1],
            "segment": ["x", "x", "y", "y", "x", "x", "y", "y"],
        }
    )
    t, _ = sweep(df)
    r = t.set_index("column").loc["segment"]
    assert r.scoring.startswith("category")
    assert r.auc_by_snapshot == {"2016-07-01": 0.0, "2016-08-01": 0.0}
    scores = diagnostics._category_scores(
        df.segment, df.churned_30.to_numpy(bool), df.snapshot_date.to_numpy()
    )
    assert scores.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
