"""Public-data validation: splits, coding, grid rule, pass marks, outputs and the run guard."""

import numpy as np
import pandas as pd
import pytest

from retention.uplift import public_validation as pv
from retention.uplift.config import load


def hillstrom_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "1) $0 - $100",
            "history": rng.gamma(2, 100, n),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
            "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
            "visit": rng.integers(0, 2, n),
            "conversion": rng.integers(0, 2, n),
            "spend": 0.0,
        }
    )


def test_split_is_stratified_and_deterministic():
    df = hillstrom_frame()
    a = pv.evaluation_half(df.segment.to_numpy())
    assert np.array_equal(a, pv.evaluation_half(df.segment.to_numpy()))
    for arm in df.segment.unique():
        n = (df.segment == arm).sum()
        assert abs(int(a[(df.segment == arm).to_numpy()].sum()) - n / 2) <= 1


def test_both_hillstrom_cells_share_one_control_arm():
    df = hillstrom_frame()
    ev = pv.evaluation_half(df.segment.to_numpy())
    mens, womens = pv.hillstrom_cells(df, ev, "visit")
    control = lambda c: set(c.rows[c.t == 0])  # noqa: E731
    assert control(mens) == control(womens)
    assert control(mens) == set(np.flatnonzero(df.segment.eq("No E-Mail")))
    # the split is made once, before the cells: a control row is on the same side in both
    for c in (mens, womens):
        assert np.array_equal(c.evaluation, ev[c.rows])


def test_hillstrom_coding_uses_training_levels_and_no_arm_column():
    df = hillstrom_frame()
    ev = pv.evaluation_half(df.segment.to_numpy())
    x = pv.hillstrom_features(df, ~ev)
    pv.check_features(x)
    assert "segment" not in x.columns and "history_segment" not in x.columns
    assert {"zip_code=Urban", "channel=Web"} <= set(x.columns)


def test_exposure_is_never_a_feature():
    df = pd.DataFrame({**{f"f{i}": [0.1] * 4 for i in range(12)}, "exposure": [1, 0, 1, 0]})
    assert "exposure" not in pv.criteo_features(df).columns
    with pytest.raises(ValueError, match="exposure"):
        pv.check_features(df)


def test_criteo_grid_subsample_is_stratified_fixed_and_from_the_training_half():
    rng = np.random.default_rng(0)
    t = (rng.random(20000) < 0.85).astype(int)
    ev = pv.evaluation_half(t)
    train_idx = np.flatnonzero(~ev)
    sub = pv.grid_subsample(train_idx, t, size=5000)
    assert len(sub) == 5000 and set(sub) <= set(train_idx)
    assert np.array_equal(sub, pv.grid_subsample(train_idx, t, size=5000))
    assert abs(t[sub].mean() - t[train_idx].mean()) < 0.002
    assert pv.GRID_SUBSAMPLE == 1_000_000


def test_grid_matches_the_registered_grid():
    g = load()["learners"]["public_grid"]
    assert len(pv.GRID) == 8
    assert {p["n_estimators"] for p in pv.GRID} == set(g["n_estimators"])
    assert {p["num_leaves"] for p in pv.GRID} == set(g["num_leaves"])
    assert {p["min_child_samples"] for p in pv.GRID} == set(g["min_child_samples"])
    assert all(p["learning_rate"] == g["learning_rate"] for p in pv.GRID)


def test_grid_tie_rule():
    p = {
        (n, lv, m): {"n_estimators": n, "num_leaves": lv, "min_child_samples": m}
        for n, lv, m in [(100, 4, 20), (100, 4, 100), (100, 15, 100), (300, 4, 100)]
    }
    # all tied: fewer trees, then fewer leaves, then larger min_child_samples
    assert pv.pick_point([(v, 0.5) for v in p.values()]) == p[(100, 4, 100)]
    # a lower loss wins outright; a 1e-12 difference is a tie
    assert pv.pick_point([(p[(300, 4, 100)], 0.4), (p[(100, 4, 20)], 0.5)]) == p[(300, 4, 100)]
    assert (
        pv.pick_point([(p[(300, 4, 100)], 0.5 - 1e-12), (p[(100, 15, 100)], 0.5)])
        == p[(100, 15, 100)]
    )
