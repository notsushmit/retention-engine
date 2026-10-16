"""Propensity design rules, folds, weights and balance on hand-made cases."""

import numpy as np
import pandas as pd
import pytest

from retention.uplift import balance, propensity


def test_column_rules():
    df = pd.DataFrame(
        {
            "const": [1.0] * 10,
            "rare_binary": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "binary": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "with_null": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "copy": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "cat": ["a"] * 6 + ["b"] * 3 + [None],
        }
    )
    numeric = ["const", "rare_binary", "binary", "with_null", "copy"]
    d = propensity.build_design(df, numeric, ["cat"])
    assert d.dropped["const"] == "constant"
    assert d.dropped["rare_binary"].startswith("binary value held by fewer than 5")
    assert d.dropped["copy"] == "identical to binary"
    assert d.dropped["cat=b"].startswith("level held by fewer than 5")
    assert d.dropped["cat=missing"].startswith("level held by fewer than 5")
    assert list(d.x.columns) == ["binary", "with_null", "with_null_missing", "cat=a"]
    assert d.x["with_null_missing"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert np.isnan(d.x["with_null"].iloc[2])  # filled later, inside the folds


def test_folds_keep_each_customer_together_and_are_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 2)  # every customer appears twice
    t = rng.integers(0, 2, len(groups))
    a = propensity.folds(t, groups)
    assert np.array_equal(a, propensity.folds(t, groups))
    assert set(np.unique(a)) == {0, 1, 2, 3, 4}
    for g in np.unique(groups):
        assert len(set(a[groups == g])) == 1


def test_pick_c_breaks_ties_toward_the_smaller_c():
    assert propensity.pick_c({0.01: 0.5, 0.1: 0.4, 1.0: 0.4, 10.0: 0.45}) == 0.1
    assert propensity.pick_c({0.01: 0.3, 0.1: 0.4}) == 0.01
    # losses equal to within floating-point noise are ties, in either direction
    assert propensity.pick_c({0.1: 0.4, 1.0: 0.4 - 1e-13}) == 0.1
    assert propensity.pick_c({0.1: 0.4 - 1e-13, 1.0: 0.4}) == 0.1
    assert propensity.pick_c({0.1: 0.4, 1.0: 0.4 - 1e-6}) == 1.0


def test_oof_propensity_is_out_of_fold_and_learns_the_signal():
    rng = np.random.default_rng(1)
    n = 600
    x = pd.DataFrame({"a": rng.normal(size=n), "noise": rng.normal(size=n)})
    t = (rng.random(n) < 1 / (1 + np.exp(-2 * x["a"].to_numpy()))).astype(int)
    fid = propensity.folds(t, np.arange(n))
    c, losses = propensity.choose_c(x, t, fid, [0.01, 0.1, 1.0, 10.0])
    assert c in losses and set(losses) == {0.01, 0.1, 1.0, 10.0}
    e = propensity.oof_propensity(x, t, fid, c)
    assert np.all((e > 0) & (e < 1))
    assert np.corrcoef(e, x["a"])[0, 1] > 0.8
    # a fold's predictions do not change when that fold's treatment labels change
    t2 = t.copy()
    t2[fid == 0] = 1 - t2[fid == 0]
    e2 = propensity.oof_propensity(x, t2, fid, c)
    assert np.array_equal(e[fid == 0], e2[fid == 0])
    assert np.array_equal(e, propensity.oof_propensity(x, t, fid, c))


def test_trim_mask_is_inclusive():
    e = np.array([0.04, 0.05, 0.5, 0.95, 0.96])
    assert propensity.trim_mask(e, 0.05, 0.95).tolist() == [False, True, True, True, False]


def test_stabilised_weights_hand_case():
    # p = 2 / 4 = 0.5; treated 0.5 / e, untreated 0.5 / (1 - e)
    t = np.array([1, 1, 0, 0])
    e = np.array([0.5, 0.25, 0.5, 0.75])
    assert propensity.stabilised_weights(t, e).tolist() == [1.0, 2.0, 1.0, 2.0]


def test_kish_ess_hand_case():
    # (1 + 1 + 2)^2 / (1 + 1 + 4) = 16 / 6
    assert propensity.kish_ess(np.array([1.0, 1.0, 2.0])) == pytest.approx(16 / 6)
    assert propensity.kish_ess(np.ones(10)) == 10


def test_smd_hand_case():
    # treated 1, 2, 3 (mean 2, variance 1); untreated 2, 3, 4 (mean 3, variance 1)
    x = np.array([1, 2, 3, 2, 3, 4], float)
    t = np.array([1, 1, 1, 0, 0, 0])
    assert balance.smd(x, t) == pytest.approx(-1.0)
    # weight 4 on the treated 3 moves the treated mean to (1 + 2 + 12) / 6 = 2.5
    w = np.array([1, 1, 4, 1, 1, 1], float)
    assert balance.smd(x, t, w) == pytest.approx(-0.5)  # variances stay unweighted


def test_smd_zero_variance_rule():
    t = np.array([1, 1, 0, 0])
    assert balance.smd(np.array([1.0, 1.0, 1.0, 1.0]), t) == 0.0
    assert balance.smd(np.array([1.0, 1.0, 0.0, 0.0]), t) == float("inf")


def test_balance_table_marks_the_threshold():
    x = pd.DataFrame({"a": [1, 2, 3, 2, 3, 4], "b": [1, 2, 3, 1, 2, 3]}, dtype=float)
    t = np.array([1, 1, 1, 0, 0, 0])
    table = balance.balance_table(x, t, np.ones(6))
    assert table.set_index("column")["passes"].to_dict() == {"a": False, "b": True}
    assert table.set_index("column").loc["b", "smd_after"] == 0.0


def test_balance_fill_uses_the_pooled_median_of_all_units():
    x = pd.DataFrame({"a": [1.0, None, 3.0, 10.0], "a_missing": [0.0, 1.0, 0.0, 0.0]})
    filled = balance.fill_for_balance(x)
    assert filled["a"].tolist() == [1.0, 3.0, 3.0, 10.0]  # median of 1, 3, 10
    assert filled["a_missing"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_tied_max_names_every_tied_column_in_any_order():
    a = 0.2732695862595
    for vals, labels in (
        ([a, -(a - 1e-13), 0.1], ["month_2016_09", "days_until_expiry", "other"]),
        ([0.1, -(a - 1e-13), a], ["other", "days_until_expiry", "month_2016_09"]),
        ([-(a + 1e-13), a, 0.1], ["days_until_expiry", "month_2016_09", "other"]),
    ):
        got = balance.tied_max(pd.Series(vals).abs(), pd.Series(labels))
        assert got == ["days_until_expiry", "month_2016_09"]
    assert balance.tied_max(pd.Series([0.3, 0.2]), pd.Series(["a", "b"])) == ["a"]
