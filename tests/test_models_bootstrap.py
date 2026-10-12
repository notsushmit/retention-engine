import numpy as np
import pytest

from retention.models.bootstrap import (
    Draws,
    FoldScores,
    combine_months,
    ece_interpretable,
    paired_auc_difference,
    verdict,
    verdicts,
)


def fold(n=400, seed=0, customers=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = y + rng.normal(0, 1, n)
    b = y + rng.normal(0, 1.5, n)
    c = np.arange(n) if customers is None else customers
    return FoldScores(np.asarray(c), y, a, b)


def test_identical_scorers_give_a_zero_interval():
    f = fold()
    out = paired_auc_difference([FoldScores(f.customer, f.y, f.score_a, f.score_a)], 200, 1)
    assert (out.point, out.lo, out.hi) == (0.0, 0.0, 0.0)


def test_perfect_against_reversed_gives_one():
    y = np.array([0, 1] * 100)
    f = FoldScores(np.arange(200), y, y.astype(float), -y.astype(float))
    out = paired_auc_difference([f], 200, 1)
    assert out.point == 1.0 and out.lo == 1.0 and out.hi == 1.0


def test_a_customer_in_two_folds_is_drawn_into_both_or_neither():
    draws = Draws([np.array(["x", "y", "z"]), np.array(["z", "w", "x"])], seed=7)
    zeros = 0
    for _ in range(500):
        w1, w2 = draws.next()
        assert w1[0] == w2[2] and w1[2] == w2[0]  # x and z carry the same count in both folds
        zeros += w1[0] == 0
    assert 0 < zeros < 500  # x is sometimes left out of both


def test_the_same_seed_gives_the_same_interval():
    f = [fold(seed=1), fold(seed=2, customers=np.arange(200, 600))]
    a = paired_auc_difference(f, 300, 42)
    b = paired_auc_difference(f, 300, 42)
    assert (a.lo, a.hi) == (b.lo, b.hi) and np.array_equal(a.replicates, b.replicates)
    assert paired_auc_difference(f, 300, 43).lo != a.lo  # another seed, another draw


def test_the_point_estimate_is_the_mean_of_the_fold_differences():
    f1, f2 = fold(seed=1), fold(seed=2, customers=np.arange(400, 800))
    both = paired_auc_difference([f1, f2], 50, 0).point
    one = [paired_auc_difference([f], 50, 0).point for f in (f1, f2)]
    assert both == pytest.approx(np.mean(one))


@pytest.mark.parametrize(
    "d, lo, hi, expected",
    [
        (0.012, 0.003, 0.021, {0.005: "win", 0.01: "win", 0.02: "below_threshold"}),
        (0.007, 0.001, 0.013, {0.005: "win", 0.01: "below_threshold", 0.02: "below_threshold"}),
        (0.030, 0.021, 0.040, {0.005: "win", 0.01: "win", 0.02: "win"}),
        (
            0.004,
            -0.002,
            0.010,
            {0.005: "no_difference", 0.01: "no_difference", 0.02: "no_difference"},
        ),
        (-0.004, -0.009, -0.001, {0.005: "loss", 0.01: "loss", 0.02: "loss"}),
    ],
)
def test_every_verdict_branch_at_each_threshold(d, lo, hi, expected):
    assert verdicts(d, lo, hi) == expected


def test_interval_touching_zero_is_no_difference():
    assert verdict(0.02, 0.0, 0.04) == "no_difference"
    assert verdict(-0.02, -0.04, 0.0) == "no_difference"


def test_two_test_months_win_only_together():
    assert combine_months(["win", "win"]) == "win"
    assert combine_months(["win", "no_difference"]) == "mixed"
    assert combine_months(["loss", "loss"]) == "loss"


def test_ece_interpretability_rule():
    assert ece_interpretable(0.010, 0.006, 0.014)
    assert not ece_interpretable(0.010, 0.004, 0.014)  # lower end below 0.5x
    assert not ece_interpretable(0.010, 0.006, 0.016)  # upper end above 1.5x
    assert not ece_interpretable(0.0, 0.0, 0.0)


def test_resampling_rows_understates_the_interval_when_customers_repeat():
    # 60 customers, each with 25 identical rows: the information is in 60 customers,
    # not 1,500 rows. Row resampling treats the copies as independent.
    base = fold(n=60, seed=3)
    rep = 25
    f = FoldScores(
        np.repeat(base.customer, rep),
        np.repeat(base.y, rep),
        np.repeat(base.score_a, rep),
        np.repeat(base.score_b, rep),
    )
    by_customer = paired_auc_difference([f], 1000, 9, cluster="customer")
    by_row = paired_auc_difference([f], 1000, 9, cluster="row")
    assert by_customer.point == by_row.point
    width_c, width_r = by_customer.hi - by_customer.lo, by_row.hi - by_row.lo
    assert width_c > 3 * width_r  # about sqrt(25) = 5 times wider in theory


def test_a_fold_with_no_lapses_gives_a_labelled_not_computable_interval():
    n = 50
    f = FoldScores(np.arange(n), np.zeros(n, int), np.arange(n) * 1.0, -np.arange(n) * 1.0)
    out = paired_auc_difference([f], 100, 1)
    assert out.status == "not computable" and np.isnan(out.point) and "lapse" in out.reason
    assert verdict(out.point, out.lo, out.hi) == "not_computable"


def test_replicates_without_lapses_make_the_interval_not_computable():
    # One lapse among 30 customers: many resamples contain no lapse.
    y = np.zeros(30, int)
    y[0] = 1
    f = FoldScores(np.arange(30), y, np.linspace(1, 0, 30), np.linspace(0, 1, 30))
    out = paired_auc_difference([f], 200, 3)
    assert out.point == 1.0  # computable on the original data: AUC 1 minus AUC 0
    assert out.status == "not computable" and "replicates not computable" in out.reason
    assert np.isnan(out.lo) and np.isnan(out.hi)


def test_one_and_four_workers_give_identical_replicates_and_intervals():
    from retention.models import bootstrap as b

    f = [fold(seed=1), fold(seed=2, customers=np.arange(200, 600))]
    b.use_workers(1)
    one = paired_auc_difference(f, 301, 42)
    b.use_workers(4)
    try:
        four = paired_auc_difference(f, 301, 42)
        rows = paired_auc_difference(f, 101, 5, cluster="row")
    finally:
        b.use_workers(1)
    assert np.array_equal(one.replicates, four.replicates)
    assert (one.point, one.lo, one.hi) == (four.point, four.lo, four.hi)
    assert np.array_equal(
        rows.replicates, paired_auc_difference(f, 101, 5, cluster="row").replicates
    )


def test_workers_report_not_computable_replicates_too():
    from retention.models import bootstrap as b

    y = np.zeros(30, int)
    y[0] = 1
    f = FoldScores(np.arange(30), y, np.linspace(1, 0, 30), np.linspace(0, 1, 30))
    b.use_workers(4)
    try:
        four = paired_auc_difference([f], 200, 3)
    finally:
        b.use_workers(1)
    assert four.reason == paired_auc_difference([f], 200, 3).reason
