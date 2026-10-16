"""Learners: AIPW hand case, recovery on randomized data (registered size and tolerance)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from retention.uplift import learners
from retention.uplift.config import load

# Registered in configs/uplift.yaml (kkbox.outcome_step.recovery_tests) before this file.
N, SEED, BASE, EFFECT, TRUE_AVERAGE, TOL, GAP = 4000, 0, 0.30, 0.10, 0.05, 0.04, 0.04


def synthetic():
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame({"x1": rng.normal(size=N), "x2": rng.normal(size=N)})
    t = np.zeros(N, dtype=int)
    t[rng.permutation(N)[: N // 2]] = 1
    p = BASE + EFFECT * (x["x1"].to_numpy() > 0) * t
    y = (rng.random(N) < p).astype(float)
    folds = np.full(N, -1)
    for k, (_, test) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(N), t)
    ):
        folds[test] = k
    return x, y, t, folds


def test_registered_recovery_values_match_the_config():
    r = load()["kkbox"]["outcome_step"]["recovery_tests"]
    assert "n = 4000, seed 0, half treated, base rate 0.30" in r["data"]
    assert "0.05 +/- 0.04" in r["pass_average"]
    assert "at least 0.04" in r["pass_heterogeneity"]


def test_aipw_hand_case():
    # psi_i = mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)
    #   unit 1: 0.5 + (1 - 0.8) / 0.5        =  0.9
    #   unit 2: 0.4 + (0 - 0.6) / 0.5        = -0.8
    #   unit 3: 0.3 - (1 - 0.4) / (1 - 0.25) = -0.5
    #   unit 4: 0.4 - (0 - 0.1) / (1 - 0.75) =  0.8
    #   mean = 0.4 / 4 = 0.1
    y = np.array([1.0, 0.0, 1.0, 0.0])
    t = np.array([1, 1, 0, 0])
    mu1 = np.array([0.8, 0.6, 0.7, 0.5])
    mu0 = np.array([0.3, 0.2, 0.4, 0.1])
    e = np.array([0.5, 0.5, 0.25, 0.75])
    assert learners.aipw(y, t, mu0, mu1, e) == pytest.approx(0.1)


def test_aipw_refuses_propensity_0_or_1():
    with pytest.raises(learners.NotComputable, match="exactly 0 or 1"):
        learners.aipw(np.ones(2), np.array([1, 0]), np.zeros(2), np.ones(2), np.array([1.0, 0.5]))


def test_recovery_at_the_registered_size_and_tolerance():
    x, y, t, folds = synthetic()
    e = np.full(N, t.mean())
    p = learners.KKBOX_FIXED
    mu0, mu1 = learners.t_learner(x, y, t, folds, "classifier", p)
    tau_t = mu1 - mu0
    tau_x = learners.x_learner(x, y, t, folds, "classifier", p, e)
    avg = {
        "aipw": learners.aipw(y, t, mu0, mu1, e),
        "t": float(tau_t.mean()),
        "x": float(tau_x.mean()),
    }
    for name, value in avg.items():
        assert abs(value - TRUE_AVERAGE) <= TOL, (name, value)
    pos = x["x1"].to_numpy() > 0
    for name, tau in (("t", tau_t), ("x", tau_x)):
        gap = tau[pos].mean() - tau[~pos].mean()
        assert gap >= GAP, (name, gap)


def test_t_learner_predictions_are_out_of_fold():
    x, y, t, folds = synthetic()
    p = learners.KKBOX_FIXED
    mu0, mu1 = learners.t_learner(x, y, t, folds, "classifier", p)
    y2 = y.copy()
    y2[folds == 0] = 1 - y2[folds == 0]
    a0, a1 = learners.t_learner(x, y2, t, folds, "classifier", p)
    assert np.array_equal(mu0[folds == 0], a0[folds == 0])
    assert np.array_equal(mu1[folds == 0], a1[folds == 0])


def test_learners_are_deterministic():
    x, y, t, folds = synthetic()
    e = np.full(N, 0.5)
    a = learners.estimate(x, y, t, folds, e, "classifier", learners.KKBOX_FIXED)
    b = learners.estimate(x, y, t, folds, e, "classifier", learners.KKBOX_FIXED)
    assert a == b


def test_kkbox_fixed_setting_matches_the_config():
    k = load()["learners"]
    assert learners.KKBOX_FIXED == {**k["kkbox_fixed"]}
    assert learners.COMMON == k["lightgbm_common"]


def test_one_valued_training_arm_predicts_that_value_and_is_counted():
    x = pd.DataFrame({"a": np.arange(40, dtype=float)})
    t = np.tile([0, 1], 20)
    y = np.isin(np.arange(40), [1, 3, 5, 7]).astype(float)  # treated responders in both folds
    folds = (np.arange(40) // 2) % 2  # both arms in every fold
    log = learners.FitLog()
    mu0, _ = learners.t_learner(x, y, t, folds, "classifier", learners.KKBOX_FIXED, log)
    assert np.all(mu0 == 0.0)
    assert log.constant_fits == 2 and log.fits == 4


def test_a_fold_with_too_few_units_in_an_arm_is_not_computable():
    x = pd.DataFrame({"a": np.arange(6, dtype=float)})
    t = np.array([1, 1, 1, 1, 0, 1])
    folds = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(learners.NotComputable, match="fewer than 2"):
        learners.t_learner(x, np.zeros(6), t, folds, "regressor", learners.KKBOX_FIXED)


def test_naive_difference_hand_case():
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    t = np.array([1, 1, 1, 0, 0, 0])
    assert learners.naive_difference(y, t) == pytest.approx(2 / 3 - 1 / 3)
