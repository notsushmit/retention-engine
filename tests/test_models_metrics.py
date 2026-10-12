"""Every metric on a hand-made case with a known answer, plus library cross-checks."""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from retention.models import metrics as m


def logit(p):
    return math.log(p / (1 - p))


# --- AUC ----------------------------------------------------------------------------


def test_auc_hand_case():
    assert m.auc([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1]) == 0.75


def test_auc_tied_pair_counts_half():
    assert m.auc([1, 0], [0.5, 0.5]) == 0.5


def test_auc_weights_equal_repeated_rows():
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.3, 0.1])
    assert m.auc(y, s, np.array([2, 1, 1, 3])) == m.auc(
        np.repeat(y, [2, 1, 1, 3]), np.repeat(s, [2, 1, 1, 3])
    )


def test_auc_matches_scikit_learn_on_random_data_with_ties():
    rng = np.random.default_rng(1)
    y, s, w = rng.integers(0, 2, 3000), rng.integers(0, 50, 3000), rng.integers(1, 4, 3000)
    assert m.auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


# --- stop day and time-dependent AUC -----------------------------------------------------

# Stop day C = 59: a, b, c lapse on days 59, 58 and 3; d and e are followed exactly to 59.
TIME = np.array([59, 58, 3, 59, 59])
OBS = np.array([1, 1, 1, 0, 0])


def test_lapse_on_stop_day_is_observed_and_censored_rows_are_lapse_free_through_it():
    assert m.status_at(TIME, OBS, 59).tolist() == [1, 1, 1, 0, 0]


def test_lapse_on_day_before_stop_day():
    assert m.status_at(TIME, OBS, 58).tolist() == [0, 1, 1, 0, 0]


def test_status_beyond_the_stop_day_raises():
    with pytest.raises(ValueError, match="beyond the stop day"):
        m.status_at(TIME, OBS, 60)


def test_td_auc_at_day_t_equals_auc_of_the_status():
    score = np.array([0.6, 0.7, 0.9, 0.2, 0.65])
    assert m.td_auc(TIME, OBS, score, 59) == m.auc([1, 1, 1, 0, 0], score)


# --- concordance --------------------------------------------------------------------


def test_concordance_with_a_censored_customer():
    # Lapse on day 2 (risk 0.9), censored on day 1 (0.1), lapse on day 3 (0.5):
    # the only usable pair is (day 2, day 3), ordered right.
    assert m.harrell_c([2, 1, 3], [1, 0, 1], [0.9, 0.1, 0.5]) == 1.0


def test_concordance_when_the_stopped_customer_is_treated_as_a_lapse():
    assert m.harrell_c([2, 1, 3], [1, 1, 1], [0.9, 0.1, 0.5]) == pytest.approx(1 / 3)


def test_concordance_over_five_usable_pairs():
    assert m.harrell_c([1, 2, 3, 4], [1, 1, 0, 1], [4, 3, 2, 1]) == 1.0


def test_concordance_censored_on_the_lapse_day_counts_as_later_and_tied_lapses_do_not():
    # Lapse day 2 (risk 1) vs censored day 2 (risk 0): a pair, concordant.
    # Lapse day 2 (risk 1) vs lapse day 2 (risk 0): not a pair.
    assert m.harrell_c([2, 2], [1, 0], [1, 0]) == 1.0
    with pytest.raises(ValueError, match="no comparable pairs"):
        m.harrell_c([2, 2], [1, 1], [1, 0])


def test_concordance_matches_scikit_survival_on_random_data():
    sksurv = pytest.importorskip("sksurv.metrics")
    rng = np.random.default_rng(2)
    t, e, r = rng.integers(1, 40, 800), rng.integers(0, 2, 800).astype(bool), rng.normal(size=800)
    r[:50] = r[50:100]  # some tied risks
    ref = sksurv.concordance_index_censored(e, t.astype(float), r)[0]
    assert m.harrell_c(t, e, r) == pytest.approx(ref, abs=1e-12)


# --- Brier score with censoring weights ------------------------------------------------

# A lapses on day 2; B censored day 1; C censored day 3; D censored day 5; t = 4.
G_TIME, G_OBS = np.array([2, 1, 3, 5]), np.array([1, 0, 0, 0])


def test_censoring_curve_hand_case():
    g = m.censoring_curve(G_TIME, G_OBS)
    assert g(1) == 1.0  # censoring on day 1 takes effect after day 1
    assert g(2) == pytest.approx(3 / 4) and g(4) == pytest.approx(3 / 8)


def test_graf_brier_hand_case():
    # Weights: A 1/G(2) = 4/3, B 0, C 0, D 1/G(4) = 8/3; S(4) = 0.3 for A, 0.6 for D.
    surv = np.array([0.3, 0.5, 0.5, 0.6])
    expected = (0.3**2 * 4 / 3 + (1 - 0.6) ** 2 * 8 / 3) / 4
    assert expected == pytest.approx(41 / 300)
    assert m.brier_at(G_TIME, G_OBS, surv, 4) == pytest.approx(41 / 300)


def test_weights_are_exactly_one_under_a_common_stop_day():
    time, obs = np.array([2, 1, 3, 5, 5]), np.array([1, 1, 1, 0, 0])
    g = m.censoring_curve(time, obs)
    assert all(g(t) == 1.0 for t in range(0, 6))
    surv = np.array([0.3, 0.2, 0.6, 0.9, 0.7])
    plain = m.brier(m.status_at(time, obs, 4) == 0, surv)  # S is the chance of no lapse
    assert m.brier_at(time, obs, surv, 4) == pytest.approx(plain, abs=1e-15)


def test_brier_beyond_the_common_stop_day_raises():
    with pytest.raises(ValueError, match="beyond the stop day"):
        m.brier_at(np.array([2, 5, 5]), np.array([1, 0, 0]), np.array([0.5, 0.5, 0.5]), 6)


def test_plain_brier_hand_case():
    assert m.brier([0, 1], [0.2, 0.8]) == pytest.approx(0.04)


def test_ibs_of_a_constant_brier_is_that_constant():
    # Every row lapse-free and followed to day 10, S = 0.9 everywhere: Brier 0.01 on each day.
    time, obs = np.full(4, 10), np.zeros(4)
    surv = np.full((4, 3), 0.9)
    assert m.ibs(time, obs, surv, [3, 5, 7]) == pytest.approx(0.01)


def test_brier_matches_scikit_survival_on_random_data():
    sksurv = pytest.importorskip("sksurv.metrics")
    from sksurv.util import Surv

    rng = np.random.default_rng(3)
    t = rng.permutation(np.arange(1, 601)).astype(float)  # unique days: no tie conventions
    e = rng.integers(0, 2, 600).astype(bool)
    y = Surv.from_arrays(e, t)
    s = rng.uniform(0.1, 0.9, 600)
    ref = sksurv.brier_score(y, y, s[:, None], [200.0])[1][0]
    assert m.brier_at(t, e, s, 200.0) == pytest.approx(ref, rel=1e-10)


# --- log loss and average precision -----------------------------------------------------


def test_average_precision_hand_case():
    assert m.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1]) == pytest.approx(5 / 6)


def test_average_precision_matches_scikit_learn_with_ties():
    rng = np.random.default_rng(4)
    y, s = rng.integers(0, 2, 2000), rng.integers(0, 30, 2000)
    assert m.average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_log_loss_hand_case_and_clipping():
    assert m.log_loss([1, 0], [0.8, 0.2]) == pytest.approx(-math.log(0.8))
    assert math.isfinite(m.log_loss([1], [0.0]))


# --- Kaplan-Meier references --------------------------------------------------------------


def test_kaplan_meier_hand_case():
    # Lapse day 1, censored day 2, lapse day 3, censored day 4.
    s = m.kaplan_meier([1, 2, 3, 4], [1, 0, 1, 0], [0, 1, 2, 3, 4])
    assert s.tolist() == pytest.approx([1, 3 / 4, 3 / 4, 3 / 8, 3 / 8])


def test_kaplan_meier_counts_a_row_censored_on_a_lapse_day_as_at_risk():
    # Day 2: lapse and censored-at-2 both at risk, so S(2) = 1/2.
    assert m.kaplan_meier([2, 2], [1, 0], [2]).tolist() == [0.5]


def test_kaplan_meier_by_expiry_band():
    band = m.expiry_band([0, 7, 8, 30], [7, 14, 21, 29])
    assert band.tolist() == [0, 0, 1, 4]
    curves = m.kaplan_meier_by_band([1, 5, 2, 9], [1, 0, 0, 1], [0, 0, 1, 1], [3])
    assert curves[0].tolist() == [0.5] and curves[1].tolist() == [1.0]


def test_kaplan_meier_matches_lifelines():
    from lifelines import KaplanMeierFitter

    rng = np.random.default_rng(5)
    t, e = rng.integers(1, 30, 500), rng.integers(0, 2, 500)
    ref = KaplanMeierFitter().fit(t, e).survival_function_at_times([5, 10, 20]).to_numpy()
    assert m.kaplan_meier(t, e, [5, 10, 20]) == pytest.approx(ref, abs=1e-12)


# --- calibration ------------------------------------------------------------------------


def test_perfectly_calibrated_groups_give_slope_one_intercept_zero():
    y = [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    out = m.calibration_slope(y, [0.2] * 5 + [0.6] * 5)
    assert out["slope"] == pytest.approx(1, abs=1e-9)
    assert out["slope_model_intercept"] == pytest.approx(0, abs=1e-9)
    assert out["intercept"] == pytest.approx(0, abs=1e-9)


def test_slope_hand_case():
    y = [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    out = m.calibration_slope(y, [0.1] * 5 + [0.3] * 5)
    slope = (logit(0.6) - logit(0.2)) / (logit(0.3) - logit(0.1))
    assert slope == pytest.approx(1.3273, abs=1e-4)
    assert out["slope"] == pytest.approx(slope, abs=1e-9)
    assert out["slope_model_intercept"] == pytest.approx(logit(0.2) - slope * logit(0.1), abs=1e-9)
    assert out["slope_model_intercept"] == pytest.approx(1.5301, abs=1e-4)


def test_intercept_hand_case():
    out = m.calibration_slope([1, 0] * 5, [0.2] * 10)
    assert out["intercept"] == pytest.approx(logit(0.5) - logit(0.2), abs=1e-9)
    assert out["intercept"] == pytest.approx(1.3863, abs=1e-4)


def test_intercept_with_equal_probabilities_leaves_the_slope_undefined():
    assert math.isnan(m.calibration_slope([1, 0] * 5, [0.2] * 10)["slope"])


def test_calibration_clips_and_counts():
    # The clipped extremes are wrong (p = 0 lapses, p = 1 does not), so the fit is finite.
    out = m.calibration_slope([1, 0, 0, 1, 0, 1], [0.0, 1.0, 0.3, 0.6, 0.4, 0.5])
    assert out["clipped"] == 2 and math.isfinite(out["slope"])


def test_calibration_on_separable_outcomes_raises():
    with pytest.raises(ValueError, match="did not converge"):
        m.calibration_slope([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])


def test_ece_hand_case():
    assert m.ece([0, 0, 1, 1], [0.1, 0.1, 0.5, 0.5], bins=2) == pytest.approx(0.3)


# --- cells the data cannot support ------------------------------------------------------


def test_auc_cell_with_no_lapses_is_labelled_not_computable():
    out = m.cell(m.auc, [0, 0, 0], [0.1, 0.2, 0.3])
    assert out.status == m.NOT_COMPUTABLE and math.isnan(out.value)
    assert "lapse" in out.reason


def test_td_auc_and_concordance_cells_with_no_lapses_are_not_computable():
    time, obs = np.array([59, 59]), np.array([0, 0])
    assert m.cell(m.td_auc, time, obs, [0.2, 0.4], 7).status == m.NOT_COMPUTABLE
    assert m.cell(m.harrell_c, time, obs, [0.2, 0.4]).status == m.NOT_COMPUTABLE


def test_calibration_cell_with_no_lapses_is_not_computable():
    out = m.cell(m.calibration_slope, [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert out.status == m.NOT_COMPUTABLE and "lapse" in out.reason


def test_a_computable_cell_is_ok():
    assert m.cell(m.auc, [1, 0], [0.9, 0.1]) == m.Cell(1.0)


def test_a_day_beyond_the_stop_day_is_still_an_error_not_a_cell():
    with pytest.raises(ValueError, match="beyond the stop day"):
        m.cell(m.td_auc, np.array([59, 59]), np.array([1, 0]), [0.2, 0.4], 60)


def test_a_singular_hessian_is_not_computable_on_every_platform(monkeypatch):
    # Linux OpenBLAS hits a singular matrix on separated outcomes where macOS
    # runs to the iteration limit; force the singular path here.
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(m.np.linalg, "solve", singular)
    with pytest.raises(m.NotComputable, match="singular Hessian"):
        m.calibration_slope([0, 1, 0, 1], [0.2, 0.6, 0.3, 0.7])
    assert (
        m.cell(m.calibration_slope, [0, 1, 0, 1], [0.2, 0.6, 0.3, 0.7]).status == m.NOT_COMPUTABLE
    )
