import numpy as np
import pandas as pd
import pytest
import yaml

from retention.models.aft import AFTModel, aft_bounds
from retention.models.cox import CoxModel, CoxPreprocessor
from retention.models.discrete_time import DiscreteTimeModel, expand
from retention.models.protocol import load_protocol

P = load_protocol()
RAW = yaml.safe_load(open("configs/evaluation.yaml"))
ENDS = RAW["survival"]["discrete_time"]["interval_ends"]


def survival_rows(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 30, n)
    auto = rng.integers(0, 2, n)
    lapse_now = rng.random(n) < 1 / (1 + np.exp(-(-2 + -1.5 * auto)))
    time = np.where(lapse_now, days, 59)
    return pd.DataFrame(
        {
            "days_until_expiry": days,
            "days_until_paid_expiry": days,
            "last_is_auto_renew": auto,
            "secs_per_day_28": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(2, 900, n)),
            "active_days_28": np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 29, n)),
            "last_payment_method_id": pd.array(rng.choice([41, 36, 38], n), dtype="Int16"),
            "time": time,
            "event": lapse_now.astype(int),
        }
    )


COLS = [
    "days_until_expiry",
    "days_until_paid_expiry",
    "last_is_auto_renew",
    "secs_per_day_28",
    "active_days_28",
    "last_payment_method_id",
]


# --- discrete time ---------------------------------------------------------------------


def test_expand_hand_cases():
    rows = pd.DataFrame(
        {"time": [10, 30, 31, 0], "event": [1, 0, 0, 1], "days_until_expiry": [9, 5, 5, 0]}
    )
    long = expand(rows, ENDS, for_fit=True)
    by = {r: long[long.source_row == r][["interval_end", "y"]].values.tolist() for r in range(4)}
    assert by[0] == [[7, 0], [14, 1]]  # lapse on day 10: survives 0-7, lapses in 8-14
    assert by[1] == [
        [7, 0],
        [14, 0],
        [21, 0],
        [30, 0],
    ]  # censored on 30: interval 31-37 not reached
    assert by[2] == by[1]  # censored on 31: 31-37 only partly observed, left out
    assert by[3] == [[7, 1]]  # lapse on day 0
    assert set(long.interval_end_minus_days_until_expiry) >= {7 - 9, 14 - 9}


def test_discrete_time_probabilities_rise_with_days_and_need_interval_ends():
    rows = survival_rows()
    cand = dict(
        zip(
            ("num_leaves", "n_estimators", "learning_rate", "min_child_samples"),
            RAW["classifier"]["candidates"]["c01"],
            strict=True,
        )
    )
    m = DiscreteTimeModel(
        True,
        ENDS,
        cand,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(rows, COLS)
    p, _ = m.lapse_probability(rows.head(200), [7, 30, 59])
    assert (np.diff(p, axis=1) >= -1e-12).all() and ((p >= 0) & (p <= 1)).all()
    with pytest.raises(ValueError, match="interval ends"):
        m.lapse_probability(rows.head(5), [8])


# --- AFT ----------------------------------------------------------------------------


def test_aft_bounds_count_time_from_day_one():
    lower, upper = aft_bounds(np.array([0, 5, 31]), np.array([1, 1, 0]))
    assert lower.tolist() == [1, 6, 33] and upper[:2].tolist() == [1, 6] and np.isinf(upper[2])


def test_aft_probabilities_rise_with_days_and_reruns_are_identical():
    rows = survival_rows()
    fixed = RAW["survival"]["xgboost_aft"]["fixed_params"] | {"num_boost_round": 30}

    def fit():
        return AFTModel(
            "normal", 1.0, fixed, RAW["classifier"]["categorical"], P.threads, P.seed
        ).fit(rows, COLS)

    p1, _ = fit().lapse_probability(rows.head(300), [7, 30, 59])
    p2, _ = fit().lapse_probability(rows.head(300), [7, 30, 59])
    assert np.array_equal(p1, p2)
    assert (np.diff(p1, axis=1) >= 0).all()


# --- Cox ----------------------------------------------------------------------------


def test_cox_preprocessing_constants_come_from_fit_rows_only():
    fit, other = survival_rows(seed=0), survival_rows(seed=1)
    other["secs_per_day_28"] *= 100  # a very different scoring set
    prep = CoxPreprocessor(RAW["classifier"]["categorical"]).fit(
        fit, ["secs_per_day_28", "active_days_28"]
    )
    before = {c: dict(v) for c, v in prep.constants.items()}
    prep.transform(other)
    assert prep.constants == before
    assert prep.constants["secs_per_day_28"]["median"] == pytest.approx(
        fit.secs_per_day_28.median()
    )


def test_one_flag_per_null_pattern_and_copies_dropped():
    rows = survival_rows()
    rows["copy_of_auto"] = rows.last_is_auto_renew
    rows["complement_of_auto"] = 1 - rows.last_is_auto_renew
    rows["same_nulls"] = rows.secs_per_day_28 * 2  # same null pattern as secs_per_day_28
    cols = [
        "last_is_auto_renew",
        "copy_of_auto",
        "complement_of_auto",
        "secs_per_day_28",
        "same_nulls",
        "active_days_28",
    ]
    prep = CoxPreprocessor([]).fit(rows, cols)
    assert sorted(prep.flags) == ["missing:active_days_28", "missing:secs_per_day_28"]
    assert prep.flags["missing:secs_per_day_28"] == ["secs_per_day_28", "same_nulls"]
    assert {"copy_of_auto", "complement_of_auto"} <= set(prep.dropped)


def test_one_hot_with_frozen_levels_counts_unseen_as_other():
    fit = survival_rows()
    prep = CoxPreprocessor(["last_payment_method_id"]).fit(fit, ["last_payment_method_id"])
    assert prep.reference["last_payment_method_id"] in {"41", "36", "38"}
    score = fit.head(10).copy()
    score["last_payment_method_id"] = pd.array([99] * 10, dtype="Int16")
    x = prep.transform(score)
    assert prep.unseen["last_payment_method_id"] == 10 and (x.to_numpy() == 0).all()


def test_cox_inputs_omit_paid_expiry_and_stratified_omits_days_until_expiry():
    rows = survival_rows()
    bands = RAW["survival"]["cox"]["strata_bands"]["p2"]
    plain = CoxModel(False, 0.1, [], bands, P.threads).fit(rows, COLS[:5])
    strat = CoxModel(True, 0.1, [], bands, P.threads).fit(rows, COLS[:5])
    assert "days_until_paid_expiry" not in plain.inputs and "days_until_expiry" in plain.inputs
    assert "days_until_expiry" not in strat.inputs
    p = plain.lapse_probability(rows.head(100), [7, 30, 59])
    assert (np.diff(p, axis=1) >= -1e-12).all()
    assert strat.lapse_probability(rows.head(100), [7]).shape == (100, 1)


def test_stratified_cox_batch_predictions_keep_row_order():
    rows = survival_rows(n=2000)
    bands = RAW["survival"]["cox"]["strata_bands"]["p2"]
    m = CoxModel(True, 0.1, [], bands, P.threads).fit(rows, COLS[:5])
    batch = m.lapse_probability(rows.head(60), [7])[:, 0]
    one = [m.lapse_probability(rows.iloc[[i]], [7])[0, 0] for i in range(60)]
    assert np.allclose(batch, one)


def test_a_cox_fit_reports_whether_it_beats_the_no_covariate_model():
    rows = survival_rows()
    m = CoxModel(False, 0.1, [], RAW["survival"]["cox"]["strata_bands"]["p2"], P.threads).fit(
        rows, COLS[:5]
    )
    assert m.fit_ok and m.log_likelihood >= m.null_log_likelihood
    assert m.summary()["fit_ok"] is True


def test_stratified_cox_predictions_follow_their_customer_in_any_row_order():
    # Regression test: lifelines groups stratified predictions by stratum.
    rows = survival_rows(n=2000)
    bands = RAW["survival"]["cox"]["strata_bands"]["p2"]
    m = CoxModel(True, 0.1, [], bands, P.threads).fit(rows, COLS[:5])
    scored = rows.head(300).reset_index(drop=True)
    in_order = m.lapse_probability(scored, [7])[:, 0]
    perm = np.random.default_rng(1).permutation(len(scored))
    shuffled = m.lapse_probability(scored.iloc[perm].reset_index(drop=True), [7])[:, 0]
    assert np.allclose(shuffled, in_order[perm])


def test_the_proportional_hazards_test_refuses_more_than_the_subsample_size():
    from retention.models.survival_stage import PH_MAX_ROWS, ph_by_group

    rows = survival_rows(n=PH_MAX_ROWS + 1)
    with pytest.raises(ValueError, match="fixed subsample"):
        ph_by_group(None, rows)
