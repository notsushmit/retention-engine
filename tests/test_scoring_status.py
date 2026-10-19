"""Scoring, part 1: each customer's status, unseen levels refused, the three probabilities."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from retention.api import scoring as sc
from retention.explain.contributions import SavedModel
from retention.explain.model import booster_sha256, fit_models
from retention.explain.reasons import probability_text
from retention.models.protocol import load_protocol
from test_registry import SETTINGS, synthetic

P = load_protocol()
MAIN = tuple(P.main_segments)


def status_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["a", "b", "c", "d", "e", "f"],
            "in_exp30": [True, True, True, True, False, False],
            "segment": [
                "paid",
                "discount100_only",
                "trial_only",
                "promo_only",
                "promo_only",
                "paid",
            ],
            "days_until_expiry": [3, 9, 2, 6, 40, 75],
        }
    )


def test_status_codes_follow_the_contract():
    s = sc.population_status(status_table(), MAIN).set_index("msno")
    assert s.status.to_dict() == {
        "a": sc.SCORED,
        "b": sc.SCORED,
        "c": sc.OUTSIDE,
        "d": sc.OUTSIDE,
        "e": sc.OUTSIDE,
        "f": sc.OUTSIDE,
    }
    assert s.code.to_dict() == {
        "a": None,
        "b": None,
        "c": sc.TRIAL,
        "d": sc.PROMO,
        "e": sc.BEYOND,  # promo-only beyond 30 days counts as beyond, as the registered counts do
        "f": sc.BEYOND,
    }
    assert s.next_expiry_in_days.to_dict() == {"a": 3, "b": 9, "c": 2, "d": 6, "e": 40, "f": 75}


def test_population_counts_have_the_registered_keys():
    counts = sc.population_counts(sc.population_status(status_table(), MAIN))
    assert counts == {
        "scored": 2,
        "outside_population_segment_trial_only": 1,
        "outside_population_segment_promo_only": 1,
        "outside_population_expiry_beyond_30_days": 2,
        "total_with_coverage": 6,
    }


def test_a_segment_outside_the_contract_is_refused():
    t = status_table()
    t.loc[0, "segment"] = "unknown_kind"
    with pytest.raises(ValueError, match="outside the contract"):
        sc.population_status(t, MAIN)


# --- unseen category levels ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def c01() -> dict[int, SavedModel]:
    out = {}
    for k, h in enumerate((7, 30, 59)):
        model, platt = fit_models(synthetic(2000, k), synthetic(800, 20 + k), SETTINGS, h, P)
        out[h] = SavedModel(h, model, platt.a, platt.b, booster_sha256(model))
    return out


def test_an_unseen_level_is_named_and_a_missing_value_is_not(c01):
    rows = synthetic(4, 99)
    rows["cat"] = pd.array(["x", "w", None, "y"], dtype="string")
    assert sc.unseen_feature(c01[7].model, rows).tolist() == [None, "cat", None, None]


def test_unseen_rows_are_refused_and_leave_the_scored_set(c01):
    rows = synthetic(3, 98).assign(msno=["a", "b", "z"])
    rows["cat"] = pd.array(["x", "w", "y"], dtype="string")
    status = sc.population_status(
        pd.DataFrame(
            {"msno": ["a", "b", "z"], "in_exp30": True, "segment": "paid", "days_until_expiry": 4}
        ),
        MAIN,
    )
    status, keep = sc.refuse_unseen(status, rows, [m.model for m in c01.values()])
    s = status.set_index("msno")
    assert (s.loc["b", "status"], s.loc["b", "code"]) == (sc.OUTSIDE, "unseen_level:cat")
    assert s.loc["a", "status"] == sc.SCORED and keep.tolist() == [True, False, True]


# --- probabilities -------------------------------------------------------------------------------


def test_final_probabilities_are_ordered_and_moved_rows_are_flagged(c01):
    rows = synthetic(3000, 7)
    pr = sc.probabilities(c01, rows, P.seed)
    assert (np.diff(pr.final, axis=1) >= 0).all()
    changed = np.abs(pr.final - pr.before_fix).max(axis=1) > 1e-12
    assert np.array_equal(pr.moved, changed)
    assert pr.summary["rows_changed"] == int(pr.moved.sum())
    assert pr.text()[0, 0] == probability_text(float(pr.final[0, 0]))


def test_all_three_horizons_are_required(c01):
    with pytest.raises(ValueError, match="need c01"):
        sc.probabilities({7: c01[7], 30: c01[30]}, synthetic(5, 1), P.seed)


def test_december_2016_boundaries():
    assert not sc.in_december_2016(date(2016, 11, 30))
    assert sc.in_december_2016(date(2016, 12, 1)) and sc.in_december_2016(date(2016, 12, 31))
    assert not sc.in_december_2016(date(2017, 1, 1))
