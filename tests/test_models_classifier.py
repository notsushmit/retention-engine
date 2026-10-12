import numpy as np
import pandas as pd
import yaml

from retention.models.classifier import (
    Classifier,
    candidate_params,
    feature_set_columns,
    score_summary,
)
from retention.models.protocol import load_protocol

P = load_protocol()
RAW = yaml.safe_load(open("configs/evaluation.yaml"))


def rows(n=4000, seed=0, methods=(41, 36, 38)):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "days_until_expiry": rng.integers(0, 30, n),
            "last_is_auto_renew": rng.integers(0, 2, n),
            "last_payment_method_id": pd.array(rng.choice(methods, n), dtype="Int16"),
            "secs_per_day_28": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(2, 900, n)),
        }
    )
    logit = -3 + 2.5 * (df.days_until_expiry <= 7) - 1.5 * df.last_is_auto_renew
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def model(candidate="c00"):
    return Classifier(
        ["days_until_expiry", "last_is_auto_renew", "last_payment_method_id", "secs_per_day_28"],
        RAW["classifier"]["categorical"],
        candidate_params(RAW["classifier"]["candidates"][candidate]),
        RAW["classifier"]["fixed_params"],
        P.threads,
        P.seed,
    )


def test_feature_sets_have_the_registered_sizes():
    sizes = {
        fs: len(feature_set_columns(d["drop"]))
        for fs, d in RAW["selection"]["feature_sets"].items()
    }
    assert sizes == {"fs1": 40, "fs2": 44, "fs3": 46}
    assert "age" not in feature_set_columns([])  # the profile group is never an input


def test_the_same_fit_twice_with_four_threads_gives_identical_scores():
    fit, score = rows(seed=0), rows(seed=1)
    a = model("c01").fit(fit).predict(score)[0]
    b = model("c01").fit(fit).predict(score)[0]
    assert np.array_equal(a, b)
    assert model().fit(fit).booster.params["num_threads"] == 4


def test_unseen_category_levels_are_counted_and_scored_as_missing():
    fit, score = rows(seed=0), rows(seed=1, methods=(41, 36, 99))
    p, unseen = model().fit(fit).predict(score)
    assert unseen["last_payment_method_id"] == int((score.last_payment_method_id == 99).sum()) > 0
    assert np.isfinite(p).all()


def test_score_summary_and_importance():
    fit = rows()
    m = model().fit(fit)
    s = score_summary(m.predict(fit)[0])
    assert len(s["deciles"]) == 9 and s["distinct"] > 10 and 0 < s["mean"] < 1
    assert set(m.importance()) == set(m.columns)
