"""Report rows and sentences: interval first, the label everywhere, aggregates only."""

import numpy as np
import pytest

from retention.uplift import report
from retention.uplift.config import load
from retention.uplift.kkbox_bootstrap import Result

COMPUTED = Result(
    {"aipw": 0.02, "t_learner": 0.01},
    "computed",
    None,
    {"aipw": (-0.1, 0.15), "t_learner": (-0.08, 0.1)},
    {"aipw": np.zeros(3), "t_learner": np.zeros(3)},
    1000,
    0.5,
)
NOT_COMPUTABLE = Result(
    {"aipw": 0.3},
    "not computable",
    "resample 4: a kept unit has a propensity of exactly 0 or 1",
    resamples_done=3,
)


def test_interval_columns_come_before_the_estimate():
    t = report.table(report.rows(COMPUTED, "primary"))
    cols = list(t.columns)
    assert cols.index("ci_low") < cols.index("ci_high") < cols.index("estimate")
    assert t.label.eq("balance not met").all()


def test_rows_refuse_any_unit_level_field():
    bad = report.rows(COMPUTED, "primary")
    bad[0]["msno"] = "x"
    with pytest.raises(ValueError, match="only aggregates"):
        report.table(bad)


def test_every_sentence_carries_the_label_and_the_interval_scope():
    for r in report.rows(COMPUTED, "primary") + report.rows(NOT_COMPUTABLE, "none"):
        s = report.sentence(r)
        assert "balance not met, assumptions unsupported" in s.lower()
        assert "not a finding" in s
        assert report.INTERVAL_SCOPE in s


def test_interval_appears_before_the_point_estimate_in_sentences():
    s = report.sentence(report.rows(COMPUTED, "primary")[0])
    assert s.index("95% interval -0.100 to +0.150") < s.index("point estimate +0.020")


def test_not_computable_rows_state_the_reason_and_carry_no_interval():
    (r,) = report.rows(NOT_COMPUTABLE, "none")
    assert r["ci_low"] is None and r["ci_high"] is None
    assert r["interval_status"].startswith("not computable: resample 4")
    assert "not computable" in report.sentence(r)


def test_no_advice_or_causal_wording():
    words = ("should", "recommend", "offer them", "causes", "will reduce", "keeps customers")
    for r in report.rows(COMPUTED, "primary") + report.rows(NOT_COMPUTABLE, "none"):
        s = report.sentence(r).lower()
        assert not any(w in s for w in words)


def test_label_and_reading_match_the_config():
    k = load()["kkbox"]
    assert report.LABEL in k["outcome_read"]["label"]
    assert report.READING in k["outcome_step"]["reading"]
    assert "not the uncertainty of the trimming itself" in k["outcome_step"]["interval_scope"]
