"""The Scala port (exact copy) and the comparison, including the variant FITTED TO TRAIN."""

from dataclasses import replace
from datetime import date

import pandas as pd

from conftest import CFG
from retention.labels.compare import REASONS, compare
from retention.labels.official import scala_labels
from test_timeline import row

JAN = (date(2016, 1, 1), date(2016, 1, 31), date(2016, 2, 1), date(2016, 2, 29))


def scala(clean_tx, rows):
    return scala_labels(clean_tx(rows), *JAN)


def test_glued_string_tie_break(clean_tx):
    # same day, two plans: "993041" sorts after "1493041" in descending text order,
    # so the 149 row comes last and its expiry is "the" expiry
    out = scala(
        clean_tx,
        [row(20160110, 20160210, price=149, paid=149), row(20160110, 20160220, price=99, paid=99)],
    )
    assert str(out.loc["a", "last_expire"].date()) == "2016-02-10"


def test_strict_29_vs_30_day_gap(clean_tx):
    out = scala(
        clean_tx,
        [
            row(20160110, 20160210, m="g29"),
            row(20160310, 20160410, m="g29"),
            row(20160110, 20160210, m="g30"),
            row(20160311, 20160411, m="g30"),
        ],
    )
    assert out.loc["g29", "gap"] == 29 and out.loc["g29", "is_churn"] == 0
    assert out.loc["g30", "gap"] == 30 and out.loc["g30", "is_churn"] == 1  # the Scala's `< 30`


def test_cancels_move_expiry_earlier_and_no_renewal_is_churn(clean_tx):
    out = scala(
        clean_tx,
        [
            row(20160110, 20160220, m="cut"),
            row(20160205, 20160205, cancel=1, m="cut"),
            row(20160308, 20160408, m="cut"),  # 32 days after the cut
            row(20160110, 20160220, m="none"),
            row(20160110, 20160220, m="cancels"),
            row(20160225, 20160220, cancel=1, m="cancels"),
            row(20160110, 20160320, m="march"),  # expiry outside February: not a candidate
        ],
    )
    assert out.loc["cut", "gap"] == 32 and out.loc["cut", "is_churn"] == 1
    assert out.loc["none", "is_churn"] == 1 and out.loc["cancels", "is_churn"] == 1
    assert not out.loc["march", "candidate"] and pd.isna(out.loc["march", "is_churn"])


def test_placeholder_cancel_cuts_in_the_exact_port(clean_tx):
    out = scala(
        clean_tx,
        [
            row(20160110, 20160210),
            row(20160205, 19700101, cancel=1, pm=24),
            row(20160215, 20160315),
        ],
    )
    assert out.loc["a", "is_churn"] == 1  # the Scala compares 1970-01-01 as a date


def run_compare(clean_tx, rows, train):
    cfg = replace(CFG, as_of=date(2016, 6, 30), snapshots=(date(2016, 2, 1),)).validate()
    table, report = compare(
        clean_tx(rows), pd.Series(train, name="is_churn"), cfg, date(2016, 2, 1)
    )
    return table.set_index("msno"), report


def test_fitted_rules_and_reason_codes(clean_tx):
    rows = [
        # renews, then a March cancel cuts coverage: ours churned, train and fitted did not
        row(20160101, 20160210, m="mar"),
        row(20160210, 20160310, m="mar"),
        row(20160305, 20160305, cancel=1, m="mar"),
        # pays early and never again: ours retained, train (and the fitted rule) churned
        row(20160101, 20160220, m="early"),
        row(20160215, 20160320, m="early"),
        # renewal paid nothing: ours retained, train churned; the fitted rule cannot explain it
        row(20160101, 20160210, m="zero"),
        row(20160212, 20160312, m="zero", price=149, paid=0),
        # agrees
        row(20160101, 20160210, m="ok"),
        row(20160210, 20160310, m="ok"),
    ]
    table, report = run_compare(clean_tx, rows, {"mar": 0, "early": 1, "zero": 1, "ok": 0})
    assert (
        table.loc["mar", "coverage_ours"] == 1 and table.loc["mar", "coverage_fitted_to_train"] == 0
    )
    assert table.loc["mar", "coverage_ours_reason"] == "march_cancel"
    assert (
        table.loc["early", "coverage_ours"] == 0
        and table.loc["early", "coverage_fitted_to_train"] == 1
    )
    assert table.loc["early", "coverage_ours_reason"] == "early_renewal"
    assert table.loc["zero", "coverage_ours_reason"] == "zero_paid_renewal"
    assert table.loc["zero", "coverage_fitted_to_train_reason"] == "zero_paid_renewal"
    assert table.loc["zero", "paid_ours"] == 1  # paid-based: a zero-paid row is no renewal
    assert pd.isna(table.loc["ok", "coverage_ours_reason"])
    for fam in ("coverage", "paid"):
        for label in ("ours", "fitted_to_train"):
            r = report[fam][label]
            assert sum(r["reasons"].values()) == r["disagreements"]
            assert set(r["reasons"]) <= set(REASONS)
