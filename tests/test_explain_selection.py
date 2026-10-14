"""Selection of listed groups, strength labels, position, other factors, and fact summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from retention.explain.config import parse_explain_config
from retention.explain.describe import fact_summary, listed_mask
from retention.explain.selection import select, shown_mask, strength_codes
from test_explain_contributions import _raw_config

LN2, LN15, LN12 = math.log(2.0), math.log(1.5), math.log(1.2)
GROUPS = {
    "a": ["fa"],
    "b": ["fb"],
    "c": ["fc"],
    "d": ["fd"],
    "e": ["fe"],
    "tenure": ["tenure_days"],
    "hidden": ["registered_via"],
}
COLUMNS = [c for cols in GROUPS.values() for c in cols]
CFG = parse_explain_config(
    _raw_config(GROUPS, {"a": ["fa"]}, hidden=["hidden"], known={"tenure": "tenure_days"}),
    COLUMNS,
)
NAMES = CFG.group_names


def rows_with_tenure(known: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"tenure_days": pd.array([400 if k else None for k in known], "Int32")})


def run(c_rows, b: float = 1.0, tenure_known=None):
    c = np.asarray(c_rows, dtype=float)
    known = tenure_known if tenure_known is not None else [True] * len(c)
    shown = shown_mask(CFG, rows_with_tenure(known))
    return select(CFG, c / b, c.sum(axis=1) / b, b, shown)


def test_strength_boundaries_are_inclusive():
    below = lambda v: np.nextafter(v, 0.0)  # noqa: E731
    vals = np.array([LN2, below(LN2), LN15, below(LN15), LN12, below(LN12), 0.0])
    assert strength_codes(vals, CFG.cutoffs()).tolist() == [3, 2, 2, 1, 1, 0, 0]


def test_top_three_raising_and_top_two_lowering_with_group_order_ties():
    #        a     b     c     d     e    tenure hidden
    row = [0.30, 0.90, 0.30, 0.50, 0.25, -0.50, 0.0]
    s = run([row])
    assert [NAMES[i] for i in s.raising[0]] == ["b", "d", "a"]  # a ties c; a comes first
    assert s.raising_strength[0].tolist() == [3, 2, 1]
    assert [NAMES[i] for i in s.lowering[0] if i >= 0] == ["tenure"]
    row2 = [-0.30, -0.90, -0.20, -0.50, 0.0, 0.0, 0.0]
    s2 = run([row2])
    assert [NAMES[i] for i in s2.lowering[0]] == ["b", "d"]  # capped at two
    assert s2.lowering_strength[0].tolist() == [3, 2]


def test_hidden_and_missing_tenure_are_never_listed():
    #        a     b     c     d     e    tenure hidden
    rows = [
        [0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 5.00],  # hidden is the largest raise
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -5.0],  # hidden is the largest lowering
        [0.00, 0.30, 0.00, 0.00, 0.00, 3.00, 0.00],  # tenure missing: not listed
        [0.00, 0.30, 0.00, 0.00, 0.00, 3.00, 0.00],  # tenure known: listed
    ]
    s = run(rows, tenure_known=[True, True, False, True])
    hidden, tenure = NAMES.index("hidden"), NAMES.index("tenure")
    assert not (s.raising == hidden).any() and not (s.lowering == hidden).any()
    assert not (s.raising[2] == tenure).any()
    assert s.raising[3][0] == tenure
    assert s.other[0] == pytest.approx(5.0) and s.other[1] == pytest.approx(-5.0)
    assert s.other[2] == pytest.approx(3.0)
    assert np.max(np.abs(s.listed_sum() + s.other - s.d)) <= 1e-12


def test_listed_plus_other_equals_d_within_1e_12_on_random_rows():
    rng = np.random.default_rng(0)
    c = rng.normal(scale=0.5, size=(2000, len(NAMES)))
    s = run(c, b=0.9988, tenure_known=list(rng.random(2000) < 0.85))
    assert np.max(np.abs(s.listed_sum() + s.other - s.d)) <= 1e-12


def test_position_and_the_no_group_flags():
    small = LN12 / 4
    #        a      b      c      d      e    tenure hidden
    rows = [
        [small, small, small, small, small, 0.0, 0.0],  # higher, but no group reaches slight
        [-small, -small, -small, -small, -small, 0.0, 0.0],  # lower, no group
        [0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  # close, nothing stands out
        [0.5, -0.45, 0.0, 0.0, 0.0, 0.0, 0.0],  # close, but listed both ways
    ]
    s = run(rows)
    assert s.position.tolist() == [1, -1, 0, 0]
    assert s.many_small_raising().tolist() == [True, False, False, False]
    assert s.many_small_lowering().tolist() == [False, True, False, False]
    assert s.none_stand_out().tolist() == [False, False, True, False]
    assert (s.n_raising[3], s.n_lowering[3]) == (1, 1)


def test_other_factors_label():
    #        a     b    c    d    e  tenure hidden
    rows = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # other raises strong
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45],  # other lowers moderate
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],  # small net effect
    ]
    s = run(rows)
    assert s.other_strength.tolist() == [3, 2, 0]
    assert np.sign(s.other).tolist() == [1, -1, 1]


def test_scale_b_multiplies_every_contribution():
    c = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert run(c * 1.0, b=1.0).raising[0][0] == 0
    raw = c / 2.0  # with b = 2 the calibrated contribution is 0.2 again
    s = select(CFG, raw, raw.sum(axis=1), 2.0, shown_mask(CFG, rows_with_tenure([True])))
    assert s.c[0, 0] == pytest.approx(0.2) and s.raising[0][0] == 0


def test_a_slope_that_is_not_positive_raises():
    c = np.zeros((1, len(NAMES)))
    with pytest.raises(ValueError):
        select(CFG, c, c.sum(axis=1), 0.0, np.ones_like(c, dtype=bool))


def test_listed_mask():
    idx = np.array([[2, 0, -1], [-1, -1, -1]])
    m = listed_mask(idx, 4)
    assert m.tolist() == [[True, False, True, False], [False, False, False, False]]


# --- fact summaries ----------------------------------------------------------------------------


def test_numeric_fact_summary_with_reference_point():
    s = fact_summary(pd.Series([0.5, 2.0, None, 0.9], dtype="Float64"), "ratio_secs_7_vs_base")
    assert s["missing"] == {"share": 0.25, "count": 1, "of": 4}
    assert s["reference_points"]["< 1"] == {"share": 2 / 3, "count": 2, "of": 3}
    assert s["quantiles"]["p50"] == pytest.approx(0.9)


def test_two_reference_points_for_plan_length():
    s = fact_summary(pd.Series([30, 30, 0, 90, 410], dtype="Int32"), "last_plan_days")
    assert s["reference_points"]["== 30"]["count"] == 2
    assert s["reference_points"]["== 0"]["count"] == 1


def test_boolean_and_category_fact_summaries():
    b = fact_summary(pd.Series([1, 0, 1, None], dtype="Int8"), "last_is_auto_renew")
    assert b["true"] == {"share": 2 / 3, "count": 2, "of": 3}
    k = fact_summary(pd.Series(["paid", "trial", "paid", "promo"], dtype="string"), "last_row_kind")
    assert k["levels"]["paid"]["count"] == 2 and k["levels"]["discount100"]["count"] == 0
