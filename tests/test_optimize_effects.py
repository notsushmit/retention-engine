from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retention.optimize.config import load_actions_config
from retention.optimize.effects import build_options, missing_deadline_count

CFG = load_actions_config()
BASE = CFG.base
R, P, D = 0, 1, 2  # reminder, autorenew_prompt, discount


def customers(**over):
    rows = {
        "msno": ["A", "B", "C"],
        "p30": [0.5, 0.02, 0.3],
        "auto_renew": ["off", "on", "off"],
        "price_per_day": [149 / 30, 99 / 30, 149 / 30],
        "days_30_90": [40.0, 55.0, 30.0],
        "days_30_365": [200.0, 300.0, 100.0],
        "days_until_paid_expiry": pd.array([10, 5, None], dtype="Int32"),
    }
    rows.update(over)
    return pd.DataFrame(rows)


def test_hand_case_base_point():
    o = build_options(customers(), CFG, BASE)
    assert o.actions == ("reminder", "autorenew_prompt", "discount")
    value_a = 0.5 * 149 / 30 * 40
    assert o.value[0] == pytest.approx(value_a)
    assert o.effect[0] == pytest.approx([0.5 * 0.03, 0.5 * 0.04, 0.5 * 0.10])
    assert o.cost[0, R] == 1.0 and o.cost[0, P] == 1.0
    assert o.cost[0, D] == pytest.approx(0.134 * 149 * (1 - 0.45))
    assert o.net[0] == pytest.approx(
        [value_a * 0.015 - 1, value_a * 0.02 - 1, value_a * 0.05 - 0.134 * 149 * 0.55]
    )
    assert o.allowed[0].all()


def test_prompt_not_offered_and_discount_too_late_for_b():
    o = build_options(customers(), CFG, BASE)
    assert o.allowed[1].tolist() == [True, False, False]  # deadline 5 < discount lead time 7
    assert o.effect[1, R] == 0.0  # reminder, auto-renew on, base cut 0
    assert o.net[1, R] == -1.0
    assert o.net[1, P] == 0.0 and o.cost[1, D] == 0.0


def test_missing_deadline_allows_no_action():
    c = customers()
    o = build_options(c, CFG, BASE)
    assert not o.allowed[2].any() and (o.net[2] == 0).all()
    assert missing_deadline_count(c) == 1


def test_lead_time_equal_to_deadline_is_allowed():
    o = build_options(
        customers(days_until_paid_expiry=pd.array([7, 2, 2], dtype="Int32")), CFG, BASE
    )
    assert o.allowed[0, D] and o.allowed[1, R] and o.allowed[2, R] and not o.allowed[2, D]


def test_absolute_cut_is_capped_at_p():
    c = customers(p30=[0.01, 0.02, 0.3])
    o = build_options(c, CFG, CFG.variants["absolute"])
    assert o.effect[0] == pytest.approx([0.005, 0.01, 0.01])  # discount 2 points capped at p = 1%
    assert o.effect[2].sum() == 0.0  # no deadline


def test_discount_cost_scaled_by_margin_variant():
    a = build_options(customers(), CFG, BASE)
    b = build_options(customers(), CFG, CFG.variants["discount_scaled_by_margin"])
    assert b.cost[0, D] == pytest.approx(a.cost[0, D] * 0.5)
    assert b.cost[0, R] == a.cost[0, R]


def test_value_horizon_and_margin_and_cost_levels():
    o = build_options(customers(), CFG, CFG.variants["value_365"])
    assert o.value[0] == pytest.approx(0.5 * 149 / 30 * 200)
    o = build_options(customers(), CFG, CFG.variants["margin_high"])
    assert o.value[0] == pytest.approx(1.0 * 149 / 30 * 40)
    o = build_options(customers(), CFG, CFG.variants["cost_high"])
    assert o.cost[0, R] == 2.0 and o.cost[0, P] == 2.0


def test_low_effect_gives_zero_effect_and_negative_or_zero_net():
    o = build_options(customers(), CFG, CFG.variants["low_effect"])
    assert (o.effect == 0).all()
    assert (o.net <= 0).all()


def test_zero_price_customer_has_zero_value_and_no_positive_net():
    o = build_options(customers(price_per_day=[0.0, 0.0, 0.0]), CFG, CFG.variants["optimistic"])
    assert (o.value == 0).all() and (o.net <= 0).all()


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="'off' or 'on'"):
        build_options(customers(auto_renew=[0, 1, 0]), CFG, BASE)
    with pytest.raises(ValueError, match="probability"):
        build_options(customers(p30=[0.5, np.nan, 0.1]), CFG, BASE)
    with pytest.raises(ValueError, match="one row per customer"):
        build_options(customers(msno=["A", "A", "C"]), CFG, BASE)


def test_unknown_scenario_level_is_an_error():
    with pytest.raises(KeyError):
        build_options(customers(), CFG, replace(BASE, level="medium"))
