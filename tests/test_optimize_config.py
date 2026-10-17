import copy

import pytest
import yaml

from retention.optimize.config import (
    ACTIONS_CONFIG,
    NO_LIMIT,
    load_actions_config,
    parse_actions_config,
)


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(ACTIONS_CONFIG.read_text())


def test_real_config_loads_with_the_registered_values():
    c = load_actions_config()
    assert c.segment_codes == {"off": 0, "on": 1}
    assert c.eligible == {
        "reminder": ("off", "on"),
        "autorenew_prompt": ("off",),
        "discount": ("off", "on"),
    }
    assert c.action_order == ("reminder", "autorenew_prompt", "discount")
    assert c.cuts["relative"]["discount"]["off"] == {"low": 0.0, "base": 0.10, "high": 0.25}
    assert c.cuts["absolute"]["reminder"]["on"] == pytest.approx(
        {"low": 0.0, "base": 0.0025, "high": 0.0075}
    )
    assert c.budgets == (0.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0, NO_LIMIT)
    assert c.discount_depth == 0.134
    assert c.lead_time_days == {"reminder": 2, "autorenew_prompt": 2, "discount": 7}
    assert (c.jaccard_min, c.regret_max, c.stricter) == (0.70, 0.10, (0.80, 0.05))
    assert c.random_draws == 1000 and c.random_seed == 20161001
    assert c.timing_stop_hours == 3.0
    assert c.exact_time_limit == 120.0 and c.unproven_statement_share == 0.10
    assert c.projection_stop_hours == 3.0 and c.projection_checks == (1, 10)
    assert len(c.variants) == 12


def test_variants_change_only_what_they_name():
    c = load_actions_config()
    s = c.scenarios()
    assert list(s)[0] == "base"
    assert s["pessimistic"].margin == 0.3 and s["pessimistic"].cost_level == "high"
    assert s["pessimistic"].level == "base" and s["pessimistic"].form == "relative"
    assert s["contact_limit"].contact_limit == 5000 and s["base"].contact_limit is None
    assert s["value_365"].value_horizon_days == 365
    assert s["discount_scaled_by_margin"].discount_cost_scaled_by_margin is True


def test_bare_off_and_on_are_refused(raw):
    bad = copy.deepcopy(raw)
    bad["scenarios"]["relative"]["cuts"]["reminder"] = {
        False: {"low": 0.0, "base": 0.03, "high": 0.08},
        True: {"low": 0.0, "base": 0.0, "high": 0.02},
    }
    with pytest.raises(ValueError, match="not text"):
        parse_actions_config(bad)


def test_effect_range_must_start_at_zero_and_be_ordered(raw):
    bad = copy.deepcopy(raw)
    bad["scenarios"]["relative"]["cuts"]["discount"]["off"]["low"] = 0.01
    with pytest.raises(ValueError, match="low must be 0"):
        parse_actions_config(bad)
    bad = copy.deepcopy(raw)
    bad["scenarios"]["absolute"]["cuts_pp"]["discount"]["off"]["base"] = 9.0
    with pytest.raises(ValueError, match="low <= base <= high"):
        parse_actions_config(bad)


def test_prompt_effect_for_auto_renew_on_is_refused(raw):
    bad = copy.deepcopy(raw)
    bad["scenarios"]["relative"]["cuts"]["autorenew_prompt"]["on"] = {
        "low": 0.0,
        "base": 0.01,
        "high": 0.02,
    }
    with pytest.raises(ValueError, match="offered to"):
        parse_actions_config(bad)


def test_banned_segment_column_is_refused(raw):
    bad = copy.deepcopy(raw)
    bad["segment"]["column"] = "city"
    with pytest.raises(ValueError, match="banned"):
        parse_actions_config(bad)


def test_label_must_say_not_estimates(raw):
    bad = copy.deepcopy(raw)
    bad["label"] = "assumed effect"
    with pytest.raises(ValueError, match="not estimates"):
        parse_actions_config(bad)


def test_empty_value_is_refused(raw):
    bad = copy.deepcopy(raw)
    bad["costs"]["reminder"]["base"] = None
    with pytest.raises(ValueError, match="empty"):
        parse_actions_config(bad)


def test_variant_with_unknown_field_is_refused(raw):
    bad = copy.deepcopy(raw)
    bad["sensitivity"]["variants"]["absolute"] = {"shape": "flat"}
    with pytest.raises(ValueError, match="unknown fields"):
        parse_actions_config(bad)
