import numpy as np
import pytest

from retention.optimize.config import load_actions_config
from retention.optimize.effects import Options
from retention.optimize.solve import NONE
from retention.optimize.stability import (
    Comparison,
    action_agreement,
    action_ranks,
    budget_curve,
    compare,
    discount_sentence,
    jaccard,
    overall_verdict,
    ranked_by_loss,
    robust_core,
    robust_core_sentence,
)

CFG = load_actions_config()
TH = (CFG.jaccard_min, CFG.regret_max)
STRICT = CFG.stricter
ACTIONS = ("reminder", "autorenew_prompt", "discount")
N = NONE


def opts_with_net(net):
    net = np.asarray(net, float)
    n = len(net)
    return Options(
        msno=np.array([f"m{i}" for i in range(n)]),
        actions=ACTIONS,
        p=np.zeros(n),
        value=np.zeros(n),
        effect=np.zeros_like(net),
        cost=np.ones_like(net),
        net=net,
        allowed=np.ones_like(net, bool),
        scenario=CFG.base,
    )


def test_hand_made_pair_is_unstable():
    # base targets customers 0-3; the variant's own plan targets 2-5 and is worth much more
    net = [[1, 0, 0]] * 2 + [[5, 0, 0]] * 4
    o = opts_with_net(net)
    base = np.array([0, 0, 0, 0, N, N])
    variant = np.array([N, N, 0, 0, 0, 0])
    c = compare("absolute", 10.0, o, base, variant, TH, STRICT)
    assert c.jaccard == pytest.approx(2 / 6)
    assert c.loss == pytest.approx(20 - 12)
    assert c.relative_regret == pytest.approx(8 / 20)
    assert c.stable is False and c.stable_stricter is False
    assert overall_verdict([c]) == ("unstable", ["absolute"])


def test_identical_plans_are_stable():
    o = opts_with_net([[1, 2, 3]] * 3)
    plan = np.array([2, N, 0])
    c = compare("margin_low", 10.0, o, plan, plan.copy(), TH, STRICT)
    assert (c.jaccard, c.loss, c.relative_regret, c.action_agreement) == (1.0, 0.0, 0.0, 1.0)
    assert c.stable is True


def test_thresholds_and_the_stricter_reading_is_separate():
    # base {0, 1, 2}; variant {0, 1, 2, 3}: Jaccard 0.75; loss 2 of 32 = 6.25%
    o = opts_with_net([[10, 0, 0]] * 3 + [[2, 0, 0]])
    c = compare("cost_low", 5.0, o, np.array([0, 0, 0, N]), np.array([0, 0, 0, 0]), TH, STRICT)
    assert c.jaccard == 0.75 and c.relative_regret == pytest.approx(2 / 32)
    assert c.stable is True and c.stable_stricter is False


def test_jaccard_edges():
    assert jaccard(np.array([N, N]), np.array([N, N])) == 1.0
    assert jaccard(np.array([N, N]), np.array([0, N])) == 0.0
    assert action_agreement(np.array([0, N]), np.array([N, 1])) is None
    assert action_agreement(np.array([0, 1, 2]), np.array([0, 2, 2])) == pytest.approx(2 / 3)


def test_low_effect_loss_is_the_spend_and_is_not_judged():
    o = opts_with_net([[-1, 0, 0], [-2, 0, 0]])  # zero effect: net = -cost
    base = np.array([0, 0])
    c = compare("low_effect", 5.0, o, base, np.array([N, N]), TH, STRICT)
    assert c.loss == pytest.approx(3.0) and c.relative_regret is None and c.stable is None
    assert overall_verdict([c]) == ("robust within the registered ranges", [])


def test_overall_names_every_failing_variant_and_ranks_by_loss():
    def comp(name, stable, loss):
        return Comparison(name, 10.0, loss, 0.0, 1.0, 1.0, stable, stable, False)

    rows = [
        comp("absolute", False, 9.0),
        comp("margin_low", True, 1.0),
        comp("value_365", None, 3.0),
    ]
    assert overall_verdict(rows) == ("unstable", ["absolute", "value_365"])
    rows.append(comp("low_effect", None, 50.0))
    assert ranked_by_loss(rows) == [("absolute", 9.0), ("value_365", 3.0), ("margin_low", 1.0)]


def test_robust_core_and_its_sentence():
    base = np.array([0, 0, 2, N])
    variants = {"absolute": np.array([0, N, 1, N]), "low_effect": np.array([N, N, N, N])}
    assert robust_core(base, variants) == 2  # low_effect ignored
    assert robust_core(base, {"absolute": np.array([N, N, N, 0])}) == 0
    assert robust_core_sentence(0) == "No customer is chosen under every assumption."
    assert robust_core_sentence(1).startswith("1 customer is chosen")
    assert robust_core_sentence(2500).startswith("2,500 customers are chosen")


def test_base_plan_breaking_the_variant_contact_limit_is_flagged():
    o = opts_with_net([[1, 0, 0]] * 3)
    c = compare("contact_limit", 5.0, o, np.array([0, 0, 0]), np.array([0, N, N]), TH, STRICT, 1)
    assert c.base_breaks_variant_limit and c.loss < 0


def test_budget_curve():
    assert budget_curve([(1000.0, 50.0), (0.0, 0.0), (2500.0, 80.0)]) == [
        (0.0, 0.0, None),
        (1000.0, 50.0, 0.05),
        (2500.0, 80.0, 0.02),
    ]


def test_discount_rank_and_both_sentences():
    o = opts_with_net([[1, 0, 9], [2, 0, 0], [3, 0, 0]])
    assert action_ranks(o, np.array([2, 0, 0])) == {
        "discount": 1,
        "reminder": 2,
        "autorenew_prompt": 3,
    }
    assert action_ranks(o, np.array([N, N, N]))["reminder"] == 1  # all zero: action order
    assert "depends" in discount_sentence(1, 2)
    assert "did not change in this run" in discount_sentence(2, 2)
