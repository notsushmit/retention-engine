import numpy as np
import pytest

from retention.optimize.config import load_actions_config
from retention.optimize.effects import Options
from retention.optimize.solve import NONE, greedy
from retention.optimize.strategies import (
    baseline_choices,
    best_action,
    fill,
    projected_hours,
    random_draws,
    ranking,
    summarize,
)

BASE = load_actions_config().base
ACTIONS = ("reminder", "autorenew_prompt", "discount")
INDEX = {a: k for k, a in enumerate(ACTIONS)}


def make(cost, net, p=None, effect=None, value=None, allowed=None, msno=None):
    cost, net = np.asarray(cost, float), np.asarray(net, float)
    n = len(cost)
    allowed = np.ones_like(cost, bool) if allowed is None else np.asarray(allowed, bool)
    return Options(
        msno=np.array(msno or [f"m{i:03d}" for i in range(n)]),
        actions=ACTIONS,
        p=np.asarray(p if p is not None else np.linspace(0.9, 0.1, n), float),
        value=np.asarray(value if value is not None else np.ones(n), float),
        effect=np.asarray(effect if effect is not None else np.zeros_like(cost), float),
        cost=np.where(allowed, cost, 0.0),
        net=np.where(allowed, net, 0.0),
        allowed=allowed,
        scenario=BASE,
    )


def test_fill_skips_what_does_not_fit_and_continues():
    cost = np.array([[4.0], [7.0], [3.0], [5.0], [1.0]])
    action = np.zeros(5, dtype=np.int64)
    choice = fill(np.arange(5), action, cost, 9.0, None)
    # 4 fits (5 left), 7 skipped, 3 fits (2 left), 5 skipped, 1 fits (1 left)
    assert (choice != NONE).tolist() == [True, False, True, False, True]


def test_fill_respects_the_count_and_passes_over_customers_without_action():
    cost = np.ones((4, 1))
    action = np.array([0, NONE, 0, 0])
    choice = fill(np.arange(4), action, cost, None, 2)
    assert choice.tolist() == [0, NONE, 0, NONE]
    choice = fill(np.arange(4), action, cost, 100.0, 1)
    assert choice.tolist() == [0, NONE, NONE, NONE]


@pytest.mark.parametrize("seed", range(200))
def test_fill_matches_a_plain_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    cost = rng.uniform(0.5, 10, size=(n, 1)).round(1)
    action = np.where(rng.random(n) > 0.1, 0, NONE)
    order = rng.permutation(n)
    budget = float(rng.uniform(0, 80))
    limit = None if rng.random() < 0.5 else int(rng.integers(0, n + 1))
    expect = np.full(n, NONE)
    remaining, count = budget, 0
    for i in order:
        if action[i] == NONE or (limit is not None and count >= limit):
            continue
        if cost[i, 0] <= remaining + 1e-12:
            expect[i] = 0
            remaining -= cost[i, 0]
            count += 1
    got = fill(order, action, cost, budget, limit)
    assert got.tolist() == expect.tolist()
    assert cost[got == 0, 0].sum() <= budget + 1e-9


def test_best_action_even_negative_with_tie_rules():
    opts = make([[1, 2, 3], [1, 1, 5], [1, 1, 1]], [[-1, -3, -2], [2, 2, 2], [0, 0, 0]])
    opts.allowed[2] = False
    assert best_action(opts).tolist() == [0, 0, NONE]  # -1 best; equal net and cost: order


def test_ranking_ties_by_msno():
    assert ranking(np.array([0.5, 0.9, 0.5]), np.array(["c", "a", "b"])).tolist() == [1, 2, 0]


def test_baselines_target_by_their_rankings():
    # p: risk ranking 0,1,2; effect x value ranks 2 first
    opts = make(
        cost=[[1, 1, 1]] * 3,
        net=[[1, 0, 0], [2, 0, 0], [3, 0, 0]],
        p=[0.9, 0.5, 0.1],
        effect=[[0.1, 0, 0], [0.1, 0, 0], [0.5, 0, 0]],
        value=[1, 1, 10],
    )
    opt = greedy(opts, 1.0)
    out = baseline_choices(opts, 1.0, None, opt, INDEX)
    assert (out["highest_risk"] != NONE).tolist() == [True, False, False]
    assert (out["highest_assumed_uplift_x_value"] != NONE).tolist() == [False, False, True]
    assert (out["optimizer"] != NONE).tolist() == [False, False, True]
    assert (out["no_action"] == NONE).all()


def test_fixed_action_is_the_reminder_ranked_by_risk():
    allowed = [[False, True, True], [True, True, True], [True, True, True]]
    opts = make([[1, 1, 1]] * 3, [[1, 5, 9]] * 3, p=[0.9, 0.5, 0.1], allowed=allowed)
    out = baseline_choices(opts, 1.0, None, greedy(opts, 1.0), INDEX)
    assert out["fixed_action_reminder"].tolist() == [NONE, 0, NONE]


def test_no_limit_baselines_target_as_many_as_the_optimizer():
    opts = make([[1, 1, 1]] * 4, [[1, 0, 0], [-1, -1, -1], [2, 0, 0], [-1, -1, -1]])
    opt = greedy(opts, None)
    assert opt.contacts == 2
    out = baseline_choices(opts, None, None, opt, INDEX)
    assert (out["highest_risk"] != NONE).sum() == 2


def test_contact_limit_applies_to_baselines():
    opts = make([[1, 1, 1]] * 4, [[1, 0, 0]] * 4)
    out = baseline_choices(opts, 100.0, 2, greedy(opts, 100.0), INDEX)
    assert (out["highest_risk"] != NONE).sum() == 2


def test_random_draws_reproduce_and_summarize():
    rng = np.random.default_rng(3)
    opts = make(rng.uniform(1, 3, (50, 3)), rng.normal(0, 2, (50, 3)), effect=rng.random((50, 3)))
    opt = greedy(opts, 20.0)
    a = random_draws(opts, 20.0, None, opt, 30, seed=20161001)
    b = random_draws(opts, 20.0, None, opt, 30, seed=20161001)
    assert a == b
    assert a["expected_cost"]["p97_5"] <= 20.0 + 1e-9
    assert a["net_value"]["p2_5"] <= a["net_value"]["mean"] <= a["net_value"]["p97_5"]


def test_summarize_hand_case():
    opts = make([[2, 0, 5]], [[1, 0, 4]], effect=[[0.1, 0, 0.3]], value=[30])
    s = summarize(opts, np.array([2]))
    assert s["expected_retained_customers"] == pytest.approx(0.3)
    assert s["expected_retained_value"] == pytest.approx(9.0)
    assert s["expected_cost"] == 5 and s["net_value"] == 4 and s["contacted_discount"] == 1


def test_projected_hours():
    assert projected_hours(90.0, 112) == pytest.approx(2.8)
