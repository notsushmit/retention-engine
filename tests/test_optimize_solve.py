import itertools
import math

import numpy as np
import pytest

from retention.optimize.config import load_actions_config
from retention.optimize.effects import Options
from retention.optimize.solve import (
    NONE,
    ExactCheckError,
    Plan,
    breaches,
    checked_plan,
    customer_hull,
    greedy,
    highs,
    lp_bound,
    repair,
    solve,
    totals,
)

BASE = load_actions_config().base
ACTIONS = ("reminder", "autorenew_prompt", "discount")


def make(cost, net, allowed=None, msno=None):
    cost, net = np.asarray(cost, float), np.asarray(net, float)
    n = cost.shape[0]
    allowed = np.ones_like(cost, bool) if allowed is None else np.asarray(allowed, bool)
    return Options(
        msno=np.array(msno or [f"m{i:03d}" for i in range(n)]),
        actions=ACTIONS,
        p=np.zeros(n),
        value=np.zeros(n),
        effect=np.zeros_like(cost),
        cost=np.where(allowed, cost, 0.0),
        net=np.where(allowed, net, 0.0),
        allowed=allowed,
        scenario=BASE,
    )


def brute(opts, budget, limit=None):
    best = 0.0
    for combo in itertools.product(range(-1, len(ACTIONS)), repeat=opts.n):
        c = np.array(combo)
        idx = np.flatnonzero(c != NONE)
        if not opts.allowed[idx, c[idx]].all():
            continue
        cost, net, contacts = totals(opts, c)
        if (budget is None or cost <= budget) and (limit is None or contacts <= limit):
            best = max(best, net)
    return best


def random_case(rng, n):
    cost = rng.uniform(0.5, 10, size=(n, 3)).round(2)
    net = rng.normal(2, 4, size=(n, 3)).round(2)
    allowed = rng.random((n, 3)) > 0.2
    return make(cost, net, allowed)


def test_hand_case_greedy_upgrades_to_the_discount():
    # A: reminder (cost 1, net 5) or discount (cost 10, net 20); B: reminder (cost 1, net 2)
    opts = make(
        [[1, 0, 10], [1, 0, 0]],
        [[5, 0, 20], [2, 0, 0]],
        [[True, False, True], [True, False, False]],
    )
    plan = greedy(opts, 11)
    assert plan.choice.tolist() == [2, 0] and plan.net == 22 and plan.cost == 11
    assert brute(opts, 11) == 22
    # Budget 10: greedy funds both reminders (net 7) and cannot afford the 9-unit upgrade;
    # the optimum is the discount alone (20). The gap stays inside the registered bound.
    g = greedy(opts, 10)
    assert g.net == 7 and brute(opts, 10) == highs(opts, 10).net == 20
    assert g.net >= g.lp_bound - opts.net.max()


def test_hull_drops_dominated_and_non_concave_options():
    # reminder (1, 1), prompt (2, 4), discount (3, 4.5): reminder lies under the line to prompt
    assert customer_hull(np.array([1, 2, 3.0]), np.array([1, 4, 4.5]), np.ones(3, bool)) == [1, 2]
    # dearer and not better is dropped; negative net never enters
    assert customer_hull(np.array([1, 2, 3.0]), np.array([3, 3, -1]), np.ones(3, bool)) == [0]


@pytest.mark.parametrize("seed", range(40))
def test_tiny_cases_match_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    opts = random_case(rng, int(rng.integers(1, 6)))
    budget = float(rng.choice([0.0, 3.0, 7.5, 15.0, 40.0]))
    best = brute(opts, budget)
    g = greedy(opts, budget)
    h = highs(opts, budget)
    assert h.net == pytest.approx(best, abs=1e-9)
    assert g.net <= best + 1e-9
    assert lp_bound(opts, budget) >= best - 1e-9
    assert g.net >= g.lp_bound - opts.net.max(initial=0.0) - 1e-9
    limit = int(rng.integers(0, 4))
    assert highs(opts, budget, limit).net == pytest.approx(brute(opts, budget, limit), abs=1e-9)


@pytest.mark.parametrize("seed", range(500))
def test_never_over_budget_or_contact_limit(seed):
    rng = np.random.default_rng(10_000 + seed)
    opts = random_case(rng, int(rng.integers(1, 40)))
    budget = float(rng.uniform(0, 60))
    limit = int(rng.integers(0, 10))
    for plan, lim in ((greedy(opts, budget), None), (solve(opts, budget, limit), limit)):
        assert breaches(opts, plan, budget, lim) == []
        assert plan.cost <= budget and (lim is None or plan.contacts <= lim)
        idx = np.flatnonzero(plan.choice != NONE)
        assert (opts.net[idx, plan.choice[idx]] > 0).all()  # negative net never chosen


def test_no_limit_gives_every_customer_their_best_positive_action():
    opts = make([[1, 2, 9], [1, 1, 1], [5, 5, 5]], [[1, 3, 8], [-1, -2, 0], [2, 2, 1]])
    plan = greedy(opts, None)
    assert plan.choice.tolist() == [2, NONE, 0]  # equal net 2 and cost 5: action order
    assert plan.net == 10


def test_tie_rules_cheaper_then_action_order():
    opts = make([[3, 2, 9]], [[4, 4, 4]])
    assert greedy(opts, None).choice.tolist() == [1]  # equal net: the cheaper one
    opts = make([[2, 2, 9]], [[4, 4, 1]])
    assert greedy(opts, None).choice.tolist() == [0]  # equal net and cost: action order


def test_equal_steps_across_customers_break_by_msno():
    opts = make([[5, 0, 0], [5, 0, 0]], [[5, 0, 0], [5, 0, 0]], msno=["b", "a"])
    plan = greedy(opts, 5)
    assert plan.choice.tolist() == [NONE, 0]  # msno "a" first


def test_input_order_does_not_change_the_plan():
    rng = np.random.default_rng(7)
    opts = random_case(rng, 30)
    plan = greedy(opts, 25)
    perm = rng.permutation(opts.n)
    shuffled = make(opts.cost[perm], opts.net[perm], opts.allowed[perm], list(opts.msno[perm]))
    plan2 = greedy(shuffled, 25)
    a = dict(zip(opts.msno, plan.choice, strict=True))
    b = dict(zip(shuffled.msno, plan2.choice, strict=True))
    assert a == b


def test_highs_twice_gives_the_identical_selection():
    rng = np.random.default_rng(11)
    opts = random_case(rng, 200)
    a, b = highs(opts, 150), highs(opts, 150)
    assert (a.choice == b.choice).all() and a.net == b.net


def test_highs_not_below_greedy():
    for seed in range(20):
        opts = random_case(np.random.default_rng(300 + seed), 60)
        assert highs(opts, 40).net >= greedy(opts, 40).net - 1e-6


def test_injected_budget_breach_is_reported():
    opts = make([[4, 0, 0], [4, 0, 0]], [[3, 0, 0], [2, 0, 0]])
    bad = Plan(np.array([0, 0]), 8, 5, 2, "test")
    assert breaches(opts, bad, 5, 1) == [
        "expected cost 8.0 above the budget 5",
        "2 contacts above the limit 1",
    ]
    fixed = repair(opts, bad, 5, None)
    assert fixed.choice.tolist() == [0, NONE] and fixed.repairs == 1  # lowest net per cost first


def test_not_allowed_action_is_a_breach():
    opts = make([[1, 1, 1]], [[1, 1, 1]], [[True, False, True]])
    assert breaches(opts, Plan(np.array([1]), 0, 0, 1, "test"), None, None) == [
        "an action that is not allowed was chosen"
    ]


def test_zero_budget_and_empty_options():
    opts = make([[1, 1, 1]], [[1, 1, 1]])
    assert greedy(opts, 0).choice.tolist() == [NONE]
    empty = make([[1, 1, 1]], [[-1, -1, -1]])
    assert highs(empty, 10).choice.tolist() == [NONE] and lp_bound(empty, 10) == 0


def test_lp_bound_takes_the_misfit_in_part():
    opts = make([[2, 0, 0], [4, 0, 0]], [[4, 0, 0], [4, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    assert lp_bound(opts, 4) == pytest.approx(4 + 4 * 2 / 4)
    assert math.isclose(greedy(opts, 4).net, 4)


def stub(choice, proven, calls=None):
    """A HiGHS stand-in returning a fixed selection and proof status."""

    def fn(opts, budget, contact_limit, time_limit):
        if calls is not None:
            calls.append(time_limit)
        c = np.array(choice)
        return Plan(c, *totals(opts, c), "highs", extra={"proven": proven, "dual": 99.0})

    return fn


def test_checked_plan_passes_the_registered_time_limit():
    calls = []
    opts = make([[1, 0, 0]], [[2, 0, 0]])
    checked_plan(
        opts, 5.0, None, load_actions_config().exact_time_limit, None, stub([0], True, calls)
    )
    assert calls == [120.0]


def test_budget_only_unproven_keeps_greedy_and_marks_the_cell():
    opts = make([[1, 0, 10], [1, 0, 0]], [[5, 0, 20], [2, 0, 0]])
    plan, chk = checked_plan(opts, 11.0, None, 120.0, None, stub([NONE, NONE], False))
    assert plan.solver == "greedy" and plan.choice.tolist() == [2, 0]
    assert chk["exact_check"] == "not proven" and chk["highs"] == 0  # below greedy, not a failure
    assert chk["greedy_gap_to_lp_bound"] == pytest.approx(chk["lp_bound"] - plan.net)


def test_budget_only_proven_below_greedy_stops():
    opts = make([[1, 0, 10], [1, 0, 0]], [[5, 0, 20], [2, 0, 0]])
    with pytest.raises(ExactCheckError, match="below greedy"):
        checked_plan(opts, 11.0, None, 120.0, None, stub([NONE, NONE], True))


def test_contact_limit_unproven_uses_highs_only_if_at_least_greedy():
    opts = make([[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[5, 0, 0], [4, 0, 0], [3, 0, 0]])
    better = stub([0, 0, NONE], False)
    plan, chk = checked_plan(opts, 10.0, 2, 120.0, None, better)
    assert plan.solver == "highs" and chk["plan_from"] == "highs" and plan.net == 9
    worse = stub([NONE, NONE, 0], False)
    plan, chk = checked_plan(opts, 10.0, 2, 120.0, None, worse)
    assert chk["plan_from"] == "greedy" and chk["exact_check"] == "not proven"
    assert plan.choice.tolist() == [0, 0, NONE] and plan.contacts == 2  # greedy cut to the limit
    assert breaches(opts, plan, 10.0, 2) == []


def test_contact_limit_proven_highs_is_the_plan():
    opts = make([[1, 0, 0], [1, 0, 0]], [[5, 0, 0], [4, 0, 0]])
    plan, chk = checked_plan(opts, 10.0, 1, 120.0, None, stub([0, NONE], True))
    assert chk["exact_check"] == "proven" and plan.choice.tolist() == [0, NONE]


def test_real_highs_reports_proof_status():
    opts = random_case(np.random.default_rng(5), 30)
    plan, chk = checked_plan(opts, 20.0, None, 120.0)
    assert chk["exact_check"] == "proven" and chk["highs"] >= chk["greedy"] - 1e-6


class Res:
    def __init__(self, status, x, message="m", dual=-10.0):
        self.status, self.x, self.message, self.mip_dual_bound = status, x, message, dual


def fake_milp(monkeypatch, res):
    import retention.optimize.solve as solve_module

    monkeypatch.setattr(solve_module, "milp", lambda **kw: res)


STATUS_OPTS = make([[1, 0, 0], [1, 0, 0]], [[5, 0, 0], [4, 0, 0]])


@pytest.mark.parametrize(
    ("status", "has_x", "proven", "feasible"),
    [(0, True, True, True), (1, True, False, True), (1, False, False, False)],
)
def test_handled_highs_statuses(monkeypatch, status, has_x, proven, feasible):
    fake_milp(monkeypatch, Res(status, np.array([1.0, 0.0]) if has_x else None))
    plan = highs(STATUS_OPTS, 10.0, None, 120.0)
    assert plan.extra["proven"] is proven and plan.extra["feasible"] is feasible
    if feasible:
        assert plan.choice.tolist() == [0, NONE] and plan.extra["dual"] == 10.0
    else:
        assert plan.choice.tolist() == [NONE, NONE] and plan.net == -np.inf


@pytest.mark.parametrize(
    ("status", "has_x"), [(2, False), (3, False), (4, False), (4, True), (0, False), (7, True)]
)
def test_impossible_highs_statuses_stop(monkeypatch, status, has_x):
    from retention.optimize.solve import HighsStatusError

    fake_milp(monkeypatch, Res(status, np.array([1.0, 0.0]) if has_x else None))
    with pytest.raises(HighsStatusError, match="status"):
        highs(STATUS_OPTS, 10.0, None, 120.0)


def no_solution_stub(opts, budget, contact_limit, time_limit):
    c = np.full(opts.n, NONE, dtype=np.int64)
    return Plan(c, 0.0, -np.inf, 0, "highs", extra={"proven": False, "feasible": False})


def test_no_feasible_solution_budget_only_keeps_greedy():
    from retention.optimize.solve import NO_SOLUTION

    opts = make([[1, 0, 10], [1, 0, 0]], [[5, 0, 20], [2, 0, 0]])
    plan, chk = checked_plan(opts, 11.0, None, 120.0, None, no_solution_stub)
    assert plan.solver == "greedy" and plan.choice.tolist() == [2, 0]
    assert chk["exact_check"] == NO_SOLUTION and chk["highs"] is None
    assert not chk["highs_proven"] and not chk["highs_feasible"]


def test_no_feasible_solution_contact_limit_uses_greedy_cut_to_the_limit():
    from retention.optimize.solve import NO_SOLUTION

    opts = make([[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[5, 0, 0], [4, 0, 0], [3, 0, 0]])
    plan, chk = checked_plan(opts, 10.0, 2, 120.0, None, no_solution_stub)
    assert chk["plan_from"] == "greedy" and chk["exact_check"] == NO_SOLUTION
    assert plan.choice.tolist() == [0, 0, NONE] and breaches(opts, plan, 10.0, 2) == []
    # even when greedy's capped value is 0, the empty HiGHS stand-in is never chosen
    zero = make([[1, 0, 0]], [[-1, 0, 0]])
    plan, chk = checked_plan(zero, 10.0, 1, 120.0, None, no_solution_stub)
    assert chk["plan_from"] == "greedy"
