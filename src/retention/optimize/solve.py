"""Choose at most one action per customer to maximize net value within a budget.

Greedy (the plan solver without a contact limit): each customer's allowed
actions with positive net value, plus "none" at (cost 0, net 0), are reduced to
their upper convex hull in (cost, net): sorted by cost, a dearer option is kept
only if it adds net value, and an option is dropped when the next step's extra
net value per extra cost is at least its own (strict concavity, so a customer's
steps are always funded in order). The steps of all customers are funded by
extra net per extra cost (descending), then extra net (descending), then msno,
then action order; a step that does not fit is skipped, and that customer's
later steps with it. With one budget constraint this is optimal up to one
customer: greedy >= LP bound - the largest single customer net value.

LP bound: the same steps taken in order, with the first one that does not fit
taken in part. It bounds every plan from above.

HiGHS (scipy.optimize.milp): the same problem as a 0-1 program, the exact check
on the greedy plan, and the plan solver when a contact limit applies.

After either solver, expected cost and contacts are checked exactly (math.fsum);
if a tolerance overshoots, chosen actions are removed, lowest net per cost first,
until the plan fits, and each removal is counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from retention.optimize.effects import Options

NONE = -1


@dataclass
class Plan:
    choice: np.ndarray  # action index per customer, NONE for no action
    cost: float
    net: float
    contacts: int
    solver: str
    lp_bound: float | None = None
    repairs: int = 0
    status: str = "optimal"
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    customer: int
    index: int  # position on the customer's hull, 1 = first action
    action: int
    d_cost: float
    d_net: float

    @property
    def efficiency(self) -> float:
        return math.inf if self.d_cost == 0 else self.d_net / self.d_cost


def customer_hull(cost: np.ndarray, net: np.ndarray, allowed: np.ndarray) -> list[int]:
    """Action indices on one customer's upper hull, cheapest first ("none" implied)."""
    cand = [j for j in range(len(cost)) if allowed[j] and net[j] > 0]
    # cheaper first; for equal cost the higher net, then action order
    cand.sort(key=lambda j: (cost[j], -net[j], j))
    pts: list[int] = []
    for j in cand:
        if pts and cost[j] == cost[pts[-1]]:
            continue  # same cost, lower or equal net, or later in action order
        if pts and net[j] <= net[pts[-1]]:
            continue  # dearer and not better
        pts.append(j)
        while len(pts) >= 2:
            c0 = cost[pts[-3]] if len(pts) >= 3 else 0.0
            n0 = net[pts[-3]] if len(pts) >= 3 else 0.0
            a, b = pts[-2], pts[-1]
            e1 = math.inf if cost[a] == c0 else (net[a] - n0) / (cost[a] - c0)
            e2 = (net[b] - net[a]) / (cost[b] - cost[a])
            if e2 >= e1:
                pts.pop(-2)  # the middle option is not on the strictly concave hull
            else:
                break
    return pts


def steps(opts: Options) -> list[Step]:
    out: list[Step] = []
    for i in range(opts.n):
        hull = customer_hull(opts.cost[i], opts.net[i], opts.allowed[i])
        c0 = n0 = 0.0
        for s, j in enumerate(hull, start=1):
            out.append(Step(i, s, j, opts.cost[i, j] - c0, opts.net[i, j] - n0))
            c0, n0 = opts.cost[i, j], opts.net[i, j]
    order = np.lexsort(
        (
            [st.action for st in out],
            [st.index for st in out],
            np.array([str(opts.msno[st.customer]) for st in out]),
            [-st.d_net for st in out],
            [-st.efficiency for st in out],
        )
    )
    return [out[k] for k in order]


def totals(opts: Options, choice: np.ndarray) -> tuple[float, float, int]:
    idx = np.flatnonzero(choice != NONE)
    cost = math.fsum(opts.cost[idx, choice[idx]].tolist())
    net = math.fsum(opts.net[idx, choice[idx]].tolist())
    return cost, net, len(idx)


def greedy(opts: Options, budget: float | None, ordered: list[Step] | None = None) -> Plan:
    """The greedy plan; budget None means no limit. `ordered` reuses steps(opts)."""
    ordered = steps(opts) if ordered is None else ordered
    remaining = math.inf if budget is None else float(budget)
    choice = np.full(opts.n, NONE, dtype=np.int64)
    at = np.zeros(opts.n, dtype=np.int64)  # hull position reached
    blocked = np.zeros(opts.n, dtype=bool)
    for st in ordered:
        if blocked[st.customer] or at[st.customer] != st.index - 1:
            continue
        if st.d_cost <= remaining:
            remaining -= st.d_cost
            choice[st.customer] = st.action
            at[st.customer] = st.index
        else:
            blocked[st.customer] = True
    cost, net, contacts = totals(opts, choice)
    plan = Plan(choice, cost, net, contacts, "greedy", lp_bound=lp_bound(opts, budget, ordered))
    return repair(opts, plan, budget, None)


def lp_bound(opts: Options, budget: float | None, ordered: list[Step] | None = None) -> float:
    """The LP relaxation's value: hull steps in order, the first misfit taken in part."""
    ordered = steps(opts) if ordered is None else ordered
    remaining = math.inf if budget is None else float(budget)
    total = 0.0
    for st in ordered:
        if st.d_cost <= remaining:
            remaining -= st.d_cost
            total += st.d_net
        else:
            if st.d_cost > 0 and remaining > 0:
                total += st.d_net * remaining / st.d_cost
            break
    return total


def highs(
    opts: Options,
    budget: float | None,
    contact_limit: int | None = None,
    time_limit: float | None = None,
) -> Plan:
    """The exact 0-1 program; only options with positive net value can be chosen.

    One row per customer with at least one candidate (at most one action), one
    budget row, one contact row. `time_limit` (seconds) is off unless given;
    whether the optimum was proven and HiGHS's dual bound go in plan.extra.
    """
    cand_i, cand_j = np.nonzero(opts.allowed & (opts.net > 0))
    m = len(cand_i)
    choice = np.full(opts.n, NONE, dtype=np.int64)
    if m == 0:
        cost, net, contacts = totals(opts, choice)
        return Plan(choice, cost, net, contacts, "highs", extra={"proven": True, "dual": 0.0})
    netv = opts.net[cand_i, cand_j]
    costv = opts.cost[cand_i, cand_j]
    _, row = np.unique(cand_i, return_inverse=True)
    k = int(row.max()) + 1
    rows = [sparse.csr_matrix((np.ones(m), (row, np.arange(m))), shape=(k, m))]
    lo, hi = [np.zeros(k)], [np.ones(k)]
    if budget is not None:
        rows.append(sparse.csr_matrix(costv.reshape(1, m)))
        lo.append(np.array([0.0]))
        hi.append(np.array([float(budget)]))
    if contact_limit is not None:
        rows.append(sparse.csr_matrix(np.ones((1, m))))
        lo.append(np.array([0.0]))
        hi.append(np.array([float(contact_limit)]))
    options = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    res = milp(
        c=-netv,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            sparse.vstack(rows).tocsr(), np.concatenate(lo), np.concatenate(hi)
        ),
        options=options,
    )
    if res.x is None:
        raise RuntimeError(f"HiGHS found no solution: {res.message}")
    take = res.x > 0.5
    choice[cand_i[take]] = cand_j[take]
    extra = {"proven": res.status == 0, "dual": float(-res.mip_dual_bound), "message": res.message}
    plan = Plan(choice, *totals(opts, choice), "highs", status=res.message, extra=extra)
    return repair(opts, plan, budget, contact_limit)


def repair(opts: Options, plan: Plan, budget: float | None, contact_limit: int | None) -> Plan:
    """Remove chosen actions, lowest net per cost first, until the plan fits exactly."""
    idx = np.flatnonzero(plan.choice != NONE)
    j = plan.choice[idx]
    eff = opts.net[idx, j] / np.where(opts.cost[idx, j] > 0, opts.cost[idx, j], np.inf)
    order = idx[np.lexsort((-np.arange(len(idx)), eff))]  # lowest efficiency first
    choice = plan.choice.copy()
    cost, _, contacts = totals(opts, choice)
    removed = 0
    for i in order:
        over_budget = budget is not None and cost > budget
        over_limit = contact_limit is not None and contacts > contact_limit
        if not (over_budget or over_limit):
            break
        choice[i] = NONE
        removed += 1
        cost, _, contacts = totals(opts, choice)
    if removed == 0:
        return plan
    cost, net, contacts = totals(opts, choice)
    return Plan(
        choice,
        cost,
        net,
        contacts,
        plan.solver,
        plan.lp_bound,
        plan.repairs + removed,
        plan.status,
        plan.extra,
    )


def breaches(
    opts: Options, plan: Plan, budget: float | None, contact_limit: int | None
) -> list[str]:
    """The runtime guard: every rule a plan must keep, checked exactly."""
    out = []
    cost, _, contacts = totals(opts, plan.choice)
    if budget is not None and cost > budget:
        out.append(f"expected cost {cost!r} above the budget {budget!r}")
    if contact_limit is not None and contacts > contact_limit:
        out.append(f"{contacts} contacts above the limit {contact_limit}")
    idx = np.flatnonzero(plan.choice != NONE)
    if not opts.allowed[idx, plan.choice[idx]].all():
        out.append("an action that is not allowed was chosen")
    if (plan.choice[idx] >= len(opts.actions)).any():
        out.append("unknown action index")
    return out


def solve(opts: Options, budget: float | None, contact_limit: int | None = None) -> Plan:
    """The registered plan: greedy without a contact limit, HiGHS with one."""
    if contact_limit is None:
        return greedy(opts, budget)
    plan = highs(opts, budget, contact_limit)
    plan.lp_bound = None
    return plan


class ExactCheckError(RuntimeError):
    """A proven HiGHS optimum below greedy, or greedy outside its bound: a bug, the run stops."""


def checked_plan(
    opts: Options,
    budget: float | None,
    contact_limit: int | None,
    time_limit: float | None,
    ordered: list[Step] | None = None,
    highs_fn=highs,
) -> tuple[Plan, dict]:
    """One cell's plan and its exact check (configs/actions.yaml, exact_check).

    Budget only: greedy is the plan. A proven HiGHS optimum must be at least greedy's
    net value minus 1e-6; an unproven one only marks the cell "not proven", with
    greedy's gap to the LP bound reported. With a contact limit: a proven HiGHS
    solution is the plan; an unproven one is the plan only if its net value is at
    least greedy's (greedy cut to the limit by the repair), otherwise greedy's.
    """
    g = greedy(opts, budget, ordered)
    max_single = float(opts.net.max(initial=0.0))
    if g.net < g.lp_bound - max_single - 1e-9:
        raise ExactCheckError(f"greedy {g.net} outside its bound {g.lp_bound} - {max_single}")
    h = highs_fn(opts, budget, contact_limit, time_limit)
    proven = bool(h.extra.get("proven"))
    check = {
        "greedy": g.net,
        "highs": h.net,
        "highs_proven": proven,
        "exact_check": "proven" if proven else "not proven",
        "highs_dual": h.extra.get("dual"),
        "lp_bound": g.lp_bound,
        "greedy_gap_to_lp_bound": g.lp_bound - g.net,
        "max_single_net": max_single,
    }
    if contact_limit is None:
        if proven and h.net < g.net - 1e-6:
            raise ExactCheckError(f"proven HiGHS optimum {h.net} below greedy {g.net}")
        return g, check
    capped = repair(opts, g, budget, contact_limit)
    check["greedy"] = capped.net
    if proven or h.net >= capped.net:
        check["plan_from"] = "highs"
        return h, check
    check["plan_from"] = "greedy"
    return capped, check
