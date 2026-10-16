"""Per customer and action: the assumed effect, the expected cost and the net value.

Every effect here is an assumption from configs/actions.yaml, never an estimate.

- Relative form: treated p = p * (1 - cut). Absolute form: treated p =
  max(p - cut, 0), so the effect never exceeds p.
- Value (price units) = margin * price per day * expected paid days in (30, H]
  given not lapsed by day 30, H = 90 or the 365-day extrapolation.
- Cost: reminder and auto-renew prompt cost their registered amount per customer;
  the discount costs depth * 30 * price per day * (1 - treated p), the forgone
  price paid by everyone who renews, times the margin only in the variant that
  scales it.
- Allowed: the action is offered to the customer's auto-renew level, and its
  lead time is at most the days until the paid expiry. A missing deadline allows
  no action.
- Net value = value * effect - cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retention.optimize.config import ActionsConfig, Scenario

CUSTOMER_COLUMNS = [
    "msno",
    "p30",
    "auto_renew",  # "off" or "on"
    "price_per_day",
    "days_30_90",
    "days_30_365",
    "days_until_paid_expiry",
]


@dataclass(frozen=True)
class Options:
    """Arrays of shape (customers, actions); actions in the registered tie order."""

    msno: np.ndarray
    actions: tuple[str, ...]
    p: np.ndarray
    value: np.ndarray  # per customer, includes the margin
    effect: np.ndarray
    cost: np.ndarray
    net: np.ndarray
    allowed: np.ndarray  # bool
    scenario: Scenario

    @property
    def n(self) -> int:
        return len(self.msno)


def check_customers(customers: pd.DataFrame) -> None:
    missing = set(CUSTOMER_COLUMNS) - set(customers.columns)
    if missing:
        raise ValueError(f"customer table lacks {sorted(missing)}")
    if not customers.auto_renew.isin(["off", "on"]).all():
        raise ValueError("auto_renew must be 'off' or 'on'")
    p = customers.p30.to_numpy(dtype=float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p30 must be a probability")
    if customers.msno.duplicated().any():
        raise ValueError("one row per customer")


def build_options(customers: pd.DataFrame, cfg: ActionsConfig, sc: Scenario) -> Options:
    check_customers(customers)
    actions = cfg.action_order
    p = customers.p30.to_numpy(dtype=float)
    price = customers.price_per_day.to_numpy(dtype=float)
    days = customers[f"days_30_{sc.value_horizon_days}"].to_numpy(dtype=float)
    value = sc.margin * price * days
    segment = customers.auto_renew.to_numpy()
    deadline = pd.to_numeric(customers.days_until_paid_expiry, errors="coerce").to_numpy(
        dtype=float
    )

    n, k = len(p), len(actions)
    effect = np.zeros((n, k))
    cost = np.zeros((n, k))
    allowed = np.zeros((n, k), dtype=bool)
    for j, a in enumerate(actions):
        cuts = cfg.cuts[sc.form][a]
        cut = np.array([cuts[s][sc.level] if s in cuts else 0.0 for s in segment])
        if sc.form == "relative":
            effect[:, j] = p * cut
        else:
            effect[:, j] = np.minimum(p, cut)
        offered = np.isin(segment, cfg.eligible[a])
        in_time = ~np.isnan(deadline) & (cfg.lead_time_days[a] <= np.nan_to_num(deadline, nan=-1))
        allowed[:, j] = offered & in_time
        if a == "discount":
            scale = sc.margin if sc.discount_cost_scaled_by_margin else 1.0
            cost[:, j] = cfg.discount_depth * 30.0 * price * (1.0 - (p - effect[:, j])) * scale
        else:
            cost[:, j] = cfg.contact_cost[a][sc.cost_level]
    effect[~allowed] = 0.0
    cost[~allowed] = 0.0
    net = value[:, None] * effect - cost
    net[~allowed] = 0.0
    return Options(
        msno=customers.msno.to_numpy(),
        actions=actions,
        p=p,
        value=value,
        effect=effect,
        cost=cost,
        net=net,
        allowed=allowed,
        scenario=sc,
    )


def missing_deadline_count(customers: pd.DataFrame) -> int:
    return int(pd.to_numeric(customers.days_until_paid_expiry, errors="coerce").isna().sum())
