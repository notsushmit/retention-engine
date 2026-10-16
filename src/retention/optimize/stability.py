"""How much the plan changes when an assumption moves (configs/actions.yaml, sensitivity).

Each variant's plan is compared with the base plan at the same budget:
- regret (headline): loss = the variant's optimum minus the base plan's net value
  scored under the variant, in price units; relative = loss / variant optimum
  when that optimum is above 0 (otherwise not defined);
- Jaccard of the targeted customers (two empty plans = 1; empty against
  non-empty = 0);
- action agreement among customers in both plans.
Stable = Jaccard >= 0.70 and relative regret <= 10% (the user's choice, not based
on data); the stricter reading (0.80, 5%) is reported, never judged. The
low-effect variant is never part of the overall verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from retention.optimize.effects import Options
from retention.optimize.solve import NONE

EXCLUDED_FROM_VERDICT = ("low_effect",)


@dataclass(frozen=True)
class Comparison:
    variant: str
    budget: float | None
    loss: float
    relative_regret: float | None
    jaccard: float
    action_agreement: float | None
    stable: bool | None  # None when relative regret is not defined
    stable_stricter: bool | None
    base_breaks_variant_limit: bool


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a != NONE, b != NONE
    union = int((sa | sb).sum())
    return 1.0 if union == 0 else int((sa & sb).sum()) / union


def action_agreement(a: np.ndarray, b: np.ndarray) -> float | None:
    both = (a != NONE) & (b != NONE)
    return None if not both.any() else float((a[both] == b[both]).mean())


def scored_net(opts: Options, choice: np.ndarray) -> float:
    idx = np.flatnonzero(choice != NONE)
    return math.fsum(opts.net[idx, choice[idx]].tolist())


def judge(jac: float, rel: float | None, jaccard_min: float, regret_max: float) -> bool | None:
    if rel is None:
        return None
    return jac >= jaccard_min and rel <= regret_max


def compare(
    variant: str,
    budget: float | None,
    variant_opts: Options,
    base_choice: np.ndarray,
    variant_choice: np.ndarray,
    thresholds: tuple[float, float],
    stricter: tuple[float, float],
    contact_limit: int | None = None,
) -> Comparison:
    optimum = scored_net(variant_opts, variant_choice)
    loss = optimum - scored_net(variant_opts, base_choice)
    rel = loss / optimum if optimum > 0 else None
    jac = jaccard(base_choice, variant_choice)
    return Comparison(
        variant=variant,
        budget=budget,
        loss=loss,
        relative_regret=rel,
        jaccard=jac,
        action_agreement=action_agreement(base_choice, variant_choice),
        stable=judge(jac, rel, *thresholds),
        stable_stricter=judge(jac, rel, *stricter),
        base_breaks_variant_limit=contact_limit is not None
        and int((base_choice != NONE).sum()) > contact_limit,
    )


def overall_verdict(at_headline: list[Comparison]) -> tuple[str, list[str]]:
    """'robust within the registered ranges' or 'unstable', with the failing variants."""
    judged = [c for c in at_headline if c.variant not in EXCLUDED_FROM_VERDICT]
    failed = [c.variant for c in judged if c.stable is not True]
    return ("unstable", failed) if failed else ("robust within the registered ranges", [])


def ranked_by_loss(at_headline: list[Comparison]) -> list[tuple[str, float]]:
    """Variants other than low_effect, largest loss first (ties by name)."""
    rows = [(c.variant, c.loss) for c in at_headline if c.variant not in EXCLUDED_FROM_VERDICT]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def robust_core(base_choice: np.ndarray, variant_choices: dict[str, np.ndarray]) -> int:
    """Customers chosen in the base plan and in every variant except low_effect."""
    keep = base_choice != NONE
    for name, choice in variant_choices.items():
        if name not in EXCLUDED_FROM_VERDICT:
            keep &= choice != NONE
    return int(keep.sum())


def robust_core_sentence(count: int) -> str:
    if count == 0:
        return "No customer is chosen under every assumption."
    if count == 1:
        return "1 customer is chosen under every assumption except zero effect."
    return f"{count:,} customers are chosen under every assumption except zero effect."


def budget_curve(points: list[tuple[float, float]]) -> list[tuple[float, float, float | None]]:
    """(budget, net value, extra net per extra price unit from the previous level)."""
    out: list[tuple[float, float, float | None]] = []
    prev = None
    for b, net in sorted(points):
        extra = None if prev is None or b == prev[0] else (net - prev[1]) / (b - prev[0])
        out.append((b, net, extra))
        prev = (b, net)
    return out


def action_ranks(opts: Options, choice: np.ndarray) -> dict[str, int]:
    """Actions ranked by net value contributed (1 = most), ties by action order."""
    idx = np.flatnonzero(choice != NONE)
    contrib = {
        a: math.fsum(opts.net[idx[choice[idx] == k], k].tolist())
        for k, a in enumerate(opts.actions)
    }
    order = sorted(opts.actions, key=lambda a: (-contrib[a], opts.actions.index(a)))
    return {a: r for r, a in enumerate(order, start=1)}


def discount_sentence(rank_full_cost: int, rank_margin_scaled: int) -> str:
    if rank_full_cost != rank_margin_scaled:
        return "the discount's rank depends on whether its cost is scaled by margin"
    return (
        "the discount's rank did not change in this run, but the choice of cost rule "
        "remains an assumption the result rests on"
    )
