"""configs/actions.yaml as a checked, typed object.

Checks when it loads: the auto-renew levels are the strings "off" and "on" (bare
YAML reads them as booleans), every effect range starts at 0 and is ordered, an
action has effects exactly for the levels it is offered to, no banned column is
the segment, the effect label says the numbers are not estimates, and no value
is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ACTIONS_CONFIG = Path("configs/actions.yaml")
LEVELS = ("low", "base", "high")
SEGMENTS = ("off", "on")
FORMS = ("relative", "absolute")
NO_LIMIT = None  # the "no_limit" budget level


@dataclass(frozen=True)
class Scenario:
    """One point of the sensitivity analysis."""

    name: str
    form: str
    level: str
    margin: float
    cost_level: str
    discount_cost_scaled_by_margin: bool
    value_horizon_days: int
    contact_limit: int | None
    snapshot: date

    def fields(self) -> dict[str, Any]:
        """The scenario fields every output row carries."""
        return {
            "scenario": self.name,
            "form": self.form,
            "level": self.level,
            "margin": self.margin,
            "cost_level": self.cost_level,
            "discount_cost_scaled_by_margin": self.discount_cost_scaled_by_margin,
            "value_horizon_days": self.value_horizon_days,
            "contact_limit": self.contact_limit,
            "snapshot": self.snapshot,
        }


@dataclass(frozen=True)
class ActionsConfig:
    label: str
    effect_basis: str
    segment_column: str
    segment_codes: dict[str, int]  # "off" -> 0, "on" -> 1
    never_used: tuple[str, ...]
    action_order: tuple[str, ...]  # tie-break order, "none" excluded
    eligible: dict[str, tuple[str, ...]]  # action -> segments it is offered to
    # form -> action -> segment -> level -> cut as a fraction (absolute cuts converted from points)
    cuts: dict[str, dict[str, dict[str, dict[str, float]]]]
    snapshots: tuple[date, ...]
    margins: dict[str, float]
    value_interval_ends: tuple[int, ...]
    extrapolation_from: tuple[int, int]  # the interval whose daily hazard is held constant
    contact_cost: dict[str, dict[str, float]]  # reminder and autorenew_prompt -> level -> cost
    discount_depth: float
    lead_time_days: dict[str, int]
    budgets: tuple[float | None, ...]  # NO_LIMIT for "no_limit"
    contact_limit_case: int
    random_draws: int
    random_seed: int
    base: Scenario
    variants: dict[str, Scenario]
    jaccard_min: float
    regret_max: float
    stricter: tuple[float, float]
    headline_budget_level: float | None | str  # "not yet registered" until its own commit
    memory_ceiling_gb: float
    threads: int
    caption: str
    observed_label: str
    effect_prefix: str
    disclaimers: tuple[str, ...]
    banned_words: tuple[str, ...]
    retained_allowed: tuple[str, ...]
    uplift_allowed: str
    works_exception: str
    external_names: tuple[str, ...]
    external_exempt: tuple[str, ...]
    timing_stop_hours: float
    exact_time_limit: float  # seconds per HiGHS solve
    unproven_statement_share: float
    reproduction_cells: tuple[tuple[int, date, tuple[str, ...]], ...]
    dt01_fit_snapshots: tuple[date, ...]
    dt01_labels_as_of: date
    seed: int

    def scenarios(self) -> dict[str, Scenario]:
        """The base point first, then the variants in registered order."""
        return {"base": self.base, **self.variants}


def _date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def _empty_values(d: Any, path: str = "") -> list[str]:
    if isinstance(d, dict):
        out = [f"{path}/{k!r} (key is not text)" for k in d if not isinstance(k, str)]
        for k, v in d.items():
            out += _empty_values(v, f"{path}/{k}")
        return out
    if isinstance(d, list):
        return [p for i, v in enumerate(d) for p in _empty_values(v, f"{path}[{i}]")]
    return [path] if d is None else []


def _ranges(raw: dict[str, Any], scale: float, form: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for action, by_segment in raw.items():
        out[action] = {}
        for segment, levels in by_segment.items():
            if segment not in SEGMENTS:
                raise ValueError(f"{form} {action}: segment {segment!r} is not 'off' or 'on'")
            values = [float(levels[k]) * scale for k in LEVELS]
            if values[0] != 0.0 or not values[0] <= values[1] <= values[2] or values[2] > 1.0:
                raise ValueError(
                    f"{form} {action} {segment}: low must be 0 and low <= base <= high <= 1"
                )
            out[action][segment] = dict(zip(LEVELS, values, strict=True))
    return out


def _scenario(name: str, raw: dict[str, Any]) -> Scenario:
    limit = raw["contact_limit"]
    return Scenario(
        name=name,
        form=raw["form"],
        level=raw["level"],
        margin=float(raw["margin"]),
        cost_level=raw["cost_level"],
        discount_cost_scaled_by_margin=bool(raw["discount_cost_scaled_by_margin"]),
        value_horizon_days=int(raw["value_horizon_days"]),
        contact_limit=None if limit == "none" else int(limit),
        snapshot=_date(raw["snapshot"]),
    )


def parse_actions_config(raw: dict[str, Any]) -> ActionsConfig:
    empty = _empty_values(raw)
    if empty:
        raise ValueError(f"empty values or non-text keys: {empty}")
    if "not estimates" not in raw["label"] or raw["effect_basis"] != "assumed":
        raise ValueError("the effect label must say the numbers are assumed and not estimates")

    seg = raw["segment"]
    codes = dict(seg["levels"])
    if codes != {"off": 0, "on": 1}:
        raise ValueError(f"segment levels must be {{'off': 0, 'on': 1}}, got {codes!r}")
    never = tuple(seg["never_used"])
    if seg["column"] in never or any(n in seg["column"] for n in never):
        raise ValueError(f"segment column {seg['column']!r} is a banned column")

    actions = {a: tuple(v["eligible"]) for a, v in raw["actions"].items() if a != "none"}
    for a, segs in actions.items():
        if not set(segs) <= set(SEGMENTS):
            raise ValueError(f"{a}: eligibility {segs!r} is not a subset of {SEGMENTS}")
    order = tuple(raw["optimizer"]["ties"]["action_order"])
    if set(order) != set(actions):
        raise ValueError(f"action order {order} does not match the actions {tuple(actions)}")

    sc = raw["scenarios"]
    cuts = {
        "relative": _ranges(sc["relative"]["cuts"], 1.0, "relative"),
        "absolute": _ranges(sc["absolute"]["cuts_pp"], 0.01, "absolute"),
    }
    for form, by_action in cuts.items():
        if set(by_action) != set(actions):
            raise ValueError(f"{form}: effects for {set(by_action)}, actions {set(actions)}")
        for a, by_segment in by_action.items():
            if set(by_segment) != set(actions[a]):
                raise ValueError(
                    f"{form} {a}: effects for {set(by_segment)}, offered to {set(actions[a])}"
                )

    value = raw["value"]
    ends = (30, 37, 44, 51, 59, 67, 74, 81, 90)
    if (
        f"interval ends {', '.join(map(str, ends))}"
        not in value["expected_paid_days_30_90"]["computation"]
    ):
        raise ValueError("the value interval ends differ from the registered ones")
    costs = raw["costs"]
    budgets = tuple(NO_LIMIT if b == "no_limit" else float(b) for b in raw["budget"]["levels"])
    sens = raw["sensitivity"]
    base_raw = dict(sens["base_point"])
    base = _scenario("base", base_raw)
    variants = {}
    for name, change in sens["variants"].items():
        extra = set(change) - set(base_raw)
        if extra:
            raise ValueError(f"variant {name} changes unknown fields {extra}")
        variants[name] = _scenario(name, {**base_raw, **change})
    if base.contact_limit is not None:
        raise ValueError("the base point has no contact limit")
    stab = sens["stability"]
    th = stab["thresholds"]
    strict = th["stricter_reading"]
    level = stab["headline_budget"]["level"]
    words = raw["wording_rules"]
    refit = raw["dt01_refit"]
    if refit["fit_rows"] != "P2 main rows of 2016-08-01 and 2016-09-01":
        raise ValueError(f"unexpected dt01 fit rows {refit['fit_rows']!r}")
    cells = tuple(
        (int(c["horizon"]), _date(c["snapshot"]), tuple(str(p) for p in c["path"]))
        for c in refit["reproduction"]["cells"]
    )
    rnd = raw["strategies"]["random"]
    return ActionsConfig(
        label=raw["label"],
        effect_basis=raw["effect_basis"],
        segment_column=seg["column"],
        segment_codes=codes,
        never_used=never,
        action_order=order,
        eligible=actions,
        cuts=cuts,
        snapshots=tuple(_date(s) for s in raw["population"]["snapshots"]),
        margins={k: float(v) for k, v in value["margin"].items()},
        value_interval_ends=ends,
        extrapolation_from=(81, 90),
        contact_cost={
            a: {k: float(v) for k, v in costs[a].items()} for a in ("reminder", "autorenew_prompt")
        },
        discount_depth=float(costs["discount"]["depth"]),
        lead_time_days={a: int(raw["lead_time_days"][a]) for a in actions},
        budgets=budgets,
        contact_limit_case=int(
            raw["optimizer"]["constraints"]["contact_limit"]["case"]["total_contacted_max"]
        ),
        random_draws=int(rnd["draws"]),
        random_seed=int(str(rnd["seed"]).split()[0]),
        base=base,
        variants=variants,
        jaccard_min=float(th["stable"].split("jaccard >= ")[1].split()[0]),
        regret_max=float(th["stable"].split("relative regret <= ")[1].split()[0]),
        stricter=(float(strict["jaccard"]), float(strict["relative_regret"])),
        headline_budget_level=level
        if isinstance(level, str)
        else (NO_LIMIT if level == "no_limit" else float(level)),
        memory_ceiling_gb=float(raw["pass_marks"]["resources"]["memory_ceiling_gb"]),
        threads=int(raw["pass_marks"]["resources"]["threads"]),
        caption=raw["strategies"]["table_caption"],
        observed_label=raw["observed_churn_check"]["label"],
        effect_prefix=words["effect_sentence_prefix"],
        disclaimers=tuple(words["fixed_disclaimers"]),
        banned_words=tuple(words["banned_words"]),
        retained_allowed=tuple(words["retained_allowed_only_in"]),
        uplift_allowed=words["uplift_allowed_only_as"],
        works_exception=words["works_exception"],
        external_names=tuple(words["external_figures_ban"]["names"]),
        external_exempt=tuple(words["external_figures_ban"]["exempt"]),
        timing_stop_hours=float(raw["random_baseline_timing"]["stop_hours"]),
        exact_time_limit=float(raw["exact_check"]["time_limit_seconds"]),
        unproven_statement_share=float(
            raw["exact_check"]["unproven_report"]["statement_above_share"]
        ),
        reproduction_cells=cells,
        dt01_fit_snapshots=(date(2016, 8, 1), date(2016, 9, 1)),
        dt01_labels_as_of=_date(refit["labels_as_of"]),
        seed=int(refit["seed"]),
    )


def load_actions_config(path: Path = ACTIONS_CONFIG) -> ActionsConfig:
    return parse_actions_config(yaml.safe_load(path.read_text()))
