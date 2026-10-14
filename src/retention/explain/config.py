"""configs/explain.yaml as a checked, typed object.

Checks when it loads: every model column sits in exactly one group, every fact
column sits inside its own group, the hidden groups and the groups shown only
when known exist, and the strength cutoffs are ordered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

EXPLAIN_CONFIG = Path("configs/explain.yaml")
STRENGTHS = ("slight", "moderate", "strong")  # code 1, 2, 3; 0 = below the slight cutoff


@dataclass(frozen=True)
class ExplainConfig:
    horizons: tuple[int, ...]
    snapshots: tuple[date, ...]
    groups: dict[str, list[str]]  # in tie-break order
    hidden: tuple[str, ...]
    shown_only_when_known: dict[str, str]  # group -> column that must be known
    facts: dict[str, list[str]]
    strength_multipliers: dict[str, float]
    top_raising: int
    top_lowering: int
    position_multiplier: float
    other_small_multiplier: float
    max_additivity_error: float
    max_base_value_error: float
    memory_ceiling_gb: float
    reproduction_cells: tuple[tuple[int, date, tuple[str, ...]], ...]

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    def cutoffs(self) -> tuple[float, float, float]:
        """ln of the slight, moderate and strong multipliers."""
        m = self.strength_multipliers
        return tuple(math.log(m[s]) for s in STRENGTHS)  # type: ignore[return-value]

    @property
    def position_cutoff(self) -> float:
        return math.log(self.position_multiplier)

    @property
    def other_small_cutoff(self) -> float:
        return math.log(self.other_small_multiplier)


def _date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def parse_explain_config(raw: dict[str, Any], columns: list[str]) -> ExplainConfig:
    groups = {g: list(cols) for g, cols in raw["groups"].items()}
    seen: dict[str, str] = {}
    for g, cols in groups.items():
        for c in cols:
            if c in seen:
                raise ValueError(f"column {c!r} is in groups {seen[c]!r} and {g!r}")
            seen[c] = g
    missing, extra = set(columns) - set(seen), set(seen) - set(columns)
    if missing or extra:
        raise ValueError(f"groups do not cover the model columns: missing {missing}, extra {extra}")
    hidden = tuple(raw["hidden_groups"])
    known = dict(raw.get("shown_only_when_known", {}))
    for g in (*hidden, *known):
        if g not in groups:
            raise ValueError(f"group {g!r} is not defined")
    for g, c in known.items():
        if c not in groups[g]:
            raise ValueError(f"{c!r} is not in group {g!r}")
    facts = {g: list(cols) for g, cols in raw["facts"].items()}
    for g, cols in facts.items():
        if g not in groups or g in hidden:
            raise ValueError(f"facts for {g!r}, which is not a shown group")
        outside = [c for c in cols if c not in groups[g]]
        if outside:
            raise ValueError(f"facts of {g!r} come from outside the group: {outside}")
    mult = {s: float(raw["strength"][s]) for s in STRENGTHS}
    if not 1 < mult["slight"] < mult["moderate"] < mult["strong"]:
        raise ValueError(f"strength multipliers must rise above 1: {mult}")
    model, rows, contrib = raw["model"], raw["rows"], raw["contributions"]
    cells = tuple(
        (int(c["horizon"]), _date(c["snapshot"]), tuple(c["path"]))
        for c in model["reproduction"]["cells"]
    )
    return ExplainConfig(
        horizons=tuple(int(h) for h in model["horizons"]),
        snapshots=tuple(_date(t) for t in rows["snapshots"]),
        groups=groups,
        hidden=hidden,
        shown_only_when_known=known,
        facts=facts,
        strength_multipliers=mult,
        top_raising=int(raw["selection"]["top_raising"]),
        top_lowering=int(raw["selection"]["top_lowering"]),
        position_multiplier=float(raw["position_cutoff"]),
        other_small_multiplier=float(raw["other_factors"]["small_below"]),
        max_additivity_error=float(contrib["max_additivity_error"]),
        max_base_value_error=float(contrib["max_base_value_error"]),
        memory_ceiling_gb=float(contrib["memory_ceiling_gb"]),
        reproduction_cells=cells,
    )


def load_explain_config(columns: list[str], path: Path = EXPLAIN_CONFIG) -> ExplainConfig:
    return parse_explain_config(yaml.safe_load(path.read_text()), columns)
