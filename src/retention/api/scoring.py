"""The scoring function shared by the API and the parity run: status and probabilities.

Status (configs/serving.yaml, contract), for every customer with coverage at T:
- expiry more than 30 days ahead (not in_exp30): outside_population,
  expiry_beyond_30_days, whatever the segment;
- in_exp30 and trial-only or promo-only: outside_population, segment_trial_only
  or segment_promo_only;
- in_exp30 and paid or 100%-discount-only (the training population): scored,
  unless a category value was never seen by the models: outside_population,
  unseen_level:<feature> (refused, never guessed).
Every outside row keeps next_expiry_in_days (days_until_expiry). A null value
inside a feature row stays in population (native missing values).

Probabilities: c01 at 7, 30 and 59 days, Platt per horizon, then the registered
clip, pool-adjacent-violators fix and clip (retention.api.ordering, which calls
Phase 3's calibrate unchanged). A row "moved" when the fix changed any of its
three values by more than 1e-12, the threshold calibrate counts with. Display
text follows Phase 4's rule (one decimal; "under 0.1%" and "over 99.9%").

Reasons (amendment A3): tenure is never shown as a reason. Two routes, which
share no masking call:
- the service route builds its mask from the config alone: hidden groups and
  every group shown only when known (tenure) are hidden, whatever the row's
  tenure_days; then Phase 4's select and Renderer run unchanged, and the output
  frame is assembled column for column as Phase 4's reason_rows assembles it;
- the self-check route calls Phase 4's unmodified reason_rows with tenure_days
  set to missing, so Phase 4's own shown_mask hides tenure.
Their equality shows only that the masking matches Phase 4's own missing-tenure
behaviour; it does not validate the wording.

Expected paid days and the action block (Phase 7 code, unchanged):
- dt01's survival curve at every interval end, as Phase 7's value step computes
  it; Phase 7's customer_table turns it into expected paid days in days 31 to 90
  given no lapse by day 30 (amendment A2's quantity, days_30_90);
- the action: Phase 7's build_options and best_action under the base scenario,
  no budget, from the 30-day probability before the fix (D1): the reasons
  frame's probability column, the column Phase 7 read from reasons.parquet;
- a customer with no usable price is valued at 0 (D2) and carries the flag
  "no usable price", read as "no action pays".
Every action block carries effect_basis "assumed", the scenario, the instability
note and the no-estimate line, from configs/serving.yaml. Amendment A4: the action
is named highest_net_value_action; pays is true only when its net value is above
0, and a block that does not pay reads "no action pays".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from retention.api.labels import LABELS, Labels
from retention.api.ordering import HORIZONS, final_probabilities, platt_from
from retention.explain.build_reasons import caveat_flags, reason_rows
from retention.explain.config import STRENGTHS, ExplainConfig
from retention.explain.contributions import (
    CheckFailed,
    SavedModel,
    chunk_contributions,
    chunk_table,
    group_sums,
)
from retention.explain.reasons import Renderer, platt_probability, probability_text
from retention.explain.selection import POSITIONS, select, strength_codes
from retention.models.classifier import Classifier
from retention.models.protocol import Protocol
from retention.optimize.config import ActionsConfig, Scenario
from retention.optimize.effects import Options, build_options
from retention.optimize.solve import NONE
from retention.optimize.strategies import best_action
from retention.optimize.value import INTERVAL_ENDS_ALL, customer_table

SCORED = "scored"
OUTSIDE = "outside_population"
NOT_ACTIVE = "not_active"
TRIAL = "segment_trial_only"
PROMO = "segment_promo_only"
BEYOND = "expiry_beyond_30_days"
UNSEEN = "unseen_level"
MOVED_THRESHOLD = 1e-12
DECEMBER_2016 = (date(2016, 12, 1), date(2017, 1, 1))  # [start, end)


def population_status(status_rows: pd.DataFrame, main_segments: tuple[str, ...]) -> pd.DataFrame:
    """msno, status, code (None when scored) and next_expiry_in_days for every P1 row."""
    seg = status_rows.segment.astype(str).to_numpy()
    in_window = status_rows.in_exp30.astype(bool).to_numpy()
    known = set(main_segments) | {"trial_only", "promo_only"}
    unknown = sorted(set(seg) - known)
    if unknown:
        raise ValueError(f"segments outside the contract: {unknown}")
    code = np.where(
        ~in_window,
        BEYOND,
        np.where(seg == "trial_only", TRIAL, np.where(seg == "promo_only", PROMO, "")),
    )
    out = pd.DataFrame(
        {
            "msno": status_rows.msno.astype(str).to_numpy(),
            "status": np.where(code == "", SCORED, OUTSIDE),
            "code": pd.Series([c or None for c in code], dtype=object),
            "next_expiry_in_days": status_rows.days_until_expiry.to_numpy(),
        }
    )
    return out


def population_counts(status: pd.DataFrame) -> dict[str, int]:
    """The registered population_counts keys, recomputed from the status table."""
    code = status.code.fillna("")
    return {
        "scored": int((status.status == SCORED).sum()),
        "outside_population_segment_trial_only": int((code == TRIAL).sum()),
        "outside_population_segment_promo_only": int((code == PROMO).sum()),
        "outside_population_expiry_beyond_30_days": int((code == BEYOND).sum()),
        "total_with_coverage": int(len(status)),
    }


def unseen_feature(model: Classifier, rows: pd.DataFrame) -> np.ndarray:
    """Per row, the first category feature whose value the model never saw (None if none)."""
    out = np.full(len(rows), None, dtype=object)
    for c in model.cats:
        values = rows[c].astype("object")
        bad = values.notna().to_numpy() & ~values.isin(model.levels[c]).to_numpy()
        out[bad & pd.isna(out)] = c
    return out


def refuse_unseen(status: pd.DataFrame, rows: pd.DataFrame, models: list[Classifier]):
    """Mark scored rows with an unseen category value as outside_population, unseen_level:<f>.

    Returns the updated status table and the boolean mask of `rows` still scored.
    """
    first = np.full(len(rows), None, dtype=object)
    for m in models:
        f = unseen_feature(m, rows)
        first = np.where(pd.isna(first), f, first)
    refused = pd.Series(first, index=rows.msno.astype(str).to_numpy()).dropna()
    status = status.copy()
    hit = status.msno.isin(refused.index).to_numpy()
    status.loc[hit, "status"] = OUTSIDE
    status.loc[hit, "code"] = [f"{UNSEEN}:{refused[m]}" for m in status.msno[hit]]
    return status, np.array([f is None for f in first])


@dataclass
class Probabilities:
    raw: np.ndarray  # rows x 3 (7, 30, 59 days), the classifiers' raw probabilities
    before_fix: np.ndarray  # Platt per horizon, clipped
    final: np.ndarray  # after the ordering fix, clipped again
    moved: np.ndarray  # bool per row
    summary: dict[str, Any]  # calibrate's counts: rows_changed, max_change, ...

    def text(self) -> np.ndarray:
        """Display text of the final probabilities (rows x 3)."""
        return np.vectorize(probability_text, otypes=[object])(self.final)


def probabilities(c01: dict[int, SavedModel], rows: pd.DataFrame, seed: int) -> Probabilities:
    """The three final probabilities of every row, through the registered order."""
    if sorted(c01) != sorted(HORIZONS):
        raise ValueError(f"need c01 at {HORIZONS} days, got {sorted(c01)}")
    raw = np.column_stack([c01[h].model.predict(rows)[0] for h in HORIZONS])
    cals = [platt_from(c01[h].platt_a, c01[h].platt_b, seed) for h in HORIZONS]
    res = final_probabilities(raw, cals)
    moved = np.abs(res["final"] - res["before_fix"]).max(axis=1) > MOVED_THRESHOLD
    summary = {k: v for k, v in res.items() if k not in ("before_fix", "final")}
    return Probabilities(raw, res["before_fix"], res["final"], moved, summary)


def in_december_2016(snapshot: date) -> bool:
    return DECEMBER_2016[0] <= snapshot < DECEMBER_2016[1]


# --- reasons (amendment A3) --------------------------------------------------------------


def forced_hidden_mask(cfg: ExplainConfig, n: int) -> np.ndarray:
    """rows x groups, from the config alone: hidden groups and every group shown only
    when known (tenure) are never listed. No row value is read."""
    mask = np.ones((n, len(cfg.groups)), dtype=bool)
    for k, g in enumerate(cfg.group_names):
        if g in cfg.hidden or g in cfg.shown_only_when_known:
            mask[:, k] = False
    return mask


def contribution_part(
    model: SavedModel, rows: pd.DataFrame, cfg: ExplainConfig, columns: list[str], horizon: int
) -> pd.DataFrame:
    """Phase 4's contributions part (raw score, base value, phi, group sums, d) for `rows`."""
    raw, contrib = chunk_contributions(model.model, rows)
    gsum = group_sums(contrib, columns, cfg.groups)
    sel = select(cfg, gsum, raw - contrib[:, -1], model.platt_b, forced_hidden_mask(cfg, len(rows)))
    table = chunk_table(
        rows, horizon, raw, contrib, gsum, columns, cfg.group_names, sel, model.platt_b
    )
    return table.to_pandas()


def service_reasons(
    cfg: ExplainConfig,
    renderer: Renderer,
    part: pd.DataFrame,
    facts: pd.DataFrame,
    horizon: int,
    snapshot: date,
    platt: tuple[float, float],
    protocol: Protocol,
    phrases_sha256: str,
) -> pd.DataFrame:
    """The reasons the service shows: Phase 4's selection and wording with tenure hidden.

    Columns and order are those of reasons.parquet; each is assembled exactly as
    Phase 4's reason_rows assembles it, from Phase 4's own functions.
    """
    names = cfg.group_names
    df = part.merge(facts, on="msno", how="left", validate="one_to_one")
    if len(df) != len(part):
        raise CheckFailed("facts do not match the contributions rows")
    b = float(df.platt_b.iloc[0])
    raw_minus_base = (df.raw_score - df.base_value).to_numpy()
    sel = select(
        cfg,
        df[[f"g_{n}" for n in names]].to_numpy(),
        raw_minus_base,
        b,
        forced_hidden_mask(cfg, len(df)),
    )
    prob = platt_probability(df.raw_score.to_numpy(), *platt)
    flags = caveat_flags(df.msno, snapshot, protocol)
    fact_cols = sorted({c for cs in cfg.facts.values() for c in cs})
    records = df[fact_cols].astype(object).where(df[fact_cols].notna(), None).to_dict("records")
    keys = list(flags)
    texts, cancel = [], np.zeros(len(df), bool)
    for i, f in enumerate(records):
        caveats = tuple(k for k in keys if flags[k][i])
        out = renderer.render(sel, i, f, horizon, float(prob[i]), caveats)
        texts.append("\n".join(out.lines))
        cancel[i] = out.partly_cancel
    rows = np.arange(len(df))
    pos_s = np.where(sel.position != 0, strength_codes(np.abs(sel.d), cfg.cutoffs()), 0)

    def word(code):
        return [STRENGTHS[k - 1] if k > 0 else None for k in code]

    data: dict[str, Any] = {
        "msno": df.msno.astype(str).to_numpy(),
        "snapshot_date": np.full(len(df), snapshot),
        "horizon": np.full(len(df), horizon, dtype=np.int16),
        "raw_score": df.raw_score.to_numpy(),
        "base_value": df.base_value.to_numpy(),
        "platt_a": np.full(len(df), platt[0]),
        "platt_b": np.full(len(df), b),
        "d": sel.d,
        "probability": prob,
        "probability_text": [probability_text(float(p)) for p in prob],
        "position": [POSITIONS[int(p) + 1] for p in sel.position],
        "position_strength": word(pos_s),
    }
    for kind, idx, strength in (
        ("raising", sel.raising, sel.raising_strength),
        ("lowering", sel.lowering, sel.lowering_strength),
    ):
        for j in range(idx.shape[1]):
            ok = idx[:, j] >= 0
            data[f"{kind}_{j + 1}"] = [
                names[k] if o else None for k, o in zip(idx[:, j], ok, strict=True)
            ]
            data[f"{kind}_{j + 1}_strength"] = word(strength[:, j])
            data[f"{kind}_{j + 1}_c"] = np.where(ok, sel.c[rows, np.maximum(idx[:, j], 0)], np.nan)
    data["other_direction"] = np.where(
        sel.other_strength == 0, "small", np.where(sel.other > 0, "raises", "lowers")
    )
    data["other_strength"] = word(sel.other_strength)
    data["other"] = sel.other
    data["partly_cancel"] = cancel
    data |= {f"caveat_{k}": v for k, v in flags.items()}
    data["text"] = texts
    data["phrases_sha256"] = np.full(len(df), phrases_sha256)
    return pd.DataFrame(data)


def self_check_reasons(
    cfg: ExplainConfig,
    renderer: Renderer,
    part: pd.DataFrame,
    facts: pd.DataFrame,
    horizon: int,
    snapshot: date,
    platt: tuple[float, float],
    protocol: Protocol,
    phrases_sha256: str,
) -> pd.DataFrame:
    """Phase 4's unmodified reason_rows with every shown-only-when-known column missing."""
    masked = facts.copy()
    for column in cfg.shown_only_when_known.values():
        masked[column] = pd.NA
    return reason_rows(
        cfg, renderer, part, masked, horizon, snapshot, platt, protocol, phrases_sha256
    )


# --- expected paid days and the action block -----------------------------------------------


def survival(dt01, rows: pd.DataFrame) -> np.ndarray:
    """S(t) at every interval end (rows x 12), as Phase 7's value step computes it."""
    return 1 - dt01.lapse_probability(rows, list(INTERVAL_ENDS_ALL))[0]


def customers(
    rows: pd.DataFrame, surv: np.ndarray, reasons_30: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Phase 7's customer table; p30 is the 30-day reasons frame's probability (D1)."""
    p30 = reasons_30[["msno", "snapshot_date", "probability"]].rename(
        columns={"probability": "p30"}
    )
    return customer_table(rows, surv, p30, prices)


@dataclass
class Actions:
    opts: Options
    best: np.ndarray  # index into opts.actions, NONE when no action is allowed
    has_price: np.ndarray

    def frame(self) -> pd.DataFrame:
        """Per customer (amendment A4): the allowed action with the highest net value,
        its expected retained value, net value and cost ("none" and 0 when no action
        is allowed), and pays (true only when the net value is above 0)."""
        i = np.arange(self.opts.n)
        j = np.maximum(self.best, 0)
        chosen = self.best != NONE
        name = np.array([*self.opts.actions, "none"], dtype=object)
        net = np.where(chosen, self.opts.net[i, j], 0.0)
        return pd.DataFrame(
            {
                "msno": self.opts.msno,
                LABELS.action_field: name[np.where(chosen, self.best, len(self.opts.actions))],
                "expected_retained_value": np.where(
                    chosen, self.opts.value * self.opts.effect[i, j], 0.0
                ),
                "net_value": net,
                "cost": np.where(chosen, self.opts.cost[i, j], 0.0),
                "pays": chosen & (net > 0),
                "has_price": self.has_price,
            }
        )


def actions(table: pd.DataFrame, cfg: ActionsConfig) -> Actions:
    """Phase 7's per-customer best action under the base scenario, no budget."""
    opts = build_options(table, cfg, cfg.base)
    return Actions(opts, best_action(opts), table.has_price.to_numpy(dtype=bool))


def scenario_of(sc: Scenario, labels: Labels = LABELS) -> dict[str, Any]:
    """The scenario fields of an action block: Phase 7's base scenario and A4's text."""
    if sc.discount_cost_scaled_by_margin or sc.value_horizon_days != 90:
        raise ValueError("the scenario differs from the one whose wording is registered")
    return {
        "form": sc.form,
        "level": sc.level,
        "margin": sc.margin,
        "cost_level": sc.cost_level,
        "discount_cost_rule": labels.scenario_text["discount_cost_rule"],
        "value_horizon": labels.scenario_text["value_horizon"],
        "budget": labels.scenario_text["budget"],
    }


def action_labels(sc: Scenario, labels: Labels = LABELS) -> dict[str, Any]:
    """The fixed fields every action block carries."""
    scenario = scenario_of(sc, labels)
    if tuple(scenario) != labels.scenario_keys:
        raise ValueError("the registered scenario fields differ from the block's")
    return {
        "effect_basis": labels.effect_basis,
        "scenario": scenario,
        "instability_note": labels.instability_note,
        "no_estimate_line": labels.no_estimate_line,
        "probability_basis": labels.probability_basis,
    }


REQUIRED_ACTION_FIELDS = ("effect_basis", "scenario", "instability_note", "no_estimate_line")


def action_blocks(acts: Actions, sc: Scenario, labels: Labels = LABELS) -> list[dict[str, Any]]:
    """One block per customer under amendment A4, with D1's label and D2's flag."""
    fixed = action_labels(sc, labels)
    out = []
    for r in acts.frame().to_dict("records"):
        block = {
            labels.action_field: r[labels.action_field],
            "expected_retained_value": float(r["expected_retained_value"]),
            "net_value": float(r["net_value"]),
            "cost": float(r["cost"]),
            "pays": bool(r["pays"]),
            **fixed,
        }
        if not block["pays"]:
            block["reads_as"] = labels.reads_as
        if not r["has_price"]:
            block["price"] = labels.no_price_flag
        check_action_block(block, labels)
        out.append(block)
    return out


def check_action_block(block: dict[str, Any], labels: Labels = LABELS) -> None:
    """Refuse a block missing a registered field or breaking A4's reading."""
    missing = [k for k in REQUIRED_ACTION_FIELDS if block.get(k) in (None, "", {})]
    if missing:
        raise ValueError(f"action block lacks {missing}")
    if block["effect_basis"] != "assumed":
        raise ValueError("an action block's effect_basis must be 'assumed'")
    if block["pays"] != (block["net_value"] > 0):
        raise ValueError("pays must be true exactly when the net value is above 0")
    if block["pays"] == ("reads_as" in block):
        raise ValueError("reads_as is shown exactly when no action pays")
    if "price" in block and block["pays"]:
        raise ValueError("a customer without a usable price cannot have an action that pays")


def paid_days_block(days: float, labels: Labels = LABELS) -> dict[str, Any]:
    """Amendment A2's field, with its name and labels."""
    return {
        "field": labels.paid_days_field,
        "value": float(days),
        "name": labels.paid_days_name,
        "labels": list(labels.paid_days_labels),
    }
