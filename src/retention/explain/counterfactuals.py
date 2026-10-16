"""Counterfactuals of the frozen 30-day churn model (configs/uplift.yaml, counterfactuals;
wording in configs/counterfactual_phrases.yaml).

A counterfactual is the model's calibrated score under a changed input, not the effect of
any action. One lever is registered: auto-renew, from off to on only; it is never switched
off. It is offered for a row only when the comparable group among the model's training
rows (customers expiring within 30 days at 2016-08-01 and 2016-09-01) holds at least 10
rows with auto-renew on; the group shares plan length, list price and payment method, the
listening-days band and the days-until-expiry band. A change is kept only if it lowers the
calibrated score by at least the registered minimum.

Only features are read (and the pre-T labels columns that choose the rows); the frozen
booster is loaded with its SHA-256 and never refit.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retention.models.metrics import logit

LEVER = "last_is_auto_renew"
HORIZON = 30
SUPPORT_MIN = 10
TRAINING_SNAPSHOTS = [date(2016, 8, 1), date(2016, 9, 1)]
GROUP_COLUMNS = ["last_plan_days", "last_list_price", "last_payment_method_id"]
LISTEN_BANDS = [(0, 0, "0"), (1, 7, "1-7"), (8, 20, "8-20"), (21, 28, "21-28")]
DUE_BANDS = [(0, 6, "0-6"), (7, 13, "7-13"), (14, 20, "14-20"), (21, 29, "21-29")]
STATUSES = ["counterfactual", "below_threshold", "no_support", "already_on"]
PHRASES = Path("configs/counterfactual_phrases.yaml")
CONFIG = Path("configs/uplift.yaml")


# --- support ---------------------------------------------------------------------------


def _band(values: pd.Series, bands: list[tuple[int, int, str]]) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series("other", index=values.index, dtype=object)
    for lo, hi, name in bands:
        out[(v >= lo) & (v <= hi)] = name
    return out


def comparable_key(rows: pd.DataFrame) -> pd.Series:
    """The comparable group of each row, as one string."""
    parts = [rows[c].astype(str) for c in GROUP_COLUMNS]
    parts.append(_band(rows.active_days_28, LISTEN_BANDS))
    parts.append(_band(rows.days_until_expiry, DUE_BANDS))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def support_table(training_rows: pd.DataFrame) -> set[str]:
    """Comparable groups with at least SUPPORT_MIN auto-renew-on rows in the training rows."""
    on = training_rows[pd.to_numeric(training_rows[LEVER]) == 1]
    counts = comparable_key(on).value_counts()
    return set(counts.index[counts >= SUPPORT_MIN])


def supported(rows: pd.DataFrame, table: set[str]) -> np.ndarray:
    return comparable_key(rows).isin(table).to_numpy()


# --- the search ------------------------------------------------------------------------


def calibrated(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt: 1 / (1 + exp(-(a + b * logit(p))))."""
    z, _ = logit(np.asarray(p, dtype=float))
    return 1 / (1 + np.exp(-(a + b * z)))


def apply_lever(rows: pd.DataFrame, eligible: np.ndarray) -> pd.DataFrame:
    """A copy with auto-renew set from off to on for eligible rows; nothing else changes."""
    changed = rows.copy()
    off = pd.to_numeric(rows[LEVER]).to_numpy() == 0
    flip = eligible & off
    changed.loc[flip, LEVER] = 1
    return changed


def changed_columns(original: pd.DataFrame, changed: pd.DataFrame) -> set[str]:
    """Columns whose value differs in any row (nulls equal to nulls)."""
    out = set()
    for c in original.columns:
        a, b = original[c], changed[c]
        same = (a == b) | (a.isna() & b.isna())
        if not bool(same.all()):
            out.add(c)
    return out


@dataclass(frozen=True)
class Searched:
    old: np.ndarray
    new: np.ndarray  # nan where no change was scored
    status: np.ndarray


def search(
    rows: pd.DataFrame,
    score: Callable[[pd.DataFrame], np.ndarray],
    support: np.ndarray,
    a: float,
    b: float,
    min_reduction: float,
) -> Searched:
    """Score each row as it is and, where the lever applies, with auto-renew on."""
    off = pd.to_numeric(rows[LEVER]).to_numpy() == 0
    old = calibrated(score(rows), a, b)
    new = np.full(len(rows), np.nan)
    eligible = off & support
    if eligible.any():
        changed = apply_lever(rows, eligible)
        new_all = calibrated(score(changed), a, b)
        new[eligible] = new_all[eligible]
    status = np.select(
        [~off, ~support, (old - new) >= min_reduction],
        ["already_on", "no_support", "counterfactual"],
        "below_threshold",
    )
    return Searched(old, new, status)


def coverage(snapshots: pd.Series, status: np.ndarray) -> dict[str, dict[str, int]]:
    """Rows per status (the reason a row has no counterfactual) by snapshot."""
    df = pd.DataFrame({"snapshot": pd.to_datetime(snapshots).dt.date.astype(str), "status": status})
    out = {}
    for snap, g in df.groupby("snapshot"):
        out[snap] = {s: int((g.status == s).sum()) for s in STATUSES}
    return out


# --- wording ---------------------------------------------------------------------------

CAUSAL_WORDS = r"\b(causes?|caused|because|due to|leads? to|drives?|results? in)\b"
ADVICE_WORDS = r"\b(should|turn on|switch|increase|improve|reduce|try|consider)\b"
LEVER_VERBS = r"\b(enable|offer|give|ask|encourage|prompt|upgrade|move to|set up)\b"
CLAIM_WORDS = r"\b(help|helps|keep|keeps|retain|save|prevent|lower churn)\b"
NEVER_NAMED = r"\b(age|gender|registered|registration|channel|tenure|city)\b"
FACT_FIELD = re.compile(r"\{[a-z_]+\}")


def load_phrases(path: Path = PHRASES) -> dict:
    return yaml.safe_load(path.read_text())


def phrase_lines(phrases: dict) -> dict[str, str]:
    return {
        "counterfactual_line": phrases["counterfactual_line"],
        "caveat_line": phrases["caveat_line"],
        **{f"no_counterfactual.{k}": v for k, v in phrases["no_counterfactual"].items()},
    }


def phrase_problems(phrases: dict) -> list[str]:
    """Every rule the registered wording breaks (an empty list means none)."""
    problems = []
    for name, line in phrase_lines(phrases).items():
        for pattern, what in (
            (CAUSAL_WORDS, "causal"),
            (ADVICE_WORDS, "advice"),
            (LEVER_VERBS, "lever verb"),
            (CLAIM_WORDS, "action claim"),
            (NEVER_NAMED, "never-named"),
        ):
            if re.search(pattern, line, re.IGNORECASE):
                problems.append(f"{name} has a {what} word")
        if re.search(r"\d", FACT_FIELD.sub("", line)):
            problems.append(f"{name} has a digit outside a fact field")
    return problems


def render(status: str, old: float, new: float, phrases: dict, min_reduction: float) -> list[str]:
    """The lines shown for one row."""
    fmt = phrases["score_format"]
    if status == "counterfactual":
        line = phrases["counterfactual_line"].format(
            horizon=HORIZON, new=fmt.format(new), old=fmt.format(old)
        )
        return [line, phrases["caveat_line"]]
    if status == "no_support":
        return [phrases["no_counterfactual"]["no_support"]]
    if status == "below_threshold":
        points = phrases["min_points_format"].format(min_reduction * 100)
        return [
            phrases["no_counterfactual"]["below_threshold"].format(
                horizon=HORIZON, min_points=points
            )
        ]
    return []  # auto-renew already on: the lever does not apply and nothing is shown


def registered_min_reduction(path: Path = CONFIG) -> float:
    return float(yaml.safe_load(path.read_text())["counterfactuals"]["min_reduction"])


# --- the run ---------------------------------------------------------------------------


class LeverBreach(RuntimeError):
    """A change touched a feature outside the lever or switched auto-renew off."""


def guard_change(original: pd.DataFrame, changed: pd.DataFrame) -> None:
    """The Done-when rule, checked on every snapshot: only the lever changes, and only up."""
    outside = changed_columns(original, changed) - {LEVER}
    if outside:
        raise LeverBreach(f"features outside the lever changed: {sorted(outside)}")
    before = pd.to_numeric(original[LEVER]).to_numpy()
    after = pd.to_numeric(changed[LEVER]).to_numpy()
    if np.any((before == 1) & (after != 1)) or np.any(after < before):
        raise LeverBreach("auto-renew was switched off")


def run(
    processed: Path,
    model_dir: Path,
    protocol,
    settings,
    snapshots: list[date],
    min_reduction: float,
    phrases: dict,
) -> tuple[dict, pd.DataFrame]:
    """Counterfactuals for every explained row; returns the log and the per-row table."""
    from retention.explain.contributions import EXPECTED_SHA256, load_saved_model
    from retention.explain.model import p2_main_rows

    problems = phrase_problems(phrases)
    if problems:
        raise ValueError(f"the registered wording breaks its rules: {problems}")
    saved = load_saved_model(model_dir, HORIZON, settings, protocol, EXPECTED_SHA256[HORIZON])
    training = pd.concat(
        [p2_main_rows(processed, t, settings.columns, protocol) for t in TRAINING_SNAPSHOTS],
        ignore_index=True,
    )
    table = support_table(training)

    def score(rows: pd.DataFrame) -> np.ndarray:
        return saved.model.predict(rows)[0]

    parts, cov, support_counts = [], {}, {}
    for t in snapshots:
        rows = p2_main_rows(processed, t, settings.columns, protocol)
        sup = supported(rows, table)
        off = pd.to_numeric(rows[LEVER]).to_numpy() == 0
        guard_change(rows, apply_lever(rows, sup & off))
        res = search(rows, score, sup, saved.platt_a, saved.platt_b, min_reduction)
        support_counts[t.isoformat()] = {
            "off_rows": int(off.sum()),
            "off_rows_with_training_support": int((off & sup).sum()),
        }
        cov |= coverage(rows.snapshot_date, res.status)
        parts.append(
            pd.DataFrame(
                {
                    "msno": rows.msno.to_numpy(),
                    "snapshot_date": rows.snapshot_date.to_numpy(),
                    "score": res.old,
                    "score_with_auto_renew_on": res.new,
                    "status": res.status,
                }
            )
        )
    log = {
        "meaning": phrases["caveat_line"],
        "booster_sha256": saved.sha256,
        "training_rows": len(training),
        "supported_groups": len(table),
        "support_by_snapshot": support_counts,
        "coverage_by_snapshot": cov,
        "min_reduction": min_reduction,
    }
    return log, pd.concat(parts, ignore_index=True)
