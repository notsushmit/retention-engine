"""score-parity: every registered step 4 mark, judged exactly, on 2016-10-01 and 2016-11-01.

Models come through the registry alias (retention-scoring@serving), as the service
will load them. Rows come from the batch inputs (retention.api.batch); every value
comes from the step 4 scoring functions. Marks (configs/serving.yaml, step4 and
contract): reload_classifier, reload_dt01 (13-field summaries on 2016-11-01),
paid_days_parity, reasons under amendment A3 (parity on the rows where Phase 4
listed no tenure, the self-check on the rows where it did, zero tenure),
final_probability_tie, action_parity under A4's field name, population_counts,
unseen_levels. Two rules stop the run: population counts that differ from the
registered ones, and a customer without a usable price who has an allowed action
(A4). Reported beside the marks, deciding nothing: the before-fix paths count and
A4's recount of paying blocks next to the registered figures. The original
reasons_parity mark is reported as "amended before any run; not judged as
written". No tolerance is added anywhere; a difference is not met.

The self-check shows only that the masking matches Phase 4's own missing-tenure
behaviour; it does not validate the wording.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retention.api import scoring as sc
from retention.api.batch import SnapshotInputs
from retention.api.config import file_sha256
from retention.explain.build_reasons import read_reasons
from retention.explain.model import compare_summary, logged_summary
from retention.models.classifier import score_summary
from retention.optimize.config import ActionsConfig
from retention.optimize.effects import build_options
from retention.optimize.strategies import best_action

SLOTS = ("raising_1", "raising_2", "raising_3", "lowering_1", "lowering_2")
ACTION_FIELDS = ("expected_retained_value", "net_value", "cost")
SNAPSHOTS = (date(2016, 10, 1), date(2016, 11, 1))


class StopRun(RuntimeError):
    """A registered stop rule fired; the run ends and reports it."""


# --- exact comparison -----------------------------------------------------------------------------


def _missing(v: Any) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _plain(v: Any) -> Any:
    """A value for comparison: every missing marker is one missing value; dates as dates."""
    if _missing(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, np.generic):
        return v.item()
    return v


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, key: str = "msno") -> dict[str, Any]:
    """Row-by-row, column-by-column exact comparison on the same keys.

    Floats must be equal as floats (a missing value equals only a missing value);
    every other value must be equal after the missing markers (None, NaN, NA) are
    read as one. Returns the differing rows per column and the largest float gap.
    """
    if list(a.columns) != list(b.columns):
        return {"equal": False, "columns": "the column lists differ"}
    if sorted(a[key]) != sorted(b[key]) or a[key].duplicated().any():
        return {"equal": False, "rows": "the key sets differ or repeat"}
    x = a.sort_values(key, kind="stable").reset_index(drop=True)
    y = b.sort_values(key, kind="stable").reset_index(drop=True)
    differ: dict[str, int] = {}
    largest = 0.0
    for c in x.columns:
        u, v = x[c].to_numpy(dtype=object), y[c].to_numpy(dtype=object)
        bad = 0
        for p, q in zip(u, v, strict=True):
            p, q = _plain(p), _plain(q)
            if isinstance(p, float) and isinstance(q, float):
                if p != q:
                    bad += 1
                    largest = max(largest, abs(p - q))
            elif p != q:
                bad += 1
        if bad:
            differ[c] = bad
    return {"equal": not differ, "rows": len(x), "differing_by_column": differ, "largest": largest}


# --- the marks ------------------------------------------------------------------------------------


def judge_summary(values: np.ndarray, logged: dict[str, Any]) -> dict[str, Any]:
    """A 13-field raw-score summary against the logged one, exactly (Phase 4's comparison)."""
    fields = compare_summary(logged, score_summary(values))
    return {"met": all(f["equal"] for f in fields), "fields": fields}


def judge_population(found: dict[str, int], expected: dict[str, int]) -> dict[str, Any]:
    """Registered counts; any difference stops the run."""
    if found != expected:
        raise StopRun(f"population counts differ: found {found}, registered {expected}")
    return {"met": True, "found": found, "registered": expected}


def no_price_rule(acts: sc.Actions) -> dict[str, Any]:
    """A4: a customer without a usable price may have no allowed action; else stop."""
    bad = (~acts.has_price) & acts.opts.allowed.any(axis=1)
    if bad.any():
        raise StopRun(f"{int(bad.sum())} customers without a usable price have an allowed action")
    return {"met": True, "customers_without_price": int((~acts.has_price).sum())}


def judge_fix(pr: sc.Probabilities, mark: dict[str, Any], tolerance: float) -> dict[str, Any]:
    s = pr.summary
    violations = int((np.diff(pr.final, axis=1) < 0).any(axis=1).sum())
    found = {
        "rows": int(pr.final.shape[0]),
        "rows_changed": int(s["rows_changed"]),
        "max_change": float(s["max_change"]),
        "order_violations_after_fix": violations,
    }
    met = (
        found["rows"] == mark["rows"]
        and found["rows_changed"] == mark["rows_changed"]
        and abs(found["max_change"] - float(mark["max_change"])) <= tolerance
        and violations == mark["order_violations_after_fix"]
    )
    return {"met": met, "found": found, "registered": mark}


def tenure_listed(reasons: pd.DataFrame) -> np.ndarray:
    """Rows where any reason slot holds the tenure group."""
    return (reasons[list(SLOTS)] == "tenure").any(axis=1).to_numpy()


def zero_tenure(served: pd.DataFrame) -> dict[str, Any]:
    slots = int(tenure_listed(served).sum())
    texts = int(served.text.str.lower().str.contains("member for", regex=False).sum())
    return {"met": slots == 0 and texts == 0, "tenure_slots": slots, "member_for_texts": texts}


def before_fix_paths(pr: sc.Probabilities, served: dict[int, pd.DataFrame]) -> dict[str, Any]:
    """Descriptive: rows where the two registered paths give different probabilities."""
    out = {}
    for k, h in enumerate((7, 30)):
        a = pr.before_fix[:, k]
        b = served[h].probability.to_numpy(dtype=float)
        diff = a != b
        out[str(h)] = {
            "rows": int(len(a)),
            "rows_differing": int(diff.sum()),
            "largest_difference": float(np.max(np.abs(a - b))) if len(a) else 0.0,
        }
    return out


def action_reference(value_rows: pd.DataFrame, cfg: ActionsConfig) -> pd.DataFrame:
    """Phase 7's best action, recomputed from value.parquet's rows (the parity reference)."""
    opts = build_options(value_rows, cfg, cfg.base)
    return sc.Actions(opts, best_action(opts), value_rows.has_price.to_numpy(dtype=bool)).frame()


def judge_actions(served: pd.DataFrame, reference: pd.DataFrame, field: str) -> dict[str, Any]:
    cols = ["msno", field, *ACTION_FIELDS]
    return compare_frames(served[cols], reference[cols])


def recount_paying(acts: sc.Actions, registered: dict[str, int]) -> dict[str, Any]:
    """A4's recount of blocks that pay, next to the registered docs figures (decides nothing)."""
    f = acts.frame()
    return {
        "blocks": int(len(f)),
        "blocks_that_pay": int(f.pays.sum()),
        "registered": registered,
        "same": int(len(f)) == registered["blocks"]
        and int(f.pays.sum()) == registered["blocks_that_pay"],
    }


def check_pinned(path: Path, sha: str) -> None:
    found = file_sha256(path)
    if found != sha:
        raise StopRun(f"{path.name} SHA-256 {found} differs from the pinned {sha}")


# --- one snapshot ---------------------------------------------------------------------------------


@dataclass
class Context:
    models: Any  # registry.ServingModels
    explain_cfg: Any
    renderer: Any
    phrases_sha256: str
    columns: list[str]
    actions_cfg: ActionsConfig
    protocol: Any
    reasons: pd.DataFrame  # reasons.parquet, read once
    value: pd.DataFrame  # value.parquet, read once
    rehearsal: dict[str, Any]
    serving: dict[str, Any]
    population_expected: dict[str, int]
    population_snapshot: date


def snapshot_marks(ctx: Context, inputs: SnapshotInputs, dry: bool = False) -> dict[str, Any]:
    """Every per-snapshot mark; raises StopRun when a stop rule fires.

    dry: a subset of customers (the dry run). Population counts are recorded, not
    judged; the no-price rule is reported, not stopping; the reference files are
    narrowed to the subset's customers (in a full run they are not, so a missing
    row is a difference).
    """
    t = inputs.snapshot
    out: dict[str, Any] = {"snapshot": t.isoformat(), "rows": len(inputs.rows)}
    c01, dt01 = ctx.models.c01, ctx.models.dt01
    status = sc.population_status(inputs.status_rows, tuple(ctx.protocol.main_segments))
    out["population"] = sc.population_counts(status)
    if t == ctx.population_snapshot and not dry:  # the stop rule fires before any long step
        judge_population(out["population"], ctx.population_expected)
    subset = set(inputs.rows.msno) if dry else None
    _, keep = sc.refuse_unseen(status, inputs.rows, [m.model for m in c01.values()])
    out["unseen_level_rows"] = int((~keep).sum())
    rows = inputs.rows
    pr = sc.probabilities(c01, rows, ctx.protocol.seed)
    out["probabilities"] = {k: v for k, v in pr.summary.items()}
    # reasons: the service route on every row, the self-check on Phase 4's tenure rows
    served, checks = {}, {}
    for h in (7, 30):
        part = sc.contribution_part(c01[h], rows, ctx.explain_cfg, ctx.columns, h)
        args = (h, t, (c01[h].platt_a, c01[h].platt_b), ctx.protocol, ctx.phrases_sha256)
        s = sc.service_reasons(ctx.explain_cfg, ctx.renderer, part, inputs.facts, *args)
        ref = ctx.reasons[(ctx.reasons.snapshot_date == t) & (ctx.reasons.horizon == h)]
        if subset is not None:
            ref = ref[ref.msno.isin(subset)]
        ref = ref.reset_index(drop=True)
        listed = set(ref.msno[tenure_listed(ref)])
        mask = s.msno.isin(listed).to_numpy()
        parity = compare_frames(
            s[~mask].reset_index(drop=True), ref[~ref.msno.isin(listed)].reset_index(drop=True)
        )
        sub = part[part.msno.isin(listed)].reset_index(drop=True)
        self_ = sc.self_check_reasons(
            ctx.explain_cfg, ctx.renderer, sub, inputs.facts[inputs.facts.msno.isin(listed)], *args
        )
        check = compare_frames(s[mask].reset_index(drop=True), self_)
        checks[str(h)] = {
            "parity_rows": int((~mask).sum()),
            "parity": parity,
            "self_check_rows": int(mask.sum()),
            "self_check": check,
            "zero_tenure": zero_tenure(s),
        }
        served[h] = s
    out["reasons"] = checks
    out["before_fix_paths"] = before_fix_paths(pr, served)
    # paid days and actions
    surv = sc.survival(dt01, rows)
    table = sc.customers(rows, surv, served[30], inputs.prices)
    vals = ctx.value[ctx.value.snapshot_date == t]
    if subset is not None:
        vals = vals[vals.msno.isin(subset)]
    vals = vals.reset_index(drop=True)
    paid = compare_frames(
        table[["msno", "days_30_90"]].reset_index(drop=True), vals[["msno", "days_30_90"]]
    )
    out["paid_days"] = paid
    acts = sc.actions(table, ctx.actions_cfg)
    try:
        out["no_price_rule"] = no_price_rule(acts)
    except StopRun as e:
        if not dry:
            raise
        out["no_price_rule"] = {"met": False, "found": str(e)}
    field = ctx.serving["contract"]["action_reading"]["field"]
    out["actions"] = judge_actions(acts.frame(), action_reference(vals, ctx.actions_cfg), field)
    docs = ctx.serving["contract"]["action_reading"]["docs_figures"][t.isoformat()]
    out["paying_blocks"] = recount_paying(acts, docs)
    out["_probabilities"] = pr  # kept for the 2016-11-01 tie, not written
    out["_rows"] = rows
    return out


def reload_marks(ctx: Context, rows: pd.DataFrame) -> dict[str, Any]:
    """reload_classifier and reload_dt01 on 2016-11-01's rows, against rehearsal.json."""
    marks = ctx.serving["step4"]["pass_marks"]
    out = {}
    for name, model_of in (
        ("reload_classifier", lambda h, r: ctx.models.c01[h].model.predict(r)[0]),
        ("reload_dt01", lambda h, r: ctx.models.dt01.lapse_probability(r, [h])[0][:, 0]),
    ):
        cells = {}
        for cell in marks[name]["cells"]:
            path = tuple(cell["path"])
            logged = logged_summary(ctx.rehearsal, path[:-1])
            cells[str(cell["horizon"])] = judge_summary(model_of(cell["horizon"], rows), logged)
        out[name] = {"met": all(c["met"] for c in cells.values()), "cells": cells}
    return out


def overall(report: dict[str, Any]) -> list[str]:
    """Every mark that is not met, by name (empty when all are met)."""
    failed = []
    for name, mark in report["marks"].items():
        if not mark.get("met", False):
            failed.append(name)
    return failed


def json_ready(obj: Any) -> Any:
    return json.loads(
        json.dumps(obj, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o))
    )


def read_inputs(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """reasons.parquet (sorted as Phase 4 reads it) and value.parquet, with dates as dates."""
    reasons = read_reasons(processed / "explain" / "reasons.parquet")
    reasons["snapshot_date"] = pd.to_datetime(reasons.snapshot_date).dt.date
    value = pd.read_parquet(processed / "optimize" / "value.parquet")
    value["snapshot_date"] = pd.to_datetime(value.snapshot_date).dt.date
    return reasons, value


def assemble_marks(
    results: dict[date, dict[str, Any]],
    reload: dict[str, Any],
    serving: dict[str, Any],
) -> dict[str, Any]:
    """The registered marks from the per-snapshot results, each with met true or false."""
    step4 = serving["step4"]
    more = step4["more_pass_marks"]
    a3 = step4["amendments"][0]
    tol = float(serving["ordering"]["pass_marks"]["fix"]["max_change_tolerance"])
    nov = date.fromisoformat(more["final_probability_tie"]["snapshot"])
    marks: dict[str, Any] = dict(reload)
    pr = results[nov]["_probabilities"]
    marks["final_probability_tie"] = judge_fix(pr, more["final_probability_tie"], tol)
    pop = more["population_counts"]
    marks["population_counts"] = {
        "met": results[nov]["population"] == pop["expected"],
        "found": results[nov]["population"],
        "registered": pop["expected"],
    }
    marks["unseen_levels"] = {
        "met": results[nov]["unseen_level_rows"] == more["unseen_levels"]["expected"],
        "found": results[nov]["unseen_level_rows"],
    }
    for t, res in results.items():
        key = t.isoformat()
        for h in ("7", "30"):
            r = res["reasons"][h]
            listed = a3["counts"][key][h]
            rows = res["rows"]
            marks[f"reasons_parity_without_tenure {key} {h}"] = {
                "met": r["parity"]["equal"] and r["parity_rows"] == rows - listed,
                "rows": r["parity_rows"],
                "registered_rows": rows - listed,
                "comparison": r["parity"],
            }
            marks[f"self_check_with_tenure {key} {h}"] = {
                "met": r["self_check"]["equal"] and r["self_check_rows"] == listed,
                "rows": r["self_check_rows"],
                "registered_rows": listed,
                "comparison": r["self_check"],
            }
            marks[f"zero_tenure {key} {h}"] = r["zero_tenure"]
        marks[f"paid_days_parity {key}"] = {"met": res["paid_days"]["equal"], **res["paid_days"]}
        marks[f"action_parity {key}"] = {"met": res["actions"]["equal"], **res["actions"]}
        marks[f"no_price_rule {key}"] = res["no_price_rule"]
    return marks


def descriptive(results: dict[date, dict[str, Any]]) -> dict[str, Any]:
    return {
        t.isoformat(): {
            "before_fix_paths": res["before_fix_paths"],
            "paying_blocks": res["paying_blocks"],
        }
        for t, res in results.items()
    }


# --- the dry run --------------------------------------------------------------------------


DRY_SNAPSHOT = date(2016, 11, 1)
DRY_BUCKET = 0


def dry_subset(inputs: SnapshotInputs, keep_bucket: int = DRY_BUCKET) -> SnapshotInputs:
    """The inputs of one md5 bucket's customers, every table in its own order."""
    from retention.data.sample import bucket

    def only(df: pd.DataFrame) -> pd.DataFrame:
        mask = np.array([bucket(m) == keep_bucket for m in df.msno.astype(str)], dtype=bool)
        return df[mask].reset_index(drop=True)

    return SnapshotInputs(
        inputs.snapshot,
        only(inputs.status_rows),
        only(inputs.rows),
        only(inputs.facts),
        only(inputs.prices),
    )


def dry_summary(res: dict[str, Any]) -> dict[str, Any]:
    """The dry run's comparisons only: no registered full-count mark is judged."""
    out: dict[str, Any] = {
        "snapshot": res["snapshot"],
        "rows": res["rows"],
        "population_recorded_not_judged": res["population"],
        "unseen_level_rows": res["unseen_level_rows"],
        "fix_on_subset": {
            k: res["probabilities"][k] for k in ("rows_changed", "max_change", "clipped")
        },
        "reasons": {
            h: {
                "parity_rows": r["parity_rows"],
                "parity_equal": r["parity"]["equal"],
                "parity_differing": r["parity"].get("differing_by_column"),
                "self_check_rows": r["self_check_rows"],
                "self_check_equal": r["self_check"]["equal"],
                "self_check_differing": r["self_check"].get("differing_by_column"),
                "zero_tenure": r["zero_tenure"],
            }
            for h, r in res["reasons"].items()
        },
        "paid_days_equal": res["paid_days"]["equal"],
        "paid_days_differing": res["paid_days"].get("differing_by_column"),
        "actions_equal": res["actions"]["equal"],
        "actions_differing": res["actions"].get("differing_by_column"),
        "no_price_rule": res["no_price_rule"],
        "before_fix_paths": res["before_fix_paths"],
        "paying_blocks_on_subset": {
            k: res["paying_blocks"][k] for k in ("blocks", "blocks_that_pay")
        },
    }
    out["all_comparisons_equal"] = bool(
        all(
            r["parity_equal"] and r["self_check_equal"] and r["zero_tenure"]["met"]
            for r in out["reasons"].values()
        )
        and out["paid_days_equal"]
        and out["actions_equal"]
        and out["no_price_rule"]["met"]
    )
    return out
