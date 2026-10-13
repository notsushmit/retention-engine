"""The test stage: fit the frozen models once, score every registered cell, log once.

Two entry points share this code:
- the real run: the test guard opens the session (all checks, then the test
  run and the one TestUnlock); months come from protocol.test_months(unlock);
  outcomes are read through the outcome lock with the unlock; every result is
  written to that test run at the end, in one pass; no metric is printed.
- the rehearsal: no guard, no unlock, no MLflow run. The same pipeline runs on
  2016 stand-in months (2017-01-01 -> 2016-11-01, 2017-02-01 -> 2016-12-01;
  the 59-day demonstration on 2016-11-01, whose 59-day outcomes are
  observable) and writes a JSON file; the console shows cell statuses only.

All fits use the frozen choices (configs/choices.yaml): fit rows 2016-08/09 and
Platt calibrators on the calibration half of 2016-10, labels as of 2017-01-01.
Discrimination verdicts use raw scores; calibration criteria use the final
probabilities (calibrate, clip, ordering fix, clip). Every metric goes through
the "not computable" path, so thin cells are labelled, never fatal.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from retention.features.spec import FEATURE_COLUMNS, PRIMARY_FEATURES
from retention.models.baselines import Baselines
from retention.models.bootstrap import (
    FoldScores,
    bootstrap,
    combine_months,
    ece_interpretable,
    paired_auc_difference,
    verdicts,
)
from retention.models.calibration import Platt, calibrate
from retention.models.classifier import Classifier, candidate_params, score_summary
from retention.models.data import load_labels
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.frames import (
    fit_rows,
    half_mask,
    horizon_target,
    scored_month_rows,
    survival_fit_rows,
)
from retention.models.metrics import (
    NOT_COMPUTABLE,
    CalibrationStatistic,
    SortedProbabilities,
    SortedScores,
    average_precision,
    brier,
    cell,
    expiry_band,
    harrell_c,
    ibs,
    kaplan_meier,
    kaplan_meier_by_band,
    log_loss,
)
from retention.models.protocol import EVAL_CONFIG, Protocol, TestUnlock
from retention.models.report import (
    DESCRIPTIVE_BREAKDOWNS,
    Cell,
    breakdown_group,
    cell_rows,
    summarize,
)
from retention.models.resources import peak_rss_gb

CHOICES = Path("configs/choices.yaml")
HORIZONS = (7, 30, 59)
REHEARSAL_MONTHS = {
    7: (date(2016, 11, 1), date(2016, 12, 1)),
    30: (date(2016, 11, 1),),
    59: (date(2016, 11, 1),),
}
REHEARSAL_LIMITS = [
    "Calibration on 2016-11 and 2016-12 says nothing about calibration in 2017.",
    "Month-level facts differ: never-seen share, segment mix, event counts and the stop day "
    "(2016 months are followed for 90 days; 2017-01 for 59 and 2017-02 for 28).",
    "The 59-day demonstration is rehearsed on 2016-11, whose 59-day outcomes are observable.",
    "Whether 2017's elevated 7-day rate behaves like 2016-11's is only shown by the real run.",
]
NEVER_SEEN_REFERENCE = (date(2016, 8, 1), date(2016, 9, 1), date(2016, 10, 1))
# Peak RSS ceiling of the main process (post-registration amendment; the
# registered 6 GB limit in configs/evaluation.yaml is recorded as exceeded).
MEMORY_CEILING_GB = 10.0


class MemoryCeiling(RuntimeError):
    """The main process went above MEMORY_CEILING_GB."""


def print_progress(message: str) -> None:
    """Progress goes to stdout: counts, step names, seconds and memory, never a metric or status."""
    print(f"progress: {message}", flush=True)


def planned_cells(months: dict[int, tuple[date, ...]]) -> int:
    """How many cells the stage records, from the months alone.

    Per primary (7- or 30-day) month: the cell, never-seen, without
    100%-discount-only, by segment, last row, sensitivities and P1 (7); per
    59-day month: the cell and never-seen (2); one pooled 7-day summary; one
    survival-metrics cell per scored month; one shape check per 30-day month.
    """
    primary = len(months[7]) + len(months[30])
    every = {m for ms in months.values() for m in ms}
    return 7 * primary + 2 * len(months[59]) + 1 + len(every) + len(months[30])


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


class Checkpoint:
    """Rehearsal only: completed cells on disk, so a stopped rehearsal resumes.

    Its contents are never printed or logged; the caller deletes it after the
    single final write. Saved cells are reused only when the fingerprint (every
    scored input and the bootstrap settings) matches; otherwise it starts empty.
    Every cell passes through the same JSON round trip, resumed or not.
    """

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path, self.fingerprint, self.cells = path, fingerprint, {}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") == fingerprint:
                self.cells = saved["cells"]

    def save(self, key: str, data: Any) -> Any:
        data = json.loads(json.dumps(data, default=_json_default))
        self.cells[key] = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "cells": self.cells}))
        tmp.replace(self.path)
        return data


def scored_fingerprint(scored: dict, replicates: int, seed: int, buckets: int | None) -> str:
    """SHA-256 of each scored month's outcomes, customers and scores, and the bootstrap settings."""
    h = hashlib.sha256(json.dumps([replicates, seed, buckets]).encode())

    def add(obj):
        if isinstance(obj, dict):
            for k in sorted(obj, key=str):
                h.update(str(k).encode())
                add(obj[k])
        elif isinstance(obj, np.ndarray):
            h.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, (int, float, str)):
            h.update(repr(obj).encode())

    for key in sorted(scored, key=str):
        entry = scored[key]
        rows = entry["rows"]
        h.update(str(key).encode())
        h.update(pd.util.hash_pandas_object(rows.msno, index=False).to_numpy().tobytes())
        for c in ("days_to_event", "event_observed"):
            h.update(rows[c].to_numpy().tobytes())
        add(entry["raw"])
        add(entry["final"])
    return h.hexdigest()


def resolve_months(protocol: Protocol, unlock: TestUnlock | None, rehearsal: bool):
    """Scored months per horizon. The rehearsal never calls protocol.test_months."""
    if rehearsal:
        return {h: tuple(m) for h, m in REHEARSAL_MONTHS.items()}
    return dict(protocol.test_months(unlock).months)


def _interval(iv) -> dict[str, Any]:
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi, "status": iv.status, "reason": iv.reason}


def _boot(customers, stat, replicates, seed) -> dict[str, Any]:
    return _interval(bootstrap([customers], lambda f, w: stat(w), replicates, seed))


def auc_stat(y: np.ndarray, score: np.ndarray):
    """A bootstrap statistic: AUC of `score` under customer weights (scores sorted once)."""
    s = SortedScores(score)
    return lambda w: s.auc(y, w)


def calibration_stat(y: np.ndarray, p: np.ndarray, key: str):
    """A bootstrap statistic: calibration slope or intercept under customer weights.

    Only the regression it reports; each replicate starts from the full-sample estimate.
    """
    return CalibrationStatistic(y, p, key, warm=True)


def ece_stat(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """A bootstrap statistic: ECE under customer weights (probabilities sorted once)."""
    s = SortedProbabilities(y, p, bins)
    return lambda w: s.ece(w)


def auc_value(y: np.ndarray, score: np.ndarray) -> float:
    return SortedScores(score).auc(y)


def stop_day(rows: pd.DataFrame) -> int:
    """The single stop day C of a scored month (every censored row stops on it)."""
    censored = rows.days_to_event[rows.event_observed == 0].unique()
    if len(censored) != 1:
        raise ValueError(f"expected one stop day, found {sorted(censored)}")
    return int(censored[0])


# --- one cell ------------------------------------------------------------------------


def model_metrics(y, raw, final, customers, replicates, seed) -> dict[str, Any]:
    """Every per-model metric of one cell; each is a value or a labelled 'not computable'."""
    out: dict[str, Any] = {
        "auc_raw": _boot(customers, auc_stat(y, raw), replicates, seed),
        "auc_final": cell(lambda: SortedScores(final).auc(y)).__dict__
        if final is not None
        else None,
        "pr_auc": cell(average_precision, y, raw).__dict__,
        "distinct_raw": int(np.unique(raw).size),
        "score_summary_raw": score_summary(raw),
    }
    if final is not None:
        out |= {
            "log_loss": log_loss(y, final),
            "brier": brier(y, final),
            "distinct_final": int(np.unique(final).size),
            "slope": _boot(customers, calibration_stat(y, final, "slope"), replicates, seed),
            "intercept": _boot(
                customers, calibration_stat(y, final, "intercept"), replicates, seed
            ),
        }
        e = _boot(customers, ece_stat(y, final, 10), replicates, seed)
        rate = float(y.mean()) if len(y) else float("nan")
        e["share_of_rate"] = e["point"] / rate if rate else float("nan")
        e["interpretable"] = bool(
            e["status"] == "ok" and ece_interpretable(e["point"], e["lo"], e["hi"])
        )
        out["ece"] = e
    return out


def paired(customers, y, a, b, replicates, seed) -> dict[str, Any]:
    iv = paired_auc_difference([FoldScores(customers, y, a, b)], replicates, seed)
    out = _interval(iv)
    out["verdicts"] = {str(k): v for k, v in verdicts(iv.point, iv.lo, iv.hi).items()}
    return out


def calibration_verdict(slope: dict, intercept: dict, limits: dict) -> dict[str, Any]:
    if slope["status"] != "ok" or intercept["status"] != "ok":
        return {"status": NOT_COMPUTABLE, "pass": False}
    s_lo, s_hi = limits["slope"]
    i_lo, i_hi = limits["intercept"]
    ok = s_lo <= slope["point"] <= s_hi and i_lo <= intercept["point"] <= i_hi
    borderline = not (
        s_lo <= slope["lo"]
        and slope["hi"] <= s_hi
        and i_lo <= intercept["lo"]
        and intercept["hi"] <= i_hi
    )
    return {"status": "ok", "pass": bool(ok), "borderline": bool(borderline)}


def evaluate_cell(
    y: np.ndarray,
    customers: np.ndarray,
    raw: dict[str, np.ndarray],
    final: dict[str, np.ndarray | None],
    comparisons: list[tuple[str, str]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Per-model metrics and paired comparisons for one (horizon, month) cell."""
    out: dict[str, Any] = {"rows": int(len(y)), "lapses": int(y.sum())}
    if len(y) == 0:
        out["status"] = NOT_COMPUTABLE
        out["reason"] = "no rows"
        return out
    out["models"] = {
        m: model_metrics(y, raw[m], final.get(m), customers, replicates, seed) for m in raw
    }
    out["comparisons"] = {
        f"{a} minus {b}": paired(customers, y, raw[a], raw[b], replicates, seed)
        for a, b in comparisons
    }
    out["status"] = "ok"
    return out


# --- the stage ---------------------------------------------------------------------


def run_test_stage(
    protocol: Protocol,
    processed: Path,
    cache: Path,
    replicates: int,
    unlock: TestUnlock | None = None,
    rehearsal: bool = False,
    buckets: int | None = None,
    progress: Callable[[str], None] = print_progress,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Fit, score and evaluate every registered cell; results are returned, never written here.

    `checkpoint` (rehearsal only) keeps completed cells on disk; the real run
    writes no partial result anywhere.
    """
    if not rehearsal and not isinstance(unlock, TestUnlock):
        raise PermissionError("the real test stage needs the TestUnlock from the test session")
    if checkpoint is not None and not rehearsal:
        raise PermissionError("a checkpoint is allowed only in the rehearsal")
    t_start = time.time()

    def step(message: str) -> None:
        peak = peak_rss_gb()
        progress(f"{message}, {time.time() - t_start:.0f} s, peak {peak:.2f} GB")
        if peak > MEMORY_CEILING_GB:
            raise MemoryCeiling(f"peak RSS {peak:.2f} GB above the {MEMORY_CEILING_GB} GB ceiling")

    raw_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    choices = yaml.safe_load(CHOICES.read_text())["choices"]
    seed, threads = protocol.seed, protocol.threads
    cats = raw_cfg["classifier"]["categorical"]
    # The profile group is never a primary input, so the registered categorical
    # list does not cover it; in the profile ablation its code and text columns
    # are categoricals, like the registered code columns (post-registration detail).
    profile_cats = [*cats, "city", "gender", "age_bucket"]
    fixed = raw_cfg["classifier"]["fixed_params"]
    cands = raw_cfg["classifier"]["candidates"]
    dt_cfg = raw_cfg["survival"]["discrete_time"]
    edges = raw_cfg["baselines"]["banded_logistic"]["bands"]
    km_edges = raw_cfg["metrics"]["km_bands"]
    limits = raw_cfg["calibration"]
    floor_e, floor_n = raw_cfg["never_seen"]["min_events"], raw_cfg["never_seen"]["min_non_events"]

    drop = {fs: d["drop"] for fs, d in raw_cfg["selection"]["feature_sets"].items()}
    primary_cols = [
        c for c in PRIMARY_FEATURES if FEATURE_COLUMNS[c] not in drop[choices["feature_set"]]
    ]
    profile = [c for c, g in FEATURE_COLUMNS.items() if g == "profile"]
    variants = {
        "primary": primary_cols,
        "plus_counts_365": [c for c in PRIMARY_FEATURES],  # the full primary set (fs3)
        "minus_segment_state": [c for c in primary_cols if FEATURE_COLUMNS[c] != "segment_state"],
        "plus_profile": primary_cols + profile,
    }
    all_cols = list(dict.fromkeys(c for cols in variants.values() for c in cols))
    months = resolve_months(protocol, unlock, rehearsal)
    tf = protocol.test_fit
    snaps = (*tf.fit, tf.calibration.snapshot)
    oct_ = tf.calibration.snapshot

    def split(rows):
        fit = rows[rows.snapshot_date.dt.date.isin(tf.fit)]
        cal = rows[rows.snapshot_date.dt.date == oct_]
        return fit, cal[half_mask(cal.msno, "calibration", protocol)]

    # --- fit every model once ------------------------------------------------------
    t0 = time.time()
    clf_params = {
        h: candidate_params(cands[choices["classifier_hyperparameters"][h]]) for h in HORIZONS
    }
    dt_params = candidate_params(cands[choices["classifier_hyperparameters"][7]])
    fitted: dict[str, Any] = {"classifier": {}, "survival": {}, "banded": {}, "platt": {}}
    for pop in ("p2", "p1"):
        cal_rows_h = {}
        for h in HORIZONS:
            rows = fit_rows(protocol, processed, cache, tf.as_of, snaps, all_cols, h, pop, buckets)
            fit, cal = split(rows)
            step(f"fit rows loaded ({pop}, {h} days)")
            cal_rows_h[h] = cal
            for v, cols in variants.items():
                if pop == "p1" and v != "primary":
                    continue
                key = (pop, v, h)
                vc = profile_cats if v == "plus_profile" else cats
                m = Classifier(cols, vc, clf_params[h], fixed, threads, seed).fit(fit)
                fitted["classifier"][key] = m
                fitted["platt"][("classifier",) + key] = Platt(seed).fit(
                    m.predict(cal)[0], cal.y.to_numpy()
                )
            if pop == "p2":
                fitted["banded"][h] = Baselines(fit, edges, threads, seed)
            step(f"classifiers fitted ({pop}, {h} days)")
        srows = survival_fit_rows(
            protocol, processed, cache, tf.as_of, snaps, all_cols, pop, buckets
        )
        sfit, _ = split(srows)
        step(f"survival rows loaded ({pop})")
        specs = (
            {"dt01": ("dt01", "primary")}
            if pop == "p1"
            else {
                "dt01": ("dt01", "primary"),
                "dt02": ("dt02", "primary"),
                **{f"dt01_{v}": ("dt01", v) for v in variants if v != "primary"},
            }
        )
        for name, (cid, v) in specs.items():
            offset = dt_cfg["candidates"][cid]["expiry_offset_input"]
            vc = profile_cats if v == "plus_profile" else cats
            m = DiscreteTimeModel(
                offset, dt_cfg["interval_ends"], dt_params, fixed, vc, threads, seed
            )
            m.fit(sfit, variants[v])
            fitted["survival"][(pop, name)] = m
            step(f"survival model fitted ({pop}, {name})")
            p_cal = m.lapse_probability(cal_rows_h[7], list(HORIZONS))[0]
            for k, h in enumerate(HORIZONS):
                fitted["platt"][("survival", pop, name, h)] = Platt(seed).fit(
                    p_cal[:, k], cal_rows_h[h].y.to_numpy()
                )
        if pop == "p2":
            cal_p2 = dict(cal_rows_h)
            km_overall = kaplan_meier(sfit.time, sfit.event, list(range(0, 91)))
            band = expiry_band(sfit.days_until_expiry.to_numpy(dtype=float), km_edges)
            km_band = kaplan_meier_by_band(sfit.time, sfit.event, band, list(range(0, 91)))
    # Registered sensitivity: dt01 also fitted on the partly observed check month.
    sens_snaps = (*snaps, protocol.check_month)
    srows = survival_fit_rows(
        protocol, processed, cache, tf.as_of, sens_snaps, all_cols, "p2", buckets
    )
    sfit = srows[srows.snapshot_date.dt.date.isin((*tf.fit, protocol.check_month))]
    step("survival rows loaded (p2, with 2016-11)")
    m = DiscreteTimeModel(
        dt_cfg["candidates"]["dt01"]["expiry_offset_input"],
        dt_cfg["interval_ends"],
        dt_params,
        fixed,
        cats,
        threads,
        seed,
    ).fit(sfit, primary_cols)
    fitted["survival"][("p2", "dt01_plus_2016_11")] = m
    p_cal = m.lapse_probability(cal_p2[7], list(HORIZONS))[0]
    for k, h in enumerate(HORIZONS):
        fitted["platt"][("survival", "p2", "dt01_plus_2016_11", h)] = Platt(seed).fit(
            p_cal[:, k], cal_p2[h].y.to_numpy()
        )
    fit_seconds = round(time.time() - t0, 1)
    step("survival model fitted (p2, dt01_plus_2016_11)")

    # --- score every month --------------------------------------------------------
    seen = set(
        load_labels(
            processed / "labels.parquet", list(NEVER_SEEN_REFERENCE), ["segment"], protocol
        ).msno
    )
    all_months = sorted({m for ms in months.values() for m in ms})
    scored: dict[tuple[str, date], dict[str, Any]] = {}
    for month in all_months:
        for pop in ("p2_all", "p1"):
            rows = scored_month_rows(protocol, processed, month, all_cols, pop, unlock, buckets)
            entry: dict[str, Any] = {"rows": rows, "C": stop_day(rows)}
            mpop = "p2" if pop == "p2_all" else "p1"
            raw, final, unseen = {}, {}, {}
            for v in variants:
                if mpop == "p1" and v != "primary":
                    continue
                cols3 = []
                for h in HORIZONS:
                    p, u = fitted["classifier"][(mpop, v, h)].predict(rows)
                    cols3.append(p)
                    unseen[f"classifier_{v}_h{h}"] = u
                name = "classifier" if v == "primary" else f"classifier_{v}"
                rs = np.column_stack(cols3)
                cals = [fitted["platt"][("classifier", mpop, v, h)] for h in HORIZONS]
                c = calibrate(rs, cals)
                raw[name], final[name] = rs, c
            for (spop, name), m in fitted["survival"].items():
                if spop != mpop:
                    continue
                ends = [e for e in dt_cfg["interval_ends"] if e <= max(entry["C"], max(HORIZONS))]
                curve, u = m.lapse_probability(rows, ends)
                unseen[name] = u
                rs = curve[:, [ends.index(h) for h in HORIZONS]]
                cals = [fitted["platt"][("survival", mpop, name, h)] for h in HORIZONS]
                raw[name], final[name] = rs, calibrate(rs, cals)
                entry.setdefault("curves", {})[name] = (ends, curve)
            if mpop == "p2":
                for h in HORIZONS:
                    raw[f"banded_h{h}"] = fitted["banded"][h].score("banded_logistic", rows)
            entry |= {"raw": raw, "final": final, "unseen": unseen}
            scored[(pop, month)] = entry
            step(f"month scored ({pop})")

    # --- evaluate every registered cell ------------------------------------------------
    results: dict[str, Any] = {"cells": [], "rehearsal": rehearsal, "fit_seconds": fit_seconds}
    goal: dict[tuple, str] = {}
    total = planned_cells(months)
    ckpt = None
    if checkpoint is not None:
        ckpt = Checkpoint(checkpoint, scored_fingerprint(scored, replicates, seed, buckets))
        step(f"checkpoint holds {len(ckpt.cells)} of {total} cells")

    def k_of(h):
        return HORIZONS.index(h)

    def cell_inputs(entry, h, mask):
        rows = entry["rows"][mask]
        y = horizon_target(rows, h)
        r = {
            "survival_dt01": entry["raw"]["dt01"][mask][:, k_of(h)],
            "classifier": entry["raw"]["classifier"][mask][:, k_of(h)],
        }
        f = {
            "survival_dt01": entry["final"]["dt01"]["final"][mask][:, k_of(h)],
            "classifier": entry["final"]["classifier"]["final"][mask][:, k_of(h)],
        }
        return rows, y, r, f

    def record(section, cell_id, compute, *args):
        key = f"{section}|{cell_id}"
        if ckpt is not None and key in ckpt.cells:
            data = ckpt.cells[key]
        else:
            data = compute(*args)
            if ckpt is not None:
                data = ckpt.save(key, data)
        status = data.get("status", "ok") if isinstance(data, dict) else "ok"
        results["cells"].append(
            {
                "section": section,
                "cell": cell_id,
                "status": status,
                "reason": data.get("reason", "") if isinstance(data, dict) else "",
            }
        )
        results.setdefault(section, {})[cell_id] = data
        step(f"cell {len(results['cells'])} of {total} done ({section})")
        return data

    # --- one function per cell type; record() runs it unless the checkpoint has it ---

    def main_cell(section, entry, y, cust, r, f, comps):
        data = evaluate_cell(y, cust, r, f, comps, replicates, seed)
        data["stop_day"] = entry["C"]
        data["order_fix"] = {
            n: {k: v for k, v in entry["final"][n].items() if k not in ("before_fix", "final")}
            for n in ("classifier", "dt01")
        }
        if section == "primary" and data["status"] == "ok":
            sm = data["models"]["survival_dt01"]
            data["calibration_verdict_survival"] = calibration_verdict(
                sm["slope"], sm["intercept"], limits
            )
        return data

    def never_seen_cell(rows, y, cust, r, comps):
        ns = ~rows.msno.isin(seen).to_numpy()
        yn = y[ns]
        if yn.sum() < floor_e or (len(yn) - yn.sum()) < floor_n:
            return {
                "status": "not interpretable",
                "reason": (
                    f"{int(yn.sum())} events, {int(len(yn) - yn.sum())} non-events "
                    f"(floor {floor_e}/{floor_n})"
                ),
                "rows": int(ns.sum()),
                "lapses": int(yn.sum()),
            }
        return {
            "status": "ok",
            "rows": int(ns.sum()),
            "lapses": int(yn.sum()),
            "comparisons": {
                f"{a} minus {b}": paired(cust[ns], yn, r[a][ns], r[b][ns], replicates, seed)
                for a, b in comps[:3]
            },
        }

    def without_discount100_cell(rows, y, cust, r, f, comps):
        paid = (rows.segment == "paid").to_numpy()
        return {
            "status": "ok",
            "rows": int(paid.sum()),
            "lapses": int(y[paid].sum()),
            "comparisons": {
                f"{a} minus {b}": paired(
                    cust[paid], y[paid], r[a][paid], r[b][paid], replicates, seed
                )
                for a, b in comps[:2]
            },
            **{
                f"survival_{key}": _boot(
                    cust[paid],
                    calibration_stat(y[paid], f["survival_dt01"][paid], key),
                    replicates,
                    seed,
                )
                for key in ("slope", "intercept")
            },
        }

    def segment_cell(entry, h):
        # Every P2 segment; trial-only and promo-only are outside the training population.
        rows_all = entry["rows"]
        seg_out = {}
        for seg, g in rows_all.groupby("segment"):
            idx = rows_all.index.get_indexer(g.index)
            yg = horizon_target(g, h)
            seg_out[seg] = {
                "rows": int(len(g)),
                "lapses": int(yg.sum()),
                "rate": float(yg.mean()),
                "in_training_population": seg in protocol.main_segments,
                "auc_survival_dt01": cell(
                    auc_value, yg, entry["raw"]["dt01"][idx][:, k_of(h)]
                ).__dict__,
                "auc_classifier": cell(
                    auc_value, yg, entry["raw"]["classifier"][idx][:, k_of(h)]
                ).__dict__,
            }
        return {"status": "ok", "segments": seg_out}

    def last_row_cell(rows, y, cust, r, f):
        b = DESCRIPTIVE_BREAKDOWNS["last_row"]
        groups = np.array([breakdown_group(v, b) for v in rows.last_row_kind.astype("object")])
        bo = {}
        for gname in [*b["groups"], "missing"]:
            gm = groups == gname
            if not gm.any():
                continue
            yg = y[gm]
            bo[gname] = {
                "rows": int(gm.sum()),
                "share_of_rows": float(gm.mean()),
                "lapses": int(yg.sum()),
                "lapse_rate": float(yg.mean()),
                **{
                    f"auc_{mname}": _boot(cust[gm], auc_stat(yg, r[mname][gm]), replicates, seed)
                    for mname in ("survival_dt01", "classifier")
                },
                **{
                    f"{key}_{mname}": _boot(
                        cust[gm], calibration_stat(yg, f[mname][gm], key), replicates, seed
                    )
                    for mname in ("survival_dt01", "classifier")
                    for key in ("slope", "intercept")
                },
            }
        return {"status": "ok", "groups": bo}

    def sensitivity_cell(entry, main, h, y, cust, r):
        # Ablations and the survival sensitivity (paired against the primary model).
        abl = {}
        for v in variants:
            if v == "primary":
                continue
            av = entry["raw"][f"classifier_{v}"][main][:, k_of(h)]
            sv = entry["raw"][f"dt01_{v}"][main][:, k_of(h)]
            abl[f"classifier {v} minus classifier"] = paired(
                cust, y, av, r["classifier"], replicates, seed
            )
            abl[f"survival_dt01 {v} minus survival_dt01"] = paired(
                cust, y, sv, r["survival_dt01"], replicates, seed
            )
        sens = entry["raw"]["dt01_plus_2016_11"][main][:, k_of(h)]
        abl["survival_dt01 plus 2016-11 minus survival_dt01"] = paired(
            cust, y, sens, r["survival_dt01"], replicates, seed
        )
        return {"status": "ok", "comparisons": abl}

    def p1_cell(p1, h):
        y1 = horizon_target(p1["rows"], h)
        return evaluate_cell(
            y1,
            p1["rows"].msno.to_numpy(),
            {
                "survival_dt01": p1["raw"]["dt01"][:, k_of(h)],
                "classifier": p1["raw"]["classifier"][:, k_of(h)],
            },
            {
                "survival_dt01": p1["final"]["dt01"]["final"][:, k_of(h)],
                "classifier": p1["final"]["classifier"]["final"][:, k_of(h)],
            },
            [("survival_dt01", "classifier")],
            replicates,
            seed,
        )

    def pooled_cell():
        # Both 7-day months jointly; a summary, never the headline.
        folds = {"survival_dt01 minus classifier": [], "survival_dt01 minus banded_logistic": []}
        for month in months[7]:
            entry = scored[("p2_all", month)]
            main = entry["rows"].segment.isin(protocol.main_segments).to_numpy()
            rows, y, r, _ = cell_inputs(entry, 7, main)
            cust = rows.msno.to_numpy()
            folds["survival_dt01 minus classifier"].append(
                FoldScores(cust, y, r["survival_dt01"], r["classifier"])
            )
            folds["survival_dt01 minus banded_logistic"].append(
                FoldScores(cust, y, r["survival_dt01"], entry["raw"]["banded_h7"][main])
            )
        pooled = {
            name: _interval(paired_auc_difference(fs, replicates, seed))
            for name, fs in folds.items()
        }
        return {"status": "ok", "comparisons": pooled}

    def survival_metrics_cell(entry):
        # Concordance and integrated Brier score, with skill.
        main = entry["rows"].segment.isin(protocol.main_segments).to_numpy()
        rows = entry["rows"][main]
        C = entry["C"]
        t, e = rows.days_to_event.to_numpy(), rows.event_observed.to_numpy()
        surv_out = {"stop_day": C}
        grid_primary = [d for d in dt_cfg["interval_ends"] if d <= C]
        grid_daily = list(range(1, C + 1))
        band = expiry_band(rows.days_until_expiry.to_numpy(dtype=float), km_edges)
        ref_overall = np.tile(km_overall, (len(rows), 1))
        ref_band = np.stack([km_band.get(b, km_overall) for b in band])
        for name in ("dt01", "dt02", "dt01_plus_2016_11"):
            ends, curve = entry["curves"][name]
            S = 1 - curve[main]
            # Survival on every day 1..C: linear between interval ends (S = 1 at day 0).
            xs = np.array([0, *ends], dtype=float)
            S_full = np.column_stack([np.ones(len(S)), S])
            S_daily = np.vstack([np.interp(grid_daily, xs, row) for row in S_full])
            idx_p = [ends.index(d) for d in grid_primary]
            risk = 1 - S[:, ends.index(max(grid_primary))]
            m_out = {
                "concordance": cell(harrell_c, t, e, risk, C).__dict__,
                "ibs_primary_grid": cell(ibs, t, e, S[:, idx_p], grid_primary).__dict__,
                "ibs_every_day": cell(ibs, t, e, S_daily, grid_daily).__dict__,
            }
            for ref_name, ref in (("km_overall", ref_overall), ("km_by_band", ref_band)):
                ref_p = cell(ibs, t, e, ref[:, grid_primary], grid_primary).__dict__
                if m_out["ibs_primary_grid"]["status"] == "ok" and ref_p["status"] == "ok":
                    m_out[f"skill_vs_{ref_name}"] = (
                        1 - m_out["ibs_primary_grid"]["value"] / ref_p["value"]
                    )
            surv_out[name] = m_out
        return {"status": "ok", **surv_out}

    def shape_cell(entry):
        # Days 8-30 lapses predicted by the uncalibrated curve vs observed, and how
        # often the three separate classifiers cross (descriptive).
        main = entry["rows"].segment.isin(protocol.main_segments).to_numpy()
        rows = entry["rows"][main]
        ends, curve = entry["curves"]["dt01"]
        S7, S30 = 1 - curve[main][:, ends.index(7)], 1 - curve[main][:, ends.index(30)]
        observed = int(
            (
                (rows.event_observed == 1) & (rows.days_to_event > 7) & (rows.days_to_event <= 30)
            ).sum()
        )
        rs = entry["raw"]["classifier"][main]
        return {
            "status": "ok",
            "predicted_lapses_days_8_30": float((S7 - S30).sum()),
            "observed_lapses_days_8_30": observed,
            "classifier_crossing_share": float((np.diff(rs, axis=1) < 0).any(axis=1).mean()),
            "note": (
                "the curve between 7 and 30 days is uncalibrated; the crossing share "
                "describes three separate classifiers"
            ),
        }

    # --- evaluate, in the registered order ----------------------------------------------
    for h in (7, 30, 59):
        section = "demonstration_59" if h == 59 else "primary"
        for month in months[h]:
            entry = scored[("p2_all", month)]
            main = entry["rows"].segment.isin(protocol.main_segments).to_numpy()
            rows, y, r, f = cell_inputs(entry, h, main)
            cust = rows.msno.to_numpy()
            r |= {
                "banded_logistic": entry["raw"][f"banded_h{h}"][main],
                "survival_dt02": entry["raw"]["dt02"][main][:, k_of(h)],
            }
            f |= {
                "banded_logistic": entry["raw"][f"banded_h{h}"][main],
                "survival_dt02": entry["final"]["dt02"]["final"][main][:, k_of(h)],
            }
            comps = [
                ("survival_dt01", "classifier"),
                ("survival_dt01", "banded_logistic"),
                ("classifier", "banded_logistic"),
                ("survival_dt02", "classifier"),
                ("survival_dt02", "survival_dt01"),
            ]
            cid = f"{h} days, {month}"
            data = record(section, cid, main_cell, section, entry, y, cust, r, f, comps)
            if section == "primary" and data["status"] == "ok":
                for comp, against in (
                    ("survival_dt01 minus classifier", "lightgbm_classifier"),
                    ("survival_dt01 minus banded_logistic", "strongest_simple_baseline"),
                ):
                    goal[("discrimination", h, month, against)] = data["comparisons"][comp][
                        "verdicts"
                    ]["0.01"]
                cv = data["calibration_verdict_survival"]
                goal[("calibration", h, month, "survival")] = (
                    "pass" if cv["pass"] else ("not_computable" if cv["status"] != "ok" else "fail")
                )
            # Never-seen subset (comparisons between models only).
            record(f"{section}_never_seen", cid, never_seen_cell, rows, y, cust, r, comps)
            if section != "primary":
                continue
            record(
                "without_discount100_only",
                cid,
                without_discount100_cell,
                rows,
                y,
                cust,
                r,
                f,
                comps,
            )
            record("by_segment", cid, segment_cell, entry, h)
            record("descriptive_last_row", cid, last_row_cell, rows, y, cust, r, f)
            record("sensitivities", cid, sensitivity_cell, entry, main, h, y, cust, r)
            record("p1_secondary", cid, p1_cell, scored[("p1", month)], h)

    record("pooled_7_day_summary", "7 days, both months", pooled_cell)
    for against in ("lightgbm_classifier", "strongest_simple_baseline"):
        per = [goal.get(("discrimination", 7, m, against), "missing") for m in months[7]]
        results.setdefault("combined_7_day", {})[against] = {
            "per_month": per,
            "combined": combine_months(per),
        }
    for month in all_months:
        record("survival_metrics", str(month), survival_metrics_cell, scored[("p2_all", month)])
    for month in months[30]:
        record("shape_checks", str(month), shape_cell, scored[("p2_all", month)])

    results["unseen_levels"] = {str(m): scored[("p2_all", m)]["unseen"] for m in all_months}
    results["goal_results"] = {"|".join(map(str, k)): v for k, v in goal.items()}
    goal_cells = [
        Cell(kind, h, m, against)
        for h in (7, 30)
        for m in months[h]
        for kind, against in (
            ("discrimination", "lightgbm_classifier"),
            ("discrimination", "strongest_simple_baseline"),
            ("calibration", "survival"),
        )
    ]
    results["goal_cells"] = cell_rows(goal_cells, goal)
    results["goal_summary"] = summarize(goal_cells, goal)
    results["seconds"] = round(time.time() - t_start, 1)
    if rehearsal:
        results["rehearsal_months"] = {str(h): [str(m) for m in ms] for h, ms in months.items()}
        results["rehearsal_limits"] = REHEARSAL_LIMITS
    return results
