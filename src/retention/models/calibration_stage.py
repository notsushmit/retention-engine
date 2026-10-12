"""Calibration stage on 2016 data. Decides nothing on test (Platt at every horizon there).

Part A, f7b_cal (7 days, descriptive): models fitted on 2016-08 (labels as of
2016-11-01), Platt and isotonic fitted on 2016-09, scored on 2016-11. The
registered rule (isotonic replaces Platt only if the paired interval of the
Brier difference, isotonic minus Platt, lies below 0) is evaluated and
reported; on test the method is Platt at every horizon whatever this shows.

Part B, the October pre-test check: the test pipeline's models (fit on
2016-08/09, labels as of 2017-01-01: classifier c01 per horizon and the
discrete-time model dt01), Platt fitted on the calibration half of 2016-10
(labels as of 2017-01-01) and evaluated on the selection half (final labels),
at 7, 30 and 59 days, through the registered order (calibrate, clip, ordering
fix, clip). Reported: slope and intercept with intervals, ECE with its interval
and the interpretability rule, AUC of raw and final scores, distinct values,
the ordering fix on both halves, and the test-pipeline fit times.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml

from retention.models.bootstrap import bootstrap, ece_interpretable
from retention.models.calibration import Isotonic, Platt, calibrate
from retention.models.classifier import Classifier, candidate_params, feature_set_columns
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.frames import fit_rows, half_mask, score_rows, survival_fit_rows
from retention.models.metrics import SortedScores, brier, calibration_slope, ece
from retention.models.protocol import EVAL_CONFIG, Protocol, Slice
from retention.models.stages import _interval, assert_no_label_building_code
from retention.models.tracking import Tracking, start

HORIZONS = (7, 30, 59)


def _iv(customers: np.ndarray, stat, replicates: int, seed: int) -> dict[str, Any]:
    return _interval(bootstrap([customers], lambda f, w: stat(w), replicates, seed))


def calibration_report(y, p_raw, p_final, p_before, customers, replicates, seed) -> dict[str, Any]:
    out = {
        "rows": int(len(y)),
        "events": int(y.sum()),
        "rate": float(y.mean()),
        "mean_final": float(p_final.mean()),
        "auc_raw": SortedScores(p_raw).auc(y),
        "auc_final": SortedScores(p_final).auc(y),
        "distinct_raw": int(np.unique(p_raw).size),
        "distinct_final": int(np.unique(p_final).size),
        "slope_before_fix": calibration_slope(y, p_before)["slope"],
        "intercept_before_fix": calibration_slope(y, p_before)["intercept"],
    }
    for key in ("slope", "intercept"):
        out[key] = _iv(
            customers, lambda w, k=key: calibration_slope(y, p_final, w)[k], replicates, seed
        )
    e = _iv(customers, lambda w: ece(y, p_final, 10, w), replicates, seed)
    e["share_of_rate"] = e["point"] / out["rate"] if out["rate"] else float("nan")
    e["interpretable"] = bool(ece_interpretable(e["point"], e["lo"], e["hi"]))
    out["ece"] = e
    return out


def calibration_stage(
    protocol: Protocol, processed: Path, cache: Path, tr: Tracking, replicates: int
) -> dict[str, Any]:
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    seed, threads = protocol.seed, protocol.threads
    cats = raw["classifier"]["categorical"]
    cols = feature_set_columns(raw["selection"]["feature_sets"]["fs1"]["drop"])
    c01 = candidate_params(raw["classifier"]["candidates"]["c01"])
    fixed = raw["classifier"]["fixed_params"]
    dt = raw["survival"]["discrete_time"]
    floor = raw["calibration_method"]["isotonic_min_events"]
    data_files = {
        "labels": processed / "labels.parquet",
        "features": processed / "features.parquet",
    }

    def classifier(fit):
        return Classifier(cols, cats, c01, fixed, threads, seed).fit(fit)

    def dt01(fit):
        offset = dt["candidates"]["dt01"]["expiry_offset_input"]
        return DiscreteTimeModel(offset, dt["interval_ends"], c01, fixed, cats, threads, seed).fit(
            fit, cols
        )

    out: dict[str, Any] = {}

    # --- Part A: f7b_cal, 7 days ----------------------------------------------------
    fold = protocol.fold("f7b_cal")
    both = (fold.fit[0], fold.calibration.snapshot)
    rows = fit_rows(protocol, processed, cache, fold.as_of, both, cols, 7)
    fit = rows[rows.snapshot_date.dt.date == fold.fit[0]]
    cal = rows[rows.snapshot_date.dt.date == fold.calibration.snapshot]
    srows = survival_fit_rows(protocol, processed, cache, fold.as_of, both, cols)
    sfit = srows[srows.snapshot_date.dt.date == fold.fit[0]]
    scored = score_rows(protocol, processed, fold.score, cols, 7)
    y, cust = scored.y.to_numpy(), scored.msno.to_numpy()
    yc = cal.y.to_numpy()
    part_a = {"calibration_events": int(yc.sum()), "isotonic_eligible": int(yc.sum()) >= floor}
    models = {"classifier_c01": classifier(fit), "dt01": dt01(sfit)}
    for name, m in models.items():
        if name == "dt01":
            pc, ps = (
                m.lapse_probability(cal, [7])[0][:, 0],
                m.lapse_probability(scored, [7])[0][:, 0],
            )
        else:
            pc, ps = m.predict(cal)[0], m.predict(scored)[0]
        platt = Platt(seed).fit(pc, yc).predict(ps)
        iso = Isotonic(floor).fit(pc, yc).predict(ps)

        def diff(w, iso=iso, platt=platt):
            return brier(y, iso, w) - brier(y, platt, w)

        d = _iv(cust, diff, replicates, seed)
        part_a[name] = {
            "brier_platt": brier(y, platt),
            "brier_isotonic": brier(y, iso),
            "isotonic_minus_platt": d,
            "isotonic_would_replace_platt": d["status"] == "ok" and d["hi"] < 0,
            "platt": {k: calibration_slope(y, platt)[k] for k in ("slope", "intercept")},
            "isotonic": {
                k: calibration_slope(y, np.clip(iso, 1e-6, 1 - 1e-6))[k]
                for k in ("slope", "intercept")
            },
            "distinct_platt": int(np.unique(platt).size),
            "distinct_isotonic": int(np.unique(iso).size),
        }
    out["f7b_cal"] = part_a
    with start(
        tr,
        "calibration_fold",
        "calibration_fold-f7b_cal-both-platt_isotonic-h7-fs1",
        {"fold": "f7b_cal", "horizon": 7, "decides": "nothing on test"},
        data_files=data_files,
    ):
        mlflow.log_dict(part_a, "f7b_cal.json")

    # --- Part B: October halves with the test pipeline's models ------------------------
    tf = protocol.test_fit
    snaps = (*tf.fit, tf.calibration.snapshot)
    oct_ = tf.calibration.snapshot
    times: dict[str, float] = {}
    raw_c, raw_s, ys, yc_ = {}, {}, {}, {}
    # The selection and calibration halves hold the same customers at every
    # horizon (P2 main rows of 2016-10); only the outcome changes with h.
    sel = cal_keep = None
    for h in HORIZONS:
        r = fit_rows(protocol, processed, cache, tf.as_of, snaps, cols, h)
        fit_h = r[r.snapshot_date.dt.date.isin(tf.fit)]
        cal_h = r[r.snapshot_date.dt.date == oct_]
        cal_h = cal_h[half_mask(cal_h.msno, "calibration", protocol)]
        sel = score_rows(protocol, processed, Slice(oct_, "selection"), cols, h)
        t0 = time.time()
        m = classifier(fit_h)
        times[f"classifier_c01_h{h}"] = round(time.time() - t0, 1)
        raw_c[("classifier_c01", h)], raw_s[("classifier_c01", h)] = (
            m.predict(cal_h)[0],
            m.predict(sel)[0],
        )
        yc_[h], ys[h] = cal_h.y.to_numpy(), sel.y.to_numpy()
        cal_keep = cal_h
    s = survival_fit_rows(protocol, processed, cache, tf.as_of, snaps, cols)
    t0 = time.time()
    m = dt01(s[s.snapshot_date.dt.date.isin(tf.fit)])
    times["dt01"] = round(time.time() - t0, 1)
    times["dt01_expanded_rows"] = m.expanded_rows
    pc, ps = (
        m.lapse_probability(cal_keep, list(HORIZONS))[0],
        m.lapse_probability(sel, list(HORIZONS))[0],
    )
    for k, h in enumerate(HORIZONS):
        raw_c[("dt01", h)], raw_s[("dt01", h)] = pc[:, k], ps[:, k]

    part_b: dict[str, Any] = {
        "fit_seconds": times,
        "calibration_events": {h: int(yc_[h].sum()) for h in HORIZONS},
    }
    cust_s = sel.msno.to_numpy()
    for name in ("classifier_c01", "dt01"):
        cals = [Platt(seed).fit(raw_c[(name, h)], yc_[h]) for h in HORIZONS]
        rs = np.column_stack([raw_s[(name, h)] for h in HORIZONS])
        rc = np.column_stack([raw_c[(name, h)] for h in HORIZONS])
        on_s, on_c = calibrate(rs, cals), calibrate(rc, cals)
        entry = {
            "platt": {h: {"a": c.a, "b": c.b} for h, c in zip(HORIZONS, cals, strict=True)},
            "raw_crossing_share": float((np.diff(rs, axis=1) < 0).any(axis=1).mean()),
            "order_fix_selection": {
                k: v for k, v in on_s.items() if k not in ("before_fix", "final")
            },
            "order_fix_calibration": {
                k: v for k, v in on_c.items() if k not in ("before_fix", "final")
            },
            "by_horizon": {},
        }
        for flag_key in ("order_fix_selection", "order_fix_calibration"):
            entry[flag_key]["flag_over_1_percent"] = entry[flag_key]["share_changed"] > 0.01
        for k, h in enumerate(HORIZONS):
            entry["by_horizon"][h] = calibration_report(
                ys[h],
                rs[:, k],
                on_s["final"][:, k],
                on_s["before_fix"][:, k],
                cust_s,
                replicates,
                seed,
            )
        part_b[name] = entry
    out["october"] = part_b
    with start(
        tr,
        "calibration_fold",
        "calibration_fold-2016-10-both-platt-h7-fs1",
        {
            "snapshot": str(oct_),
            "fit": ",".join(map(str, tf.fit)),
            "fit_label_as_of": str(tf.as_of),
            "calibrator": "platt",
            "decides": "nothing",
        },
        data_files=data_files,
    ):
        mlflow.log_dict(part_b, "october.json")
    assert_no_label_building_code()
    return out
