"""Fold stage for the survival families: Cox, XGBoost AFT and discrete-time.

Each candidate is fitted on the two fold fit sets (2016-08 labelled as of
2016-10-01, shared by f7a and f30; 2016-08/09 labelled as of 2016-11-01 for
f7b), on the chosen feature set fs1, with survival labels (time, event).
It is scored as 1 - S(h) on the folds' scored rows: f7a (7 days), f7b (7
days), f30 (30 days). The registered ladder picks the best candidate within
each family on the fold-mean AUC over f7a, f7b and f30, then the family.

Every candidate is reported next to the chosen classifier (c01) and banded
logistic: paired AUC differences per fold and for the fold mean. Also: the
proportional-hazards test per feature group for the Cox fits on the larger
fit set, the Cox preprocessing constants on both fit sets, and the
pre-registered expectations, printed beside the results.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from lifelines.statistics import proportional_hazard_test

from retention.data.sample import bucket
from retention.features.spec import FEATURE_COLUMNS
from retention.models.aft import AFTModel
from retention.models.baselines import Baselines
from retention.models.bootstrap import FoldScores, paired_auc_difference
from retention.models.classifier import Classifier, candidate_params, feature_set_columns
from retention.models.cox import STRATUM, CoxModel
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.frames import fit_rows, score_rows, survival_fit_rows
from retention.models.metrics import SortedScores
from retention.models.protocol import EVAL_CONFIG, Protocol
from retention.models.selection import Candidate, auc_comparison, ladder
from retention.models.stages import _interval, assert_no_label_building_code
from retention.models.tracking import Tracking, run_name, start

FOLDS = ("f7a", "f7b", "f30")
PH_BUCKETS = 2
FIT_SET = {"f7a": "aug", "f30": "aug", "f7b": "aug_sep"}


def _group(design_column: str) -> str:
    base = design_column.removeprefix("missing:").split("=")[0]
    return FEATURE_COLUMNS.get(base, "other")


def ph_by_group(model: CoxModel, rows) -> dict[str, Any]:
    """Proportional-hazards test (rank time transform) summarised per feature group."""
    df = model._frame(rows).assign(time=rows.time.to_numpy(), event=rows.event.to_numpy())
    with model.threads.limit():
        res = proportional_hazard_test(model.fitter, df, time_transform="rank")
    summary = res.summary
    out: dict[str, Any] = {}
    for col, r in summary.iterrows():
        name = col if isinstance(col, str) else col[0]
        g = out.setdefault(_group(name), {"columns": 0, "statistics": [], "p_below_0.01": 0})
        g["columns"] += 1
        g["statistics"].append(float(r["test_statistic"]))
        g["p_below_0.01"] += int(r["p"] < 0.01)
    for g in out.values():
        stats = sorted(g.pop("statistics"))
        g["median_statistic"] = float(np.median(stats))
        g["max_statistic"] = float(stats[-1])
    return out


def survival_stage(
    protocol: Protocol, processed: Path, cache: Path, tr: Tracking, replicates: int
) -> dict[str, Any]:
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    seed, threads = protocol.seed, protocol.threads
    cats = raw["classifier"]["categorical"]
    cols = feature_set_columns(raw["selection"]["feature_sets"]["fs1"]["drop"])
    surv = raw["survival"]
    c01 = candidate_params(raw["classifier"]["candidates"]["c01"])
    fixed = raw["classifier"]["fixed_params"]
    data_files = {
        "labels": processed / "labels.parquet",
        "features": processed / "features.parquet",
    }

    # Fit sets (survival labels) and scored rows.
    fa, fb = protocol.fold("f7a"), protocol.fold("f7b")
    fit_sets = {
        "aug": survival_fit_rows(protocol, processed, cache, fa.as_of, fa.fit, cols),
        "aug_sep": survival_fit_rows(protocol, processed, cache, fb.as_of, fb.fit, cols),
    }
    scored, reference = {}, {}
    for name in FOLDS:
        fold = protocol.fold(name)
        rows = score_rows(protocol, processed, fold.score, cols, fold.horizon)
        cfit = fit_rows(protocol, processed, cache, fold.as_of, fold.fit, cols, fold.horizon)
        clf = Classifier(cols, cats, c01, fixed, threads, seed).fit(cfit).predict(rows)[0]
        banded = Baselines(cfit, raw["baselines"]["banded_logistic"]["bands"], threads, seed).score(
            "banded_logistic", rows
        )
        scored[name] = rows
        reference[name] = {"classifier_c01": clf, "banded_logistic": banded}

    # Candidates.
    def make(cid: str):
        if cid.startswith("cox"):
            strat, pen = surv["cox"]["candidates"][cid]
            return CoxModel(strat, pen, cats, surv["cox"]["strata_bands"]["p2"], threads)
        if cid.startswith("aft"):
            dist, sigma = surv["xgboost_aft"]["candidates"][cid]
            return AFTModel(dist, sigma, surv["xgboost_aft"]["fixed_params"], cats, threads, seed)
        offset = surv["discrete_time"]["candidates"][cid]["expiry_offset_input"]
        ends = surv["discrete_time"]["interval_ends"]
        return DiscreteTimeModel(offset, ends, c01, fixed, cats, threads, seed)

    family_of = {"cox": "cox", "aft": "xgboost_aft", "dt": "discrete_time"}
    ids = [
        *surv["cox"]["candidates"],
        *surv["xgboost_aft"]["candidates"],
        *surv["discrete_time"]["candidates"],
    ]
    scores: dict[str, dict[str, np.ndarray]] = {}
    details: dict[str, Any] = {}
    fitted: dict[tuple[str, str], Any] = {}
    for cid in ids:
        family = family_of[cid.rstrip("0123456789")]
        scores[cid], details[cid] = {}, {"family": family, "fit_seconds": {}, "unseen_levels": {}}
        for set_name, rows in fit_sets.items():
            t0 = time.time()
            model = make(cid).fit(rows, cols)
            details[cid]["fit_seconds"][set_name] = round(time.time() - t0, 1)
            fitted[(cid, set_name)] = model
            if hasattr(model, "expanded_rows"):
                details[cid].setdefault("expanded_rows", {})[set_name] = model.expanded_rows
        for name in FOLDS:
            model = fitted[(cid, FIT_SET[name])]
            h = protocol.fold(name).horizon
            out = model.lapse_probability(scored[name], [h])
            p, unseen = (out, {}) if isinstance(out, np.ndarray) else out
            if isinstance(model, CoxModel):
                unseen = dict(model.prep.unseen)
            scores[cid][name] = p[:, 0]
            details[cid]["unseen_levels"][name] = unseen
        details[cid]["status"] = "ok"
        if isinstance(fitted[(cid, "aug")], CoxModel):
            details[cid]["cox_design"] = {s: fitted[(cid, s)].summary() for s in fit_sets}
            failed = [s for s in fit_sets if not fitted[(cid, s)].fit_ok]
            if failed:
                details[cid]["status"] = (
                    "failed fit: log-likelihood below the no-covariate model on "
                    + ", ".join(failed)
                )
        y = {n: scored[n].y.to_numpy() for n in FOLDS}
        aucs = {n: SortedScores(scores[cid][n]).auc(y[n]) for n in FOLDS}
        details[cid]["auc_by_fold"] = aucs
        details[cid]["fold_mean_auc"] = float(np.mean(list(aucs.values())))
        with start(
            tr,
            "fold",
            run_name("fold", "f7a_f7b_f30", family, cid, 7, "fs1"),
            {
                "candidate": cid,
                "family": family,
                "feature_set": "fs1",
                "folds": ",".join(FOLDS),
                "fit_sets": "2016-08 as of 2016-10-01; 2016-08/09 as of 2016-11-01",
                "seed": seed,
            },
            data_files=data_files,
        ):
            mlflow.log_metrics({f"auc.{n}": v for n, v in aucs.items()})

    def fold_scores(cid: str) -> list[FoldScores]:
        return [
            FoldScores(
                scored[n].msno.to_numpy(), scored[n].y.to_numpy(), scores[cid][n], scores[cid][n]
            )
            for n in FOLDS
        ]

    def versus(cid: str, ref: str) -> dict[str, Any]:
        per = {}
        for n in FOLDS:
            f = FoldScores(
                scored[n].msno.to_numpy(), scored[n].y.to_numpy(), scores[cid][n], reference[n][ref]
            )
            per[n] = _interval(paired_auc_difference([f], replicates, seed))
        joint = [
            FoldScores(
                scored[n].msno.to_numpy(), scored[n].y.to_numpy(), scores[cid][n], reference[n][ref]
            )
            for n in FOLDS
        ]
        return {
            "per_fold": per,
            "fold_mean": _interval(paired_auc_difference(joint, replicates, seed)),
        }

    usable = [cid for cid in ids if details[cid]["status"] == "ok"]
    for cid in ids:
        if cid not in usable:
            details[cid]["versus_classifier_c01"] = details[cid]["status"]
            continue
        details[cid]["versus_classifier_c01"] = versus(cid, "classifier_c01")
        joint = [
            FoldScores(
                scored[n].msno.to_numpy(),
                scored[n].y.to_numpy(),
                scores[cid][n],
                reference[n]["banded_logistic"],
            )
            for n in FOLDS
        ]
        details[cid]["versus_banded_logistic"] = {
            "fold_mean": _interval(paired_auc_difference(joint, replicates, seed))
        }

    # Ladder within each family, then across families.
    n_groups = len({FEATURE_COLUMNS[c] for c in cols})
    compare = auc_comparison({cid: fold_scores(cid) for cid in ids}, replicates, seed)
    winners, steps = {}, {}
    for family in ("cox", "xgboost_aft", "discrete_time"):
        cands = [
            Candidate(c, family, n_groups, len(cols))
            for c in usable
            if details[c]["family"] == family
        ]
        chosen, st = ladder(cands, compare)
        winners[family], steps[family] = chosen, st
    family_choice, family_steps = ladder(list(winners.values()), compare)

    def step_rows(st):
        return [
            {
                "candidate": s.candidate,
                "against": s.current,
                "d": s.comparison.d,
                "lo": s.comparison.lo,
                "hi": s.comparison.hi,
                "by_fold": list(s.comparison.candidate_by_fold),
                "current_by_fold": list(s.comparison.current_by_fold),
                "interval_above_zero": s.interval_above_zero,
                "better_on_every_fold": s.better_on_every_fold,
                "replaced": s.replaced,
            }
            for s in st
        ]

    # Proportional hazards per feature group, on a fixed customer subsample of the
    # larger fit set (md5 buckets below PH_BUCKETS): the test's cost grows much
    # faster than linearly with rows (15,050 rows 1.3 s; 30,279 rows 58 s;
    # 60,902 rows 1,605 s), so all 303,072 rows are not feasible. Cox is refitted
    # on the subsample, so model and test use the same rows. Descriptive only.
    big = fit_sets["aug_sep"]
    sub = big[np.array([bucket(m) < PH_BUCKETS for m in big.msno])].reset_index(drop=True)
    ph = {
        "subsample": {
            "md5_buckets_below": PH_BUCKETS,
            "rows": len(sub),
            "lapses": int(sub.event.sum()),
        }
    }
    for cid in ("cox01", "cox03"):
        ph[cid] = ph_by_group(make(cid).fit(sub, cols), sub)
    logged = raw["cox_preprocessing"]["logged_columns"]
    constants = {s: fitted[("cox01", s)].prep.logged_constants(logged) for s in fit_sets}

    out = {
        "candidates": details,
        "reference_auc": {
            n: {r: SortedScores(v).auc(scored[n].y.to_numpy()) for r, v in reference[n].items()}
            for n in FOLDS
        },
        "within_family": {
            f: {"chosen": w.id, "steps": step_rows(steps[f])} for f, w in winners.items()
        },
        "family": {
            "chosen": family_choice.id,
            "family": family_choice.family,
            "steps": step_rows(family_steps),
        },
        "proportional_hazards_by_group": ph,
        "cox_constants_by_fit_set": constants,
        "expectations": raw["expectations"],
        "stratum_column": STRATUM,
    }
    with start(
        tr,
        "fold",
        run_name("fold", "f7a_f7b_f30", "survival", "chosen", 7, "fs1"),
        {"rule": "registered ladder", "folds": ",".join(FOLDS)},
        tags={
            "chosen_survival_candidate": family_choice.id,
            "chosen_survival_family": family_choice.family,
            **{f"chosen.{f}": w.id for f, w in winners.items()},
        },
        data_files=data_files,
    ) as run:
        mlflow.log_dict({k: out[k] for k in ("within_family", "family")}, "ladder.json")
    out["summary_run_id"] = run.info.run_id
    assert_no_label_building_code()
    return out
