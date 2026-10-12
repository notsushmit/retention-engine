"""Fold stages. Each runs in a model process that reads cached label builds only.

baseline_stage: every simple baseline on f7a, f7b (7 days) and f30 (30 days):
each fold's AUC with its customer-bootstrap interval, the fold-mean AUC per
horizon with its joint interval, the strongest simple baseline per horizon
(highest fold-mean AUC, as registered), and each baseline's paired AUC
difference to the strongest.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml

from retention.features.spec import FEATURE_COLUMNS
from retention.models.baselines import INPUTS, NAMES, Baselines
from retention.models.bootstrap import FoldScores, bootstrap, paired_auc_difference
from retention.models.classifier import (
    Classifier,
    candidate_params,
    feature_set_columns,
    score_summary,
)
from retention.models.frames import fit_rows, score_rows
from retention.models.metrics import NOT_COMPUTABLE, SortedScores
from retention.models.protocol import EVAL_CONFIG, Protocol
from retention.models.selection import Candidate, auc_comparison, ladder, simplest_first
from retention.models.tracking import Tracking, run_name, start

BASELINE_FOLDS = {7: ("f7a", "f7b"), 30: ("f30",)}
LABEL_CODE = "retention.labels.snapshots"


def assert_no_label_building_code() -> None:
    if LABEL_CODE in sys.modules:
        raise RuntimeError("a model process must not load the label-building code")


def auc_interval(
    customers: list[np.ndarray],
    ys: list[np.ndarray],
    scores: list[np.ndarray],
    replicates: int,
    seed: int,
):
    """Fold-mean AUC of one score with its customer-bootstrap interval."""
    sorted_scores = [SortedScores(s) for s in scores]

    def stat(f: int, w: np.ndarray | None) -> float:
        return sorted_scores[f].auc(ys[f], w)

    return bootstrap(customers, stat, replicates, seed)


def _interval(iv) -> dict[str, Any]:
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi, "status": iv.status, "reason": iv.reason}


def baseline_stage(
    protocol: Protocol, processed: Path, cache: Path, tr: Tracking, replicates: int
) -> dict[str, Any]:
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    edges = raw["baselines"]["banded_logistic"]["bands"]
    seed = protocol.seed
    data_files = {
        "labels": processed / "labels.parquet",
        "features": processed / "features.parquet",
    }
    out: dict[str, Any] = {"folds": {}, "horizons": {}}
    scored: dict[str, dict[str, Any]] = {}

    for h, names in BASELINE_FOLDS.items():
        for name in names:
            fold = protocol.fold(name)
            fit = fit_rows(protocol, processed, cache, fold.as_of, fold.fit, INPUTS, h)
            rows = score_rows(protocol, processed, fold.score, INPUTS, h)
            b = Baselines(fit, edges, protocol.threads, seed)
            scores = {n: b.score(n, rows) for n in NAMES}
            scored[name] = {"rows": rows, "scores": scores}
            fold_out = {
                "fit_rows": len(fit),
                "fit_events": int(fit.y.sum()),
                "scored_rows": len(rows),
                "scored_events": int(rows.y.sum()),
                "empty_bands_dropped": b.dropped_bands,
                "auc": {},
            }
            for n, s in scores.items():
                iv = auc_interval(
                    [rows.msno.to_numpy()], [rows.y.to_numpy()], [s], replicates, seed
                )
                fold_out["auc"][n] = _interval(iv)
                params = {
                    "fold": name,
                    "horizon": h,
                    "fit_snapshots": ",".join(map(str, fold.fit)),
                    "fit_label_as_of": str(fold.as_of),
                    "scored_snapshot": str(fold.score.snapshot),
                    "scored_half": fold.score.half or "all",
                    "scored_label_as_of": "final labels.parquet",
                    "population": "P2 main",
                    "seed": seed,
                    "replicates": replicates,
                }
                with start(
                    tr,
                    "fold",
                    run_name("fold", name, "baseline", n, h, "baseline_inputs"),
                    params,
                    data_files=data_files,
                ):
                    mlflow.log_metrics(
                        {
                            k: v
                            for k, v in {
                                "auc": iv.point,
                                "auc_lo": iv.lo,
                                "auc_hi": iv.hi,
                                "scored_rows": len(rows),
                                "scored_events": int(rows.y.sum()),
                            }.items()
                            if not np.isnan(v)
                        }
                    )
            out["folds"][name] = fold_out

        # Fold-mean AUC per baseline (joint customer bootstrap across the horizon's folds).
        customers = [scored[n]["rows"].msno.to_numpy() for n in names]
        ys = [scored[n]["rows"].y.to_numpy() for n in names]
        horizon_out: dict[str, Any] = {"folds": list(names), "fold_mean_auc": {}}
        for b_name in NAMES:
            iv = auc_interval(
                customers, ys, [scored[n]["scores"][b_name] for n in names], replicates, seed
            )
            horizon_out["fold_mean_auc"][b_name] = _interval(iv)
        ranked = sorted(
            (v["point"], k) for k, v in horizon_out["fold_mean_auc"].items() if v["status"] == "ok"
        )
        strongest = ranked[-1][1]
        horizon_out["strongest"] = strongest
        horizon_out["difference_to_strongest"] = {}
        for b_name in NAMES:
            if b_name == strongest:
                continue
            folds = [
                FoldScores(
                    scored[n]["rows"].msno.to_numpy(),
                    scored[n]["rows"].y.to_numpy(),
                    scored[n]["scores"][b_name],
                    scored[n]["scores"][strongest],
                )
                for n in names
            ]
            iv = paired_auc_difference(folds, replicates, seed)
            horizon_out["difference_to_strongest"][b_name] = _interval(iv)
        with start(
            tr,
            "fold",
            run_name("fold", "_".join(names), "baseline", "strongest", h, "baseline_inputs"),
            {"horizon": h, "folds": ",".join(names), "rule": "highest fold-mean AUC"},
            tags={"strongest_simple_baseline": strongest},
            data_files=data_files,
        ) as run:
            mlflow.log_metrics(
                {f"fold_mean_auc.{k}": v["point"] for k, v in horizon_out["fold_mean_auc"].items()}
            )
        horizon_out["summary_run_id"] = run.info.run_id
        out["horizons"][str(h)] = horizon_out

    out["status_values"] = ["ok", NOT_COMPUTABLE]
    assert_no_label_building_code()
    return out


# --- the LightGBM classifier ----------------------------------------------------------

STAGE1_FOLDS = ("f7a", "f7b", "f30")
STAGE2_FOLDS = {7: ("f7a", "f7b"), 30: ("f30",)}


def classifier_stage(
    protocol: Protocol, processed: Path, cache: Path, tr: Tracking, replicates: int
) -> dict[str, Any]:
    """Stage 1 (feature set, c00) and stage 2 (13 candidates per horizon), with the ladder.

    Every result is reported next to the banded logistic baseline: the paired
    AUC difference per fold and for the fold mean.
    """
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    seed, threads = protocol.seed, protocol.threads
    cands = {cid: candidate_params(v) for cid, v in raw["classifier"]["candidates"].items()}
    fixed, cats = raw["classifier"]["fixed_params"], raw["classifier"]["categorical"]
    fsets = {
        fs: feature_set_columns(d["drop"]) for fs, d in raw["selection"]["feature_sets"].items()
    }
    edges = raw["baselines"]["banded_logistic"]["bands"]
    data_files = {
        "labels": processed / "labels.parquet",
        "features": processed / "features.parquet",
    }

    rows: dict[str, dict[str, Any]] = {}
    for name in STAGE1_FOLDS:
        fold = protocol.fold(name)
        fit = fit_rows(protocol, processed, cache, fold.as_of, fold.fit, fsets["fs3"], fold.horizon)
        score = score_rows(protocol, processed, fold.score, fsets["fs3"], fold.horizon)
        banded = Baselines(fit, edges, threads, seed).score("banded_logistic", score)
        rows[name] = {"fold": fold, "fit": fit, "score": score, "banded": banded}

    fitted: dict[tuple[str, str, str], dict[str, Any]] = {}

    def run(name: str, fs: str, cid: str, stage: str) -> dict[str, Any]:
        key = (name, fs, cid)
        if key in fitted:
            return fitted[key]
        r = rows[name]
        t0 = time.time()
        model = Classifier(fsets[fs], cats, cands[cid], fixed, threads, seed).fit(r["fit"])
        p, unseen = model.predict(r["score"])
        y = r["score"].y.to_numpy()
        result = {
            "p": p,
            "unseen": unseen,
            "auc": SortedScores(p).auc(y),
            "model": model,
            "seconds": round(time.time() - t0, 1),
        }
        fold = r["fold"]
        with start(
            tr,
            "fold",
            run_name("fold", name, "lightgbm", cid, fold.horizon, fs),
            {
                "fold": name,
                "horizon": fold.horizon,
                "stage": stage,
                "feature_set": fs,
                "candidate": cid,
                **cands[cid],
                "fit_snapshots": ",".join(map(str, fold.fit)),
                "fit_label_as_of": str(fold.as_of),
                "scored_snapshot": str(fold.score.snapshot),
                "scored_half": fold.score.half or "all",
                "population": "P2 main",
                "seed": seed,
            },
            data_files=data_files,
        ) as mlrun:
            mlflow.log_metrics(
                {
                    "auc": result["auc"],
                    "auc_minus_banded": result["auc"] - SortedScores(r["banded"]).auc(y),
                }
            )
        result["run_id"] = mlrun.info.run_id
        fitted[key] = result
        return result

    def fold_scores(names: tuple[str, ...], fs: str, cid: str, stage: str) -> list[FoldScores]:
        out = []
        for n in names:
            res = run(n, fs, cid, stage)
            s = rows[n]["score"]
            out.append(FoldScores(s.msno.to_numpy(), s.y.to_numpy(), res["p"], res["p"]))
        return out

    def versus_banded(names: tuple[str, ...], fs: str, cid: str) -> dict[str, Any]:
        per_fold = {}
        for n in names:
            s = rows[n]["score"]
            fsc = FoldScores(
                s.msno.to_numpy(), s.y.to_numpy(), fitted[(n, fs, cid)]["p"], rows[n]["banded"]
            )
            per_fold[n] = _interval(paired_auc_difference([fsc], replicates, seed))
        joint = [
            FoldScores(
                rows[n]["score"].msno.to_numpy(),
                rows[n]["score"].y.to_numpy(),
                fitted[(n, fs, cid)]["p"],
                rows[n]["banded"],
            )
            for n in names
        ]
        mean = (
            _interval(paired_auc_difference(joint, replicates, seed))
            if len(names) > 1
            else per_fold[names[0]]
        )
        return {"per_fold": per_fold, "fold_mean": mean}

    def step_rows(steps) -> list[dict[str, Any]]:
        return [
            {
                "candidate": s.candidate,
                "against": s.current,
                "d": s.comparison.d,
                "lo": s.comparison.lo,
                "hi": s.comparison.hi,
                "candidate_by_fold": list(s.comparison.candidate_by_fold),
                "current_by_fold": list(s.comparison.current_by_fold),
                "interval_above_zero": s.interval_above_zero,
                "better_on_every_fold": s.better_on_every_fold,
                "replaced": s.replaced,
            }
            for s in steps
        ]

    def candidate_rows(names, fs, ids) -> dict[str, Any]:
        out = {}
        for cid in ids:
            aucs = {n: fitted[(n, fs, cid)]["auc"] for n in names}
            entry = {
                "auc_by_fold": aucs,
                "fold_mean_auc": float(np.mean(list(aucs.values()))),
                "versus_banded": versus_banded(names, fs, cid),
                "fit_seconds": {n: fitted[(n, fs, cid)]["seconds"] for n in names},
            }
            if len(names) == 2:
                entry["fold_gap"] = aucs[names[1]] - aucs[names[0]]
            out[cid] = entry
        return out

    out: dict[str, Any] = {}

    # Stage 1: feature set, with c00, over f7a, f7b and f30.
    fs_cands = [
        Candidate(fs, "lightgbm", len({FEATURE_COLUMNS[c] for c in cols}), len(cols))
        for fs, cols in fsets.items()
    ]
    scores1 = {c.id: fold_scores(STAGE1_FOLDS, c.id, "c00", "feature_set") for c in fs_cands}
    chosen_fs, steps1 = ladder(fs_cands, auc_comparison(scores1, replicates, seed))
    out["stage1"] = {
        "folds": list(STAGE1_FOLDS),
        "order": [c.id for c in simplest_first(fs_cands)],
        "chosen": chosen_fs.id,
        "steps": step_rows(steps1),
    }
    out["stage1"]["candidates"] = {
        c.id: {
            "auc_by_fold": {n: fitted[(n, c.id, "c00")]["auc"] for n in STAGE1_FOLDS},
            "fold_mean_auc": float(
                np.mean([fitted[(n, c.id, "c00")]["auc"] for n in STAGE1_FOLDS])
            ),
            "versus_banded": versus_banded(STAGE1_FOLDS, c.id, "c00"),
        }
        for c in fs_cands
    }
    fs = chosen_fs.id

    # Stage 2: hyperparameters per horizon on the chosen feature set.
    out["stage2"] = {}
    for h, names in STAGE2_FOLDS.items():
        hp_cands = [
            Candidate(
                cid, "lightgbm", chosen_fs.feature_groups, chosen_fs.columns, tuned=cid != "c00"
            )
            for cid in cands
        ]
        scores2 = {c.id: fold_scores(names, fs, c.id, "hyperparameters") for c in hp_cands}
        chosen_c, steps2 = ladder(hp_cands, auc_comparison(scores2, replicates, seed))
        chosen_rows = {}
        for n in names:
            res = fitted[(n, fs, chosen_c.id)]
            chosen_rows[n] = {
                "score_summary": score_summary(res["p"]),
                "unseen_levels": res["unseen"],
                "gain_importance": res["model"].importance(),
            }
        out["stage2"][str(h)] = {
            "folds": list(names),
            "order": [c.id for c in simplest_first(hp_cands)],
            "chosen": chosen_c.id,
            "default_kept": chosen_c.id == "c00",
            "steps": step_rows(steps2),
            "candidates": candidate_rows(names, fs, list(cands)),
            "chosen_model": chosen_rows,
        }
        with start(
            tr,
            "fold",
            run_name("fold", "_".join(names), "lightgbm", "chosen", h, fs),
            {
                "horizon": h,
                "folds": ",".join(names),
                "feature_set": fs,
                "rule": "registered ladder",
            },
            tags={"chosen_candidate": chosen_c.id, "chosen_feature_set": fs},
            data_files=data_files,
        ) as srun:
            mlflow.log_dict(out["stage2"][str(h)]["steps"], "ladder_steps.json")
        out["stage2"][str(h)]["summary_run_id"] = srun.info.run_id

    # Rerun check: the chosen 7-day model refitted on f7a gives identical scores.
    c7 = out["stage2"]["7"]["chosen"]
    first = fitted[("f7a", fs, c7)]["p"]
    again = (
        Classifier(fsets[fs], cats, cands[c7], fixed, threads, seed)
        .fit(rows["f7a"]["fit"])
        .predict(rows["f7a"]["score"])[0]
    )
    out["rerun_identical"] = bool(np.array_equal(first, again))
    out["fits"] = len(fitted)
    assert_no_label_building_code()
    return out
