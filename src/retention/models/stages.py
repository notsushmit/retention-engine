"""Fold stages. Each runs in a model process that reads cached label builds only.

baseline_stage: every simple baseline on f7a, f7b (7 days) and f30 (30 days):
each fold's AUC with its customer-bootstrap interval, the fold-mean AUC per
horizon with its joint interval, the strongest simple baseline per horizon
(highest fold-mean AUC, as registered), and each baseline's paired AUC
difference to the strongest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml

from retention.models.baselines import INPUTS, NAMES, Baselines
from retention.models.bootstrap import FoldScores, bootstrap, paired_auc_difference
from retention.models.frames import fit_rows, score_rows
from retention.models.metrics import NOT_COMPUTABLE, SortedScores
from retention.models.protocol import EVAL_CONFIG, Protocol
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
