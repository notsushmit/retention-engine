"""Assemble the frozen choices from the MLflow runs that made them, and check them.

Every choice is read from the run that recorded it (by exact run name; a run
tagged `invalid` is never cited; a name with no valid run, or more than one,
stops the freeze). Checks before writing: every registered stage cites at
least one run; no cited run is tagged invalid; every cited run was made under
the binding config hash; the choices agree with the tags of the runs cited.
"""

from __future__ import annotations

from typing import Any

import yaml
from mlflow.tracking import MlflowClient

from retention.models.guard import HEADER, blob_sha256, required_stages
from retention.models.report import DESCRIPTIVE_BREAKDOWNS, DESCRIPTIVE_MODELS
from retention.models.tracking import CONFIG, Tracking, config_commits, sha256_file

FS = "fs1"
STAGE1 = [
    f"fold-{f}-lightgbm-c00-h{h}-{fs}"
    for fs in ("fs1", "fs2", "fs3")
    for f, h in (("f7a", 7), ("f7b", 7), ("f30", 30))
]
NAMES = {
    "feature_set": STAGE1,
    "classifier_hyperparameters": [
        "fold-f7a_f7b-lightgbm-chosen-h7-fs1",
        "fold-f30-lightgbm-chosen-h30-fs1",
    ],
    "survival_candidate": ["fold-f7a_f7b_f30-survival-chosen-h7-fs1"],
    "survival_family": ["fold-f7a_f7b_f30-survival-chosen-h7-fs1"],
    "calibration_method": [
        "calibration_fold-f7b_cal-both-platt_isotonic-h7-fs1",
        "calibration_fold-2016-10-both-platt-h7-fs1",
    ],
    "strongest_simple_baseline": [
        "fold-f7a_f7b-baseline-strongest-h7-baseline_inputs",
        "fold-f30-baseline-strongest-h30-baseline_inputs",
    ],
    "descriptive_models": ["fold-f7a_f7b_f30-survival-chosen-h7-fs1"],
}


def find_run(client: MlflowClient, tr: Tracking, name: str):
    runs = client.search_runs(
        [tr.experiment_id], filter_string=f"attributes.run_name = '{name}'", max_results=50
    )
    valid = [r for r in runs if "invalid" not in r.data.tags and r.info.status == "FINISHED"]
    if len(valid) != 1:
        raise ValueError(f"run {name!r}: {len(valid)} valid finished runs, expected exactly 1")
    return valid[0]


def assemble(tr: Tracking) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, bool, str]]]:
    client = MlflowClient(tr.uri)
    runs = {name: find_run(client, tr, name) for names in NAMES.values() for name in names}
    tags = {name: r.data.tags for name, r in runs.items()}
    h7, h30 = (tags[n] for n in NAMES["classifier_hyperparameters"])
    surv = tags[NAMES["survival_family"][0]]
    b7, b30 = (tags[n] for n in NAMES["strongest_simple_baseline"])
    choices = {
        "feature_set": h7["chosen_feature_set"],
        "classifier_hyperparameters": {
            7: h7["chosen_candidate"],
            30: h30["chosen_candidate"],
            59: h30["chosen_candidate"],  # the 59-day demonstration inherits the 30-day choice
        },
        "survival_candidate": {
            f: surv[f"chosen.{f}"] for f in ("cox", "xgboost_aft", "discrete_time")
        },
        "survival_family": {
            "family": surv["chosen_survival_family"],
            "candidate": surv["chosen_survival_candidate"],
        },
        "calibration_method": {
            7: "platt",
            30: "platt",
            59: "platt",
            "note": (
                "Platt at every horizon on test, as registered: isotonic fails the 1,000-event "
                "floor at 7 days (976) and has no calibration fold at 30 or 59 days"
            ),
        },
        "strongest_simple_baseline": {
            7: b7["strongest_simple_baseline"],
            30: b30["strongest_simple_baseline"],
            59: b30["strongest_simple_baseline"],  # inherits the 30-day choice
        },
        "descriptive_breakdowns": DESCRIPTIVE_BREAKDOWNS,
        "descriptive_models": DESCRIPTIVE_MODELS,
    }
    run_ids = {stage: [runs[n].info.run_id for n in names] for stage, names in NAMES.items()}

    binding = config_commits(tr.repo)[-1]
    binding_hash = blob_sha256(tr.repo, binding, CONFIG)
    checks = []
    missing = [s for s in required_stages(tr.repo) if not run_ids.get(s)]
    checks.append(
        ("every registered stage cites at least one run", not missing, f"missing: {missing}")
    )
    invalid = [n for n, t in tags.items() if "invalid" in t]
    checks.append(("no cited run is tagged invalid", not invalid, f"invalid: {invalid}"))
    other = [n for n, t in tags.items() if t.get("config_sha256") != binding_hash]
    checks.append(
        (
            "every cited run was made under the binding config hash",
            not other,
            f"other hash: {other}",
        )
    )
    checks.append(
        (
            "working-tree config equals the binding blob",
            sha256_file(tr.repo / CONFIG) == binding_hash,
            "",
        )
    )
    agree = (
        choices["feature_set"] == h30["chosen_feature_set"] == FS
        and choices["survival_family"]["candidate"]
        == choices["survival_candidate"]["discrete_time"]
    )
    checks.append(("choices agree with the cited runs' tags", agree, ""))
    return choices, run_ids, checks


def preview(tr: Tracking, choices: dict[str, Any], run_ids: dict[str, Any]) -> str:
    """The text freeze() would write, with the freeze run id still to be assigned."""
    commits = config_commits(tr.repo)
    doc = {
        "frozen": True,
        "preregistration_commits": commits,
        "binding_commit": commits[-1],
        "evaluation_config_sha256": sha256_file(tr.repo / CONFIG),
        "freeze_run_id": "<assigned when the freeze run is created>",
        "choices": choices,
        "run_ids": run_ids,
    }
    return HEADER + yaml.safe_dump(doc, sort_keys=False)
