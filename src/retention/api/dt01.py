"""dt01 for the service: refit once, saved with its model JSON, rebuilt from the files.

Phase 7 refit dt01 (configs/actions.yaml, dt01_refit) and wrote only the booster
text (data/interim/<sample>/optimize/model/dt01.txt, SHA-256 6d2edd15...a7338d);
it scored dt01 in the process that fitted it, so the category levels and columns
were never written and the file alone cannot be scored.

save-dt01 refits dt01 exactly as Phase 7 did (optimize.value.refit_dt01) and
judges it by Phase 7's registered reproduction cells (rehearsal fingerprints at 7,
30 and 59 days on 2016-11-01, exact) and by the expected booster SHA-256, which
must also equal Phase 7's saved file. Only then does it write dt01.txt and
model_dt01.json (columns, category levels, interval ends, settings) to the
service's model folder. Phase 7's folder is not touched.

load_dt01 rebuilds the scoring object from the two files, after both hashes pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

from retention.api.config import file_sha256
from retention.api.saved import verify_files
from retention.explain.model import library_versions, p2_main_rows
from retention.models.classifier import Classifier
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import Protocol
from retention.optimize.config import ActionsConfig
from retention.optimize.value import Dt01Settings, refit_dt01, reproduce

BOOSTER = "dt01.txt"
META = "model_dt01.json"


def booster_text(model: DiscreteTimeModel) -> str:
    return model.model.booster.model_to_string()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def meta_of(model: DiscreteTimeModel, s: Dt01Settings, sha: str) -> dict[str, Any]:
    """Everything needed to score the booster again, beside its hash."""
    return {
        "model": "dt01",
        "booster_sha256": sha,
        "feature_columns": list(s.columns),
        "columns": list(model.columns),
        "categorical": list(model.categorical),
        "categorical_levels": {c: model.model.levels[c] for c in model.model.cats},
        "ends": list(model.ends),
        "offset_input": bool(model.offset_input),
        "candidate": dict(model.candidate),
        "fixed": dict(model.fixed),
    }


def run_save_check(
    protocol: Protocol,
    processed: Path,
    cache: Path,
    s: Dt01Settings,
    cfg: ActionsConfig,
    expected_sha: str,
    phase7_booster: Path,
) -> tuple[dict[str, Any], DiscreteTimeModel]:
    """Refit, reproduce, compare hashes; returns the report and the model. Writes nothing."""
    model, n_fit = refit_dt01(protocol, processed, cache, s, cfg)
    snapshots = sorted({t for _, t, _ in cfg.reproduction_cells})
    rows = {t: p2_main_rows(processed, t, s.columns, protocol) for t in snapshots}
    rehearsal = json.loads((processed / "rehearsal.json").read_text())
    table, failures = reproduce(model, rehearsal, rows, cfg)
    sha = text_sha256(booster_text(model))
    phase7_sha = file_sha256(phase7_booster) if phase7_booster.is_file() else None
    if sha != expected_sha:
        failures.append(f"booster SHA-256 {sha} differs from --expect-sha {expected_sha}")
    if phase7_sha != expected_sha:
        failures.append(f"Phase 7's {phase7_booster} has SHA-256 {phase7_sha}, not {expected_sha}")
    report = {
        "step": "save-dt01",
        "fit_rows": n_fit,
        "expanded_rows": model.expanded_rows,
        "reproduction": table,
        "booster_sha256": sha,
        "phase7_booster_sha256": phase7_sha,
        "expected_sha256": expected_sha,
        "failures": failures,
        "passed": not failures,
        "library_versions": library_versions(),
    }
    return report, model


def save(model: DiscreteTimeModel, s: Dt01Settings, out_dir: Path, expected_sha: str) -> dict:
    """Write dt01.txt and model_dt01.json; refuses a booster whose hash differs."""
    text = booster_text(model)
    sha = text_sha256(text)
    if sha != expected_sha:
        raise ValueError(f"booster SHA-256 {sha} differs from the expected {expected_sha}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BOOSTER).write_text(text)
    (out_dir / META).write_text(json.dumps(meta_of(model, s, sha), indent=2) + "\n")
    return {BOOSTER: file_sha256(out_dir / BOOSTER), META: file_sha256(out_dir / META)}


def load_dt01(directory: Path, expected: dict[str, str], protocol: Protocol) -> DiscreteTimeModel:
    """dt01 rebuilt from its two files, after both match `expected` ({file name: SHA-256})."""
    if sorted(expected) != sorted([BOOSTER, META]):
        raise ValueError(f"expected hashes must name exactly {BOOSTER} and {META}")
    verify_files(directory, expected)
    meta = json.loads((directory / META).read_text())
    if meta["booster_sha256"] != expected[BOOSTER]:
        raise ValueError("model_dt01.json names another booster")
    model = DiscreteTimeModel(
        meta["offset_input"],
        list(meta["ends"]),
        dict(meta["candidate"]),
        dict(meta["fixed"]),
        list(meta["categorical"]),
        protocol.threads,
        protocol.seed,
    )
    model.columns = list(meta["columns"])
    clf = Classifier(
        model.columns, model.categorical, model.candidate, model.fixed, model.threads, model.seed
    )
    clf.cats = [c for c in model.categorical if c in model.columns]
    clf.levels = {c: meta["categorical_levels"][c] for c in clf.cats}
    clf.booster = lgb.Booster(model_str=(directory / BOOSTER).read_text())
    model.model = clf
    return model
