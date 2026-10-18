"""The scoring models in the MLflow registry: one version under one alias, checked by hash.

configs/serving.yaml, `registry`, names the registered model (retention-scoring),
the alias (serving) and the eight-file bundle with each file's SHA-256: c01 at 7,
30 and 59 days (booster and model JSON each) and dt01 (booster and model JSON).
The bundle must equal the `ordering.saved_*` registrations and Phase 4's booster
hashes, so the two places that register hashes cannot drift apart.

register: refuses if the registered model already exists (MLflow itself allows an
alias to be set again); checks every source file; copies the eight files and a
manifest.json (the eight hashes, the git head, the config's SHA-256) to a staging
folder; logs them in one run of the `retention-serving` experiment; creates
version 1 from that folder and sets the alias.

load: resolves the alias, downloads the version's files, refuses any missing,
extra or changed file and a manifest whose hashes differ from the bundle, and only
then parses the models. Runs are tagged with repository-relative paths only.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from retention.api.config import load_serving
from retention.api.dt01 import BOOSTER as DT01_BOOSTER
from retention.api.dt01 import META as DT01_META
from retention.api.dt01 import load_dt01
from retention.api.saved import (
    ModelRefused,
    saved_7_30_registration,
    saved_59_registration,
    saved_dt01_registration,
    verify_files,
)
from retention.explain.contributions import SavedModel, load_saved_model
from retention.explain.model import Settings
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import Protocol
from retention.models.tracking import CONFIG as EVAL_CONFIG
from retention.models.tracking import check_no_absolute_paths

EXPERIMENT = "retention-serving"
MANIFEST = "manifest.json"
SOURCE = "src/retention/api/__main__.py"
C01_HORIZONS = (7, 30, 59)


class AlreadyRegistered(RuntimeError):
    """The registered model already exists; registration runs once."""


@dataclass(frozen=True)
class RegistrySettings:
    name: str
    alias: str
    bundle: dict[str, str]
    tracking_uri: str
    artifact_root: Path


def tracking_uri(repo: Path, env: dict[str, str] | None = None) -> str:
    """MLFLOW_TRACKING_URI when set; otherwise the project's SQLite store."""
    env = dict(os.environ) if env is None else env
    if env.get("MLFLOW_TRACKING_URI"):
        return env["MLFLOW_TRACKING_URI"]
    uri = yaml.safe_load((repo / EVAL_CONFIG).read_text())["tracking"]["uri"]
    return "sqlite:///" + str(repo / uri.removeprefix("sqlite:///"))


def registry_settings(
    repo: Path, cfg: dict[str, Any] | None = None, env: dict[str, str] | None = None
) -> RegistrySettings:
    reg = (cfg if cfg is not None else load_serving())["registry"]
    if reg.get("versions_under_alias") != 1:
        raise ValueError("the registry must hold exactly one version under its alias")
    root = yaml.safe_load((repo / EVAL_CONFIG).read_text())["tracking"]["artifact_root"]
    return RegistrySettings(
        name=reg["registered_model"],
        alias=reg["alias"],
        bundle=dict(reg["bundle"]),
        tracking_uri=tracking_uri(repo, env),
        artifact_root=repo / root,
    )


def bundle_sources(settings: RegistrySettings, cfg: dict[str, Any] | None = None) -> dict:
    """{file name: source path}; refuses a bundle that differs from ordering.saved_*."""
    sources, registered = {}, {}
    for directory, files in (
        saved_7_30_registration(cfg),
        saved_59_registration(cfg),
        saved_dt01_registration(cfg),
    ):
        registered |= files
        sources |= {name: directory / name for name in files}
    if registered != settings.bundle:
        raise ValueError("the registry bundle differs from the ordering.saved_* registrations")
    return sources


# --- registration ------------------------------------------------------------------------


def stage_bundle(
    sources: dict[str, Path], expected: dict[str, str], staging: Path, head: str, config_sha: str
) -> dict[str, Any]:
    """Check every source file, copy it to `staging`, write the manifest."""
    for name, path in sources.items():
        verify_files(path.parent, {name: expected[name]})
    staging.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        shutil.copyfile(path, staging / name)
    verify_files(staging, expected)
    manifest = {
        "files": dict(sorted(expected.items())),
        "git_head": head,
        "config_sha256": config_sha,
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def client_for(settings: RegistrySettings) -> MlflowClient:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if not settings.tracking_uri.startswith("sqlite:///"):
        raise ValueError("the model registry needs the SQLite store, not a folder store")
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient()


def registered_model_exists(client: MlflowClient, name: str) -> bool:
    try:
        client.get_registered_model(name)
        return True
    except MlflowException:
        return False


def experiment_id(client: MlflowClient, settings: RegistrySettings) -> str:
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(EXPERIMENT, artifact_location=settings.artifact_root.as_uri())


def register(
    settings: RegistrySettings,
    sources: dict[str, Path],
    staging: Path,
    head: str,
    config_sha: str,
) -> dict[str, Any]:
    """Log the bundle once and set the alias; refuses if the model is already registered."""
    client = client_for(settings)
    if registered_model_exists(client, settings.name):
        raise AlreadyRegistered(f"{settings.name} is already registered; registration runs once")
    manifest = stage_bundle(sources, settings.bundle, staging, head, config_sha)
    tags = {
        "run_kind": "serving_registration",
        "git_commit": head,
        "config_sha256": config_sha,
        "mlflow.source.name": SOURCE,
    }
    check_no_absolute_paths(tags)
    with mlflow.start_run(experiment_id=experiment_id(client, settings), tags=tags) as run:
        mlflow.log_artifacts(str(staging), artifact_path="bundle")
    client.create_registered_model(settings.name)
    version = client.create_model_version(
        settings.name, source=f"{run.info.artifact_uri}/bundle", run_id=run.info.run_id
    )
    client.set_registered_model_alias(settings.name, settings.alias, version.version)
    return {"run_id": run.info.run_id, "version": int(version.version), "manifest": manifest}


# --- loading -----------------------------------------------------------------------------


def download_bundle(settings: RegistrySettings, workdir: Path) -> tuple[Path, Any]:
    """The alias's version downloaded to `workdir`; a missing alias is refused."""
    client = client_for(settings)
    try:
        version = client.get_model_version_by_alias(settings.name, settings.alias)
    except MlflowException as e:
        raise ModelRefused(f"{settings.name}@{settings.alias} is not registered") from e
    workdir.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(workdir))
    return Path(path), version


def verify_bundle(
    folder: Path, settings: RegistrySettings
) -> tuple[dict[str, Any], dict[str, str]]:
    """Every bundle file present and unchanged, nothing extra, manifest equal to the bundle.

    Returns the manifest and each downloaded file's SHA-256.
    """
    found = verify_files(folder, settings.bundle)
    extra = sorted({p.name for p in folder.iterdir()} - set(settings.bundle) - {MANIFEST})
    if extra:
        raise ModelRefused(f"unexpected files in the registered bundle: {extra}")
    if not (folder / MANIFEST).is_file():
        raise ModelRefused(f"{MANIFEST} is missing")
    manifest = json.loads((folder / MANIFEST).read_text())
    if manifest.get("files") != dict(sorted(settings.bundle.items())):
        raise ModelRefused(f"{MANIFEST} lists other hashes than the registered bundle")
    return manifest, found


@dataclass
class ServingModels:
    c01: dict[int, SavedModel]
    dt01: DiscreteTimeModel
    version: int
    manifest: dict[str, Any]
    files: dict[str, str]  # SHA-256 of each downloaded file, after it matched the bundle


def load_serving_models(
    settings: RegistrySettings, model_settings: Settings, protocol: Protocol, workdir: Path
) -> ServingModels:
    """c01 at 7, 30 and 59 days and dt01 from the alias's version, after every check."""
    folder, version = download_bundle(settings, workdir)
    manifest, files = verify_bundle(folder, settings)
    c01 = {
        h: load_saved_model(
            folder, h, model_settings, protocol, settings.bundle[f"booster_{h}d.txt"]
        )
        for h in C01_HORIZONS
    }
    dt01 = load_dt01(
        folder,
        {DT01_BOOSTER: settings.bundle[DT01_BOOSTER], DT01_META: settings.bundle[DT01_META]},
        protocol,
    )
    return ServingModels(c01, dt01, int(version.version), manifest, files)
