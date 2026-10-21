"""The registry source rebase (configs/serving.yaml, step7.registry_source) and a read-only load.

A version's source is an absolute path on the machine that registered it. The loader
rebases it onto this repository's mlruns/artifacts, refuses anything else, and the eight
SHA-256 checks still decide. The store must load from a read-only copy with no write and
no migration (step7.mlruns_mount): any write or migration stops step 7.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
from pathlib import Path

import mlflow
import pytest

from retention.api import registry as reg
from retention.api.saved import ModelRefused
from test_registry import SETTINGS, P, bundle_dir, keep_tracking_uri, registered  # noqa: F401

RUN = "0123456789abcdef0123456789abcdef"
SCHEMA = "dc11669786a5"  # the alembic head of the installed MLflow 3.17.0 (D-235)


# --- the rule -----------------------------------------------------------------------------------


def test_a_source_under_mlruns_artifacts_is_rebased_onto_this_repository(tmp_path):
    source = f"file:///elsewhere/repo/mlruns/artifacts/{RUN}/artifacts/bundle"
    root = tmp_path / "mlruns" / "artifacts"
    assert reg.rebased_source(source, root) == (root / RUN / "artifacts" / "bundle").as_uri()


def test_on_the_registering_machine_the_rebase_changes_nothing(tmp_path):
    root = tmp_path / "mlruns" / "artifacts"
    source = (root / RUN / "artifacts" / "bundle").as_uri()
    assert reg.rebased_source(source, root) == source


@pytest.mark.parametrize(
    "source",
    [
        f"file:///repo/mlruns/artifacts/{RUN}/artifacts/bundle/../../x",
        f"file:///repo/mlruns/../mlruns/artifacts/{RUN}/artifacts/bundle",
        f"file:///repo/other/artifacts/{RUN}/artifacts/bundle",
        f"file:///repo/mlruns/artifacts/{RUN}/artifacts/elsewhere",
        "file:///repo/mlruns/artifacts/not-a-run-id/artifacts/bundle",
        f"s3://bucket/mlruns/artifacts/{RUN}/artifacts/bundle",
        f"/repo/mlruns/artifacts/{RUN}/artifacts/bundle",
    ],
)
def test_any_other_source_is_refused(tmp_path, source):
    with pytest.raises(ModelRefused):
        reg.rebased_source(source, tmp_path / "mlruns" / "artifacts")


def test_the_marker_is_the_registered_artifact_root():
    s = reg.registry_settings(Path.cwd(), env={})
    assert s.artifact_marker == "mlruns/artifacts" and s.artifact_root.parts[-2:] == (
        "mlruns",
        "artifacts",
    )


# --- a moved registry ---------------------------------------------------------------------------


def moved(tmp_path, bundle_dir) -> reg.RegistrySettings:  # noqa: F811
    """Registered in repo A, copied to repo B, A deleted; settings for B."""
    a, b = tmp_path / "a", tmp_path / "b"
    s, _ = registered(a, bundle_dir)
    shutil.copytree(a, b)
    shutil.rmtree(a)
    return reg.RegistrySettings(
        s.name, s.alias, s.bundle, f"sqlite:///{b / 'mlflow.db'}", b / "mlruns" / "artifacts"
    )


def test_a_moved_registry_loads_from_its_new_place_with_every_check(tmp_path, bundle_dir):  # noqa: F811
    s = moved(tmp_path, bundle_dir)
    v = reg.client_for(s).get_model_version_by_alias(s.name, s.alias)
    assert str(tmp_path / "a") in v.source  # the stored source still names the old place
    models = reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")
    assert models.version == 1 and models.files == dict(sorted(bundle_dir[1].items()))


def test_without_the_rebase_a_moved_registry_is_not_found(tmp_path, bundle_dir, monkeypatch):  # noqa: F811
    s = moved(tmp_path, bundle_dir)
    monkeypatch.setattr(reg, "rebased_source", lambda source, root, marker="": source)
    with pytest.raises(Exception, match="No such artifact"):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


# --- a read-only store: no write, no migration (A(4)) -----------------------------------------


def snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    """Every file under `folder`: its SHA-256 and modification time in nanoseconds."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(folder))] = (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mtime_ns,
            )
    return out


def schema_version(db: Path) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute("select version_num from alembic_version").fetchone()[0]
    finally:
        con.close()


def make_read_only(folder: Path) -> None:
    for p in [folder, *folder.rglob("*")]:
        mode = stat.S_IMODE(p.stat().st_mode)
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def make_writable(folder: Path) -> None:
    for p in [folder, *folder.rglob("*")]:
        p.chmod(stat.S_IMODE(p.stat().st_mode) | stat.S_IWUSR)


def read_only_load(store: Path, settings: reg.RegistrySettings, workdir: Path, model_settings):
    """Load the bundle from a read-only `store`; returns (before, after, models)."""
    make_read_only(store)
    try:
        assert not os.access(store / "mlflow.db", os.W_OK)
        before = snapshot(store)
        models = reg.load_serving_models(settings, model_settings, P, workdir)
        after = snapshot(store)
    finally:
        make_writable(store)
    return before, after, models


def test_a_read_only_store_loads_with_no_write_and_no_migration(tmp_path, bundle_dir):  # noqa: F811
    s = moved(tmp_path, bundle_dir)
    store = tmp_path / "b"
    assert schema_version(store / "mlflow.db") == SCHEMA
    before, after, models = read_only_load(store, s, tmp_path / "load", SETTINGS)
    assert after == before  # no file written, added or touched
    assert schema_version(store / "mlflow.db") == SCHEMA  # no migration
    assert models.files == dict(sorted(bundle_dir[1].items()))


REAL_DB = Path("mlruns/mlflow.db")


@pytest.mark.skipif(not REAL_DB.is_file(), reason="needs the local registry (mlruns/)")
def test_the_real_registry_loads_read_only_from_a_copy(tmp_path, monkeypatch):
    """The registered version 1, copied to another place and made read-only, loads with
    every check, no write and no migration; the copy's files are the ones read."""
    from retention.explain.model import load_settings

    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy("configs/evaluation.yaml", repo / "configs" / "evaluation.yaml")
    shutil.copytree("mlruns", repo / "mlruns")
    s = reg.registry_settings(repo, env={})
    v = reg.client_for(s).get_model_version_by_alias(s.name, s.alias)
    assert str(repo) not in v.source  # the stored source names the original repository
    assert reg.rebased_source(v.source, s.artifact_root, s.artifact_marker).startswith(
        (repo / "mlruns" / "artifacts").as_uri()
    )
    mlflow.set_tracking_uri(s.tracking_uri)
    before, after, models = read_only_load(
        repo / "mlruns", s, tmp_path / "load", load_settings(horizons=(7, 30, 59))
    )
    assert after == before
    assert schema_version(repo / "mlruns" / "mlflow.db") == SCHEMA
    assert models.version == 1 and models.files == dict(sorted(s.bundle.items()))
