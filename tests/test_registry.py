"""The model registry: one version under one alias; every file checked at register and load."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from retention.api import __main__ as cli
from retention.api import dt01 as d
from retention.api import registry as reg
from retention.api.config import file_sha256, load_serving
from retention.api.saved import ModelRefused
from retention.explain.model import Refit, Settings, booster_sha256, fit_models, save_models
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import load_protocol
from retention.models.tracking import check_no_absolute_paths
from retention.optimize.value import INTERVAL_ENDS_ALL
from test_save_dt01 import COLS, RAW, S, rows

P = load_protocol()
REPO = Path.cwd()
SMALL = {"num_leaves": 7, "n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 20}
SETTINGS = Settings(
    columns=["a", "b", "cat"],
    categorical=["cat"],
    fixed=yaml.safe_load(Path("configs/evaluation.yaml").read_text())["classifier"]["fixed_params"],
    candidates={h: SMALL for h in (7, 30, 59)},
)


def synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["x", "y", "z"], size=n)
    z = 1.2 * a - 0.8 * b + np.where(cat == "y", 1.0, 0.0) - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "cat": pd.array(cat, dtype="string"), "y": y})


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """Eight hand-made files: c01 at 7, 30 and 59 days and a small dt01."""
    out = tmp_path_factory.mktemp("bundle_src")
    for k, h in enumerate((7, 30, 59)):
        model, platt = fit_models(synthetic(2000, k), synthetic(800, 10 + k), SETTINGS, h, P)
        save_models(out, {h: Refit(h, model, platt, 2000, 800)}, {h: booster_sha256(model)})
    dt = DiscreteTimeModel(
        False,
        list(INTERVAL_ENDS_ALL),
        SMALL,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(rows(), COLS)
    d.save(dt, S, out, d.text_sha256(d.booster_text(dt)))
    names = [f"booster_{h}d.txt" for h in (7, 30, 59)] + [f"model_{h}d.json" for h in (7, 30, 59)]
    names += [d.BOOSTER, d.META]
    hashes = {n: file_sha256(out / n) for n in names}
    return out, hashes


@pytest.fixture(autouse=True)
def keep_tracking_uri():
    before = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(before)


def settings_for(tmp: Path, hashes: dict[str, str]) -> reg.RegistrySettings:
    return reg.RegistrySettings(
        name="test-scoring",
        alias="serving",
        bundle=dict(hashes),
        tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
        artifact_root=tmp / "mlruns" / "artifacts",  # the layout step7.registry_source accepts
    )


def registered(tmp: Path, bundle_dir) -> tuple[reg.RegistrySettings, dict]:
    src, hashes = bundle_dir
    s = settings_for(tmp, hashes)
    done = reg.register(s, {n: src / n for n in hashes}, tmp / "staging", "abc1234", "f" * 64)
    return s, done


def store_folder(s: reg.RegistrySettings) -> Path:
    v = reg.client_for(s).get_model_version_by_alias(s.name, s.alias)
    return Path(urlparse(v.source).path)


# --- settings ---------------------------------------------------------------------------------


def test_bundle_equals_the_ordering_registrations():
    s = reg.registry_settings(REPO, env={})
    sources = reg.bundle_sources(s)
    assert sorted(sources) == sorted(s.bundle) and len(s.bundle) == 8
    assert (s.name, s.alias) == ("retention-scoring", "serving")


def test_a_bundle_differing_from_the_ordering_registrations_is_refused():
    cfg = copy.deepcopy(load_serving())
    cfg["registry"]["bundle"]["dt01.txt"] = "0" * 64
    s = reg.registry_settings(REPO, cfg, env={})
    with pytest.raises(ValueError, match="differs"):
        reg.bundle_sources(s, cfg)


def test_tracking_uri_comes_from_the_environment_else_the_project_store():
    assert reg.tracking_uri(REPO, {"MLFLOW_TRACKING_URI": "sqlite:///x.db"}) == "sqlite:///x.db"
    default = reg.tracking_uri(REPO, {})
    assert default == "sqlite:///" + str(REPO / "mlruns/mlflow.db")


def test_a_folder_store_is_refused(tmp_path, bundle_dir):
    s = settings_for(tmp_path, bundle_dir[1])
    s = reg.RegistrySettings(s.name, s.alias, s.bundle, tmp_path.as_uri(), s.artifact_root)
    with pytest.raises(ValueError, match="SQLite"):
        reg.client_for(s)


# --- register and load --------------------------------------------------------------------------


def test_register_sets_the_alias_on_version_1_and_loads_back(tmp_path, bundle_dir):
    s, done = registered(tmp_path, bundle_dir)
    assert done["version"] == 1
    assert done["manifest"] == {
        "files": dict(sorted(bundle_dir[1].items())),
        "git_head": "abc1234",
        "config_sha256": "f" * 64,
    }
    models = reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")
    assert models.version == 1 and sorted(models.c01) == [7, 30, 59]
    for h, m in models.c01.items():
        assert m.sha256 == bundle_dir[1][f"booster_{h}d.txt"]
    r = rows(200, seed=3)
    original = d.load_dt01(bundle_dir[0], {n: bundle_dir[1][n] for n in (d.BOOSTER, d.META)}, P)
    a, _ = original.lapse_probability(r, list(INTERVAL_ENDS_ALL))
    b, _ = models.dt01.lapse_probability(r, list(INTERVAL_ENDS_ALL))
    assert np.array_equal(a, b)


def test_loaded_summary_records_every_downloaded_file_hash(tmp_path, bundle_dir):
    s, _ = registered(tmp_path, bundle_dir)
    summary = cli.loaded_summary(reg.load_serving_models(s, SETTINGS, P, tmp_path / "load"))
    assert summary["downloaded_files_sha256"] == dict(sorted(bundle_dir[1].items()))
    assert len(summary["downloaded_files_sha256"]) == 8


def test_a_second_registration_is_refused_and_creates_no_version(tmp_path, bundle_dir):
    s, _ = registered(tmp_path, bundle_dir)
    src, hashes = bundle_dir
    with pytest.raises(reg.AlreadyRegistered):
        reg.register(s, {n: src / n for n in hashes}, tmp_path / "staging2", "abc1234", "f" * 64)
    versions = reg.client_for(s).search_model_versions(f"name='{s.name}'")
    assert [int(v.version) for v in versions] == [1]


def test_register_refuses_a_changed_source_and_registers_nothing(tmp_path, bundle_dir):
    src, hashes = bundle_dir
    copy_dir = tmp_path / "src"
    shutil.copytree(src, copy_dir)
    data = bytearray((copy_dir / "booster_59d.txt").read_bytes())
    data[-2] ^= 1
    (copy_dir / "booster_59d.txt").write_bytes(bytes(data))
    s = settings_for(tmp_path, hashes)
    with pytest.raises(ModelRefused, match="booster_59d"):
        reg.register(s, {n: copy_dir / n for n in hashes}, tmp_path / "staging", "abc", "f" * 64)
    assert not reg.registered_model_exists(reg.client_for(s), s.name)


def test_load_refuses_a_missing_alias(tmp_path, bundle_dir):
    s = settings_for(tmp_path, bundle_dir[1])
    with pytest.raises(ModelRefused, match="not registered"):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


@pytest.mark.parametrize("name", ["booster_7d.txt", "model_30d.json", d.BOOSTER, d.META])
def test_load_refuses_a_changed_file_in_the_store(tmp_path, bundle_dir, name):
    s, _ = registered(tmp_path, bundle_dir)
    path = store_folder(s) / name
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ModelRefused, match=name.replace(".", r"\.")):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


def test_load_refuses_a_missing_file_in_the_store(tmp_path, bundle_dir):
    s, _ = registered(tmp_path, bundle_dir)
    (store_folder(s) / "model_59d.json").unlink()
    with pytest.raises(ModelRefused, match="missing"):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


def test_load_refuses_an_extra_file_in_the_store(tmp_path, bundle_dir):
    s, _ = registered(tmp_path, bundle_dir)
    (store_folder(s) / "booster_90d.txt").write_text("extra\n")
    with pytest.raises(ModelRefused, match="unexpected"):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


def test_load_refuses_a_manifest_listing_other_hashes(tmp_path, bundle_dir):
    s, _ = registered(tmp_path, bundle_dir)
    path = store_folder(s) / reg.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["files"]["dt01.txt"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ModelRefused, match="manifest"):
        reg.load_serving_models(s, SETTINGS, P, tmp_path / "load")


def test_registration_run_tags_hold_no_absolute_path(tmp_path, bundle_dir):
    s, done = registered(tmp_path, bundle_dir)
    tags = MlflowClient().get_run(done["run_id"]).data.tags
    check_no_absolute_paths(tags)
    assert tags["mlflow.source.name"] == reg.SOURCE
    assert tags["run_kind"] == "serving_registration"


# --- the commands' guards -------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["register", "verify-serving"])
def test_commands_refuse_a_dirty_tree(monkeypatch, command):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main([command])


@pytest.mark.parametrize(
    ("command", "report"),
    [("register", cli.REGISTER_REPORT), ("verify-serving", cli.VERIFY_REPORT)],
)
def test_commands_run_once(monkeypatch, tmp_path, command, report):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    (tmp_path / report).write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        cli.main([command])
