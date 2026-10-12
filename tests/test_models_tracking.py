from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from model_fixtures import make_repo, tracking_uri
from retention.models.tracking import (
    AbsolutePathError,
    check_no_absolute_paths,
    run_name,
    setup,
    sha256_file,
    start,
)


@pytest.fixture
def tracked(tmp_path):
    repo = make_repo(tmp_path)
    (tmp_path / "store").mkdir()
    data = tmp_path / "labels.parquet"
    data.write_bytes(b"not really parquet, but hashed the same way")
    return setup(repo, tracking_uri(tmp_path)), data


def logged(tr, run_id):
    return MlflowClient(tr.uri).get_run(run_id).data


def one_run(tr, data, params=None):
    params = params or {"horizon": 7, "fit_snapshots": "2016-08-01", "feature_set": "fs3"}
    name = run_name("fold", "f7a", "lightgbm", "c00", 7, "fs3")
    with start(tr, "fold", name, params, data_files={"labels": data}) as run:
        mlflow.log_metric("auc", 0.75)
    return run.info.run_id


def test_run_name_format():
    assert run_name("fold", "f7a", "lightgbm", "c00", 7, "fs3") == "fold-f7a-lightgbm-c00-h7-fs3"
    with pytest.raises(ValueError):
        run_name("fold", "f 7a", "lightgbm", "c00", 7, "fs3")


def test_a_run_logs_backend_source_hashes_versions_and_resources(tracked):
    tr, data = tracked
    d = logged(tr, one_run(tr, data))
    assert d.tags["run_kind"] == "fold" and d.tags["tracking_backend"] == "sqlite"
    assert d.tags["mlflow.source.name"] == "src/retention/models/__main__.py"
    assert d.tags["sha256.labels"] == sha256_file(data)
    assert d.tags["config_sha256"] == sha256_file(tr.repo / "configs/evaluation.yaml")
    assert len(d.tags["preregistration_commit"]) == 7 and d.tags["git_dirty"] == "false"
    assert float(d.tags["timing.peak_rss_gb"]) > 0 and "version.lightgbm" in d.tags
    assert d.params["horizon"] == "7" and d.metrics["auc"] == 0.75


def test_no_tag_or_parameter_holds_an_absolute_path(tracked):
    tr, data = tracked
    d = logged(tr, one_run(tr, data))
    home = str(Path.home())
    for v in [*d.tags.values(), *d.params.values()]:
        assert "/Users/" not in v and home not in v


def test_an_absolute_path_in_a_parameter_is_refused(tracked):
    tr, data = tracked
    with pytest.raises(AbsolutePathError):
        one_run(tr, data, {"path": str(Path.home() / "x.parquet")})
    with pytest.raises(AbsolutePathError):
        check_no_absolute_paths({"p": "/Users/someone/file"})


def test_two_runs_with_identical_inputs_log_identical_values(tracked):
    tr, data = tracked
    a, b = logged(tr, one_run(tr, data)), logged(tr, one_run(tr, data))
    assert a.params == b.params and a.metrics == b.metrics

    def stable(tags):  # timing.* are measurements of the run itself
        return {k: v for k, v in tags.items() if not k.startswith("timing.")}

    assert stable(a.tags) == stable(b.tags)


def test_a_modified_tracked_file_marks_the_run_dirty(tracked):
    tr, data = tracked
    (tr.repo / "README.md").write_text("changed\n")
    assert logged(tr, one_run(tr, data)).tags["git_dirty"] == "true"


def test_an_unknown_run_kind_is_refused(tracked):
    tr, data = tracked
    with pytest.raises(ValueError, match="unknown run kind"):
        with start(tr, "exploration", "x", {}):
            pass


def test_the_folder_store_is_the_fallback_and_is_tagged(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)  # restored after the test
    repo = make_repo(tmp_path)
    (tmp_path / "a_file").write_text("a file where the store's folder should be\n")
    tr = setup(repo, f"sqlite:///{tmp_path / 'a_file' / 'mlflow.db'}")
    assert tr.backend == "folder"
    data = tmp_path / "d.bin"
    data.write_bytes(b"x")
    assert logged(tr, one_run(tr, data)).tags["tracking_backend"] == "folder"
