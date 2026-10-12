"""The test guard in a temporary git repository: every refusal, and the one success."""

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient

from model_fixtures import commit, make_repo, sh, tracking_uri
from retention.models.guard import CHOICES, GuardRefusal, dry_run, freeze, open_test_session
from retention.models.protocol import TestUnlock, load_protocol
from retention.models.tracking import CONFIG, setup, start

CHOICES_DOC = {
    "feature_set": "fs3",
    "classifier_hyperparameters": {7: "c00", 30: "c00", 59: "c00"},
    "survival_candidate": {"cox": "cox01", "xgboost_aft": "aft01", "discrete_time": "dt01"},
    "survival_family": "cox",
    "calibration_method": {7: "platt", 30: "platt", 59: "platt"},
}


def fold_run(tr, name="fold-f7a-lightgbm-c00-h7-fs3"):
    with start(tr, "fold", name, {"horizon": 7}) as run:
        mlflow.log_metric("auc", 0.7)
    return run.info.run_id


@pytest.fixture
def env(tmp_path):
    repo = make_repo(tmp_path)
    (tmp_path / "store").mkdir()
    tr = setup(repo, tracking_uri(tmp_path))
    return tr, {"feature_set": fold_run(tr), "survival_family": fold_run(tr)}


def frozen(tr, runs, **kw):
    freeze(tr, CHOICES_DOC, runs, **kw)
    return commit(tr.repo, "Freeze choices")


def edit_config(tr, line):
    path = tr.repo / CONFIG
    path.write_text(path.read_text() + f"# {line}\n")


def refused(tr, check, **kw):
    with pytest.raises(GuardRefusal, match=f"check {check}:"):
        open_test_session(tr, **kw)


# --- success, exactly once -------------------------------------------------------------


def test_success_unlocks_once_records_the_run_and_a_second_attempt_refuses(env):
    tr, runs = env
    frozen(tr, runs)
    unlock, run_id = open_test_session(tr)
    assert isinstance(unlock, TestUnlock)
    assert load_protocol().test_months(unlock).months[30]  # the months are now readable
    tags = MlflowClient(tr.uri).get_run(run_id).data.tags
    assert tags["run_kind"] == "test" and tags["unlock_issued"] == "true"
    refused(tr, 9)


def test_unlock_still_cannot_be_constructed_anywhere_else(env):
    with pytest.raises(PermissionError):
        TestUnlock()


# --- each refusal ------------------------------------------------------------------------


def test_refuses_a_dirty_tree(env):
    tr, runs = env
    frozen(tr, runs)
    (tr.repo / "README.md").write_text("changed\n")
    refused(tr, 1)


def test_refuses_choices_that_are_not_committed(env):
    tr, runs = env
    freeze(tr, CHOICES_DOC, runs)  # written but not committed (untracked)
    refused(tr, 2)


def hand_edit(tr, change):
    path = tr.repo / CHOICES
    doc = yaml.safe_load(path.read_text())
    change(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    commit(tr.repo, "Edit choices by hand")


def test_refuses_choices_that_are_not_frozen(env):
    tr, runs = env
    frozen(tr, runs)
    hand_edit(tr, lambda d: d.update(frozen=False))
    refused(tr, 2)


def test_refuses_incomplete_choices(env):
    tr, runs = env
    frozen(tr, runs)
    hand_edit(tr, lambda d: d["choices"].pop("survival_family"))
    refused(tr, 2)


def test_freeze_itself_refuses_incomplete_choices(env):
    tr, runs = env
    with pytest.raises(ValueError, match="no choice"):
        freeze(tr, {**CHOICES_DOC, "calibration_method": None}, runs)


def test_refuses_a_hand_edited_choices_file(env):
    tr, runs = env
    frozen(tr, runs)
    hand_edit(tr, lambda d: d["choices"].update(feature_set="fs1"))
    refused(tr, 3)


def test_refuses_a_binding_commit_that_is_not_an_ancestor(env):
    tr, runs = env
    sh(tr.repo, "checkout", "-q", "-b", "side")
    edit_config(tr, "on a side branch")
    side = commit(tr.repo, "Side change")
    sh(tr.repo, "checkout", "-q", "main")
    frozen(tr, runs, binding_commit=side)
    refused(tr, 4)


def test_refuses_when_the_binding_blob_has_another_hash(env):
    tr, runs = env
    first = sh(tr.repo, "rev-parse", "--short=7", "HEAD")
    edit_config(tr, "second version")
    commit(tr.repo, "Change config")
    frozen(tr, runs, binding_commit=first)  # records the hash of the second version
    refused(tr, 5)


def test_refuses_when_the_working_tree_config_has_another_hash(env):
    tr, runs = env
    frozen(tr, runs)
    edit_config(tr, "changed after the freeze")
    commit(tr.repo, "Change config after freeze")
    refused(tr, 6)


def test_refuses_a_later_commit_that_changed_the_config_even_if_reverted(env):
    tr, runs = env
    frozen(tr, runs)
    original = (tr.repo / CONFIG).read_text()
    edit_config(tr, "temporary change")
    commit(tr.repo, "Change config")
    (tr.repo / CONFIG).write_text(original)
    commit(tr.repo, "Revert config")
    refused(tr, 7)


def test_refuses_a_cited_run_that_does_not_exist(env):
    tr, runs = env
    frozen(tr, {**runs, "survival_family": "0" * 32})
    refused(tr, 8)


def test_refuses_a_cited_run_made_under_another_config(env):
    tr, runs = env
    MlflowClient(tr.uri).set_tag(runs["feature_set"], "config_sha256", "f" * 64)
    frozen(tr, runs)
    refused(tr, 8)


def test_refuses_when_a_finished_test_run_exists(env):
    tr, runs = env
    frozen(tr, runs)
    with start(tr, "test", "test", {}):
        pass
    refused(tr, 9)


def test_a_crashed_test_run_may_be_repeated_only_with_a_reason(env):
    tr, runs = env
    frozen(tr, runs)
    client = MlflowClient(tr.uri)
    crashed = client.create_run(tr.experiment_id, tags={"run_kind": "test"})
    client.set_terminated(crashed.info.run_id, "FAILED")
    refused(tr, 9)
    unlock, run_id = open_test_session(tr, rerun_reason="process killed by the OS")
    tags = client.get_run(run_id).data.tags
    assert tags["rerun_reason"] == "process killed by the OS"
    assert tags["rerun_of"] == crashed.info.run_id


# --- later non-config commits and the dry run ------------------------------------------------


def test_a_later_commit_that_changes_only_other_files_does_not_trip_check_7(env):
    tr, runs = env
    frozen(tr, runs)
    workflow = tr.repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: ci\n")
    commit(tr.repo, "Change CI only")
    unlock, _ = open_test_session(tr)
    assert isinstance(unlock, TestUnlock)


def test_dry_run_before_the_freeze_passes_and_records_nothing(env):
    tr, _ = env
    results = dry_run(tr)
    assert [r.check for r in results] == list(range(1, 10))
    assert {r.check: r.status for r in results if r.status != "passed"} == {
        2: "needs choices",
        3: "needs choices",
        8: "needs choices",
    }
    assert not MlflowClient(tr.uri).search_runs(
        [tr.experiment_id], filter_string="tags.run_kind = 'test'"
    )


def test_dry_run_after_the_freeze_passes_every_check(env):
    tr, runs = env
    frozen(tr, runs)
    assert {r.status for r in dry_run(tr)} == {"passed"}


def test_dry_run_reports_a_failure_without_raising(env):
    tr, _ = env
    (tr.repo / "README.md").write_text("changed\n")
    status = {r.check: r.status for r in dry_run(tr)}
    assert status[1] == "failed" and status[4] == "passed"
