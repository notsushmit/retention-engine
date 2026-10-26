"""The real-sample drift report, run on the synthetic tables (nothing here reads the 20% sample).

The same code takes the replay tables, the batch table, the scoring function and the daily counts
as arguments; here they come from the synthetic path. Its numbers must equal the synthetic run's
own, which reads the same rows by another route.
"""

from __future__ import annotations

import json
from datetime import date

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from ops_group import needs_ops_group
from retention.api import __main__ as api_cli
from retention.api import ops as api_ops
from retention.ops import __main__ as ops_cli
from retention.ops import drift as dr
from retention.ops import real_drift as rd
from retention.synthetic import drift as sd
from retention.synthetic.pipeline import fit_models
from test_ops_drift import synthetic_run  # noqa: F401 (fixture)

MAIN = ("paid", "discount100_only")


def snapshot(setup, day: str, features=None) -> pa.Table:
    path = features or setup.processed / "features.parquet"
    return ds.dataset(path).to_table(filter=pc.field("snapshot_date") == date.fromisoformat(day))


@pytest.fixture(scope="module")
def real_run(synthetic_run, tmp_path_factory):  # noqa: F811
    setup, synth, out = synthetic_run
    models = fit_models(setup)
    cfg = setup.raw["drift"]
    work = tmp_path_factory.mktemp("real_drift")
    dr.log_defaults(work)
    result = rd.run(
        fit_tables=[snapshot(setup, str(d)) for d in cfg["reference_fit_snapshots"]],
        prediction_table=snapshot(setup, str(cfg["reference_prediction_snapshot"])),
        unshifted_table=snapshot(setup, str(cfg["current_snapshot"])),
        shifted_table=snapshot(
            setup, str(cfg["current_snapshot"]), out / "synthetic_features_shifted.parquet"
        ),
        predict=lambda rows: sd.final_probabilities(models, rows),
        window=30,
        main=MAIN,
        batch_reference=sd.daily_events(setup, str(cfg["batch_reference_month"])),
        batch_current=sd.daily_events(setup, str(cfg["batch_current_month"])),
        out_dir=work,
        store_date=str(cfg["current_snapshot"]),
        replays={"unshifted": {"database": 1}, "shifted": {"database": 2}},
    )
    return setup, synth, result, work


@needs_ops_group
def test_both_replays_give_the_numbers_of_the_synthetic_run_read_by_another_route(real_run):
    _, synth, result, _ = real_run
    for name in ("unshifted", "shifted"):
        assert result[name]["drift_share"] == pytest.approx(synth[name]["drift_share"], abs=0.03)
        assert result[name]["columns_compared"] == 47
    assert result["margin"]["required"] == 0.20


@needs_ops_group
def test_the_report_gives_the_registered_result_and_the_group_mark_side_by_side(real_run):
    _, _, result, work = real_run
    both = json.loads((work / "drift_real_report.json").read_text())
    reg, grp = both["dataset_level_as_registered"], both["group_level"]
    assert reg["met"] == (reg["margin"]["margin_met"] and reg["margin"]["shifted_alert"])
    assert grp["group"] == "usage" and grp["group_columns"] == 23 and grp["unvalidated"] is True
    assert grp["label"] == "Group-level mark: added after a result was seen; unvalidated."
    assert grp["share"] == grp["drifted_in_group"] / 23 and grp["rule_share"] == 0.5
    assert grp["group_rule_met"] is (grp["share"] >= 0.5)
    assert grp["met"] is (grp["group_rule_met"] and grp["margin_rule_met"])
    assert both["label"] == dr.LABEL
    # neither replaces the other: each is present and computed from the same two reports
    assert grp["margin_rule_met"] == reg["margin"]["margin_met"]


@needs_ops_group
def test_the_group_drifted_columns_are_usage_columns_only(real_run):
    _, _, result, _ = real_run
    assert set(result["group"]["drifted_columns"]) <= set(dr.usage_columns())


@needs_ops_group
def test_the_file_the_api_serves_is_not_labelled_synthetic(real_run):
    _, _, result, work = real_run
    body = api_ops.drift_body(work)
    assert body["status"] == "monitored" and body["synthetic"] is False
    assert body["label"] == dr.LABEL
    assert body["columns_compared"] == {"count": 47, "hidden_by_rule": 0}
    assert body["current"]["replays"] == {"unshifted": {"database": 1}, "shifted": {"database": 2}}


@needs_ops_group
def test_the_group_mark_is_a_function_of_the_two_reports_only():
    usage = dr.usage_columns()

    def report(drifted: int, share: float):
        cols = [{"column": c, "drifted": i < drifted} for i, c in enumerate(usage)]
        return {"columns": cols, "drift_share": share, "dataset_alert": share >= 0.5}

    low = dr.group_mark(report(0, 0.0), report(11, 0.40))  # 11 of 23 = 0.478
    assert low["group_rule_met"] is False and low["met"] is False
    high = dr.group_mark(report(0, 0.0), report(12, 0.30))  # 12 of 23 = 0.5217, margin 0.30
    assert high["group_rule_met"] is True and high["met"] is True
    no_margin = dr.group_mark(report(0, 0.30), report(12, 0.40))  # margin only 0.10
    assert no_margin["group_rule_met"] is True and no_margin["met"] is False


def test_the_usage_group_has_23_columns_among_the_47():
    assert len(dr.usage_columns()) == 23


@needs_ops_group
def test_the_report_refuses_to_run_before_the_defaults_are_logged(tmp_path):
    with pytest.raises(RuntimeError, match="log the library defaults first"):
        rd.run(
            fit_tables=[], prediction_table=None, unshifted_table=None, shifted_table=None,
            predict=None, window=30, main=MAIN, batch_reference=None, batch_current=None,
            out_dir=tmp_path, store_date="2016-11-01", replays={},
        )  # fmt: skip


# --- the command -----------------------------------------------------------------------------


class Args:
    fraction = None
    defaults = False


def test_the_command_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        ops_cli.report_cmd(Args(), gather=lambda a: pytest.fail("gathered"))


@needs_ops_group
def test_the_defaults_flag_writes_the_defaults_and_nothing_else(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ops_cli, "ops_dir", lambda fraction: tmp_path)
    a = Args()
    a.defaults = True
    assert ops_cli.report_cmd(a, gather=lambda x: pytest.fail("gathered")) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [dr.DEFAULTS_FILE]


def test_the_command_runs_once(monkeypatch, tmp_path):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(ops_cli, "ops_dir", lambda fraction: tmp_path)
    (tmp_path / ops_cli.REPORT).write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        ops_cli.report_cmd(Args(), gather=lambda a: pytest.fail("gathered"))


def test_the_command_needs_both_replay_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(ops_cli, "ops_dir", lambda fraction: tmp_path / "ops")
    monkeypatch.setattr(api_cli, "out_dir", lambda fraction: tmp_path / "serving")
    with pytest.raises(FileNotFoundError):
        ops_cli.report_cmd(Args(), gather=lambda a: pytest.fail("gathered"))


def test_the_registered_snapshots_and_months():
    assert ops_cli.FIT_SNAPSHOTS == ("2016-08-01", "2016-09-01")
    assert ops_cli.PREDICTION_SNAPSHOT == "2016-10-01"
    assert (ops_cli.BATCH_REFERENCE_MONTH, ops_cli.BATCH_CURRENT_MONTH) == ("2016-09", "2016-10")
    assert pq  # the report reads parquet tables only; no Valkey client is imported
    import inspect

    assert "redis" not in inspect.getsource(ops_cli) and "valkey" not in inspect.getsource(rd)
