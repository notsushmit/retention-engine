"""score-parity on hand-made data: exact comparison, each judge, the stop rules, the guards."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from retention.api import __main__ as cli
from retention.api import parity as pa
from retention.api import scoring as sc
from retention.api.config import load_serving
from test_scoring_action import ACT, table

SERVING = load_serving()
OCT, NOV = date(2016, 10, 1), date(2016, 11, 1)


# --- exact comparison -------------------------------------------------------------------


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["b", "a", "c"],
            "x": [0.1, 0.2, np.nan],
            "s": ["u", None, "w"],
            "d": [NOV, NOV, NOV],
        }
    )


def test_identical_frames_compare_equal_in_any_row_order():
    a = frame()
    b = a.iloc[[2, 0, 1]].reset_index(drop=True)
    assert pa.compare_frames(a, b)["equal"]


def test_one_float_step_is_a_difference():
    a, b = frame(), frame()
    b.loc[0, "x"] = np.nextafter(0.1, 1.0)
    r = pa.compare_frames(a, b)
    assert not r["equal"] and r["differing_by_column"] == {"x": 1} and r["largest"] > 0


def test_missing_markers_are_one_missing_value_but_differ_from_a_value():
    a, b = frame(), frame()
    b["s"] = pd.array(["u", pd.NA, "w"], dtype="object")
    b["x"] = [0.1, 0.2, None]
    assert pa.compare_frames(a, b)["equal"]
    b.loc[1, "s"] = "v"
    assert pa.compare_frames(a, b)["differing_by_column"] == {"s": 1}


def test_dates_and_timestamps_compare_as_dates():
    a, b = frame(), frame()
    b["d"] = pd.Timestamp(NOV)
    assert pa.compare_frames(a, b)["equal"]


def test_other_columns_or_keys_are_not_equal():
    a = frame()
    assert not pa.compare_frames(a, a.drop(columns="s"))["equal"]
    assert not pa.compare_frames(a, a.assign(msno=["b", "a", "z"]))["equal"]
    assert not pa.compare_frames(pd.concat([a, a.iloc[:1]]), pd.concat([a, a.iloc[:1]]))["equal"]


# --- judges ---------------------------------------------------------------------------------------


def test_summary_judge_is_exact():
    values = np.random.default_rng(0).uniform(0, 1, 500)
    logged = pa.score_summary(values)
    assert pa.judge_summary(values, logged)["met"]
    nudged = values.copy()
    k = int(np.argmax(nudged))  # the maximum is one of the 13 fields, so this must show
    nudged[k] = np.nextafter(nudged[k], 2.0)
    assert not pa.judge_summary(nudged, logged)["met"]


def test_population_difference_stops_the_run():
    exp = SERVING["step4"]["more_pass_marks"]["population_counts"]["expected"]
    assert pa.judge_population(dict(exp), dict(exp))["met"]
    with pytest.raises(pa.StopRun, match="population counts differ"):
        pa.judge_population(dict(exp) | {"scored": exp["scored"] - 1}, dict(exp))


def test_population_stop_fires_before_any_model_work(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("model work started")

    monkeypatch.setattr(sc, "refuse_unseen", refuse)
    monkeypatch.setattr(sc, "probabilities", refuse)
    status_rows = pd.DataFrame(
        {"msno": ["a"], "in_exp30": [True], "segment": ["paid"], "days_until_expiry": [3]}
    )
    inputs = SimpleNamespace(snapshot=NOV, status_rows=status_rows, rows=pd.DataFrame())
    ctx = SimpleNamespace(
        models=SimpleNamespace(c01={}, dt01=None),
        protocol=SimpleNamespace(main_segments=("paid", "discount100_only"), seed=1),
        population_snapshot=NOV,
        population_expected={"scored": 2},
    )
    with pytest.raises(pa.StopRun):
        pa.snapshot_marks(ctx, inputs)


def test_no_price_customer_with_an_allowed_action_stops_the_run():
    tab = table(200, seed=1, no_price_every=4)
    acts = sc.actions(tab, ACT)
    assert acts.opts.allowed[~acts.has_price].any()  # this hand-made table does allow some
    with pytest.raises(pa.StopRun, match="without a usable price"):
        pa.no_price_rule(acts)
    clean = sc.Actions(acts.opts, acts.best, np.ones(acts.opts.n, bool))
    assert pa.no_price_rule(clean)["met"]


def probabilities(final: np.ndarray, before: np.ndarray) -> sc.Probabilities:
    change = np.abs(final - before)
    moved = change.max(axis=1) > 1e-12
    summary = {"rows_changed": int(moved.sum()), "max_change": float(change.max())}
    return sc.Probabilities(before, before, final, moved, summary)


def test_fix_judge_counts_rows_and_the_largest_change():
    before = np.array([[0.2, 0.1, 0.3], [0.1, 0.2, 0.3]])
    final = np.array([[0.15, 0.15, 0.3], [0.1, 0.2, 0.3]])
    pr = probabilities(final, before)
    mark = {"rows": 2, "rows_changed": 1, "max_change": 0.05, "order_violations_after_fix": 0}
    tol = 1e-12
    assert pa.judge_fix(pr, mark, tol)["met"]
    assert not pa.judge_fix(pr, mark | {"rows_changed": 2}, tol)["met"]
    assert not pa.judge_fix(pr, mark | {"max_change": 0.05 + 1e-9}, tol)["met"]


def test_tenure_is_counted_in_slots_and_member_for_in_texts():
    df = pd.DataFrame(
        {
            "raising_1": ["tenure", "plan_length"],
            "raising_2": [None, None],
            "raising_3": [None, None],
            "lowering_1": [None, "tenure"],
            "lowering_2": [None, None],
            "text": ["Member for 6 months.", "ok"],
        }
    )
    assert pa.tenure_listed(df).tolist() == [True, True]
    assert pa.zero_tenure(df) == {"met": False, "tenure_slots": 2, "member_for_texts": 1}
    clean = df.assign(raising_1=None, lowering_1=None, text="ok")
    assert pa.zero_tenure(clean)["met"]


def test_before_fix_paths_counts_rows_that_differ():
    before = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    pr = probabilities(before, before)
    served = {
        7: pd.DataFrame({"probability": [0.1, np.nextafter(0.2, 1.0)]}),
        30: pd.DataFrame({"probability": [0.2, 0.3]}),
    }
    out = pa.before_fix_paths(pr, served)
    assert out["7"]["rows_differing"] == 1 and out["7"]["largest_difference"] > 0
    assert out["30"]["rows_differing"] == 0


def test_action_reference_is_phase_7_and_the_judge_is_exact():
    tab = table(300, seed=2)
    ref = pa.action_reference(tab, ACT)
    served = sc.actions(tab, ACT).frame()
    field = SERVING["contract"]["action_reading"]["field"]
    assert pa.judge_actions(served, ref, field)["equal"]
    served.loc[0, "net_value"] = np.nextafter(served.loc[0, "net_value"], 1e9)
    assert pa.judge_actions(served, ref, field)["differing_by_column"] == {"net_value": 1}


def test_recount_of_paying_blocks_is_reported_next_to_the_registered_figures():
    acts = sc.actions(table(100, seed=4), ACT)
    pays = int(acts.frame().pays.sum())
    r = pa.recount_paying(acts, {"blocks": 100, "blocks_that_pay": pays})
    assert r["same"] and r["blocks_that_pay"] == pays
    assert not pa.recount_paying(acts, {"blocks": 100, "blocks_that_pay": pays + 1})["same"]


def test_a_reference_file_with_another_hash_stops_the_run(tmp_path):
    f = tmp_path / "value.parquet"
    f.write_bytes(b"not the pinned file")
    with pytest.raises(pa.StopRun, match="differs from the pinned"):
        pa.check_pinned(f, "0" * 64)


# --- assembling the marks -------------------------------------------------------------------------


def fake_results(listed_offset: int = 0, equal: bool = True) -> dict:
    a3 = SERVING["step4"]["amendments"][0]["counts"]
    more = SERVING["step4"]["more_pass_marks"]
    tie = more["final_probability_tie"]
    results = {}
    for t, n in ((OCT, 144_996), (NOV, 169_673)):
        key = t.isoformat()
        reasons = {}
        for h in ("7", "30"):
            listed = a3[key][h] + listed_offset
            reasons[h] = {
                "parity_rows": n - listed,
                "parity": {"equal": equal},
                "self_check_rows": listed,
                "self_check": {"equal": True},
                "zero_tenure": {"met": True},
            }
        final = np.zeros((tie["rows"], 3)) if t == NOV else np.zeros((1, 3))
        pr = SimpleNamespace(
            final=final,
            summary={"rows_changed": tie["rows_changed"], "max_change": tie["max_change"]},
        )
        results[t] = {
            "rows": n,
            "reasons": reasons,
            "paid_days": {"equal": True},
            "actions": {"equal": True},
            "no_price_rule": {"met": True},
            "population": more["population_counts"]["expected"],
            "unseen_level_rows": 0,
            "_probabilities": pr,
        }
    return results


RELOAD = {"reload_classifier": {"met": True}, "reload_dt01": {"met": True}}


def test_all_marks_met_when_every_part_is():
    marks = pa.assemble_marks(fake_results(), RELOAD, SERVING)
    assert pa.overall({"marks": marks}) == []
    assert len(marks) == 2 + 3 + 2 * (2 * 3 + 3)


def test_a_failed_part_is_named():
    marks = pa.assemble_marks(fake_results(equal=False), RELOAD, SERVING)
    failed = pa.overall({"marks": marks})
    assert failed and all(f.startswith("reasons_parity_without_tenure") for f in failed)


def test_row_counts_other_than_a3_s_are_not_met():
    marks = pa.assemble_marks(fake_results(listed_offset=1), RELOAD, SERVING)
    failed = pa.overall({"marks": marks})
    assert any(f.startswith("self_check_with_tenure") for f in failed)


# --- the command's guards -------------------------------------------------------------------------


def test_score_parity_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["score-parity"])


def test_score_parity_runs_once(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    (tmp_path / cli.STEP4_REPORT).write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        cli.main(["score-parity"])


# --- the dry run --------------------------------------------------------------------------------


def stub_report(processed, repo, dry):
    return {"dry_run": dry, "passed": True}


@pytest.fixture()
def no_process_guard(monkeypatch):
    """The label-code guard is about the real process; the test session has loaded that code."""
    monkeypatch.setattr(cli, "assert_no_label_building_code", lambda: None)


def test_dry_run_never_writes_the_real_report_and_ignores_its_guard(
    monkeypatch, tmp_path, no_process_guard
):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    monkeypatch.setattr(cli, "parity_report", stub_report)
    (tmp_path / cli.STEP4_REPORT).write_text("kept")  # a real report already there
    assert cli.main(["score-parity", "--dry-run"]) == 0
    assert (tmp_path / cli.STEP4_REPORT).read_text() == "kept"
    assert (tmp_path / cli.DRY_REPORT).exists()


def test_dry_run_leaves_the_real_run_unblocked(monkeypatch, tmp_path, no_process_guard):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    monkeypatch.setattr(cli, "parity_report", stub_report)
    assert cli.main(["score-parity", "--dry-run"]) == 0
    assert not (tmp_path / cli.STEP4_REPORT).exists()
    assert cli.main(["score-parity"]) == 0  # the real run's guard was never tripped
    assert (tmp_path / cli.STEP4_REPORT).exists()


def test_dry_run_refuses_to_overwrite_its_own_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: tmp_path)
    monkeypatch.setattr(cli, "parity_report", stub_report)
    (tmp_path / cli.DRY_REPORT).write_text("{}")
    with pytest.raises(SystemExit, match="refuses to overwrite"):
        cli.main(["score-parity", "--dry-run"])


def test_dry_run_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.main(["score-parity", "--dry-run"])


def test_dry_subset_keeps_one_bucket_in_each_table_s_order():
    from retention.api.batch import SnapshotInputs
    from retention.data.sample import bucket

    ids = [f"customer{i:03d}" for i in range(400)][::-1]
    df = pd.DataFrame({"msno": ids, "x": range(400)})
    sub = pa.dry_subset(SnapshotInputs(NOV, df, df, df, df))
    expected = [m for m in ids if bucket(m) == 0]
    assert 0 < len(expected) < 400
    for t in (sub.status_rows, sub.rows, sub.facts, sub.prices):
        assert t.msno.tolist() == expected


def test_the_dry_run_does_not_judge_population(monkeypatch):
    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached

    monkeypatch.setattr(sc, "refuse_unseen", reached)
    status_rows = pd.DataFrame(
        {"msno": ["a"], "in_exp30": [True], "segment": ["paid"], "days_until_expiry": [3]}
    )
    inputs = SimpleNamespace(snapshot=NOV, status_rows=status_rows, rows=pd.DataFrame({"msno": []}))
    ctx = SimpleNamespace(
        models=SimpleNamespace(c01={}, dt01=None),
        protocol=SimpleNamespace(main_segments=("paid", "discount100_only"), seed=1),
        population_snapshot=NOV,
        population_expected={"scored": 2},
    )
    with pytest.raises(Reached):  # past the population step: nothing judged there
        pa.snapshot_marks(ctx, inputs, dry=True)
    with pytest.raises(pa.StopRun):
        pa.snapshot_marks(ctx, inputs, dry=False)


def test_dry_summary_is_equal_only_when_every_comparison_is():
    r = {
        "parity_rows": 9,
        "parity": {"equal": True},
        "self_check_rows": 1,
        "self_check": {"equal": True},
        "zero_tenure": {"met": True},
    }
    res = {
        "snapshot": "2016-11-01",
        "rows": 10,
        "population": {},
        "unseen_level_rows": 0,
        "probabilities": {"rows_changed": 1, "max_change": 0.1, "clipped": 0},
        "reasons": {"7": r, "30": r},
        "paid_days": {"equal": True},
        "actions": {"equal": True},
        "no_price_rule": {"met": True},
        "before_fix_paths": {},
        "paying_blocks": {"blocks": 10, "blocks_that_pay": 1},
    }
    assert pa.dry_summary(res)["all_comparisons_equal"]
    assert not pa.dry_summary(res | {"actions": {"equal": False}})["all_comparisons_equal"]
    bad = r | {"zero_tenure": {"met": False}}
    assert not pa.dry_summary(res | {"reasons": {"7": bad, "30": r}})["all_comparisons_equal"]
