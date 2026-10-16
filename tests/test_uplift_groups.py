"""Customer groups: thresholds, assignment, reproduction, the registered check, outputs."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.uplift import groups
from retention.uplift import public_validation as pv
from retention.uplift.config import load

sys.path.insert(0, str(Path(__file__).parent))
from test_uplift_public_validation import synthetic_public  # noqa: E402


def test_registered_cells_match_the_config():
    g = load()["groups"]
    assert g["cells"] == groups.CELLS
    assert g["not_built"] == groups.NOT_BUILT


def test_thresholds_hand_case():
    # treated rate 3/4, control rate 1/4: delta = 0.5 * 0.5 = 0.25, m = 0.25
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0], float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert groups.thresholds(y, t) == {"delta": 0.25, "m": 0.25}


def test_assignment_boundaries():
    delta, m = 0.1, 0.3
    tau = np.array([0.1, 0.2, -0.1, -0.3, 0.05, 0.05, -0.05])
    mu0 = np.array([0.1, 0.9, 0.9, 0.1, 0.3, 0.29, 0.5])
    assert groups.assign(tau, mu0, delta, m).tolist() == [
        "persuadable",
        "persuadable",
        "do_not_disturb",
        "do_not_disturb",
        "stays_anyway",
        "lost_cause",
        "stays_anyway",
    ]


@pytest.fixture(scope="module")
def validated(tmp_path_factory):
    out = tmp_path_factory.mktemp("public")
    hill, crit = synthetic_public()
    log = pv.run_validation(hill, crit, out, "abc1234", b=20, grid=pv.GRID[:2], grid_size=1000)
    scores = pd.read_parquet(out / "uplift_scores.parquet")
    return hill, crit, scores, log, out


def test_end_to_end_groups_on_synthetic_data(validated):
    hill, crit, scores, log, out = validated
    res = groups.run_groups(hill, crit, scores, log, out, b=20, grid_size=1000)
    assert set(res["cells"]) == set(groups.CELLS)
    assert res["not_built"] == {
        "hillstrom_mens_vs_none_visit": "not built: the cell did not beat random"
    }
    for key, rec in res["cells"].items():
        assert rec["reproduction_max_abs_diff"] <= groups.REPRODUCTION_TOL
        assert sum(rec["counts"].values()) == log["cells"][key]["evaluation_rows"]
        assert set(rec["check"]) == set(groups.GROUPS)
        assert rec["persuadable_largest"] in {"yes", "no", "not computable"}
        for r in rec["check"].values():
            if r["status"] == "computed":
                assert list(r)[-3:] == ["ci_low", "ci_high", "estimate"]  # interval before estimate
    members = pd.read_parquet(out / "group_members.parquet")
    assert set(members.dataset) <= {"hillstrom", "criteo"}
    assert "no KKBox customer is assigned to a group" in res["note"]


def test_a_changed_logged_score_stops_the_step(validated):
    hill, crit, scores, log, out = validated
    bad = scores.copy()
    bad.loc[bad.cell == "womens_vs_none", "uplift_t_learner"] += 1e-6
    with pytest.raises(groups.ReproductionFailed, match="differs"):
        groups.run_groups(hill, crit, bad, log, out, b=20, grid_size=1000)


def test_assignment_reads_no_evaluation_outcome(validated):
    hill, crit, scores, log, _ = validated
    built = {pv.cell_key(c): (c, x) for c, x, _ in pv.build_cells(hill, crit, 1000)}
    cell, x = built["hillstrom_womens_vs_none_visit"]
    mine = scores[
        (scores.dataset == "hillstrom")
        & (scores.cell == "womens_vs_none")
        & (scores.outcome == "visit")
    ]
    mine = mine.reset_index(drop=True)
    chosen = log["cells"]["hillstrom_womens_vs_none_visit"]["chosen_grid_points"]
    a_rec, a_mem = groups.group_cell(cell, x, chosen, mine, b=20)
    y = cell.y.copy()
    y[cell.evaluation] = 1 - y[cell.evaluation]  # flip every evaluation-half outcome
    b_rec, b_mem = groups.group_cell(dataclasses.replace(cell, y=y), x, chosen, mine, b=20)
    assert a_rec["thresholds"] == b_rec["thresholds"]
    assert a_mem.group.equals(b_mem.group)
    assert a_rec["counts"] == b_rec["counts"]


def test_no_kkbox_row_can_be_written(tmp_path):
    members = pd.DataFrame(
        {"dataset": ["kkbox"], "cell": ["x"], "row": [1], "group": ["persuadable"]}
    )
    with pytest.raises(ValueError, match="public datasets only"):
        groups.write_members(members, tmp_path / "m.parquet")
    with pytest.raises(ValueError, match="exactly"):
        groups.write_members(members.assign(msno="a"), tmp_path / "m.parquet")


def test_check_reports_empty_arms_as_not_computable():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    t = np.array([1, 1, 0, 0])
    g = np.array(["persuadable", "persuadable", "lost_cause", "lost_cause"])
    out = groups.group_check(y, t, g, b=20)
    assert out["persuadable"]["status"] == "not computable: an arm is empty"
    assert groups.persuadable_largest(out) == "not computable"
