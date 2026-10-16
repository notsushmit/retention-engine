"""Public-data validation: splits, coding, grid rule, pass marks, outputs and the run guard."""

import numpy as np
import pandas as pd
import pytest

from retention.uplift import public_validation as pv
from retention.uplift.config import load


def hillstrom_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "1) $0 - $100",
            "history": rng.gamma(2, 100, n),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
            "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
            "visit": rng.integers(0, 2, n),
            "conversion": rng.integers(0, 2, n),
            "spend": 0.0,
        }
    )


def test_split_is_stratified_and_deterministic():
    df = hillstrom_frame()
    a = pv.evaluation_half(df.segment.to_numpy())
    assert np.array_equal(a, pv.evaluation_half(df.segment.to_numpy()))
    for arm in df.segment.unique():
        n = (df.segment == arm).sum()
        assert abs(int(a[(df.segment == arm).to_numpy()].sum()) - n / 2) <= 1


def test_both_hillstrom_cells_share_one_control_arm():
    df = hillstrom_frame()
    ev = pv.evaluation_half(df.segment.to_numpy())
    mens, womens = pv.hillstrom_cells(df, ev, "visit")
    control = lambda c: set(c.rows[c.t == 0])  # noqa: E731
    assert control(mens) == control(womens)
    assert control(mens) == set(np.flatnonzero(df.segment.eq("No E-Mail")))
    # the split is made once, before the cells: a control row is on the same side in both
    for c in (mens, womens):
        assert np.array_equal(c.evaluation, ev[c.rows])


def test_hillstrom_coding_uses_training_levels_and_no_arm_column():
    df = hillstrom_frame()
    ev = pv.evaluation_half(df.segment.to_numpy())
    x = pv.hillstrom_features(df, ~ev)
    pv.check_features(x)
    assert "segment" not in x.columns and "history_segment" not in x.columns
    assert {"zip_code=Urban", "channel=Web"} <= set(x.columns)


def test_exposure_is_never_a_feature():
    df = pd.DataFrame({**{f"f{i}": [0.1] * 4 for i in range(12)}, "exposure": [1, 0, 1, 0]})
    assert "exposure" not in pv.criteo_features(df).columns
    with pytest.raises(ValueError, match="exposure"):
        pv.check_features(df)


def test_criteo_grid_subsample_is_stratified_fixed_and_from_the_training_half():
    rng = np.random.default_rng(0)
    t = (rng.random(20000) < 0.85).astype(int)
    ev = pv.evaluation_half(t)
    train_idx = np.flatnonzero(~ev)
    sub = pv.grid_subsample(train_idx, t, size=5000)
    assert len(sub) == 5000 and set(sub) <= set(train_idx)
    assert np.array_equal(sub, pv.grid_subsample(train_idx, t, size=5000))
    assert abs(t[sub].mean() - t[train_idx].mean()) < 0.002
    assert pv.GRID_SUBSAMPLE == 1_000_000


def test_grid_matches_the_registered_grid():
    g = load()["learners"]["public_grid"]
    assert len(pv.GRID) == 8
    assert {p["n_estimators"] for p in pv.GRID} == set(g["n_estimators"])
    assert {p["num_leaves"] for p in pv.GRID} == set(g["num_leaves"])
    assert {p["min_child_samples"] for p in pv.GRID} == set(g["min_child_samples"])
    assert all(p["learning_rate"] == g["learning_rate"] for p in pv.GRID)


def test_grid_tie_rule():
    p = {
        (n, lv, m): {"n_estimators": n, "num_leaves": lv, "min_child_samples": m}
        for n, lv, m in [(100, 4, 20), (100, 4, 100), (100, 15, 100), (300, 4, 100)]
    }
    # all tied: fewer trees, then fewer leaves, then larger min_child_samples
    assert pv.pick_point([(v, 0.5) for v in p.values()]) == p[(100, 4, 100)]
    # a lower loss wins outright; a 1e-12 difference is a tie
    assert pv.pick_point([(p[(300, 4, 100)], 0.4), (p[(100, 4, 20)], 0.5)]) == p[(300, 4, 100)]
    assert (
        pv.pick_point([(p[(300, 4, 100)], 0.5 - 1e-12), (p[(100, 15, 100)], 0.5)])
        == p[(100, 15, 100)]
    )


def iv(est, lo, hi):
    from retention.uplift.qini import Interval

    return Interval(est, lo, hi, np.zeros(1))


# Q(1) = 0.02 in every hand case, so the size mark (b) is A >= 0.10 * 0.02 / 2 = 0.001.
Q1 = 0.02


def test_pass_marks_a_and_b_both_pass():
    v = pv.judge(iv(0.004, 0.002, 0.006), iv(0.03, 0.01, 0.05), Q1)
    assert v["a"] and v["b_judged"] and v["b"] and v["met"] and v["status"] == "met"
    assert v["b_threshold"] == pytest.approx(0.001)


def test_pass_marks_a_passes_b_fails():
    # the interval of A is above 0, but A = 0.0008 is below the minimum size 0.001
    v = pv.judge(iv(0.0008, 0.0002, 0.0014), iv(0.03, 0.01, 0.05), Q1)
    assert v["a"] and v["b_judged"] and v["b"] is False
    assert not v["met"] and v["status"] == "not met"


def test_pass_marks_a_fails():
    v = pv.judge(iv(0.004, -0.001, 0.009), iv(0.03, 0.01, 0.05), Q1)
    assert not v["a"] and not v["met"] and v["status"] == "not met"


def test_b_not_judged_and_passed_on_a_alone_counts_as_not_met():
    v = pv.judge(iv(0.004, 0.002, 0.006), iv(0.01, -0.005, 0.025), Q1)
    assert v["a"] and not v["b_judged"] and v["b"] is None
    assert not v["met"]
    assert v["status"] == "passed on (a) alone; average effect not distinguishable from zero"
    # (b) not judged and (a) failing: plain "not met"
    w = pv.judge(iv(0.004, -0.002, 0.01), iv(0.01, -0.005, 0.025), Q1)
    assert not w["met"] and w["status"] == "not met"


def test_done_when_needs_all_three_primary_cells():
    met = {"met": True}
    alone = {"met": False}
    all_met = dict.fromkeys(pv.PRIMARY, met)
    assert pv.done_when(all_met) == {"met": True, "failing_primary_cells": []}
    one = {**all_met, "hillstrom_womens_vs_none_visit": alone}
    assert pv.done_when(one) == {
        "met": False,
        "failing_primary_cells": ["hillstrom_womens_vs_none_visit"],
    }
    with pytest.raises(ValueError, match="missing"):
        pv.done_when({"criteo_visit": met})
    # secondary cells never count
    assert pv.done_when({**all_met, "criteo_conversion": alone})["met"]


def test_primary_cells_match_the_config():
    assert load()["validation"]["cells"]["primary"] == pv.PRIMARY


def test_sanity_mark():
    naive = iv(0.03, 0.01, 0.05)
    assert pv.sanity(0.02, naive) and pv.sanity(0.05, naive)
    assert not pv.sanity(0.06, naive)


def score_frame(dataset="hillstrom"):
    return pd.DataFrame(
        {
            "dataset": [dataset],
            "cell": ["mens_vs_none"],
            "outcome": ["visit"],
            "row": [3],
            "uplift_t_learner": [0.1],
            "uplift_x_learner": [0.1],
            "outcome_probability": [0.2],
        }
    )


def test_no_kkbox_row_can_be_written_to_uplift_scores(tmp_path):
    path = tmp_path / "uplift_scores.parquet"
    pv.write_uplift_scores(score_frame(), path)
    assert path.exists()
    with pytest.raises(ValueError, match="public datasets only"):
        pv.write_uplift_scores(score_frame("kkbox"), tmp_path / "x.parquet")
    with_customer = score_frame().assign(msno="abc")
    with pytest.raises(ValueError, match="exactly"):
        pv.write_uplift_scores(with_customer, tmp_path / "y.parquet")
    assert not (tmp_path / "x.parquet").exists() and not (tmp_path / "y.parquet").exists()


def test_run_once_guard(tmp_path):
    out = tmp_path / "public"
    marker = pv.begin_run(out, "abc1234")
    assert marker["first_attempt"]["commit"] == "abc1234" and (out / pv.MARKER).exists()
    # an attempt that started fitting and did not finish: a repeat needs a stated reason
    with pytest.raises(pv.RunOnce, match="stated reason"):
        pv.begin_run(out, "abc1234")
    marker = pv.begin_run(out, "abc1234", "crashed during the Criteo bootstrap")
    assert marker["repeats"][0]["reason"] == "crashed during the Criteo bootstrap"
    # once the log exists, every further run refuses, with or without a reason
    (out / pv.LOG).write_text("{}")
    for reason in (None, "any reason"):
        with pytest.raises(pv.RunOnce, match="runs once"):
            pv.guard(out, reason)


def synthetic_public(seed: int = 0):
    rng = np.random.default_rng(seed)
    hill = hillstrom_frame(3000, seed)
    high = (hill.history > hill.history.median()).to_numpy()
    p = 0.15 + 0.15 * high * hill.segment.eq("Mens E-Mail").to_numpy()
    p += 0.03 * hill.segment.eq("Womens E-Mail").to_numpy()
    hill["visit"] = (rng.random(len(hill)) < p).astype(int)
    hill["conversion"] = (rng.random(len(hill)) < p / 5).astype(int)
    n = 6000
    crit = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    crit["treatment"] = (rng.random(n) < 0.85).astype(int)
    crit["exposure"] = crit.treatment * (rng.random(n) < 0.04)
    pc = 0.05 + 0.10 * (crit.f0.to_numpy() > 0) * crit.treatment.to_numpy()
    crit["visit"] = (rng.random(n) < pc).astype(int)
    crit["conversion"] = (rng.random(n) < pc / 5).astype(int)
    return hill, crit


def test_end_to_end_on_small_synthetic_data(tmp_path):
    hill, crit = synthetic_public()
    out = tmp_path / "public"
    log = pv.run_validation(hill, crit, out, "abc1234", b=20, grid=pv.GRID[:2], grid_size=1000)
    assert (out / pv.MARKER).exists()  # written before the first fit
    assert not (out / pv.LOG).exists()  # the log is written by write_log, last
    pv.write_log(out, log)
    keys = {
        "hillstrom_mens_vs_none_visit",
        "hillstrom_womens_vs_none_visit",
        "hillstrom_mens_vs_none_conversion",
        "hillstrom_womens_vs_none_conversion",
        "criteo_visit",
        "criteo_conversion",
    }
    assert set(log["cells"]) == keys
    for rec in log["cells"].values():
        area = rec["qini_area"]["x_learner"]
        assert list(area) == ["ci_low", "ci_high", "estimate"]  # interval before estimate
        assert set(rec["pass_marks"]) == {"a", "b_judged", "b", "b_threshold", "met", "status"}
        assert rec["timing"]["seconds_per_resample"] >= 0
        assert "exposure" not in str(rec["chosen_grid_points"])
    assert set(log["done_when_first_criterion"]) == {"met", "failing_primary_cells"}
    # the strong, heterogeneous effect built into the men's cell is found
    assert log["cells"]["hillstrom_mens_vs_none_visit"]["qini_area"]["x_learner"]["estimate"] > 0
    curves = pd.read_parquet(out / "qini_curves.parquet")
    assert len(curves) == 6 * 4 * 100
    scores = pd.read_parquet(out / "uplift_scores.parquet")
    assert set(scores.dataset) == {"hillstrom", "criteo"}
    assert list(scores.columns) == pv.SCORE_COLUMNS
    assert len(scores) == sum(rec["evaluation_rows"] for rec in log["cells"].values())
    assert len(log["budget"]) == 6  # one timing projection per cell
    with pytest.raises(pv.RunOnce, match="runs once"):
        pv.run_validation(hill, crit, out, "abc1234", b=20, grid=pv.GRID[:2], grid_size=1000)
