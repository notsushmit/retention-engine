"""Customer-clustered bootstrap: grouping, determinism, the not-computable rule, the budget."""

import numpy as np
import pandas as pd
import pytest

from retention.uplift import kkbox_bootstrap as kb
from retention.uplift import learners


def test_a_drawn_customer_brings_all_of_their_units():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    idx = kb.resample(groups, np.random.default_rng(0))
    drawn = groups[idx]
    for g in np.unique(drawn):
        units = np.flatnonzero(groups == g)
        copies = [int((idx == u).sum()) for u in units]
        assert len(set(copies)) == 1 and copies[0] >= 1  # every unit of g drawn equally often


def test_bootstrap_is_deterministic_and_forms_percentile_intervals():
    groups = np.repeat(np.arange(50), 2)
    values = np.arange(100, dtype=float)

    def stat(idx, rng):
        return {"mean": float(values[idx].mean())}

    a = kb.bootstrap("m", stat, {"mean": values.mean()}, groups, b=200, seed=0)
    b = kb.bootstrap("m", stat, {"mean": values.mean()}, groups, b=200, seed=0)
    assert a.status == "computed" and np.array_equal(a.replicates["mean"], b.replicates["mean"])
    low, high = a.intervals["mean"]
    assert low < values.mean() < high
    assert (low, high) == tuple(np.percentile(a.replicates["mean"], [2.5, 97.5]))


def test_one_not_computable_resample_makes_the_interval_not_computable():
    groups = np.arange(20)
    calls = []

    def stat(idx, rng):
        calls.append(1)
        if len(calls) == 7:
            raise learners.NotComputable("a kept unit has a propensity of exactly 0 or 1")
        return {"x": 1.0}

    out = kb.bootstrap("m", stat, {"x": 1.0}, groups, b=100, seed=0)
    assert out.status == "not computable" and out.intervals is None
    assert out.reason == "resample 7: a kept unit has a propensity of exactly 0 or 1"
    assert out.estimates == {"x": 1.0} and len(calls) == 7


def test_budget_stops_when_the_projection_exceeds_three_hours():
    # each clock call advances 20 s: 2 s per resample, so (990 + 5000) resamples take 3.3 h
    ticks = iter(np.arange(0.0, 1e6, 20.0))
    clock = lambda: next(ticks)  # noqa: E731
    budget = kb.Budget(step_started=clock(), later_work=5000, clock=clock)
    with pytest.raises(kb.OverBudget, match="exceeds 3.0 h"):
        kb.bootstrap(
            "m",
            lambda i, r: {"x": 0.0},
            {"x": 0.0},
            np.arange(10),
            b=1000,
            budget=budget,
            clock=clock,
        )
    assert budget.projections[0]["bootstrap"] == "m"


def test_budget_passes_a_short_run_and_records_the_projection():
    budget = kb.Budget(step_started=0.0, later_work=10, clock=lambda: 1.0)
    kb.bootstrap(
        "m",
        lambda i, r: {"x": 0.0},
        {"x": 0.0},
        np.arange(10),
        b=20,
        budget=budget,
        clock=lambda: 1.0,
    )
    assert budget.projections == [
        {"bootstrap": "m", "seconds_per_resample": 0.0, "projected_total_seconds": 1.0}
    ]


def test_effect_statistic_keeps_trimming_membership_and_refits_on_kept_units(monkeypatch):
    rng = np.random.default_rng(1)
    n = 200
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.3).astype(float)
    groups = np.repeat(np.arange(n // 2), 2)
    kept = np.arange(n) % 3 != 0  # an arbitrary fixed membership
    seen = []
    real = learners.estimate

    def spy(xk, yk, tk, fk, ek, kind, params, log=None):
        seen.append(len(yk))
        return real(xk, yk, tk, fk, ek, kind, params, log)

    monkeypatch.setattr(learners, "estimate", spy)
    stat = kb.effect_statistic(x, t, y, groups, kept, 1.0, "classifier", learners.KKBOX_FIXED)
    idx = kb.resample(groups, np.random.default_rng(0))
    out = stat(idx, np.random.default_rng(0))
    assert set(out) == {"aipw", "t_learner", "x_learner"}
    assert seen == [int(kept[idx].sum())]  # only the resampled units that were kept originally


def test_naive_statistic_uses_only_the_given_rows():
    t = np.array([1, 1, 0, 0, 1, 0])
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    rows = np.array([True, True, True, True, False, False])
    stat = kb.naive_statistic(t, y, rows)
    assert stat(np.arange(6), None) == {"naive": 1.0}
