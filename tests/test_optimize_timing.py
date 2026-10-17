"""The registered projection rechecks, with stubbed solve times (no real solve is timed)."""

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

import retention.optimize.__main__ as cli
from retention.optimize.config import load_actions_config
from retention.optimize.solve import NO_SOLUTION, NONE, Plan
from retention.optimize.strategies import TimeGuard
from test_optimize_report import synthetic_customers

CFG = load_actions_config()
RANDOM_SECONDS = 13.2  # the synthetic random-baseline timing measured before the build


def guard() -> TimeGuard:
    return TimeGuard(CFG.projection_stop_hours, RANDOM_SECONDS, 84, 112, CFG.projection_checks)


def run_first_cells(g: TimeGuard, solve: float) -> dict:
    assert g.record(False, 0.0, RANDOM_SECONDS) is None  # base point, budget 0
    return g.record(True, solve, solve + RANDOM_SECONDS)  # base point, budget 1,000


def test_stop_fires_when_the_first_budget_limited_cell_takes_120_s():
    check = run_first_cells(guard(), 120.0)
    # 146.4 elapsed + 120 x 83 + 13.2 x 110 = 11,558.4 s
    assert check["projected_hours"] == pytest.approx(11558.4 / 3600)
    assert check["stop"] and check["measured_solve_seconds"] == [120.0]


def test_stop_does_not_fire_at_5_s():
    check = run_first_cells(guard(), 5.0)
    assert check["projected_hours"] == pytest.approx((31.4 + 5 * 83 + 13.2 * 110) / 3600)
    assert not check["stop"]


def test_second_check_after_10_budget_limited_cells_uses_their_mean():
    g = guard()
    assert not run_first_cells(g, 5.0)["stop"]
    for _ in range(8):
        assert g.record(True, 150.0, 150.0 + RANDOM_SECONDS) is None
    check = g.record(True, 150.0, 150.0 + RANDOM_SECONDS)  # the 10th budget-limited cell
    assert check["after_budget_limited_cells"] == 10
    assert check["mean_solve_seconds"] == pytest.approx((5 + 9 * 150) / 10)
    assert check["stop"]
    assert len(g.checks) == 2 and g.record(True, 1.0, 1.0) is None  # no third check


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def run_plan(tmp_path, monkeypatch, solve_seconds: float, highs_for=None) -> int:
    """highs_for(scenario, budget) may return a HiGHS stand-in for that cell."""
    out = tmp_path
    pd.concat(
        [
            synthetic_customers(200, date(2016, 11, 1), 1),
            synthetic_customers(150, date(2016, 10, 1), 2),
        ],
        ignore_index=True,
    ).to_parquet(out / "value.parquet")
    (out / "dt01_refit.json").write_text(json.dumps({"passed": True}))
    cfg = replace(CFG, headline_budget_level=1000.0, random_draws=5)
    fake = FakeClock()
    real_checked = cli.checked_plan

    def stubbed(opts, budget, limit, time_limit, ordered=None):
        if budget is not None and budget > 0:
            fake.t += solve_seconds
        stand_in = highs_for(opts.scenario.name, budget) if highs_for else None
        if stand_in is not None:
            return real_checked(opts, budget, limit, time_limit, ordered, stand_in)
        return real_checked(opts, budget, limit, time_limit, ordered)

    monkeypatch.setattr(cli, "clock", fake)
    monkeypatch.setattr(cli, "checked_plan", stubbed)
    monkeypatch.setattr(cli, "time_random", lambda *a, **k: RANDOM_SECONDS)
    monkeypatch.setattr(cli, "load_actions_config", lambda: cfg)
    monkeypatch.setattr(cli, "out_dir", lambda fraction: out)
    monkeypatch.setattr(cli, "require_clean", lambda repo: None)
    monkeypatch.setattr(cli, "head_commit", lambda repo: "test000")
    return cli.plan(type("A", (), {"fraction": None})())


def test_plan_step_stops_at_120_s_and_reports_measured_seconds(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="above 3.0 h after 1 budget-limited cells"):
        run_plan(tmp_path, monkeypatch, 120.0)
    stopped = json.loads((tmp_path / "plan_stopped.json").read_text())
    assert stopped["projection_checks"][0]["measured_solve_seconds"] == [120.0]
    lines = [json.loads(x) for x in (tmp_path / "plan_progress.jsonl").read_text().splitlines()]
    last = lines[-1]  # the cell that triggered the stop is logged before the stop
    assert (last["event"], last["scenario"], last["budget"]) == ("done", "base", "1,000")
    assert last["solve_seconds"] == 120.0
    assert not (tmp_path / "plan_report.json").exists()


def test_plan_step_runs_through_at_5_s(tmp_path, monkeypatch):
    assert run_plan(tmp_path, monkeypatch, 5.0) == 0
    rep = json.loads((tmp_path / "plan_report.json").read_text())
    checks = rep["timing"]["projection_checks"]
    assert [c["after_budget_limited_cells"] for c in checks] == [1, 10]
    assert not any(c["stop"] for c in checks)
    assert all("solve_seconds" in c for c in rep["solver_checks"])


def no_solution(opts, budget, contact_limit, time_limit):
    c = np.full(opts.n, NONE, dtype=np.int64)
    extra = {"proven": False, "feasible": False, "dual": None, "message": "Time limit reached"}
    return Plan(c, 0.0, -np.inf, 0, "highs", extra=extra)


def test_plan_step_runs_through_a_cell_without_a_feasible_solution(tmp_path, monkeypatch):
    def highs_for(scenario, budget):
        if (scenario, budget) in (("absolute", 1000.0), ("contact_limit", 2500.0)):
            return no_solution
        return None

    assert run_plan(tmp_path, monkeypatch, 5.0, highs_for) == 0
    rep = json.loads((tmp_path / "plan_report.json").read_text())
    s = rep["solver_summary"]
    assert s["no_feasible_solution"] == 2 and s["unproven"] >= 2 and s["breaches"] == 0
    marked = [c for c in rep["solver_checks"] if c["exact_check"] == NO_SOLUTION]
    assert {(c["scenario"], c["budget"]) for c in marked} == {
        ("absolute", "1,000"),
        ("contact_limit", "2,500"),
    }
    lines = [json.loads(x) for x in (tmp_path / "plan_progress.jsonl").read_text().splitlines()]
    assert (
        sum(x["event"] == "start" for x in lines) == sum(x["event"] == "done" for x in lines) == 112
    )


def test_an_unexpected_error_keeps_the_failing_cell(tmp_path, monkeypatch):
    def boom(opts, budget, contact_limit, time_limit):
        raise ValueError("solver crashed")

    def highs_for(scenario, budget):
        return boom if (scenario, budget) == ("absolute", 1000.0) else None

    with pytest.raises(ValueError, match="solver crashed"):
        run_plan(tmp_path, monkeypatch, 5.0, highs_for)
    stopped = json.loads((tmp_path / "plan_stopped.json").read_text())
    assert (stopped["scenario"], stopped["budget"]) == ("absolute", "1,000")
    assert stopped["reason"] == "ValueError: solver crashed"
    last = json.loads((tmp_path / "plan_progress.jsonl").read_text().splitlines()[-1])
    assert (last["event"], last["scenario"], last["budget"]) == ("stopped", "absolute", "1,000")


def test_plan_step_refuses_to_start_over_an_earlier_run(tmp_path, monkeypatch):
    (tmp_path / "plan_progress.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="earlier run"):
        run_plan(tmp_path, monkeypatch, 5.0)
