"""The outcome command's order and outputs, with the heavy parts stubbed (no real data)."""

import json

import numpy as np
import pandas as pd
import pytest

import retention.uplift.__main__ as cmd
from retention.uplift import kkbox_bootstrap, outcome, placebo, propensity, report


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    n = 120
    rng = np.random.default_rng(0)
    analysed = pd.DataFrame(
        {
            "msno": [f"c{i // 2}" for i in range(n)],
            "snapshot_date": pd.to_datetime(["2016-09-01"] * n),
            "treated": np.tile([0, 1], n // 2),
            "paid_expiry_date": pd.to_datetime(["2016-09-30"] * n),
            "expiry_date": pd.to_datetime(["2016-10-31"] * n),
        }
    )
    cov = pd.DataFrame(
        {"has_any_log_before_T": 1, "first_log_in_window": 0, "days_since_last_listen": 1.0},
        index=range(n),
    )
    design = propensity.Design(pd.DataFrame({"a": rng.normal(size=n)}))
    order = []
    monkeypatch.setattr(cmd, "KKBOX_OUT", tmp_path)
    monkeypatch.setattr(cmd, "_require_clean", lambda: None)
    monkeypatch.setattr(cmd, "_git", lambda *a: "abc1234\n")
    monkeypatch.setattr(cmd, "_inputs", lambda drop_members=False: (analysed, cov, design, None))
    monkeypatch.setattr(cmd, "_design", lambda c, drop_members=False: design)
    monkeypatch.setattr(
        cmd, "_fit_propensity", lambda d, t, g, grid: (np.arange(n) % 5, 1.0, np.full(n, 0.5))
    )
    monkeypatch.setattr(cmd, "_check_saved_scores", lambda a, e, f: 0.0)

    def days(logs, units):
        order.append("placebo")
        return np.ones(len(units))

    monkeypatch.setattr(placebo, "placebo_days", days)
    monkeypatch.setattr(placebo, "check_leak", lambda count: None)

    def est(name, *args):
        order.append(f"estimate {name}")
        return kkbox_bootstrap.Result(
            {"aipw": 0.0, "t_learner": 0.0, "x_learner": 0.0},
            "computed",
            None,
            {"aipw": (-1, 1), "t_learner": (-1, 1), "x_learner": (-1, 1)},
            None,
            1000,
        )

    monkeypatch.setattr(cmd, "_estimate", est)
    monkeypatch.setattr(
        cmd,
        "_naive",
        lambda name, *a: kkbox_bootstrap.Result(
            {"naive": 0.0}, "computed", None, {"naive": (-1, 1)}
        ),
    )

    def read(units, **kw):
        order.append("read")
        assert (tmp_path / "placebo.json").exists(), "the placebo log must exist before the read"
        assert kw["placebo_log"] == tmp_path / "placebo.json"
        return np.zeros(len(units))

    monkeypatch.setattr(outcome, "read_outcome", read)
    return tmp_path, order


def test_placebo_is_computed_and_logged_before_the_outcome_is_read(stubbed):
    tmp, order = stubbed
    cmd.outcome_step()
    assert order[:3] == ["placebo", "estimate placebo", "read"]
    assert order.index("read") < order.index("estimate primary")


def test_outputs_carry_the_label_and_the_reading(stubbed):
    tmp, _ = stubbed
    log = cmd.outcome_step()
    assert log["label"] == report.LABEL and log["reading"] == report.READING
    assert all(r["label"] == report.LABEL for r in log["table"])
    assert all("balance not met" in s.lower() for s in log["sentences"])
    assert json.loads((tmp / "placebo.json").read_text())["label"] == report.LABEL
    table = pd.read_parquet(tmp / "outcome_table.parquet")
    assert list(table.columns) == report.COLUMNS and "msno" not in table.columns
    assert (tmp / "outcome_first_run.json").read_text() == (tmp / "outcome.json").read_text()


def test_a_second_run_refuses_once_the_log_exists(stubbed):
    tmp, _ = stubbed
    cmd.outcome_step()
    with pytest.raises(outcome.ReadOnce, match="read once"):
        cmd.outcome_step()
