import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.models.classifier import feature_set_columns, score_summary
from retention.optimize.config import NO_LIMIT, load_actions_config
from retention.optimize.price import price_table
from retention.optimize.value import (
    INTERVAL_ENDS_ALL,
    customer_table,
    disagreement,
    dt01_settings,
    expected_days,
    headline_level,
    reproduce,
)

CFG = load_actions_config()
ENDS = INTERVAL_ENDS_ALL


def curve(values_by_end):
    return np.array([[values_by_end[e] for e in ENDS]])


def test_flat_survival_gives_60_and_335_days():
    d = expected_days(np.ones((2, len(ENDS))))
    assert d.days_30_90.tolist() == [60.0, 60.0]
    assert d.days_30_365.tolist() == [335.0, 335.0]


def test_hand_drawn_curve_trapezoid():
    s = {7: 0.99, 14: 0.98, 21: 0.95, 30: 0.8, 37: 0.8, 44: 0.6, 51: 0.6, 59: 0.4, 67: 0.4,
         74: 0.4, 81: 0.2, 90: 0.0}  # fmt: skip
    r = [s[e] / 0.8 for e in (30, 37, 44, 51, 59, 67, 74, 81, 90)]
    w = [7, 7, 7, 8, 8, 7, 7, 9]
    expect = sum((r[k] + r[k + 1]) / 2 * w[k] for k in range(8))
    d = expected_days(curve(s))
    assert d.days_30_90[0] == pytest.approx(expect)
    assert d.days_30_365[0] == pytest.approx(expect)  # S(90) = 0: no tail


def test_constant_hazard_tail_closed_form():
    q = 0.99
    s = {e: 1.0 for e in ENDS}
    s[81] = 0.9
    s[90] = 0.9 * q**9
    d = expected_days(curve(s))
    r90 = s[90]
    tail = r90 * (1 + q) / 2 * (1 - q**275) / (1 - q)
    # daily trapezoid sum equals the closed form
    daily = sum((r90 * q**k + r90 * q ** (k + 1)) / 2 for k in range(275))
    assert tail == pytest.approx(daily)
    assert d.days_30_365[0] - d.days_30_90[0] == pytest.approx(tail)


def test_zero_survival_at_30_gives_zero_days():
    d = expected_days(np.zeros((1, len(ENDS))))
    assert d.days_30_90[0] == 0.0 and d.days_30_365[0] == 0.0


@pytest.mark.parametrize(
    ("spend", "level"),
    [(30000.0, 25000.0), (1000.0, "no_limit"), (999.0, "no_limit"), (1000.5, 1000.0),
     (1e9, 50000.0), (10000.0, 5000.0)],
)  # fmt: skip
def test_headline_level_rule(spend, level):
    assert headline_level(spend, CFG.budgets) == level
    assert NO_LIMIT in CFG.budgets


def test_disagreement_top_decile_below_dt01_median():
    n = 20
    t = pd.DataFrame(
        {
            "msno": [f"m{i:02d}" for i in range(n)],
            "p30": np.linspace(0.9, 0.01, n),  # m00, m01 are the top decile
            "dt01_lapse_30": [0.001, 0.9] + list(np.linspace(0.1, 0.5, n - 2)),
        }
    )
    assert disagreement(t) == {"top_decile": 2, "disagree": 1, "share": 0.5}


class FakeModel:
    def __init__(self, p):
        self.p = p

    def lapse_probability(self, rows, days):
        return self.p[days[0]][:, None], {}


def test_reproduction_is_exact_and_names_the_differing_field():
    rng = np.random.default_rng(0)
    p = {h: rng.random(50) for h in (7, 30, 59)}
    rehearsal = {"primary": {}, "demonstration_59": {}}
    for h, _snap, path in CFG.reproduction_cells:
        node = rehearsal
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node["score_summary_raw"] = json.loads(json.dumps(score_summary(p[h])))
    rows = {date(2016, 11, 1): pd.DataFrame({"x": range(50)})}
    table, failures = reproduce(FakeModel(p), rehearsal, rows, CFG)
    assert failures == [] and len(table) == 3
    p[30] = p[30].copy()
    p[30][np.argmax(p[30])] += 1e-12  # changes the logged max
    _, failures = reproduce(FakeModel(p), rehearsal, rows, CFG)
    assert failures == ["dt01 30 days 2016-11-01: differs in ['mean', 'max']"] or failures == [
        "dt01 30 days 2016-11-01: differs in ['max']"
    ]


def test_customer_table_refuses_missing_probability():
    rows = pd.DataFrame(
        {
            "msno": ["a", "b"],
            "snapshot_date": pd.to_datetime(["2016-11-01"] * 2),
            "last_is_auto_renew": [0, 1],
            "days_until_paid_expiry": [5, 20],
        }
    )
    surv = np.ones((2, len(ENDS)))
    prices = pd.DataFrame({"msno": ["a", "b"], "price_per_day": [5.0, 0.0], "has_price": [1, 0]})
    p30 = pd.DataFrame({"msno": ["a"], "snapshot_date": [date(2016, 11, 1)], "p30": [0.1]})
    with pytest.raises(ValueError, match="no calibrated p30"):
        customer_table(rows, surv, p30, prices)
    p30 = pd.DataFrame(
        {"msno": ["a", "b"], "snapshot_date": [date(2016, 11, 1)] * 2, "p30": [0.1, 0.2]}
    )
    t = customer_table(rows, surv, p30, prices)
    assert t.auto_renew.tolist() == ["off", "on"] and t.days_30_90.tolist() == [60.0, 60.0]


def test_dt01_settings_are_the_frozen_ones():
    s = dt01_settings()
    raw = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
    drop = raw["selection"]["feature_sets"]["fs1"]["drop"]
    assert s.columns == feature_set_columns(drop)
    assert s.ends == list(ENDS) and s.offset_input is False


def test_price_table_counts(tmp_path):
    rows = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                ["2016-01-05", "2016-03-01", "2016-11-30", "2016-12-01", "2015-12-31"]
            ).date,
            "is_cancel": [0, 0, 0, 0, 0],
            "actual_amount_paid": [149, 149, 99, 149, 149],
            "plan_list_price": [149, 149, 99, 149, 149],
            "payment_plan_days": [30, 30, 30, 30, 30],
        }
    )
    path = tmp_path / "tx.parquet"
    rows.to_parquet(path)
    out = price_table(path)
    assert out["rows"] == 3
    assert out["top"][0] == {
        "plan_days": 30,
        "list_price": 149,
        "paid": 149,
        "n": 2,
        "share": 2 / 3,
    }


def test_run_once_guard(tmp_path):
    from retention.optimize.__main__ import run_once

    run_once(tmp_path / "absent.json")
    (tmp_path / "done.json").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        run_once(tmp_path / "done.json")


VALUE_STEP_CODE = """
import sys
import numpy as np, pandas as pd
import retention.optimize.__main__, retention.optimize.value
from retention.optimize.config import load_actions_config
from retention.optimize.effects import build_options
from retention.optimize.solve import greedy
from retention.optimize.value import expected_days
cfg = load_actions_config()
d = expected_days(np.ones((3, 12)))
c = pd.DataFrame({"msno": list("abc"), "p30": [0.5, 0.1, 0.02], "auto_renew": ["off", "on", "off"],
    "price_per_day": [5.0, 5.0, 0.0], "days_30_90": d.days_30_90, "days_30_365": d.days_30_365,
    "days_until_paid_expiry": pd.array([10, 3, None], dtype="Int32")})
greedy(build_options(c, cfg, cfg.base), None)
assert "retention.labels.snapshots" not in sys.modules, "label-building code loaded"
print("ok")
"""


def test_value_step_never_loads_label_building_code():
    out = subprocess.run(
        [sys.executable, "-c", VALUE_STEP_CODE], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_value_step_source_never_builds_labels_or_unlocks():
    for f in ("value.py", "__main__.py", "price.py"):
        src = (Path("src/retention/optimize") / f).read_text()
        for banned in (
            "labels_as_of(",
            "build_labels_as_of",
            "TestUnlock",
            "test_months",
            "score-test",
        ):
            assert banned not in src, (f, banned)
