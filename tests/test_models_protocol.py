import ast
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_info

from retention.data.sample import bucket
from retention.models.protocol import (
    GUARD_FUNCTION,
    GUARD_MODULE,
    ProtocolError,
    TestUnlock,
    fully_observed,
    load_protocol,
    membership,
    parse_protocol,
)

RAW = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
D = date.fromisoformat


def raw():
    return deepcopy(RAW)


def test_project_protocol_loads_as_registered():
    p = load_protocol()
    assert p.threads.n == 4 and p.purge_days == 30 and p.seed == 20161001
    assert p.fold("f7a").fit == (D("2016-08-01"),) and p.fold("f7a").as_of == D("2016-10-01")
    assert p.fold("f7b").fit == (D("2016-08-01"), D("2016-09-01"))
    assert p.fold("f7b_cal").calibration.snapshot == D("2016-09-01")
    assert p.fold("f30").score.half == "selection"
    assert p.test_fit.fit == (D("2016-08-01"), D("2016-09-01"))
    assert p.test_fit.calibration.half == "calibration"
    assert p.check_month == D("2016-11-01") and p.locked_from == D("2017-01-01")


def test_purge_boundary_is_inclusive():
    assert fully_observed(D("2016-08-01"), 7, D("2016-09-07"), 30)
    assert not fully_observed(D("2016-08-01"), 7, D("2016-09-06"), 30)


def test_fit_snapshot_shifted_one_month_breaks_the_purge_rule():
    r = raw()
    r["folds"]["f7a"]["fit"] = [D("2016-09-01")]  # 2016-09-01 + 37 days = 2016-10-08 > as_of
    with pytest.raises(ProtocolError, match="purge rule"):
        parse_protocol(r)


def test_calibration_shifted_one_month_breaks_the_purge_rule():
    r = raw()
    r["folds"]["f7b_cal"]["calibration"]["snapshot"] = D("2016-10-01")
    with pytest.raises(ProtocolError, match="purge rule"):
        parse_protocol(r)


def test_test_pipeline_calibration_shifted_one_month_breaks_the_purge_rule():
    r = raw()
    r["test"]["calibration"]["snapshot"] = D("2016-11-01")  # + 59 + 30 days = 2017-01-29
    with pytest.raises(ProtocolError, match="purge rule"):
        parse_protocol(r)


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["folds"]["f7a"].update(fit=[D("2017-01-01")], as_of=D("2017-03-01")),
        lambda r: r["folds"]["f7b"]["score"].update(snapshot=D("2017-01-01")),
        lambda r: r.update(check_month=D("2017-02-01")),
        lambda r: r["test"].update(fit=[D("2016-08-01"), D("2017-01-01")]),
    ],
    ids=["fold fit", "fold score", "check month", "test fit"],
)
def test_locked_months_cannot_fill_a_fold_check_or_fit_role(edit):
    r = raw()
    edit(r)
    with pytest.raises(ProtocolError, match="locked date"):
        parse_protocol(r)


def test_test_month_before_the_locked_date_fails():
    r = raw()
    r["test"]["months"][30] = [D("2016-10-01")]
    with pytest.raises(ProtocolError, match="before the locked date"):
        parse_protocol(r)


def test_test_months_need_an_unlock():
    p = load_protocol()
    with pytest.raises(PermissionError):
        p.test_months(None)
    with pytest.raises(PermissionError):
        p.test_months(object())


def test_unlock_cannot_be_created_outside_the_guard():
    with pytest.raises(PermissionError):
        TestUnlock()
    # A function with the guard's module name and function name, but defined in
    # another file, is still refused.
    fake = {"__name__": GUARD_MODULE, "TestUnlock": TestUnlock}
    exec(compile(f"def {GUARD_FUNCTION}():\n    return TestUnlock()\n", "fake.py", "exec"), fake)
    with pytest.raises(PermissionError):
        fake[GUARD_FUNCTION]()


def test_only_the_guard_module_constructs_an_unlock():
    callers = set()
    for path in Path("src/retention").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "TestUnlock":
                callers.add(path.name)
    assert callers <= {"guard.py"}


def test_no_fold_or_check_role_names_a_locked_month():
    p = load_protocol()
    named = {p.check_month, *p.test_fit.fit, p.test_fit.calibration.snapshot}
    for f in p.folds.values():
        named |= {*f.fit, f.score.snapshot} | ({f.calibration.snapshot} if f.calibration else set())
    assert max(named) < p.locked_from


def test_halves_follow_md5_bucket_parity_and_never_overlap(tmp_path):
    ids = [f"customer{i}" for i in range(400)]
    rows = [(m, D(d), True, "paid") for d in ("2016-09-01", "2016-10-01") for m in ids]
    labels = pd.DataFrame(rows, columns=["msno", "snapshot_date", "in_exp30", "segment"])
    labels["churned_7"] = 1  # an outcome column that membership must not return
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path)
    p = load_protocol()
    mem = membership(path, [D("2016-10-01")], p, p2_main_only=True)
    assert "churned_7" not in mem.columns
    expected = {m: "selection" if bucket(m) % 2 == 0 else "calibration" for m in ids}
    assert dict(zip(mem.msno, mem.half, strict=True)) == expected
    both = membership(path, [D("2016-09-01"), D("2016-10-01")], p)
    assert both.groupby("msno").half.nunique().max() == 1
    assert 0 < (mem.half == "selection").sum() < len(ids)


def test_thread_setting_reaches_every_library():
    threads = load_protocol().threads
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200)
    booster = lgb.train({**threads.lightgbm(), "verbosity": -1}, lgb.Dataset(X, y), 2)
    assert booster.params["num_threads"] == 4
    bst = xgb.train(threads.xgboost(), xgb.DMatrix(X, y), 2)
    assert json.loads(bst.save_config())["learner"]["generic_param"]["nthread"] == "4"
    HistGradientBoostingClassifier(max_iter=2).fit(X, y)  # loads scikit-learn's OpenMP
    with threads.limit():
        pools = threadpool_info()
        assert pools and all(tp["num_threads"] == 4 for tp in pools)
