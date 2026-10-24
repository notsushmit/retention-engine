"""The drift code (configs/ops.yaml, R1): the registered rules, on hand-made and synthetic data.

The synthetic comparison is run on the synthetic dataset only. Its numbers are reported against the
registered margin; no test here asserts that the registered mark is met.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ops_group import needs_ops_group
from retention.api import ops as api_ops
from retention.features.spec import FEATURE_COLUMNS, PRIMARY_FEATURES
from retention.ops import drift as dr
from retention.synthetic import pipeline as pl

R1 = yaml.safe_load(Path("configs/ops.yaml").read_text())["drift"]


# --- the registered columns ------------------------------------------------------------------


def test_the_compared_columns_are_the_registered_47():
    cols = dr.compared_columns()
    assert len(cols) == 47 == int(str(R1["compared"]["count"]).split()[0])
    assert cols[-3:] == ["p7", "p30", "p59"] == R1["compared"]["predictions"]
    assert not set(cols) & set(dr.NEVER_COMPARED)
    assert set(cols[:-3]) == set(PRIMARY_FEATURES) - set(dr.NEVER_COMPARED)


def test_no_compared_column_names_a_never_used_attribute():
    from retention.api import schemas as sch

    assert sch.never_used_hits(" ".join(dr.compared_columns())) == []


def test_the_registered_label_is_the_one_the_config_holds():
    assert dr.LABEL == R1["label"]


# --- the registered rules --------------------------------------------------------------------


def report(share: float, alert: bool) -> dict:
    return {"drift_share": share, "dataset_alert": alert}


@pytest.mark.parametrize(
    ("unshifted", "shifted", "alert", "margin_met", "met"),
    [
        (0.05, 0.30, False, True, False),  # margin only: not met (both are required)
        (0.05, 0.60, True, True, True),
        (0.05, 0.24, True, False, False),  # alert only: not met
        (0.40, 0.60, True, True, True),  # exactly the margin is enough
        (0.40, 0.55, True, False, False),
    ],
)
def test_the_margin_and_alert_rules_need_both(unshifted, shifted, alert, margin_met, met):
    r = dr.margin_result(report(unshifted, False), report(shifted, alert))
    assert r["required"] == 0.20 == R1["pass_marks"]["margin_rule"]["margin"]
    assert (r["margin_met"], r["met"]) == (margin_met, met)
    assert r["difference"] == pytest.approx(shifted - unshifted)


def test_the_batch_size_band_is_mean_plus_minus_three_sample_deviations():
    ref = pd.Series([10, 12, 11, 9, 13, 10, 12])
    cur = pd.Series({20160901: 11, 20160902: 40, 20160903: 0})
    out = dr.batch_size_indicator(ref, cur)
    mean, sd = ref.mean(), ref.std(ddof=1)
    assert out["band_low"] == pytest.approx(mean - 3 * sd) and out["band_high"] == pytest.approx(
        mean + 3 * sd
    )
    assert out["days_outside_band"] == {"20160902": 40, "20160903": 0}
    assert out["mark"] == "none"


def test_a_code_column_has_the_same_labels_whether_it_is_a_float_or_an_integer_column():
    """Found after the first synthetic run: 38.0 and 38 were two categories."""
    as_float = pd.Series([38.0, 39.0, np.nan, 38.0])
    as_int = pd.Series([38, 39, 38, 38], dtype="int16")
    assert dr._category(as_float).dropna().tolist() == ["38", "39", "38"]
    assert dr._category(as_int).tolist() == ["38", "39", "38", "38"]
    got = dr._category(pd.Series(["paid", None, "grace"]))
    assert got.iloc[[0, 2]].tolist() == ["paid", "grace"] and pd.isna(got.iloc[1])


# --- the library, with the telemetry variables set -------------------------------------------


@needs_ops_group
def test_the_library_defaults_are_read_and_logged_before_a_report(tmp_path):
    with pytest.raises(RuntimeError, match="not logged yet"):
        dr.run_report(pd.DataFrame(), pd.DataFrame(), tmp_path)
    logged = json.loads(dr.log_defaults(tmp_path).read_text())
    assert logged["dataset_drift_share"] == 0.5
    assert logged["stattests"]["wasserstein"]["default_threshold"] == 0.1
    assert logged["stattests"]["ks"]["default_threshold"] == 0.05
    assert logged["evidently_version"] == "0.7.23"


@needs_ops_group
def test_a_tiny_report_flags_a_shifted_column_and_not_an_unshifted_one(tmp_path):
    dr.log_defaults(tmp_path)
    rng = np.random.default_rng(3)
    cols = dr.compared_columns()

    def frame(shift: dict[str, float]) -> pd.DataFrame:
        data = {c: rng.normal(shift.get(c, 0.0), 1, 400) for c in cols}
        for c in dr.CATEGORICAL:
            data[c] = rng.integers(0, 3, 400)
        return pd.DataFrame(data)

    r = dr.run_report(frame({}), frame({"plays_28": 3.0}), tmp_path)
    drifted = {c["column"] for c in r["columns"] if c["drifted"]}
    assert "plays_28" in drifted and "p7" not in drifted
    assert r["columns_compared"] == 47 and r["drift_share"] == r["columns_drifted"] / 47


# --- the synthetic comparison ----------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    from retention.ops.telemetry import telemetry_off
    from retention.synthetic import drift as sd

    telemetry_off()
    root = tmp_path_factory.mktemp("synthetic_drift")
    raw = yaml.safe_load(pl.CONFIG.read_text())
    raw["data"]["interim_dir"] = str(root / "interim")
    raw["labels"]["processed_dir"] = str(root / "processed")
    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    setup = pl.Setup(path, root=root)
    pl.run_all(setup, generated=root / "generated")
    out = setup.processed / "ops"
    with pytest.raises(RuntimeError, match="log the library defaults first"):
        sd.run(setup)
    dr.log_defaults(out)
    return setup, sd.run(setup), out


@needs_ops_group
def test_the_synthetic_comparison_compares_47_columns_and_writes_synthetic_labelled_files(
    synthetic_run,
):
    setup, result, out = synthetic_run
    for name in ("unshifted", "shifted"):
        d = json.loads((out / f"synthetic_drift_{name}.json").read_text())
        assert d["synthetic"] is True and d["columns_compared"] == 47
        assert d["shifted"] is (name == "shifted")
    assert (out / "SYNTHETIC").read_text().strip().startswith("Synthetic data generated")
    assert result["margin"]["required"] == 0.20


@needs_ops_group
def test_the_shift_moves_listening_columns_and_not_transaction_columns(synthetic_run):
    _, result, _ = synthetic_run
    usage = {c for c, g in FEATURE_COLUMNS.items() if g == "usage"}
    tx = {c for c, g in FEATURE_COLUMNS.items() if g == "transactions"}  # noqa: E501

    def drifted(r):
        return {c["column"] for c in r["columns"] if c["drifted"]}

    un, sh = drifted(result["unshifted"]), drifted(result["shifted"])
    assert len((sh - un) & usage) > 0
    assert (sh - un) & tx == set()  # transactions are never touched


@needs_ops_group
def test_the_unshifted_store_is_reported_whatever_it_shows(synthetic_run):
    _, result, _ = synthetic_run
    u = result["unshifted"]
    assert 0.0 <= u["drift_share"] <= 1.0 and isinstance(u["dataset_alert"], bool)
    assert result["margin"]["unshifted_alert"] is u["dataset_alert"]


@needs_ops_group
def test_the_drift_file_the_run_writes_is_served_by_the_api_reader(synthetic_run):
    _, result, out = synthetic_run
    body = api_ops.drift_body(out)
    assert body["status"] == "monitored" and body["synthetic"] is True
    assert body["columns_compared"] == {"count": 47, "hidden_by_rule": 0}
    assert body["label"].startswith(dr.LABEL) and "Synthetic data" in body["label"]
    assert body["drift_share"] == result["shifted"]["drift_share"]
    assert body["margin_met"] is result["margin"]["met"]
    assert len(body["columns_drifted"]) == result["shifted"]["columns_drifted"]


@needs_ops_group
def test_the_comparison_reads_no_outcome_and_names_no_customer(synthetic_run):
    _, _, out = synthetic_run
    text = "".join(f.read_text() for f in out.glob("*.json"))
    assert "synthetic_" not in text.replace("synthetic_drift", "").replace("synthetic_user", "")
    from retention.api import schemas as sch

    assert sch.never_used_hits(text) == []
    import inspect

    src = inspect.getsource(dr) + inspect.getsource(
        __import__("retention.synthetic.drift", fromlist=["x"])
    )
    assert "churned_" not in src and "days_to_event" not in src
