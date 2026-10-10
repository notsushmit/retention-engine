from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.data.quality import classify, failures, snapshot_status, thresholds
from retention.labels.config import parse_label_config

RAW = yaml.safe_load(Path("configs/kkbox.yaml").read_text())["labels"]
Q = parse_label_config({**RAW, "snapshots": {"dates": ["2016-01-01"]}}).data_quality
COLS = ["batch", "churn", "listening", "returns"]


def table(rows: dict[str, tuple]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLS)


# Twelve ordinary months: churn-like values 0.050 / 0.052 / 0.054, batch 8 / 8.5 / 9.
NORMAL = {
    f"2016-{m:02d}": (
        8 + (m % 3) * 0.5,
        0.05 + (m % 3) * 0.002,
        0.05 + (m % 3) * 0.002,
        0.2 + (m % 3) * 0.002,
    )
    for m in range(1, 13)
}


def test_thresholds_are_one_sided_median_plus_k_mad():
    t = table({**NORMAL, "2017-01": (9.0, 0.0, 0.0, 0.0)})
    th = thresholds(t, 3.0)
    v = t.churn
    mad = 1.4826 * (v - v.median()).abs().median()
    assert th["churn"]["fail_above"] == pytest.approx(v.median() + 3 * mad)
    assert not failures(t, th).loc["2017-01"].any()  # very low values never fail


def test_batch_split_at_largest_gap():
    t = table(
        {
            k: (b, 0.05, 0.05, 0.2)
            for k, b in zip("abcdef", [4.4, 4.9, 5.0, 7.0, 7.5, 8.0], strict=True)
        }
    )
    th = thresholds(t, 3.0)["batch"]
    assert (th["gap_low"], th["gap_high"], th["fail_below"]) == (5.0, 7.0, 6.0)
    assert failures(t, thresholds(t, 3.0)).batch.tolist() == [True, True, True, False, False, False]


def test_status_rules():
    t = table(
        {
            **NORMAL,
            "ok": (8.5, 0.052, 0.052, 0.202),
            "one_fail": (8.5, 0.30, 0.052, 0.202),
            "two_fail": (8.5, 0.30, 0.30, 0.202),
            "batch_low": (3.0, 0.052, 0.052, 0.202),
            "two_assessed": (8.5, 0.052, np.nan, np.nan),
            "three_assessed": (8.5, 0.052, 0.052, np.nan),
        }
    )
    out, _ = classify(t, Q)
    s = out.status
    assert s["ok"] == "reliable"
    assert s["one_fail"] == "uncertain"
    assert s["two_fail"] == "unreliable"
    assert s["batch_low"] == "uncertain" and out.fails["batch_low"] == "batch"
    assert s["two_assessed"] == "uncertain"  # fewer than 3 assessed
    assert s["three_assessed"] == "reliable"  # missing indicators are not failures
    assert out.n_assessed["three_assessed"] == 3


def test_month_that_flips_with_k_is_uncertain():
    t = table({**NORMAL, "edge": (8.5, 0.052, 0.052, 0.202)})
    th = thresholds(t, Q.mad_k)
    low = thresholds(t, Q.mad_k * (1 - Q.sensitivity))["churn"]["fail_above"]
    edge = (low + th["churn"]["fail_above"]) / 2  # fails at the lower k only
    t.loc["edge", "churn"] = edge
    out, _ = classify(t, Q)
    assert out.n_fail["edge"] == 0 and out.flips["edge"] and out.status["edge"] == "uncertain"


def test_snapshot_status_window_and_unknown():
    cfg = parse_label_config(
        {
            **RAW,
            "snapshots": {
                "dates": ["2016-07-01", "2016-11-01", "2017-01-01"],
                "allow_partial": True,
            },
        }
    )
    months = pd.Series(
        {
            "2016-06": "unreliable",
            "2016-07": "reliable",
            "2016-08": "reliable",
            "2016-09": "reliable",
            "2016-10": "reliable",
            "2016-11": "reliable",
            "2016-12": "uncertain",
            "2017-01": "reliable",
            "2017-02": "reliable",
        }
    )
    s = snapshot_status(months, cfg).set_index(["snapshot", "horizon"]).status
    assert s[(date(2016, 7, 1), 60)] == "reliable"  # July to Aug 30
    assert s[(date(2016, 11, 1), 7)] == "reliable"
    assert s[(date(2016, 11, 1), 30)] == "uncertain"  # 2016-11-01 + 30 days reaches Dec 1
    assert s[(date(2017, 1, 1), 30)] == "reliable"
    assert s[(date(2017, 1, 1), 60)] == "unknown"  # past the last labelable expiry
    with_lookback = parse_label_config(
        {**RAW, "lookback_days": 28, "snapshots": {"dates": ["2016-07-01"]}}
    )
    s2 = snapshot_status(months, with_lookback).set_index(["snapshot", "horizon"]).status
    assert s2[(date(2016, 7, 1), 7)] == "unreliable"  # lookback reaches June


def test_batch_split_always_creates_a_low_group():
    """Known weakness: with one regime the largest gap still splits off the lowest values."""
    t = table({k: (b, 0.05, 0.05, 0.2) for k, b in zip("abc", [8.0, 8.5, 9.0], strict=True)})
    assert failures(t, thresholds(t, 3.0)).batch.tolist() == [True, False, False]
