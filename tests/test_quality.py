from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.data.quality import (
    classify,
    completeness_status,
    failures,
    pre_t_status,
    snapshot_status,
    thresholds,
)
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
    assert s2[(date(2016, 7, 1), 7)] == "reliable"  # labels ignore the lookback


def test_batch_split_always_creates_a_low_group():
    """Known weakness: with one regime the largest gap still splits off the lowest values."""
    t = table({k: (b, 0.05, 0.05, 0.2) for k, b in zip("abc", [8.0, 8.5, 9.0], strict=True)})
    assert failures(t, thresholds(t, 3.0)).batch.tolist() == [True, False, False]


# Output of `classify` on the 20% sample (month_quality.parquet), and the
# pre-T completeness status each month must get (batch, churn, listening only).
GOLDEN = [
    # month, assessed, fails, fails_low_k, fails_high_k, four-indicator status, completeness
    ("2015-01", "batch", "", "", "", "uncertain", "uncertain"),
    ("2015-02", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2015-03", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    (
        "2015-04",
        "batch,churn,listening,returns",
        "churn,listening,returns",
        "churn,listening,returns",
        "churn,listening,returns",
        "unreliable",
        "unreliable",
    ),
    ("2015-05", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2015-06", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2015-07", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2015-08", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    (
        "2015-09",
        "batch,churn,listening,returns",
        "returns",
        "churn,listening,returns",
        "returns",
        "uncertain",
        "uncertain",
    ),
    (
        "2015-10",
        "batch,churn,listening,returns",
        "batch",
        "batch",
        "batch",
        "uncertain",
        "uncertain",
    ),
    (
        "2015-11",
        "batch,churn,listening,returns",
        "batch",
        "batch",
        "batch",
        "uncertain",
        "uncertain",
    ),
    (
        "2015-12",
        "batch,churn,listening,returns",
        "batch",
        "batch",
        "batch",
        "uncertain",
        "uncertain",
    ),
    ("2016-01", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    (
        "2016-02",
        "batch,churn,listening,returns",
        "batch",
        "batch,listening",
        "batch",
        "uncertain",
        "uncertain",
    ),
    (
        "2016-03",
        "batch,churn,listening,returns",
        "batch,churn,listening",
        "batch,churn,listening",
        "batch,churn,listening",
        "unreliable",
        "unreliable",
    ),
    (
        "2016-04",
        "batch,churn,listening,returns",
        "batch",
        "batch,listening",
        "batch",
        "uncertain",
        "uncertain",
    ),
    (
        "2016-05",
        "batch,churn,listening,returns",
        "batch,listening",
        "batch,listening,returns",
        "batch",
        "unreliable",
        "unreliable",
    ),
    (
        "2016-06",
        "batch,churn,listening,returns",
        "batch",
        "batch",
        "batch",
        "uncertain",
        "uncertain",
    ),
    ("2016-07", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2016-08", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2016-09", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2016-10", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2016-11", "batch,churn,listening,returns", "", "", "", "reliable", "reliable"),
    ("2016-12", "batch,churn,listening,returns", "returns", "returns", "", "uncertain", "reliable"),
    ("2017-01", "batch,churn,listening", "", "", "", "reliable", "reliable"),
    ("2017-02", "batch,churn,listening", "", "", "", "reliable", "reliable"),
    ("2017-03", "batch,churn", "", "", "", "uncertain", "uncertain"),
]
GOLDEN_COLS = ["assessed", "fails", "fails_low_k", "fails_high_k", "status", "completeness"]


def golden() -> pd.DataFrame:
    return pd.DataFrame([g[1:] for g in GOLDEN], index=[g[0] for g in GOLDEN], columns=GOLDEN_COLS)


def test_completeness_status_golden():
    g = golden()
    got = completeness_status(g, Q)
    assert got.to_dict() == g.completeness.to_dict()
    assert list(g.index[got != g.status]) == ["2016-12"]  # the only month that changes


def test_completeness_needs_all_three_assessed():
    assert Q.min_assessed == 3  # all three completeness indicators must be assessed
    g = pd.DataFrame(
        {
            "assessed": ["batch,churn,listening", "batch,churn,returns", "batch,churn"],
            "fails": ["", "", ""],
            "fails_low_k": ["", "", ""],
            "fails_high_k": ["", "", ""],
        },
        index=["three", "returns_not_counted", "two"],
    )
    assert completeness_status(g, Q).to_dict() == {
        "three": "reliable",
        "returns_not_counted": "uncertain",
        "two": "uncertain",
    }


def test_completeness_ignores_returns_failures_and_flips():
    g = pd.DataFrame(
        {
            "assessed": ["batch,churn,listening,returns"] * 2,
            "fails": ["returns", "churn,returns"],
            "fails_low_k": ["returns", "churn,returns"],
            "fails_high_k": ["", "churn"],
        },
        index=["returns_only", "churn_plus_returns"],
    )
    assert completeness_status(g, Q).to_dict() == {
        "returns_only": "reliable",
        "churn_plus_returns": "uncertain",
    }


def test_pre_t_status_monthly_snapshots():
    months = completeness_status(golden(), Q)
    s = pre_t_status(months, [date(2016, 7, 1), date(2016, 8, 1), date(2017, 1, 1)], 28)
    assert s[date(2016, 7, 1)] == "uncertain"  # June 2016: halved month-end batch
    assert s[date(2016, 8, 1)] == "reliable"
    assert s[date(2017, 1, 1)] == "reliable"  # December 2016 passes without returns


def test_pre_t_status_window_spanning_two_months_takes_worse():
    months = pd.Series({"2016-05": "unreliable", "2016-06": "reliable", "2016-07": "reliable"})
    s = pre_t_status(months, [date(2016, 6, 10), date(2016, 7, 29), date(2016, 7, 1)], 28)
    assert s[date(2016, 6, 10)] == "unreliable"  # days 2016-05-13 to 2016-06-09
    assert s[date(2016, 7, 29)] == "reliable"  # days 2016-07-01 to 2016-07-28
    assert s[date(2016, 7, 1)] == "reliable"  # June only; T itself is not touched
    assert pre_t_status(months, [date(2015, 1, 1)], 28)[date(2015, 1, 1)] == "uncertain"
    with pytest.raises(ValueError):
        pre_t_status(months, [date(2016, 7, 1)], 0)


MONTH_QUALITY = Path("data/processed/kkbox/sample_020/month_quality.parquet")


@pytest.mark.skipif(not MONTH_QUALITY.exists(), reason="20% sample not built")
def test_golden_matches_sample_month_quality():
    m = pd.read_parquet(MONTH_QUALITY).set_index("month")
    g = golden()
    for col in ["assessed", "fails", "fails_low_k", "fails_high_k", "status"]:
        assert m[col].fillna("").to_dict() == g[col].to_dict(), col
