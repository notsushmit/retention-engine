"""Usage features: seconds cleaning, window boundaries, ratios, slopes, cohort ratio."""

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from retention.features import spec
from retention.features.usage import UsageSource, slope_4w

T = date(2016, 8, 1)
LOG_COLS = [
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]


def log(msno: str, offset: int, secs: float = 1000.0, plays: int = 10, full: int = 4) -> tuple:
    """One customer-day `offset` days before T (offset 0 is T itself)."""
    d = T - timedelta(days=offset)
    return (msno, d.year * 10000 + d.month * 100 + d.day, plays - full, 0, 0, 0, full, 8, secs)


def build(rows, customers=None, population=None, t=T, **kw) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("raw_logs", pd.DataFrame(rows, columns=LOG_COLS))
    customers = customers or sorted({r[0] for r in rows})
    src = UsageSource(con, "raw_logs", customers)
    return src.features(t, population or customers, **kw).set_index("msno")


def test_secs_cleaning_before_aggregation():
    rows = [
        log("a", 3, secs=-9.223372e15),  # sentinel: missing, day still active
        log("a", 2, secs=1e5),  # over 24 hours: capped
        log("a", 1, secs=100.0),
        log("b", 1, secs=9.223372e15),  # positive sentinel: missing
        log("b", 2, secs=-5.0),  # negative: missing
    ]
    f = build(rows)
    assert f.loc["a", "active_days_last7"] == 3
    assert f.loc["a", "secs_per_day_last7"] == pytest.approx((0 + 86_400 + 100) / 7)
    assert f.loc["a", "invalid_secs_days_28"] == 2
    assert f.loc["a", "plays_last7"] == 30  # play counts untouched
    assert f.loc["b", "secs_per_day_last7"] == 0.0
    assert f.loc["b", "active_days_last7"] == 2
    assert f.loc["b", "invalid_secs_days_28"] == 2


# offset: (last7, base21, 28, week) membership; week None means no weekly block.
BOUNDARIES = {
    29: (0, 0, 0, None),
    28: (0, 1, 1, "wk1"),
    22: (0, 1, 1, "wk1"),
    21: (0, 1, 1, "wk2"),
    8: (0, 1, 1, "wk3"),
    7: (1, 0, 1, "wk4"),
    1: (1, 0, 1, "wk4"),
    0: (0, 0, 0, None),
}


def test_window_boundaries():
    rows = [log(f"c{o}", o) for o in BOUNDARIES]
    f = build(rows, keep_weeks=True)
    for o, (last7, base21, d28, week) in BOUNDARIES.items():
        r = f.loc[f"c{o}"]
        assert (r.active_days_last7, r.active_days_base21, r.active_days_28) == (
            last7,
            base21,
            d28,
        ), o
        for wk in spec.WEEKS:
            assert r[f"a_{wk}"] == int(wk == week), (o, wk)
    assert spec.WINDOWS["wk1"] == (22, 28) and spec.WINDOWS["wk2"] == (15, 21)
    assert spec.WINDOWS["base21"] == (8, 28) and spec.WINDOWS["last7"] == (1, 7)
    # Day T is in no window and does not count as listening before T.
    assert f.loc["c0", "has_any_log_before_T"] == 0
    assert pd.isna(f.loc["c0", "days_since_last_listen"])
    assert f.loc["c29", "days_since_last_listen"] == 29
    assert f.loc["c29", "first_log_in_window"] == 0  # first listen before the window
    assert f.loc["c28", "first_log_in_window"] == 1


def test_rows_on_or_after_t_are_ignored():
    base = [log("x", 1), log("x", 12)]
    later = [log("x", 0, secs=1e12, plays=10**6), log("x", -5, secs=-1.0)]
    a = build(base)
    b = build(base + later)
    pd.testing.assert_frame_equal(a, b)


def test_slope_signs_and_bounds():
    s, r = slope_4w(np.array([[100, 80, 60, 40], [40, 60, 80, 100]]))
    assert s[0] < 0 < s[1]
    x = 7.0
    _, rel = slope_4w(np.array([[0, 0, 0, x], [x, 0, 0, 0], [0, 0, 0, 0], [5, 5, 5, 5]]))
    assert rel[0] == pytest.approx(1.2) and rel[1] == pytest.approx(-1.2)
    assert np.isnan(rel[2])
    assert rel[3] == 0.0
    rng = np.random.default_rng(0)
    _, rel = slope_4w(rng.exponential(size=(10_000, 4)) * rng.integers(0, 2, size=(10_000, 4)))
    rel = rel[~np.isnan(rel)]
    assert rel.min() >= -1.2 - 1e-12 and rel.max() <= 1.2 + 1e-12


def test_sql_slope_matches_numpy_reference():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(1000):
        m = f"r{i}"
        rows.append(log(m, 40))  # first listen before the window: no newcomers
        p = rng.uniform(0, 1)
        for o in range(1, 29):
            if rng.uniform() < p:
                rows.append(log(m, o, secs=float(rng.exponential(3000))))
    for o in (3, 10, 17, 24):  # every week equal
        rows.append(log("flat", o, secs=500.0))
    rows.append(log("flat", 40))
    f = build(rows)
    daily = pd.DataFrame(rows, columns=LOG_COLS)
    daily["day"] = pd.to_datetime(daily.date.astype(str))
    daily["o"] = (pd.Timestamp(T) - daily.day).dt.days
    daily["total_secs"] = (daily.total_secs * 1000).round() / 1000  # whole milliseconds
    for kind, value in (("secs", "total_secs"), ("active", None)):
        pts = np.zeros((len(f), 4))
        idx = {m: i for i, m in enumerate(f.index)}
        for w, wk in enumerate(spec.WEEKS):
            lo, hi = spec.WINDOWS[wk]
            sel = daily[daily.o.between(lo, hi)]
            agg = sel.groupby("msno")[value].sum() / 7 if value else sel.groupby("msno").size()
            for m, v in agg.items():
                pts[idx[m], w] = v
        slope, rel = slope_4w(pts)
        none = f.active_days_28.to_numpy() == 0
        assert f[f"slope_{kind}_4w"].isna().to_numpy().tolist() == none.tolist()
        ok = ~none
        np.testing.assert_allclose(
            f[f"slope_{kind}_4w"].to_numpy()[ok], slope[ok], rtol=1e-12, atol=1e-9
        )
        np.testing.assert_allclose(
            f[f"rel_slope_{kind}_4w"].to_numpy()[ok], rel[ok], rtol=1e-12, atol=1e-9
        )
    assert f.loc["flat", "slope_secs_4w"] == pytest.approx(0.0, abs=1e-12)
    assert f.loc["flat", "rel_slope_active_4w"] == pytest.approx(0.0, abs=1e-12)


def test_baseline_rule_and_zero_last_week():
    rows = [log("two", o) for o in (9, 15)] + [log("two", 2)]
    rows += [log("three", o) for o in (9, 15, 23)]  # nothing last week
    rows += [log("nosecs", o, secs=-1.0) for o in (9, 15, 23)] + [log("nosecs", 3)]
    f = build(rows)
    assert pd.isna(f.loc["two", "ratio_secs_7_vs_base"])
    assert pd.isna(f.loc["two", "ratio_active_7_vs_base"])
    assert f.loc["three", "ratio_secs_7_vs_base"] == 0.0
    assert f.loc["three", "ratio_active_7_vs_base"] == 0.0
    assert pd.isna(f.loc["nosecs", "ratio_secs_7_vs_base"])  # baseline seconds all missing
    assert f.loc["nosecs", "ratio_active_7_vs_base"] == pytest.approx((1 / 7) / (3 / 21))


def test_newcomer_and_customer_without_logs():
    rows = [log("new", 10), log("new", 2)]
    f = build(rows, customers=["new", "silent"])
    assert f.loc["new", "first_log_in_window"] == 1
    assert pd.isna(f.loc["new", "slope_secs_4w"]) and pd.isna(f.loc["new", "rel_slope_active_4w"])
    s = f.loc["silent"]
    assert s.has_any_log_before_T == 0 and s.first_log_in_window == 0
    assert s.active_days_28 == 0 and s.secs_per_day_28 == 0.0 and s.plays_28 == 0
    for c in (
        "days_since_last_listen",
        "secs_per_active_day_28",
        "complete_share_28",
        "ratio_secs_7_vs_base",
        "ratio_secs_vs_cohort",
        "slope_secs_4w",
        "rel_slope_secs_4w",
    ):
        assert pd.isna(s[c]), c


def test_cohort_ratio_independent_of_chunks():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(301):
        for o in range(1, 29):
            if rng.uniform() < 0.5:
                rows.append(log(f"k{i}", o, secs=float(rng.exponential(2000))))
    customers = sorted({r[0] for r in rows} | {"nolog1", "nolog2"})
    one = build(rows, customers=customers, n_chunks=1)
    for n in (3, 7):
        pd.testing.assert_frame_equal(one, build(rows, customers=customers, n_chunks=n))
    valid = one.ratio_secs_vs_cohort.dropna()
    assert len(valid) > 200
    assert valid.median() == pytest.approx(1.0, abs=1e-12)
    ratio = one.ratio_secs_7_vs_base
    assert (one.ratio_secs_vs_cohort.isna() == ratio.isna()).all()
