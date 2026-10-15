"""Daily batch indicator on hand-made data."""

import numpy as np
import pandas as pd

from retention.trends import batch

CFG = batch.IndicatorConfig(
    block_days=4,
    first_day=2,
    last_day=27,
    gap_below_ratio=0.60,
    baseline_months=3,
    min_baseline_values=2,
    clean_months=("2016-04", "2016-05"),
    max_false_flags=1,
    known_gap_days=(),
    known_non_gap_days=(),
    noise_mad_scale=1.4826,
)


def flat(first="2016-01-01", last="2016-06-30", normal=100.0, end_day=900.0) -> pd.Series:
    """Flat days of `normal`, with the month's last day carrying `end_day`."""
    idx = pd.date_range(first, last, freq="D")
    s = pd.Series(normal, index=idx)
    s[idx.is_month_end] = end_day
    return s


def row(ind: pd.DataFrame, day: str) -> pd.Series:
    d = pd.Timestamp(day)
    return ind[(ind.start <= d) & (ind.end >= d)].iloc[0]


def test_flat_series_has_no_flags() -> None:
    ind = batch.indicator(flat(), CFG)
    assert not ind.flag.any()
    assert np.isclose(row(ind, "2016-05-10").ratio, 1.0)


def test_single_day_gap_is_flagged_and_baseline_is_median() -> None:
    s = flat()
    s["2016-04-05"] = 30.0
    s["2016-05-05"] = 100.0
    ind = batch.indicator(s, CFG)
    assert row(ind, "2016-04-05").flag
    # the bad April day does not drag the May baseline: median of Feb, Mar, Apr
    assert row(ind, "2016-05-05").baseline == 100.0


def test_ratio_at_threshold_is_not_a_gap() -> None:
    s = flat()
    s["2016-04-05"] = 60.0
    assert not row(batch.indicator(s, CFG), "2016-04-05").flag
    s["2016-04-05"] = 59.0
    assert row(batch.indicator(s, CFG), "2016-04-05").flag


def test_batch_moved_inside_month_end_block_is_not_flagged() -> None:
    s = flat()
    s["2016-05-30"] = s["2016-05-31"]  # the 31st batch lands on the 30th
    s["2016-05-31"] = 100.0
    ind = batch.indicator(s, CFG)
    assert not row(ind, "2016-05-31").flag
    assert np.isclose(row(ind, "2016-05-31").ratio, 1.0)


def test_missing_month_end_block_is_flagged() -> None:
    s = flat()
    s["2016-05-31"] = 100.0  # the batch is missing
    ind = batch.indicator(s, CFG)
    r = row(ind, "2016-05-31")
    assert r.flag and r.kind == "block" and r.start == pd.Timestamp("2016-05-28")


def test_months_of_different_length_align_by_position_from_the_end() -> None:
    # 30-day April and 31-day May both carry the batch on their last day
    ind = batch.indicator(flat(), CFG)
    assert not ind[ind.kind == "block"].flag.any()
    assert ind[ind.kind == "block"].baseline.notna().sum() >= 2


def test_not_enough_history_means_not_judged() -> None:
    ind = batch.indicator(flat(), CFG)
    jan = ind[ind.month == pd.Period("2016-01")]
    assert jan.baseline.isna().all() and not jan.flag.any()


def test_gap_dates_and_week_flags() -> None:
    s = flat()
    s["2016-05-31"] = 100.0
    ind = batch.indicator(s, CFG)
    gaps = batch.gap_dates(ind)
    assert pd.Timestamp("2016-05-28") in gaps and pd.Timestamp("2016-05-31") in gaps
    weeks = pd.DatetimeIndex(["2016-04-25", "2016-05-02", "2016-06-27"])
    wf = batch.week_flags(weeks, gaps)
    # 2016-04-25 + 36 days = 2016-05-31 is inside; 2016-05-02 too; 2016-06-27 is after
    assert bool(wf["2016-04-25"]) and bool(wf["2016-05-02"]) and not bool(wf["2016-06-27"])


def test_weekday_rule_off_for_flat_and_on_for_strong_effect() -> None:
    cfg = batch.IndicatorConfig(
        **{**CFG.__dict__, "clean_months": tuple(f"2016-0{m}" for m in range(2, 7))}
    )
    rng = np.random.default_rng(1)
    s = flat()
    noisy = s * rng.normal(1.0, 0.03, len(s))
    assert not batch.weekday_multipliers(noisy, cfg)["use_weekday"]
    wd = pd.Series(np.where(s.index.dayofweek == 2, 1.4, 1.0), index=s.index)
    assert batch.weekday_multipliers(noisy * wd, cfg)["use_weekday"]


def test_deterministic() -> None:
    a = batch.indicator(flat(), CFG)
    b = batch.indicator(flat(), CFG)
    pd.testing.assert_frame_equal(a, b)
