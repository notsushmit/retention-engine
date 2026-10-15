"""Alert pieces on hand-made cases: flags, contributions, templates and their structure."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.explain.reasons import ADVICE_WORDS, CAUSAL_WORDS
from retention.trends import alerts

CFG = yaml.safe_load(Path("configs/changepoints.yaml").read_text())["alerts"]
FIELD = re.compile(r"\{[a-z_0-9]+\}")


def alert(**kw) -> dict:
    base = {
        "series": "the paid series",
        "week": "2016-09-05",
        "before_rate": 0.049,
        "after_rate": 0.056,
        "direction": "up",
        "size_class": "shift",
        "series_returned_to_level": False,
        "return_week": None,
        "top_segments": [{"dimension": "plan_length", "value": "8-30"}],
        "flags": {
            "renewal_gap_dates": [],
            "unreliable_months": [],
            "uncertain_months": [],
            "left_truncation_risk": False,
            "known_periods": [],
        },
    }
    base.update(kw)
    return base


def test_contributions_sum_to_total_change_within_a_dimension() -> None:
    before = pd.DataFrame(
        {
            "dimension": ["plan"] * 2,
            "value": ["a", "b"],
            "events": [3000, 1000],
            "churned": [150, 100],
        }
    )
    after = pd.DataFrame(
        {
            "dimension": ["plan"] * 2,
            "value": ["a", "b"],
            "events": [2000, 2000],
            "churned": [120, 220],
        }
    )
    rows, small, totals = alerts.contributions(before, after, min_events=100)
    total = (120 + 220) / 4000 - (150 + 100) / 4000
    assert not small
    assert sum(r["rate_contribution"] + r["mix_contribution"] for r in rows) == pytest.approx(total)
    assert totals["plan"] == pytest.approx(total)
    # b's rate went 0.10 -> 0.11 at mean share 0.375; a's 0.05 -> 0.06 at mean share 0.625
    by = {r["value"]: r for r in rows}
    assert by["b"]["rate_contribution"] == pytest.approx(0.01 * 0.375)
    assert by["a"]["rate_contribution"] == pytest.approx(0.01 * 0.625)


def test_small_segments_are_listed_not_ranked_and_totals_still_add_up() -> None:
    before = pd.DataFrame(
        {"dimension": ["c"] * 2, "value": ["x", "y"], "events": [5000, 50], "churned": [250, 10]}
    )
    after = pd.DataFrame(
        {"dimension": ["c"] * 2, "value": ["x", "y"], "events": [5000, 60], "churned": [300, 12]}
    )
    rows, small, totals = alerts.contributions(before, after, min_events=1838)
    assert [r["value"] for r in rows] == ["x"]
    assert [s["value"] for s in small] == ["y"]
    assert totals["c"] == pytest.approx((312 / 5060) - (260 / 5050))


def test_flags_and_priority() -> None:
    wk = pd.Timestamp("2016-12-05")
    f = alerts.week_flags(wk, [pd.Timestamp("2016-12-05")], {"2016-12": "uncertain"}, CFG)
    assert f["renewal_gap_dates"] and f["uncertain_months"] == ["2016-12"]
    assert alerts.primary_template(alerts.merge_flags([f]), small=False) == "renewal_gap"
    f2 = alerts.week_flags(pd.Timestamp("2016-03-14"), [], {"2016-03": "unreliable"}, CFG)
    assert alerts.primary_template(alerts.merge_flags([f2]), small=False) == "unreliable_month"
    f3 = alerts.week_flags(pd.Timestamp("2016-09-12"), [], {}, CFG)
    assert f3["known_periods"] == ["the September 2016 level shift"]
    assert alerts.primary_template(alerts.merge_flags([f3]), small=True) == "known_period"
    assert (
        alerts.primary_template(
            alerts.merge_flags([alerts.week_flags(pd.Timestamp("2016-07-04"), [], {}, CFG)]),
            small=True,
        )
        == "small"
    )


def test_quiet_sd_uses_only_unflagged_weeks_of_long_segments() -> None:
    rates = np.array([0.05, 0.06] * 6 + [0.20, 0.21, 0.19])
    segs = [(0, 12), (12, 15)]
    flagged = np.zeros(15, dtype=bool)
    q = alerts.quiet_week_sd(rates, segs, flagged)
    assert q["segments"] == 1 and q["weeks"] == 12
    assert q["sd"] == pytest.approx(np.std([0.05, 0.06] * 6, ddof=1), rel=0.1)


def test_every_template_is_clean() -> None:
    texts = {**CFG["sentence"]["templates"], **CFG["sentence"]["pieces"]}
    forbidden = (
        [CAUSAL_WORDS, ADVICE_WORDS]
        + [rf"\b{re.escape(w)}\b" for w in CFG["sentence"]["forbidden_lever"]]
        + [re.escape(p) for p in CFG["sentence"]["forbidden_phrases"]]
    )
    for name, text in texts.items():
        fixed = FIELD.sub("", text)
        assert not re.search(r"\d", fixed), f"{name} has a digit outside a fact field"
        for pat in forbidden:
            assert not re.search(pat, fixed, flags=re.I), f"{name} matches {pat}"
        for field in FIELD.findall(text):
            assert field in text


def test_rendered_sentences_for_flagged_and_normal_alerts() -> None:
    gap = alert(
        flags={
            **alert()["flags"],
            "renewal_gap_dates": ["2016-12-05"],
            "gap_same_weeks": ["2016-12-05"],
            "uncertain_months": ["2016-12"],
        },
        series_returned_to_level=True,
        return_week="2016-12-12",
    )
    s = alerts.render(gap, CFG)
    assert s.startswith("This change coincides with a data gap on 2016-12-05;")
    assert "back within its earlier range by the week of 2016-12-12" in s
    assert "also in 2016-12, marked uncertain" in s
    for bad in ("churn shift", "churn worsen", "target", "focus on", "address", "fix", "act "):
        assert bad not in s
    normal = alerts.render(alert(), CFG)
    assert normal.startswith("The weekly churn rate for the paid series rose from 4.9% to 5.6%")
    assert (
        normal.endswith("largest rate changes are in plan length 8-30 days.")
        or "plan length" in normal
    )
    none = alerts.render(alert(top_segments=[]), CFG)
    assert "no segment met the minimum size" in none
    two = alerts.render(
        alert(
            top_segments=[
                {"dimension": "plan_length", "value": "8-30"},
                {"dimension": "city", "value": "5"},
            ]
        ),
        CFG,
    )
    assert "plan length 8-30 days and city 5" in two


@pytest.mark.parametrize("weeks", [1, 4])
def test_spike_alert_never_uses_normal_template_or_churn_shift(weeks: int) -> None:
    flags = {
        **alert()["flags"],
        "renewal_gap_dates": ["2016-12-05"],
        "gap_same_weeks": ["2016-12-05"],
    }
    for size_class in ("shift", "small"):
        a = alert(flags=flags, size_class=size_class, series_returned_to_level=weeks == 1)
        s = alerts.render(a, CFG)
        assert alerts.primary_template(flags, size_class == "small") != "normal"
        assert not s.startswith("The weekly churn rate")
        assert "churn shift" not in s and "churn worsen" not in s
        assert not re.search(r"\b(target|focus on|address|fix|act)\b", s)


def test_missing_city_and_channel_are_labelled_and_kept_in_the_sum_check() -> None:
    d = pd.DataFrame(
        {
            "last_plan_days": [30, 30, 7, 100],
            "last_payment_method_id": [41, 41, 38, 41],
            "last_is_auto_renew": [1, 1, 0, 1],
            "city": pd.array([1, None, 5, None], dtype="Int16"),
            "registered_via": pd.array([7, None, 9, 3], dtype="Int16"),
            "tenure_days": [10, 400, None, 900],
            "churned": [0, 1, 1, 0],
        }
    )
    cols = alerts.segment_columns(d)
    assert list(cols["city"]) == ["1", "missing", "5", "missing"]
    assert list(cols["registration_channel"]) == ["7", "missing", "9", "3"]
    assert list(cols["tenure"]) == ["0-89", "365-729", "missing", "730+"]
    for name, col in cols.items():
        assert col.notna().all() and len(col) == len(d), name
    # every dimension holds the same events, so totals agree
    frames = []
    for name, col in cols.items():
        g = (
            pd.DataFrame({"value": col, "churned": d.churned})
            .groupby("value")
            .churned.agg(events="size", churned="sum")
        )
        g.insert(0, "dimension", name)
        frames.append(g.reset_index())
    f = pd.concat(frames)
    assert f.groupby("dimension").events.sum().nunique() == 1


def test_returned_to_level_is_null_without_a_band_or_a_later_segment() -> None:
    assert alerts.returned_to_level(0.05, float("nan"), 0.05) == ([None, None], None)
    assert alerts.returned_to_level(0.05, None, 0.05) == ([None, None], None)
    band, ret = alerts.returned_to_level(0.05, 0.004, None)
    assert band == pytest.approx([0.042, 0.058]) and ret is None
    assert alerts.returned_to_level(0.05, 0.004, 0.055)[1] is True
    assert alerts.returned_to_level(0.05, 0.004, 0.07)[1] is False


def test_no_returned_fragment_when_not_computable() -> None:
    s = alerts.render(alert(series_returned_to_level=None), CFG)
    assert "back within its earlier range" not in s


def test_renewal_window_wording_and_short_window_never_nan() -> None:
    f = {**alert()["flags"], "renewal_gap_dates": ["2016-12-05"], "gap_same_weeks": []}
    s = alerts.render(alert(flags=f), CFG)
    assert s.startswith(
        "This change from the week of 2016-09-05 falls in the 30-day renewal window"
    )
    assert "data gap on 2016-12-05" in s and "coincides" not in s
    short = alert(before_rate=None, after_rate=None, size_class="not_computable", direction=None)
    t = alerts.render(short, CFG)
    assert "nan" not in t.lower() and "%" not in t
    assert t.startswith("A change point was detected from the week of 2016-09-05;")
