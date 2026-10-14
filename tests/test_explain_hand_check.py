"""Hand check: the draw, the raw recomputation of facts, the claim checks and caveats."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.explain.hand_check import (
    CLAIMS,
    LUNAR_NEW_YEAR,
    RawCustomer,
    assign_strata,
    claim_holds,
    daily_listening,
    draw,
    expected_caveats,
    load_hand_check_config,
    raw_facts,
    same,
    window_has,
)
from retention.explain.reasons import PHRASES
from retention.models.frames import half_mask
from retention.models.protocol import load_protocol

P = load_protocol()
HC = load_hand_check_config()
T = date(2016, 11, 1)


def test_lunar_new_year_window():
    assert window_has(date(2017, 2, 1), LUNAR_NEW_YEAR)
    assert not window_has(date(2017, 1, 1), LUNAR_NEW_YEAR)
    assert window_has(date(2017, 2, 25), LUNAR_NEW_YEAR)  # T - 28
    assert not window_has(date(2017, 2, 26), LUNAR_NEW_YEAR)
    assert not window_has(date(2017, 1, 28), LUNAR_NEW_YEAR)  # day T is in no window


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "last_row_kind": ["paid"] * n,
        "never_seen": [False] * n,
        "last_is_auto_renew": [1] * n,
        "top_decile_30": [False] * n,
        "last_row_is_cancel": [0] * n,
        "d_30": [1.0] * n,
        "has_any_log_before_T": [1] * n,
    }
    return pd.DataFrame({**base, **cols})


def test_strata_first_match_wins():
    rows = frame(
        last_row_kind=["trial", "paid", "paid", "paid", "paid", "paid", "paid", "paid"],
        never_seen=[True, True, False, False, False, False, False, False],
        top_decile_30=[True, False, True, False, False, False, False, False],
        last_is_auto_renew=[0, 1, 1, 0, 1, 1, 1, 1],
        last_row_is_cancel=[1, 0, 0, 0, 1, 0, 0, 0],
        d_30=[0.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0],
        has_any_log_before_T=[0, 1, 1, 1, 1, 1, 0, 1],
    )
    got = assign_strata(rows, HC["strata"], np.log(1.2)).tolist()
    assert got == [
        "zero_paid_last_row",  # trial beats every later rule
        "never_seen",
        "auto_renew_high_risk",
        "manual_payer",
        "last_row_cancel",
        "close_to_average",
        "no_listening_before_t",
        "any_other",
    ]


def test_draw_counts_and_determinism():
    rng = np.random.default_rng(0)
    rows = []
    for t in ("2016-10-01", "2016-11-01", "2017-01-01", "2017-02-01"):
        for i in range(200):
            rows.append({"msno": f"m{i:03d}", "snapshot_date": pd.Timestamp(t)})
    rows = pd.DataFrame(rows)
    names = [s["name"] for s in HC["strata"]]
    rows["stratum"] = rng.choice(names, size=len(rows))
    a, b = draw(rows, HC, 20161001), draw(rows, HC, 20161001)
    assert a.equals(b) and len(a) == 20
    counts = a.sample_stratum.value_counts().to_dict()
    for s in HC["strata"]:
        assert counts[s["name"]] == s["n"]
    assert counts["month 2017-01-01"] == 2 and counts["month 2017-02-01"] == 2
    behaviour = a[~a.sample_stratum.str.startswith("month")]
    assert set(behaviour.snapshot_date.dt.date.astype(str)) <= {"2016-10-01", "2016-11-01"}
    assert not draw(rows, HC, 20161002).equals(a)


def logs(days_secs: list[tuple[date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [d for d, _ in days_secs],
            "num_25": 1,
            "num_50": 0,
            "num_75": 0,
            "num_985": 0,
            "num_100": 2,
            "total_secs": [s for _, s in days_secs],
        }
    )


def test_daily_listening_cleans_seconds_like_the_features():
    d = T - timedelta(days=3)
    out = daily_listening(logs([(d, 100.0), (d, 2e15), (d, -5.0), (T - timedelta(days=2), 1e6)]))
    assert out.secs.tolist() == [100.0, 86_400.0]  # sentinel and negative count as 0; cap
    assert out.plays.tolist() == [9, 3] and out.full.tolist() == [6, 2]


def tx(rows):
    cols = [
        "seq",
        "d",
        "row_kind",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "payment_method_id",
        "membership_expire_date",
        "expiry",
        "is_cancel",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_raw_facts_from_transactions_and_logs():
    t = T
    trans = tx(
        [
            (0, date(2016, 8, 5), "paid", 30, 149, 149, 1, 41, None, date(2016, 9, 5), 0),
            (1, date(2016, 9, 5), "paid", 30, 149, 149, 1, 41, None, date(2016, 10, 5), 0),
            # same day, same expiry: the higher amount wins (as in the feature order)
            (2, date(2016, 10, 5), "discount100", 30, 149, 0, 0, 38, None, date(2016, 11, 5), 0),
            (3, date(2016, 10, 5), "paid", 30, 149, 99, 1, 41, None, date(2016, 11, 5), 0),
            (4, date(2016, 10, 20), "paid", 30, 149, 149, 1, 41, None, date(2016, 10, 20), 1),
        ]
    )
    # base window (days 8-28 before T): 2 listening days only; last 7 days: 3 days
    days = [(t - timedelta(days=k), 600.0) for k in (2, 3, 5, 10, 20)] + [(date(2016, 1, 3), 60.0)]
    raw = RawCustomer(
        trans,
        daily_listening(logs(days)),
        {"registration_init_time": date(2016, 4, 1)},
        date(2016, 11, 5),
    )
    f = raw_facts(t, raw)
    assert f["days_until_expiry"] == 4
    assert (f["last_row_kind"], f["last_amount_paid"], f["last_is_auto_renew"]) == ("paid", 99, 1)
    assert f["days_since_last_payment"] == (t - date(2016, 10, 5)).days
    assert f["last_row_is_cancel"] == 1
    assert f["active_days_28"] == 5 and f["days_since_last_listen"] == 2
    assert f["has_any_log_before_T"] == 1 and f["first_log_in_window"] == 0
    assert f["secs_per_day_28"] == 3000.0 / 28
    assert f["ratio_secs_7_vs_base"] is None  # 2 base days: below the 3-day minimum
    assert f["complete_share_28"] == 10 / 15
    assert f["tenure_days"] == (t - date(2016, 4, 1)).days


def test_tenure_is_masked_without_listening_and_newcomer_flag():
    no_logs = pd.DataFrame(columns=["day", "secs", "plays", "full"])
    raw = RawCustomer(tx([]), no_logs, {"registration_init_time": date(2015, 1, 1)}, None)
    f = raw_facts(T, raw)
    assert f["tenure_days"] is None and f["has_any_log_before_T"] == 0
    assert f["days_since_last_listen"] is None and f["first_log_in_window"] == 0
    raw2 = RawCustomer(tx([]), daily_listening(logs([(T - timedelta(days=4), 10.0)])), None, None)
    assert raw_facts(T, raw2)["first_log_in_window"] == 1


PHRASE_FACTS = yaml.safe_load(PHRASES.read_text())["groups"]


def test_missing_ratio_claim_accepts_fewer_than_three_base_days():
    spec = PHRASE_FACTS["listening_trend"]["facts"]["ratio_secs_7_vs_base"]
    text = spec["when_missing"]
    for days, ok in ((0, True), (1, True), (2, True), (3, False)):
        got, why = claim_holds("ratio_secs_7_vs_base", text, spec, {"_active_days_base21": days})
        assert got is ok and str(days) in why


def test_every_none_missing_zero_or_tiny_text_has_a_claim_rule():
    for group in PHRASE_FACTS.values():
        for col, spec in group["facts"].items():
            if not isinstance(spec, dict):
                continue
            for slot in ("when_zero", "when_missing", "when_tiny"):
                if spec.get(slot):
                    assert (col, slot) in CLAIMS, (col, slot)


def test_no_paid_plan_claim_ignores_cancel_rows_with_an_amount():
    trans = tx(
        [
            (0, date(2016, 10, 3), "discount100", 30, 149, 0, 0, 38, None, date(2016, 11, 2), 0),
            (1, date(2016, 10, 9), "paid", 30, 149, 149, 0, 38, None, date(2016, 11, 2), 1),
        ]
    )
    raw = raw_facts(
        T,
        RawCustomer(
            trans, pd.DataFrame(columns=["day", "secs", "plays", "full"]), None, date(2016, 11, 2)
        ),
    )
    assert raw["days_since_last_payment"] is None and raw["_paid_rows"] == 0
    spec = PHRASE_FACTS["renewal_timing"]["facts"]["days_since_last_payment"]
    assert claim_holds("days_since_last_payment", spec["when_missing"], spec, raw)[0]


def test_hand_check_label_reads_are_audited(tmp_path, monkeypatch):
    from retention.explain import hand_check as hc
    from retention.explain import model as em
    from retention.explain.__main__ import audit_label_reads, require_label_reads

    monkeypatch.setattr(hc, "load_labels", hc.load_labels)  # restored after the test
    monkeypatch.setattr(em, "load_labels", em.load_labels)
    labels = pd.DataFrame(
        {
            "msno": ["a", "b"],
            "snapshot_date": [T, T],
            "segment": ["paid", "paid"],
            "expiry_date": [date(2016, 11, 5), date(2016, 11, 20)],
            "churned_7": [0, 1],
        }
    )
    (tmp_path / "labels.parquet").write_bytes(b"")
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    reads: list[dict] = []
    with pytest.raises(RuntimeError):
        require_label_reads(reads)
    audit_label_reads(reads)
    assert hc.known_expiry(tmp_path, T, P) == {"a": date(2016, 11, 5), "b": date(2016, 11, 20)}
    assert hc.seen_customers(tmp_path, [T], P) == {"a", "b"}
    assert [r["columns"] for r in reads] == [["expiry_date"], ["segment"]]
    require_label_reads(reads)
    with pytest.raises(PermissionError):
        hc.load_labels(tmp_path / "labels.parquet", [T], ["churned_7"], P)


def test_same_and_caveats():
    assert same(None, float("nan")) and same(3, 3.0) and not same(None, 0)
    assert same(0.1 + 0.2, 0.3) and not same(1.0, 1.001) and same("paid", "paid")
    cal = next(
        m for m in (f"x{i}" for i in range(100)) if half_mask(pd.Series([m]), "calibration", P)[0]
    )
    sel = next(
        m
        for m in (f"x{i}" for i in range(100))
        if not half_mask(pd.Series([m]), "calibration", P)[0]
    )
    assert expected_caveats(date(2016, 10, 1), cal, P) == ("october_2016_calibration_half",)
    assert expected_caveats(date(2016, 10, 1), sel, P) == ("months_used_for_settings",)
    assert expected_caveats(date(2016, 11, 1), sel, P) == ("months_used_for_settings",)
    assert expected_caveats(date(2017, 1, 1), sel, P) == ()
    assert expected_caveats(date(2017, 2, 1), sel, P) == ("february_2017_gap",)
