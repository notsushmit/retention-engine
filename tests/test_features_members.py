"""Member features: age cleaning and buckets, gender, tenure, and the pre-T listening mask."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from retention.features.members import MEMBER_COLUMNS, mask_members, member_features

T = date(2016, 8, 1)
START = date(2015, 1, 1)


def members(rows) -> pd.DataFrame:
    cols = ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]
    df = pd.DataFrame(rows, columns=cols)
    df["registration_init_time"] = pd.to_datetime(df.registration_init_time)
    return df


def build(rows, logs=None, t=T) -> pd.DataFrame:
    m = members(rows)
    names = list(m.msno) + [n for n in (logs or {}) if n not in set(m.msno)]
    has = [int((logs or {}).get(n, 1)) for n in names]
    pop = pd.DataFrame({"msno": names, "has_any_log_before_T": has})
    return member_features(m, pop, t, START).set_index("msno")


AGES = {
    0: (None, 1, 0, 1, "missing"),
    9: (None, 0, 1, 1, "missing"),
    10: (10, 0, 0, 0, "10-17"),
    17: (17, 0, 0, 0, "10-17"),
    18: (18, 0, 0, 0, "18-21"),
    21: (21, 0, 0, 0, "18-21"),
    22: (22, 0, 0, 0, "22-25"),
    25: (25, 0, 0, 0, "22-25"),
    26: (26, 0, 0, 0, "26-29"),
    29: (29, 0, 0, 0, "26-29"),
    30: (30, 0, 0, 0, "30-34"),
    34: (34, 0, 0, 0, "30-34"),
    35: (35, 0, 0, 0, "35-44"),
    44: (44, 0, 0, 0, "35-44"),
    45: (45, 0, 0, 0, "45-100"),
    100: (100, 0, 0, 0, "45-100"),
    101: (None, 0, 1, 1, "missing"),
}


def test_age_cleaning_flags_and_bucket_edges():
    f = build([(f"a{bd}", 5, bd, "male", 9, "2014-01-01") for bd in AGES])
    for bd, (age, zero, oor, missing, bucket) in AGES.items():
        r = f.loc[f"a{bd}"]
        assert pd.isna(r.age) if age is None else r.age == age, bd
        assert (r.age_zero, r.age_out_of_range, r.age_missing) == (zero, oor, missing), bd
        assert r.age_bucket == bucket, bd


def test_gender_city_and_profile_blank():
    f = build(
        [
            ("blank", 1, 0, None, 7, "2014-01-01"),
            ("city1_only", 1, 30, "female", 7, "2014-01-01"),
            ("no_gender", 4, 0, None, 9, "2014-01-01"),
        ]
    )
    assert f.loc["blank", "gender"] == "missing" and f.loc["blank", "gender_missing"] == 1
    assert f.loc["blank", "profile_blank"] == 1
    assert f.loc["city1_only", "profile_blank"] == 0 and f.loc["city1_only", "gender"] == "female"
    assert f.loc["no_gender", "profile_blank"] == 0  # city is not 1
    assert f.loc["no_gender", "registered_via"] == 9 and f.loc["no_gender", "city"] == 4


def test_tenure_and_history_truncated():
    limit = (T - START).days + 30  # 608
    f = build(
        [
            ("day_before", 1, 0, None, 7, "2016-07-31"),
            ("on_t", 1, 0, None, 7, "2016-08-01"),
            ("after_t", 1, 0, None, 7, "2017-04-29"),
            ("trunc", 1, 0, None, 7, str(pd.Timestamp(T) - pd.Timedelta(days=limit + 1))[:10]),
            ("edge", 1, 0, None, 7, str(pd.Timestamp(T) - pd.Timedelta(days=limit))[:10]),
        ]
    )
    assert f.loc["day_before", "tenure_days"] == 1
    for m in ("on_t", "after_t"):
        assert pd.isna(f.loc[m, "tenure_days"]) and pd.isna(f.loc[m, "history_truncated"])
    assert f.loc["trunc", "history_truncated"] == 1 and f.loc["edge", "history_truncated"] == 0


def test_mask_nulls_every_member_column_without_pre_t_listening():
    rows = [
        ("quiet", 13, 30, "male", 9, "2012-05-01"),
        ("listener", 13, 30, "male", 9, "2012-05-01"),
    ]
    f = build(rows, logs={"quiet": 0, "listener": 1, "norow_quiet": 0, "norow_listener": 1})
    for c in MEMBER_COLUMNS:
        assert pd.isna(f.loc["quiet", c]), c
        assert pd.isna(f.loc["norow_quiet", c]), c
    assert f.loc["listener", "tenure_days"] > 0 and f.loc["listener", "has_member_record"] == 1
    assert f.loc["listener", "age"] == 30 and f.loc["listener", "history_truncated"] == 1
    nr = f.loc["norow_listener"]
    assert nr.has_member_record == 0
    for c in MEMBER_COLUMNS:
        if c != "has_member_record":
            assert pd.isna(nr[c]), c


def test_mask_members_directly():
    df = pd.DataFrame({c: [1, 2, 3] for c in MEMBER_COLUMNS})
    df.insert(0, "msno", ["a", "b", "c"])
    df["other"] = [7, 8, 9]
    out = mask_members(df, np.array([1, 0, 1]))
    assert out.loc[1, MEMBER_COLUMNS].isna().all()
    assert out.loc[[0, 2], MEMBER_COLUMNS].notna().all().all()
    assert out.other.tolist() == [7, 8, 9] and out.msno.tolist() == ["a", "b", "c"]
    assert df.loc[1, MEMBER_COLUMNS].notna().all()  # input not modified
    with pytest.raises(KeyError):
        mask_members(df.drop(columns="age"), np.array([1, 0, 1]))
