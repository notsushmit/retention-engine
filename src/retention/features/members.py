"""Member features from members_v3, masked for customers without pre-T listening.

members_v3 is one undated snapshot taken after the data ends, and whether a
customer has a row in it follows log activity at any time, including after T.
So every column taken from it is computed first and then masked as the last
step (`mask_members`): for rows with has_any_log_before_T = 0, all of them are
null, whether or not a members row exists. Their nullness then depends only on
listening before T.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from retention.data.clean import AGE_RANGE
from retention.features.spec import AGE_BUCKET_EDGES, columns_in

MEMBER_COLUMNS = [
    *columns_in("member_primary"),
    *columns_in("profile"),
    "has_member_record",
    "history_truncated",
]
AGE_BUCKETS = [
    f"{lo}-{hi - 1}" for lo, hi in zip(AGE_BUCKET_EDGES[:-1], AGE_BUCKET_EDGES[1:], strict=True)
]
HISTORY_SLACK_DAYS = 30


def member_features(
    members: pd.DataFrame, population: pd.DataFrame, t: date, data_start: date
) -> pd.DataFrame:
    """Member columns for every row of `population` (msno, has_any_log_before_T) at T."""
    m = members.set_index("msno")
    df = population[["msno", "has_any_log_before_T"]].reset_index(drop=True).join(m, on="msno")
    has = df.msno.isin(m.index).to_numpy()
    out = pd.DataFrame({"msno": df.msno.to_numpy()})
    reg = pd.to_datetime(df.registration_init_time)
    tenure = (pd.Timestamp(t) - reg).dt.days
    out["tenure_days"] = tenure.where(reg < pd.Timestamp(t)).astype("Int32").array
    out["registered_via"] = df.registered_via.astype("Int16").array

    bd = df.bd.astype("Int32")
    valid = bd.between(AGE_RANGE[0], AGE_RANGE[1])
    out["age"] = bd.where(valid).astype("Int16").array
    out["age_zero"] = (bd == 0).astype("Int8").array
    out["age_out_of_range"] = ((bd != 0) & ~valid).astype("Int8").array
    out["age_missing"] = (~valid).astype("Int8").array
    bucket = pd.cut(
        bd.where(valid).astype("float64"), list(AGE_BUCKET_EDGES), right=False, labels=AGE_BUCKETS
    )
    out["age_bucket"] = bucket.astype(object).where(valid, "missing").array
    gender = df.gender.where(df.gender.isin(["male", "female"]), "missing")
    out["gender"] = gender.array
    out["gender_missing"] = (gender == "missing").astype("Int8").array
    out["city"] = df.city.astype("Int16").array
    out["profile_blank"] = (
        (out.age_missing == 1) & (out.gender_missing == 1) & (out.city == 1)
    ).astype("Int8")
    out["has_member_record"] = pd.array(has.astype("int8"), dtype="Int8")
    limit = (t - data_start).days + HISTORY_SLACK_DAYS
    out["history_truncated"] = (out.tenure_days > limit).astype("Int8")

    # No members row: every value from the table is null, not a default.
    no_row = ~has
    for c in MEMBER_COLUMNS:
        if c != "has_member_record":
            out.loc[no_row, c] = None
    out[["age_bucket", "gender"]] = out[["age_bucket", "gender"]].astype(object)
    out = mask_members(out, df.has_any_log_before_T.to_numpy())
    return out[["msno", *MEMBER_COLUMNS]]


def mask_members(df: pd.DataFrame, has_any_log_before_t: np.ndarray) -> pd.DataFrame:
    """Null every members_v3 column, metadata included, where there is no log before T."""
    missing = [c for c in MEMBER_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"member columns missing before masking: {missing}")
    out = df.copy()
    masked = np.asarray(has_any_log_before_t) != 1
    for c in MEMBER_COLUMNS:
        out.loc[masked, c] = None
    return out
