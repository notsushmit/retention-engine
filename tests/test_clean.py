"""clean.py on small synthetic CSVs converted to Parquet the way `make data` writes them."""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from conftest import CFG, write_interim  # noqa: E402
from retention.data.clean import ValidationError, clean

GOOD_V1 = [
    "a,41,30,149,149,1,20160101,20160131,0",  # paid
    "a,41,30,149,149,1,20160101,20160131,0",  # identical duplicate within v1
    "a,41,30,149,149,1,20160131,20160301,0",  # same-day same-plan subscribe ...
    "a,41,30,149,149,1,20160131,20160131,1",  # ... and cancel
    "b,17,240,0,0,0,20160105,19700101,0",  # placeholder, non-cancel: imputed 3 days
    "b,24,30,149,149,1,20160106,19700101,1",  # placeholder, cancel: no expiry
    "c,35,7,0,0,0,20160110,20160117,0",  # trial (7 days)
    "c,41,30,0,0,0,20160120,20160219,0",  # promo (30 days free)
    "d,38,30,149,0,0,20160201,20160302,0",  # 100% discount
    "d,38,30,149,149,0,20160310,20160301,0",  # expiry before transaction
    "e,41,30,149,149,1,20160101,20290101,0",  # far future
]
GOOD_V2 = [
    "a,41,30,149,149,1,20160101,20160131,0",  # identical to a v1 row
    "a,41,30,149,129,1,20160101,20160131,0",  # same customer and date, different row: kept
]
MEMBERS = [
    "msno,city,bd,gender,registered_via,registration_init_time",
    "a,1,0,,7,20150101",
    "b,1,25,male,7,20150101",
    "c,1,999,other,7,20150101",
    "d,1,30,female,7,20170415",
    "e,1,30,female,7,20150101",
]
TRAIN = ["msno,is_churn", "a,0", "b,1"]


def write(tmp: Path, v1=GOOD_V1, v2=GOOD_V2, members=MEMBERS, train=TRAIN) -> Path:
    return write_interim(tmp, v1, v2, members, train)


@pytest.fixture
def good(tmp_path):
    out = tmp_path / "out"
    report = clean(write(tmp_path), out, CFG)
    return report, duckdb.sql(f"select * from '{out}/transactions.parquet'").df(), out


def test_clean_run_passes_and_counts_duplicates(good):
    report, tx, _ = good
    assert report["structural_ok"]
    assert report["duplicates"] == {
        "identical_within_v1": 1,
        "identical_within_v2": 0,
        "identical_across_v1_v2": 1,
        "rows_dropped": 2,
    }
    assert len(tx) == len(GOOD_V1) + len(GOOD_V2) - 2
    # same customer and date but a different amount is a different row, kept from v2
    assert ((tx.msno == "a") & (tx.actual_amount_paid == 129) & (tx.source == "v2")).sum() == 1


def test_row_id_is_source_and_row_number(good):
    _, tx, _ = good
    assert tx.row_id.is_unique
    first = tx[(tx.source == "v1") & (tx.source_row == 0)].iloc[0]
    assert first.row_id == "v1:0" and first.msno == "a"


def test_row_flags(good):
    report, tx, _ = good
    flags = report["quality_flags"]["transactions"]
    assert flags["expiry_placeholder"] == 2
    assert flags["expiry_imputed"] == 1
    assert flags["expiry_far_future"] == 1
    assert flags["expiry_before_transaction"] == 1
    assert flags["sameday_sameplan_cancel"] == 1
    b = tx[tx.msno == "b"].set_index("is_cancel")
    assert b.loc[0, "expiry_date"] == pd.Timestamp("2016-01-08")  # 2016-01-05 + 3 days (method 17)
    assert pd.isna(b.loc[1, "expiry_date"])  # a placeholder cancel never cuts coverage

    def kind(msno, day):
        return tx[(tx.msno == msno) & (tx.transaction_date == pd.Timestamp(day))].row_kind.item()

    assert kind("c", "2016-01-10") == "trial"
    assert kind("c", "2016-01-20") == "promo"
    assert kind("d", "2016-02-01") == "discount100"


def test_member_flags_raw_values_kept(good):
    report, _, out = good
    m = duckdb.sql(f"select * from '{out}/members.parquet'").df().set_index("msno")
    assert m.loc["c", "bd"] == 999 and m.loc["c", "gender"] == "other"
    assert report["quality_flags"]["members"] == {
        "age_zero": 1,
        "age_out_of_range": 1,
        "gender_missing": 1,
        "gender_unexpected": 1,
        "registration_after_data_end": 1,
    }


def test_validation_json_separates_structural_and_quality(good):
    _, _, out = good
    v = json.loads((out / "validation.json").read_text())["clean"]
    assert v["structural_ok"] and v["structural_failures"] == {}
    assert "quality_flags" in v and "structural" in v


@pytest.mark.parametrize(
    "v1, check",
    [
        (["  ,41,30,149,149,1,20160101,20160131,0"], "transactions.null_msno"),
        (["a,41,30,149,149,1,20160231,20160331,0"], "transactions.unparseable_transaction_date"),
        (["a,41,30,149,149,1,2016-01-01,20160131,0"], "transactions.unparseable_transaction_date"),
        (["a,41,30,149,149,1,20160101,abc,0"], "transactions.unparseable_membership_expire_date"),
        (["a,41,30,149,-5,1,20160101,20160131,0"], "transactions.negative_actual_amount_paid"),
        (["a,41,30,149,149,1,20160101,20160131,2"], "transactions.not_binary_is_cancel"),
        (["a,99,30,0,0,0,20160101,19700101,0"], "transactions.placeholder_method_without_coverage"),
        (["a,17,30,0,0,0,20160101,19700101,0"], "transactions.placeholder_method_without_coverage"),
    ],
)
def test_structural_failures_stop_the_run(tmp_path, v1, check):
    out = tmp_path / "out"
    with pytest.raises(ValidationError):
        clean(write(tmp_path, v1=v1, v2=[]), out, CFG)
    v = json.loads((out / "validation.json").read_text())["clean"]
    assert not v["structural_ok"]
    assert v["structural_failures"].get(check, 0) >= 1, v["structural_failures"]
    assert not (out / "transactions.parquet").exists()


def test_duplicate_msno_in_train_fails(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValidationError, match="train.duplicate_msno"):
        clean(write(tmp_path, train=["msno,is_churn", "a,0", "a,1"]), out, CFG)
