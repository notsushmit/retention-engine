"""Shared helpers: write synthetic raw tables the way `make data` does, then clean them."""

from pathlib import Path

import duckdb
import pytest
import yaml

from retention.data.clean import clean
from retention.labels.config import parse_label_config
from valkey_guard import pytest_configure, pytest_runtest_makereport, valkey  # noqa: F401

CFG = parse_label_config(
    {
        **yaml.safe_load(Path("configs/kkbox.yaml").read_text())["labels"],
        "snapshots": {"dates": ["2016-01-01"]},
    }
)
TX_HEADER = (
    "msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
    "is_auto_renew,transaction_date,membership_expire_date,is_cancel"
)
MEMBERS = ["msno,city,bd,gender,registered_via,registration_init_time", "a,1,30,male,7,20150101"]
TRAIN = ["msno,is_churn", "a,0"]


def write_interim(tmp: Path, v1, v2=(), members=MEMBERS, train=TRAIN) -> Path:
    interim = tmp / "interim"
    interim.mkdir(parents=True)
    files = {
        "transactions": [TX_HEADER, *v1],
        "transactions_v2": [TX_HEADER, *v2],
        "members_v3": members,
        "train": train,
        "train_v2": train,
    }
    for name, lines in files.items():
        csv = tmp / f"{name}.csv"
        csv.write_text("\n".join(lines) + "\n")
        duckdb.sql(
            f"copy (select * from read_csv('{csv}', header = true, all_varchar = true)) "
            f"to '{interim / name}.parquet'"
        )
    return interim


@pytest.fixture
def clean_tx(tmp_path):
    """clean_tx(v1_rows, v2_rows) -> the clean transactions table as a DataFrame."""
    counter = iter(range(10**6))

    def run(v1, v2=()):
        tmp = tmp_path / f"run{next(counter)}"
        out = tmp / "out"
        clean(write_interim(tmp, v1, v2), out, CFG)
        return duckdb.sql(f"select * from '{out}/transactions.parquet'").df()

    return run
