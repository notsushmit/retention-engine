"""Typed, validated KKBox tables for one customer sample.

Reads the sampled Parquet from `make data`, casts every column strictly, unions
v1 and v2 transactions, drops only fully identical rows, and adds row-level
flags. Nothing here depends on a snapshot date: customer-level segments are
computed later from rows dated before each snapshot.

Structural problems (missing IDs, values that do not cast, duplicate customers
in label or member tables, negative amounts, non-binary flags, placeholder
expiries with no frozen coverage) stop the run: validation.json is written but
no table is. Data-quality flags are counted and reported only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import duckdb

from retention.data.schema import LABELS, MEMBERS, TRANSACTIONS, structural_checks, typed_select
from retention.labels.config import LabelConfig

TX_DATA_COLUMNS = list(TRANSACTIONS)
TX_FLAGS = [
    "expiry_placeholder",
    "expiry_imputed",
    "expiry_far_future",
    "expiry_before_transaction",
    "sameday_sameplan_cancel",
]
ROW_KINDS = ["cancel", "paid", "discount100", "trial", "promo"]
# Ages outside this range are flagged, not changed; bd = 0 is flagged separately.
AGE_RANGE = (10, 100)
GENDERS = ("male", "female")


class ValidationError(RuntimeError):
    """Raised when structural checks fail; the report is still written."""


def _counts(con: duckdb.DuckDBPyConnection, checks: dict[str, str]) -> dict[str, int]:
    return {name: int(con.sql(sql).fetchone()[0]) for name, sql in checks.items()}


def _load_transactions(con: duckdb.DuckDBPyConnection, v1: Path, v2: Path) -> None:
    sel = typed_select(TRANSACTIONS)
    con.execute(
        f"""create table tx_raw as
        select 'v1' as source, file_row_number as source_row, {sel}
        from read_parquet(?, file_row_number = true)
        union all
        select 'v2' as source, file_row_number as source_row, {sel}
        from read_parquet(?, file_row_number = true)""",
        [str(v1), str(v2)],
    )


def _duplicates(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    cols = ", ".join(TX_DATA_COLUMNS)
    row = con.sql(
        f"""with g as (select {cols}, count(*) filter (where source = 'v1') n1,
                              count(*) filter (where source = 'v2') n2
                       from tx_raw group by all)
            select coalesce(sum(greatest(n1 - 1, 0)), 0), coalesce(sum(greatest(n2 - 1, 0)), 0),
                   count(*) filter (where n1 > 0 and n2 > 0),
                   coalesce(sum(n1 + n2 - 1), 0)
            from g"""
    ).fetchone()
    return {
        "identical_within_v1": int(row[0]),
        "identical_within_v2": int(row[1]),
        "identical_across_v1_v2": int(row[2]),
        "rows_dropped": int(row[3]),
    }


def _build_transactions(con: duckdb.DuckDBPyConnection, cfg: LabelConfig) -> None:
    cols = ", ".join(TX_DATA_COLUMNS)
    cov = (
        ", ".join(f"({m}, {p}, {d})" for (m, p), d in sorted(cfg.placeholder_coverage_days.items()))
        or "(null, null, null)"
    )
    con.execute(
        f"""create table tx_dedup as
        select * from tx_raw
        qualify row_number() over (partition by {cols} order by source, source_row) = 1"""
    )
    con.execute(
        f"""create table tx as
        with cov(payment_method_id, payment_plan_days, days) as (values {cov}),
        subs as (select distinct msno, transaction_date, payment_method_id, payment_plan_days,
                   plan_list_price
                 from tx_dedup where is_cancel = 0),
        base as (
          select d.source || ':' || d.source_row as row_id, d.source, d.source_row,
            {", ".join("d." + c for c in TX_DATA_COLUMNS)},
            d.membership_expire_date < date '{cfg.placeholder_expiry_before}' as expiry_placeholder,
            d.membership_expire_date > date '{cfg.far_future_expiry_after}' as expiry_far_future,
            cov.days as placeholder_days,
            s.msno is not null as same_day_same_plan_sub
          from tx_dedup d
          left join cov using (payment_method_id, payment_plan_days)
          left join subs s on s.msno = d.msno and s.transaction_date = d.transaction_date
            and s.payment_method_id = d.payment_method_id
            and s.payment_plan_days = d.payment_plan_days
            and s.plan_list_price = d.plan_list_price),
        flagged as (
          select * exclude (same_day_same_plan_sub),
            expiry_placeholder and is_cancel = 0 as expiry_imputed,
            not expiry_placeholder and is_cancel = 0
              and membership_expire_date < transaction_date - interval 1 day
              as expiry_before_transaction,
            is_cancel = 1 and same_day_same_plan_sub as sameday_sameplan_cancel,
            case when not expiry_placeholder then membership_expire_date
                 when is_cancel = 0
                   then (transaction_date + placeholder_days * interval 1 day)::DATE
            end as expiry_date
          from base)
        select * exclude (placeholder_days),
          expiry_date - transaction_date as coverage_days,
          case when is_cancel = 1 then 'cancel'
               when actual_amount_paid > 0 then 'paid'
               when plan_list_price > 0 then 'discount100'
               when expiry_date - transaction_date <= {cfg.trial_max_coverage_days} then 'trial'
               else 'promo' end as row_kind
        from flagged"""
    )


def _table_summary(
    con: duckdb.DuckDBPyConnection, table: str, date_cols: list[str]
) -> dict[str, Any]:
    rows, customers = con.sql(f"select count(*), count(distinct msno) from {table}").fetchone()
    out: dict[str, Any] = {"rows": rows, "customers": customers}
    for c in date_cols:
        lo, hi = con.sql(f"select min({c}), max({c}) from {table}").fetchone()
        out[c] = {"min": str(lo), "max": str(hi)}
    return out


def clean(interim: Path, out_dir: Path, cfg: LabelConfig) -> dict[str, Any]:
    """Build the typed tables in `out_dir`; raise ValidationError on structural problems."""
    t0 = time.time()
    con = duckdb.connect()
    con.sql("set enable_progress_bar = false")
    _load_transactions(con, interim / "transactions.parquet", interim / "transactions_v2.parquet")
    con.execute(
        f"create table members_raw as select {typed_select(MEMBERS)} from read_parquet(?)",
        [str(interim / "members_v3.parquet")],
    )
    for name in ("train", "train_v2"):
        con.execute(
            f"create table {name}_raw as select {typed_select(LABELS)} from read_parquet(?)",
            [str(interim / f"{name}.parquet")],
        )

    structural: dict[str, dict[str, int]] = {
        "transactions": _counts(con, structural_checks("tx_raw", TRANSACTIONS))
    }
    structural["members"] = _counts(con, structural_checks("members_raw", MEMBERS))
    structural["members"]["duplicate_msno"] = int(
        con.sql("select count(*) - count(distinct msno) from members_raw").fetchone()[0]
    )
    for name in ("train", "train_v2"):
        structural[name] = _counts(con, structural_checks(f"{name}_raw", LABELS))
        structural[name]["duplicate_msno"] = int(
            con.sql(f"select count(*) - count(distinct msno) from {name}_raw").fetchone()[0]
        )

    duplicates = _duplicates(con)
    _build_transactions(con, cfg)
    structural["transactions"]["placeholder_method_without_coverage"] = int(
        con.sql("select count(*) from tx where expiry_imputed and expiry_date is null").fetchone()[
            0
        ]
    )
    data_end = con.sql("select max(transaction_date) from tx").fetchone()[0]
    con.execute(
        f"""create table members as
        select msno, city, bd, gender, registered_via, registration_init_time,
          bd = 0 as age_zero,
          bd <> 0 and (bd < {AGE_RANGE[0]} or bd > {AGE_RANGE[1]}) as age_out_of_range,
          gender is null as gender_missing,
          gender is not null and gender not in {GENDERS} as gender_unexpected,
          registration_init_time > ?::DATE as registration_after_data_end
        from members_raw order by msno""",
        [data_end],
    )

    tx_flags = {
        f: int(con.sql(f"select count(*) from tx where {f}").fetchone()[0]) for f in TX_FLAGS
    }
    tx_flags.update(
        {
            f"row_kind_{k}": int(
                con.sql(f"select count(*) from tx where row_kind = '{k}'").fetchone()[0]
            )
            for k in ROW_KINDS
        }
    )
    member_flags = {
        f: int(con.sql(f"select count(*) from members where {f}").fetchone()[0])
        for f in [
            "age_zero",
            "age_out_of_range",
            "gender_missing",
            "gender_unexpected",
            "registration_after_data_end",
        ]
    }
    failures = {f"{t}.{k}": v for t, checks in structural.items() for k, v in checks.items() if v}
    report: dict[str, Any] = {
        "interim_dir": str(interim),
        "data_end": str(data_end),
        "structural_ok": not failures,
        "structural": structural,
        "structural_failures": failures,
        "quality_flags": {"transactions": tx_flags, "members": member_flags},
        "duplicates": duplicates,
        "tables": {
            "transactions": _table_summary(
                con, "tx", ["transaction_date", "membership_expire_date", "expiry_date"]
            ),
            "members": _table_summary(con, "members", ["registration_init_time"]),
            "train": _table_summary(con, "train_raw", []),
            "train_v2": _table_summary(con, "train_v2_raw", []),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if not failures:
        outputs = {
            "transactions": "select * from tx order by msno, transaction_date, source, source_row",
            "members": "select * from members",
            "train": "select msno, is_churn from train_raw order by msno",
            "train_v2": "select msno, is_churn from train_v2_raw order by msno",
        }
        for name, sql in outputs.items():
            tmp = out_dir / f"{name}.parquet.tmp"
            con.execute(f"copy ({sql}) to '{tmp}' (format parquet, compression zstd)")
            tmp.replace(out_dir / f"{name}.parquet")
        report["output_bytes"] = {n: (out_dir / f"{n}.parquet").stat().st_size for n in outputs}
    report["seconds"] = round(time.time() - t0, 1)
    write_validation(out_dir, "clean", report)
    if failures:
        raise ValidationError(f"structural checks failed: {failures}")
    return report


def write_validation(out_dir: Path, section: str, content: dict[str, Any]) -> Path:
    """Replace one section of validation.json, keeping the others."""
    path = out_dir / "validation.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = content
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return path
