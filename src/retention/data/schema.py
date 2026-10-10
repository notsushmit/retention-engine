"""Explicit column types for the KKBox tables and SQL to cast raw values strictly.

Every raw value is read as text and cast with TRY_CAST or a strict YYYYMMDD
parse. A value that is present but does not cast is a structural error: it is
counted, never silently turned into NULL downstream.
"""

from __future__ import annotations

# Column -> DuckDB type. "DATE" columns hold YYYYMMDD text or integers in the raw files.
TRANSACTIONS = {
    "msno": "VARCHAR",
    "payment_method_id": "SMALLINT",
    "payment_plan_days": "SMALLINT",
    "plan_list_price": "INTEGER",
    "actual_amount_paid": "INTEGER",
    "is_auto_renew": "TINYINT",
    "transaction_date": "DATE",
    "membership_expire_date": "DATE",
    "is_cancel": "TINYINT",
}
MEMBERS = {
    "msno": "VARCHAR",
    "city": "SMALLINT",
    "bd": "INTEGER",
    "gender": "VARCHAR",
    "registered_via": "SMALLINT",
    "registration_init_time": "DATE",
}
LABELS = {"msno": "VARCHAR", "is_churn": "TINYINT"}

BINARY = {"is_auto_renew", "is_cancel", "is_churn"}
NON_NEGATIVE = {"payment_plan_days", "plan_list_price", "actual_amount_paid"}


def cast_expr(column: str, dtype: str) -> str:
    """SQL that casts the raw column; NULL when the raw value is missing or does not cast."""
    text = f"nullif(trim({column}::VARCHAR), '')"
    if dtype == "DATE":
        return (
            f"(case when regexp_full_match({text}, '[0-9]{{8}}') "
            f"then try_strptime({text}, '%Y%m%d')::DATE end)"
        )
    if dtype == "VARCHAR":
        return text
    return f"try_cast({text} as {dtype})"


def typed_select(schema: dict[str, str], prefix: str = "") -> str:
    """Typed columns plus `_raw_<col>` text copies used to count failed casts."""
    typed = [f"{cast_expr(prefix + c, t)} as {c}" for c, t in schema.items()]
    raw = [f"nullif(trim({prefix}{c}::VARCHAR), '') as _raw_{c}" for c in schema]
    return ", ".join(typed + raw)


def structural_checks(table: str, schema: dict[str, str]) -> dict[str, str]:
    """Named SQL counts that must all be zero for the table to be usable."""
    checks = {"null_msno": f"select count(*) from {table} where msno is null"}
    for col, dtype in schema.items():
        if col == "msno":
            continue
        label = "unparseable" if dtype == "DATE" else "uncastable"
        checks[f"{label}_{col}"] = (
            f"select count(*) from {table} where _raw_{col} is not null and {col} is null"
        )
        if col in BINARY:
            checks[f"not_binary_{col}"] = f"select count(*) from {table} where {col} not in (0, 1)"
        if col in NON_NEGATIVE:
            checks[f"negative_{col}"] = f"select count(*) from {table} where {col} < 0"
    return checks
