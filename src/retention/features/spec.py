"""Feature registry: every column of features.parquet, in one place.

Each column records its group (the unit Phase 3 drops in an ablation), whether
it is a model input, where it comes from, what it means, the window it reads,
the months that window touches, and why it can be null. The dictionary table
in docs/features.md is generated from this registry.

Features for snapshot T read only rows dated strictly before T.
"""

from __future__ import annotations

from dataclasses import dataclass

from retention.data.clean import AGE_RANGE

# Usage windows, in days, all ending on T - 1.
WINDOW_SHORT = 7
WINDOW_LONG = 28
# Baseline for "last week versus own earlier baseline": weeks 2-4 before T.
BASELINE_DAYS = WINDOW_LONG - WINDOW_SHORT
# Usage windows as inclusive day-offset ranges, offset = T - day (1 is T - 1).
# Day T itself (offset 0) is in no window. wk1 is the oldest week, wk4 the newest.
WINDOWS: dict[str, tuple[int, int]] = {
    "last7": (1, WINDOW_SHORT),
    "base21": (WINDOW_SHORT + 1, WINDOW_LONG),
    "28": (1, WINDOW_LONG),
    "wk1": (22, 28),
    "wk2": (15, 21),
    "wk3": (8, 14),
    "wk4": (1, 7),
}
WEEKS = ["wk1", "wk2", "wk3", "wk4"]  # oldest to newest
BASELINE_MIN_ACTIVE_DAYS = 3
# Fixed window for transaction counts (no growth with T, unlike all-history counts).
COUNT_WINDOW = 365

# total_secs cleaning, fixed constants applied to each daily row before any sum.
SECS_PER_DAY_MAX = 86_400  # values from here up to the sentinel limit are capped
SECS_SENTINEL_ABS = 1e15  # |value| at or above this, or below 0, is missing

# Age buckets: [lower, upper) edges over the valid range AGE_RANGE (inclusive).
AGE_BUCKET_EDGES = (AGE_RANGE[0], 18, 22, 26, 30, 35, 45, AGE_RANGE[1] + 1)

KEYS = ["msno", "snapshot_date"]

GROUPS = {
    "usage": "Listening in the 28 days before T, and recency of listening.",
    "labels_state": "Coverage state at T, copied from the labels table.",
    "segment_state": "Customer segment and paid state at T, copied from the labels table.",
    "transactions": "The last transaction before T and recency of transactions.",
    "counts_365": "Transaction counts in the 365 days before T.",
    "member_primary": "Registration date and channel from the members table.",
    "profile": "Self-reported profile fields from the members table (sensitivity only).",
    "metadata": "Row facts for filtering and diagnostics; never model inputs.",
}
# Groups outside the primary feature set: built and stored, used only in sensitivity runs.
SENSITIVITY_GROUPS = {"profile"}

PRE_TREATMENT = "pre-treatment discount history"

_LOGS_28 = "days T-28 to T-1; logs have rows on every day, so no completeness check"
_LOGS_ALL = "every day from 2015-01-01 to T-1; logs have rows on every day"
_STATE = "all rows before T; the last month is covered by pre_t_status"
_LAST_ROW = "from the month of the last transaction to T-1; see last_txn_span_status"
_COUNTS = (
    "the 365 days before T; from 2015-05-01 on this always includes an unreliable "
    "month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months"
)
_MEMBERS = "none: the members table is one undated snapshot taken after the data ends"
_MASK = "no log before T (member columns are masked), or no members row"


@dataclass(frozen=True)
class Column:
    name: str
    group: str
    source: str  # labels | usage | transactions | members | derived
    description: str
    window: str
    months: str
    null_reason: str = ""  # empty: never null
    window_days: int | None = None  # fixed window length, if any
    tags: tuple[str, ...] = ()

    @property
    def model_input(self) -> bool:
        return self.group not in ("metadata", "key")


def _usage(name: str, description: str, null_reason: str = "", window_days: int = 28) -> Column:
    window = {7: "last 7 days", 21: "days T-28 to T-8", 28: "last 28 days"}[window_days]
    return Column(name, "usage", "usage", description, window, _LOGS_28, null_reason, window_days)


COLUMNS: list[Column] = [
    Column("msno", "key", "labels", "Customer id.", "-", "-"),
    Column("snapshot_date", "key", "labels", "Snapshot date T.", "-", "-"),
    # --- usage -----------------------------------------------------------------
    Column(
        "has_any_log_before_T",
        "usage",
        "usage",
        "1 if the customer has any listening row dated before T.",
        "all history",
        _LOGS_ALL,
    ),
    Column(
        "days_since_last_listen",
        "usage",
        "usage",
        "Days from the last listening day to T.",
        "all history",
        _LOGS_ALL,
        "no listening before T (has_any_log_before_T = 0)",
    ),
    Column(
        "first_log_in_window",
        "usage",
        "usage",
        "1 if the first listening day ever falls in the 28 days before T (a newcomer).",
        "all history",
        _LOGS_ALL,
    ),
    _usage("active_days_last7", "Days with listening in the last 7 days.", window_days=7),
    _usage("active_days_base21", "Days with listening in weeks 2-4 before T.", window_days=21),
    _usage("active_days_28", "Days with listening in the last 28 days."),
    _usage("secs_per_day_last7", "Seconds listened per calendar day, last 7 days.", window_days=7),
    _usage("secs_per_day_base21", "Seconds listened per calendar day, weeks 2-4.", window_days=21),
    _usage("secs_per_day_28", "Seconds listened per calendar day, last 28 days."),
    _usage(
        "secs_per_active_day_28",
        "Seconds listened per active day, last 28 days.",
        "no listening in the last 28 days",
    ),
    _usage("plays_last7", "Songs played (any length), last 7 days.", window_days=7),
    _usage("plays_28", "Songs played (any length), last 28 days."),
    _usage(
        "unq_last7", "Sum over days of distinct songs played that day, last 7 days.", window_days=7
    ),
    _usage("unq_28", "Sum over days of distinct songs played that day, last 28 days."),
    _usage(
        "complete_share_last7",
        "Share of plays that reached the end of the song (num_100 / plays), last 7 days.",
        "no plays in the last 7 days",
        7,
    ),
    _usage(
        "complete_share_28",
        "Share of plays that reached the end of the song (num_100 / plays), last 28 days.",
        "no plays in the last 28 days",
    ),
    _usage(
        "ratio_secs_7_vs_base",
        "secs_per_day_last7 / secs_per_day_base21. Raw, never clipped.",
        "fewer than 3 active days in weeks 2-4, or no seconds recorded there",
    ),
    _usage(
        "ratio_active_7_vs_base",
        "(active_days_last7 / 7) / (active_days_base21 / 21). Raw, never clipped.",
        "fewer than 3 active days in weeks 2-4",
    ),
    _usage(
        "ratio_secs_vs_cohort",
        "ratio_secs_7_vs_base divided by its median over all customers at the same T "
        "with a valid ratio.",
        "ratio_secs_7_vs_base is null",
    ),
    _usage(
        "slope_secs_4w",
        "Least-squares slope of seconds per day over the four weekly blocks, "
        "oldest to newest, in seconds per day per week.",
        "newcomer (first_log_in_window = 1) or no listening in the last 28 days",
    ),
    _usage(
        "rel_slope_secs_4w",
        "slope_secs_4w divided by the mean of the four weekly points; between -1.2 and 1.2.",
        "as slope_secs_4w",
    ),
    _usage(
        "slope_active_4w",
        "Least-squares slope of active days per week over the four weekly blocks.",
        "newcomer (first_log_in_window = 1) or no listening in the last 28 days",
    ),
    _usage(
        "rel_slope_active_4w",
        "slope_active_4w divided by the mean of the four weekly points; between -1.2 and 1.2.",
        "as slope_active_4w",
    ),
    # --- copied from the labels table ---------------------------------------------
    Column(
        "days_until_expiry",
        "labels_state",
        "labels",
        "Days from T to the effective coverage expiry known at T.",
        "all history",
        _STATE,
    ),
    Column(
        "days_until_paid_expiry",
        "labels_state",
        "derived",
        "Days from T to the expiry of paid coverage known at T (negative if it has passed).",
        "all history",
        _STATE,
        "the customer never paid before T (paid_state = never_paid)",
    ),
    Column(
        "segment",
        "segment_state",
        "labels",
        "paid, discount100_only, promo_only or trial_only, from the row kinds before T.",
        "all history",
        _STATE,
    ),
    Column(
        "paid_state",
        "segment_state",
        "labels",
        "Paid coverage at T: active, grace (ended within 30 days), lapsed or never_paid.",
        "all history",
        _STATE,
    ),
    # --- transactions ---------------------------------------------------------------
    Column(
        "last_plan_days",
        "transactions",
        "transactions",
        "Plan length in days on the last non-cancel row before T; 0 means not recorded.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_plan_days_missing",
        "transactions",
        "transactions",
        "1 if last_plan_days is 0 (not recorded).",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_list_price",
        "transactions",
        "transactions",
        "List price on the last non-cancel row before T.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_amount_paid",
        "transactions",
        "transactions",
        "Amount paid on the last non-cancel row before T.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_discount_ratio",
        "transactions",
        "transactions",
        "1 - amount paid / list price on the last non-cancel row; negative when paid exceeds list.",
        "last row",
        _LAST_ROW,
        "list price is 0 on the last row",
        tags=(PRE_TREATMENT,),
    ),
    Column(
        "last_is_auto_renew",
        "transactions",
        "transactions",
        "Auto-renew flag on the last non-cancel row before T.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_payment_method_id",
        "transactions",
        "transactions",
        "Payment method on the last non-cancel row before T (categorical).",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "last_row_kind",
        "transactions",
        "transactions",
        "paid, discount100, promo or trial: kind of the last non-cancel row before T.",
        "last row",
        _LAST_ROW,
        tags=(PRE_TREATMENT,),
    ),
    Column(
        "last_row_is_cancel",
        "transactions",
        "transactions",
        "1 if the last row of any kind before T is a cancel.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "days_since_last_txn",
        "transactions",
        "transactions",
        "Days from the last non-cancel row to T.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "days_since_last_payment",
        "transactions",
        "transactions",
        "Days from the last non-cancel row with an amount paid above 0 to T.",
        "all history",
        _LAST_ROW,
        "no non-cancel row with an amount paid above 0 before T",
    ),
    *(
        Column(
            name,
            "counts_365",
            "transactions",
            description,
            "last 365 days",
            _COUNTS,
            null_reason,
            COUNT_WINDOW,
            tags,
        )
        for name, description, null_reason, tags in [
            ("n_txn_365", "Non-cancel rows in the 365 days before T.", "", ()),
            ("n_paid_365", "Rows with an amount paid above 0 in the 365 days before T.", "", ()),
            (
                "n_discounted_365",
                "Rows with 0 < paid < list price, or a 100% discount on a priced plan, "
                "in the 365 days before T.",
                "",
                (PRE_TREATMENT,),
            ),
            ("n_cancels_365", "Cancel rows in the 365 days before T.", "", ()),
            (
                "days_since_last_cancel",
                "Days from the last cancel row to T, within the 365 days before T.",
                "no cancel row in the 365 days before T",
                (),
            ),
            (
                "n_prior_lapses_365",
                "Non-cancel rows in the 365 days before T dated more than 30 days after "
                "the effective expiry left by the row before them (returns after a lapse).",
                "",
                (),
            ),
        ]
    ),
    # --- members ----------------------------------------------------------------------
    Column(
        "tenure_days",
        "member_primary",
        "members",
        "Days from the registration date to T.",
        "registration date",
        _MEMBERS,
        _MASK + ", or registration on or after T",
    ),
    Column(
        "registered_via",
        "member_primary",
        "members",
        "Registration channel code (categorical).",
        "-",
        _MEMBERS,
        _MASK,
    ),
    Column(
        "age",
        "profile",
        "members",
        f"Age in years if between {AGE_RANGE[0]} and {AGE_RANGE[1]}.",
        "-",
        _MEMBERS,
        _MASK + ", or age 0 or outside the valid range",
    ),
    Column(
        "age_missing",
        "profile",
        "members",
        "1 if age is 0 or outside the valid range.",
        "-",
        _MEMBERS,
        _MASK,
    ),
    Column("age_zero", "profile", "members", "1 if age is 0.", "-", _MEMBERS, _MASK),
    Column(
        "age_out_of_range",
        "profile",
        "members",
        "1 if age is not 0 and outside the valid range.",
        "-",
        _MEMBERS,
        _MASK,
    ),
    Column(
        "age_bucket",
        "profile",
        "members",
        "Age bucket: missing, 10-17, 18-21, 22-25, 26-29, 30-34, 35-44, 45-100.",
        "-",
        _MEMBERS,
        _MASK,
    ),
    Column("gender", "profile", "members", "male, female or missing.", "-", _MEMBERS, _MASK),
    Column("gender_missing", "profile", "members", "1 if gender is missing.", "-", _MEMBERS, _MASK),
    Column(
        "city",
        "profile",
        "members",
        "City code (categorical); code 1 mostly appears with an unfilled profile.",
        "-",
        _MEMBERS,
        _MASK,
    ),
    Column(
        "profile_blank",
        "profile",
        "members",
        "1 if age is missing, gender is missing and city is 1.",
        "-",
        _MEMBERS,
        _MASK,
    ),
    # --- metadata (never model inputs) -------------------------------------------------
    Column(
        "pre_t_status",
        "metadata",
        "derived",
        "Worst transaction-completeness status of the months touched by days T-28 to T-1.",
        "last 28 days",
        "the month(s) touched by days T-28 to T-1",
        window_days=WINDOW_LONG,
    ),
    *(
        Column(
            f"snapshot_usable_{h}",
            "metadata",
            "derived",
            f"label_reliable_{h} and pre_t_status = reliable.",
            f"T-28 to T+{h}",
            f"the month(s) touched by days T-28 to T+{h}",
        )
        for h in (7, 30, 60)
    ),
    Column(
        "last_txn_span_status",
        "metadata",
        "derived",
        "Worst completeness status of the months from the last non-cancel row to T-1.",
        "last row",
        _LAST_ROW,
    ),
    Column(
        "n_txn_365_in_nonreliable_months",
        "metadata",
        "derived",
        "Non-cancel rows in the 365 days before T dated in months that are not "
        "completeness-reliable.",
        "last 365 days",
        _COUNTS,
        window_days=COUNT_WINDOW,
    ),
    Column(
        "share_txn_365_in_nonreliable_months",
        "metadata",
        "derived",
        "n_txn_365_in_nonreliable_months / n_txn_365.",
        "last 365 days",
        _COUNTS,
        "no non-cancel row in the 365 days before T",
        COUNT_WINDOW,
    ),
    Column(
        "counts_365_days_observed",
        "metadata",
        "derived",
        "Days of the 365-day count window that fall on or after the data start "
        "(2015-01-01); below 365 for every snapshot before 2016-01-01.",
        "last 365 days",
        _COUNTS,
        window_days=COUNT_WINDOW,
    ),
    Column(
        "history_truncated",
        "metadata",
        "derived",
        "1 if tenure_days exceeds the days since 2015-01-01 plus 30: early history is not "
        "in the data.",
        "registration date",
        _MEMBERS,
        "tenure_days is null (including every row with no log before T)",
    ),
    Column(
        "has_member_record",
        "metadata",
        "members",
        "1 if the customer has a members row. Not a signal: presence follows log activity "
        "at any time, including after T.",
        "-",
        _MEMBERS,
        "no log before T (member columns are masked)",
    ),
    Column(
        "invalid_secs_days_28",
        "metadata",
        "usage",
        "Days in the last 28 with a total_secs value that was set to missing or capped.",
        "last 28 days",
        _LOGS_28,
        window_days=WINDOW_LONG,
    ),
]

BY_NAME = {c.name: c for c in COLUMNS}
FEATURE_COLUMNS: dict[str, str] = {c.name: c.group for c in COLUMNS if c.model_input}
PRIMARY_FEATURES = [n for n, g in FEATURE_COLUMNS.items() if g not in SENSITIVITY_GROUPS]
METADATA_COLUMNS = [c.name for c in COLUMNS if c.group == "metadata"]
LABEL_SOURCED = [c.name for c in COLUMNS if c.source == "labels" and c.group != "key"]
PRE_TREATMENT_COLUMNS = [c.name for c in COLUMNS if PRE_TREATMENT in c.tags]


def columns_in(group: str) -> list[str]:
    return [c.name for c in COLUMNS if c.group == group]


def longest_state_window() -> int:
    """Longest fixed window outside the 365-day counts: the pre-T completeness lookback."""
    return max(
        c.window_days
        for c in COLUMNS
        if c.window_days is not None and c.window_days != COUNT_WINDOW
    )


DICTIONARY_START = "<!-- dictionary:start -->"
DICTIONARY_END = "<!-- dictionary:end -->"


def dictionary_markdown() -> str:
    """The feature dictionary tables, one per group, for docs/features.md."""
    out: list[str] = []
    for group, title in [("key", "Keys"), *GROUPS.items()]:
        cols = [c for c in COLUMNS if c.group == group]
        out.append(f"### {group}\n")
        out.append(f"{title}\n" if group != "key" else "Keys of the table.\n")
        out.append("| Column | Source | Meaning | Window | Months touched | Null when |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for c in cols:
            meaning = c.description + (f" ({', '.join(c.tags)})" if c.tags else "")
            null = c.null_reason or "never"
            out.append(
                f"| `{c.name}` | {c.source} | {meaning} | {c.window} | {c.months} | {null} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_docs(text: str) -> str:
    """docs/features.md with the dictionary section replaced by the generated tables."""
    head, start, rest = text.partition(DICTIONARY_START)
    _, end, tail = rest.partition(DICTIONARY_END)
    if not start or not end:
        raise ValueError("docs/features.md lacks the dictionary markers")
    return f"{head}{DICTIONARY_START}\n\n{dictionary_markdown()}\n{DICTIONARY_END}{tail}"


# --- storage types: one arrow type per column -------------------------------------
_TYPE_GROUPS: dict[str, list[str]] = {
    "string": [
        "msno",
        "segment",
        "paid_state",
        "last_row_kind",
        "age_bucket",
        "gender",
        "pre_t_status",
        "last_txn_span_status",
    ],
    "date32": ["snapshot_date"],
    "bool": ["snapshot_usable_7", "snapshot_usable_30", "snapshot_usable_60"],
    "int8": [
        "has_any_log_before_T",
        "first_log_in_window",
        "last_plan_days_missing",
        "last_is_auto_renew",
        "last_row_is_cancel",
        "age_missing",
        "age_zero",
        "age_out_of_range",
        "gender_missing",
        "profile_blank",
        "has_member_record",
        "history_truncated",
    ],
    "int16": [
        "registered_via",
        "city",
        "age",
        "last_payment_method_id",
        "counts_365_days_observed",
    ],
    "int32": [
        "days_since_last_listen",
        "active_days_last7",
        "active_days_base21",
        "active_days_28",
        "plays_last7",
        "plays_28",
        "unq_last7",
        "unq_28",
        "invalid_secs_days_28",
        "days_until_expiry",
        "days_until_paid_expiry",
        "last_plan_days",
        "last_list_price",
        "last_amount_paid",
        "days_since_last_txn",
        "days_since_last_payment",
        "n_txn_365",
        "n_paid_365",
        "n_discounted_365",
        "n_cancels_365",
        "days_since_last_cancel",
        "n_prior_lapses_365",
        "n_txn_365_in_nonreliable_months",
        "tenure_days",
    ],
}
DTYPES: dict[str, str] = {name: kind for kind, names in _TYPE_GROUPS.items() for name in names} | {
    c.name: "float64" for c in COLUMNS if not any(c.name in v for v in _TYPE_GROUPS.values())
}

# --- allowed values of categorical and code columns ------------------------------------
STATUSES = ["reliable", "uncertain", "unreliable"]
VOCAB: dict[str, list] = {
    "segment": ["paid", "discount100_only", "promo_only", "trial_only"],
    "paid_state": ["active", "grace", "lapsed", "never_paid"],
    "last_row_kind": ["paid", "discount100", "promo", "trial"],
    "age_bucket": [
        *(
            f"{lo}-{hi - 1}"
            for lo, hi in zip(AGE_BUCKET_EDGES[:-1], AGE_BUCKET_EDGES[1:], strict=True)
        ),
        "missing",
    ],
    "gender": ["male", "female", "missing"],
    "pre_t_status": STATUSES,
    "last_txn_span_status": STATUSES,
    # Codes seen anywhere in the members and transactions tables (all dates).
    "city": [1, *range(3, 23)],
    "registered_via": [*range(1, 12), 13, 14, 16, 17, 18, 19],
    "last_payment_method_id": [*range(2, 9), *range(10, 42)],
}
CODE_COLUMNS = ["city", "registered_via", "last_payment_method_id"]


def arrow_schema():
    """The exact schema of features.parquet, in registry order."""
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "date32": pa.date32(),
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "float64": pa.float64(),
    }
    return pa.schema([(c.name, types[DTYPES[c.name]]) for c in COLUMNS])
