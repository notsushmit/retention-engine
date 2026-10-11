# Features

One row per customer and snapshot date T, keyed one-to-one to the labels table (`msno`, `snapshot_date`): 4,306,893 rows for the 24 monthly snapshots of the 20% sample. Build it with `make features`; read it with `retention.features.build.read_features`, which gives nullable pandas types and no object columns.

## The one rule

Every feature for snapshot T reads only rows dated strictly before T. A transaction or listening row dated on T itself, or later, is never read. This holds for every column, including the `last_*` columns of customers in grace or already lapsed at T: their "last row" is the last one before T, never the renewal that arrives on or after T. Nothing about the renewal after T (its date, price, discount or kind) is a feature. The one table without dates, the members table, is handled separately below.

The tests check the rule by changing the data after T and requiring identical features: every row dated on or after T deleted; those rows replaced with extreme values, with extra rows dated exactly T for every customer; those rows moved to other customers; and the transaction rows before T put in reverse file order. They run on a small synthetic dataset in the test suite and on the real 20% sample (`scripts/check_feature_leakage.py`). Deliberately leaky versions of three features (a last transaction that reads day T, a 7-day window that includes day T, a cohort median taken over all snapshots at once) are caught by these tests in exactly the mutated column, which shows the tests can fail.

## Column groups

Each column belongs to one group, recorded in the registry (`src/retention/features/spec.py`, `FEATURE_COLUMNS`), so a model can drop a whole group in an ablation.

| Group | Use | Columns |
| --- | --- | --- |
| usage | model input | listening in the 28 days before T, recency, newcomer flag |
| labels_state | model input | days until the coverage and the paid expiry known at T |
| segment_state | model input; report results with and without | segment and paid state at T |
| transactions | model input | the last transaction before T, recency of transactions |
| counts_365 | model input; report results with and without | counts in the 365 days before T |
| member_primary | model input | registration date (as tenure) and channel |
| profile | sensitivity runs only | age, gender, city and profile flags |
| metadata | never a model input | data-quality and filtering facts about the row |

`days_until_expiry`, `segment` and `paid_state` are copied unchanged from the labels table; the build stops if they differ from it.

## Listening

The listening logs have one row per customer and day, and rows on every day from 2015-01-01 to 2017-03-31, so the usage features need no data-completeness check.

**Seconds.** `total_secs` is cleaned on each daily row before anything is summed: values below 0 and overflow values (absolute value 1e15 or more) are treated as missing and count as 0 seconds, and values above 86,400 (more than the seconds in a day) are capped at 86,400. A day with a missing seconds value still counts as an active day, because every log row has at least one play. The play counts are used as they are. `invalid_secs_days_28` (metadata) counts the days in the last 28 with a missing or capped value.

**Windows.** Every window ends on T - 1. A day's offset is T minus its date, so offset 1 is the day before T and day T itself is in no window.

| Window | Offsets | Days |
| --- | --- | --- |
| last 7 days (`last7`) | 1-7 | T-7 to T-1 |
| baseline (`base21`), weeks 2-4 | 8-28 | T-28 to T-8 |
| last 28 days (`28`) | 1-28 | T-28 to T-1 |
| wk1 (oldest week) | 22-28 | T-28 to T-22 |
| wk2 | 15-21 | T-21 to T-15 |
| wk3 | 8-14 | T-14 to T-8 |
| wk4 (newest week) | 1-7 | T-7 to T-1 |

Rates are per calendar day, so a silent day counts as zero, not as a gap.

**Last week against the customer's own baseline.** `ratio_secs_7_vs_base` is seconds per day in the last 7 days over seconds per day in weeks 2-4; `ratio_active_7_vs_base` does the same for active days. They are stored raw and never clipped. A ratio is null when the baseline has fewer than 3 active days, because one or two days are too few to call a habit; those customers are still described by `active_days_28` and `days_since_last_listen`. A customer who listened in the baseline but not last week gets 0, not null.

**Against everyone else at the same date.** `ratio_secs_vs_cohort` divides a customer's `ratio_secs_7_vs_base` by the median of that ratio over all customers in the labels table at the same T who have a valid ratio, computed from listening before T only. It removes movements that every customer shares in a given week. The median itself is not stored as a column: a value that is the same for every row of a snapshot would let a model recognise snapshots. A live scorer cannot recompute the median from one customer, so it needs a stored daily median, computed the same way over the population being scored.

**Trend.** `slope_secs_4w` is the least-squares slope of seconds per day across the four weekly blocks, oldest to newest:

    slope = (-1.5 * wk1 - 0.5 * wk2 + 0.5 * wk3 + 1.5 * wk4) / 5

in seconds per day per week, so a falling trend is negative. `rel_slope_secs_4w` divides it by the mean of the four points and always lies between -1.2 and 1.2 (1.2 when only the newest week has listening, -1.2 when only the oldest has). `slope_active_4w` and `rel_slope_active_4w` do the same for active days. Weeks without listening count as 0. The slopes are null for a newcomer whose first-ever listen falls inside the 28 days (`first_log_in_window` = 1), because the weeks before the first listen are not silence, and for anyone with no listening in the 28 days; the relative slope is also null when the mean is 0.

A customer who never listened before T gets 0 for counts and totals, null for recency, ratios and slopes, and `has_any_log_before_T` = 0, kept apart from newcomers.

## Transactions

The transaction features read the subscription timeline: every transaction in order, with the effective expiry after each row.

**The last row.** `last_*` columns describe the last non-cancel row before T. `last_plan_days` is 0 on blank-plan paid rows, where the plan length was not recorded; `last_plan_days_missing` marks them, and 0 there means "not recorded", not a zero-day plan. `last_discount_ratio` is 1 minus amount paid over list price; it is null when the list price is 0 (trials, promotions and blank-plan rows) and is kept as it is when negative (paid above list price).

**Fixed 365-day windows.** Counts look back 365 days, not over all history. All-history counts would grow with T just because the data starts on 2015-01-01, so training and test months would differ for no reason. For snapshots before 2016-01-01 part of the window lies before the data starts; `counts_365_days_observed` (metadata) records how much of it is observed. A prior lapse is a non-cancel row dated more than 30 days after the effective expiry left by the row before it.

**Exposure to months with missing transactions.** From 2015-05 on, every 365-day window includes at least one month in which transactions are missing. Two metadata columns record, per customer, how many of the counted transactions fall in months that are not completeness-reliable, and what share. The share falls as those months slide out of the window, so the earlier snapshots are about twice as exposed as the latest ones.

**Discount history before T.** `last_discount_ratio`, `n_discounted_365` and `last_row_kind` describe discounts the customer received before T. They are pre-treatment history: a model of the effect of a discount at the next renewal may use them to describe who was offered discounts before, but nothing about that next renewal is ever a feature.

## Same-day ties and exact sums

Some customers have two or more non-cancel rows on the same day with the same expiry, for example a payment and a 100% discount. The raw files give no order between such rows. The last row before T is then chosen by content: the highest amount paid, then list price, plan days, payment method, auto-renew and row kind, all descending; rows equal in all of these are interchangeable. The subscription timeline orders these rows by the same content rule, so no feature, effective expiry, spell or label depends on the order of rows in the raw files. On the 20% sample the rule decides the `last_*` values of 916 rows in all (at most 0.11% of a snapshot; 3 to 77 rows per snapshot from 2016-03 on), and the labels are unchanged by it.

Each day's listening seconds are rounded to whole milliseconds and summed as integers, so every sum is exact and the table is the same bit for bit however the database orders its additions. On the sample the rounding moves no value by more than 1e-10.

The transaction features read these timeline columns: `msno`, `seq`, `transaction_date`, `is_cancel`, `row_kind`, `payment_plan_days`, `plan_list_price`, `actual_amount_paid`, `is_auto_renew`, `payment_method_id`, `expiry_date` and `eff_expiry_after`. A prior lapse is a non-cancel row dated more than 30 days after the effective expiry left by the row before it; spell tables are not read.

## Metadata that exists only offline

`pre_t_status` and `snapshot_usable_7/30/60` are for choosing snapshots when training and evaluating, never model inputs. They cannot be computed at scoring time: the completeness status of month T-1 uses that month's churn rate, which is measured up to 30 days after its expiries, so it looks past T. `snapshot_usable_h` also uses the label reliability of the window after T.

## The members table

The members table (city, age, gender, registration channel and registration date) is a single undated snapshot taken after the data ends; registrations run to 2017-04-29. Its values are as they were then, not as they were at T. Whether a customer has a row in it at all appears to follow log activity at any time, including after T: among customers with no listening before T, almost all of those who listen later have a row and almost none of the others do. So presence or absence in this table is not a signal. Every column taken from it, including the metadata columns `has_member_record` and `history_truncated`, is null for customers with no listening before T, whether or not a row exists, and nothing derived from presence in this table may be used for filtering or reporting.

**What member features are.** `tenure_days` (days from registration to T, null if registration is on or after T) and `registered_via` are model inputs. Age, gender and city form the profile group, used only in sensitivity runs, because the profile may have been filled in after T. Age is kept when it is between 10 and 100; 0 and anything outside the range are missing, with flags (`age_zero`, `age_out_of_range`, `age_missing`). The buckets are 10-17, 18-21, 22-25, 26-29, 30-34, 35-44 and 45-100, plus "missing". Gender is male, female or "missing". `profile_blank` marks rows where age and gender are missing and city is 1; in this data city 1 occurs mostly together with a missing age and gender.

Age, gender and registration channel are not actionable and are not causes. They must not be presented as the reason a customer leaves, and they are never levers in "what would change this customer's outcome" analyses.

## Nulls

A null means the value is not defined or not observable at T; a 0 means it was measured and is zero. Nothing is imputed in this table. Categorical columns use an explicit "missing" category for a field that is empty, and null only where the whole source is unavailable (no members row, or masked). The null rule of every column is in the dictionary below. The build writes the share of nulls per column and snapshot (`features_null_report.parquet`) and stops if a column is null in more than 1% of any snapshot without a documented reason, or if a column documented as never null has a single null.

## Checks in the build

The table is written to a temporary file, checked, and only then renamed to `features.parquet`; a failed check leaves no partial table. Per snapshot, the build checks that every part has the labels table's rows in the same order, that columns copied from the labels table are identical to it, that member columns are null wherever there is no listening before T, and that independently computed groups agree (for example, `paid_state` is "never_paid" exactly when `days_since_last_payment` is null). On the written file it checks the exact column list and types, keys equal to the labels table in both directions, that every categorical value is in its allowed list, and the null report. Building the table twice, or one snapshot alone, gives the same values bit for bit.

## Feature dictionary

Generated from `src/retention/features/spec.py`; do not edit by hand.

<!-- dictionary:start -->

### key

Keys of the table.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `msno` | labels | Customer id. | - | - | never |
| `snapshot_date` | labels | Snapshot date T. | - | - | never |

### usage

Listening in the 28 days before T, and recency of listening.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `has_any_log_before_T` | usage | 1 if the customer has any listening row dated before T. | all history | every day from 2015-01-01 to T-1; logs have rows on every day | never |
| `days_since_last_listen` | usage | Days from the last listening day to T. | all history | every day from 2015-01-01 to T-1; logs have rows on every day | no listening before T (has_any_log_before_T = 0) |
| `first_log_in_window` | usage | 1 if the first listening day ever falls in the 28 days before T (a newcomer). | all history | every day from 2015-01-01 to T-1; logs have rows on every day | never |
| `active_days_last7` | usage | Days with listening in the last 7 days. | last 7 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `active_days_base21` | usage | Days with listening in weeks 2-4 before T. | days T-28 to T-8 | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `active_days_28` | usage | Days with listening in the last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `secs_per_day_last7` | usage | Seconds listened per calendar day, last 7 days. | last 7 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `secs_per_day_base21` | usage | Seconds listened per calendar day, weeks 2-4. | days T-28 to T-8 | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `secs_per_day_28` | usage | Seconds listened per calendar day, last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `secs_per_active_day_28` | usage | Seconds listened per active day, last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | no listening in the last 28 days |
| `plays_last7` | usage | Songs played (any length), last 7 days. | last 7 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `plays_28` | usage | Songs played (any length), last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `unq_last7` | usage | Sum over days of distinct songs played that day, last 7 days. | last 7 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `unq_28` | usage | Sum over days of distinct songs played that day, last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |
| `complete_share_last7` | usage | Share of plays that reached the end of the song (num_100 / plays), last 7 days. | last 7 days | days T-28 to T-1; logs have rows on every day, so no completeness check | no plays in the last 7 days |
| `complete_share_28` | usage | Share of plays that reached the end of the song (num_100 / plays), last 28 days. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | no plays in the last 28 days |
| `ratio_secs_7_vs_base` | usage | secs_per_day_last7 / secs_per_day_base21. Raw, never clipped. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | fewer than 3 active days in weeks 2-4, or no seconds recorded there |
| `ratio_active_7_vs_base` | usage | (active_days_last7 / 7) / (active_days_base21 / 21). Raw, never clipped. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | fewer than 3 active days in weeks 2-4 |
| `ratio_secs_vs_cohort` | usage | ratio_secs_7_vs_base divided by its median over all customers at the same T with a valid ratio. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | ratio_secs_7_vs_base is null |
| `slope_secs_4w` | usage | Least-squares slope of seconds per day over the four weekly blocks, oldest to newest, in seconds per day per week. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | newcomer (first_log_in_window = 1) or no listening in the last 28 days |
| `rel_slope_secs_4w` | usage | slope_secs_4w divided by the mean of the four weekly points; between -1.2 and 1.2. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | as slope_secs_4w |
| `slope_active_4w` | usage | Least-squares slope of active days per week over the four weekly blocks. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | newcomer (first_log_in_window = 1) or no listening in the last 28 days |
| `rel_slope_active_4w` | usage | slope_active_4w divided by the mean of the four weekly points; between -1.2 and 1.2. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | as slope_active_4w |

### labels_state

Coverage state at T, copied from the labels table.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `days_until_expiry` | labels | Days from T to the effective coverage expiry known at T. | all history | all rows before T; the last month is covered by pre_t_status | never |
| `days_until_paid_expiry` | derived | Days from T to the expiry of paid coverage known at T (negative if it has passed). | all history | all rows before T; the last month is covered by pre_t_status | the customer never paid before T (paid_state = never_paid) |

### segment_state

Customer segment and paid state at T, copied from the labels table.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `segment` | labels | paid, discount100_only, promo_only or trial_only, from the row kinds before T. | all history | all rows before T; the last month is covered by pre_t_status | never |
| `paid_state` | labels | Paid coverage at T: active, grace (ended within 30 days), lapsed or never_paid. | all history | all rows before T; the last month is covered by pre_t_status | never |

### transactions

The last transaction before T and recency of transactions.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `last_plan_days` | transactions | Plan length in days on the last non-cancel row before T; 0 means not recorded. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_plan_days_missing` | transactions | 1 if last_plan_days is 0 (not recorded). | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_list_price` | transactions | List price on the last non-cancel row before T. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_amount_paid` | transactions | Amount paid on the last non-cancel row before T. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_discount_ratio` | transactions | 1 - amount paid / list price on the last non-cancel row; negative when paid exceeds list. (pre-treatment discount history) | last row | from the month of the last transaction to T-1; see last_txn_span_status | list price is 0 on the last row |
| `last_is_auto_renew` | transactions | Auto-renew flag on the last non-cancel row before T. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_payment_method_id` | transactions | Payment method on the last non-cancel row before T (categorical). | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_row_kind` | transactions | paid, discount100, promo or trial: kind of the last non-cancel row before T. (pre-treatment discount history) | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `last_row_is_cancel` | transactions | 1 if the last row of any kind before T is a cancel. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `days_since_last_txn` | transactions | Days from the last non-cancel row to T. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `days_since_last_payment` | transactions | Days from the last non-cancel row with an amount paid above 0 to T. | all history | from the month of the last transaction to T-1; see last_txn_span_status | no non-cancel row with an amount paid above 0 before T |

### counts_365

Transaction counts in the 365 days before T.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `n_txn_365` | transactions | Non-cancel rows in the 365 days before T. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `n_paid_365` | transactions | Rows with an amount paid above 0 in the 365 days before T. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `n_discounted_365` | transactions | Rows with 0 < paid < list price, or a 100% discount on a priced plan, in the 365 days before T. (pre-treatment discount history) | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `n_cancels_365` | transactions | Cancel rows in the 365 days before T. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `days_since_last_cancel` | transactions | Days from the last cancel row to T, within the 365 days before T. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | no cancel row in the 365 days before T |
| `n_prior_lapses_365` | transactions | Non-cancel rows in the 365 days before T dated more than 30 days after the effective expiry left by the row before them (returns after a lapse). | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |

### member_primary

Registration date and channel from the members table.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `tenure_days` | members | Days from the registration date to T. | registration date | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row, or registration on or after T |
| `registered_via` | members | Registration channel code (categorical). | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |

### profile

Self-reported profile fields from the members table (sensitivity only).

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `age` | members | Age in years if between 10 and 100. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row, or age 0 or outside the valid range |
| `age_missing` | members | 1 if age is 0 or outside the valid range. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `age_zero` | members | 1 if age is 0. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `age_out_of_range` | members | 1 if age is not 0 and outside the valid range. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `age_bucket` | members | Age bucket: missing, 10-17, 18-21, 22-25, 26-29, 30-34, 35-44, 45-100. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `gender` | members | male, female or missing. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `gender_missing` | members | 1 if gender is missing. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `city` | members | City code (categorical); code 1 mostly appears with an unfilled profile. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |
| `profile_blank` | members | 1 if age is missing, gender is missing and city is 1. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked), or no members row |

### metadata

Row facts for filtering and diagnostics; never model inputs.

| Column | Source | Meaning | Window | Months touched | Null when |
| --- | --- | --- | --- | --- | --- |
| `pre_t_status` | derived | Worst transaction-completeness status of the months touched by days T-28 to T-1. | last 28 days | the month(s) touched by days T-28 to T-1 | never |
| `snapshot_usable_7` | derived | label_reliable_7 and pre_t_status = reliable. | T-28 to T+7 | the month(s) touched by days T-28 to T+7 | never |
| `snapshot_usable_30` | derived | label_reliable_30 and pre_t_status = reliable. | T-28 to T+30 | the month(s) touched by days T-28 to T+30 | never |
| `snapshot_usable_60` | derived | label_reliable_60 and pre_t_status = reliable. | T-28 to T+60 | the month(s) touched by days T-28 to T+60 | never |
| `last_txn_span_status` | derived | Worst completeness status of the months from the last non-cancel row to T-1. | last row | from the month of the last transaction to T-1; see last_txn_span_status | never |
| `n_txn_365_in_nonreliable_months` | derived | Non-cancel rows in the 365 days before T dated in months that are not completeness-reliable. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `share_txn_365_in_nonreliable_months` | derived | n_txn_365_in_nonreliable_months / n_txn_365. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | no non-cancel row in the 365 days before T |
| `counts_365_days_observed` | derived | Days of the 365-day count window that fall on or after the data start (2015-01-01); below 365 for every snapshot before 2016-01-01. | last 365 days | the 365 days before T; from 2015-05-01 on this always includes an unreliable month (2015-04, 2016-03 or 2016-05); see share_txn_365_in_nonreliable_months | never |
| `history_truncated` | derived | 1 if tenure_days exceeds the days since 2015-01-01 plus 30: early history is not in the data. | registration date | none: the members table is one undated snapshot taken after the data ends | tenure_days is null (including every row with no log before T) |
| `has_member_record` | members | 1 if the customer has a members row. Not a signal: presence follows log activity at any time, including after T. | - | none: the members table is one undated snapshot taken after the data ends | no log before T (member columns are masked) |
| `invalid_secs_days_28` | usage | Days in the last 28 with a total_secs value that was set to missing or capped. | last 28 days | days T-28 to T-1; logs have rows on every day, so no completeness check | never |

<!-- dictionary:end -->
