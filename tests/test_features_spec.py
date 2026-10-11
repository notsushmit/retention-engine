"""The feature registry: groups, model inputs, constants and the generated dictionary."""

import re
from pathlib import Path

import yaml

from retention.features import spec

LABEL_ONLY = {
    "expiry_date",
    "in_exp30",
    "in_expiry_month",
    "any_imputed_expiry",
    "stacked_far_future",
    "left_truncation_risk",
    "event_observed",
    "days_to_event",
    "paid_expiry_date",
    "paid_event_observed",
    "paid_days_to_event",
}
LABEL_PATTERNS = r"^(paid_)?churned_|^label_reliable_|^reliability_status_"
QUALITY = {
    "pre_t_status",
    "last_txn_span_status",
    "has_member_record",
    "history_truncated",
    "invalid_secs_days_28",
    "n_txn_365_in_nonreliable_months",
    "share_txn_365_in_nonreliable_months",
    "expiry_imputed",
    "expiry_placeholder",
}


def test_names_unique_and_complete():
    names = [c.name for c in spec.COLUMNS]
    assert len(names) == len(set(names))
    for c in spec.COLUMNS:
        assert c.group in {*spec.GROUPS, "key"}, c.name
        assert c.source in {"labels", "usage", "transactions", "members", "derived"}, c.name
        assert c.description and c.window and c.months, c.name


def test_feature_columns_hold_no_metadata_label_or_quality_column():
    features = set(spec.FEATURE_COLUMNS)
    assert not features & set(spec.METADATA_COLUMNS)
    assert not features & set(spec.KEYS)
    assert not features & LABEL_ONLY
    assert not [n for n in features if re.search(LABEL_PATTERNS, n)]
    assert not features & QUALITY
    assert set(spec.METADATA_COLUMNS) >= QUALITY & {c.name for c in spec.COLUMNS}


def test_groups_nonempty_and_profile_is_sensitivity_only():
    for group in spec.GROUPS:
        assert spec.columns_in(group), group
    profile = set(spec.columns_in("profile"))
    assert spec.SENSITIVITY_GROUPS == {"profile"}
    assert not profile & set(spec.PRIMARY_FEATURES)
    assert {"age", "gender", "city", "profile_blank"} <= profile
    assert set(spec.columns_in("member_primary")) == {"tenure_days", "registered_via"}


def test_pre_treatment_and_label_sourced_columns():
    assert set(spec.PRE_TREATMENT_COLUMNS) == {
        "last_discount_ratio",
        "n_discounted_365",
        "last_row_kind",
    }
    assert set(spec.LABEL_SOURCED) == {"days_until_expiry", "segment", "paid_state"}


def test_lookback_equals_longest_state_window():
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())["labels"]
    assert raw["lookback_days"] == spec.longest_state_window() == spec.WINDOW_LONG == 28
    # The 365-day counts are longer; they carry exposure metadata instead.
    for name in spec.columns_in("counts_365"):
        assert "share_txn_365_in_nonreliable_months" in spec.BY_NAME[name].months


def test_constants():
    assert spec.BASELINE_DAYS == 21
    assert spec.AGE_BUCKET_EDGES == (10, 18, 22, 26, 30, 35, 45, 101)
    assert spec.SECS_PER_DAY_MAX == 86_400 and spec.SECS_SENTINEL_ABS == 1e15


def test_docs_dictionary_is_generated_from_registry():
    text = Path("docs/features.md").read_text()
    assert spec.render_docs(text) == text, "run the docs generator"
    documented = set(re.findall(r"^\| `([^`]+)` \|", text, flags=re.M))
    assert documented == {c.name for c in spec.COLUMNS}
