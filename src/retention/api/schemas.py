"""The service's response models: every registered label is a required fixed value.

Each fixed string is a Literal built from retention.api.labels (configs/serving.yaml),
so a response missing a label, or carrying another value, cannot be built. Extra
fields are refused. The models also check the reading rules: probabilities never
fall from 7 to 30 to 59 days; amendment A4's pays and reads_as; D2's flag only on
a block that does not pay; no "recommend" in a block that does not pay; no "member
for" in a reason (amendment A3); December 2016's withheld reasons and warning; no
outside code naming a never-used attribute (amendment A5). response_hits is the
response scan with amendment A6's identifier exemption.

D1, stated here as the registry asks: the action block is computed from the 30-day
probability before the ordering fix, as in Phase 7, so it can differ from the
final 30-day probability for the rows the fix moves (7,632 training-population
rows on 2016-11-01).
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retention.api.labels import DASH, LABELS
from retention.optimize.config import load_actions_config

if LABELS.action_field != "highest_net_value_action":
    raise ValueError("the registered action field name differs from the schema's")

ACTION_NAMES = (*load_actions_config().action_order, "none")
ActionName = Literal.__getitem__(ACTION_NAMES)  # the allowed actions and "none"
SCORED, OUTSIDE, NOT_ACTIVE = LABELS.statuses
NEVER = re.compile(r"\b(" + "|".join(map(re.escape, LABELS.never_in_any_body)) + r")\b", re.I)


def never_used_hits(text: str) -> list[str]:
    """Every never-used attribute named in `text` (whole words, any letter case)."""
    return [m.group(0) for m in NEVER.finditer(text)]


def response_hits(text: str, status_code: int, own_id: str | None) -> list[str]:
    """The response scan under amendment A6 (configs/serving.yaml, msno_scan_exemption).

    An identifier exemption, not a relaxation of the never-used rule: only the value
    of the top-level msno field is skipped, and only when it equals `own_id`, the
    requested customer's own ID (for an /at-risk row, scanned row by row, the listed
    customer's own ID). Every error body (status 400 and above), every key, every
    other field, the same string anywhere else, a nested msno, and another
    customer's ID in msno are scanned in full.
    """
    if status_code >= 400 or own_id is None:
        return never_used_hits(text)
    try:
        body = json.loads(text)
    except ValueError:
        return never_used_hits(text)
    if not isinstance(body, dict):
        return never_used_hits(text)
    hits = []
    for key, value in body.items():
        hits += never_used_hits(json.dumps(key))
        if key == LABELS.msno_exempt_field and isinstance(value, str) and value == own_id:
            continue
        hits += never_used_hits(json.dumps(value))
    return hits


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- the scored response ------------------------------------------------------------------


class Probability(Strict):
    probability: float = Field(ge=0.0, le=1.0)
    text: str


class Probability59(Probability):
    grade: Literal[LABELS.grade_59]


class Risk(Strict):
    days_7: Probability
    days_30: Probability
    days_59: Probability59
    moved_by_ordering_fix: bool

    @model_validator(mode="after")
    def ordered(self) -> Risk:
        p = (self.days_7.probability, self.days_30.probability, self.days_59.probability)
        if not p[0] <= p[1] <= p[2]:
            raise ValueError("final probabilities must not fall from 7 to 30 to 59 days")
        return self


class Reason(Strict):
    horizon: Literal[7, 30]
    probability_before_fix: float = Field(ge=0.0, le=1.0)
    text: str
    explains: Literal[LABELS.reasons_fixed_line]

    @model_validator(mode="after")
    def no_member_for(self) -> Reason:
        if "member for" in self.text.lower():
            raise ValueError("a reason may not name how long the customer has been a member")
        return self


class PaidDays(Strict):
    field: Literal[LABELS.paid_days_field]
    value: float
    name: Literal[LABELS.paid_days_name]
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def registered_labels(self) -> PaidDays:
        if self.labels != LABELS.paid_days_labels:
            raise ValueError("the paid-days labels differ from the registered ones")
        return self


class Archetype(Strict):
    value: None
    status: Literal[LABELS.archetype_status]


class Scenario(Strict):
    form: str
    level: str
    margin: float
    cost_level: str
    discount_cost_rule: Literal[LABELS.scenario_text["discount_cost_rule"]]
    value_horizon: Literal[LABELS.scenario_text["value_horizon"]]
    budget: Literal[LABELS.scenario_text["budget"]]


class ActionBlock(Strict):
    """Computed from the 30-day probability before the ordering fix, as in Phase 7; it can
    differ from the final 30-day probability for the rows the fix moves."""

    highest_net_value_action: ActionName
    expected_retained_value: float
    net_value: float
    cost: float
    pays: bool
    reads_as: Literal[LABELS.reads_as] | None = None
    price: Literal[LABELS.no_price_flag] | None = None
    effect_basis: Literal[LABELS.effect_basis]
    scenario: Scenario
    instability_note: Literal[LABELS.instability_note]
    no_estimate_line: Literal[LABELS.no_estimate_line]
    probability_basis: Literal[LABELS.probability_basis]

    @model_validator(mode="after")
    def reading(self) -> ActionBlock:
        if self.pays != (self.net_value > 0):
            raise ValueError("pays must be true exactly when the net value is above 0")
        if self.pays == (self.reads_as is not None):
            raise ValueError("reads_as is shown exactly when no action pays")
        if self.price is not None and self.pays:
            raise ValueError("a customer without a usable price cannot have an action that pays")
        if self.highest_net_value_action == "none" and (
            self.expected_retained_value or self.net_value or self.cost or self.pays
        ):
            raise ValueError("no allowed action means zero values and no pay")
        if not self.pays:
            body = json.dumps(self.model_dump(mode="json")).lower()
            for word in LABELS.never_in_unpaid:
                if word in body:
                    raise ValueError("a block that does not pay may not recommend anything")
        return self


class ModelBlock(Strict):
    registered_model: str
    version: int
    files: dict[str, str]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def registered_files(self) -> ModelBlock:
        if self.files != LABELS.bundle:
            raise ValueError("the model files differ from the registered bundle")
        return self


class Scored(Strict):
    status: Literal[SCORED]
    msno: str
    features_as_of: date
    risk: Risk
    reasons: tuple[Reason, Reason] | None
    reasons_withheld: Literal[LABELS.reasons_withheld_code] | None = None
    data_warning: Literal[LABELS.december_warning] | None = None
    paid_days: PaidDays
    archetype: Archetype
    action: ActionBlock
    model: ModelBlock

    @model_validator(mode="after")
    def december(self) -> Scored:
        dec = self.features_as_of.year == 2016 and self.features_as_of.month == 12
        if dec:
            if self.reasons is not None or self.reasons_withheld is None:
                raise ValueError("reasons are withheld in December 2016")
            if self.data_warning is None:
                raise ValueError("December 2016 scores carry the data warning")
        else:
            if self.reasons is None or self.reasons_withheld or self.data_warning:
                raise ValueError("outside December 2016 both reasons are shown, no warning")
            if tuple(r.horizon for r in self.reasons) != (7, 30):
                raise ValueError("reasons come at 7 and 30 days, in that order")
        return self


# --- the other answers ---------------------------------------------------------------------


class Outside(Strict):
    status: Literal[OUTSIDE]
    msno: str
    features_as_of: date
    code: str
    next_expiry_in_days: int | None

    @model_validator(mode="after")
    def registered_code(self) -> Outside:
        prefix = LABELS.unseen_prefix + ":"
        if self.code not in LABELS.outside_codes and not (
            self.code.startswith(prefix) and len(self.code) > len(prefix)
        ):
            raise ValueError("the code is not one of the registered outside-population codes")
        if never_used_hits(self.code):  # amendment A5
            raise ValueError("the code names an attribute that is never shown; use A5's code")
        return self


class NotActive(Strict):
    status: Literal[NOT_ACTIVE]
    msno: str
    features_as_of: date


# --- GET /at-risk ---------------------------------------------------------------------------------


class PlanField(Strict):
    """Plan membership, a batch fact from the Phase 7 plan for the store's date."""

    in_plan: bool
    action: Literal[ACTION_NAMES[:-1]] | None  # type: ignore[valid-type]
    label: Literal[LABELS.plan_label]

    @model_validator(mode="after")
    def member_has_action(self) -> PlanField:
        if self.in_plan != (self.action is not None):
            raise ValueError("a plan action is shown exactly when the customer is in the plan")
        return self


class AtRiskRow(Strict):
    rank: int = Field(ge=1)
    msno: str
    probability: float = Field(ge=0.0, le=1.0)
    text: str
    grade: Literal[LABELS.grade_59] | None = None
    auto_renew: Literal["on", "off"]
    plan: PlanField | None


class AtRiskFilters(Strict):
    auto_renew: Literal["on", "off"] | None
    in_plan: bool | None


class AtRiskCounts(Strict):
    scored: int
    outside_population: dict[str, int]
    not_active: int


class AtRiskPage(Strict):
    store_date: date
    horizon: Literal[7, 30, 59]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    filters: AtRiskFilters
    matching: int = Field(ge=0)
    header_note: Literal[LABELS.at_risk_header_note]
    plan_status: Literal[LABELS.plan_label, LABELS.no_plan_status]
    counts: AtRiskCounts
    rows: tuple[AtRiskRow, ...]

    @model_validator(mode="after")
    def reading(self) -> AtRiskPage:
        if len(self.rows) > self.limit:
            raise ValueError("more rows than the limit")
        if [r.rank for r in self.rows] != list(
            range(self.offset + 1, self.offset + 1 + len(self.rows))
        ):
            raise ValueError("ranks must follow the offset without gaps")
        if any((r.grade is not None) != (self.horizon == 59) for r in self.rows):
            raise ValueError("every 59-day probability, and only it, is a demonstration")
        no_plan = self.plan_status == LABELS.no_plan_status
        if any((r.plan is None) != no_plan for r in self.rows):
            raise ValueError("plan fields are shown exactly when a plan exists for the date")
        p = [r.probability for r in self.rows]
        if any(a < b for a, b in zip(p, p[1:], strict=False)):
            raise ValueError("rows must be ordered by final probability, descending")
        return self


# --- GET /alerts ----------------------------------------------------------------------------------


class HiddenByRule(Strict):
    """How much each hiding rule removed from one trend alert (api.alerts and A7)."""

    top_segments: int = Field(ge=0)
    too_small_segments: int = Field(ge=0)
    segments: int = Field(ge=0)  # top_segments + too_small_segments
    dimension_totals: int = Field(ge=0)

    @model_validator(mode="after")
    def total(self) -> HiddenByRule:
        if self.segments != self.top_segments + self.too_small_segments:
            raise ValueError("segments hidden must be the sum of both segment lists")
        return self


class TrendAlert(Strict):
    """One Phase 5 trend alert as stored, with the hidden dimensions removed and counted."""

    kind: Literal["trend"]
    knowable_from: date
    labels: tuple[str, ...]
    hidden_by_rule: HiddenByRule
    alert_id: str
    week: date
    series: str
    before_rate: float
    after_rate: float
    size_points: float
    note: str | None
    direction: str
    size_class: str
    windows: dict[str, Any]
    series_returned_to_level: Any
    return_week: str | None
    band: list[Any]
    flags: dict[str, Any]
    data_quality_flags: list[str]
    top_segments: list[dict[str, Any]]
    too_small_segments: list[dict[str, Any]]
    dimension_totals: dict[str, float]
    noise_assumption: dict[str, Any]
    sentence: str

    @model_validator(mode="after")
    def registered(self) -> TrendAlert:
        if self.labels != LABELS.trend_labels:
            raise ValueError("a trend alert carries exactly the three registered labels")
        hidden = set(LABELS.hidden_dimensions)
        shown = [x.get("dimension") for x in (*self.top_segments, *self.too_small_segments)]
        if hidden & set(shown) or hidden & set(self.dimension_totals):
            raise ValueError("a hidden dimension is still in the alert")
        return self


class EventAlert(Strict):
    kind: Literal["event"]
    msno: str
    date: date
    label: Literal[LABELS.event_label]


class Alerts(Strict):
    store_date: date
    trend_in_file: int = Field(ge=0)
    trend_knowable: int = Field(ge=0)
    trend: tuple[TrendAlert, ...]
    event: tuple[EventAlert, ...]

    @model_validator(mode="after")
    def counted(self) -> Alerts:
        if self.trend_knowable != len(self.trend) or self.trend_knowable > self.trend_in_file:
            raise ValueError("the knowable count must equal the alerts shown")
        return self


class ErrorBody(Strict):
    status_code: Literal[404, 409, 422, 503]
    detail: str
    reason: str | None = None  # 503 only: why the service is not ready

    @model_validator(mode="after")
    def registered_detail(self) -> ErrorBody:
        if self.detail != LABELS.errors[self.status_code]:
            raise ValueError("the error detail differs from the registered one")
        if self.reason is not None and self.status_code != 503:
            raise ValueError("only a 503 carries a reason")
        return self


def error_body(status_code: int, reason: str | None = None) -> ErrorBody:
    return ErrorBody(status_code=status_code, detail=LABELS.errors[status_code], reason=reason)


# --- GET /summary, /plans and /model (configs/serving.yaml, dashboard_endpoints) ----------------
# No customer ID appears in any of the three; each body is scanned with no exemption.


class Bucket59(Strict):
    value: float = Field(ge=0.0)
    grade: Literal[LABELS.grade_59]


class Buckets(Strict):
    """Expected counts from the final probabilities (c01, after the fix), scored customers only."""

    population: int = Field(ge=0)
    days_1_7: float = Field(ge=0.0)
    days_8_30: float = Field(ge=0.0)
    days_31_59: Bucket59
    label: Literal[DASH.bucket_label]


class SummaryCounts(AtRiskCounts):
    outside_share_of_active: float | None = Field(ge=0.0, le=1.0)
    outside_label: Literal[DASH.outside_label]


class ReasonCell(Strict):
    group: str
    customers: int | None = Field(ge=0)
    share: float | None = Field(ge=0.0, le=1.0)
    suppressed: bool

    @model_validator(mode="after")
    def suppressed_cells_are_empty(self) -> ReasonCell:
        if self.suppressed != (self.customers is None) or self.suppressed != (self.share is None):
            raise ValueError("a suppressed cell has no count and no share; a shown cell has both")
        return self


class ReasonSegment(Strict):
    segment: str
    customers: int | None = Field(ge=0)
    suppressed: bool
    cells: tuple[ReasonCell, ...]

    @model_validator(mode="after")
    def whole_segment(self) -> ReasonSegment:
        if self.suppressed != (self.customers is None):
            raise ValueError("a suppressed segment has no count")
        if self.suppressed and self.cells:
            raise ValueError("a suppressed segment shows no cells")
        return self


class HiddenReasons(Strict):
    groups: int = Field(ge=0)
    segments: int = Field(ge=0)
    customers: int = Field(ge=0)


class SummaryReasons(Strict):
    horizon: Literal[7, 30]
    top_n: int = Field(ge=1)
    population: int = Field(ge=0)  # at most top_n: fewer when fewer customers are scored
    label: str
    explains: Literal[LABELS.reasons_fixed_line]
    min_cell_size: int = Field(ge=1)
    suppressed_cells: int = Field(ge=0)
    hidden_by_rule: HiddenReasons
    segments: tuple[ReasonSegment, ...]

    @model_validator(mode="after")
    def reading(self) -> SummaryReasons:
        if self.label != DASH.reasons_label.replace("<horizon>", str(self.horizon)):
            raise ValueError("the reasons label differs from the registered one")
        if self.population > self.top_n:
            raise ValueError("more customers than the registered top_n")
        n = self.min_cell_size
        for seg in self.segments:
            if seg.customers is not None and seg.customers < n:
                raise ValueError("a segment under the minimum cell size is shown")
            for c in seg.cells:
                if c.customers is not None and c.customers < n:
                    raise ValueError("a cell under the minimum cell size is shown")
        return self


class Summary(Strict):
    store_date: date
    counts: SummaryCounts
    buckets: Buckets
    reasons: tuple[SummaryReasons, SummaryReasons] | None
    reasons_withheld: Literal[LABELS.reasons_withheld_code] | None = None
    data_warning: Literal[LABELS.december_warning] | None = None

    @model_validator(mode="after")
    def december(self) -> Summary:
        dec = self.store_date.year == 2016 and self.store_date.month == 12
        if dec != (self.reasons is None) or dec != (self.reasons_withheld is not None):
            raise ValueError("reasons are withheld in December 2016 only")
        if dec != (self.data_warning is not None):
            raise ValueError("December 2016 carries the data warning, no other month does")
        if self.reasons is not None and tuple(r.horizon for r in self.reasons) != (7, 30):
            raise ValueError("reason shares come at 7 and 30 days, in that order")
        return self


PLAN_VALUES = (
    "expected_retained_customers",
    "expected_retained_value",
    "expected_cost",
    "net_value",
    "contacted",
    "contacted_reminder",
    "contacted_autorenew_prompt",
    "contacted_discount",
)


class PlanRange(Strict):
    """The random strategy's spread over its draws (p2_5 and p97_5 of each value)."""

    label: Literal[DASH.range_label]
    p2_5: dict[str, float | None]
    p97_5: dict[str, float | None]

    @model_validator(mode="after")
    def every_value(self) -> PlanRange:
        if set(self.p2_5) != set(PLAN_VALUES) or set(self.p97_5) != set(PLAN_VALUES):
            raise ValueError("the range must cover every plan value")
        return self


class PlanRow(Strict):
    strategy: str
    budget_label: str
    expected_retained_customers: float | None
    expected_retained_value: float | None
    expected_cost: float | None
    net_value: float | None
    contacted: float | None
    contacted_reminder: float | None
    contacted_autorenew_prompt: float | None
    contacted_discount: float | None
    range: PlanRange | None

    @model_validator(mode="after")
    def random_only(self) -> PlanRow:
        if (self.range is not None) != (self.strategy == "random"):
            raise ValueError("a range is shown for the random strategy only")
        return self


class Plans(Strict):
    store_date: date
    plan_status: Literal[DASH.plan_label, DASH.no_plan_status]
    scenario: Literal["base"]
    effect_prefix: Literal[DASH.effect_prefix]
    caption: Literal[DASH.caption]
    effect_basis: Literal[DASH.effect_basis]
    instability_note: Literal[DASH.instability_note]
    no_estimate_line: Literal[DASH.no_estimate_line]
    dates: Literal[DASH.dates]
    money_unit: Literal[DASH.money_unit]
    rows: tuple[PlanRow, ...]

    @model_validator(mode="after")
    def plan_or_none(self) -> Plans:
        if (self.plan_status == DASH.no_plan_status) != (not self.rows):
            raise ValueError("rows are shown exactly when a plan exists for the date")
        return self


class CalibrationRow(Strict):
    horizon: Literal[7, 30]
    events: int = Field(ge=0)
    slope_before_fix: float
    intercept_before_fix: float
    slope_after_fix: float
    intercept_after_fix: float


class Calibration(Strict):
    label: Literal[DASH.calibration_label]
    no_59: Literal[DASH.no_59]
    rows: int = Field(ge=0)
    rows_changed: int = Field(ge=0)
    horizons: tuple[CalibrationRow, CalibrationRow]

    @model_validator(mode="after")
    def seven_then_thirty(self) -> Calibration:
        if tuple(h.horizon for h in self.horizons) != (7, 30):
            raise ValueError("calibration rows come at 7 and 30 days, in that order")
        return self


class ModelInfo(ModelBlock):
    model_checksums_match: bool
    calibration: Calibration
    drift: Literal[DASH.drift]
    retraining: Literal[DASH.retraining]
    grade_59: Literal[LABELS.grade_59]
