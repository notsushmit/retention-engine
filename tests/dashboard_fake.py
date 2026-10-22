"""A fake scoring API for the dashboard tests: hand-made answers served over httpx.MockTransport.

Every answer is built through the API's own response schemas (retention.api.schemas),
so a fake answer the real API could not give cannot be built. Answers that can carry
customer IDs (the /at-risk rows, cancel events, a /score answer) carry IDs from IDS,
so a test that looks for leaked IDs can fail.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx

from retention.api import schemas as sch
from retention.api import service as svc
from retention.api.labels import DASH, LABELS

NOV = date(2016, 11, 1)
IDS = [f"fakeid{i:02d}AbCdEfGh" for i in range(25)]
CANCEL_ID = "fakecancelXyZ0123"
OUT_ID, INACTIVE_ID = "fakeoutsideQq01", "fakeinactiveRr02"
SCORED_ID = IDS[0]


def dump(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


def health(ready: bool = True, produced: int = 0, consumed: int = 0) -> dict[str, Any]:
    return dump(
        svc.Health(
            ready=ready,
            reason=None if ready else "no snapshot is loaded (the store has no date)",
            store_date=NOV if ready else None,
            replay_progress=svc.ReplayProgress(
                produced=produced, consumed=consumed, expected_events=3727472
            ),
            model_checksums_match=True,
            config_sha256="a" * 64,
            valkey_reachable=True,
        )
    )


COUNTS = {
    "scored": 169673,
    "outside_population": {
        "segment_trial_only": 8890,
        "segment_promo_only": 31,
        "expiry_beyond_30_days": 51692,
    },
    "not_active": 195234,
}


def at_risk(horizon: int = 30, limit: int = 50, offset: int = 0, **filters) -> dict[str, Any]:
    in_plan = filters.get("in_plan")
    ids = IDS[offset : offset + limit]
    rows = []
    for i, m in enumerate(ids):
        member = i % 3 == 0
        if in_plan is not None and member != (in_plan == "true"):
            continue
        p = round(0.9 - 0.01 * (offset + i), 6)
        rows.append(
            sch.AtRiskRow(
                rank=offset + len(rows) + 1,
                msno=m,
                probability=p,
                text=f"{round(p * 100)}%",
                grade=LABELS.grade_59 if horizon == 59 else None,
                auto_renew="off" if i % 2 else "on",
                plan=sch.PlanField(
                    in_plan=member, action="reminder" if member else None, label=LABELS.plan_label
                ),
            )
        )
    return dump(
        sch.AtRiskPage(
            store_date=NOV,
            horizon=horizon,
            offset=offset,
            limit=limit,
            filters=sch.AtRiskFilters(auto_renew=filters.get("auto_renew"), in_plan=None),
            matching=len(IDS),
            header_note=LABELS.at_risk_header_note,
            plan_status=LABELS.plan_label,
            counts=sch.AtRiskCounts(**COUNTS),
            rows=tuple(rows),
        )
    )


def trend_alert() -> sch.TrendAlert:
    seg = {"dimension": "auto_renew", "value": "on", "events_before": 10, "events_after": 12}
    return sch.TrendAlert(
        kind="trend",
        knowable_from=date(2016, 4, 13),
        labels=LABELS.trend_labels,
        hidden_by_rule=sch.HiddenByRule(
            top_segments=2, too_small_segments=1, segments=3, dimension_totals=3
        ),
        alert_id="cp1",
        week=date(2016, 3, 7),
        series="the paid series",
        before_rate=0.11,
        after_rate=0.18,
        size_points=7.0,
        note=None,
        direction="up",
        size_class="shift",
        windows={},
        series_returned_to_level=None,
        return_week=None,
        band=[None, None],
        flags={},
        data_quality_flags=["unreliable_month"],
        top_segments=[seg],
        too_small_segments=[],
        dimension_totals={"auto_renew": 0.06},
        noise_assumption={"status": "exploratory"},
        sentence="The weekly lapse rate of the paid series moved up by 7.0 points.",
    )


def alerts() -> dict[str, Any]:
    return dump(
        sch.Alerts(
            store_date=NOV,
            trend_in_file=10,
            trend_knowable=1,
            trend=(trend_alert(),),
            event=(
                sch.EventAlert(
                    kind="event", msno=CANCEL_ID, date=date(2016, 10, 3), label=LABELS.event_label
                ),
            ),
        )
    )


def reasons(horizon: int, cells: list[tuple[str, int | None]]) -> sch.SummaryReasons:
    shown = [n for _, n in cells if n is not None]
    pop = 1000
    return sch.SummaryReasons(
        horizon=horizon,
        top_n=1000,
        population=pop,
        label=DASH.reasons_label.replace("<horizon>", str(horizon)),
        explains=LABELS.reasons_fixed_line,
        min_cell_size=20,
        suppressed_cells=len(cells) - len(shown),
        hidden_by_rule=sch.HiddenReasons(groups=0, segments=0, customers=0),
        segments=(
            sch.ReasonSegment(
                segment="all",
                customers=pop,
                suppressed=False,
                cells=tuple(
                    sch.ReasonCell(
                        group=g,
                        customers=n,
                        share=None if n is None else n / pop,
                        suppressed=n is None,
                    )
                    for g, n in cells
                ),
            ),
            sch.ReasonSegment(segment="paid", customers=907, suppressed=False, cells=()),
            sch.ReasonSegment(
                segment="discount100_only", customers=None, suppressed=True, cells=()
            ),
        ),
    )


R7 = [("renewal_timing", 1000)]
R30 = [("renewal_timing", 823), ("payment_setup", 90), ("cancellation", 75), ("plan_length", None)]


def summary() -> dict[str, Any]:
    return dump(
        sch.Summary(
            store_date=NOV,
            counts=sch.SummaryCounts(
                **COUNTS,
                outside_share_of_active=51692 / 230286,
                outside_label=DASH.outside_label,
            ),
            buckets=sch.Buckets(
                population=169673,
                days_1_7=3159.7587096676675,
                days_8_30=4871.568506211308,
                days_31_59=sch.Bucket59(value=3408.1308071068447, grade=LABELS.grade_59),
                label=DASH.bucket_label,
            ),
            reasons=(reasons(7, R7), reasons(30, R30)),
        )
    )


BUDGETS = ["0", "1,000", "5,000", "no limit"]


def plans(t: date = NOV) -> dict[str, Any]:
    rows = []
    if t == NOV:
        for strategy in ("optimizer", "highest_risk", "random"):
            for k, budget in enumerate(BUDGETS):
                vals = {v: float(k * 100 + j) for j, v in enumerate(sch.PLAN_VALUES)}
                if strategy == "optimizer" and budget == "5,000":
                    vals["net_value"] = 5312.564758518844
                    vals["contacted"] = 4887.0
                spread = None
                if strategy == "random":
                    spread = sch.PlanRange(
                        label=DASH.range_label,
                        p2_5={v: x - 1 for v, x in vals.items()},
                        p97_5={v: x + 1 for v, x in vals.items()},
                    )
                rows.append(
                    sch.PlanRow(strategy=strategy, budget_label=budget, **vals, range=spread)
                )
    return dump(
        sch.Plans(
            store_date=t,
            plan_status=DASH.plan_label if rows else DASH.no_plan_status,
            scenario="base",
            effect_prefix=DASH.effect_prefix,
            caption=DASH.caption,
            effect_basis=DASH.effect_basis,
            instability_note=DASH.instability_note,
            no_estimate_line=DASH.no_estimate_line,
            dates=DASH.dates,
            money_unit=DASH.money_unit,
            rows=tuple(rows),
        )
    )


def model() -> dict[str, Any]:
    cal = {
        7: (
            937,
            0.9556452448959333,
            -0.06712515413029518,
            0.9806757005295119,
            0.0010183554314821036,
        ),
        30: (
            3388,
            1.005399403794472,
            -0.0012376814371337342,
            1.0004532529100434,
            -0.016306194832806185,
        ),
    }
    return dump(
        sch.ModelInfo(
            registered_model="retention-scoring",
            version=1,
            files=dict(LABELS.bundle),
            config_sha256="a" * 64,
            model_checksums_match=True,
            calibration=sch.Calibration(
                label=DASH.calibration_label,
                no_59=DASH.no_59,
                rows=72395,
                rows_changed=3348,
                horizons=tuple(
                    sch.CalibrationRow(
                        horizon=h,
                        events=c[0],
                        slope_before_fix=c[1],
                        intercept_before_fix=c[2],
                        slope_after_fix=c[3],
                        intercept_after_fix=c[4],
                    )
                    for h, c in cal.items()
                ),
            ),
            drift=DASH.drift,
            retraining=DASH.retraining,
            grade_59=LABELS.grade_59,
        )
    )


def scenario() -> sch.Scenario:
    return sch.Scenario(
        form="relative",
        level="base",
        margin=0.5,
        cost_level="base",
        discount_cost_rule=LABELS.scenario_text["discount_cost_rule"],
        value_horizon=LABELS.scenario_text["value_horizon"],
        budget=LABELS.scenario_text["budget"],
    )


def scored(msno: str = SCORED_ID) -> dict[str, Any]:
    return dump(
        sch.Scored(
            status=sch.SCORED,
            msno=msno,
            features_as_of=NOV,
            risk=sch.Risk(
                days_7=sch.Probability(probability=0.21, text="21%"),
                days_30=sch.Probability(probability=0.48, text="48%"),
                days_59=sch.Probability59(probability=0.61, text="61%", grade=LABELS.grade_59),
                moved_by_ordering_fix=False,
            ),
            reasons=tuple(
                sch.Reason(
                    horizon=h,
                    probability_before_fix=p,
                    text="Higher than typical: the renewal is due soon.",
                    explains=LABELS.reasons_fixed_line,
                )
                for h, p in ((7, 0.2), (30, 0.47))
            ),
            paid_days=sch.PaidDays(
                field=LABELS.paid_days_field,
                value=41.5,
                name=LABELS.paid_days_name,
                labels=LABELS.paid_days_labels,
            ),
            archetype=sch.Archetype(value=None, status=LABELS.archetype_status),
            action=sch.ActionBlock(
                highest_net_value_action="reminder",
                expected_retained_value=3.2,
                net_value=1.2,
                cost=2.0,
                pays=True,
                effect_basis=LABELS.effect_basis,
                scenario=scenario(),
                instability_note=LABELS.instability_note,
                no_estimate_line=LABELS.no_estimate_line,
                probability_basis=LABELS.probability_basis,
            ),
            model=sch.ModelBlock(
                registered_model="retention-scoring",
                version=1,
                files=dict(LABELS.bundle),
                config_sha256="a" * 64,
            ),
        )
    )


def outside() -> dict[str, Any]:
    return dump(
        sch.Outside(
            status=sch.OUTSIDE,
            msno=OUT_ID,
            features_as_of=NOV,
            code="expiry_beyond_30_days",
            next_expiry_in_days=45,
        )
    )


def not_active() -> dict[str, Any]:
    return dump(sch.NotActive(status=sch.NOT_ACTIVE, msno=INACTIVE_ID, features_as_of=NOV))


class FakeApi:
    """Answers by path; `overrides` replaces an answer: (status, body)."""

    def __init__(self, overrides: dict[str, tuple[int, dict]] | None = None):
        self.overrides = overrides or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.overrides:
            status, body = self.overrides[path]
            return httpx.Response(status, json=body)
        q = dict(request.url.params)
        if path == "/health":
            return httpx.Response(200, json=health())
        if path == "/at-risk":
            kw = {k: int(v) for k, v in q.items() if k in ("horizon", "limit", "offset")}
            return httpx.Response(
                200,
                json=at_risk(
                    **kw, **{k: v for k, v in q.items() if k in ("auto_renew", "in_plan")}
                ),
            )
        if path == "/alerts":
            return httpx.Response(200, json=alerts())
        if path == "/summary":
            return httpx.Response(200, json=summary())
        if path == "/plans":
            return httpx.Response(200, json=plans())
        if path == "/model":
            return httpx.Response(200, json=model())
        if path == "/score":
            msno = json.loads(request.content)["msno"]
            if msno == OUT_ID:
                return httpx.Response(200, json=outside())
            if msno == INACTIVE_ID:
                return httpx.Response(200, json=not_active())
            if msno in IDS:
                return httpx.Response(200, json=scored(msno))
            return httpx.Response(404, json=dump(sch.error_body(404)))
        return httpx.Response(404, json={"detail": "Not Found"})


def not_ready() -> dict[str, tuple[int, dict]]:
    """Overrides for a service with an empty store."""
    body = dump(sch.error_body(503, "no snapshot is loaded (the store has no date)"))
    return {
        "/health": (503, health(ready=False)),
        **{p: (503, body) for p in ("/at-risk", "/alerts", "/summary", "/plans")},
    }


def client(api: FakeApi | None = None):
    from dashboard.client import Client

    return Client("http://127.0.0.1:8000", transport=httpx.MockTransport(api or FakeApi()))
