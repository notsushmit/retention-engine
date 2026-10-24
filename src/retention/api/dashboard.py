"""GET /summary, /plans and /model: the aggregates the dashboard reads.

configs/serving.yaml, dashboard_endpoints. No customer ID appears in any body.

/summary, for the store's one date:
- the stored /at-risk header counts and the share of active customers outside the
  population because they expire more than 30 days ahead;
- three expected counts over every scored customer, from the stored final
  probabilities (c01 after the ordering fix): the sum of p7, of p30 - p7 and of
  p59 - p30 (a demonstration, as every 59-day probability), summed with math.fsum
  in msno order; no survival curve is used;
- at 7 and 30 days, the reason group the model lists first as raising the
  probability (raising_1, through the service's own reasons path with amendment A3's
  masking), counted among the top_n customers ranked highest at that horizon, by
  main segment and for both together. A cell or segment with fewer customers than
  the registered minimum is suppressed and counted; there is no secondary
  suppression, so a suppressed cell can be derived from its segment's total. A
  group or segment whose name holds a hidden dimension or a never-used word is
  removed and counted, as the alerts' hiding rule (amendment A7) does.
The body is computed once per store date and kept.

/plans: the rows of the pinned Phase 7 plan summary for the base scenario and the
store's date, every strategy and budget level, values as stored; the random
strategy carries its p2_5 and p97_5 spread. Nothing is re-planned.

/model: the registered model's files and the pinned calibration table before and
after the ordering fix; drift and retraining are not part of this phase.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from retention.api import ops, store
from retention.api import schemas as sch
from retention.api import scoring as sc
from retention.api.labels import DASH, LABELS

REASON_HORIZONS = (7, 30)
ALL = "all"
SCENARIO = "base"
HIDDEN = re.compile(
    r"\b("
    + "|".join(map(re.escape, (*LABELS.hidden_dimensions, *LABELS.never_in_any_body)))
    + r")\b",
    re.I,
)


def hidden_name(name: str) -> bool:
    """A group or segment name holding a hidden dimension or a never-used word."""
    return bool(HIDDEN.search(name.replace("_", " "))) or bool(HIDDEN.search(name))


# --- /summary: counts and buckets ---------------------------------------------------------------


def summary_counts(stored: dict[str, int]) -> sch.SummaryCounts:
    head = sch.AtRiskCounts.model_validate(
        {
            "scored": stored["scored"],
            "outside_population": {
                k.removeprefix("outside_population_"): v
                for k, v in stored.items()
                if k.startswith("outside_population_")
            },
            "not_active": stored["not_active"],
        }
    )
    active = head.scored + sum(head.outside_population.values())
    beyond = head.outside_population.get(sc.BEYOND, 0)
    return sch.SummaryCounts(
        **head.model_dump(),
        outside_share_of_active=beyond / active if active else None,
        outside_label=DASH.outside_label,
    )


def buckets(rankings: dict[int, list[tuple[str, float]]]) -> sch.Buckets:
    """Expected counts from the final probabilities, every scored customer, msno order."""
    p = {h: dict(rankings[h]) for h in (7, 30, 59)}
    if not set(p[7]) == set(p[30]) == set(p[59]):
        raise ValueError("the three rankings hold different customers")
    msnos = sorted(p[7])
    return sch.Buckets(
        population=len(msnos),
        days_1_7=math.fsum(p[7][m] for m in msnos),
        days_8_30=math.fsum(p[30][m] - p[7][m] for m in msnos),
        days_31_59=sch.Bucket59(
            value=math.fsum(p[59][m] - p[30][m] for m in msnos), grade=LABELS.grade_59
        ),
        label=DASH.bucket_label,
    )


# --- /summary: reason shares --------------------------------------------------------------------


def stored_rows(r, msnos: list[str], snapshot: date) -> pd.DataFrame:
    """The stored rows of scored customers, rebuilt as the batch loader types them."""
    docs = [store.customer(r, m) for m in msnos]
    if any(d is None or d["status"] != sch.SCORED for d in docs):
        raise ValueError("a ranked customer has no scored document")
    records = [d["row"] for d in docs]
    rows = store.rows_frame(records, list(records[0]))
    rows.insert(0, "msno", msnos)
    rows["snapshot_date"] = pd.Timestamp(snapshot)
    return rows


def first_groups(ctx, rows: pd.DataFrame, horizon: int, snapshot: date) -> pd.Series:
    """raising_1 of the served reasons (None when no group is listed), in `rows`' order."""
    c01 = ctx.models.c01[horizon]
    part = sc.contribution_part(c01, rows, ctx.explain_cfg, ctx.columns, horizon)
    facts = rows[["msno", *ctx.fact_columns]].reset_index(drop=True)
    served = sc.service_reasons(
        ctx.explain_cfg,
        ctx.renderer,
        part,
        facts,
        horizon,
        snapshot,
        (c01.platt_a, c01.platt_b),
        ctx.protocol,
        ctx.phrases_sha256,
    )
    if served.msno.astype(str).tolist() != rows.msno.astype(str).tolist():
        raise ValueError("the reasons changed the row order")
    return served.raising_1


def reason_shares(
    horizon: int,
    segment: pd.Series,
    group: pd.Series,
    main_segments: tuple[str, ...],
    top_n: int,
    min_cell: int,
) -> sch.SummaryReasons:
    """Counts of the first listed group by main segment and for all, with suppression."""
    frame = pd.DataFrame(
        {"segment": segment.astype(str).to_numpy(), "group": group.fillna(DASH.no_group)}
    )
    if not set(frame.segment) <= set(main_segments):
        raise ValueError("a ranked customer outside the main segments")
    hidden_groups = sorted({g for g in frame.group if hidden_name(g)})
    hidden_segments = [s for s in main_segments if hidden_name(s)]
    dropped = frame.group.isin(hidden_groups) | frame.segment.isin(hidden_segments)
    kept = frame[~dropped]
    suppressed, out = 0, []
    for seg in (ALL, *[s for s in main_segments if s not in hidden_segments]):
        part = kept if seg == ALL else kept[kept.segment == seg]
        counts = part.group.value_counts()
        groups = sorted(counts.index, key=lambda g: (-int(counts[g]), g))
        if len(part) < min_cell:
            suppressed += len(groups)
            out.append(sch.ReasonSegment(segment=seg, customers=None, suppressed=True, cells=()))
            continue
        cells = []
        for g in groups:
            n = int(counts[g])
            small = n < min_cell
            suppressed += small
            cells.append(
                sch.ReasonCell(
                    group=g,
                    customers=None if small else n,
                    share=None if small else n / len(part),
                    suppressed=small,
                )
            )
        out.append(
            sch.ReasonSegment(
                segment=seg, customers=len(part), suppressed=False, cells=tuple(cells)
            )
        )
    return sch.SummaryReasons(
        horizon=horizon,
        top_n=top_n,
        population=len(frame),
        label=DASH.reasons_label.replace("<horizon>", str(horizon)),
        explains=LABELS.reasons_fixed_line,
        min_cell_size=min_cell,
        suppressed_cells=suppressed,
        hidden_by_rule=sch.HiddenReasons(
            groups=len(hidden_groups), segments=len(hidden_segments), customers=int(dropped.sum())
        ),
        segments=tuple(out),
    )


def summary_body(
    ctx,
    r,
    store_date: date,
    rankings: dict[int, list[tuple[str, float]]],
    top_n: int = DASH.top_n,
    min_cell: int = DASH.min_cell_size,
) -> dict[str, Any]:
    december = sc.in_december_2016(store_date)
    reasons = None
    if not december:
        main = tuple(ctx.protocol.main_segments)
        reasons = []
        for h in REASON_HORIZONS:
            top = [m for m, _ in rankings[h][:top_n]]
            rows = stored_rows(r, top, store_date)
            groups = first_groups(ctx, rows, h, store_date)
            reasons.append(reason_shares(h, rows.segment, groups, main, top_n, min_cell))
        reasons = tuple(reasons)
    body = sch.Summary(
        store_date=store_date,
        counts=summary_counts(store.counts(r)),
        buckets=buckets(rankings),
        reasons=reasons,
        reasons_withheld=LABELS.reasons_withheld_code if december else None,
        data_warning=LABELS.december_warning if december else None,
    )
    return body.model_dump(mode="json")


class SummaryCache:
    """The /summary body of the current store date; rebuilt only when the date moves."""

    def __init__(self, build: Callable[..., dict[str, Any]] = summary_body):
        self._build = build
        self._lock = threading.Lock()
        self._body: tuple[date, dict[str, Any]] | None = None

    def get(self, ctx, r, store_date: date, rankings) -> dict[str, Any]:
        with self._lock:
            if self._body is None or self._body[0] != store_date:
                self._body = (store_date, self._build(ctx, r, store_date, rankings))
            return self._body[1]


# --- /plans --------------------------------------------------------------------------------------


def value(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def plans_body(summary: pd.DataFrame, store_date: date) -> dict[str, Any]:
    """The pinned plan summary's base-scenario rows for the store's date, as stored."""
    sel = summary[(summary.scenario == SCENARIO) & (summary.snapshot == store_date.isoformat())]
    if sel.duplicated(["strategy", "budget_label"]).any():
        raise ValueError("a strategy and budget appear twice in the plan summary")
    rows = []
    for rec in sel.to_dict("records"):
        spread = None
        if rec["strategy"] == "random":
            spread = sch.PlanRange(
                label=DASH.range_label,
                p2_5={k: value(rec[f"{k}_p2_5"]) for k in sch.PLAN_VALUES},
                p97_5={k: value(rec[f"{k}_p97_5"]) for k in sch.PLAN_VALUES},
            )
        rows.append(
            sch.PlanRow(
                strategy=rec["strategy"],
                budget_label=rec["budget_label"],
                **{k: value(rec[k]) for k in sch.PLAN_VALUES},
                range=spread,
            )
        )
    body = sch.Plans(
        store_date=store_date,
        plan_status=DASH.plan_label if rows else DASH.no_plan_status,
        scenario=SCENARIO,
        effect_prefix=DASH.effect_prefix,
        caption=DASH.caption,
        effect_basis=DASH.effect_basis,
        instability_note=DASH.instability_note,
        no_estimate_line=DASH.no_estimate_line,
        dates=DASH.dates,
        money_unit=DASH.money_unit,
        rows=tuple(rows),
    )
    return body.model_dump(mode="json")


# --- /model --------------------------------------------------------------------------------------


def calibration(table: dict[str, Any]) -> sch.Calibration:
    return sch.Calibration(
        label=DASH.calibration_label,
        no_59=DASH.no_59,
        rows=table["rows"],
        rows_changed=table["rows_changed"],
        horizons=tuple(
            sch.CalibrationRow(
                horizon=h,
                **{
                    k: table[str(h)][k]
                    for k in (
                        "events",
                        "slope_before_fix",
                        "intercept_before_fix",
                        "slope_after_fix",
                        "intercept_after_fix",
                    )
                },
            )
            for h in REASON_HORIZONS
        ),
    )


def model_body(ctx, checksums_match: bool, ops_dir=None) -> dict[str, Any]:
    body = sch.ModelInfo(
        registered_model=ctx.registered_model,
        version=ctx.models.version,
        files=dict(ctx.models.files),
        config_sha256=ctx.config_sha256,
        model_checksums_match=checksums_match,
        calibration=calibration(ctx.pinned.calibration),
        drift=ops.drift_line(ops_dir),
        retraining=DASH.retraining,
        grade_59=LABELS.grade_59,
        last_retrain=ops.last_retrain(ops_dir),
    )
    return body.model_dump(mode="json")
