"""The service layer: stored rows into the registered answers, through the step 4 functions.

One store date at a time (configs/serving.yaml, step5). For POST /score:
- the service is ready only when the downloaded model files still match the
  registered bundle, both pinned files (plans and alerts) still match their pins,
  Valkey answers and the store has a date (a half-finished load leaves none);
  otherwise 503 with every reason found; a pinned file that differs at startup
  refuses startup, its error naming the file and both hashes;
- an as_of that differs from the store's date is 409;
- a customer with a stored document is scored or outside_population as stored; a
  known customer without one is not_active; anyone else is 404 (a customer in the
  members file who never transacted before the store's date included).

scored_bodies is the batch path: it turns any number of scored rows of one snapshot
into Scored bodies with the step 4 functions, in the order score-parity uses them
(final probabilities; reasons by the service route, amendment A3; expected paid
days, A2; the action block from the 30-day probability before the fix, D1, D2,
A4). /score calls it through score_one.

score_one is a workaround (D-199), chosen after the exact-equality test failed,
not a fix of the cause: Phase 7's expected_days sums each row's eight trapezoid
terms with NumPy's sum(axis=1), which adds them in another order for a one-row
array than for an array of two or more rows, so one row alone can differ from
the batch in the last bit of the expected paid days (and the action values built
from them). score_one therefore scores the customer's row together with a copy
of it under a placeholder ID and returns the first body only. It relies on
NumPy's summation behaviour; the exact-equality tests fail if that changes.

Every 59-day probability is a demonstration (grade_59). The action block is
computed from the 30-day probability before the ordering fix, as in Phase 7.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, StrictStr, field_validator
from redis.exceptions import RedisError

from retention.api import alerts, at_risk, registry, store
from retention.api import schemas as sch
from retention.api import scoring as sc
from retention.api.config import SERVING_CONFIG, file_sha256, load_serving
from retention.api.labels import LABELS
from retention.api.saved import ModelRefused, verify_files
from retention.explain.config import ExplainConfig, load_explain_config
from retention.explain.model import Settings, load_settings
from retention.explain.reasons import PHRASES, Renderer
from retention.models.protocol import Protocol, load_protocol
from retention.optimize.config import ActionsConfig, load_actions_config

REASON_HORIZONS = (7, 30)
PAIR_MSNO = "pair-row:not-a-customer"  # never a KKBox ID ("-" and ":" are not in them)
C01_HORIZONS = (7, 30, 59)


class NotReady(RuntimeError):
    """503, with the reason the service cannot answer."""


class UnknownCustomer(LookupError):
    """404."""


class DateDiffers(ValueError):
    """409: as_of differs from the store's date."""


class PinRefused(RuntimeError):
    """A pinned file (step5.pins) is missing or its SHA-256 differs from the pin."""


# --- requests and health -----------------------------------------------------------------------


class ScoreRequest(sch.Strict):
    msno: StrictStr = Field(min_length=1)
    as_of: StrictStr | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("as_of")
    @classmethod
    def a_calendar_date(cls, v: str | None) -> str | None:
        if v is not None:
            date.fromisoformat(v)  # a day that does not exist is malformed input
        return v

    def as_of_date(self) -> date | None:
        return None if self.as_of is None else date.fromisoformat(self.as_of)


class ReplayProgress(sch.Strict):
    produced: int
    consumed: int
    expected_events: int


class Health(sch.Strict):
    ready: bool
    reason: str | None  # why not ready; None when ready
    store_date: date | None
    replay_progress: ReplayProgress | None
    model_checksums_match: bool
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    valkey_reachable: bool


# --- the loaded context ------------------------------------------------------------------------


# --- the pinned files ----------------------------------------------------------------------------


@dataclass
class Pinned:
    """The two pinned batch files, read once at startup and never written."""

    pins: dict[str, tuple[Path, str]]  # name: (path, pinned SHA-256)
    plans: dict[date, dict[str, str]]  # the registered plan cell, {snapshot: {msno: action}}
    alerts: dict[str, Any]  # the Phase 5 alerts run


def registered_pins(cfg: dict[str, Any] | None = None) -> dict[str, tuple[Path, str]]:
    pins = (cfg if cfg is not None else load_serving())["step5"]["pins"]
    return {name: (Path(p["path"]), p["sha256"]) for name, p in pins.items()}


def pin_problems(pins: dict[str, tuple[Path, str]]) -> list[str]:
    """Each pinned file that is missing or differs, named with both hashes."""
    out = []
    for path, sha in pins.values():
        if not path.is_file():
            out.append(f"pinned file {path.name} is missing")
            continue
        found = file_sha256(path)
        if found != sha:
            out.append(f"pinned file {path.name} SHA-256 {found} differs from the pinned {sha}")
    return out


def load_pinned(pins: dict[str, tuple[Path, str]]) -> Pinned:
    """Both pinned files after their hashes match; refuses (PinRefused) otherwise."""
    if sorted(pins) != ["alerts", "plans"]:
        raise PinRefused("the pins must name exactly the plans and the alerts files")
    problems = pin_problems(pins)
    if problems:
        raise PinRefused("; ".join(problems))
    plans = at_risk.plan_cell(pd.read_parquet(pins["plans"][0]))
    run = json.loads(pins["alerts"][0].read_text())
    alerts.check_run(run)
    return Pinned(dict(pins), plans, run)


@dataclass
class Context:
    models: registry.ServingModels
    registered_model: str
    bundle: dict[str, str]  # the registered bundle
    bundle_dir: Path  # the downloaded files the models were parsed from
    workdir: Path  # removed at shutdown
    explain_cfg: ExplainConfig
    renderer: Renderer
    phrases_sha256: str
    columns: list[str]  # the classifiers' columns
    fact_columns: list[str]
    actions_cfg: ActionsConfig
    protocol: Protocol
    config_sha256: str
    replay_expected_events: int
    pinned: Pinned
    at_risk_index: at_risk.IndexCache

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def downloaded_folder(workdir: Path) -> Path:
    """The one folder in `workdir` holding the downloaded bundle and its manifest."""
    found = [p.parent for p in workdir.rglob(registry.MANIFEST)]
    if len(found) != 1:
        raise ModelRefused(f"expected one downloaded bundle, found {len(found)}")
    return found[0]


def load_context(
    settings: registry.RegistrySettings,
    model_settings: Settings,
    protocol: Protocol,
    workdir: Path,
    cfg: dict[str, Any] | None = None,
    pins: dict[str, tuple[Path, str]] | None = None,
) -> Context:
    """The models through the alias with every registry check, then everything scoring needs.

    Refuses (ModelRefused) when a downloaded file's SHA-256 differs from the bundle the
    response schemas register, so a model block can never name other files, and
    (PinRefused) when a pinned file differs from its pin. `pins` defaults to the
    registered ones (step5.pins).
    """
    cfg = cfg if cfg is not None else load_serving()
    pinned = load_pinned(registered_pins(cfg) if pins is None else pins)
    models = registry.load_serving_models(settings, model_settings, protocol, workdir)
    if dict(models.files) != sch.LABELS.bundle:
        raise ModelRefused("the loaded model files differ from the registered bundle")
    explain_cfg = load_explain_config(model_settings.columns)
    facts = sorted({c for cs in explain_cfg.facts.values() for c in cs})
    return Context(
        models=models,
        registered_model=settings.name,
        bundle=dict(settings.bundle),
        bundle_dir=downloaded_folder(workdir),
        workdir=workdir,
        explain_cfg=explain_cfg,
        renderer=Renderer.from_files(explain_cfg),
        phrases_sha256=file_sha256(PHRASES),
        columns=list(model_settings.columns),
        fact_columns=facts,
        actions_cfg=load_actions_config(),
        protocol=protocol,
        config_sha256=file_sha256(SERVING_CONFIG),
        replay_expected_events=int(cfg["api"]["health"]["replay_expected_events"]),
        pinned=pinned,
        at_risk_index=at_risk.IndexCache(),
    )


def default_context(repo: Path | None = None) -> Context:
    """The service's context: the project's registry, settings and protocol."""
    repo = Path.cwd() if repo is None else repo
    workdir = Path(tempfile.mkdtemp(prefix="retention-serving-"))
    try:
        return load_context(
            registry.registry_settings(repo),
            load_settings(horizons=C01_HORIZONS),
            load_protocol(),
            workdir,
        )
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


# --- readiness ---------------------------------------------------------------------------------


def model_problem(ctx: Context) -> str | None:
    """None when every downloaded file still matches the bundle; otherwise what differs."""
    try:
        verify_files(ctx.bundle_dir, ctx.bundle)
    except ModelRefused as e:
        return f"model files differ from the registered bundle: {e}"
    return None


def health(ctx: Context, r) -> Health:
    reasons = []
    problem = model_problem(ctx)
    if problem:
        reasons.append(problem)
    reasons += pin_problems(ctx.pinned.pins)
    reachable, t, progress = True, None, None
    try:
        r.ping()
        t = store.store_date(r)
        produced, consumed = (int(v or 0) for v in r.mget(list(store.KEYS.replay)))
        progress = ReplayProgress(
            produced=produced, consumed=consumed, expected_events=ctx.replay_expected_events
        )
    except (RedisError, OSError):
        reachable = False
        reasons.append("Valkey cannot be reached")
    if reachable and t is None:
        reasons.append("no snapshot is loaded (the store has no date)")
    return Health(
        ready=not reasons,
        reason="; ".join(reasons) or None,
        store_date=t,
        replay_progress=progress,
        model_checksums_match=problem is None,
        config_sha256=ctx.config_sha256,
        valkey_reachable=reachable,
    )


def ready_date(ctx: Context, r) -> date:
    """The store's date; NotReady with every reason otherwise."""
    h = health(ctx, r)
    if not h.ready:
        raise NotReady(h.reason)
    return h.store_date


# --- the batch path ----------------------------------------------------------------------------


def scored_bodies(
    ctx: Context, snapshot: date, rows: pd.DataFrame, prices: pd.DataFrame
) -> list[dict[str, Any]]:
    """Scored bodies (JSON-ready) for scored rows of one snapshot, in `rows`' order."""
    c01, msno = ctx.models.c01, rows.msno.astype(str).tolist()
    pr = sc.probabilities(c01, rows, ctx.protocol.seed)
    text = pr.text()
    facts = rows[["msno", *ctx.fact_columns]].reset_index(drop=True)
    served = {}
    for h in REASON_HORIZONS:
        part = sc.contribution_part(c01[h], rows, ctx.explain_cfg, ctx.columns, h)
        platt = (c01[h].platt_a, c01[h].platt_b)
        served[h] = sc.service_reasons(
            ctx.explain_cfg,
            ctx.renderer,
            part,
            facts,
            h,
            snapshot,
            platt,
            ctx.protocol,
            ctx.phrases_sha256,
        )
    table = sc.customers(rows, sc.survival(ctx.models.dt01, rows), served[30], prices)
    blocks = sc.action_blocks(sc.actions(table, ctx.actions_cfg), ctx.actions_cfg.base)
    for frame in (served[7], served[30], table):
        if frame.msno.astype(str).tolist() != msno:
            raise ValueError("a scoring step changed the row order")
    december = sc.in_december_2016(snapshot)
    model = sch.ModelBlock(
        registered_model=ctx.registered_model,
        version=ctx.models.version,
        files=dict(ctx.models.files),
        config_sha256=ctx.config_sha256,
    )
    out = []
    for i, m in enumerate(msno):
        risk = {
            f"days_{h}": {"probability": float(pr.final[i, k]), "text": text[i, k]}
            for k, h in enumerate(C01_HORIZONS)
        }
        risk["days_59"]["grade"] = LABELS.grade_59
        reasons = None
        if not december:
            reasons = tuple(
                sch.Reason(
                    horizon=h,
                    probability_before_fix=float(served[h].probability.iloc[i]),
                    text=served[h].text.iloc[i],
                    explains=LABELS.reasons_fixed_line,
                )
                for h in REASON_HORIZONS
            )
        body = sch.Scored(
            status=sch.SCORED,
            msno=m,
            features_as_of=snapshot,
            risk=sch.Risk(**risk, moved_by_ordering_fix=bool(pr.moved[i])),
            reasons=reasons,
            reasons_withheld=LABELS.reasons_withheld_code if december else None,
            data_warning=LABELS.december_warning if december else None,
            paid_days=sch.PaidDays(**sc.paid_days_block(float(table.days_30_90.iloc[i]))),
            archetype=sch.Archetype(value=None, status=LABELS.archetype_status),
            action=sch.ActionBlock(**blocks[i]),
            model=model,
        )
        out.append(body.model_dump(mode="json"))
    return out


def score_one(
    ctx: Context, snapshot: date, rows: pd.DataFrame, prices: pd.DataFrame
) -> dict[str, Any]:
    """One customer's Scored body, scored in a two-row block with a placeholder copy
    (the D-199 workaround); only the customer's own body is returned."""
    if len(rows) != 1 or len(prices) != 1:
        raise ValueError("score_one takes one row and one price")
    msno = str(rows.msno.iloc[0])
    if msno == PAIR_MSNO:
        raise ValueError("the placeholder ID is not a customer")
    pair_rows = pd.concat([rows, rows.assign(msno=PAIR_MSNO)], ignore_index=True)
    pair_prices = pd.concat([prices, prices.assign(msno=PAIR_MSNO)], ignore_index=True)
    first, _ = scored_bodies(ctx, snapshot, pair_rows, pair_prices)
    if first["msno"] != msno:
        raise ValueError("the pair returned another customer's body first")
    return first


# --- one customer from the store ---------------------------------------------------------------


def stored_inputs(
    msno: str, doc: dict[str, Any], snapshot: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored row and price rebuilt as the batch loader types them."""
    row = doc["row"]
    rows = store.rows_frame([row], list(row))
    rows.insert(0, "msno", [msno])
    rows["snapshot_date"] = pd.Timestamp(snapshot)
    p = doc["price"]
    prices = pd.DataFrame(
        {"msno": [msno], "price_per_day": [p["price_per_day"]], "has_price": [p["has_price"]]}
    )
    return rows, prices


def answer(ctx: Context, r, req: ScoreRequest) -> dict[str, Any]:
    """The /score body; raises NotReady, DateDiffers or UnknownCustomer."""
    t = ready_date(ctx, r)
    as_of = req.as_of_date()
    if as_of is not None and as_of != t:
        raise DateDiffers(req.as_of)
    doc = store.customer(r, req.msno)
    if doc is None:
        if store.is_known(r, req.msno):
            body = sch.NotActive(status=sch.NOT_ACTIVE, msno=req.msno, features_as_of=t)
            return body.model_dump(mode="json")
        raise UnknownCustomer()
    if doc["status"] == sch.OUTSIDE:
        body = sch.Outside(
            status=sch.OUTSIDE,
            msno=req.msno,
            features_as_of=t,
            code=doc["code"],
            next_expiry_in_days=doc["next_expiry_in_days"],
        )
        return body.model_dump(mode="json")
    if doc["status"] != sch.SCORED:
        raise ValueError("a stored document with an unregistered status")
    rows, prices = stored_inputs(req.msno, doc, t)
    return score_one(ctx, t, rows, prices)
