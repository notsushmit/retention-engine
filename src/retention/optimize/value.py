"""The value step: the dt01 refit, its survival curves, and each customer's value inputs.

dt01 was not saved in Phase 3, so it is refit once (configs/actions.yaml,
dt01_refit), exactly as the test stage fits it: the frozen choices, P2 main rows
of 2016-08-01 and 2016-09-01 from the cached label build as of 2017-01-01 (read,
never built: this process never loads the label-building code), the 7-day
classifier hyperparameters, protocol seed and threads.

Pass marks: peak memory under the ceiling, and the exact reproduction of the raw
score summaries the 2016 rehearsal logged for dt01 at 7, 30 and 59 days on
2016-11-01 (same P2 main rows, same order; floats equal after a JSON round trip).
Any difference: nothing is saved. Causes to examine first: row order, a library
version, the thread count. The check is never loosened.

Value inputs per customer (P2 main at 2016-10-01 and 2016-11-01):
- p30: the calibrated 30-day probability from the explained classifier c01
  (reasons.parquet); the risk input;
- S(t) at every interval end from dt01's raw curve, used for its shape only:
  days_30_90 = trapezoid of S(t)/S(30) on 30, 37, ..., 90 (0 to 60 days);
  days_30_365 adds days 90-365 with the 81-90 daily hazard held constant
  (a labelled extrapolation);
- price per day from the last paid row before T (price.py), the auto-renew
  level and the days until the paid expiry from the features at T.
Risk (c01) and value (dt01) come from different models; how often they disagree
is reported (descriptive).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from retention.explain.model import compare_summary, logged_summary
from retention.models.classifier import candidate_params, feature_set_columns, score_summary
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.frames import survival_fit_rows
from retention.models.protocol import EVAL_CONFIG, Protocol
from retention.optimize.config import ActionsConfig

CHOICES = Path("configs/choices.yaml")
INTERVAL_ENDS_ALL = (7, 14, 21, 30, 37, 44, 51, 59, 67, 74, 81, 90)
VALUE_ENDS = (30, 37, 44, 51, 59, 67, 74, 81, 90)
EXTRA_COLUMNS = ("last_is_auto_renew", "days_until_paid_expiry")


@dataclass(frozen=True)
class Dt01Settings:
    columns: list[str]
    categorical: list[str]
    params: dict[str, Any]
    fixed: dict[str, Any]
    offset_input: bool
    ends: list[int]


def dt01_settings(eval_path: Path = EVAL_CONFIG, choices_path: Path = CHOICES) -> Dt01Settings:
    """The frozen dt01 settings, assembled as the test stage assembles them."""
    raw = yaml.safe_load(eval_path.read_text())
    choices = yaml.safe_load(choices_path.read_text())["choices"]
    if choices["survival_family"] != {"family": "discrete_time", "candidate": "dt01"}:
        raise ValueError("the frozen survival choice is not dt01")
    drop = raw["selection"]["feature_sets"][choices["feature_set"]]["drop"]
    dt = raw["survival"]["discrete_time"]
    cand = raw["classifier"]["candidates"][choices["classifier_hyperparameters"][7]]
    ends = list(dt["interval_ends"])
    if tuple(ends) != INTERVAL_ENDS_ALL:
        raise ValueError(f"interval ends {ends} differ from {INTERVAL_ENDS_ALL}")
    return Dt01Settings(
        columns=feature_set_columns(drop),
        categorical=list(raw["classifier"]["categorical"]),
        params=candidate_params(cand),
        fixed=dict(raw["classifier"]["fixed_params"]),
        offset_input=bool(dt["candidates"]["dt01"]["expiry_offset_input"]),
        ends=ends,
    )


# --- refit and reproduction ------------------------------------------------------------


def refit_dt01(
    protocol: Protocol, processed: Path, cache: Path, s: Dt01Settings, cfg: ActionsConfig
) -> tuple[DiscreteTimeModel, int]:
    tf = protocol.test_fit
    if tuple(tf.fit) != cfg.dt01_fit_snapshots or tf.as_of != cfg.dt01_labels_as_of:
        raise ValueError("the protocol's test fit differs from the registered dt01 refit")
    if protocol.seed != cfg.seed:
        raise ValueError("the protocol seed differs from the registered one")
    snaps = (*tf.fit, tf.calibration.snapshot)  # the cached build the test stage read
    rows = survival_fit_rows(protocol, processed, cache, tf.as_of, snaps, s.columns, "p2")
    fit = rows[rows.snapshot_date.dt.date.isin(tf.fit)]
    model = DiscreteTimeModel(
        s.offset_input, s.ends, s.params, s.fixed, s.categorical, protocol.threads, protocol.seed
    ).fit(fit, s.columns)
    return model, len(fit)


def reproduce(
    model: DiscreteTimeModel,
    rehearsal: dict[str, Any],
    rows_by_snapshot: dict[date, pd.DataFrame],
    cfg: ActionsConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    table, failures = [], []
    for h, snapshot, path in cfg.reproduction_cells:
        rows = rows_by_snapshot[snapshot]
        p, _ = model.lapse_probability(rows, [h])
        if path[-1] != "score_summary_raw":
            raise ValueError(f"registered path {path} does not end in score_summary_raw")
        logged = logged_summary(rehearsal, path[:-1])
        fields = compare_summary(logged, score_summary(p[:, 0]))
        table.append(
            {"horizon": h, "snapshot": snapshot.isoformat(), "rows": len(rows), "fields": fields}
        )
        bad = [f["field"] for f in fields if not f["equal"]]
        if bad:
            failures.append(f"dt01 {h} days {snapshot}: differs in {bad}")
    return table, failures


# --- expected paid days ------------------------------------------------------------------


def expected_days(surv: np.ndarray, ends: tuple[int, ...] = INTERVAL_ENDS_ALL) -> pd.DataFrame:
    """days_30_90 and days_30_365 from S(t) at every interval end (rows x ends)."""
    surv = np.asarray(surv, dtype=float)
    col = {e: k for k, e in enumerate(ends)}
    s30 = surv[:, col[30]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(s30[:, None] > 0, surv[:, [col[e] for e in VALUE_ENDS]] / s30[:, None], 0.0)
    widths = np.diff(VALUE_ENDS)
    d90 = ((r[:, :-1] + r[:, 1:]) / 2 * widths).sum(axis=1)
    s81, s90 = surv[:, col[81]], surv[:, col[90]]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s81 > 0, np.clip(s90 / s81, 0.0, 1.0) ** (1 / 9), 0.0)  # daily survival
    r90 = r[:, -1]
    n = 365 - 90
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = np.where(q < 1, (1 - q**n) / (1 - q), float(n))
    tail = r90 * (1 + q) / 2 * geo  # daily trapezoid over days 90-365
    return pd.DataFrame({"days_30_90": d90, "days_30_365": d90 + tail})


# --- the customer table ------------------------------------------------------------------


def as_date(values: pd.Series) -> pd.Series:
    """Snapshot dates as datetime.date, whether they arrive as dates or timestamps.

    Parquet date columns come back from DuckDB and pandas as datetime64; comparing
    those with a datetime.date matches nothing, so every snapshot filter uses this.
    """
    return pd.Series(pd.to_datetime(values).dt.date, index=values.index)


def read_p30(reasons: Path, snapshots: list[date]) -> pd.DataFrame:
    days = ", ".join(f"date '{t.isoformat()}'" for t in snapshots)
    con = duckdb.connect()
    try:
        out = con.sql(
            f"select msno, snapshot_date::date as snapshot_date, probability as p30 "
            f"from read_parquet('{reasons}') where horizon = 30 and snapshot_date in ({days})"
        ).df()
    finally:
        con.close()
    return out.assign(snapshot_date=as_date(out.snapshot_date))


def first_run_sha_failure(out_dir: Path, sha: str) -> str | None:
    """With a kept first run (dt01_refit_first_run.json), the rerun's booster must match it."""
    first = out_dir / "dt01_refit_first_run.json"
    if not first.exists():
        return None
    expected = json.loads(first.read_text())["booster_sha256"]
    if sha != expected:
        return f"booster SHA-256 {sha} differs from the first run's {expected}"
    return None


def customer_table(
    rows: pd.DataFrame, surv: np.ndarray, p30: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """One row per customer of one snapshot; the row counts must agree or this raises."""
    days = expected_days(surv)
    out = pd.DataFrame(
        {
            "msno": rows.msno.to_numpy(),
            "snapshot_date": as_date(rows.snapshot_date).to_numpy(),
            "auto_renew": np.where(rows.last_is_auto_renew.to_numpy() == 1, "on", "off"),
            "days_until_paid_expiry": rows.days_until_paid_expiry.astype("Int32").to_numpy(),
            "dt01_lapse_30": 1 - surv[:, INTERVAL_ENDS_ALL.index(30)],
            "days_30_90": days.days_30_90.to_numpy(),
            "days_30_365": days.days_30_365.to_numpy(),
        }
    )
    for k, e in enumerate(INTERVAL_ENDS_ALL):
        out[f"s_{e}"] = surv[:, k]
    p = p30.assign(snapshot_date=as_date(p30.snapshot_date))
    out = out.merge(p, on=["msno", "snapshot_date"], how="left", validate="one_to_one")
    pr = prices[["msno", "price_per_day", "has_price"]]
    out = out.merge(pr, on="msno", how="left", validate="one_to_one")
    if out.p30.isna().any():
        raise ValueError(f"{int(out.p30.isna().sum())} customers have no calibrated p30")
    if out.price_per_day.isna().any():
        raise ValueError("price lookup lost customers")
    return out


def disagreement(table: pd.DataFrame) -> dict[str, Any]:
    """Customers in c01's top decile of risk whose dt01 30-day lapse is below dt01's median."""
    order = np.lexsort((table.msno.astype(str).to_numpy(), -table.p30.to_numpy()))
    top = np.zeros(len(table), dtype=bool)
    top[order[: math.ceil(len(table) / 10)]] = True
    below = table.dt01_lapse_30.to_numpy() < np.median(table.dt01_lapse_30.to_numpy())
    n_top = int(top.sum())
    n = int((top & below).sum())
    return {"top_decile": n_top, "disagree": n, "share": n / n_top if n_top else None}


def headline_level(spend: float, levels: tuple[float | None, ...]) -> float | str:
    """The registered rule: the largest scan level strictly below the spend; 'no limit' if
    the spend is at or below the smallest positive level."""
    positive = sorted(b for b in levels if b is not None and b > 0)
    if spend <= positive[0]:
        return "no_limit"
    return max(b for b in positive if b < spend)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
