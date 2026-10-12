"""The check month (2016-11): what drives its high 7-day lapse rate. Descriptive only.

Nothing here changes a choice or a registered value. Only 2016 data is read:
the check month and the training months 2016-08 to 2016-10 (and, for the
never-seen definition, the three monthly snapshots before each month).

Definitions, fixed before computing:
- Rows: P2 (expiry within 30 days of T). Segment tables cover every segment;
  the tenure, never-seen and model tables cover the main population (paid and
  100%-discount-only).
- Never seen at month M: no row of any segment in the three monthly snapshots
  before M (for 2016-11: 2016-08, 09 and 10). Returning: the rest.
- Tenure bands (tenure_days, from registration): < 90 days, 90-364, 1-2 years,
  2-5 years, 5+ years, missing (no listening before T masks member columns).
- Rates carry Wilson 95% intervals; one row per customer and month.
- Last row before T paid or zero-paid: a zero-paid last row (trial, promo or
  100% discount) is exactly where days_until_paid_expiry differs from
  days_until_expiry. Added after the first run showed the 2016-11 excess sits
  there; it is a breakdown, not a rule.
- "Close to the training months" (question 3): the 2016-11 rate of returning,
  paid-segment customers lies between the lowest lower bound and the highest
  upper bound of the three training months' 95% intervals. If it does, the
  elevated rate is called a customer-mix effect; if not, it stays unexplained.
- The model is the chosen 7-day classifier (c01 on fs1) fitted on the f7b fit
  rows (2016-08/09, labels as of 2016-11-01) and scored on 2016-11; its raw
  probabilities are not calibrated (calibration is a later step).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from retention.models.bootstrap import bootstrap
from retention.models.classifier import Classifier, candidate_params, feature_set_columns
from retention.models.data import load_features, load_labels
from retention.models.frames import fit_rows, score_rows
from retention.models.metrics import NotComputable, SortedScores, calibration_slope
from retention.models.protocol import EVAL_CONFIG, Protocol

MONTHS = [date(2016, 8, 1), date(2016, 9, 1), date(2016, 10, 1), date(2016, 11, 1)]
TENURE_EDGES = [
    (0, 90, "< 90 days"),
    (90, 365, "90-364 days"),
    (365, 730, "1-2 years"),
    (730, 1826, "2-5 years"),
    (1826, 10**6, "5+ years"),
]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def tenure_band(days: pd.Series) -> pd.Series:
    out = pd.Series("missing", index=days.index, dtype="object")
    d = days.astype("Float64")
    for lo, hi, name in TENURE_EDGES:
        out[(d >= lo) & (d < hi)] = name
    return out


def month_rows(protocol: Protocol, processed: Path, month: date) -> pd.DataFrame:
    """P2 rows of every segment for one month, with churned_7, tenure and never-seen."""
    lab = load_labels(
        processed / "labels.parquet", [month], ["in_exp30", "segment", "churned_7"], protocol
    )
    lab = lab[lab.in_exp30.astype(bool)].copy()
    cols = ["tenure_days", "days_until_expiry", "days_until_paid_expiry"]
    feats = load_features(processed / "features.parquet", [month], cols)
    lab = lab.merge(feats[["msno", *cols]], on="msno", how="left", validate="one_to_one")
    zero_paid = (lab.days_until_paid_expiry != lab.days_until_expiry).fillna(True)
    lab["last_row"] = np.where(zero_paid, "zero-paid last row", "paid last row")
    before = [
        date(month.year + (month.month - k - 1) // 12, (month.month - k - 1) % 12 + 1, 1)
        for k in (1, 2, 3)
    ]
    seen = set(load_labels(processed / "labels.parquet", before, ["segment"], protocol).msno)
    lab["never_seen"] = ~lab.msno.isin(seen)
    lab["tenure_band"] = tenure_band(lab.tenure_days)
    lab["main"] = lab.segment.isin(protocol.main_segments)
    return lab


def group_table(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    total_rows, total_lapses = len(df), int(df.churned_7.sum())
    out = []
    for key, g in df.groupby(by, dropna=False):
        k, n = int(g.churned_7.sum()), len(g)
        rate, lo, hi = wilson(k, n)
        out.append(
            {
                "group": str(key),
                "rows": n,
                "lapses": k,
                "rate": rate,
                "lo": lo,
                "hi": hi,
                "share_of_rows": n / total_rows,
                "share_of_lapses": k / total_lapses if total_lapses else float("nan"),
            }
        )
    return out


def _metric_interval(customers, stat, replicates, seed) -> dict[str, Any]:
    iv = bootstrap([customers], lambda f, w: stat(w), replicates, seed)
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi, "status": iv.status, "reason": iv.reason}


def model_checks(protocol, processed, cache, replicates) -> dict[str, Any]:
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    fold = protocol.fold("f7b")
    cols = feature_set_columns(raw["selection"]["feature_sets"]["fs1"]["drop"])
    fit = fit_rows(protocol, processed, cache, fold.as_of, fold.fit, cols, 7)
    rows = score_rows(protocol, processed, fold.score, cols, 7)
    before = [date(2016, 8, 1), date(2016, 9, 1), date(2016, 10, 1)]
    seen = set(load_labels(processed / "labels.parquet", before, ["segment"], protocol).msno)
    never = ~rows.msno.isin(seen).to_numpy()
    paid = (rows.segment == "paid").to_numpy()
    y = rows.y.to_numpy()

    def model(columns):
        return Classifier(
            columns,
            raw["classifier"]["categorical"],
            candidate_params(raw["classifier"]["candidates"]["c01"]),
            raw["classifier"]["fixed_params"],
            protocol.threads,
            protocol.seed,
        ).fit(fit)

    full = model(cols)
    p = full.predict(rows)[0]
    seed, cust = protocol.seed, rows.msno.to_numpy()

    def auc_on(mask, score=p):
        s = SortedScores(score[mask])
        return lambda w: s.auc(y[mask], None if w is None else w[mask])

    def calib_on(mask, key):
        def stat(w):
            return calibration_slope(y[mask], p[mask], None if w is None else w[mask])[key]

        return stat

    out: dict[str, Any] = {
        "rows": len(rows),
        "lapses": int(y.sum()),
        "never_seen_rows": int(never.sum()),
        "never_seen_lapses": int(y[never].sum()),
    }
    groups = {
        "all P2 main": np.ones(len(y), bool),
        "never seen": never,
        "returning": ~never,
        "paid segment only (without 100%-discount-only)": paid,
    }
    out["auc"] = {g: _metric_interval(cust, auc_on(m), replicates, seed) for g, m in groups.items()}
    out["calibration_raw"] = {
        g: {
            k: _metric_interval(cust, calib_on(m, k), replicates, seed)
            for k in ("slope", "intercept")
        }
        for g, m in groups.items()
    }
    out["mean_score_vs_rate"] = {
        g: {"mean_score": float(p[m].mean()), "rate": float(y[m].mean())} for g, m in groups.items()
    }

    def diff(m1, m2):
        a, b = auc_on(m1), auc_on(m2)
        return lambda w: a(w) - b(w)

    out["auc_difference"] = {
        "never seen minus returning": _metric_interval(cust, diff(never, ~never), replicates, seed),
        "paid only minus all P2 main": _metric_interval(
            cust, diff(paid, np.ones(len(y), bool)), replicates, seed
        ),
    }

    # Question 4: refit without days_until_paid_expiry (descriptive, f7b rows only).
    reduced_cols = [c for c in cols if c != "days_until_paid_expiry"]
    reduced = model(reduced_cols)
    q = reduced.predict(rows)[0]
    sp, sq = SortedScores(p), SortedScores(q)
    out["without_days_until_paid_expiry"] = {
        "auc_full_minus_without": _metric_interval(
            cust, lambda w: sp.auc(y, w) - sq.auc(y, w), replicates, seed
        ),
        "gain_share_full": _shares(full.importance()),
        "gain_share_without": _shares(reduced.importance()),
    }
    differ = (rows.days_until_paid_expiry != rows.days_until_expiry).fillna(True).to_numpy()
    kinds = rows.last_row_kind.astype("object").to_numpy()
    out["expiry_columns_differ"] = {
        "rows": int(differ.sum()),
        "share": float(differ.mean()),
        "lapses": int(y[differ].sum()),
        "last_row_kind_when_they_differ": pd.Series(kinds[differ]).value_counts().to_dict(),
        "last_row_kind_when_equal": pd.Series(kinds[~differ]).value_counts().to_dict(),
    }
    for name, mask in (("rows where they differ", differ), ("rows where they are equal", ~differ)):
        try:
            out["without_days_until_paid_expiry"][f"auc_full_minus_without, {name}"] = (
                _metric_interval(
                    cust,
                    lambda w, m=mask: (
                        SortedScores(p[m]).auc(y[m], None if w is None else w[m])
                        - SortedScores(q[m]).auc(y[m], None if w is None else w[m])
                    ),
                    replicates,
                    seed,
                )
            )
        except NotComputable as err:
            out["without_days_until_paid_expiry"][f"auc_full_minus_without, {name}"] = str(err)
    return out


def _shares(gain: dict[str, float]) -> dict[str, float]:
    total = sum(gain.values())
    return {k: v / total for k, v in list(gain.items())[:8]}


def check_month(
    protocol: Protocol, processed: Path, cache: Path, replicates: int
) -> dict[str, Any]:
    tables: dict[str, Any] = {
        "segment": {},
        "tenure_band": {},
        "never_seen": {},
        "last_row": {},
        "never_seen_by_last_row": {},
        "returning_paid": {},
    }
    for m in MONTHS:
        df = month_rows(protocol, processed, m)
        main = df[df.main]
        tables["segment"][str(m)] = group_table(df, "segment")
        tables["tenure_band"][str(m)] = group_table(main, "tenure_band")
        seen_group = np.where(main.never_seen, "never seen", "returning")
        tables["never_seen"][str(m)] = group_table(main.assign(group=seen_group), "group")
        tables["last_row"][str(m)] = group_table(main, "last_row")
        both = pd.Series(seen_group, index=main.index) + ", " + main.last_row
        tables["never_seen_by_last_row"][str(m)] = group_table(main.assign(group=both), "group")
        rp = main[~main.never_seen & (main.segment == "paid")]
        rate, lo, hi = wilson(int(rp.churned_7.sum()), len(rp))
        tables["returning_paid"][str(m)] = {
            "rows": len(rp),
            "lapses": int(rp.churned_7.sum()),
            "rate": rate,
            "lo": lo,
            "hi": hi,
        }
    train = [tables["returning_paid"][str(m)] for m in MONTHS[:3]]
    nov = tables["returning_paid"][str(MONTHS[3])]
    band = (min(t["lo"] for t in train), max(t["hi"] for t in train))
    close = band[0] <= nov["rate"] <= band[1]
    tables["question_3"] = {
        "training_band": band,
        "check_month_rate": nov["rate"],
        "close": bool(close),
        "verdict": "customer-mix effect" if close else "unexplained",
    }
    tables["model"] = model_checks(protocol, processed, cache, replicates)
    return tables
