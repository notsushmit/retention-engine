"""Comparison with the competition labels in train.csv (February 2017 expiries).

Three labels are compared with train for the customers whose effective expiry
at 2017-02-01 falls in February 2017:

- ours: churned_classic from our own rule (coverage-based and paid-based);
- fitted_to_train: our rule plus three habits found BY LOOKING AT TRAIN'S
  LABELS, so its agreement is fitted, not independent: cancels count only up to
  the end of the expiry month (2017-02-28), and a renewal counts only if it is
  dated on or after the expiry (and at most 30 days after it). The inclusive
  30-day boundary is already part of our rule;
- scala_official: the exact port of the published labeller (official.py), for
  the customers it treats as candidates.

Every disagreement gets exactly one reason code, the first that applies in
REASONS. The first three codes only apply to our own labels; for the fitted
variant they cannot explain anything, because the variant already includes them.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from retention.labels.config import LabelConfig
from retention.labels.official import scala_labels
from retention.labels.snapshots import LabelFamily
from retention.labels.timeline import as_datetimes

REASONS = [
    "march_cancel",  # train ignores cancels after the expiry month; fixed by that rule alone
    "early_renewal",  # train ignores renewals dated before the expiry; fixed by that rule alone
    "march_cancel_and_early_renewal",  # fixed only by both rules together
    "zero_paid_renewal",  # the first renewal in the window paid nothing
    "same_day_same_plan_pair",  # a same-day subscribe-and-cancel of one plan in the window
    "plan_change_at_renewal",  # the first renewal changes plan length or list price
    "cancel_in_window",  # another cancel in the window
    "other",
]
OWN_ONLY = REASONS[:3]


def _metrics(train: pd.Series, label: pd.Series) -> dict[str, Any]:
    ok = label.notna()
    y, p = train[ok].astype(int), label[ok].astype(int)
    tp, fp, fn = (
        int(((p == 1) & (y == 1)).sum()),
        int(((p == 1) & (y == 0)).sum()),
        int(((p == 0) & (y == 1)).sum()),
    )
    po = float((y == p).mean())
    pe = float(y.mean() * p.mean() + (1 - y.mean()) * (1 - p.mean()))
    return {
        "n": int(ok.sum()),
        "unknown": int((~ok).sum()),
        "agreement": round(100 * po, 2),
        "churn_recall": round(100 * tp / (tp + fn), 2) if tp + fn else None,
        "churn_precision": round(100 * tp / (tp + fp), 2) if tp + fp else None,
        "kappa": round((po - pe) / (1 - pe), 4),
        "train_churn_pct": round(100 * float(y.mean()), 2),
        "label_churn_pct": round(100 * float(p.mean()), 2),
    }


def _window_facts(
    fam: LabelFamily, state: pd.DataFrame, t: pd.Timestamp, w: int, paid_only: bool
) -> pd.DataFrame:
    """Facts about rows dated in [T, expiry + w] used by the early-renewal rule and reason codes."""
    tl = fam.timeline.join(state.expiry, on="msno", how="inner")
    win = tl[(tl.transaction_date >= t) & (tl.transaction_date <= tl.expiry + pd.Timedelta(days=w))]
    renew = win[(win.is_cancel == 0) & ((win.actual_amount_paid > 0) | (not paid_only))]
    first = renew.groupby("msno").head(1).set_index("msno")
    before = fam.timeline[(fam.timeline.transaction_date < t) & (fam.timeline.is_cancel == 0)]
    plan_before = (
        before.groupby("msno").tail(1).set_index("msno")[["payment_plan_days", "plan_list_price"]]
    )
    out = pd.DataFrame(index=state.index)
    out["on_time_renewal"] = out.index.isin(renew[renew.transaction_date >= renew.expiry].msno)
    out["first_renewal_paid0"] = first.actual_amount_paid.reindex(out.index).eq(0)
    pb = plan_before.reindex(first.index)
    changed = (first.payment_plan_days != pb.payment_plan_days) | (
        first.plan_list_price != pb.plan_list_price
    )
    out["first_renewal_plan_change"] = changed.reindex(out.index).fillna(False).astype(bool)
    out["sameday_pair"] = out.index.isin(win[win.sameday_sameplan_cancel].msno)
    out["cancel_in_window"] = out.index.isin(win[win.is_cancel == 1].msno)
    return out


def _early(base: pd.Series, facts: pd.DataFrame) -> pd.Series:
    """Train's habit: without a renewal dated on or after the expiry, the customer churned."""
    return base.where(base.isna() | (base == 1), (~facts.on_time_renewal).astype(float))


def _reason(row: pd.Series, label: str, applicable: list[str]) -> str:
    checks = {
        "march_cancel": row["febcancel"] == row["train"],
        "early_renewal": row["early"] == row["train"],
        "march_cancel_and_early_renewal": row["fitted_to_train"] == row["train"],
        "zero_paid_renewal": row["first_renewal_paid0"],
        "same_day_same_plan_pair": row["sameday_pair"],
        "plan_change_at_renewal": row["first_renewal_plan_change"],
        "cancel_in_window": row["cancel_in_window"],
    }
    return next((r for r in applicable if checks[r]), "other")


def _family(
    tx: pd.DataFrame,
    train: pd.Series,
    cfg: LabelConfig,
    renewals: str,
    t: pd.Timestamp,
    cancel_until: pd.Timestamp,
):
    w = cfg.renewal_window_days
    fam = LabelFamily(tx, w, renewals)
    feb = LabelFamily(tx[(tx.is_cancel == 0) | (tx.transaction_date <= cancel_until)], w, renewals)
    state = fam.state(t, cfg.snapshot_includes_same_day_rows)
    group = pd.Series("no_state_at_T", index=train.index)
    exp = state.expiry.reindex(train.index)
    nxt = t + pd.offsets.MonthBegin()
    group[exp < t] = "expiry_before_T"
    group[exp >= nxt] = "expiry_after_month"
    group[(exp >= t) & (exp < nxt)] = "feb_expiry"
    cohort = state[state.index.isin(group.index[group == "feb_expiry"])]
    df = pd.DataFrame({"train": train.reindex(cohort.index)})
    df["ours"] = fam.outcome(cohort, t, cfg).churned_classic_all.astype(float)
    df["febcancel"] = feb.outcome(
        feb.state(t, False).reindex(cohort.index), t, cfg
    ).churned_classic_all.astype(float)
    facts = _window_facts(fam, cohort, t, w, renewals == "paid")
    df = df.join(facts)
    df["early"] = _early(df.ours, facts)
    df["fitted_to_train"] = _early(df.febcancel, facts)
    for label, applicable in (("ours", REASONS[:-1]), ("fitted_to_train", REASONS[3:-1])):
        dis = df[label].notna() & (df[label] != df.train)
        df[f"{label}_reason"] = None
        df.loc[dis, f"{label}_reason"] = df[dis].apply(_reason, axis=1, args=(label, applicable))
    return group, df


def _rate(df: pd.DataFrame, by: str) -> dict[str, dict[str, float]]:
    g = df.groupby(by).train.agg(
        n="size", train_churn_pct=lambda s: round(100 * float(s.mean()), 2)
    )
    return {
        str(k): {"n": int(v.n), "train_churn_pct": float(v.train_churn_pct)}
        for k, v in g.iterrows()
    }


def rule_tables(
    tx: pd.DataFrame, train: pd.Series, cfg: LabelConfig, t: pd.Timestamp, sc: pd.DataFrame
) -> dict[str, Any]:
    """Train's churn rate in the situations that separate its labels from ours (coverage-based)."""
    tr = train.rename("train")
    out: dict[str, Any] = {}
    c = sc[sc.candidate & sc.gap.between(26, 36)].join(tr, how="inner")
    out["by_scala_gap_days"] = _rate(c.assign(gap=c.gap.astype(int)), "gap")
    fam = LabelFamily(tx, cfg.renewal_window_days, "coverage")
    st = fam.state(t, cfg.snapshot_includes_same_day_rows)
    st = st[(st.expiry >= t) & (st.expiry < t + pd.offsets.MonthBegin())]
    tl = fam.timeline.join(st.expiry, on="msno", how="inner")
    win = tl[
        (tl.transaction_date >= t)
        & (tl.transaction_date <= tl.expiry + pd.Timedelta(days=cfg.renewal_window_days))
    ]
    ren = win[win.is_cancel == 0]
    first = ren.groupby("msno").transaction_date.min()
    cut = win[(win.is_cancel == 1) & win.affects_coverage].join(
        first.rename("first"), on="msno", how="inner"
    )
    cut = cut[cut.transaction_date >= cut["first"]].groupby("msno").transaction_date.max()
    after = ren.join(cut.rename("cancel"), on="msno", how="inner")
    again = set(after[after.transaction_date > after.cancel].msno)
    b = pd.DataFrame({"cancel_month": cut.dt.strftime("%Y-%m")})
    out["cancel_after_renewal_by_month"] = _rate(
        b[~b.index.isin(again)].join(tr, how="inner"), "cancel_month"
    )
    early = first[first < st.expiry.reindex(first.index)].index
    on_time = set(ren[ren.transaction_date >= ren.expiry].msno)
    e = pd.DataFrame({"later_on_time_renewal": [m in on_time for m in early]}, index=early)
    out["early_first_renewal"] = _rate(e.join(tr, how="inner"), "later_on_time_renewal")
    paid = ren.groupby("msno").actual_amount_paid.max()
    kind = ren[ren.msno.isin(paid.index[paid == 0])].groupby("msno").row_kind.first()
    out["only_zero_paid_renewals_by_kind"] = _rate(
        pd.DataFrame({"kind": kind}).join(tr, how="inner"), "kind"
    )
    return out


def compare(
    tx: pd.DataFrame, train: pd.Series, cfg: LabelConfig, snapshot: date = date(2017, 2, 1)
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tx = as_datetimes(tx)
    tx = tx[tx.transaction_date <= pd.Timestamp(cfg.as_of)]
    t = pd.Timestamp(snapshot)
    cancel_until = t + pd.offsets.MonthBegin() - pd.Timedelta(days=1)
    table = pd.DataFrame({"train_is_churn": train})
    report: dict[str, Any] = {"snapshot": str(snapshot), "cancel_until": str(cancel_until.date())}
    for renewals in ("coverage", "paid"):
        group, df = _family(tx, train, cfg, renewals, t, cancel_until)
        p = f"{renewals}_"
        table[p + "group"] = group
        for c in ("ours", "fitted_to_train", "ours_reason", "fitted_to_train_reason"):
            table[p + c] = df[c].reindex(table.index)
        g = pd.DataFrame({"group": group, "train": train})
        fam_report: dict[str, Any] = {"groups": _rate(g, "group")}
        for label in ("ours", "fitted_to_train"):
            reasons = df[f"{label}_reason"].value_counts()
            n_dis = int((df[label].notna() & (df[label] != df.train)).sum())
            assert int(reasons.sum()) == n_dis
            fam_report[label] = {
                **_metrics(df.train, df[label]),
                "disagreements": n_dis,
                "reasons": {
                    r: int(reasons.get(r, 0)) for r in REASONS if r in reasons or r == "other"
                },
                "other_share_pct": round(100 * reasons.get("other", 0) / n_dis, 1)
                if n_dis
                else 0.0,
            }
        report[renewals] = fam_report
    hist_start = (t - pd.offsets.MonthBegin()).date()
    sc = scala_labels(
        tx, hist_start, (t - pd.Timedelta(days=1)).date(), t.date(), cancel_until.date()
    )
    table["scala_candidate"] = sc.candidate.reindex(table.index).fillna(False).astype(bool)
    table["scala_gap"] = sc.gap.reindex(table.index)
    table["scala_is_churn"] = sc.is_churn.reindex(table.index)
    cand = table[table.scala_candidate]
    report["rule_tables"] = rule_tables(tx, train, cfg, t, sc)
    report["scala_official"] = {
        "candidates": int(len(cand)),
        **_metrics(cand.train_is_churn, cand.scala_is_churn.astype(float)),
    }
    return table.rename_axis("msno").reset_index(), report
