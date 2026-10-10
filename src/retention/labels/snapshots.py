"""labels.parquet: one row per customer and snapshot, for every customer with coverage at T.

Built "as of" a date: only transactions dated on or before as_of are used, so
labels from full data with as_of = D equal labels from data truncated at D.

State at T (rows dated before T only): effective expiry, days until expiry,
segment, flags. Outcome: follow the spell in force at T forward to its end
(the first lapse), censored at min(T + followup_cap_days, as_of - window).

Two families (see the module docs in timeline.py for the walk):
- coverage-based: any non-cancel row extends coverage;
- paid-based (prefix `paid_`): only rows with amount paid > 0 do. Defined only
  for rows whose paid coverage is active at T (`paid_state == "active"`).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from retention.labels.config import LabelConfig
from retention.labels.timeline import as_datetimes, build_timeline

SEGMENTS = ["paid", "discount100_only", "promo_only", "trial_only"]


class LabelFamily:
    """A timeline in one renewal mode, with per-spell facts precomputed."""

    def __init__(self, tx: pd.DataFrame, window: int, renewals: str):
        self.timeline, spells = build_timeline(tx, window, renewals)
        last = (
            self.timeline[self.timeline.affects_coverage]
            .groupby(["msno", "spell_id"], sort=False)
            .tail(1)
            .set_index(["msno", "spell_id"])
        )
        self.spells = spells.set_index(["msno", "spell_id"]).join(
            last[["transaction_date", "is_cancel"]].rename(
                columns={"transaction_date": "last_change_date", "is_cancel": "last_change_cancel"}
            )
        )

    def state(self, t: pd.Timestamp, same_day: bool) -> pd.DataFrame:
        tl = self.timeline
        before = tl.transaction_date <= t if same_day else tl.transaction_date < t
        last = tl[before & tl.spell_id.notna()].groupby("msno", sort=False).tail(1)
        return last.set_index("msno")[["spell_id", "eff_expiry_after"]].rename(
            columns={"eff_expiry_after": "expiry"}
        )

    def outcome(self, state: pd.DataFrame, t: pd.Timestamp, cfg: LabelConfig) -> pd.DataFrame:
        """Event time, observation and classic label for customers in `state`."""
        w = pd.Timedelta(days=cfg.renewal_window_days)
        last_labelable = pd.Timestamp(cfg.last_labelable_expiry)
        censor = min(t + pd.Timedelta(days=cfg.followup_cap_days), last_labelable)
        key = pd.MultiIndex.from_arrays([state.index, state.spell_id.astype("int64")])
        sp = self.spells.reindex(key)
        lapse = pd.Series(sp.effective_end.to_numpy(), index=state.index)
        out = pd.DataFrame(index=state.index)
        out["event_observed"] = (lapse <= censor).astype("int8")
        days = (lapse - t).dt.days.clip(lower=0)
        out["days_to_event"] = np.where(out.event_observed == 1, days, (censor - t).days).astype(
            "int32"
        )
        for h in cfg.horizons:
            known = t + pd.Timedelta(days=h) <= last_labelable
            hit = (out.event_observed == 1) & (out.days_to_event <= h)
            out[f"churned_{h}"] = pd.array(np.where(known, hit.astype(int), pd.NA), dtype="Int8")
        renewed = (sp.last_change_cancel.to_numpy() == 0) & (sp.last_change_date.to_numpy() >= t)
        renewed |= (lapse > state.expiry + w).to_numpy()
        fits = (state.expiry + w <= pd.Timestamp(cfg.as_of)).to_numpy()
        observed = (lapse + w <= pd.Timestamp(cfg.as_of)).to_numpy()
        classic = np.where(renewed, 0, np.where(observed, 1, -1))
        out["churned_classic_all"] = pd.array(
            np.where(fits & (classic >= 0), classic, pd.NA), dtype="Int8"
        )
        return out


def _segments(tl: pd.DataFrame) -> pd.DataFrame:
    """Running flags per row: has the customer had each row kind so far."""
    kinds = pd.DataFrame({k: tl.row_kind.eq(k) for k in ("paid", "discount100", "promo", "trial")})
    kinds["imputed"] = tl.expiry_imputed
    return kinds.groupby(tl.msno.to_numpy()).cummax()


def _segment_at(
    tl: pd.DataFrame, running: pd.DataFrame, t: pd.Timestamp, same_day: bool
) -> pd.DataFrame:
    before = tl.transaction_date <= t if same_day else tl.transaction_date < t
    idx = tl.index[before]
    last = running.loc[idx].groupby(tl.msno[idx].to_numpy()).tail(1)
    last.index = tl.msno[last.index].to_numpy()
    seg = np.select(
        [last.paid, last.discount100, last.promo, last.trial],
        SEGMENTS,
        "cancel_rows_only",
    )
    return pd.DataFrame({"segment": seg, "any_imputed_expiry": last.imputed}, index=last.index)


def build_labels(
    tx: pd.DataFrame, cfg: LabelConfig, reliability: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Labels for every snapshot in cfg, from transactions dated on or before cfg.as_of."""
    as_of = pd.Timestamp(cfg.as_of)
    tx = as_datetimes(tx)
    tx = tx[tx.transaction_date <= as_of]
    w = cfg.renewal_window_days
    same_day = cfg.snapshot_includes_same_day_rows
    cov = LabelFamily(tx, w, "coverage")
    paid = LabelFamily(tx, w, "paid")
    running = _segments(cov.timeline)
    trunc_end = pd.Timestamp(cfg.data_start + timedelta(days=cfg.left_truncation_days))
    far = pd.Timestamp(cfg.far_future_expiry_after)
    frames = []
    for snap in cfg.snapshots:
        t = pd.Timestamp(snap)
        st = cov.state(t, same_day)
        st = st[st.expiry >= t]
        out = pd.DataFrame(index=st.index)
        out["snapshot_date"] = t
        out["expiry_date"] = st.expiry
        out["days_until_expiry"] = (st.expiry - t).dt.days.astype("int32")
        out["in_exp30"] = out.days_until_expiry < w
        out["in_expiry_month"] = st.expiry < t + pd.offsets.MonthBegin()
        out = out.join(_segment_at(cov.timeline, running, t, same_day))
        out["stacked_far_future"] = st.expiry > far
        out["left_truncation_risk"] = t < trunc_end
        o = cov.outcome(st, t, cfg)
        o["churned_classic"] = o.pop("churned_classic_all").where(out.in_expiry_month)
        out = out.join(o)

        pst = paid.state(t, same_day).reindex(out.index)
        out["paid_expiry_date"] = pst.expiry
        out["paid_state"] = np.select(
            [pst.expiry.isna(), pst.expiry >= t, pst.expiry >= t - pd.Timedelta(days=w)],
            ["never_paid", "active", "grace"],
            "lapsed",
        )
        active = pst[out.paid_state.eq("active")]
        po = paid.outcome(active, t, cfg)
        po["churned_classic"] = po.pop("churned_classic_all").where(
            active.expiry < t + pd.offsets.MonthBegin()
        )
        out = out.join(po.add_prefix("paid_"))
        frames.append(out)
    labels = pd.concat(frames).rename_axis("msno").reset_index()
    labels["snapshot_date"] = labels.snapshot_date.dt.date
    if reliability is not None:
        r = reliability.assign(snapshot=pd.to_datetime(reliability.snapshot).dt.date)
        wide = r.pivot(index="snapshot", columns="horizon", values="status")
        for h in wide.columns:
            labels[f"reliability_status_{h}"] = labels.snapshot_date.map(wide[h])
            labels[f"label_reliable_{h}"] = labels[f"reliability_status_{h}"].eq("reliable")
    for c in [
        c for c in labels.columns if c.startswith("paid_") and c.endswith(("_7", "_30", "_60"))
    ]:
        labels[c] = labels[c].astype("Int8")
    for c in ("paid_event_observed", "paid_churned_classic"):
        labels[c] = labels[c].astype("Int8")
    labels["paid_days_to_event"] = labels.paid_days_to_event.astype("Int32")
    return labels
