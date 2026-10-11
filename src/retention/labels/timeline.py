"""Each customer's subscription timeline: ordered rows, effective expiry, spells.

Order within a customer: transaction date, then on the same day a cancel of a
plan with no same-day subscription to it, then subscriptions, then cancels of a
plan subscribed that same day; then expiry (later last for subscriptions,
earlier last for cancels); then content (TIE_KEYS); then source file and row
number, which only orders rows that are identical in all of the above.

Walking that order once gives the effective expiry after every row:
- a subscription more than `window` days after the effective expiry starts a
  new spell; otherwise it extends the expiry to max(current, its expiry);
- a cancel sets the expiry to min(current, its expiry), never later. It is
  ignored when it is a same-day same-plan cancel, when its expiry is a
  placeholder (no expiry), before any subscription, or after the spell has
  already lapsed (dated more than `window` days past the effective expiry).
- with renewals="paid", zero-paid subscriptions (trial, promo, 100% discount)
  are ignored, so coverage counts paid time only.

Every value for a row depends only on that row and earlier rows, so the
timeline built from rows dated up to D equals the full timeline cut at D.
spells.parquet holds raw facts only; whether a lapse is observed depends on
an as_of date and is decided when labels are built.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-01", "D")


def _days(s: pd.Series) -> np.ndarray:
    """Dates as int days since 1970-01-01; NaT becomes the minimum int64."""
    v = pd.to_datetime(s).to_numpy(dtype="datetime64[D]")
    out = (v - EPOCH).astype("int64")
    out[np.isnat(v)] = np.iinfo(np.int64).min
    return out


def as_datetimes(tx: pd.DataFrame) -> pd.DataFrame:
    """Date columns as datetime64 whether they were read by DuckDB or pyarrow."""
    cols = [c for c in ("transaction_date", "membership_expire_date", "expiry_date") if c in tx]
    return tx.assign(**{c: pd.to_datetime(tx[c]) for c in cols})


# Rows tied on date, kind of row and expiry are ordered by content, so the order
# does not depend on file position: paid rows before zero-paid rows, then amount
# paid, list price, plan days, payment method, auto-renew and row kind, all
# descending. Rows equal in all of these are interchangeable; file position only
# makes the sort total.
TIE_KEYS = [
    "actual_amount_paid",
    "plan_list_price",
    "payment_plan_days",
    "payment_method_id",
    "is_auto_renew",
    "row_kind",
]


def order_rows(tx: pd.DataFrame) -> pd.DataFrame:
    """Sort transactions into timeline order and number them per customer."""
    rank = np.where(tx.is_cancel == 0, 1, np.where(tx.sameday_sameplan_cancel, 2, 0))
    exp = _days(tx.expiry_date)
    exp_key = np.where(tx.is_cancel == 0, exp, -exp)
    zero_paid = (tx.actual_amount_paid <= 0).astype("int8")
    keys = ["msno", "transaction_date", "_rank", "_exp_key", "_zero_paid", *TIE_KEYS]
    out = tx.assign(_rank=rank, _exp_key=exp_key, _zero_paid=zero_paid).sort_values(
        [*keys, "source", "source_row"],
        ascending=[True] * 5 + [False] * len(TIE_KEYS) + [True, True],
        kind="stable",
    )
    out = out.drop(columns=["_rank", "_exp_key", "_zero_paid"]).reset_index(drop=True)
    out["seq"] = out.groupby("msno").cumcount()
    return out


RENEWALS = ("coverage", "paid")


def build_timeline(
    tx: pd.DataFrame, window: int, renewals: str = "coverage"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (timeline, spells). `tx` is the clean transactions table (any row order).

    renewals="coverage": every non-cancel row can start or extend coverage.
    renewals="paid": only non-cancel rows with amount paid > 0 can; trial, promo
    and 100%-discount rows change nothing.
    """
    if renewals not in RENEWALS:
        raise ValueError(f"renewals must be one of {RENEWALS}: {renewals!r}")
    tl = order_rows(as_datetimes(tx))
    none = np.iinfo(np.int64).min
    codes = pd.factorize(tl.msno)[0]
    td = _days(tl.transaction_date)
    ex = _days(tl.expiry_date)
    cancel = (tl.is_cancel == 1).to_numpy()
    inert = (tl.sameday_sameplan_cancel | tl.expiry_date.isna()).to_numpy() & cancel
    if renewals == "paid":
        inert |= ~cancel & (tl.actual_amount_paid == 0).to_numpy()

    n = len(tl)
    eff_after = np.full(n, none)
    spell_id = np.full(n, -1)
    affects = np.zeros(n, bool)
    starts = np.zeros(n, bool)
    spells: list[tuple[int, int, int, int, int]] = []  # code, spell, start, end, next_sub

    prev, eff, sid, start = -1, none, -1, none
    for i, (c, t, e, k, z) in enumerate(
        zip(codes.tolist(), td.tolist(), ex.tolist(), cancel.tolist(), inert.tolist(), strict=True)
    ):
        if c != prev:
            if sid >= 0:
                spells.append((prev, sid, start, eff, none))
            prev, eff, sid, start = c, none, -1, none
        if k:
            if not z and eff != none and t - eff <= window and e < eff:
                eff = e
                affects[i] = True
        elif z:
            pass
        elif eff == none or t - eff > window:
            if sid >= 0:
                spells.append((c, sid, start, eff, t))
            sid, start, eff = sid + 1, t, e
            affects[i] = starts[i] = True
        elif e > eff:
            eff = e
            affects[i] = True
        eff_after[i] = eff
        spell_id[i] = sid
    if sid >= 0:
        spells.append((prev, sid, start, eff, none))

    def to_date(a: np.ndarray) -> np.ndarray:
        out = (a.astype("timedelta64[D]") + EPOCH).astype("datetime64[ns]")
        out[a == none] = np.datetime64("NaT")
        return out

    tl["eff_expiry_after"] = to_date(eff_after)
    tl["spell_id"] = pd.array(np.where(spell_id >= 0, spell_id, None), dtype="Int32")
    tl["affects_coverage"] = affects
    tl["starts_spell"] = starts

    sp = pd.DataFrame(spells, columns=["code", "spell_id", "start", "end", "next_sub"])
    sp.insert(0, "msno", tl.msno.unique()[sp.code.to_numpy()] if len(sp) else [])
    sp["gap_days"] = pd.array(
        np.where(sp.next_sub != none, sp.next_sub - sp.end, None), dtype="Int32"
    )
    spells_df = pd.DataFrame(
        {
            "msno": sp.msno,
            "spell_id": sp.spell_id.astype("int32"),
            "spell_start": to_date(sp.start.to_numpy()),
            "effective_end": to_date(sp.end.to_numpy()),
            "next_sub_date": to_date(sp.next_sub.to_numpy()),
            "gap_days": sp.gap_days,
        }
    )
    return tl, spells_df


def in_force_at(timeline: pd.DataFrame, snapshot: date, include_same_day: bool) -> pd.DataFrame:
    """Per customer, the last row counted as before the snapshot and its effective expiry.

    A row counts as before T when its transaction date is strictly before T,
    unless `include_same_day` is set (rows dated exactly T then count too).
    """
    t = pd.Timestamp(snapshot)
    before = timeline.transaction_date <= t if include_same_day else timeline.transaction_date < t
    last = timeline[before & timeline.spell_id.notna()].groupby("msno", sort=False).tail(1)
    return last.set_index("msno")[["seq", "spell_id", "eff_expiry_after"]]


def outcome_at(
    timeline: pd.DataFrame,
    spells: pd.DataFrame,
    snapshot: date,
    as_of: date,
    window: int,
    include_same_day: bool = False,
) -> pd.DataFrame:
    """State at T and what happened to the spell in force at T, as seen on `as_of`.

    Columns: expiry (effective expiry at T), spell_id, ever_paid (a paid row
    before T), lapse_date (the spell's effective end), lapse_observed (no new
    subscription within `window` days after it, visible by as_of), and classic:
    for the expiry in force at T, 1 = lapsed, 0 = renewed, NaN = not yet known.
    A customer renewed this expiry if the spell's last coverage change was a
    subscription dated on or after T, or if coverage ran past expiry + window.
    The timeline and spells must not contain rows dated after `as_of`.
    """
    t, a = pd.Timestamp(snapshot), pd.Timestamp(as_of)
    if (timeline.transaction_date > a).any():
        raise ValueError("timeline has rows dated after as_of; build it from rows <= as_of")
    state = in_force_at(timeline, snapshot, include_same_day)
    before = timeline.transaction_date <= t if include_same_day else timeline.transaction_date < t
    paid = timeline[before & timeline.row_kind.eq("paid")].msno.unique()
    last_change = (
        timeline[timeline.affects_coverage]
        .groupby(["msno", "spell_id"], sort=False)
        .tail(1)
        .set_index(["msno", "spell_id"])[["transaction_date", "is_cancel"]]
    )
    out = state.rename(columns={"eff_expiry_after": "expiry"}).drop(columns="seq")
    key = pd.MultiIndex.from_arrays([out.index, out.spell_id.astype("int64")])
    sp = spells.set_index(["msno", "spell_id"]).reindex(key)
    lc = last_change.reindex(key)
    w = pd.Timedelta(days=window)
    out["ever_paid"] = out.index.isin(paid)
    out["lapse_date"] = sp.effective_end.to_numpy()
    closed = sp.next_sub_date.notna().to_numpy()
    out["lapse_observed"] = closed | (out.lapse_date + w <= a).to_numpy()
    renewed = (lc.is_cancel.to_numpy() == 0) & (lc.transaction_date.to_numpy() >= t)
    renewed |= (out.lapse_date > out.expiry + w).to_numpy()
    out["classic"] = np.where(renewed, 0.0, np.where(out.lapse_observed, 1.0, np.nan))
    return out
