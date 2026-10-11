import random
from datetime import date, timedelta

import pandas as pd
import pytest

from retention.labels.timeline import build_timeline, in_force_at, order_rows

WINDOW = 30


def row(td, exp, cancel=0, m="a", pm=41, days=30, price=149, paid=149, ar=1):
    return f"{m},{pm},{days},{price},{paid},{ar},{td},{exp},{cancel}"


def eff(tl, msno="a"):
    """Effective expiry after the customer's last row, as YYYY-MM-DD."""
    return str(tl[tl.msno == msno].eff_expiry_after.iloc[-1].date())


def build(clean_tx, v1, v2=()):
    return build_timeline(clean_tx(v1, v2), WINDOW)


def test_gap_of_30_days_continues_31_starts_new_spell(clean_tx):
    tl, spells = build(
        clean_tx,
        [
            row(20160101, 20160131, m="a"),
            row(20160301, 20160331, m="a"),  # 30 days after 2016-01-31
            row(20160101, 20160131, m="b"),
            row(20160302, 20160401, m="b"),  # 31 days after
        ],
    )
    assert tl[tl.msno == "a"].spell_id.tolist() == [0, 0]
    assert tl[tl.msno == "b"].spell_id.tolist() == [0, 1]
    b0 = spells[(spells.msno == "b") & (spells.spell_id == 0)].iloc[0]
    assert str(b0.effective_end.date()) == "2016-01-31"
    assert str(b0.next_sub_date.date()) == "2016-03-02" and b0.gap_days == 31
    last = spells[(spells.msno == "a")].iloc[-1]
    assert pd.isna(last.next_sub_date) and pd.isna(last.gap_days)


def test_early_renewal_extends_same_spell(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160125, 20160301)])
    assert tl.spell_id.tolist() == [0, 0] and eff(tl) == "2016-03-01"


def test_cancel_cuts_then_renewal_within_window_continues(clean_tx):
    tl, spells = build(
        clean_tx,
        [
            row(20160101, 20160131, m="a"),
            row(20160110, 20160110, cancel=1, m="a"),
            row(20160205, 20160306, m="a"),  # 26 days after the cut-back expiry
            row(20160101, 20160131, m="b"),
            row(20160110, 20160110, cancel=1, m="b"),
            row(20160215, 20160316, m="b"),  # 36 days after: new spell
        ],
    )
    a = tl[tl.msno == "a"]
    assert [str(d.date()) for d in a.eff_expiry_after] == ["2016-01-31", "2016-01-10", "2016-03-06"]
    assert a.spell_id.tolist() == [0, 0, 0]
    b0 = spells[(spells.msno == "b") & (spells.spell_id == 0)].iloc[0]
    assert str(b0.effective_end.date()) == "2016-01-10" and b0.gap_days == 36


def test_late_renewal_starts_new_spell(clean_tx):
    tl, spells = build(clean_tx, [row(20160101, 20160131), row(20160320, 20160419)])
    assert tl.spell_id.tolist() == [0, 1]
    assert spells.iloc[0].gap_days == 49


def test_cancel_never_extends_coverage(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160115, 20160215, cancel=1)])
    assert eff(tl) == "2016-01-31"
    assert not tl.affects_coverage.iloc[-1]


def test_cancel_after_lapse_does_not_rewrite_closed_spell(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160315, 20160110, cancel=1)])
    assert eff(tl) == "2016-01-31" and not tl.affects_coverage.iloc[-1]


def test_same_day_plan_switch_keeps_new_subscription(clean_tx):
    # listed subscription first: ordering must still apply the old plan's cancel first
    tl, _ = build(
        clean_tx,
        [
            row(20160101, 20160131, price=149, paid=149),
            row(20160115, 20160214, price=129, paid=129),
            row(20160115, 20160115, cancel=1, price=149, paid=149),
        ],
    )
    assert tl.is_cancel.tolist() == [0, 1, 0]
    assert eff(tl) == "2016-02-14"


def test_same_day_same_plan_cancel_does_not_cut(clean_tx):
    tl, _ = build(
        clean_tx,
        [row(20160101, 20160131), row(20160131, 20160302), row(20160131, 20160131, cancel=1)],
    )
    assert tl.sameday_sameplan_cancel.sum() == 1
    assert eff(tl) == "2016-03-02"


def test_placeholder_cancel_never_cuts(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160110, 19700101, cancel=1, pm=24)])
    assert eff(tl) == "2016-01-31"


def test_placeholder_subscription_uses_imputed_expiry(clean_tx):
    tl, _ = build(clean_tx, [row(20160105, 19700101, pm=17, days=60, price=0, paid=0, ar=0)])
    assert eff(tl) == "2016-01-08"  # method 17, 60 plan days: 3 days of coverage


def test_far_future_expiry_kept(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20290101)])
    assert eff(tl) == "2029-01-01"


def test_expiry_before_transaction_does_not_extend(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160210, 20160125)])
    assert tl.spell_id.tolist() == [0, 0] and eff(tl) == "2016-01-31"


def test_v2_rows_change_in_force_expiry(clean_tx):
    v1 = [row(20160101, 20160131)]
    v2 = [row(20160115, 20170301, days=410, price=1788, paid=1788)]  # long plan, only in v2
    alone = in_force_at(build(clean_tx, v1)[0], date(2016, 2, 1), include_same_day=False)
    both = in_force_at(build(clean_tx, v1, v2)[0], date(2016, 2, 1), include_same_day=False)
    assert str(alone.loc["a", "eff_expiry_after"].date()) == "2016-01-31"
    assert str(both.loc["a", "eff_expiry_after"].date()) == "2017-03-01"


def test_row_dated_on_snapshot_is_not_before_it(clean_tx):
    tl, _ = build(clean_tx, [row(20160101, 20160131), row(20160201, 20160302)])
    strict = in_force_at(tl, date(2016, 2, 1), include_same_day=False)
    same_day = in_force_at(tl, date(2016, 2, 1), include_same_day=True)
    assert str(strict.loc["a", "eff_expiry_after"].date()) == "2016-01-31"
    assert str(same_day.loc["a", "eff_expiry_after"].date()) == "2016-03-02"


def test_duplicated_rows_change_nothing(clean_tx):
    tx = clean_tx(
        [row(20160101, 20160131), row(20160110, 20160110, cancel=1), row(20160205, 20160306)]
    )
    dup = pd.concat([tx, tx.assign(source_row=tx.source_row + 1000)], ignore_index=True)
    one, _ = build_timeline(tx, WINDOW)
    two, _ = build_timeline(dup, WINDOW)
    ends = two.groupby(["transaction_date", "is_cancel"]).eff_expiry_after.last()
    assert (
        ends.tolist()
        == one.groupby(["transaction_date", "is_cancel"]).eff_expiry_after.last().tolist()
    )


def random_rows(rng: random.Random) -> tuple[list[str], list[str]]:
    """Messy customers on a few shared dates, so same-day ties are common."""
    days = sorted({date(2016, 1, 1) + timedelta(days=rng.randrange(150)) for _ in range(25)})
    v1, v2 = [], []
    for m in "abcd":
        for d in rng.sample(days, rng.randint(3, 10)):
            td = d.strftime("%Y%m%d")
            price = rng.choice([129, 149])
            kind = rng.random()
            if kind < 0.6:
                exp = (d + timedelta(days=rng.choice([-3, 0, 7, 30, 31, 60]))).strftime("%Y%m%d")
                r = row(td, exp, m=m, price=price, paid=price)
            elif kind < 0.85:
                exp = (d + timedelta(days=rng.choice([-5, 0, 10]))).strftime("%Y%m%d")
                r = row(td, exp, cancel=1, m=m, price=price, paid=price)
            elif kind < 0.92:
                r = row(td, 19700101, cancel=1, m=m, pm=24)
            else:
                r = row(td, 19700101, m=m, pm=17, days=60, price=0, paid=0, ar=0)
            (v2 if rng.random() < 0.3 else v1).append(r)
            if rng.random() < 0.25:  # same-day partner on the same plan
                exp = (d + timedelta(days=30)).strftime("%Y%m%d")
                v1.append(row(td, exp, cancel=rng.randint(0, 1), m=m, price=price, paid=price))
    return v1, v2


COLS = ["msno", "seq", "transaction_date", "is_cancel", "expiry_date", "eff_expiry_after"]
COLS += ["spell_id", "affects_coverage", "starts_spell"]


@pytest.mark.parametrize("renewals", ["coverage", "paid"])
@pytest.mark.parametrize("seed", range(8))
def test_prefix_property(clean_tx, seed, renewals):
    rng = random.Random(seed)
    v1, v2 = random_rows(rng)
    full_tl, full_spells = build_timeline(clean_tx(v1, v2), WINDOW, renewals)
    for _ in range(3):
        cutoff = date(2016, 1, 1) + timedelta(days=rng.randrange(150))
        keep = [r for r in v1 if int(r.split(",")[6]) <= int(cutoff.strftime("%Y%m%d"))]
        keep2 = [r for r in v2 if int(r.split(",")[6]) <= int(cutoff.strftime("%Y%m%d"))]
        if not keep and not keep2:
            continue
        cut_tl, cut_spells = build_timeline(clean_tx(keep, keep2), WINDOW, renewals)
        expected = full_tl[full_tl.transaction_date <= pd.Timestamp(cutoff)][COLS]
        pd.testing.assert_frame_equal(
            cut_tl[COLS].reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
        )
        closed = cut_spells[cut_spells.next_sub_date.notna()].reset_index(drop=True)
        same = full_spells.merge(closed[["msno", "spell_id"]], on=["msno", "spell_id"])
        pd.testing.assert_frame_equal(closed, same, check_dtype=False)


def test_order_rows_is_deterministic(clean_tx):
    tx = clean_tx(random_rows(random.Random(99))[0])
    a = order_rows(tx).row_id.tolist()
    b = order_rows(tx.sample(frac=1, random_state=1)).row_id.tolist()
    assert a == b


def test_late_cancel_after_lapse_then_renewal_with_prefix(clean_tx):
    rows = [
        row(20160101, 20160131),
        row(20160315, 20160110, cancel=1),  # 44 days after the lapse: ignored
        row(20160401, 20160501),  # new spell
    ]
    tl, spells = build(clean_tx, rows)
    assert tl.affects_coverage.tolist() == [True, False, True]
    assert tl.spell_id.tolist() == [0, 0, 1]
    first = spells[spells.spell_id == 0].iloc[0]
    assert str(first.effective_end.date()) == "2016-01-31" and first.gap_days == 61
    for cutoff, n in ((20160314, 1), (20160315, 2), (20160331, 2), (20160401, 3)):
        cut, _ = build(clean_tx, [r for r in rows if int(r.split(",")[6]) <= cutoff])
        pd.testing.assert_frame_equal(cut[COLS], tl[COLS].iloc[:n], check_dtype=False)


def test_outcome_at_classic(clean_tx):
    from retention.labels.timeline import outcome_at

    rows = [
        row(20160101, 20160210, m="renews"),
        row(20160210, 20160310, m="renews"),
        row(20160101, 20160210, m="lapses"),
        row(20160101, 20160210, m="cut"),
        row(20160205, 20160310, m="cut"),
        row(20160215, 20160215, cancel=1, m="cut"),  # renewal undone inside the window
        row(20160101, 20160305, m="late"),
    ]
    tl, spells = build(clean_tx, rows)
    o = outcome_at(tl, spells, date(2016, 2, 1), date(2016, 3, 31), WINDOW)
    assert o.loc["renews", "classic"] == 0
    assert (
        o.loc["lapses", "classic"] == 1
        and str(o.loc["lapses", "lapse_date"].date()) == "2016-02-10"
    )
    assert o.loc["cut", "classic"] == 1 and str(o.loc["cut", "lapse_date"].date()) == "2016-02-15"
    assert pd.isna(o.loc["late", "classic"])  # 2016-03-05 + 30 days is after as_of
    with pytest.raises(ValueError, match="after as_of"):
        outcome_at(tl, spells, date(2016, 2, 1), date(2016, 2, 12), WINDOW)


def test_paid_mode_ignores_zero_paid_rows(clean_tx):
    rows = [row(20151201, 20151231), row(20151231, 20160107, pm=35, days=7, price=0, paid=0, ar=0)]
    cov, _ = build_timeline(clean_tx(rows), WINDOW, "coverage")
    paid, _ = build_timeline(clean_tx(rows), WINDOW, "paid")
    assert eff(cov) == "2016-01-07" and eff(paid) == "2015-12-31"
    assert paid.affects_coverage.tolist() == [True, False]


@pytest.mark.parametrize("renewals", ["coverage", "paid"])
def test_same_day_same_expiry_twins_in_either_file_order(clean_tx, renewals):
    """Twins (same day, same expiry, different price or kind) are ordered by content, not
    file position: swapping them in the input leaves every timeline column (except the
    file position itself) and every spell unchanged, in both renewal modes."""
    twins = [
        # paid row and a 100% discount on the same day with the same expiry
        (
            row("20160315", "20160415", m="a", price=99, paid=99),
            row("20160315", "20160415", m="a", price=149, paid=0),
        ),
        # two paid rows with different prices, after a lapse (one of them starts a spell)
        (
            row("20160601", "20160701", m="a", price=149, paid=149),
            row("20160601", "20160701", m="a", price=129, paid=129, pm=38),
        ),
        # twins that both start a new spell for a returning customer
        (
            row("20160901", "20161001", m="b", price=149, paid=0),
            row("20160901", "20161001", m="b", price=99, paid=99),
        ),
        # a paid twin pair next to a same-day cancel
        (row("20161101", "20161201", m="c"), row("20161101", "20161201", m="c", paid=99)),
    ]
    other = [
        row("20160101", "20160201", m="a"),
        row("20160201", "20160301", m="a"),
        row("20160101", "20160301", m="b"),
        row("20161001", "20161101", m="c"),
        row("20161101", "20161105", m="c", cancel=1),
    ]
    first = other + [r for pair in twins for r in pair]
    second = other + [r for pair in twins for r in reversed(pair)]
    results = []
    for rows in (first, second):
        tl, spells = build_timeline(clean_tx(rows), WINDOW, renewals)
        tl = tl.sort_values(["msno", "seq"]).drop(columns=["row_id", "source", "source_row"])
        results.append((tl.reset_index(drop=True), spells.reset_index(drop=True)))
    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    pd.testing.assert_frame_equal(results[0][1], results[1][1])
