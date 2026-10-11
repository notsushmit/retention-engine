"""Leakage tests: features at T must not change when rows dated on or after T change.

For every snapshot T of the synthetic dataset, the inputs are changed in two
ways and snapshot T is rebuilt: (a) every transaction and log row dated on or
after T is deleted; (b) those rows are replaced with extreme values, and every
customer gets extra rows dated exactly T. The timeline and the labels state are
rebuilt from the changed transactions. The rebuilt snapshot must have the same
keys and the same value in every one of the registry's columns.

Mutation tests wrap the real sources with deliberately leaky versions and check
that these comparisons catch each one, in the mutated column only.
"""

from __future__ import annotations

import itertools
from datetime import date

import duckdb
import pandas as pd
import pytest

from conftest import CFG, write_interim
from feature_fixtures import (
    SNAPSHOTS,
    TWINS,
    label_cfg,
    log_rows,
    members_frame,
    tx_rows,
    write_dataset,
)
from retention.data.clean import clean
from retention.features import build as build_mod
from retention.features import transactions
from retention.features.build import build, read_features
from test_timeline import row

KEY = ["snapshot_date", "msno"]
SENTINEL = -9.223372036854776e15


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("leakage")


_counter = itertools.count()


def clean_rows(root, rows: list[str]) -> pd.DataFrame:
    tmp = root / f"clean{next(_counter)}"
    clean(write_interim(tmp, rows), tmp / "out", CFG)
    return duckdb.sql(f"select * from '{tmp / 'out'}/transactions.parquet'").df()


def td(r: str) -> int:
    return int(r.split(",")[6])


def ymd(t: date) -> int:
    return int(t.strftime("%Y%m%d"))


@pytest.fixture(scope="module")
def raw():
    rows = tx_rows()
    customers = sorted({r.split(",")[0] for r in rows})
    return rows, log_rows(customers), members_frame(customers), customers


def features(root, rows, logs, members, snapshots=None) -> pd.DataFrame:
    tag = next(_counter)
    paths = write_dataset(root / f"set{tag}", clean_rows(root, rows), logs, members)
    build(paths, label_cfg(), snapshots, memory_limit="256MB")
    f = read_features(paths.out)
    return f.sort_values(KEY).reset_index(drop=True)


@pytest.fixture(scope="module")
def full(workdir, raw):
    rows, logs, members, _ = raw
    return features(workdir, rows, logs, members)


def deleted(rows, logs, t):
    return [r for r in rows if td(r) < ymd(t)], logs[logs.date < ymd(t)]


def extreme(rows, logs, customers, t):
    """Rows on or after t get extreme values; every customer gets extra rows dated t."""
    out = []
    for r in rows:
        f = r.split(",")
        if td(r) >= ymd(t):
            f[1] = "13" if f[1] != "13" else "39"  # another payment method
            f[2], f[3], f[4] = "999", "99999", "99999"  # plan days, list price, paid
            f[5] = str(1 - int(f[5]))  # auto-renew flipped
        out.append(",".join(f))
    d = ymd(t)
    for i, m in enumerate(customers):
        out.append(row(d, "20360101", m=m, pm=13, days=999, price=99999, paid=99999, ar=0))
        out.append(row(d, "19700101", m=m, pm=41, days=30, price=149, paid=149))  # placeholder
        if i % 2:
            out.append(row(d, d, m=m, cancel=1))
    logs = logs.copy()
    late = logs.date >= d
    logs.loc[late, ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]] = 10**6
    logs.loc[late, "total_secs"] = [1e12 if k % 2 else SENTINEL for k in range(late.sum())]
    extra = pd.DataFrame(
        {
            "msno": customers,
            "date": d,
            "num_25": 10**6,
            "num_50": 0,
            "num_75": 0,
            "num_985": 0,
            "num_100": 10**6,
            "num_unq": 10**6,
            "total_secs": [1e12 if k % 2 else SENTINEL for k in range(len(customers))],
        }
    )
    return out, pd.concat([logs, extra], ignore_index=True)


def reversed_pre_t(rows, logs, t):
    """Rows dated before t in reverse file order; rows on or after t unchanged."""
    pre = [r for r in rows if td(r) < ymd(t)]
    post = [r for r in rows if td(r) >= ymd(t)]
    early = logs[logs.date < ymd(t)].iloc[::-1]
    return list(reversed(pre)) + post, pd.concat([early, logs[logs.date >= ymd(t)]])


def swapped_twins(rows):
    """The same rows with every same-day twin pair of the fixture in the other order."""
    out = list(rows)
    for a, b in TWINS.values():
        i, j = out.index(a), out.index(b)
        out[i], out[j] = out[j], out[i]
    return out


def differences(a: pd.DataFrame, b: pd.DataFrame) -> set[str]:
    """Columns that differ between two builds of one snapshot; '__keys__' if keys differ."""
    if set(map(tuple, a[KEY].astype(str).to_numpy())) != set(
        map(tuple, b[KEY].astype(str).to_numpy())
    ):
        return {"__keys__"}
    a, b = a.sort_values(KEY).reset_index(drop=True), b.sort_values(KEY).reset_index(drop=True)
    out = set()
    for c in a.columns:
        try:
            pd.testing.assert_series_equal(a[c], b[c], check_names=False, check_exact=True)
        except AssertionError:
            out.add(c)
    return out


def at(f: pd.DataFrame, t: date) -> pd.DataFrame:
    return f[f.snapshot_date == pd.Timestamp(t)].reset_index(drop=True)


@pytest.mark.parametrize("t", SNAPSHOTS)
def test_delete_rows_on_or_after_t(workdir, raw, full, t):
    rows, logs, members, _ = raw
    r, lg = deleted(rows, logs, t)
    assert differences(at(full, t), features(workdir, r, lg, members, [t])) == set()


@pytest.mark.parametrize("t", SNAPSHOTS)
def test_extreme_rows_on_or_after_t(workdir, raw, full, t):
    rows, logs, members, customers = raw
    r, lg = extreme(rows, logs, customers, t)
    assert differences(at(full, t), features(workdir, r, lg, members, [t])) == set()


@pytest.mark.parametrize("t", SNAPSHOTS)
def test_reverse_pre_t_file_order(workdir, raw, full, t):
    rows, logs, members, _ = raw
    r, lg = reversed_pre_t(rows, logs, t)
    assert differences(at(full, t), features(workdir, r, lg, members, [t])) == set()


def test_twin_order_changes_nothing_in_paid_or_coverage_columns(workdir, raw, full):
    rows, logs, members, _ = raw
    other = features(workdir, swapped_twins(rows), logs, members)
    assert differences(full, other) == set()
    cols = [
        "n_prior_lapses_365",
        "days_since_last_payment",
        "paid_state",
        "days_until_paid_expiry",
        *[c for c in full.columns if c.startswith("last_")],
    ]
    for m in TWINS:  # twins on 2016-04-01; the lapse of twins_lapse ends that day
        a = full[full.msno == m].reset_index(drop=True)
        b = other[other.msno == m].reset_index(drop=True)
        assert len(a) > 0
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)
    may = at(full, date(2016, 5, 1)).set_index("msno")
    assert may.loc["twins", "last_amount_paid"] == 99  # the paid twin, by content
    assert may.loc["twins_lapse", "n_prior_lapses_365"] == 1
    assert may.loc["twins_lapse", "last_row_kind"] == "paid"


def test_old_file_order_rule_fails_the_reverse_variant(workdir, raw, monkeypatch):
    rows, logs, members, _ = raw
    monkeypatch.setattr(
        transactions,
        "LAST_ROW_ORDER",
        "d desc, expiry desc nulls last, source desc, source_row desc",
    )
    t = date(2016, 5, 1)
    old = features(workdir, rows, logs, members, [t])
    r, lg = reversed_pre_t(rows, logs, t)
    diff = differences(old, features(workdir, r, lg, members, [t]))
    assert "last_amount_paid" in diff and "last_row_kind" in diff
    assert diff <= {
        "last_list_price",
        "last_amount_paid",
        "last_discount_ratio",
        "last_row_kind",
        "last_payment_method_id",
        "last_is_auto_renew",
        "last_plan_days",
        "last_plan_days_missing",
    }


def test_edge_customers(full):
    may, jun, jul = date(2016, 5, 1), date(2016, 6, 1), date(2016, 7, 1)
    assert "first_on_t" not in set(at(full, may).msno)  # first transaction is on T
    assert "first_on_t" in set(at(full, jun).msno)
    c = at(full, jun).set_index("msno").loc["cancel_after"]
    assert c.n_cancels_365 == 0 and pd.isna(c.days_since_last_cancel)  # cancel is after T
    aug = date(2016, 8, 1)
    assert "on_t" not in set(at(full, jun).msno)  # lapsed; returns exactly on T
    assert at(full, jul).set_index("msno").loc["on_t"].n_prior_lapses_365 == 1
    assert "after_t" not in set(at(full, jul).msno)  # lapsed; returns three days after T
    assert at(full, aug).set_index("msno").loc["after_t"].n_prior_lapses_365 == 1
    q = at(full, jun).set_index("msno").loc["steady0"]
    assert q.has_any_log_before_T == 0 and pd.isna(q.tenure_days)  # never listens: masked


# --- mutation tests: each leaky variant must be caught, in its own column(s) ----------


def leaky_last_txn(monkeypatch):
    original = build_mod.TransactionSource.features

    def features_(self, t, population, **kw):
        out = original(self, t, population, **kw)
        last = (
            self.con.sql(
                f"select msno, max(d) as d from tx_rows where is_cancel = 0 "
                f"and d <= date '{t}' group by msno"
            )
            .df()
            .set_index("msno")
            .d
        )
        days = (pd.Timestamp(t) - pd.to_datetime(out.msno.map(last))).dt.days
        out["days_since_last_txn"] = days.astype("Int32").array
        return out

    monkeypatch.setattr(build_mod.TransactionSource, "features", features_)


def leaky_last7(monkeypatch):
    original = build_mod.UsageSource.features

    def features_(self, t, population, **kw):
        out = original(self, t, population, **kw)
        on_t = (
            self.con.sql(
                f"select i.msno, count(*) as n, sum(d.secs_ms) / 1000.0 as s from usage_daily d "
                f"join usage_ids i using (cid) where d.day = date '{t}' group by i.msno"
            )
            .df()
            .set_index("msno")
        )
        n = out.msno.map(on_t.n).fillna(0).astype("int32")
        s = out.msno.map(on_t.s).fillna(0.0)
        out["active_days_last7"] = out.active_days_last7 + n
        out["secs_per_day_last7"] = out.secs_per_day_last7 + s / 7
        return out

    monkeypatch.setattr(build_mod.UsageSource, "features", features_)


def global_cohort(monkeypatch, populations: dict[date, list[str]]):
    """Cohort median over every snapshot of the build at once, instead of per T."""
    original = build_mod.UsageSource.features
    cache: dict[int, float] = {}

    def features_(self, t, population, **kw):
        if id(self) not in cache:
            ratios = pd.concat(
                [original(self, s, populations[s], **kw).ratio_secs_7_vs_base for s in populations]
            )
            cache[id(self)] = float(ratios.median())
        out = original(self, t, population, **kw)
        out["ratio_secs_vs_cohort"] = out.ratio_secs_7_vs_base / cache[id(self)]
        return out

    monkeypatch.setattr(build_mod.UsageSource, "features", features_)


def test_mutation_days_since_last_txn_reading_day_t_is_caught(workdir, raw, full, monkeypatch):
    rows, logs, members, customers = raw
    t = SNAPSHOTS[3]
    r, lg = extreme(rows, logs, customers, t)
    leaky_last_txn(monkeypatch)
    # Guard 1: the build's cross-check refuses a last row dated on or after T.
    with pytest.raises(build_mod.BuildError, match="last non-cancel row dated on or after T"):
        features(workdir, r, lg, members, [t])
    # Guard 2: without that check, the leakage comparison alone catches it.
    monkeypatch.setattr(build_mod, "cross_check", lambda df: [])
    assert differences(at(full, t), features(workdir, r, lg, members, [t])) == {
        "days_since_last_txn"
    }


def test_mutation_last7_window_including_day_t_is_caught(workdir, raw, full, monkeypatch):
    rows, logs, members, customers = raw
    t = SNAPSHOTS[3]
    r, lg = extreme(rows, logs, customers, t)
    leaky_last7(monkeypatch)
    diff = differences(at(full, t), features(workdir, r, lg, members, [t]))
    assert "active_days_last7" in diff
    assert diff <= {"active_days_last7", "secs_per_day_last7"}


def test_mutation_global_cohort_median_fails_single_snapshot_test(workdir, raw, monkeypatch):
    rows, logs, members, _ = raw
    lab_paths = write_dataset(workdir / "pops", clean_rows(workdir, rows), logs, members)
    lab = pd.read_parquet(lab_paths.processed / "labels.parquet")
    pops = {t: lab[lab.snapshot_date == t].msno.tolist() for t in SNAPSHOTS}
    t = SNAPSHOTS[2]
    global_cohort(monkeypatch, pops)
    whole = features(workdir, rows, logs, members)
    global_cohort(monkeypatch, {t: pops[t]})
    alone = features(workdir, rows, logs, members, [t])
    assert differences(at(whole, t), alone) == {"ratio_secs_vs_cohort"}
