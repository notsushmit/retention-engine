"""A small synthetic dataset on disk for feature build and leakage tests.

Customers cover the edge cases: steady renewals, a lapse and return, grace,
trial-only, 100% discount, cancels, a return exactly on a snapshot date and
just after one, a newcomer listener, a customer with no logs, sentinel and
over-24-hour seconds, and members rows that are blank, out of range or missing.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from conftest import CFG
from retention.data.quality import snapshot_status
from retention.features.build import FeaturePaths
from retention.labels.snapshots import build_labels
from retention.labels.timeline import build_timeline
from test_timeline import row

SNAPSHOTS = [date(2016, m, 1) for m in range(3, 9)]
AS_OF = date(2016, 12, 31)
LOG_COLS = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]


TWINS = {
    "twins": [
        row("20160401", "20160501", m="twins", price=99, paid=99),
        row("20160401", "20160501", m="twins", price=149, paid=0),
    ],
    "twins_lapse": [  # 60 days after the last expiry: a lapse ends on the twins' day
        row("20160401", "20160515", m="twins_lapse", price=149, paid=0, pm=40),
        row("20160401", "20160515", m="twins_lapse", price=129, paid=129, pm=40),
    ],
}


def ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def monthly(m: str, start: date, end: date, **kw) -> list[str]:
    out, d = [], start
    while d <= end:
        out.append(row(ymd(d), ymd(d + timedelta(days=30)), m=m, **kw))
        d += timedelta(days=30)
    return out


def tx_rows(seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []
    for i in range(12):
        rows += monthly(f"steady{i}", date(2015, 10, 1) + timedelta(days=i), date(2016, 11, 1))
    rows += monthly("lapser", date(2015, 11, 1), date(2016, 2, 1))
    rows += monthly("lapser", date(2016, 5, 10), date(2016, 11, 1))
    rows += [row("20160401", "20160701", m="grace", days=90, price=447, paid=447)]
    rows += [row("20160705", "20160805", m="grace")]
    rows += [
        row(
            ymd(date(2016, 2, 1) + timedelta(days=7 * k)),
            ymd(date(2016, 2, 8) + timedelta(days=7 * k)),
            m="trial",
            pm=35,
            days=7,
            price=0,
            paid=0,
            ar=0,
        )
        for k in range(20)
    ]
    rows += monthly("discount", date(2016, 1, 15), date(2016, 9, 1), price=149, paid=0)
    rows += monthly("canceller", date(2015, 12, 1), date(2016, 7, 1))
    rows += [row("20160415", "20160501", m="canceller", cancel=1)]
    # Lapses and returns exactly on a snapshot date, and three days after another.
    rows += monthly("on_t", date(2016, 1, 1), date(2016, 3, 1))
    rows += monthly("on_t", date(2016, 6, 1), date(2016, 10, 1))
    rows += monthly("after_t", date(2016, 1, 1), date(2016, 4, 1))
    rows += monthly("after_t", date(2016, 7, 4), date(2016, 10, 1))
    # Same-day, same-expiry twins (a paid row and a 100% discount), ordered only by
    # content: one in the middle of a run, one on the day a lapse ends.
    rows += monthly("twins", date(2016, 1, 1), date(2016, 3, 1))
    rows += TWINS["twins"]
    rows += monthly("twins", date(2016, 5, 1), date(2016, 10, 1))
    rows += [row("20160101", "20160201", m="twins_lapse")]
    rows += TWINS["twins_lapse"]
    rows += monthly("twins_lapse", date(2016, 5, 15), date(2016, 10, 1))
    # First-ever transaction exactly on a snapshot date: not in that snapshot.
    rows += monthly("first_on_t", date(2016, 5, 1), date(2016, 10, 1))
    # The only row after 2016-06-01 is a cancel.
    rows += monthly("cancel_after", date(2016, 3, 1), date(2016, 5, 30))
    rows += [row("20160610", "20160615", m="cancel_after", cancel=1)]
    rows += [row("20160301", "20160301", m="blank", days=0, price=0, paid=149)]
    rows += monthly("blank", date(2016, 3, 2), date(2016, 10, 1), days=0, price=0, paid=149)
    for i in range(10):
        d = date(2015, 9, 1) + timedelta(days=rng.randrange(120))
        while d < date(2016, 11, 1):
            plan = rng.choice([30, 30, 30, 90])
            paid = rng.choice([149, 149, 99, 0])
            rows.append(
                row(
                    ymd(d),
                    ymd(d + timedelta(days=plan)),
                    m=f"rand{i}",
                    days=plan,
                    paid=paid,
                    ar=rng.randint(0, 1),
                    pm=rng.choice([38, 39, 40, 41]),
                )
            )
            if rng.random() < 0.15:
                rows.append(
                    row(
                        ymd(d + timedelta(days=5)),
                        ymd(d + timedelta(days=10)),
                        m=f"rand{i}",
                        cancel=1,
                    )
                )
            d += timedelta(days=plan + rng.choice([0, 0, 2, 20, 45]))
    return rows


def log_rows(customers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    out = []
    for m in customers:
        if m == "steady0":
            continue  # never listens
        first = date(2016, 7, 20) if m == "steady1" else date(2015, 11, 1)  # steady1: newcomer
        p = rng.uniform(0.1, 0.9)
        d = first
        while d <= date(2016, 9, 30):
            if rng.random() < p:
                secs = rng.expovariate(1 / 3000)
                if rng.random() < 0.01:
                    secs = rng.choice([-9.223372036854776e15, 9.223372036854776e15, -4.0, 1.2e5])
                plays = rng.randint(1, 60)
                full = rng.randint(0, plays)
                out.append(
                    (m, int(ymd(d)), plays - full, 0, 0, 0, full, rng.randint(1, plays), secs)
                )
            d += timedelta(days=1)
    return pd.DataFrame(out, columns=[*LOG_COLS, "total_secs"])


def members_frame(customers: list[str]) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(customers):
        if m in ("rand0", "rand1"):
            continue  # no members row
        bd = [0, 25, 101, 34, 45, 18][i % 6]
        gender = [None, "male", "female"][i % 3]
        city = 1 if bd == 0 else 5 + i % 10
        reg = date(2012, 1, 1) + timedelta(days=97 * i)
        if m == "steady2":
            reg = date(2016, 6, 15)  # registered after some snapshots
        rows.append((m, city, bd, gender, [7, 9, 3][i % 3], pd.Timestamp(reg)))
    cols = ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]
    return pd.DataFrame(rows, columns=cols)


def month_quality() -> pd.DataFrame:
    months = [str(p) for p in pd.period_range("2015-01", "2017-03", freq="M")]
    mq = pd.DataFrame(
        {
            "month": months,
            "assessed": "batch,churn,listening,returns",
            "fails": "",
            "fails_low_k": "",
            "fails_high_k": "",
            "status": "reliable",
        }
    )
    mq.loc[mq.month == "2016-05", ["fails", "fails_low_k", "fails_high_k", "status"]] = [
        "batch",
        "batch",
        "batch",
        "uncertain",
    ]
    mq.loc[mq.month == "2016-02", ["fails", "fails_low_k", "fails_high_k", "status"]] = [
        "returns",
        "returns",
        "returns",
        "uncertain",
    ]
    return mq


def label_cfg():
    return replace(CFG, snapshots=tuple(SNAPSHOTS), as_of=AS_OF, allow_partial=True).validate()


def write_dataset(
    root: Path, tx: pd.DataFrame, logs: pd.DataFrame, members: pd.DataFrame
) -> FeaturePaths:
    """Write the processed tables and logs the build reads; labels from these transactions."""
    processed = root / "processed"
    interim = root / "interim"
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    cfg = label_cfg()
    tl, _ = build_timeline(tx, cfg.renewal_window_days)
    tl.to_parquet(processed / "timeline.parquet", index=False)
    mq = month_quality()
    mq.to_parquet(processed / "month_quality.parquet", index=False)
    rel = snapshot_status(mq.set_index("month").status, cfg)
    build_labels(tx, cfg, rel).to_parquet(processed / "labels.parquet", index=False)
    members.to_parquet(processed / "members.parquet", index=False)
    logs.to_parquet(interim / "user_logs.parquet", index=False)
    return FeaturePaths(processed=processed, logs=(interim / "user_logs.parquet",))


def dataset(root: Path, clean_tx, seed: int = 0) -> FeaturePaths:
    tx = clean_tx(tx_rows(seed))
    customers = sorted(tx.msno.unique())
    return write_dataset(root, tx, log_rows(customers, seed), members_frame(customers))
