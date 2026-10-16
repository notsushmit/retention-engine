"""The KKBox outcome reader: what it reads, the placebo-first order, and read-once."""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.models.data import META_KEY, OUTCOME_LABEL_PREFIXES, column_kind
from retention.models.protocol import load_protocol
from retention.uplift import outcome

AUG, SEP = date(2016, 8, 1), date(2016, 9, 1)


def units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["a", "b", "c", "d"],
            "snapshot_date": pd.to_datetime([SEP, SEP, SEP, AUG]),
            "paid_expiry_date": pd.to_datetime(["2016-09-30"] * 3 + ["2016-08-31"]),
            "expiry_date": pd.to_datetime(["2016-10-31"] * 3 + ["2016-09-30"]),
            "treated": [1, 0, 1, 0],
        }
    )


def labels_file(tmp: Path) -> Path:
    """A rebuilt labels file with extra outcome columns and a 2017 snapshot, never to be read."""
    rows = [
        # msno, snapshot, paid_expiry, observed, days (lapse = T + days)
        ("a", SEP, "2016-09-30", 1, 60),  # lapse 2016-10-31 = E: churn
        ("b", SEP, "2016-09-30", 1, 61),  # lapse after E: retained at E
        ("c", SEP, "2016-09-30", 0, 90),  # censored: retained at E
        ("d", AUG, "2016-08-31", 1, 30),  # lapse 2016-08-31, before E: churn
        ("a", date(2017, 1, 1), "2017-01-31", 1, 5),
    ]
    table = pa.table(
        {
            "msno": [r[0] for r in rows],
            "snapshot_date": pa.array([r[1] for r in rows], pa.date32()),
            "paid_state": ["active"] * len(rows),
            "paid_expiry_date": pa.array([pd.Timestamp(r[2]) for r in rows], pa.timestamp("ns")),
            "paid_event_observed": pa.array([r[3] for r in rows], pa.int8()),
            "paid_days_to_event": pa.array([r[4] for r in rows], pa.int32()),
            "paid_churned_30": pa.array([1] * len(rows), pa.int8()),
            "churned_30": pa.array([1] * len(rows), pa.int8()),
        }
    )
    meta = {"as_of": "2016-12-31", "snapshots": ["2016-08-01", "2016-09-01"]}
    table = table.replace_schema_metadata({META_KEY: json.dumps(meta).encode()})
    path = tmp / "labels_asof.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def spies(tmp_path, monkeypatch):
    calls = {"as_of": [], "load": []}
    path = labels_file(tmp_path)
    real_load = outcome.load_labels

    def fake_as_of(processed, cache_dir, as_of, snapshots, protocol, base=None):
        calls["as_of"].append((as_of, tuple(snapshots)))
        return path

    def spy_load(p, snapshots, columns, protocol, unlock=None):
        calls["load"].append((list(snapshots), list(columns)))
        return real_load(p, snapshots, columns, protocol, unlock)

    monkeypatch.setattr(outcome, "labels_as_of", fake_as_of)
    monkeypatch.setattr(outcome, "load_labels", spy_load)
    return calls


def placebo_log(tmp: Path, u: pd.DataFrame, commit: str = "abc1234") -> Path:
    return outcome.write_placebo_log(tmp / "placebo.json", {"label": "balance not met"}, commit, u)


def read(tmp: Path, u: pd.DataFrame, log: Path, started, commit="abc1234", reason=None):
    return outcome.read_outcome(
        u,
        out_dir=tmp / "out",
        placebo_log=log,
        commit=commit,
        run_started=started,
        protocol=load_protocol(),
        rerun_reason=reason,
    )


def test_reads_only_the_two_outcome_columns_of_2016_snapshots_as_of_2016_12_31(tmp_path, spies):
    u = units()
    started = outcome.now() - timedelta(seconds=1)
    y = read(tmp_path, u, placebo_log(tmp_path, u), started)
    assert y.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert spies["as_of"] == [(date(2016, 12, 31), (AUG, SEP))]
    ((snaps, cols),) = spies["load"]
    assert snaps == [AUG, SEP]
    outcome_cols = [c for c in cols if column_kind(c) == "outcome"]
    assert sorted(outcome_cols) == ["paid_days_to_event", "paid_event_observed"]
    assert set(cols) - set(outcome_cols) <= {
        "msno",
        "snapshot_date",
        "paid_state",
        "paid_expiry_date",
    }
    assert all(column_kind(c) == "pre_t" for c in set(cols) - set(outcome_cols))


def test_churn_at_e_hand_cases():
    t = pd.Series(pd.to_datetime(["2016-09-01"] * 4))
    e = pd.Series(pd.to_datetime(["2016-10-31"] * 4))
    observed = pd.Series([1, 1, 1, 0])
    days = pd.Series([59, 60, 61, 90])  # lapse before E, on E, after E, censored
    assert outcome.churn_at_e(t, e, observed, days).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_untreated_units_use_their_own_full_price_renewal_expiry(tmp_path, spies):
    # unit b is untreated; its E (2016-10-31) is the expiry on its own renewal, and a lapse
    # one day later is not churn at E
    u = units()
    y = read(tmp_path, u, placebo_log(tmp_path, u), outcome.now() - timedelta(seconds=1))
    assert u.loc[1, "treated"] == 0 and y[1] == 0.0


def test_placebo_first_refusals(tmp_path, spies):
    u = units()
    started = outcome.now() - timedelta(seconds=1)
    with pytest.raises(outcome.PlaceboFirst, match="no placebo log"):
        read(tmp_path, u, tmp_path / "missing.json", started)
    stale = placebo_log(tmp_path, u, commit="0000000")
    with pytest.raises(outcome.PlaceboFirst, match="from commit"):
        read(tmp_path, u, stale, started)
    other = placebo_log(tmp_path, u.iloc[:3])
    with pytest.raises(outcome.PlaceboFirst, match="different set of units"):
        read(tmp_path, u, other, started)
    old = placebo_log(tmp_path, u)
    with pytest.raises(outcome.PlaceboFirst, match="before this run started"):
        read(tmp_path, u, old, outcome.now() + timedelta(seconds=5))
    assert spies["load"] == [] and spies["as_of"] == []  # nothing was read
    assert not (tmp_path / "out" / outcome.MARKER).exists()


def test_the_command_order_is_placebo_then_log_then_outcome():
    order = []
    result, log, y = outcome.placebo_then_outcome(
        lambda: order.append("placebo") or "p",
        lambda r: order.append(f"log {r}") or Path("placebo.json"),
        lambda path: order.append(f"read after {path}") or np.zeros(1),
    )
    assert order == ["placebo", "log p", "read after placebo.json"]


def test_read_once_marker_and_repeat_with_a_reason(tmp_path, spies):
    u = units()
    out = tmp_path / "out"
    started = outcome.now() - timedelta(seconds=1)
    read(tmp_path, u, placebo_log(tmp_path, u), started)
    marker = json.loads((out / outcome.MARKER).read_text())
    assert marker["first_attempt"]["commit"] == "abc1234" and marker["repeats"] == []
    # the run crashed before writing its log: a repeat needs a stated reason
    with pytest.raises(outcome.ReadOnce, match="stated reason"):
        read(tmp_path, u, placebo_log(tmp_path, u), started)
    read(tmp_path, u, placebo_log(tmp_path, u), started, reason="crashed while writing tables")
    marker = json.loads((out / outcome.MARKER).read_text())
    assert marker["repeats"][0]["reason"] == "crashed while writing tables"
    # once the outcome log exists, every further read refuses, with or without a reason
    (out / outcome.LOG).write_text("{}")
    for reason in (None, "any reason"):
        with pytest.raises(outcome.ReadOnce, match="read once"):
            read(tmp_path, u, placebo_log(tmp_path, u), started, reason=reason)


def test_the_marker_is_written_before_any_label_read(tmp_path, monkeypatch):
    u = units()
    seen = []

    def fake_as_of(*args, **kwargs):
        seen.append((tmp_path / "out" / outcome.MARKER).exists())
        raise RuntimeError("stop here")

    monkeypatch.setattr(outcome, "labels_as_of", fake_as_of)
    with pytest.raises(RuntimeError, match="stop here"):
        read(tmp_path, u, placebo_log(tmp_path, u), outcome.now() - timedelta(seconds=1))
    assert seen == [True]


def test_rebuilt_pre_t_state_must_match(tmp_path, spies):
    u = units()
    u.loc[0, "paid_expiry_date"] = pd.Timestamp("2016-09-29")
    with pytest.raises(outcome.OutcomeMismatch, match="pre-T state"):
        read(tmp_path, u, placebo_log(tmp_path, u), outcome.now() - timedelta(seconds=1))


def test_only_outcome_py_names_outcome_columns_and_only_the_two():
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in OUTCOME_LABEL_PREFIXES) + r")\w*"
    )
    for path in Path("src/retention/uplift").glob("*.py"):
        names = set(pattern.findall(path.read_text()))
        if path.name == "outcome.py":
            assert names == set(outcome.OUTCOME_COLUMNS), names
        else:
            assert not names, f"{path} names {names}"
