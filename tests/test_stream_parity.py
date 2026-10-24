"""stream-parity on hand-made data: the marks, the four diagnostics and the command's refusals.

Every table comes from test_stream_move's hand-made raw files (two seeds): batch through
Phase 1 and 2's own functions, the stream through the consumer and the move. Planted
differences are written into one table by hand; nothing reads the sample's data.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from id_search import assert_no_ids
from retention.api import __main__ as api_cli
from retention.data.clean import _build_transactions, _load_transactions
from retention.explain.contributions import SavedModel
from retention.explain.model import Settings, booster_sha256, fit_models, load_settings
from retention.labels.timeline import build_timeline
from retention.models.protocol import load_protocol
from retention.stream import __main__ as cli
from retention.stream import move, parity
from test_stream_move import CFG, T, batch_table, no_test_unlock, raw, replayed  # noqa: F401

P = load_protocol()
REAL = load_settings(horizons=(7, 30, 59))
COLUMNS = list(REAL.columns)
MAIN = tuple(P.main_segments)
pytestmark = pytest.mark.valkey


def tiny_models(table: pa.Table) -> dict:
    """Small classifiers fitted on the table itself (scores mean nothing; parity is the point)."""
    settings = Settings(
        columns=COLUMNS,
        categorical=list(REAL.categorical),
        fixed=yaml.safe_load(Path("configs/evaluation.yaml").read_text())["classifier"][
            "fixed_params"
        ],
        candidates={
            h: {"num_leaves": 3, "n_estimators": 5, "learning_rate": 0.1, "min_child_samples": 5}
            for h in (7, 30, 59)
        },
    )
    feats = table.to_pandas(types_mapper=move.NULLABLE.get, date_as_object=False)
    rng = np.random.default_rng(1)
    out = {}
    for h in (7, 30, 59):
        fit = feats.assign(y=rng.integers(0, 2, len(feats)))
        model, platt = fit_models(fit, fit, settings, h, P)
        out[h] = SavedModel(h, model, platt.a, platt.b, booster_sha256(model))
    return out


def batch_timeline(folder: Path) -> pd.DataFrame:
    con = duckdb.connect()
    _load_transactions(con, folder / "transactions.parquet", folder / "transactions_v2.parquet")
    _build_transactions(con, CFG)
    tx = con.sql("select * from tx order by msno, transaction_date, source, source_row").df()
    return build_timeline(tx, CFG.renewal_window_days)[0]


def planted(table: pa.Table, msno: str, column: str, value) -> pa.Table:
    df = table.to_pandas(types_mapper=move.NULLABLE.get)
    df.loc[df.msno == msno, column] = value
    return pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)


@pytest.fixture()
def case(replayed):  # noqa: F811
    r, _, members_path, folder = replayed
    tl = move.rebuild(r, CFG, None)
    stream, _ = move.feature_table(r, T, tl, CFG, members_path)
    batch = batch_table(folder, members_path)
    full = batch_timeline(folder)
    return SimpleNamespace(
        r=r,
        stream=stream,
        batch=batch,
        folder=folder,
        stream_timeline=lambda ids: move.Timelines.build(move.stored_rows(r, ids), CFG).coverage,
        batch_timeline=lambda ids: full[full.msno.isin(ids)],
        c01=tiny_models(stream),
    )


def judge(case, stream=None, batch=None) -> dict:
    stream = case.stream if stream is None else stream
    batch = case.batch if batch is None else batch
    p2 = len(parity.p2_rows(case.batch, CFG.renewal_window_days, MAIN))
    return parity.run(
        stream,
        batch,
        case.stream_timeline,
        case.batch_timeline,
        case.folder / "user_logs.parquet",
        T,
        case.c01,
        P.seed,
        COLUMNS,
        CFG.renewal_window_days,
        MAIN,
        case.batch.num_rows,
        p2,
    )


# --- the marks -----------------------------------------------------------------------------------


def test_equal_tables_meet_feature_and_score_parity(case):
    report = judge(case)
    assert (
        report["registered_68_column_parity"] == "not met as written (mis-specified before any run)"
    )
    assert report["a9_columns_judged"] == 61 and len(report["a9_columns_excluded"]) == 7
    assert report["feature_parity"]["met"] and report["diagnostics"] == {"mismatching_customers": 0}
    s = report["score_parity"]
    assert s["met"] and s["rows_compared"] == s["rows_with_identical_features"] > 50
    assert s["raw_scores_identical_where_features_identical"] == s["rows_compared"]
    assert s["largest_final_probability_gap"] == 0.0


def test_a_registered_row_count_that_differs_is_not_met(case):
    p2 = len(parity.p2_rows(case.batch, CFG.renewal_window_days, MAIN))
    report = parity.run(
        case.stream, case.batch, case.stream_timeline, case.batch_timeline,
        case.folder / "user_logs.parquet", T, case.c01, P.seed, COLUMNS,
        CFG.renewal_window_days, MAIN, case.batch.num_rows + 1, p2 + 1,
    )  # fmt: skip
    assert not report["feature_parity"]["met"] and not report["score_parity"]["met"]


# --- the diagnostics, on planted differences ------------------------------------------------------


def test_a_mismatch_on_same_day_ties_is_an_ordering_cause(case):
    """s-tie has two same-day rows tied on every timeline key (test_stream_move)."""
    batch = planted(case.batch, "s-tie", "last_payment_method_id", 36)
    d = judge(case, batch=batch)["diagnostics"]
    assert d["mismatching_customers"] == 1 and d["by_column"] == {"last_payment_method_id": 1}
    assert d["with_same_day_ties"] == 1 and d["inputs_before_t_match"] == 1
    assert d["classes"] == {"ordering_cause": 1, "phase2_finding": 0, "stream_difference": 0}


def test_a_labels_state_difference_with_matching_inputs_is_a_phase_2_finding(case):
    m = next(x for x in case.batch.column("msno").to_pylist() if not x.startswith("s-"))
    batch = planted(case.batch, m, "segment", "discount100_only")
    d = judge(case, batch=batch)["diagnostics"]
    assert d["by_column"] == {"segment": 1} and d["inputs_before_t_match"] == 1
    assert d["with_same_day_ties"] == 0
    assert d["classes"] == {"ordering_cause": 0, "phase2_finding": 1, "stream_difference": 0}


def test_a_usage_difference_is_traced_to_the_side_that_differs_from_a_recount(case):
    df = case.batch.to_pandas()
    m = df.msno[(df.active_days_28 > 0) & ~df.msno.str.startswith("s-")].iloc[0]
    batch = planted(case.batch, m, "secs_per_day_28", 123.456)
    d = judge(case, batch=batch)["diagnostics"]
    assert d["usage_mismatching"] == 1
    assert (d["usage_stream_equals_recount"], d["usage_batch_equals_recount"]) == (1, 0)
    assert d["classes"] == {"ordering_cause": 0, "phase2_finding": 0, "stream_difference": 1}


def test_a_customer_missing_from_the_stream_is_a_population_difference(case):
    m = case.stream.column("msno")[3].as_py()
    keep = pa.compute.not_equal(case.stream.column("msno"), m)
    report = judge(case, stream=case.stream.filter(keep))
    f, d = report["feature_parity"], report["diagnostics"]
    assert (
        not f["met"]
        and f["rows_only_in_batch"] == 1
        and f["labels_state_differences"] == ["population"]
    )
    assert d["by_column"] == {"population": 1}


def test_a_model_input_difference_breaks_score_parity_only_where_features_differ(case):
    p2 = parity.p2_rows(case.stream, CFG.renewal_window_days, MAIN)
    m = p2.msno.iloc[0]
    stream = planted(case.stream, m, "plays_28", int(p2.plays_28.iloc[0]) + 1000)
    s = judge(case, stream=stream)["score_parity"]
    assert not s["met"]
    assert s["rows_with_identical_features"] == s["rows_compared"] - 1
    assert s["raw_scores_identical_where_features_identical"] == s["rows_compared"] - 1


def test_the_report_holds_no_customer_id(case):
    batch = planted(case.batch, "s-tie", "last_payment_method_id", 36)
    text = json.dumps(judge(case, batch=batch), default=str)
    ids = case.batch.column("msno").to_pylist()
    assert_no_ids(text, ids, "the parity report")


# --- the command ---------------------------------------------------------------------------------


def args(db=1):
    return SimpleNamespace(fraction=None, db=db)


@pytest.mark.parametrize(
    "db, message", [(0, "database 0 holds the loaded store"), (2, "replay database")]
)
def test_the_command_refuses_database_0_and_any_but_the_replay_database(db, message):
    with pytest.raises(SystemExit, match=message):
        cli.stream_parity_cmd(args(db), gather=lambda *a: pytest.fail("gathered"))


def test_the_command_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: True)
    with pytest.raises(SystemExit, match="commit first"):
        cli.stream_parity_cmd(args(), gather=lambda *a: pytest.fail("gathered"))


def test_the_command_writes_only_its_report_once_with_no_id(case, monkeypatch, tmp_path):
    monkeypatch.setattr(api_cli, "tree_dirty", lambda repo: False)
    monkeypatch.setattr(api_cli, "head_commit", lambda repo: "abc1234")
    monkeypatch.setattr(api_cli, "out_dir", lambda fraction: tmp_path)
    monkeypatch.setattr(
        parity,
        "SCORE",
        parity.SCORE | {"rows": len(parity.p2_rows(case.batch, CFG.renewal_window_days, MAIN))},
    )

    def gather(a, t):
        return {
            "stream": case.stream,
            "batch": case.batch,
            "stream_timeline": case.stream_timeline,
            "batch_timeline": case.batch_timeline,
            "logs": case.folder / "user_logs.parquet",
            "c01": case.c01,
            "models_version": 1,
            "seed": P.seed,
            "columns": COLUMNS,
            "window": CFG.renewal_window_days,
            "main": MAIN,
        }

    code = cli.stream_parity_cmd(args(), gather=gather)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stream_parity_2016-11-01.json"]
    text = (tmp_path / "stream_parity_2016-11-01.json").read_text()
    assert_no_ids(text, case.batch.column("msno").to_pylist(), "the parity report file")
    assert json.loads(text)["registered_68_column_parity"].startswith("not met as written")
    assert code == 1  # the registered 230,286 rows are not the hand-made table's
    with pytest.raises(SystemExit, match="runs once"):
        cli.stream_parity_cmd(args(), gather=gather)


def test_the_stream_parity_date_is_the_replay_s_store_date():
    assert T == date(2016, 11, 1) == cli.source.STORE_DATE
