"""The synthetic dataset and its small-data path (configs/ops.yaml, R3).

Every check here runs on synthetic data only; none reads a KKBox file.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retention.api.docs import wording_violations
from retention.synthetic import SYNTHETIC_SENTENCE
from retention.synthetic import generate as gen
from retention.synthetic import pipeline as pl

GENERATE_SOURCE = Path("src/retention/synthetic/generate.py")
ALLOWED_IMPORTS = {
    "__future__",
    "json",
    "datetime",
    "pathlib",
    "typing",
    "numpy",
    "pandas",
    "yaml",
    "retention.synthetic",
}


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic_generated")
    gen.generate(out)
    return out


def table_hashes(path: Path) -> dict[str, str]:
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.glob("*.parquet"))
    }


def test_the_settings_are_the_registered_ones():
    c = yaml.safe_load(Path("configs/ops.yaml").read_text())["synthetic"]
    s = gen.load_settings()
    assert s == c
    assert s["seed"] == 20261024
    assert s["size_cap"] == {
        "customers": 3000,
        "log_rows": 200000,
        "total_files_bytes": 100000000,
    }


def test_the_same_seed_gives_the_same_tables(generated, tmp_path):
    gen.generate(tmp_path)
    assert table_hashes(tmp_path) == table_hashes(generated)


def test_another_seed_gives_other_tables(generated, tmp_path):
    other = dict(gen.load_settings(), seed=1)
    gen.generate(tmp_path, other)
    assert table_hashes(tmp_path) != table_hashes(generated)


def test_the_size_caps_hold(generated):
    report = json.loads((generated / "synthetic_generate_report.json").read_text())
    caps = gen.load_settings()["size_cap"]
    assert report["customers"] == caps["customers"]
    assert report["rows"]["synthetic_user_logs"] <= caps["log_rows"]
    assert report["total_bytes"] <= caps["total_files_bytes"]
    assert sum(f.stat().st_size for f in generated.glob("*.parquet")) == report["total_bytes"]


def test_a_cap_that_would_be_exceeded_refuses(tmp_path):
    s = gen.load_settings()
    for key, value in (("log_rows", 10), ("total_files_bytes", 1000)):
        capped = dict(s, size_cap=dict(s["size_cap"], **{key: value}))
        with pytest.raises(ValueError, match="registered cap"):
            gen.generate(tmp_path / key, capped)


def test_every_generated_file_is_labelled_synthetic(generated):
    names = [f.name for f in generated.iterdir()]
    assert names and all(n.startswith("synthetic_") or n == "SYNTHETIC" for n in names)
    assert (generated / "SYNTHETIC").read_text().strip() == SYNTHETIC_SENTENCE
    report = json.loads((generated / "synthetic_generate_report.json").read_text())
    assert report["synthetic"] is True and report["label"] == SYNTHETIC_SENTENCE
    ids = pd.read_parquet(generated / "synthetic_members.parquet").msno
    assert ids.str.startswith(gen.IDS_PREFIX).all() and ids.is_unique


def test_the_tables_have_the_columns_the_pipeline_reads(generated):
    tx = pd.read_parquet(generated / "synthetic_transactions.parquet")
    assert list(tx.columns) == [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ]
    assert tx.transaction_date.max() <= 20161231  # the recording stops on the data end
    logs = pd.read_parquet(generated / "synthetic_user_logs.parquet")
    assert list(logs.columns) == [
        "msno",
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ]
    assert logs.date.max() <= 20161231
    members = pd.read_parquet(generated / "synthetic_members.parquet")
    assert list(members.columns) == [
        "msno",
        "city",
        "bd",
        "gender",
        "registered_via",
        "registration_init_time",
    ]


def imported_modules(source: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
    return out


def test_the_generator_imports_no_pipeline_reader():
    mods = imported_modules(GENERATE_SOURCE)
    assert mods <= ALLOWED_IMPORTS, sorted(mods - ALLOWED_IMPORTS)
    assert not [m for m in mods if m.startswith(("retention.data", "duckdb", "pyarrow"))]


def test_the_generator_opens_no_kkbox_path(monkeypatch, tmp_path):
    opened: list[str] = []
    real_open = builtins.open
    real_path_open = Path.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_path_open(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)
    gen.generate(tmp_path)
    forbidden = ("data/raw", "data/interim", "data/processed", "kkbox")
    assert not [p for p in opened if any(f in p.lower() for f in forbidden)], opened
    assert any(p.endswith("configs/ops.yaml") for p in opened)  # the registered settings


def test_the_label_sentence_passes_the_wording_rules():
    assert wording_violations(SYNTHETIC_SENTENCE) == []


def test_the_synthetic_paths_check_refuses_other_folders(tmp_path):
    pl.check_synthetic_paths(pl.ROOT / "interim", root=pl.ROOT)
    for bad in (Path("data/interim/kkbox"), Path("data/processed/kkbox"), Path("/tmp")):
        with pytest.raises(ValueError, match="outside"):
            pl.check_synthetic_paths(bad, root=pl.ROOT)


def test_the_default_profile_reads_and_writes_only_the_synthetic_folder():
    s = pl.Setup()
    assert s.interim.resolve().is_relative_to(pl.ROOT.resolve())
    assert s.processed.resolve().is_relative_to(pl.ROOT.resolve())


def test_a_profile_that_is_not_marked_synthetic_is_refused(tmp_path):
    raw = yaml.safe_load(pl.CONFIG.read_text())
    raw["synthetic"] = False
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not a synthetic profile"):
        pl.Setup(path)


def test_a_profile_pointing_at_the_kkbox_folders_is_refused(tmp_path):
    raw = yaml.safe_load(pl.CONFIG.read_text())
    raw["data"]["interim_dir"] = "data/interim/kkbox"
    path = tmp_path / "kkbox_paths.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="outside"):
        pl.Setup(path)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_root")
    raw = yaml.safe_load(pl.CONFIG.read_text())
    raw["data"]["interim_dir"] = str(root / "interim")
    raw["labels"]["processed_dir"] = str(root / "processed")
    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    setup = pl.Setup(path, root=root)
    report = pl.run_all(setup, generated=root / "generated")
    return setup, report


def test_the_small_path_builds_labels_features_and_a_model(small_run):
    setup, report = small_run
    for f in ("labels.parquet", "features.parquet", "timeline.parquet", "month_quality.parquet"):
        assert (setup.processed / f).exists(), f
    assert report["labels_and_features"]["feature_rows"] > 1000
    for h in ("7", "30"):
        t = report["train"]["horizons"][h]
        assert t["fit_events"] > 0 and t["score_rows"] > 0 and t["feature_columns"] == 46
        assert 0.5 < t["auc_synthetic"] <= 1.0
        assert t["unseen_levels"] == dict.fromkeys(t["unseen_levels"], 0)


def test_the_small_path_stays_under_the_registered_ceiling(small_run):
    _, report = small_run
    assert report["seconds"] < 10 * 60  # the smoke job's ceiling is 10 minutes (R3)


def test_every_output_says_synthetic(small_run):
    setup, report = small_run
    assert report["synthetic"] is True and report["label"] == SYNTHETIC_SENTENCE
    assert report["train"]["synthetic"] is True
    assert "synthetic" in report["train"]["metrics_note"]
    for d in (setup.interim, setup.processed, setup.processed / "models"):
        assert (d / "SYNTHETIC").read_text().strip() == SYNTHETIC_SENTENCE
    models = sorted((setup.processed / "models").glob("*"))
    assert all(m.name.startswith("synthetic_") or m.name == "SYNTHETIC" for m in models)
    validation = json.loads((setup.processed / "validation.json").read_text())
    assert validation["synthetic"]["synthetic"] is True
    saved = json.loads((setup.processed / "synthetic_pipeline_report.json").read_text())
    assert saved["synthetic"] is True and saved["train"]["label"] == SYNTHETIC_SENTENCE


def _features_with_listening_changed(setup, tmp_path, name, first_day, last_day):
    import duckdb

    from retention.features.build import FeaturePaths, build

    logs = setup.interim / "user_logs.parquet"
    changed = tmp_path / f"{name}.parquet"
    hit = f"date >= {first_day} and date <= {last_day}"
    duckdb.sql(
        f"""copy (select msno, date, num_25, num_50, num_75, num_985, num_100, num_unq,
        case when {hit} then total_secs * 3 else total_secs end as total_secs
        from '{logs}') to '{changed}' (format parquet)"""
    )
    changed_rows = duckdb.sql(f"select count(*) from '{logs}' where {hit}").fetchone()[0]
    out = tmp_path / f"features_{name}.parquet"
    v2 = setup.interim / "user_logs_v2.parquet"
    build(FeaturePaths(setup.processed, (changed, v2)), setup.cfg, [date(2016, 9, 1)], out=out)
    return changed_rows, pd.read_parquet(out).sort_values("msno").reset_index(drop=True)


def test_changing_only_post_snapshot_listening_leaves_the_features_unchanged(small_run, tmp_path):
    """Listening changed from the snapshot day on must not move the snapshot's features."""
    setup, _ = small_run
    base = pd.read_parquet(setup.processed / "features.parquet")
    base = base[base.snapshot_date == date(2016, 9, 1)].sort_values("msno").reset_index(drop=True)
    rows, after = _features_with_listening_changed(setup, tmp_path, "after", 20160901, 20161231)
    assert rows > 0
    pd.testing.assert_frame_equal(base[after.columns], after)
    # Control: the same change dated just before the snapshot does move the features.
    rows, before = _features_with_listening_changed(setup, tmp_path, "before", 20160820, 20160831)
    assert rows > 0
    assert not base[before.columns].equals(before)


def test_the_365_day_counts_are_all_time_counts_in_the_synthetic_data(small_run):
    """The premise of the calendar-growth explanation of the two 365-day columns' drift.

    The synthetic data span 2016-01-01 to 2016-12-31, so the 365 days before every snapshot reach
    back past the first row: the look-back never truncates, n_txn_365 and n_paid_365 equal the
    customer's count of all earlier rows, and they can only grow as the snapshot moves later.
    This tests that premise on the synthetic tables. It does not test that the unshifted drift of
    those two columns in the drift run is caused by it, and it says nothing about the KKBox data.
    """
    import duckdb

    setup, _ = small_run
    tx = f"'{setup.processed}/transactions.parquet'"
    feats = f"'{setup.processed}/features.parquet'"
    q = f"""
    with t as (select msno, transaction_date d, is_cancel, actual_amount_paid paid from {tx}),
    f as (select msno, snapshot_date T, n_txn_365, n_paid_365 from {feats})
    select f.T, f.msno, f.n_txn_365, f.n_paid_365,
      (select count(*) from t where t.msno = f.msno and t.d < f.T and t.is_cancel = 0) as all_txn,
      (select count(*) from t where t.msno = f.msno and t.d < f.T and t.d >= f.T - 365
         and t.is_cancel = 0) as win_txn,
      (select count(*) from t where t.msno = f.msno and t.d < f.T and t.is_cancel = 0
         and t.paid > 0) as all_paid
    from f where f.n_txn_365 is not null
    """
    df = duckdb.sql(q).df()
    assert len(df) > 1000
    assert (df.all_txn == df.win_txn).all()  # the look-back never truncates
    assert (df.n_txn_365 == df.win_txn).all() and (df.n_paid_365 == df.all_paid).all()
    means = df.groupby("T").n_txn_365.mean().sort_index()
    assert len(means) >= 4 and means.is_monotonic_increasing  # grows as the snapshot moves
