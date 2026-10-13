import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.models import data as data_module
from retention.models import protocol as protocol_module
from retention.models import test_stage as stage_module
from retention.models.protocol import load_protocol
from retention.models.test_stage import (
    REHEARSAL_MONTHS,
    Checkpoint,
    MemoryCeiling,
    evaluate_cell,
    planned_cells,
    resolve_months,
    run_test_stage,
)

P = load_protocol()
PROCESSED = Path("data/processed/kkbox/sample_020")
CACHE = Path("data/interim/kkbox/sample_020/labels_asof")
LOCKED = date(2017, 1, 1)


def test_rehearsal_months_never_ask_for_the_test_months(monkeypatch):
    def refuse(self, unlock):
        raise AssertionError("the rehearsal asked for the test months")

    monkeypatch.setattr(protocol_module.Protocol, "test_months", refuse)
    months = resolve_months(P, None, rehearsal=True)
    assert all(m < LOCKED for ms in months.values() for m in ms)


def test_rehearsal_has_the_registered_shape():
    raw = yaml.safe_load(open("configs/evaluation.yaml"))
    shape = {int(h): len(ms) for h, ms in raw["test"]["months"].items()}
    shape |= {int(h): 1 for h in raw["test"]["demonstration"]}
    assert {h: len(ms) for h, ms in REHEARSAL_MONTHS.items()} == shape


def test_the_real_stage_refuses_without_an_unlock():
    with pytest.raises(PermissionError, match="TestUnlock"):
        run_test_stage(P, PROCESSED, CACHE, 5, unlock=None, rehearsal=False)


def test_a_cell_without_lapses_is_labelled_not_computable_and_does_not_crash():
    rng = np.random.default_rng(0)
    n = 300
    y = np.zeros(n, int)  # no lapse at this horizon
    raw = {"survival_dt01": rng.random(n), "classifier": rng.random(n)}
    out = evaluate_cell(y, np.arange(n), raw, dict(raw), [("survival_dt01", "classifier")], 20, 1)
    assert out["status"] == "ok" and out["lapses"] == 0
    m = out["models"]["survival_dt01"]
    assert m["auc_raw"]["status"] == "not computable"
    assert m["slope"]["status"] == "not computable"
    comp = out["comparisons"]["survival_dt01 minus classifier"]
    assert comp["status"] == "not computable"
    assert comp["verdicts"]["0.01"] == "not_computable"


@pytest.mark.skipif(
    not (PROCESSED / "labels.parquet").exists(), reason="needs the processed 20% sample"
)
def test_rehearsal_never_unlocks_and_never_reads_2017(monkeypatch, tmp_path):
    def no_unlock(self):
        raise AssertionError("the rehearsal constructed a TestUnlock")

    monkeypatch.setattr(protocol_module.TestUnlock, "__init__", no_unlock)
    seen: list[date] = []
    real_read = data_module._read

    def spy(path, snapshots, columns):
        seen.extend(snapshots)
        return real_read(path, snapshots, columns)

    monkeypatch.setattr(data_module, "_read", spy)
    ck = tmp_path / "checkpoint.json"
    out = run_test_stage(
        P, PROCESSED, CACHE, 5, rehearsal=True, buckets=2, checkpoint=ck, progress=lambda m: None
    )  # buckets 0-1 cover both October halves
    assert seen and max(seen) < LOCKED  # nothing dated 2017 was read at all
    assert out["rehearsal"] and out["cells"]
    assert {c["status"] for c in out["cells"]} <= {"ok", "not computable", "not interpretable"}
    assert len(out["cells"]) == planned_cells(REHEARSAL_MONTHS)

    # A second run resumes every cell from the checkpoint and returns the same results.
    def no_eval(*args, **kwargs):
        raise AssertionError("a checkpointed cell was computed again")

    monkeypatch.setattr(stage_module, "evaluate_cell", no_eval)
    monkeypatch.setattr(stage_module, "paired", no_eval)
    again = run_test_stage(
        P, PROCESSED, CACHE, 5, rehearsal=True, buckets=2, checkpoint=ck, progress=lambda m: None
    )
    timing = ("seconds", "fit_seconds")
    assert {k: v for k, v in again.items() if k not in timing} == {
        k: v for k, v in out.items() if k not in timing
    }


def test_the_rehearsal_records_the_planned_number_of_cells():
    # 3 primary cells x 7, the 59-day cell and its never-seen subset, the pooled
    # summary, survival metrics for 2 months, one shape check.
    assert planned_cells(REHEARSAL_MONTHS) == 27


def test_a_checkpoint_is_refused_outside_the_rehearsal(tmp_path):
    unlock = object.__new__(protocol_module.TestUnlock)  # simulated; the guard is not run
    with pytest.raises(PermissionError, match="only in the rehearsal"):
        run_test_stage(P, PROCESSED, CACHE, 5, unlock=unlock, checkpoint=tmp_path / "c.json")
    assert not list(tmp_path.iterdir())


def test_a_checkpoint_is_reused_only_with_the_same_fingerprint(tmp_path):
    path = tmp_path / "c.json"
    ck = Checkpoint(path, "abc")
    saved = ck.save("primary|7 days", {"status": "ok", "x": np.float64(0.25), "n": np.int64(3)})
    assert saved == {"status": "ok", "x": 0.25, "n": 3}
    assert Checkpoint(path, "abc").cells == {"primary|7 days": saved}
    assert Checkpoint(path, "other").cells == {}


def test_the_stage_stops_above_the_memory_ceiling(monkeypatch):
    monkeypatch.setattr(stage_module, "peak_rss_gb", lambda: stage_module.MEMORY_CEILING_GB + 1)

    def no_rows(*args, **kwargs):
        return pd.DataFrame(
            {"snapshot_date": pd.to_datetime(pd.Series([], dtype="object")), "msno": []}
        )

    monkeypatch.setattr(stage_module, "fit_rows", no_rows)
    with pytest.raises(MemoryCeiling, match="ceiling"):
        run_test_stage(P, PROCESSED, CACHE, 5, rehearsal=True, progress=lambda m: None)


PROGRESS = re.compile(r"^progress: (?P<msg>[a-z0-9_ ,()-]+), \d+ s, peak \d+\.\d\d GB$")
STATUS_WORDS = re.compile(
    r"\b(ok|not computable|not interpretable|pass|fail|win|loss|borderline|mixed)\b"
)


@pytest.mark.skipif(
    not (PROCESSED / "labels.parquet").exists(), reason="needs the processed 20% sample"
)
def test_the_real_path_writes_nothing_outside_the_mlflow_store(monkeypatch, tmp_path, capsys):
    """The real score-test path, unlock simulated, on 2016 stand-in months and a subset.

    Inputs are symlinked into a temporary data directory; the MLflow store is a
    temporary one. No new or changed file may appear anywhere in the temporary
    directory outside that store, or anywhere in the repository (data/ included),
    and stdout holds only progress lines and the final run summary.
    """
    import functools
    import shutil
    from types import SimpleNamespace

    from mlflow.tracking import MlflowClient

    from retention.models import __main__ as cli
    from retention.models import guard as guard_module
    from retention.models import tracking as tracking_module
    from retention.models.stages import LABEL_CODE

    # Other tests in this process may have imported the label-building code;
    # hide it so the stage's own check sees only what this run imports.
    monkeypatch.delitem(sys.modules, LABEL_CODE, raising=False)
    repo = Path.cwd()
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    shutil.copy(repo / "configs/evaluation.yaml", root / "configs/evaluation.yaml")
    processed, cache = root / "processed", root / "interim" / "labels_asof"
    for src, dst in ((PROCESSED, processed), (CACHE, cache)):
        dst.mkdir(parents=True)
        for item in src.iterdir():
            if item.is_file():
                (dst / item.name).symlink_to(item.resolve())

    real_setup = tracking_module.setup
    tr = real_setup(root)
    run_id = MlflowClient(tr.uri).create_run(tr.experiment_id).info.run_id
    unlock = object.__new__(protocol_module.TestUnlock)  # simulated; the guard is not run
    monkeypatch.setattr(tracking_module, "setup", lambda r, uri=None: real_setup(root))
    monkeypatch.setattr(guard_module, "open_test_session", lambda tr, reason=None: (unlock, run_id))
    stand_in = protocol_module.TestMonths(LOCKED, dict(REHEARSAL_MONTHS))
    monkeypatch.setattr(protocol_module.Protocol, "test_months", lambda self, u: stand_in)
    monkeypatch.setattr(cli, "processed_dir", lambda fraction: processed)
    monkeypatch.setattr(cli, "interim_dir", lambda fraction: root / "interim")
    monkeypatch.setattr(
        stage_module, "run_test_stage", functools.partial(stage_module.run_test_stage, buckets=2)
    )

    def files(base: Path, skip: tuple[str, ...]) -> dict[str, tuple[int, int]]:
        out = {}
        for p in base.rglob("*"):
            rel = p.relative_to(base)
            if rel.parts and rel.parts[0] in skip or "__pycache__" in rel.parts:
                continue
            if p.is_file():
                st = p.stat()
                out[str(rel)] = (st.st_size, st.st_mtime_ns)
        return out

    repo_skip = (".git", ".venv", "mlruns", ".pytest_cache", ".ruff_cache")
    before_repo, before_root = files(repo, repo_skip), files(root, ("mlruns",))
    capsys.readouterr()
    args = SimpleNamespace(
        rehearsal=False, buckets=None, replicates=5, rerun_reason=None, fraction=None
    )
    cli.score_test(args)
    printed = capsys.readouterr().out.splitlines()

    assert files(root, ("mlruns",)) == before_root
    assert files(repo, repo_skip) == before_repo
    run = MlflowClient(tr.uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "test_results.json" in [a.path for a in MlflowClient(tr.uri).list_artifacts(run_id)]

    *progress, summary = printed
    assert set(json.loads(summary)) == {"test_run", "seconds", "peak_rss_gb", "peak_rss_workers_gb"}
    assert progress and all(PROGRESS.match(line) for line in progress), progress
    for line in progress:
        msg = PROGRESS.match(line)["msg"]
        assert not STATUS_WORDS.search(msg), line
        assert not re.search(r"\d\.\d", msg), line  # no metric value
    assert sum("cell " in line for line in progress) == planned_cells(REHEARSAL_MONTHS)
