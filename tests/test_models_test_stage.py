from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from retention.models import data as data_module
from retention.models import protocol as protocol_module
from retention.models.protocol import load_protocol
from retention.models.test_stage import (
    REHEARSAL_MONTHS,
    evaluate_cell,
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
def test_rehearsal_never_unlocks_and_never_reads_2017(monkeypatch):
    def no_unlock(self):
        raise AssertionError("the rehearsal constructed a TestUnlock")

    monkeypatch.setattr(protocol_module.TestUnlock, "__init__", no_unlock)
    seen: list[date] = []
    real_read = data_module._read

    def spy(path, snapshots, columns):
        seen.extend(snapshots)
        return real_read(path, snapshots, columns)

    monkeypatch.setattr(data_module, "_read", spy)
    out = run_test_stage(P, PROCESSED, CACHE, 5, rehearsal=True, buckets=2)  # both halves
    assert seen and max(seen) < LOCKED  # nothing dated 2017 was read at all
    assert out["rehearsal"] and out["cells"]
    assert {c["status"] for c in out["cells"]} <= {"ok", "not computable", "not interpretable"}
