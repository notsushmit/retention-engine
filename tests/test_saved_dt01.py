"""The checked dt01 loader, and refit.py's switch to the checked 7-day and 30-day loader."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from retention.api import dt01 as d
from retention.api.config import load_serving
from retention.api.refit import saved_models
from retention.api.saved import (
    ModelRefused,
    load_saved_dt01,
    saved_7_30_registration,
    saved_dt01_registration,
)
from retention.explain.model import load_settings
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import load_protocol
from retention.optimize.value import INTERVAL_ENDS_ALL
from test_save_dt01 import COLS, RAW, S, rows

P = load_protocol()
REAL_DIR, REAL_EXPECTED = saved_dt01_registration()
EXPLAIN_DIR, _ = saved_7_30_registration()


@pytest.fixture(scope="module")
def small() -> DiscreteTimeModel:
    cand = {"num_leaves": 7, "n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 20}
    return DiscreteTimeModel(
        False,
        list(INTERVAL_ENDS_ALL),
        cand,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(rows(), COLS)


def cfg_for(directory: Path, hashes: dict[str, str]) -> dict:
    cfg = copy.deepcopy(load_serving())
    cfg["ordering"]["saved_dt01"] = {"directory": str(directory), "files": hashes}
    return cfg


def test_registration_names_both_dt01_files_with_their_hashes():
    assert REAL_DIR == Path("data/interim/kkbox/sample_020/serving/model")
    assert REAL_EXPECTED == {
        "dt01.txt": "6d2edd153b89ba149c46de3f647c0b8900a4e22a66fd365d9c927714c2a7338d",
        "model_dt01.json": "c809fba302b62463756027cada619956f6128044d36b527ce885be9ee68a81a9",
    }


def test_registration_must_name_exactly_the_two_files():
    cfg = copy.deepcopy(load_serving())
    del cfg["ordering"]["saved_dt01"]["files"]["model_dt01.json"]
    with pytest.raises(ValueError, match="exactly"):
        saved_dt01_registration(cfg)


def test_checked_loader_rebuilds_a_saved_dt01(small, tmp_path):
    hashes = d.save(small, S, tmp_path, d.text_sha256(d.booster_text(small)))
    loaded = load_saved_dt01(P, cfg_for(tmp_path, hashes))
    r = rows(300, seed=2)
    a, _ = small.lapse_probability(r, list(INTERVAL_ENDS_ALL))
    b, _ = loaded.lapse_probability(r, list(INTERVAL_ENDS_ALL))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", [d.BOOSTER, d.META])
def test_checked_loader_refuses_a_changed_file(small, tmp_path, name):
    hashes = d.save(small, S, tmp_path, d.text_sha256(d.booster_text(small)))
    data = bytearray((tmp_path / name).read_bytes())
    data[-2] ^= 1
    (tmp_path / name).write_bytes(bytes(data))
    with pytest.raises(ModelRefused, match=name.replace(".", r"\.")):
        load_saved_dt01(P, cfg_for(tmp_path, hashes))


@pytest.mark.skipif(not REAL_DIR.joinpath("dt01.txt").exists(), reason="needs the saved dt01")
def test_real_dt01_files_pass_and_load():
    model = load_saved_dt01(P)
    assert d.text_sha256(d.booster_text(model)) == REAL_EXPECTED["dt01.txt"]
    assert model.ends == list(INTERVAL_ENDS_ALL) and model.offset_input is False


# --- refit.py now loads 7 and 30 days through the checked loader ----------------------------------


def test_saved_models_refuses_files_that_are_not_the_registered_ones(tmp_path):
    for name in ("booster_7d.txt", "model_7d.json", "booster_30d.txt", "model_30d.json"):
        (tmp_path / name).write_text("not the registered file\n")
    with pytest.raises(ModelRefused):
        saved_models(tmp_path, load_settings(horizons=(7, 30, 59)), P)


@pytest.mark.skipif(not EXPLAIN_DIR.exists(), reason="needs the saved explain files")
def test_saved_models_refuses_a_changed_7_day_json(tmp_path):
    for name in ("booster_7d.txt", "model_7d.json", "booster_30d.txt", "model_30d.json"):
        shutil.copy(EXPLAIN_DIR / name, tmp_path / name)
    data = bytearray((tmp_path / "model_7d.json").read_bytes())
    data[-2] ^= 1
    (tmp_path / "model_7d.json").write_bytes(bytes(data))
    with pytest.raises(ModelRefused, match=r"model_7d\.json"):
        saved_models(tmp_path, load_settings(horizons=(7, 30, 59)), P)
