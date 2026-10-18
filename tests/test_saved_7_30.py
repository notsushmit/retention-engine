"""The saved 7-day and 30-day files: a changed booster or model JSON is refused."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from retention.api.config import load_serving
from retention.api.saved import (
    ModelRefused,
    booster_name,
    load_saved_7_30,
    meta_name,
    saved_7_30_registration,
    verify_files,
)
from retention.explain.contributions import EXPECTED_SHA256
from retention.explain.model import load_settings
from retention.models.protocol import load_protocol

P = load_protocol()
REAL_DIR, REAL_EXPECTED = saved_7_30_registration()
OCTOBER = Path("mlruns/artifacts/9bca205ae00b4017a440d16057fa60e8/artifacts/october.json")
NAMES = [meta_name(7), meta_name(30), booster_name(7), booster_name(30)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_four(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    for h in (7, 30):
        (directory / booster_name(h)).write_text(f"tree\nversion=v4\nhorizon={h}\n")
        meta = {"horizon": h, "booster_sha256": sha(directory / booster_name(h))}
        (directory / meta_name(h)).write_text(json.dumps(meta | {"platt_a": 0.1, "platt_b": 1.0}))
    return {n: sha(directory / n) for n in NAMES}


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


def test_registration_has_the_two_jsons_and_the_phase_4_boosters():
    assert REAL_DIR == Path("data/interim/kkbox/sample_020/explain/model")
    assert REAL_EXPECTED == {
        "model_7d.json": "d317109cf938b6518e73be78887befe284f5e9790322933df0250024e8ea525a",
        "model_30d.json": "c78fbc8757ac3176a4ff780bf813cc1faee282ae8873de2c1549527e319fe7bb",
        "booster_7d.txt": EXPECTED_SHA256[7],
        "booster_30d.txt": EXPECTED_SHA256[30],
    }


def test_registration_must_name_exactly_the_two_jsons():
    cfg = load_serving()
    del cfg["ordering"]["saved_7_30"]["files"]["model_30d.json"]
    with pytest.raises(ValueError, match="exactly"):
        saved_7_30_registration(cfg)


def test_unchanged_files_pass(tmp_path):
    expected = write_four(tmp_path)
    assert verify_files(tmp_path, expected) == expected


@pytest.mark.parametrize("name", NAMES)
def test_a_changed_file_is_refused(tmp_path, name):
    expected = write_four(tmp_path)
    flip_last_byte(tmp_path / name)
    with pytest.raises(ModelRefused, match=re.escape(name)):
        verify_files(tmp_path, expected)


@pytest.mark.parametrize("h", [7, 30])
def test_a_changed_platt_value_is_refused(tmp_path, h):
    expected = write_four(tmp_path)
    meta = json.loads((tmp_path / meta_name(h)).read_text())
    meta["platt_a"] += 1e-12
    (tmp_path / meta_name(h)).write_text(json.dumps(meta))
    with pytest.raises(ModelRefused, match=re.escape(meta_name(h))):
        verify_files(tmp_path, expected)


@pytest.mark.parametrize("h", [7, 30])
def test_a_changed_booster_with_its_hash_rewritten_in_the_json_is_still_refused(tmp_path, h):
    expected = write_four(tmp_path)
    flip_last_byte(tmp_path / booster_name(h))
    meta = json.loads((tmp_path / meta_name(h)).read_text())
    meta["booster_sha256"] = sha(tmp_path / booster_name(h))
    (tmp_path / meta_name(h)).write_text(json.dumps(meta))
    with pytest.raises(ModelRefused) as err:
        verify_files(tmp_path, expected)
    assert booster_name(h) in str(err.value) and meta_name(h) in str(err.value)


@pytest.mark.parametrize("h", [7, 30])
def test_a_missing_json_is_refused(tmp_path, h):
    expected = write_four(tmp_path)
    (tmp_path / meta_name(h)).unlink()
    with pytest.raises(ModelRefused, match="missing"):
        verify_files(tmp_path, expected)


@pytest.mark.skipif(not REAL_DIR.exists(), reason="needs the saved explain files")
def test_real_files_pass_and_load():
    assert verify_files(REAL_DIR, REAL_EXPECTED) == REAL_EXPECTED
    models = load_saved_7_30(load_settings(horizons=(7, 30, 59)), P)
    assert [m.horizon for m in models] == [7, 30]
    assert [m.sha256 for m in models] == [EXPECTED_SHA256[7], EXPECTED_SHA256[30]]


@pytest.mark.skipif(
    not (REAL_DIR.exists() and OCTOBER.exists()), reason="needs the saved files and MLflow"
)
def test_real_platt_values_equal_the_logged_october_run():
    models = load_saved_7_30(load_settings(horizons=(7, 30, 59)), P)
    logged = json.loads(OCTOBER.read_text())["classifier_c01"]["platt"]
    for m in models:
        assert (m.platt_a, m.platt_b) == (logged[str(m.horizon)]["a"], logged[str(m.horizon)]["b"])


@pytest.mark.skipif(not REAL_DIR.exists(), reason="needs the saved explain files")
@pytest.mark.parametrize("name", NAMES)
def test_a_changed_copy_of_a_real_file_is_refused_by_the_loader(tmp_path, name):
    for n in NAMES:
        shutil.copy(REAL_DIR / n, tmp_path / n)
    flip_last_byte(tmp_path / name)
    with pytest.raises(ModelRefused, match=re.escape(name)):
        load_saved_7_30(load_settings(horizons=(7, 30, 59)), P, directory=tmp_path)


@pytest.mark.skipif(not REAL_DIR.exists(), reason="needs the saved explain files")
def test_a_real_booster_change_with_its_json_hash_rewritten_is_refused(tmp_path):
    for n in NAMES:
        shutil.copy(REAL_DIR / n, tmp_path / n)
    flip_last_byte(tmp_path / booster_name(7))
    meta = json.loads((tmp_path / meta_name(7)).read_text())
    meta["booster_sha256"] = sha(tmp_path / booster_name(7))
    (tmp_path / meta_name(7)).write_text(json.dumps(meta, indent=2) + "\n")
    with pytest.raises(ModelRefused) as err:
        load_saved_7_30(load_settings(horizons=(7, 30, 59)), P, directory=tmp_path)
    assert booster_name(7) in str(err.value) and meta_name(7) in str(err.value)
