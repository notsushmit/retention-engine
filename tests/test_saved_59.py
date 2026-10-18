"""The saved 59-day files: a changed booster or model JSON is refused."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from retention.api.config import load_serving, ordering_marks
from retention.api.saved import (
    BOOSTER,
    META,
    ModelRefused,
    load_saved_59,
    saved_59_registration,
    verify_files,
)
from retention.explain.model import load_settings
from retention.models.protocol import load_protocol

P = load_protocol()
REAL_DIR, REAL_EXPECTED = saved_59_registration()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pair(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BOOSTER).write_text("tree\nversion=v4\nnum_class=1\n")
    (directory / META).write_text(json.dumps({"horizon": 59, "platt_a": 0.1, "platt_b": 1.0}))
    return {BOOSTER: sha(directory / BOOSTER), META: sha(directory / META)}


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


def test_registration_names_both_files_with_full_hashes():
    assert REAL_DIR == Path("data/interim/kkbox/sample_020/serving/model")
    assert sorted(REAL_EXPECTED) == [BOOSTER, META]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in REAL_EXPECTED.values())
    assert REAL_EXPECTED[BOOSTER] == (
        "21f7240dce242d3a53287cee502742431c1747b88d23d6259d7ab7709d81d13a"
    )
    assert REAL_EXPECTED[META] == (
        "e1b4bbc70fb8183921b374cd9498f1e80249d52df4517e6e42d9d58a29734cb5"
    )


def test_registration_must_name_exactly_the_two_files():
    cfg = load_serving()
    del cfg["ordering"]["saved_59"]["files"][META]
    with pytest.raises(ValueError, match="exactly"):
        saved_59_registration(cfg)


def test_unchanged_files_pass(tmp_path):
    expected = write_pair(tmp_path)
    assert verify_files(tmp_path, expected) == expected


@pytest.mark.parametrize("name", [BOOSTER, META])
def test_a_changed_file_of_either_kind_is_refused(tmp_path, name):
    expected = write_pair(tmp_path)
    flip_last_byte(tmp_path / name)
    with pytest.raises(ModelRefused, match=re.escape(name)):
        verify_files(tmp_path, expected)


@pytest.mark.parametrize("name", [BOOSTER, META])
def test_a_missing_file_of_either_kind_is_refused(tmp_path, name):
    expected = write_pair(tmp_path)
    (tmp_path / name).unlink()
    with pytest.raises(ModelRefused, match="missing"):
        verify_files(tmp_path, expected)


def test_a_changed_booster_with_its_hash_rewritten_in_the_json_is_still_refused(tmp_path):
    expected = write_pair(tmp_path)
    flip_last_byte(tmp_path / BOOSTER)
    meta = json.loads((tmp_path / META).read_text())
    meta["booster_sha256"] = sha(tmp_path / BOOSTER)
    (tmp_path / META).write_text(json.dumps(meta))
    with pytest.raises(ModelRefused) as err:
        verify_files(tmp_path, expected)
    assert BOOSTER in str(err.value) and META in str(err.value)


@pytest.mark.skipif(not REAL_DIR.exists(), reason="needs the saved 59-day files")
def test_real_files_pass_and_load_with_the_registered_platt():
    assert verify_files(REAL_DIR, REAL_EXPECTED) == REAL_EXPECTED
    model = load_saved_59(load_settings(horizons=(7, 30, 59)), P)
    marks = ordering_marks()
    assert (model.platt_a, model.platt_b) == (marks.platt_a, marks.platt_b)
    assert model.sha256 == REAL_EXPECTED[BOOSTER]


@pytest.mark.skipif(not REAL_DIR.exists(), reason="needs the saved 59-day files")
@pytest.mark.parametrize("name", [BOOSTER, META])
def test_a_changed_copy_of_a_real_file_is_refused_by_the_loader(tmp_path, name):
    for f in (BOOSTER, META):
        shutil.copy(REAL_DIR / f, tmp_path / f)
    flip_last_byte(tmp_path / name)
    with pytest.raises(ModelRefused, match=re.escape(name)):
        load_saved_59(load_settings(horizons=(7, 30, 59)), P, directory=tmp_path)
