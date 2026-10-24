"""configs/serving.yaml may only add keys after each registration, never change one."""

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

from retention.api.config import file_sha256, load_serving, ordering_marks
from retention.api.refit import SELECTION_ROWS
from retention.models.metrics import CLIP

REGISTRATIONS = {
    "4911ff6": "7b8a09a38b6f62c0bf23f167423883e630fdf198989f954264b5036e770611b0",
    "4c1f376": "b595ddc0b4e82c5806f438feb6c254039ae69cdc2533f611718ac1ca985439e8",
    "4cb85af": "761f3b96f7e9bac06f676bcbbda8c6748af5256f4b1c2dad9b208511e88ada0d",
    "6b91c1b": "10aff11c2e4e48c4be1a1b9fcdc7dc2eebe02d2cc8b7637eeb100ecc997bf6bd",  # amendment A1
    "63be031": "221ba813a83797ef9ee7cc098cc6e256d3286033c62e8e1b0c65e8998abcc40d",  # saved files
    "f6ae2e6": "b204e9d06f152ce8771a55abcc60ce7c5bcfa74dca5f6f31f53cf13f8664ed24",  # 7/30 JSONs
    "f7e63a6": "83f48a5c53699077458aefd55a777350a436dceea7df3a987905d5a4afb93429",  # A2
    "3e676f3": "f5fb37852c82f721b57321c3ff3d003893c574de8a7968c19be5ded02cab7052",  # registry
    "9e80930": "f343fd7b4fff18b57a520b7468c8936ee767c66b8230b1cd19233f4d3d498ad4",  # step 4
    "4e199b4": "a65d588c48b49b166f1d4cacf18b049e67953d949f73b714d2741cd5edcd6e73",  # A3
    "f1c46bc": "0973c46889ef7ffbe018b0fe92f58c943c1bc4dd26fc768adfc0e642e05cf2e5",  # A4
    "822def8": "c5b043114586b5138f660d4e1c76e9faa94435dda92e298d0a97f615b655dd78",  # step 5
    "21fd391": "c40e1a9206f619f11f941852d64b709c1d6efbb9b57c36de197a82ddcb600801",  # A5
    "8ec0086": "c06c7be5d7cb0f0938a069a05d3eaaaec17a3c0d1fa758e95c4e0da8edb0e63b",  # A6
    "9a5102f": "4ca6b5ae2e2f3b7d208429846f807dcbcf00f01e8ba4c14ba04b4dc1abfb6bba",  # A7
    "dc7948a": "025ef81dd872fee5c2e915e386e9099c16219714e48be50e28c1761e40745dcf",  # A8
    "ce301ce": "b5045c38d39101200db30588a439670025d964c041d644660283396aece2cecc",  # step 6
    "75ce40e": "d0139cffd0d866bf9288a471c731ddbf339961b4324cd3b2086c2f1902e93edb",  # A9
    "a6bb214": "076d7cc559df753651a72bd367e73f6c7177b9b367436976a8c132e6113d4b4f",  # step 7
    "e127920": "dc36d343dd5d789ac21314200662270580e3e4d262baec61c332261df9441928",  # step 7 images
    "caa6a39": "41e074e9ecf2cc47906b170095bad9c6b2e23911bede31e4bff233ef18cf46fc",  # step 8
    "2f30abc": "10b12f2e86c47e518aa1609650f82e0f9f0a90e4d0a6ec24a0cac03ccf628d14",  # A10
    "1cb0293": "b767fe8d93ea1300183d8961d447cd832b09b66d775506fa77a2b693afe6a77c",  # memory limit
    "ff3ee30": "3ee8032edae47da65234858ee7efb66cd1ec95106b04c989084bffcdf16c7789",  # dashboard
    "acfd3e8": "a9bb37f69bfaca948ee6d4f5c7b37782453187285d7e07d138c022db0ca055d0",  # R4
}
REHEARSAL = Path("data/processed/kkbox/sample_020/rehearsal.json")
OCTOBER = Path("mlruns/artifacts/9bca205ae00b4017a440d16057fa60e8/artifacts/october.json")


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, (*path, k))
    else:
        yield path, d


def registered_text(commit: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{commit}:configs/serving.yaml"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_registered_blob_has_the_registered_hash(commit):
    assert hashlib.sha256(registered_text(commit).encode()).hexdigest() == REGISTRATIONS[commit]


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_every_registered_value_is_unchanged(commit):
    old = dict(leaves(yaml.safe_load(registered_text(commit))))
    new = dict(leaves(load_serving()))
    assert {p: v for p, v in old.items() if new.get(p, "<missing>") != v} == {}


def test_ordering_marks_read_the_registered_values():
    m = ordering_marks()
    assert m.horizons == (7, 30, 59)
    assert m.clip == (CLIP, 1 - CLIP)
    assert m.fit_snapshots == (date(2016, 8, 1), date(2016, 9, 1))
    assert m.labels_as_of == date(2017, 1, 1)
    assert (m.threads, m.seed) == (4, 20161001)
    assert m.snapshot == date(2016, 11, 1)
    assert m.rehearsal_key[-1] == "score_summary_raw"
    assert (m.fix_rows, m.fix_rows_changed) == (169_673, 16_444)
    assert (m.fix_max_change, m.fix_max_change_tolerance) == (0.300535285645459, 1e-12)
    assert (m.platt_a, m.platt_b) == (0.02725170409239291, 1.0132027145157536)


def test_expected_summary_has_the_13_registered_fields():
    s = ordering_marks().expected_summary
    assert sorted(s) == ["deciles", "distinct", "max", "mean", "min"]
    assert len(s["deciles"]) == 9 and isinstance(s["distinct"], int)
    assert 3 + len(s["deciles"]) + 1 == 13


def test_selection_rows_constant_matches_the_config_text():
    text = load_serving()["ordering"]["descriptive"]["calibration_before_after_fix"]["slice"]
    assert f"{SELECTION_ROWS:,}" in text


def test_event_count_is_the_sum_of_its_parts():
    assert load_serving()["streaming"]["pass_marks"]["event_count"]["expected"] == (
        3_511_773 + 212_092 + 3_607
    )


@pytest.mark.skipif(not REHEARSAL.exists(), reason="needs the 20% sample's rehearsal.json")
def test_rehearsal_has_the_registered_hash_and_summary():
    m = ordering_marks()
    assert file_sha256(REHEARSAL) == m.rehearsal_sha256
    node = json.loads(REHEARSAL.read_text())
    for key in m.rehearsal_key:
        node = node[key]
    assert node == m.expected_summary
    fix = json.loads(REHEARSAL.read_text())["demonstration_59"]["59 days, 2016-11-01"]
    assert fix["rows"] == m.fix_rows
    assert fix["order_fix"]["classifier"]["rows_changed"] == m.fix_rows_changed
    assert fix["order_fix"]["classifier"]["max_change"] == m.fix_max_change


@pytest.mark.skipif(not OCTOBER.exists(), reason="needs the local MLflow artifacts")
def test_platt_values_equal_the_logged_october_run():
    m = ordering_marks()
    logged = json.loads(OCTOBER.read_text())["classifier_c01"]["platt"]["59"]
    assert (logged["a"], logged["b"]) == (m.platt_a, m.platt_b)
