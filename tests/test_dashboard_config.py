"""configs/dashboard.yaml may only add keys after its registration, never change one."""

import hashlib
import subprocess

import pytest
import yaml

DASHBOARD = "configs/dashboard.yaml"
REGISTRATIONS = {
    "fa00997": "c7a45472099be5833c14858d11606df21d011983b841e5068937983609da1cc1",  # step 0a
    "f8cbf93": "4d4345bc11a378aa3cd119e7e9b87cc487de6f07e949884b0678126ccb5e1c20",  # heatmap rules
    "5f6d6f2": "4fe3eda7d6226aac17845b7e6e18cb8d714dc98e4038bc2ee01fac0251888337",  # design tokens
    "a4d88af": "8f96465e88b402591f182e26c24d3ed4b550aa76d08cc5fa872436abcb260738",  # C1
    "a9dea38": "5408ec56a32ea935efb781cefd08e3b50ee234d8111d6e58c3d97dab83653d06",  # C2
}


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, (*path, k))
    else:
        yield path, d


def registered_text(commit: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{commit}:{DASHBOARD}"], capture_output=True, text=True, check=True
    )
    return out.stdout


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_registered_blob_has_the_registered_hash(commit):
    assert hashlib.sha256(registered_text(commit).encode()).hexdigest() == REGISTRATIONS[commit]


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_every_registered_value_is_unchanged(commit):
    old = dict(leaves(yaml.safe_load(registered_text(commit))))
    with open(DASHBOARD) as f:
        new = dict(leaves(yaml.safe_load(f)))
    assert {p: v for p, v in old.items() if new.get(p, "<missing>") != v} == {}
