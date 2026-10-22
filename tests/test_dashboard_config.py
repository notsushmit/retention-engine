"""configs/dashboard.yaml may only add keys after its registration, never change one."""

import hashlib
import subprocess

import pytest
import yaml

DASHBOARD = "configs/dashboard.yaml"
REGISTRATIONS = {
    "fa00997": "c7a45472099be5833c14858d11606df21d011983b841e5068937983609da1cc1",  # step 0a
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
