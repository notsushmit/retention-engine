"""configs/ops.yaml may only add keys after each registration, never change one."""

import hashlib
import subprocess

import pytest
import yaml

OPS = "configs/ops.yaml"
REGISTRATIONS = {
    "9e0baa3": "15efb202c205998e069d3856463377dc439e163bb10b71f48b910f27c84f93c5",  # R0 to R3
    "e583248": "b76b9ba9d7032b05ffd31adcd1bcdb3722a01937dfa1c724745f02dce5a23622",  # A13
}


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, (*path, k))
    else:
        yield path, d


def registered_text(commit: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{commit}:{OPS}"], capture_output=True, text=True, check=True
    )
    return out.stdout


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_registered_blob_has_the_registered_hash(commit):
    assert hashlib.sha256(registered_text(commit).encode()).hexdigest() == REGISTRATIONS[commit]


@pytest.mark.parametrize("commit", list(REGISTRATIONS))
def test_every_registered_value_is_unchanged(commit):
    old = dict(leaves(yaml.safe_load(registered_text(commit))))
    with open(OPS) as f:
        new = dict(leaves(yaml.safe_load(f)))
    assert {p: v for p, v in old.items() if new.get(p, "<missing>") != v} == {}


def test_the_registered_marks_that_the_user_fixed_are_present():
    with open(OPS) as f:
        c = yaml.safe_load(f)
    assert c["hosting"]["chosen"] == "(a)"
    assert c["drift"]["pass_marks"]["margin_rule"]["margin"] == 0.20
    assert c["smoke"]["job_ceiling_minutes"] == 10
    assert c["synthetic"]["seed"] == 20261024
    env = c["amendment_A13"]["environment"]
    assert env["DO_NOT_TRACK"] == "1"
    assert env["PREFECT_SERVER_ANALYTICS_ENABLED"] == "false"
