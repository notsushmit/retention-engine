"""configs/actions.yaml may only add keys after each registration, never change one.

Each commit below registered part of the Phase 7 design before anything it
judges was run. The one planned change is the headline budget level, which goes
from "not yet registered" to a number in its own commit after the value step.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

REGISTRATIONS = {
    "f020569": "89bdf1db528ef6a4ca9c64ce38599083c520310c5117de8601bb7ca3616c3115",
    "6849759": "2385faf844bb9fa165df247c6625a2c1ace44f4a59fe4a560445bf152c835480",
    "e25e3b2": "108a70f2d7df2135cc91ccf8f112b58b5638eaa340b83eea05e49902845ab1b6",
    "78d747e": "3500f60e3cad80fa114facfcf3d37c7223d8fcbc30ed3de7293ac30903818243",
    "1728a9f": "8170ed5a02b7508f7a36a8ce386879d9665b75de5e67d4cec9e703ba868ef05b",
}
PLANNED_CHANGE = ("sensitivity", "stability", "headline_budget", "level")


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, (*path, k))
    else:
        yield path, d


def registered_text(commit: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{commit}:configs/actions.yaml"],
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
    new = dict(leaves(yaml.safe_load(Path("configs/actions.yaml").read_text())))
    changed = {p: (v, new.get(p, "<missing>")) for p, v in old.items() if new.get(p) != v}
    if PLANNED_CHANGE in changed and changed[PLANNED_CHANGE][0] == "not yet registered":
        level = changed.pop(PLANNED_CHANGE)[1]
        assert level == "no_limit" or isinstance(level, int | float)
    assert changed == {}
