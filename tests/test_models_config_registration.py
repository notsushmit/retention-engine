"""The evaluation config may only add keys after registration, never change one.

7b18e2b was the binding pre-registration commit before machine-readable
settings were added; every value registered there must be unchanged.
"""

import hashlib
import subprocess
from pathlib import Path

import yaml

REGISTERED_COMMIT = "7b18e2b"
REGISTERED_SHA256 = "1ea2696d56ab87277bcaddedcce0134b4862b81354c319573c4fe5cf4a5f680f"
ADDED = {
    ("execution", "threads"),
    *(
        ("classifier", "fixed_params", k)
        for k in (
            "deterministic",
            "force_col_wise",
            "bagging_fraction",
            "feature_fraction",
            "verbosity",
        )
    ),
    ("survival", "cox", "ties"),
    ("survival", "cox", "strata_bands", "p2"),
    ("survival", "cox", "strata_bands", "p1"),
    *(
        ("survival", "xgboost_aft", "fixed_params", k)
        for k in ("max_depth", "eta", "num_boost_round", "tree_method")
    ),
}


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, (*path, k))
    else:
        yield path, d


def registered_text() -> str:
    out = subprocess.run(
        ["git", "show", f"{REGISTERED_COMMIT}:configs/evaluation.yaml"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_registered_blob_has_the_registered_hash():
    assert hashlib.sha256(registered_text().encode()).hexdigest() == REGISTERED_SHA256


def test_every_registered_value_is_unchanged_and_only_known_keys_are_added():
    old = dict(leaves(yaml.safe_load(registered_text())))
    new = dict(leaves(yaml.safe_load(Path("configs/evaluation.yaml").read_text())))
    changed = {p: (v, new.get(p, "<missing>")) for p, v in old.items() if new.get(p) != v}
    assert changed == {}
    assert set(new) - set(old) == ADDED
