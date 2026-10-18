"""The saved classifier files, loaded only if every file matches its registered SHA-256.

configs/serving.yaml registers, for 59 days (`ordering.saved_59`), the booster
text and its model JSON (columns, category levels, Platt a and b), and for 7 and
30 days (`ordering.saved_7_30`) the model JSONs saved in Phase 4; those boosters'
hashes were registered in Phase 4 (retention.explain.contributions). Any file
with another hash is refused before anything is parsed, so a changed booster
cannot pass by also editing the hash written inside its JSON, and a changed
Platt value cannot pass at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retention.api.config import file_sha256, load_serving
from retention.explain.contributions import EXPECTED_SHA256, SavedModel, load_saved_model
from retention.explain.model import Settings
from retention.models.protocol import Protocol

HORIZON = 59
BOOSTER = "booster_59d.txt"
META = "model_59d.json"
SAVED_HORIZONS = (7, 30)


class ModelRefused(RuntimeError):
    """A saved model file is missing or its SHA-256 differs from the registered one."""


def saved_59_registration(cfg: dict[str, Any] | None = None) -> tuple[Path, dict[str, str]]:
    """The registered directory and {file name: SHA-256} of the saved 59-day files."""
    reg = (cfg if cfg is not None else load_serving())["ordering"]["saved_59"]
    files = dict(reg["files"])
    if sorted(files) != sorted([BOOSTER, META]):
        raise ValueError(f"saved_59 must register exactly {BOOSTER} and {META}")
    return Path(reg["directory"]), files


def verify_files(directory: Path, expected: dict[str, str]) -> dict[str, str]:
    """Each file's SHA-256; refuses a missing file or any hash that differs."""
    found, problems = {}, []
    for name, sha in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        found[name] = file_sha256(path)
        if found[name] != sha:
            problems.append(f"{name} SHA-256 {found[name]} differs from the registered {sha}")
    if problems:
        raise ModelRefused("; ".join(problems))
    return found


def load_saved_59(
    settings: Settings,
    protocol: Protocol,
    cfg: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> SavedModel:
    """The saved 59-day booster and its Platt parameters, after both files pass."""
    registered_dir, expected = saved_59_registration(cfg)
    directory = registered_dir if directory is None else directory
    verify_files(directory, expected)
    return load_saved_model(directory, HORIZON, settings, protocol, expected[BOOSTER])


def booster_name(horizon: int) -> str:
    return f"booster_{horizon}d.txt"


def meta_name(horizon: int) -> str:
    return f"model_{horizon}d.json"


def saved_7_30_registration(cfg: dict[str, Any] | None = None) -> tuple[Path, dict[str, str]]:
    """The registered directory and {file name: SHA-256} of the 7-day and 30-day files.

    The model JSONs come from configs/serving.yaml; the boosters from Phase 4's
    registered hashes.
    """
    reg = (cfg if cfg is not None else load_serving())["ordering"]["saved_7_30"]
    metas = dict(reg["files"])
    if sorted(metas) != sorted(meta_name(h) for h in SAVED_HORIZONS):
        raise ValueError("saved_7_30 must register exactly model_7d.json and model_30d.json")
    boosters = {booster_name(h): EXPECTED_SHA256[h] for h in SAVED_HORIZONS}
    return Path(reg["directory"]), metas | boosters


def load_saved_7_30(
    settings: Settings,
    protocol: Protocol,
    cfg: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> list[SavedModel]:
    """The saved 7-day and 30-day boosters and Platt parameters, after all four files pass."""
    registered_dir, expected = saved_7_30_registration(cfg)
    directory = registered_dir if directory is None else directory
    verify_files(directory, expected)
    return [
        load_saved_model(directory, h, settings, protocol, expected[booster_name(h)])
        for h in SAVED_HORIZONS
    ]
