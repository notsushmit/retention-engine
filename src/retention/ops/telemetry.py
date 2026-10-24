"""Usage telemetry off (configs/ops.yaml, amendment A13).

evidently and prefect send usage data unless told not to. Call `telemetry_off()` before the first
import of either package; it sets the three registered variables, overriding anything already
set, and makes the Prefect home folder (a relative path inside the ignored data folder).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

OPS_CONFIG = Path("configs/ops.yaml")


def registered_environment(path: Path = OPS_CONFIG) -> dict[str, str]:
    return {
        k: str(v)
        for k, v in yaml.safe_load(path.read_text())["amendment_A13"]["environment"].items()
    }


def telemetry_off() -> dict[str, str]:
    env = registered_environment()
    os.environ.update(env)
    # Prefect creates only the last folder of its home; the parents must exist
    Path(env["PREFECT_HOME"]).mkdir(parents=True, exist_ok=True)
    return env
