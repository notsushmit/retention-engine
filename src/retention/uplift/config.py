"""The registered uplift design (configs/uplift.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG = Path("configs/uplift.yaml")


def load(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())
