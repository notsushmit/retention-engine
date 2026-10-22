"""The registered dashboard settings, read from configs/ with PyYAML.

configs/dashboard.yaml governs, with the later keys (C1 heatmap_replaced, C2
step3_conditions) taking precedence over the entries they amend, which stay in the
file unedited. The fixed disclaimers, the plan caption and the wording rules come
from configs/serving.yaml and configs/actions.yaml. Nothing under data/ is read and
the retention package is never imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def read(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text())


@dataclass(frozen=True)
class Page:
    key: str
    title: str
    group: str


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    serving: dict[str, Any]
    actions: dict[str, Any]

    @property
    def c1(self) -> dict[str, Any]:
        return self.raw["heatmap_replaced"]

    @property
    def c2(self) -> dict[str, Any]:
        return self.raw["step3_conditions"]

    # --- the source --------------------------------------------------------------------------
    @property
    def api_env(self) -> str:
        return self.raw["source"]["api_url"]["env"]

    @property
    def api_default(self) -> str:
        return self.raw["source"]["api_url"]["default"]

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        return tuple(tuple(c.split(" ", 1)) for c in self.raw["source"]["calls"])  # type: ignore[misc]

    @property
    def timeout(self) -> float:
        return float(self.raw["source"]["client_timeout_seconds"])

    @property
    def summary_timeout(self) -> float:
        return float(self.c1["timeouts"]["summary_seconds"])

    # --- pages -------------------------------------------------------------------------------
    @property
    def pages(self) -> tuple[Page, ...]:
        out = []
        for group in self.c1["pages_after"]["navigation"]:
            for key in group["pages"]:
                out.append(Page(key, self.raw["pages"][key]["title"], group["group"]))
        return tuple(out)

    def page(self, key: str) -> dict[str, Any]:
        return self.raw["pages"][key]

    # --- fixed texts -------------------------------------------------------------------------
    @property
    def banner(self) -> str:
        return self.raw["every_page"]["banner"]

    @property
    def disclaimers(self) -> tuple[str, ...]:
        return tuple(self.serving["wording"]["fixed_disclaimers"])

    @property
    def money_unit(self) -> str:
        return self.raw["every_page"]["money_unit"]

    @property
    def tokens(self) -> dict[str, str]:
        return {k: v["hex"] for k, v in self.raw["design_tokens"]["colours"].items()}

    @property
    def min_cell(self) -> int:
        return int(self.c1["suppression"]["min_cell_size"])

    @property
    def trend_labels(self) -> tuple[str, ...]:
        return tuple(self.serving["api"]["alerts"]["trend"]["labels"])

    @property
    def caption(self) -> str:
        return self.actions["strategies"]["table_caption"]


@cache
def load() -> Settings:
    return Settings(read("dashboard.yaml"), read("serving.yaml"), read("actions.yaml"))
