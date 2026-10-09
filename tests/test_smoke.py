import importlib

import pytest

import retention

SUBPACKAGES = [
    "data",
    "labels",
    "features",
    "models",
    "explain",
    "trends",
    "optimize",
    "streaming",
    "api",
    "monitoring",
]


def test_version() -> None:
    assert retention.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"retention.{name}")
