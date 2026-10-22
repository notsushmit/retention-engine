"""The dashboard reads only the API: no store client, no data reader, no file under data/."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dashboard.settings import load

S = load()
SOURCES = sorted(Path("dashboard").rglob("*.py")) + [Path("scripts/capture_pages.py")]
NEVER_IMPORTED = set(S.raw["source"]["never_imported"])
NEVER_CALLED = {"read_parquet", "read_csv", "read_feather", "read_json", "read_table", "open"}


def existing():
    return [p for p in SOURCES if p.exists()]


def test_the_registered_never_imported_list():
    assert NEVER_IMPORTED == {"redis", "duckdb", "pyarrow", "retention"}


@pytest.mark.parametrize("path", existing(), ids=str)
def test_no_forbidden_import(path):
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    assert set(found) & NEVER_IMPORTED == set()


@pytest.mark.parametrize("path", existing(), ids=str)
def test_no_file_reader_and_no_data_path(path):
    tree = ast.parse(path.read_text())
    calls = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert calls & NEVER_CALLED == set()
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef)
        and n.body
        and isinstance(n.body[0], ast.Expr)
    }
    strings = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]
    assert [s for s in strings if "data/" in s] == []


def test_the_scan_covers_the_package():
    assert Path("dashboard/client.py") in existing() and Path("dashboard/settings.py") in existing()
