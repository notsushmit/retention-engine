"""Only data.py may read files under src/retention/models/ (scan, like the TestUnlock scan).

Every other module reaches labels, cached label builds and features through
data.py's filtered loaders, so the snapshot cap, the column lists and the 2017
outcome lock cannot be skipped.
"""

import ast
from pathlib import Path

READERS = {"read_parquet", "read_table", "read_schema", "read_metadata", "ParquetFile", "dataset"}


def reads(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in READERS:
                found.append(f"{name}()")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "read_parquet" in node.value or "from '" in node.value:
                found.append(f"SQL {node.value[:40]!r}")
    return found


def test_only_data_py_reads_parquet_files():
    offenders = {
        p.name: reads(p)
        for p in Path("src/retention/models").glob("*.py")
        if p.name != "data.py" and reads(p)
    }
    assert offenders == {}


def test_the_scan_sees_reads_in_data_py():
    assert set(reads(Path("src/retention/models/data.py"))) >= {
        "dataset()",
        "read_parquet()",
        "read_schema()",
    }
