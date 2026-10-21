"""Comparing the stream's feature table with batch's, under A9's rules.

configs/serving.yaml, streaming.pass_marks.feature_parity and amendment A9: the 61
judged columns; integer, category, date and boolean values and empty positions
exact; floats within a relative 1e-12 (|a - b| <= 1e-12 * max(|a|, |b|)). Rows are
matched by msno; a customer in one table and not the other is a population
difference. Differences in the labels-state columns, and population differences,
are reported as labels-state differences (a Phase 2 finding only after the
diagnostics, D-214); every other difference is a stream difference.

The count-only diagnostics of D-214 are added with the stream-parity command.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa

from retention.api.config import load_serving
from retention.stream.move import JUDGED, LABELS_STATE

FLOAT_TOLERANCE = float(
    load_serving()["streaming"]["pass_marks"]["feature_parity"]["float_relative_tolerance"]
)


def _same_float(a: float | None, b: float | None) -> tuple[bool, float]:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        both_empty = (a is None or math.isnan(a)) and (b is None or math.isnan(b))
        return both_empty, 0.0
    if math.isnan(b):
        return False, 0.0
    if a == b:
        return True, 0.0
    rel = abs(a - b) / max(abs(a), abs(b))
    return rel <= FLOAT_TOLERANCE, rel


def compare_tables(stream: pa.Table, batch: pa.Table) -> dict[str, Any]:
    """Every judged column of the two tables, row by row on msno."""
    if list(stream.column_names) != list(JUDGED) or list(batch.column_names) != list(JUDGED):
        raise ValueError("both tables must hold exactly A9's judged columns, in order")
    s = {m: i for i, m in enumerate(stream.column("msno").to_pylist())}
    b = {m: i for i, m in enumerate(batch.column("msno").to_pylist())}
    common = sorted(set(s) & set(b))
    columns: dict[str, dict[str, Any]] = {}
    for name in JUDGED:
        sv, bv = stream.column(name).to_pylist(), batch.column(name).to_pylist()
        is_float = pa.types.is_floating(stream.schema.field(name).type)
        differing, largest = 0, 0.0
        for m in common:
            x, y = sv[s[m]], bv[b[m]]
            if is_float:
                ok, rel = _same_float(x, y)
                largest = max(largest, rel)
            else:
                ok = x == y
            differing += not ok
        if differing:
            columns[name] = {
                "differing": differing,
                "largest_relative": largest if is_float else None,
            }
    missing, extra = sorted(set(b) - set(s)), sorted(set(s) - set(b))
    labels = [c for c in columns if c in LABELS_STATE]
    other = [c for c in columns if c not in LABELS_STATE]
    return {
        "met": not columns and not missing and not extra,
        "rows_stream": len(s),
        "rows_batch": len(b),
        "rows_compared": len(common),
        "rows_only_in_batch": len(missing),
        "rows_only_in_stream": len(extra),
        "columns_differing": columns,
        "labels_state_differences": labels + (["population"] if missing or extra else []),
        "stream_differences": other,
    }
