"""Comparing the stream's feature table with batch's, under A9's rules.

configs/serving.yaml, streaming.pass_marks.feature_parity and amendment A9: the 61
judged columns; integer, category, date and boolean values and empty positions
exact; floats within a relative 1e-12 (|a - b| <= 1e-12 * max(|a|, |b|)). Rows are
matched by msno; a customer in one table and not the other is a population
difference. Differences in the labels-state columns, and population differences,
are reported as labels-state differences (a Phase 2 finding only after the
diagnostics, D-214); every other difference is a stream difference.

The four count-only diagnostics of D-214 run on the mismatching customers, in order:
the timeline rows before T (stream against batch), same-day ties, the usage columns
recomputed by Phase 2's UsageSource from the interim logs dated before T, and the
classification: a mismatch on same-day ties is an ordering cause; a labels-state
difference only, with matching inputs before T, is a Phase 2 finding; anything else is
a stream difference. Score parity scores the P2 main rows of both tables through the
API's scoring function. Every report is counts only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from retention.api import scoring as sc
from retention.api.config import load_serving
from retention.features.build import NULLABLE
from retention.features.usage import USAGE_COLUMNS, UsageSource
from retention.labels.timeline import TIE_KEYS
from retention.stream.move import A9, EXCLUDED, JUDGED, LABELS_STATE

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


# --- per customer --------------------------------------------------------------------------------


def customer_differences(stream: pa.Table, batch: pa.Table) -> dict[str, set[str]]:
    """{msno: the judged columns that differ}; "population" for a customer in one table only."""
    s = {m: i for i, m in enumerate(stream.column("msno").to_pylist())}
    b = {m: i for i, m in enumerate(batch.column("msno").to_pylist())}
    out: dict[str, set[str]] = {m: {"population"} for m in set(s) ^ set(b)}
    common = sorted(set(s) & set(b))
    for name in JUDGED:
        sv, bv = stream.column(name).to_pylist(), batch.column(name).to_pylist()
        is_float = pa.types.is_floating(stream.schema.field(name).type)
        for m in common:
            x, y = sv[s[m]], bv[b[m]]
            ok = _same_float(x, y)[0] if is_float else x == y
            if not ok:
                out.setdefault(m, set()).add(name)
    return out


# --- the four diagnostics (D-214), counts only ---------------------------------------------------

INPUT_COLUMNS = ["seq", "eff_expiry_after", "row_kind", "expiry_date"]
ORDER_KEYS = [  # timeline.order_rows: same day, same kind of row, same expiry, same content
    "msno",
    "transaction_date",
    "is_cancel",
    "sameday_sameplan_cancel",
    "expiry_date",
    *TIE_KEYS,
]
LAST_ROW_KEYS = [  # transactions.LAST_ROW_ORDER, over non-cancel rows
    "msno",
    "transaction_date",
    "expiry_date",
    "actual_amount_paid",
    "plan_list_price",
    "payment_plan_days",
    "payment_method_id",
    "is_auto_renew",
    "row_kind",
]


def _rows_by_customer(tl: pd.DataFrame, t: date) -> dict[str, list[tuple]]:
    tl = tl[pd.to_datetime(tl.transaction_date) < pd.Timestamp(t)].sort_values(["msno", "seq"])
    out: dict[str, list[tuple]] = {}
    for row in tl[["msno", *INPUT_COLUMNS]].itertuples(index=False):
        values = tuple(
            None if pd.isna(v) else str(pd.Timestamp(v).date()) if i in (1, 3) else v
            for i, v in enumerate(row[1:])
        )
        out.setdefault(row[0], []).append(values)
    return out


def inputs_match(stream_tl: pd.DataFrame, batch_tl: pd.DataFrame, t: date) -> dict[str, bool]:
    """Diagnostic 1: the same timeline rows before T (count, seq, effective expiry, row kind,
    expiry)."""
    s, b = _rows_by_customer(stream_tl, t), _rows_by_customer(batch_tl, t)
    return {m: s.get(m) == b.get(m) for m in set(s) | set(b)}


def tied(tl: pd.DataFrame, t: date) -> set[str]:
    """Diagnostic 2: customers with same-day rows before T tied on every order_rows key, or
    non-cancel rows tied on LAST_ROW_ORDER."""
    tl = tl[pd.to_datetime(tl.transaction_date) < pd.Timestamp(t)]
    keys = tl[ORDER_KEYS].astype(object).where(tl[ORDER_KEYS].notna(), "<null>")
    order = set(keys[keys.duplicated(keep=False)].msno)
    nc = tl[tl.is_cancel == 0]
    last = nc[LAST_ROW_KEYS].astype(object).where(nc[LAST_ROW_KEYS].notna(), "<null>")
    return order | set(last[last.duplicated(keep=False)].msno)


def usage_recount(logs: Path, msnos: list[str], t: date) -> pa.Table:
    """Diagnostic 3: the usage columns recomputed by Phase 2's UsageSource from the interim
    logs dated before T, for these customers only (the cohort ratio needs the whole
    population and is left out)."""
    con = duckdb.connect()
    try:
        bound = f"(select * from read_parquet('{logs}') where date < {t:%Y%m%d})"
        u = UsageSource(con, bound, msnos).features(t, msnos)
    finally:
        con.close()
    return pa.Table.from_pandas(u, preserve_index=False)


def _usage_equal(a: pa.Table, b: pa.Table, msnos: list[str]) -> set[str]:
    """Customers whose usage columns (all but the cohort ratio) are equal in a and b."""
    cols = [c for c in USAGE_COLUMNS if c != "ratio_secs_vs_cohort"]
    ai = {m: i for i, m in enumerate(a.column("msno").to_pylist())}
    bi = {m: i for i, m in enumerate(b.column("msno").to_pylist())}
    equal = set()
    for m in msnos:
        if m not in ai or m not in bi:
            continue
        ok = True
        for c in cols:
            x, y = a.column(c)[ai[m]].as_py(), b.column(c)[bi[m]].as_py()
            if isinstance(x, float) or isinstance(y, float):
                ok &= _same_float(None if x is None else float(x), None if y is None else float(y))[
                    0
                ]
            else:
                ok &= x == y
        if ok:
            equal.add(m)
    return equal


def classify(
    diffs: dict[str, set[str]], matched: dict[str, bool], ties: set[str]
) -> dict[str, str]:
    """Diagnostic 4: a mismatch on same-day ties is an ordering cause; a labels-state
    difference only, with matching inputs before T, is a Phase 2 finding; anything else
    is a stream difference."""
    labels = set(LABELS_STATE) | {"population"}
    out = {}
    for m, cols in diffs.items():
        if m in ties:
            out[m] = "ordering_cause"
        elif cols <= labels and matched.get(m, False):
            out[m] = "phase2_finding"
        else:
            out[m] = "stream_difference"
    return out


Timeline = Callable[[list[str]], pd.DataFrame]  # coverage timeline rows of these customers


def diagnostics(
    stream: pa.Table,
    batch: pa.Table,
    stream_timeline: Timeline,
    batch_timeline: Timeline,
    logs: Path,
    t: date,
) -> dict[str, Any]:
    """The four diagnostics on the mismatching customers only; counts only, no ID."""
    diffs = customer_differences(stream, batch)
    if not diffs:
        return {"mismatching_customers": 0}
    ids = sorted(diffs)
    stream_tl = stream_timeline(ids)
    matched = inputs_match(stream_tl, batch_timeline(ids), t)
    ties = tied(stream_tl, t) & set(ids)
    usage_ids = sorted(m for m, c in diffs.items() if c & set(USAGE_COLUMNS))
    recount = usage_recount(logs, usage_ids, t) if usage_ids else None
    classes = classify(diffs, matched, ties)
    by_column: dict[str, int] = {}
    for cols in diffs.values():
        for c in cols:
            by_column[c] = by_column.get(c, 0) + 1
    return {
        "mismatching_customers": len(ids),
        "by_column": dict(sorted(by_column.items())),
        "inputs_before_t_match": sum(matched.get(m, False) for m in ids),
        "inputs_before_t_differ": sum(not matched.get(m, False) for m in ids),
        "with_same_day_ties": len(ties),
        "usage_mismatching": len(usage_ids),
        "usage_stream_equals_recount": len(_usage_equal(stream, recount, usage_ids))
        if recount
        else 0,
        "usage_batch_equals_recount": len(_usage_equal(batch, recount, usage_ids))
        if recount
        else 0,
        "classes": {
            k: sum(v == k for v in classes.values())
            for k in ("ordering_cause", "phase2_finding", "stream_difference")
        },
    }


# --- score parity --------------------------------------------------------------------------------

SCORE = load_serving()["streaming"]["pass_marks"]["score_parity"]


def p2_rows(table: pa.Table, window: int, main: tuple[str, ...]) -> pd.DataFrame:
    """P2 main rows of a feature table: in_exp30 (days_until_expiry < window) and a main segment."""
    df = table.to_pandas(types_mapper=NULLABLE.get, date_as_object=False)
    keep = (df.days_until_expiry < window) & df.segment.isin(main)
    return df[keep.fillna(False).to_numpy()].sort_values("msno").reset_index(drop=True)


def score_parity(
    stream: pa.Table,
    batch: pa.Table,
    c01: dict,
    seed: int,
    columns: list[str],
    window: int,
    main: tuple[str, ...],
    expected_rows: int,
) -> dict[str, Any]:
    """Raw scores identical wherever features are identical; final probabilities within the
    registered tolerance; on the P2 main rows of both tables, through the API's scoring."""
    sp, bp = p2_rows(stream, window, main), p2_rows(batch, window, main)
    common = sorted(set(sp.msno) & set(bp.msno))
    sp = sp.set_index("msno").loc[common].reset_index()
    bp = bp.set_index("msno").loc[common].reset_index()
    models = [m.model for m in c01.values()]
    unseen = np.zeros(len(common), bool)
    for frame in (sp, bp):
        for m in models:
            unseen |= pd.notna(sc.unseen_feature(m, frame))
    sp, bp = sp[~unseen].reset_index(drop=True), bp[~unseen].reset_index(drop=True)
    same = np.ones(len(sp), bool)
    for c in columns:
        x, y = sp[c].astype(object), bp[c].astype(object)
        same &= ((x == y) | (x.isna() & y.isna())).fillna(False).to_numpy(dtype=bool)
    ps, pb = sc.probabilities(c01, sp, seed), sc.probabilities(c01, bp, seed)
    raw_same = (ps.raw == pb.raw).all(axis=1)
    tol = float(SCORE["final_probability_tolerance"])
    gap = np.abs(ps.final - pb.final).max(axis=1) if len(sp) else np.zeros(0)
    met = (
        len(set(p2_rows(stream, window, main).msno))
        == len(set(p2_rows(batch, window, main).msno))
        == expected_rows
        == len(common)
        and not unseen.any()
        and bool(same.all())
        and bool(raw_same[same].all())
        and bool((gap <= tol).all())
    )
    return {
        "met": bool(met),
        "rows_stream": int(len(p2_rows(stream, window, main))),
        "rows_batch": int(len(p2_rows(batch, window, main))),
        "rows_registered": expected_rows,
        "rows_compared": int(len(sp)),
        "rows_with_unseen_levels": int(unseen.sum()),
        "rows_with_identical_features": int(same.sum()),
        "raw_scores_identical_where_features_identical": int(raw_same[same].sum()),
        "largest_final_probability_gap": float(gap.max()) if len(gap) else 0.0,
        "final_probability_tolerance": tol,
    }


def run(
    stream: pa.Table,
    batch: pa.Table,
    stream_timeline: Timeline,
    batch_timeline: Timeline,
    logs: Path,
    t: date,
    c01: dict,
    seed: int,
    columns: list[str],
    window: int,
    main: tuple[str, ...],
    expected_feature_rows: int,
    expected_score_rows: int,
) -> dict[str, Any]:
    """The stream-parity report: counts only, no customer ID."""
    features = compare_tables(stream, batch)
    features["met"] = features["met"] and features["rows_batch"] == expected_feature_rows
    return {
        "registered_68_column_parity": A9["registered_wording"],
        "a9_columns_judged": len(JUDGED),
        "a9_columns_excluded": list(EXCLUDED),
        "feature_parity": features | {"rows_registered": expected_feature_rows},
        "diagnostics": diagnostics(stream, batch, stream_timeline, batch_timeline, logs, t),
        "score_parity": score_parity(
            stream, batch, c01, seed, columns, window, main, expected_score_rows
        ),
    }
