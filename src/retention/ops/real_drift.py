"""The drift comparison on the 20% sample's two replays (configs/ops.yaml, R1 and A15).

It reads the replay tables the two drift replays kept (the October 2016 events replayed into a
store at 2016-11-01, once as recorded and once with the listening halved), the batch feature
table, the registered models, and the daily event counts. It reads no outcome and writes
aggregates only. Reference features: the P2 main rows at 2016-08-01 and 2016-09-01 (the fit
months); reference predictions: the final probabilities of the P2 main rows at 2016-10-01 from
the batch table; current: the replay's P2 main rows with their final probabilities.

Two results are reported side by side and neither replaces the other: the dataset-level rules as
registered (R1: the margin rule and the dataset alert) and the group-level mark of amendment A15
(added after a result, unvalidated).

The tables, the scoring function and the counts come in through arguments, so the same code runs on
hand-made tables in the tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from retention.features.build import NULLABLE
from retention.ops import drift as dr

Predict = Callable[[pd.DataFrame], pd.DataFrame]  # feature rows -> p7, p30, p59


def p2_main(table: pa.Table, window: int, main: tuple[str, ...]) -> pd.DataFrame:
    """P2 main rows of a feature table: expiring within the window and a main segment."""
    df = table.to_pandas(types_mapper=NULLABLE.get, date_as_object=False)
    keep = (df.days_until_expiry < window) & df.segment.isin(main)
    return df[keep.fillna(False).to_numpy()].sort_values("msno").reset_index(drop=True)


def with_predictions(rows: pd.DataFrame, predict: Predict) -> pd.DataFrame:
    preds = predict(rows).reset_index(drop=True)
    cols = dr.compared_columns()[:-3]
    return pd.concat([rows[cols].reset_index(drop=True), preds[dr.PREDICTIONS]], axis=1)


def reference_frame(
    fit_tables: list[pa.Table],
    prediction_table: pa.Table,
    predict: Predict,
    window: int,
    main: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Features from the fit months' rows, predictions from a later month's rows; each column is
    filled for the rows it comes from and empty for the others."""
    feats = pd.concat([p2_main(t, window, main) for t in fit_tables], ignore_index=True)
    preds_rows = p2_main(prediction_table, window, main)
    preds = predict(preds_rows).reset_index(drop=True)
    frame = pd.concat([feats[dr.compared_columns()[:-3]], preds[dr.PREDICTIONS]], ignore_index=True)
    info = {
        "features": "P2 main rows at the fit months",
        "feature_rows": len(feats),
        "predictions": "final probabilities of the P2 main rows of a later month",
        "prediction_rows": len(preds),
        "note": "predictions and features come from different rows",
    }
    return frame, info


def run(
    *,
    fit_tables: list[pa.Table],
    prediction_table: pa.Table,
    unshifted_table: pa.Table,
    shifted_table: pa.Table,
    predict: Predict,
    window: int,
    main: tuple[str, ...],
    batch_reference: pd.Series,
    batch_current: pd.Series,
    out_dir: Path,
    store_date: str,
    replays: dict[str, Any],
) -> dict[str, Any]:
    """Both replays against the reference; writes the reports and the GET /drift file."""
    if not (out_dir / dr.DEFAULTS_FILE).exists():
        raise RuntimeError(f"log the library defaults first ({out_dir / dr.DEFAULTS_FILE})")
    defaults = json.loads((out_dir / dr.DEFAULTS_FILE).read_text())
    reference, ref_info = reference_frame(fit_tables, prediction_table, predict, window, main)
    plain = with_predictions(p2_main(unshifted_table, window, main), predict)
    moved = with_predictions(p2_main(shifted_table, window, main), predict)
    unshifted = dr.run_report(reference, plain, out_dir)
    shifted = dr.run_report(reference, moved, out_dir)
    margin = dr.margin_result(unshifted, shifted)
    group = dr.group_mark(unshifted, shifted)
    batch = dr.batch_size_indicator(batch_reference, batch_current)
    current = {
        "store_date": store_date,
        "rows": shifted["current_rows"],
        "shift": "listening halved in the October 2016 listening events",
        "replays": replays,
    }
    common = {"synthetic": False, "sentence": dr.LABEL}
    dr.write_json(out_dir / "drift_unshifted.json", {**common, "shifted": False, **unshifted})
    dr.write_json(out_dir / "drift_shifted.json", {**common, "shifted": True, **shifted})
    latest = dr.body(
        shifted_report=shifted,
        margin=margin,
        shifted=True,
        batch=batch,
        reference=ref_info,
        current=current,
        synthetic=False,
        defaults=defaults,
    )
    dr.write_json(out_dir / "drift_latest.json", latest)
    both = {
        "label": dr.LABEL,
        "dataset_level_as_registered": {
            "rule": "the shifted share exceeds the unshifted by at least 0.20 AND the shifted "
            "store raises the dataset alert (R1)",
            "unshifted_share": unshifted["drift_share"],
            "shifted_share": shifted["drift_share"],
            "margin": margin,
            "met": margin["met"],
        },
        "group_level": group,
        "unshifted_alert": unshifted["dataset_alert"],
        "batch_size": batch,
    }
    dr.write_json(out_dir / "drift_real_report.json", both)
    return {"unshifted": unshifted, "shifted": shifted, "margin": margin, "group": group,
            "batch": batch, "latest": latest, "report": both}  # fmt: skip
