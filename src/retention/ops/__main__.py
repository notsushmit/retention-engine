"""`python -m retention.ops drift-report [--defaults]`: the drift report of the two drift replays.

--defaults: write the library's default thresholds to the ops folder and stop (the registered
step before any report). Without it: read the two replay tables the drift replays kept
(`python -m retention.stream drift-replay`), the batch feature table and the registered models,
run the Evidently reports (configs/ops.yaml, R1) and write aggregates to the ops folder:
drift_unshifted.json, drift_shifted.json, drift_latest.json (what GET /drift serves) and
drift_real_report.json (the dataset-level result as registered and the group-level mark of A15,
side by side). Refuses a dirty tree and a second run; touches no Valkey database.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from retention.api import __main__ as api_cli
from retention.ops import drift as dr
from retention.ops import real_drift
from retention.stream import __main__ as stream_cli

REPORT = "drift_real_report.json"
FIT_SNAPSHOTS = ("2016-08-01", "2016-09-01")
PREDICTION_SNAPSHOT = "2016-10-01"
BATCH_REFERENCE_MONTH, BATCH_CURRENT_MONTH = "2016-09", "2016-10"


def ops_dir(fraction) -> Path:
    return api_cli.out_dir(fraction).parent / "ops"


def gather(args) -> dict[str, Any]:
    """Everything the report reads: the tables, the models through the registry alias, counts."""
    from datetime import date

    from retention.api import registry
    from retention.api import scoring as sc
    from retention.explain.model import load_settings
    from retention.labels.config import load_label_config
    from retention.models.__main__ import interim_dir, processed_dir
    from retention.models.protocol import load_protocol

    protocol = load_protocol()
    settings = load_settings(horizons=(7, 30, 59))
    with tempfile.TemporaryDirectory() as tmp:
        models = registry.load_serving_models(
            registry.registry_settings(Path.cwd()), settings, protocol, Path(tmp)
        )
    processed, interim = processed_dir(args.fraction), interim_dir(args.fraction)

    def snapshot(day: str):
        t = date.fromisoformat(day)
        return ds.dataset(processed / "features.parquet").to_table(
            filter=pc.field("snapshot_date") == t
        )

    def predict(rows):
        import pandas as pd

        final = sc.probabilities(models.c01, rows, protocol.seed).final
        return pd.DataFrame(final, columns=dr.PREDICTIONS)

    tx = [interim / "transactions.parquet", interim / "transactions_v2.parquet"]
    logs = [interim / "user_logs.parquet", interim / "user_logs_v2.parquet"]
    return {
        "fit_tables": [snapshot(d) for d in FIT_SNAPSHOTS],
        "prediction_table": snapshot(PREDICTION_SNAPSHOT),
        "unshifted_table": pq.read_table(stream_cli.drift_table_path(False)),
        "shifted_table": pq.read_table(stream_cli.drift_table_path(True)),
        "predict": predict,
        "window": load_label_config().renewal_window_days,
        "main": tuple(protocol.main_segments),
        "batch_reference": dr.daily_events(tx, logs, BATCH_REFERENCE_MONTH),
        "batch_current": dr.daily_events(tx, logs, BATCH_CURRENT_MONTH),
        "models_version": models.version,
    }


def replays_note(out: Path) -> dict[str, Any]:
    """Counts only, from the two replay reports."""
    notes = {}
    for shift in (False, True):
        r = json.loads((out / stream_cli.drift_report_name(shift)).read_text())
        notes[stream_cli.drift_tag(shift)] = {
            k: r[k] for k in ("database", "produced", "consumed", "git_head", "table_sha256")
        }
    return notes


def report_cmd(args, gather=gather) -> int:
    out = ops_dir(args.fraction)
    if args.defaults:
        print(dr.log_defaults(out).read_text(), end="")
        return 0
    repo = Path.cwd()
    api_cli.require_clean(repo)
    path = out / REPORT
    api_cli.run_once(path)
    replays = replays_note(api_cli.out_dir(args.fraction))
    t0 = time.time()
    g = gather(args)
    result = real_drift.run(
        fit_tables=g["fit_tables"],
        prediction_table=g["prediction_table"],
        unshifted_table=g["unshifted_table"],
        shifted_table=g["shifted_table"],
        predict=g["predict"],
        window=g["window"],
        main=g["main"],
        batch_reference=g["batch_reference"],
        batch_current=g["batch_current"],
        out_dir=out,
        store_date="2016-11-01",
        replays=replays,
    )
    stamped = api_cli.stamp(result["report"], repo, t0)
    dr.write_json(path, stamped)
    print(json.dumps(stamped, indent=2, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.ops")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("drift-report", help="the drift report of the two drift replays")
    d.add_argument("--fraction", type=float, default=None)
    d.add_argument("--defaults", action="store_true", help="log the library defaults and stop")
    args = parser.parse_args(argv)
    return report_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
