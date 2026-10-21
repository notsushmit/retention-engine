"""Command line entry: `python -m retention.stream <command>`.

stream-parity --db <replay database>: judge the registered feature and score parity of
the replayed 2016-11-01 store (configs/serving.yaml, streaming.pass_marks, amendment
A9): the stream feature table at step6.outputs.stream_features against features.parquet
on A9's 61 judged columns, with the four count-only diagnostics of D-214 on the
mismatching customers, and score parity on the P2 main rows through the API's scoring
function with the models loaded through the registry alias. Refuses database 0 and any
database but the registered replay database, a dirty tree, and a second run. Writes
stream_parity_2016-11-01.json (counts only, no customer ID) and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from retention.api import __main__ as api_cli
from retention.api.config import load_serving
from retention.stream import move, parity, producer, source

REPORT = "stream_parity_{t}.json"


def batch_features(path: Path, t) -> pa.Table:
    """features.parquet at t, A9's judged columns, as stored."""
    return ds.dataset(path).to_table(
        columns=list(move.JUDGED), filter=pc.field("snapshot_date") == t
    )


def batch_timeline(path: Path, t):
    """Loader: timeline.parquet rows of these customers dated before t only."""
    import duckdb

    def load(ids: list[str]):
        con = duckdb.connect()
        try:
            con.register("_ids", pd.DataFrame({"msno": ids}))
            return con.sql(
                f"select t.* from read_parquet('{path}') t join _ids using (msno) "
                f"where t.transaction_date < date '{t.isoformat()}'"
            ).df()
        finally:
            con.close()

    return load


def gather(args, t) -> dict:
    """Everything stream-parity reads: the replay client, both tables, the timeline loaders,
    the logs, the models through the registry alias and the settings."""
    from retention.api import registry
    from retention.explain.model import load_settings
    from retention.labels.config import load_label_config
    from retention.models.__main__ import interim_dir, processed_dir
    from retention.models.protocol import load_protocol

    step6 = load_serving()["step6"]
    r = producer.client_for(args.db)
    producer.refuse_loaded_store(r)
    cfg = load_label_config()
    protocol = load_protocol()
    settings = load_settings(horizons=(7, 30, 59))
    processed = processed_dir(args.fraction)
    with tempfile.TemporaryDirectory() as tmp:
        models = registry.load_serving_models(
            registry.registry_settings(Path.cwd()), settings, protocol, Path(tmp)
        )

    def stream_timeline(ids: list[str]):
        return move.Timelines.build(move.stored_rows(r, ids), cfg).coverage

    return {
        "stream": pq.read_table(step6["outputs"]["stream_features"]).select(list(move.JUDGED)),
        "batch": batch_features(processed / "features.parquet", t),
        "stream_timeline": stream_timeline,
        "batch_timeline": batch_timeline(processed / "timeline.parquet", t),
        "logs": interim_dir(args.fraction) / "user_logs.parquet",
        "c01": models.c01,
        "models_version": models.version,
        "seed": protocol.seed,
        "columns": list(settings.columns),
        "window": cfg.renewal_window_days,
        "main": tuple(protocol.main_segments),
    }


def stream_parity_cmd(args, gather=gather) -> int:
    serving = load_serving()
    step6 = serving["step6"]
    if args.db == int(step6["databases"]["loaded_store"]):
        raise SystemExit("database 0 holds the loaded store; step 6 never reads it as a replay")
    if args.db != int(step6["databases"]["replay"]):
        raise SystemExit(
            f"stream-parity judges the replay database ({step6['databases']['replay']})"
        )
    repo = Path.cwd()
    api_cli.require_clean(repo)
    t = source.STORE_DATE
    out = api_cli.out_dir(args.fraction)
    report_path = out / REPORT.format(t=t.isoformat())
    api_cli.run_once(report_path)
    t0 = time.time()
    g = gather(args, t)
    report = parity.run(
        g["stream"],
        g["batch"],
        g["stream_timeline"],
        g["batch_timeline"],
        g["logs"],
        t,
        g["c01"],
        g["seed"],
        g["columns"],
        g["window"],
        g["main"],
        int(serving["stream_parity_columns"]["rows"]),
        int(parity.SCORE["rows"]),
    )
    report |= {"database": args.db, "models_version": g["models_version"]}
    report = api_cli.stamp(report, repo, t0)
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    if not (report["feature_parity"]["met"] and report["score_parity"]["met"]):
        print("STREAM PARITY NOT MET (the registered 68-column wording is not met as written, A9)")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.stream")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("stream-parity", help="judge feature and score parity of the replay")
    sp.add_argument("--fraction", type=float, default=None)
    sp.add_argument("--db", type=int, required=True)
    args = parser.parse_args(argv)
    return stream_parity_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
