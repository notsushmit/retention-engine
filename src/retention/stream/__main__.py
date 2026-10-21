"""Command line entry: `python -m retention.stream <command>`.

replay --db <database> --speed full|throttled [--bucket K [--bucket J]] [--rerun-reason R]:
one replay (step6). With --bucket it is a dry run on one or two md5 buckets (D-227), not a
mark, projected to the full sample by a factor of 20 over the buckets replayed.
--rerun-reason reruns a dry run only: the earlier report and table are renamed with the
earlier report's git head as a suffix (never overwritten) and the reason is recorded in the
new report. A registered run (no --bucket) is never rerun.

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
import yaml

from retention.api import __main__ as api_cli
from retention.api.config import load_serving
from retention.stream import move, parity, producer, replay, source

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


# --- recount, cancel-ids, replay -----------------------------------------------------------------

RECOUNT = "recount_2016-10.json"
CANCEL_IDS = "cancel_ids_2016-10.json"


def interim(fraction):
    from retention.models.__main__ import interim_dir

    return interim_dir(fraction)


def begin(args, name: str, set_aside=None) -> tuple[Path, Path, float]:
    """Clean tree and the run-once guard; returns (repo, report path, start time). A dry
    run's labelled rerun sets the earlier outputs aside first (`set_aside(path)`)."""
    repo = Path.cwd()
    api_cli.require_clean(repo)
    path = api_cli.out_dir(args.fraction) / name
    if set_aside is not None:
        set_aside(path)
    api_cli.run_once(path)
    return repo, path, time.time()


def finish(report: dict, repo: Path, path: Path, t0: float) -> dict:
    report = api_cli.stamp(report, repo, t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay.json_ready(report))
    print(replay.json_ready(report), end="")
    return report


def recount_cmd(args, folder=None) -> int:
    repo, path, t0 = begin(args, RECOUNT)
    report = finish(replay.recount(folder or interim(args.fraction)), repo, path, t0)
    if not report["met"]:
        print(f"EVENT COUNT NOT MET: {report['total']} against {report['expected']}")
        return 1
    return 0


def scored_at(t, fraction) -> set[str]:
    """Customers scored at t: the P2 main population (in_exp30 and a main segment), read from
    labels.parquet's pre-T columns through the lock-aware loader."""
    from retention.models.__main__ import processed_dir
    from retention.models.data import load_labels
    from retention.models.protocol import load_protocol

    protocol = load_protocol()
    lab = load_labels(
        processed_dir(fraction) / "labels.parquet", [t], ["in_exp30", "segment"], protocol
    )
    keep = lab.in_exp30.astype(bool) & lab.segment.isin(protocol.main_segments)
    return set(lab.msno[keep.to_numpy()].astype(str))


def cancel_ids_cmd(args, folder=None, scored=None) -> int:
    repo, path, t0 = begin(args, CANCEL_IDS)
    ids = scored if scored is not None else scored_at(source.BOOTSTRAP, args.fraction)
    finish(replay.cancel_ids(folder or interim(args.fraction), ids), repo, path, t0)
    return 0


MAX_DRY_BUCKETS = 2


def buckets_of(args) -> tuple[int, ...] | None:
    """The dry run's buckets in the order given (None for a registered run)."""
    b = args.bucket
    if b is None:
        return None
    return (int(b),) if isinstance(b, int) else tuple(int(k) for k in b)


def bucket_arg(args) -> int | tuple[int, ...] | None:
    """What the reader takes: one bucket as an int (as before D-227), two as a tuple."""
    ks = buckets_of(args)
    return ks if ks is None or len(ks) > 1 else ks[0]


def bucket_tag(args) -> str:
    ks = buckets_of(args)
    return f"bucket{ks[0]}" if len(ks) == 1 else "buckets" + "_".join(map(str, ks))


def replay_name(args) -> str:
    if args.bucket is not None:
        return f"replay_dry_{bucket_tag(args)}.json"
    return f"replay_{args.speed}.json"


def set_aside(table: Path, reason: str, record: dict):
    """A dry run's labelled rerun (D-227): rename the earlier report and table with the
    earlier report's git head as a suffix, never overwriting, and note what was done."""

    def run(path: Path) -> None:
        if not path.exists():
            raise SystemExit(f"--rerun-reason given but {path.name} does not exist")
        head = json.loads(path.read_text()).get("git_head")
        if not head:
            raise SystemExit(f"{path.name} has no git_head; set it aside by hand")
        moves = [(p, p.with_name(f"{p.stem}_{head}{p.suffix}")) for p in (path, table)]
        for _, new in moves:
            if new.exists():
                raise SystemExit(f"{new} exists; nothing renamed")
        for old, new in moves:
            if old.exists():
                old.rename(new)
        record.update(
            reason=reason,
            earlier_git_head=head,
            earlier_report=moves[0][1].name,
            earlier_table=moves[1][1].name if moves[1][1].exists() else None,
        )

    return run


def check_replay_args(args) -> None:
    dbs = load_serving()["step6"]["databases"]
    if args.db == int(dbs["loaded_store"]):
        raise SystemExit("database 0 holds the loaded store; step 6 never writes it")
    want = int(dbs["determinism_replay"] if args.speed == "throttled" else dbs["replay"])
    ks = buckets_of(args)
    if ks is not None and args.speed != "full":
        raise SystemExit("the dry run replays one bucket at full speed")
    if ks is not None and not (1 <= len(ks) <= MAX_DRY_BUCKETS and len(set(ks)) == len(ks)):
        raise SystemExit("a dry run replays one bucket or two different buckets")
    if getattr(args, "rerun_reason", None) is not None:
        if ks is None:
            raise SystemExit("only a dry run is rerun; a registered run runs once")
        if not args.rerun_reason.strip():
            raise SystemExit("--rerun-reason needs a reason")
    if args.db != want:
        raise SystemExit(f"a {args.speed}-speed replay goes to database {want}")


def table_path(args) -> Path:
    registered = Path(load_serving()["step6"]["outputs"]["stream_features"])
    if args.bucket is not None:
        return registered.with_name(f"{registered.stem}_{bucket_tag(args)}.parquet")
    if args.speed == "throttled":
        return registered.with_name(f"{registered.stem}_throttled.parquet")
    return registered


def gather_replay(args) -> dict:
    """Everything a replay reads besides its sources: the client, the label settings, the
    models through the registry alias, the store columns, the scored set, the paths."""
    from retention.api import registry
    from retention.explain.config import load_explain_config
    from retention.explain.model import load_settings
    from retention.labels.config import load_label_config
    from retention.models.__main__ import processed_dir
    from retention.models.protocol import load_protocol
    from retention.optimize.value import EXTRA_COLUMNS

    protocol = load_protocol()
    settings = load_settings(horizons=(7, 30, 59))
    with tempfile.TemporaryDirectory() as tmp:
        models = registry.load_serving_models(
            registry.registry_settings(Path.cwd()), settings, protocol, Path(tmp)
        )
    explain = load_explain_config(settings.columns)
    facts = sorted({c for cs in explain.facts.values() for c in cs} | {"tenure_days"})
    columns = list(dict.fromkeys([*settings.columns, *EXTRA_COLUMNS, *facts]))
    scored = scored_at(source.BOOTSTRAP, args.fraction)
    processed = processed_dir(args.fraction)
    return {
        "r": producer.client_for(args.db),
        "folder": interim(args.fraction),
        "cfg": load_label_config(),
        "members": processed / "members.parquet",
        "cleaned": processed / "transactions.parquet",
        "is_scored": scored.__contains__,
        "c01": models.c01,
        "models_version": models.version,
        "seed": protocol.seed,
        "columns": columns,
        "main": tuple(protocol.main_segments),
        "sample_buckets": to_percent_of(args.fraction),
    }


def to_percent_of(fraction) -> int:
    from retention.data.sample import resolve_fraction, to_percent

    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())["data"]
    return to_percent(resolve_fraction(fraction, raw))


def replay_cmd(args, gather=gather_replay) -> int:
    check_replay_args(args)
    reason = getattr(args, "rerun_reason", None)
    rerun: dict = {}
    aside = set_aside(table_path(args), reason, rerun) if reason is not None else None
    repo, path, t0 = begin(args, replay_name(args), aside)
    out = api_cli.out_dir(args.fraction)
    if args.speed == "throttled":
        full = out / "replay_full.json"
        if not full.exists():
            raise SystemExit("the throttled replay runs after the full-speed one (step6.run_order)")
    g = gather(args)
    report, result = replay.run(
        g["r"], args.db, g["folder"], g["cfg"], g["members"], g["is_scored"], g["c01"],
        g["seed"], g["columns"], g["main"], table_path(args),
        throttled=args.speed == "throttled", bucket=bucket_arg(args),
        keep_link=args.bucket is not None,
    )  # fmt: skip
    report["models_version"] = g["models_version"]
    code = 0
    if args.bucket is not None:
        n = len(buckets_of(args))
        report["not_a_mark"] = f"dry run on {'one md5 bucket' if n == 1 else f'{n} md5 buckets'}"
        if rerun:
            report["rerun"] = rerun
        report["projection"] = replay.project(report, factor=g["sample_buckets"] / n)
        report["source_row_link"] = replay.link_check(
            g["cleaned"], result.link_rows, source.STORE_DATE, bucket_arg(args)
        )
        code = {"proceed": 0, "stop_and_ask": 2, "stop": 1}[report["projection"]["verdict"]]
        code = code or (0 if report["source_row_link"]["met"] else 1)
    elif args.speed == "full":
        recount = out / RECOUNT
        if not recount.exists():
            raise SystemExit("the recount runs before the replay is judged (step6.run_order)")
        expected = json.loads(recount.read_text())["total"]
        report["event_count"] = {
            "recount": expected,
            "registered": replay.EXPECTED_EVENTS,
            "produced": report["produced"],
            "consumed": report["consumed"],
            "met": report["produced"] == report["consumed"] == expected == replay.EXPECTED_EVENTS,
        }
        code = 0 if report["event_count"]["met"] else 1
    else:
        full = json.loads((out / "replay_full.json").read_text())
        report["determinism"] = {
            "full_speed_sha256": full["table_sha256"],
            "throttled_sha256": report["table_sha256"],
            "met": full["table_sha256"] == report["table_sha256"],
        }
        code = 0 if report["determinism"]["met"] else 1
    finish(report, repo, path, t0)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.stream")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("stream-parity", help="judge feature and score parity of the replay")
    sp.add_argument("--fraction", type=float, default=None)
    sp.add_argument("--db", type=int, required=True)
    rc = sub.add_parser("recount", help="count October's source rows (event_count)")
    rc.add_argument("--fraction", type=float, default=None)
    ci = sub.add_parser("cancel-ids", help="counts only: never-used words in cancel-event IDs")
    ci.add_argument("--fraction", type=float, default=None)
    rp = sub.add_parser("replay", help="one replay into a replay database")
    rp.add_argument("--fraction", type=float, default=None)
    rp.add_argument("--db", type=int, required=True)
    rp.add_argument("--speed", choices=["full", "throttled"], required=True)
    rp.add_argument("--bucket", type=int, action="append", default=None)
    rp.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    if args.command == "recount":
        return recount_cmd(args)
    if args.command == "cancel-ids":
        return cancel_ids_cmd(args)
    if args.command == "replay":
        return replay_cmd(args)
    return stream_parity_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
