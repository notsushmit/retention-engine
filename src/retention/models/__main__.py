"""Command line entry: `python -m retention.models <command>`.

Commands: protocol-check, label-check, guard-dry-run, build-labels, baselines, classifier,
survival, calibration, freeze, score-test,
check-month (descriptive analysis of 2016-11; decides nothing).

protocol-check loads configs/evaluation.yaml (which runs the purge-rule and
locked-month checks), then counts snapshot membership for the fold, fit,
calibration and check months: P2 main rows, the October customer halves,
customers in both halves of a month (must be 0), and each half's overlap with
the fit months and the check month. It reads no outcome column and no test month.

label-check builds labels at the earliest fully observed as_of (T + h + 30) for
each 2016 snapshot and horizon in `label_as_of_check`, and the fit snapshots'
survival labels at the test pipeline's as_of, and compares them with the final
labels.parquet. Any differing row is reported and the command exits with an
error. --first-only runs the first build alone, to measure its memory.

build-labels builds one cached label set (labels as of a split date for the
snapshots that split's fit and calibration roles need). It is the only command
that loads the label-building code; model commands read the cache and refuse
to build.

baselines runs the simple baselines on the folds (a model process).

guard-dry-run runs every precondition of the test session against this
repository and prints each check's status; it records nothing and unlocks
nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

from retention.data.sample import resolve_fraction, sample_dir
from retention.models.data import labels_as_of, load_labels, membership
from retention.models.protocol import EVAL_CONFIG, Protocol, load_protocol
from retention.models.resources import peak_rss_gb


def processed_dir(fraction: float | None) -> Path:
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())
    base = Path(raw["labels"]["processed_dir"])
    return sample_dir(base, resolve_fraction(fraction, raw["data"]))


def interim_dir(fraction: float | None) -> Path:
    raw = yaml.safe_load(Path("configs/kkbox.yaml").read_text())
    return sample_dir(Path(raw["data"]["interim_dir"]), resolve_fraction(fraction, raw["data"]))


PRE_T_COMPARED = ["expiry_date", "days_until_expiry", "in_exp30", "segment", "paid_state"]


def label_check_builds(protocol: Protocol) -> list[tuple[date, tuple[date, ...], list[str]]]:
    """(as_of, snapshots, outcome columns to compare) for every build of the check."""
    spec = yaml.safe_load(EVAL_CONFIG.read_text())["label_as_of_check"]
    builds = []
    for t in spec["snapshots"]:
        for h in spec["horizons"]:
            as_of = t + timedelta(days=h + protocol.purge_days)
            builds.append((as_of, (t,), [f"churned_{h}" if h in (7, 30) else f"lapse_by_{h}"]))
    tf = protocol.test_fit
    builds.append((tf.as_of, tf.fit, ["event_observed", "days_to_event"]))
    return builds


def _derive(df, columns: list[str]):
    for c in columns:
        if c.startswith("lapse_by_"):
            h = int(c.removeprefix("lapse_by_"))
            df[c] = ((df.event_observed == 1) & (df.days_to_event <= h)).astype("int8")
    return df


def label_check(
    processed: Path, cache: Path, protocol: Protocol, first_only: bool
) -> tuple[dict, int]:
    final = processed / "labels.parquet"
    rows, differing = [], 0
    for as_of, snaps, outcomes in label_check_builds(protocol):
        t0 = time.time()
        built = labels_as_of(processed, cache, as_of, snaps, protocol)
        raw_cols = [c for c in outcomes if not c.startswith("lapse_by_")]
        if len(raw_cols) < len(outcomes):
            raw_cols += ["event_observed", "days_to_event"]
        cols = list(dict.fromkeys([*PRE_T_COMPARED, *raw_cols]))
        early = _derive(load_labels(built, snaps, cols, protocol), outcomes)
        late = _derive(load_labels(final, snaps, cols, protocol), outcomes)
        keys = ["msno", "snapshot_date"]
        m = early.merge(late, on=keys, how="outer", suffixes=("_early", "_final"), indicator=True)
        diff = {"rows_only_in_one": int((m._merge != "both").sum())}
        both = m[m._merge == "both"]
        for c in [*PRE_T_COMPARED, *outcomes]:
            a, b = both[f"{c}_early"], both[f"{c}_final"]
            diff[c] = int((~((a == b).fillna(False) | (a.isna() & b.isna()))).sum())
        n_diff = sum(diff.values())
        differing += n_diff
        rows.append(
            {
                "as_of": str(as_of),
                "snapshots": [str(t) for t in snaps],
                "rows": len(early),
                "events": {c: int(early[c].sum()) for c in outcomes if c != "days_to_event"},
                "differing": diff,
                "seconds": round(time.time() - t0, 1),
                "peak_rss_gb_so_far": peak_rss_gb(),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        if first_only:
            break
    return {"builds": rows, "differing_rows_total": differing}, differing


def protocol_check(labels_path: Path, protocol: Protocol) -> dict:
    fit = list(protocol.test_fit.fit)
    oct_ = protocol.halves_snapshot
    check = protocol.check_month
    months = sorted({*fit, oct_, check})
    mem = membership(labels_path, months, protocol)
    p2 = mem[mem.in_exp30 & mem.segment.isin(protocol.main_segments)]

    # The half depends only on the customer ID, so both counts must be 0: a
    # customer in both halves within one month, or in different halves in two months.
    within = mem.groupby(["snapshot_date", "msno"]).half.nunique()
    across = mem.groupby("msno").half.nunique()
    oct_p2 = p2[p2.snapshot_date.dt.date == oct_]
    in_fit = set(mem.msno[mem.snapshot_date.dt.date.isin(fit)])
    in_check = set(mem.msno[mem.snapshot_date.dt.date == check])
    halves = {}
    for half, g in oct_p2.groupby("half"):
        ids = set(g.msno)
        halves[half] = {
            "p2_main_rows": len(g),
            "customers": len(ids),
            "also_in_fit_months": len(ids & in_fit),
            "also_in_check_month": len(ids & in_check),
        }
    return {
        "p2_main_rows": {str(d): int((p2.snapshot_date.dt.date == d).sum()) for d in months},
        "customers_in_both_halves_within_a_month": int((within > 1).sum()),
        "customers_in_different_halves_across_months": int((across > 1).sum()),
        "months_checked_for_both_halves": [str(d) for d in months],
        "october_halves": halves,
    }


def _plain(obj):
    """JSON-ready copy: numpy scalars to Python, dates and other objects to strings."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _headline_metrics(results: dict) -> dict[str, float]:
    out = {}
    for cid, data in results.get("primary", {}).items():
        if data.get("status") != "ok":
            continue
        tag = cid.replace(" days, ", "d.")
        for comp, v in data["comparisons"].items():
            if v["status"] == "ok":
                key = comp.replace(" minus ", "_vs_")
                out |= {
                    f"{tag}.{key}.d": v["point"],
                    f"{tag}.{key}.lo": v["lo"],
                    f"{tag}.{key}.hi": v["hi"],
                }
        sm = data["models"]["survival_dt01"]
        for k in ("slope", "intercept"):
            if sm[k]["status"] == "ok":
                out[f"{tag}.survival_dt01.{k}"] = sm[k]["point"]
    return out


def score_test(args) -> None:
    from retention.models.bootstrap import use_workers
    from retention.models.resources import peak_rss_children_gb
    from retention.models.stages import assert_no_label_building_code
    from retention.models.test_stage import MEMORY_CEILING_GB, run_test_stage

    t0 = time.time()
    protocol = load_protocol()
    use_workers(protocol.threads.n)
    raw = yaml.safe_load(EVAL_CONFIG.read_text())
    reps = args.replicates or raw["bootstrap"]["replicates"]
    out_dir = processed_dir(args.fraction)
    cache = interim_dir(args.fraction) / "labels_asof"

    if args.rehearsal:
        suffix = "" if args.buckets is None else f"_buckets{args.buckets}"
        # Completed cells, so a stopped rehearsal resumes; never printed or logged.
        checkpoint = interim_dir(args.fraction) / f"rehearsal_checkpoint{suffix}.json"
        results = run_test_stage(
            protocol,
            out_dir,
            cache,
            reps,
            rehearsal=True,
            buckets=args.buckets,
            checkpoint=checkpoint,
        )
        assert_no_label_building_code()
        results["peak_rss_gb"] = peak_rss_gb()
        results["peak_rss_workers_gb"] = peak_rss_children_gb()
        (out_dir / f"rehearsal{suffix}.json").write_text(
            json.dumps(_plain(results), indent=2) + "\n"
        )
        checkpoint.unlink(missing_ok=True)
        print("REHEARSAL (2016 stand-in months; no unlock; nothing logged to MLflow)")
        print("stand-in months:", results["rehearsal_months"])
        for c in results["cells"]:
            print(f"  {c['section']:28s} {c['cell']:26s} {c['status']:18s} {c['reason']}")
        print("goal cells:")
        for c in results["goal_cells"]:
            computed = c["status"] != "missing"
            print(
                f"  {c['kind']:14s} {c['horizon']:>2} days {c['month']}  {c['against']:26s}", end=""
            )
            print(f" computed: {computed}")
        print("what the rehearsal cannot show:")
        for line in results["rehearsal_limits"]:
            print("  -", line)
        print(
            json.dumps(
                {
                    "seconds": round(time.time() - t0, 1),
                    "peak_rss_gb": results["peak_rss_gb"],
                    "peak_rss_workers_gb": results["peak_rss_workers_gb"],
                    "ceiling_gb": MEMORY_CEILING_GB,
                    "registered_limit_gb": 6,
                }
            )
        )
        return

    from mlflow.tracking import MlflowClient

    from retention.models.guard import open_test_session
    from retention.models.tracking import setup

    tr = setup(Path.cwd())
    unlock, run_id = open_test_session(tr, args.rerun_reason)
    client = MlflowClient(tr.uri)
    try:
        results = run_test_stage(protocol, out_dir, cache, reps, unlock=unlock)
        assert_no_label_building_code()
        results["peak_rss_gb"] = peak_rss_gb()
        results["peak_rss_workers_gb"] = peak_rss_children_gb()
        results["seconds"] = round(time.time() - t0, 1)
        plain = _plain(results)
        client.log_dict(run_id, plain, "test_results.json")
        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        metrics = [Metric(k, float(v), now, 0) for k, v in _headline_metrics(plain).items()]
        for i in range(0, len(metrics), 500):
            client.log_batch(run_id, metrics=metrics[i : i + 500])
        client.set_tag(run_id, "timing.seconds", str(results["seconds"]))
        client.set_tag(run_id, "timing.peak_rss_gb", str(results["peak_rss_gb"]))
        client.set_terminated(run_id, "FINISHED")
    except BaseException:
        client.set_terminated(run_id, "FAILED")
        raise
    print(
        json.dumps(
            {
                "test_run": run_id,
                "seconds": results["seconds"],
                "peak_rss_gb": results["peak_rss_gb"],
                "peak_rss_workers_gb": results["peak_rss_workers_gb"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.models")
    sub = parser.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("protocol-check", help="validate the protocol and count membership")
    pc.add_argument("--fraction", type=float, default=None)
    lc = sub.add_parser("label-check", help="labels as of each split date vs the final labels")
    lc.add_argument("--fraction", type=float, default=None)
    lc.add_argument("--first-only", action="store_true", help="run only the first build")
    sub.add_parser("guard-dry-run", help="check the test-session preconditions; unlock nothing")
    bl = sub.add_parser("build-labels", help="build one cached label set in this process")
    bl.add_argument("--split", required=True, help="a fold name (for example f7a) or 'test'")
    bl.add_argument("--fraction", type=float, default=None)
    for name, text in (
        ("baselines", "simple baselines on the folds"),
        ("classifier", "LightGBM classifier stages 1 and 2 on the folds"),
        ("survival", "Cox, XGBoost AFT and discrete-time candidates on the folds"),
        ("calibration", "calibration on f7b_cal and the October halves (2016 only)"),
    ):
        stage = sub.add_parser(name, help=text)
        stage.add_argument("--fraction", type=float, default=None)
        stage.add_argument("--replicates", type=int, default=None)
    cm = sub.add_parser("check-month", help="descriptive analysis of the check month 2016-11")
    cm.add_argument("--fraction", type=float, default=None)
    cm.add_argument("--replicates", type=int, default=None)
    st = sub.add_parser("score-test", help="the test stage (guarded), or its 2016 rehearsal")
    st.add_argument("--rehearsal", action="store_true", help="2016 stand-in months; no unlock")
    st.add_argument("--buckets", type=int, default=None, help="rehearsal on a customer subset")
    st.add_argument("--replicates", type=int, default=None)
    st.add_argument("--rerun-reason", default=None, help="only to repeat a crashed test run")
    st.add_argument("--fraction", type=float, default=None)
    fz = sub.add_parser("freeze", help="write configs/choices.yaml from the MLflow runs")
    fz.add_argument(
        "--preview", action="store_true", help="print the file and checks; write nothing"
    )
    args = parser.parse_args()

    if args.command == "score-test":
        score_test(args)
        return

    if args.command == "freeze":
        from retention.models.freeze_choices import assemble, preview
        from retention.models.guard import CHOICES, freeze
        from retention.models.tracking import setup

        tr = setup(Path.cwd())
        choices, run_ids, checks = assemble(tr)
        for name, ok, detail in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ({detail})"))
        if not all(ok for _, ok, _ in checks):
            sys.exit(1)
        if args.preview:
            print(preview(tr, choices, run_ids))
            return
        run_id = freeze(tr, choices, run_ids)
        print(f"wrote {CHOICES} (freeze run {run_id}); commit it alone")
        return

    if args.command == "check-month":
        import mlflow

        from retention.models.bootstrap import use_workers
        from retention.models.check_month import check_month
        from retention.models.stages import assert_no_label_building_code
        from retention.models.tracking import setup, start

        t0 = time.time()
        protocol = load_protocol()
        use_workers(protocol.threads.n)  # identical intervals for any number of workers
        raw = yaml.safe_load(EVAL_CONFIG.read_text())
        reps = args.replicates or raw["bootstrap"]["replicates"]
        out_dir = processed_dir(args.fraction)
        tr = setup(Path.cwd())
        with start(
            tr,
            "check_month",
            "check_month-2016-11-lightgbm-c01-h7-fs1",
            {"month": "2016-11-01", "decides": "nothing", "replicates": reps},
            data_files={
                "labels": out_dir / "labels.parquet",
                "features": out_dir / "features.parquet",
            },
        ):
            report = check_month(
                protocol, out_dir, interim_dir(args.fraction) / "labels_asof", reps
            )
            mlflow.log_dict(json.loads(json.dumps(report, default=str)), "check_month.json")
        assert_no_label_building_code()
        report["seconds"] = round(time.time() - t0, 1)
        report["peak_rss_gb"] = peak_rss_gb()
        (out_dir / "check_month.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({k: report[k] for k in ("seconds", "peak_rss_gb")}))
        return

    if args.command == "build-labels":
        from retention.models.data import fit_label_requests

        t0 = time.time()
        protocol = load_protocol()
        ((name, as_of, snaps),) = [r for r in fit_label_requests(protocol) if r[0] == args.split]
        path = labels_as_of(
            processed_dir(args.fraction),
            interim_dir(args.fraction) / "labels_asof",
            as_of,
            snaps,
            protocol,
        )
        print(
            json.dumps(
                {
                    "split": name,
                    "as_of": str(as_of),
                    "snapshots": [str(t) for t in snaps],
                    "file": path.name,
                    "seconds": round(time.time() - t0, 1),
                    "peak_rss_gb": peak_rss_gb(),
                }
            )
        )
        return

    if args.command in ("baselines", "classifier", "survival", "calibration"):
        from retention.models import stages
        from retention.models.bootstrap import use_workers
        from retention.models.tracking import setup

        t0 = time.time()
        protocol = load_protocol()
        use_workers(protocol.threads.n)  # identical intervals for any number of workers
        raw = yaml.safe_load(EVAL_CONFIG.read_text())
        reps = args.replicates or raw["bootstrap"]["replicates"]
        out_dir = processed_dir(args.fraction)
        from retention.models.calibration_stage import calibration_stage
        from retention.models.survival_stage import survival_stage

        stage = {
            "baselines": stages.baseline_stage,
            "classifier": stages.classifier_stage,
            "survival": survival_stage,
            "calibration": calibration_stage,
        }
        report = stage[args.command](
            protocol, out_dir, interim_dir(args.fraction) / "labels_asof", setup(Path.cwd()), reps
        )
        report["seconds"] = round(time.time() - t0, 1)
        report["peak_rss_gb"] = peak_rss_gb()
        name = {
            "baselines": "baselines_folds.json",
            "classifier": "classifier_folds.json",
            "survival": "survival_folds.json",
            "calibration": "calibration_folds.json",
        }
        (out_dir / name[args.command]).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({k: report[k] for k in ("seconds", "peak_rss_gb")}))
        return

    if args.command == "guard-dry-run":
        from retention.models.guard import dry_run
        from retention.models.tracking import setup

        results = dry_run(setup(Path.cwd()))
        for r in results:
            print(f"check {r.check} {r.name:32s} {r.status:14s} {r.message}")
        sys.exit(1 if any(r.status == "failed" for r in results) else 0)

    t0 = time.time()
    out_dir = processed_dir(args.fraction)
    protocol = load_protocol()
    status = 0
    if args.command == "protocol-check":
        name = "protocol_check.json"
        report = protocol_check(out_dir / "labels.parquet", protocol)
    else:
        name = "label_check_first.json" if args.first_only else "label_check.json"
        cache = interim_dir(args.fraction) / "labels_asof"
        report, differing = label_check(out_dir, cache, protocol, args.first_only)
        status = 1 if differing else 0
    report["seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = peak_rss_gb()
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "builds"}, indent=2))
    sys.exit(status)


if __name__ == "__main__":
    main()
