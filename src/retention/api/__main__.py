"""Command line entry: `python -m retention.api <command>`.

refit59: refit the 59-day classifier twice, judge it by the registered pass marks
(configs/serving.yaml, ordering) and write serving/refit59_report.json. Nothing
else is written. Runs once; exits with an error if any mark fails.

refit59 --save --expect-sha 59=<hex>: the same refit and marks in a new process;
the booster and its Platt parameters are saved only if every mark passes and the
booster's SHA-256 equals the expected one. Writes refit59_report_save.json. Runs
once.

calibration-table: the descriptive slope and intercept at 7 and 30 days before
and after the fix, on the selection half of 2016-10-01 (2016 outcomes only), with
the saved 59-day booster. Writes calibration_before_after.json. Runs once.

save-dt01 --expect-sha <hex>: refit dt01 exactly as Phase 7 did, judge it by Phase
7's registered reproduction cells (configs/actions.yaml, dt01_refit) and the
expected booster SHA-256 (which Phase 7's saved dt01.txt must also have), then
write dt01.txt and model_dt01.json to the service's model folder. Writes
save_dt01_report.json. Runs once; saves nothing if anything fails.

register: put the eight-file bundle (configs/serving.yaml, registry) in the MLflow
registry as retention-scoring version 1 under the alias serving, after checking
every file; then load it back through the alias with every check. Writes
register_report.json. Runs once; refused if the model is already registered.

verify-serving: load the models through the alias exactly as the service will
(download, hash every file, check the manifest, parse). Writes
verify_serving_report.json. Runs once.

score-parity: judge every registered step 4 mark exactly, on 2016-10-01 and
2016-11-01, with the models loaded through the registry alias (retention.api.parity).
Stops on a population count that differs from the registered one or on a customer
without a usable price who has an allowed action. Writes step4_report.json. Runs
once; exits with an error if any mark is not met.

load-snapshot --snapshot <date>: replace the Valkey store with one snapshot from the
batch tables: every customer with coverage, the known customers, the ranking by
final probability at 7, 30 and 59 days (models through the registry alias) and
the /at-risk header counts; the store's date is written last. Writes
load_snapshot_<date>.json. Runs once per snapshot.

score-parity --dry-run: the same comparisons on one md5 bucket of 2016-11-01's
customers, with no registered full-count mark judged and no stop rule stopping.
Writes only step4_dry_run.json, refuses to overwrite it, and never reads or writes
step4_report.json (the run-once guard of the real run is untouched).

smoke --snapshot <date>: the step 5f smoke check of the loaded store (retention.api.smoke):
the service started in-process on the store, each endpoint called once, the
answers compared with what they should be, the index build and the readiness hash
check timed. Not a mark; its numbers are never quoted as parity or latency. Writes
smoke_<date>.json and nothing else; clean tree; runs once.

write-docs: write docs/scoring_api.md from configs/serving.yaml and the response
schemas (retention.api.docs); nothing else is written. It reads no data and is
not a run, so it does not need a clean tree.

Every other command refuses to run when a tracked file differs from HEAD, so each
report ties to a commit.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd

from retention.api import dt01 as dt01_step
from retention.api import parity, registry
from retention.api.batch import snapshot_inputs
from retention.api.config import SERVING_CONFIG, file_sha256, load_serving, ordering_marks
from retention.api.ordering import platt_from
from retention.api.refit import (
    HORIZON,
    calibration_table,
    run_check,
    saved_models,
    selection_rows,
)
from retention.explain.contributions import load_saved_model
from retention.explain.model import load_settings, save_models
from retention.models.__main__ import interim_dir, processed_dir
from retention.models.protocol import load_protocol
from retention.models.resources import peak_rss_gb
from retention.models.stages import assert_no_label_building_code
from retention.models.tracking import head_commit, tree_dirty
from retention.optimize.config import load_actions_config
from retention.optimize.value import dt01_settings

REPORT = "refit59_report.json"
REPORT_SAVE = "refit59_report_save.json"
TABLE = "calibration_before_after.json"
DT01_REPORT = "save_dt01_report.json"
REGISTER_REPORT = "register_report.json"
VERIFY_REPORT = "verify_serving_report.json"
STEP4_REPORT = "step4_report.json"
DRY_REPORT = "step4_dry_run.json"


def out_dir(fraction: float | None) -> Path:
    return processed_dir(fraction) / "serving"


def model_dir(fraction: float | None) -> Path:
    return interim_dir(fraction) / "serving" / "model"


def require_clean(repo: Path) -> None:
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the run ties to code")


def run_once(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: this step runs once and is never rerun")


def parse_expected(values: list[str]) -> str:
    """The one HORIZON=SHA256 pair for 59 days."""
    pairs = dict(v.partition("=")[::2] for v in values)
    if list(pairs) != [str(HORIZON)] or not pairs[str(HORIZON)].strip():
        raise SystemExit(f"--expect-sha takes exactly {HORIZON}=SHA256, got {values!r}")
    return pairs[str(HORIZON)].strip()


def stamp(report: dict, repo: Path, t0: float) -> dict:
    return report | {
        "git_head": head_commit(repo),
        "tree_clean": True,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
    }


def refit59(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    report_path = out / (REPORT_SAVE if args.save else REPORT)
    run_once(report_path)
    expected = parse_expected(args.expect_sha) if args.save else None
    t0 = time.time()
    protocol = load_protocol()
    marks = ordering_marks(load_serving())
    settings = load_settings(horizons=(7, 30, HORIZON))
    result = run_check(
        marks,
        protocol,
        processed_dir(args.fraction),
        interim_dir(args.fraction) / "labels_asof",
        interim_dir(args.fraction) / "explain" / "model",
        settings,
    )
    assert_no_label_building_code()
    if expected is not None and result.report["refit"]["booster_sha256"] != expected:
        result.report["failures"].append("booster SHA-256 differs from --expect-sha")
        result.report["passed"] = False
    report = stamp(result.report, repo, t0) | {"save": bool(args.save)}
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("REFIT CHECK NOT MET; nothing saved:", *report["failures"], sep="\n  ")
        return 1
    if args.save:
        written = save_models(
            model_dir(args.fraction), {HORIZON: result.refit}, {HORIZON: expected}
        )
        print("saved:", *[str(p) for p in written], sep="\n  ")
    return 0


def table(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / TABLE)
    saved_report = json.loads((out / REPORT_SAVE).read_text())
    if not saved_report.get("passed"):
        raise SystemExit(f"{REPORT_SAVE} did not pass; no 59-day booster to use")
    t0 = time.time()
    protocol = load_protocol()
    settings = load_settings(horizons=(7, 30, HORIZON))
    explain_models = saved_models(
        interim_dir(args.fraction) / "explain" / "model", settings, protocol
    )
    m59 = load_saved_model(
        model_dir(args.fraction),
        HORIZON,
        settings,
        protocol,
        saved_report["refit"]["booster_sha256"],
    )
    rows = selection_rows(
        protocol,
        processed_dir(args.fraction),
        interim_dir(args.fraction) / "labels_asof",
        settings.columns,
    )
    models = [m.model for m in (*explain_models, m59)]
    cals = [platt_from(m.platt_a, m.platt_b, protocol.seed) for m in (*explain_models, m59)]
    result = calibration_table(rows, models, cals)
    assert_no_label_building_code()
    result = stamp(result | {"descriptive": True, "pass_mark": None}, repo, t0)
    out.mkdir(parents=True, exist_ok=True)
    (out / TABLE).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def save_dt01(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / DT01_REPORT)
    expected = args.expect_sha.strip()
    t0 = time.time()
    protocol = load_protocol()
    cfg = load_actions_config()
    s = dt01_settings()
    report, model = dt01_step.run_save_check(
        protocol,
        processed_dir(args.fraction),
        interim_dir(args.fraction) / "labels_asof",
        s,
        cfg,
        expected,
        interim_dir(args.fraction) / "optimize" / "model" / "dt01.txt",
    )
    assert_no_label_building_code()
    report = stamp(report, repo, t0)
    if report["peak_rss_gb"] >= cfg.memory_ceiling_gb:
        report["failures"].append(
            f"peak RSS {report['peak_rss_gb']} GB at or above {cfg.memory_ceiling_gb} GB"
        )
        report["passed"] = False
    if report["passed"]:
        report["saved"] = dt01_step.save(model, s, model_dir(args.fraction), expected)
    out.mkdir(parents=True, exist_ok=True)
    (out / DT01_REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("SAVE-DT01 NOT MET; nothing saved:", *report["failures"], sep="\n  ")
        return 1
    return 0


def loaded_summary(models: registry.ServingModels) -> dict:
    return {
        "version": models.version,
        "manifest": models.manifest,
        "downloaded_files_sha256": dict(sorted(models.files.items())),
        "c01_booster_sha256": {str(h): m.sha256 for h, m in models.c01.items()},
        "c01_platt": {str(h): [m.platt_a, m.platt_b] for h, m in models.c01.items()},
        "dt01_booster_sha256": dt01_step.text_sha256(dt01_step.booster_text(models.dt01)),
    }


def register_cmd(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / REGISTER_REPORT)
    t0 = time.time()
    settings = registry.registry_settings(repo)
    sources = registry.bundle_sources(settings)
    protocol = load_protocol()
    with tempfile.TemporaryDirectory() as tmp:
        done = registry.register(
            settings,
            sources,
            Path(tmp) / "bundle",
            head_commit(repo),
            file_sha256(SERVING_CONFIG),
        )
        models = registry.load_serving_models(
            settings, load_settings(horizons=(7, 30, HORIZON)), protocol, Path(tmp) / "load"
        )
    report = stamp(
        {
            "registered_model": settings.name,
            "alias": settings.alias,
            "run_id": done["run_id"],
            "registered_version": done["version"],
            "loaded_back": loaded_summary(models),
            "passed": True,
        },
        repo,
        t0,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / REGISTER_REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def verify_cmd(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / VERIFY_REPORT)
    t0 = time.time()
    settings = registry.registry_settings(repo)
    with tempfile.TemporaryDirectory() as tmp:
        models = registry.load_serving_models(
            settings, load_settings(horizons=(7, 30, HORIZON)), load_protocol(), Path(tmp)
        )
    report = stamp(
        {
            "registered_model": settings.name,
            "alias": settings.alias,
            "loaded": loaded_summary(models),
            "passed": True,
        },
        repo,
        t0,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / VERIFY_REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def parity_report(processed: Path, repo: Path, dry: bool) -> dict:
    """The score-parity report (the real run or the dry run); writes nothing."""
    from retention.explain.config import load_explain_config
    from retention.explain.reasons import PHRASES, Renderer
    from retention.optimize.config import load_actions_config
    from retention.optimize.value import EXTRA_COLUMNS

    serving = load_serving()
    step4 = serving["step4"]
    report: dict = {
        "dry_run": dry,
        "original_reasons_parity": step4["amendments"][0]["original_mark"],
        "self_check_scope": (
            "the self-check shows only that the masking matches Phase 4's own "
            "missing-tenure behaviour; it does not validate the wording"
        ),
    }
    try:
        parity.check_pinned(
            processed / "explain" / "reasons.parquet",
            step4["more_pass_marks"]["reasons_parity"]["against_sha256"],
        )
        parity.check_pinned(
            processed / "optimize" / "value.parquet",
            step4["pass_marks"]["paid_days_parity"]["against_sha256"],
        )
        parity.check_pinned(
            processed / "rehearsal.json",
            serving["ordering"]["pass_marks"]["reproduction"]["source_sha256"],
        )
        protocol = load_protocol()
        settings = load_settings(horizons=(7, 30, HORIZON))
        cfg = load_explain_config(settings.columns)
        reasons, value = parity.read_inputs(processed)
        pop = step4["more_pass_marks"]["population_counts"]
        with tempfile.TemporaryDirectory() as tmp:
            models = registry.load_serving_models(
                registry.registry_settings(repo), settings, protocol, Path(tmp)
            )
        report["models"] = {
            "version": models.version,
            "downloaded_files_sha256": dict(sorted(models.files.items())),
        }
        ctx = parity.Context(
            models=models,
            explain_cfg=cfg,
            renderer=Renderer.from_files(cfg),
            phrases_sha256=file_sha256(PHRASES),
            columns=settings.columns,
            actions_cfg=load_actions_config(),
            protocol=protocol,
            reasons=reasons,
            value=value,
            rehearsal=json.loads((processed / "rehearsal.json").read_text()),
            serving=serving,
            population_expected=dict(pop["expected"]),
            population_snapshot=date.fromisoformat(pop["snapshot"]),
        )
        cols = list(dict.fromkeys([*settings.columns, *EXTRA_COLUMNS]))
        facts = sorted({c for cs in cfg.facts.values() for c in cs} | {"tenure_days"})
        if dry:
            t = parity.DRY_SNAPSHOT
            inputs = parity.dry_subset(snapshot_inputs(processed, t, cols, facts, protocol))
            report["dry"] = parity.dry_summary(parity.snapshot_marks(ctx, inputs, dry=True))
            report["passed"] = report["dry"]["all_comparisons_equal"]
            return report
        results = {}
        for t in parity.SNAPSHOTS:
            inputs = snapshot_inputs(processed, t, cols, facts, protocol)
            results[t] = parity.snapshot_marks(ctx, inputs)
        nov = ctx.population_snapshot
        reload = parity.reload_marks(ctx, results[nov]["_rows"])
        report["marks"] = parity.assemble_marks(results, reload, serving)
        report["descriptive"] = parity.descriptive(results)
        report["not_met"] = parity.overall(report)
        report["passed"] = not report["not_met"]
    except parity.StopRun as e:
        report |= {"stopped": str(e), "passed": False}
    return report


def score_parity(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    target = out / (DRY_REPORT if args.dry_run else STEP4_REPORT)
    if args.dry_run:
        if target.exists():
            raise SystemExit(f"{target} exists: the dry run refuses to overwrite its report")
    else:
        run_once(target)
    t0 = time.time()
    report = parity_report(processed_dir(args.fraction), repo, args.dry_run)
    assert_no_label_building_code()
    report = stamp(parity.json_ready(report), repo, t0)
    out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    keep = ("marks", "descriptive", "dry")
    print(json.dumps({k: report[k] for k in report if k not in keep}, indent=2))
    if not report["passed"]:
        print("SCORE-PARITY NOT MET:", report.get("stopped") or report.get("not_met"), sep="\n  ")
        return 1
    return 0


def load_snapshot_cmd(args) -> int:
    from retention.api import scoring as sc
    from retention.api import store
    from retention.api.batch import known_customers
    from retention.explain.config import load_explain_config
    from retention.optimize.value import EXTRA_COLUMNS

    repo = Path.cwd()
    require_clean(repo)
    snapshot = date.fromisoformat(args.snapshot)
    out = out_dir(args.fraction)
    report_path = out / f"load_snapshot_{snapshot.isoformat()}.json"
    run_once(report_path)
    t0 = time.time()
    protocol = load_protocol()
    settings = load_settings(horizons=(7, 30, HORIZON))
    cfg = load_explain_config(settings.columns)
    processed = processed_dir(args.fraction)
    with tempfile.TemporaryDirectory() as tmp:
        models = registry.load_serving_models(
            registry.registry_settings(repo), settings, protocol, Path(tmp)
        )
    cols = list(dict.fromkeys([*settings.columns, *EXTRA_COLUMNS]))
    facts = sorted({c for cs in cfg.facts.values() for c in cs} | {"tenure_days"})
    columns = list(dict.fromkeys([*cols, *facts]))
    inputs = snapshot_inputs(processed, snapshot, cols, facts, protocol)
    status = sc.population_status(inputs.status_rows, tuple(protocol.main_segments))
    status, keep = sc.refuse_unseen(status, inputs.rows, [m.model for m in models.c01.values()])
    rows = inputs.rows[keep].reset_index(drop=True)
    prices = inputs.prices[keep].reset_index(drop=True)
    pr = sc.probabilities(models.c01, rows, protocol.seed)
    finals = pd.DataFrame({"msno": rows.msno.astype(str)})
    for k, h in enumerate((7, 30, HORIZON)):
        finals[f"days_{h}"] = pr.final[:, k]
    known = known_customers(snapshot)
    counts = sc.population_counts(status) | {"not_active": len(known) - len(status)}
    r = store.client()
    written = store.load_snapshot(r, snapshot, status, rows, prices, columns, known, finals, counts)
    assert_no_label_building_code()
    report = stamp(
        {
            "snapshot": snapshot.isoformat(),
            "models_version": models.version,
            "counts": counts,
            "written": written,
            "valkey_used_memory_bytes": store.memory_used(r),
            "passed": True,
        },
        repo,
        t0,
    )
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def smoke_cmd(args) -> int:
    from retention.api import service as svc
    from retention.api import smoke, store

    repo = Path.cwd()
    require_clean(repo)
    snapshot = date.fromisoformat(args.snapshot)
    out = out_dir(args.fraction)
    report_path = out / f"smoke_{snapshot.isoformat()}.json"
    run_once(report_path)
    a8 = load_serving()["alerts_knowable_count"]
    if snapshot != date(2016, 11, 1):
        raise SystemExit("the knowable alert count is registered for 2016-11-01 only (A8)")
    t0 = time.time()
    report = smoke.run(svc.default_context, store.client, snapshot, a8["knowable_at_2016_11_01"])
    report = stamp(report, repo, t0)
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["unexpected"]:
        print(
            "SMOKE CHECK FOUND UNEXPECTED ANSWERS (not a mark):", *report["unexpected"], sep="\n  "
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.api")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("refit59", help="refit the 59-day classifier and judge it")
    r.add_argument("--fraction", type=float, default=None)
    r.add_argument("--save", action="store_true")
    r.add_argument("--expect-sha", action="append", default=[], metavar="59=SHA256")
    t = sub.add_parser("calibration-table", help="descriptive slopes before and after the fix")
    t.add_argument("--fraction", type=float, default=None)
    d = sub.add_parser("save-dt01", help="refit dt01, check it, save it with its model JSON")
    d.add_argument("--fraction", type=float, default=None)
    d.add_argument("--expect-sha", required=True, metavar="SHA256")
    g = sub.add_parser("register", help="register the eight-file bundle under the alias")
    g.add_argument("--fraction", type=float, default=None)
    v = sub.add_parser("verify-serving", help="load the models through the alias, checked")
    v.add_argument("--fraction", type=float, default=None)
    ls = sub.add_parser("load-snapshot", help="replace the Valkey store with one snapshot")
    ls.add_argument("--fraction", type=float, default=None)
    ls.add_argument("--snapshot", required=True, metavar="YYYY-MM-DD")
    sp = sub.add_parser("score-parity", help="judge every registered step 4 mark, once")
    sp.add_argument("--fraction", type=float, default=None)
    sp.add_argument("--dry-run", action="store_true", help="one md5 bucket, comparisons only")
    sub.add_parser("write-docs", help="write docs/scoring_api.md from the config and schemas")
    sm = sub.add_parser("smoke", help="the 5f smoke check of the loaded store (not a mark)")
    sm.add_argument("--fraction", type=float, default=None)
    sm.add_argument("--snapshot", required=True, metavar="YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return smoke_cmd(args)
    if args.command == "write-docs":
        from retention.api import docs

        print(docs.write())
        return 0
    if args.command == "score-parity":
        return score_parity(args)
    if args.command == "load-snapshot":
        return load_snapshot_cmd(args)
    if args.command == "register":
        return register_cmd(args)
    if args.command == "verify-serving":
        return verify_cmd(args)
    if args.command == "calibration-table":
        return table(args)
    if args.command == "save-dt01":
        return save_dt01(args)
    if args.save and not args.expect_sha:
        parser.error("--save needs --expect-sha 59=SHA256")
    return refit59(args)


if __name__ == "__main__":
    sys.exit(main())
