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

Every command refuses to run when a tracked file differs from HEAD, so each
report ties to a commit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retention.api import dt01 as dt01_step
from retention.api.config import load_serving, ordering_marks
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
    args = parser.parse_args(argv)
    if args.command == "calibration-table":
        return table(args)
    if args.command == "save-dt01":
        return save_dt01(args)
    if args.save and not args.expect_sha:
        parser.error("--save needs --expect-sha 59=SHA256")
    return refit59(args)


if __name__ == "__main__":
    sys.exit(main())
