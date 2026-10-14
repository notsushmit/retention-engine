"""Command line entry: `python -m retention.explain <command>`.

refit: refit the 7-day and 30-day classifiers twice each, compare their raw-score
summaries with the 2016 rehearsal's logged ones, and write refit_report.json.
Nothing else is written. Exits with an error if any pass mark fails.

refit --save --expect-sha 7=<hex> --expect-sha 30=<hex>: the same refit and
checks in a new process; the boosters and Platt parameters are saved only if
every check passes and each booster's SHA-256 equals the expected one.

contributions [--output-dir DIR]: SHAP contributions of the saved boosters for
the explained snapshots, with every pass mark checked per chunk; writes the
parts and the descriptive reports (tables B and C, illustrative rows) to DIR
(default data/processed/<sample>/explain/contributions). Every read of
labels.parquet is audited: an outcome column stops the run.

compare-contributions A B: whether two runs' outputs are identical after
sorting by key.

hand-check: the 20-row hand check (configs/explain.yaml, hand_check): each row
rendered at 7 and 30 days next to its raw rows, with the automatic checks;
written to data/processed/<sample>/explain/hand_check_<seed>.md (and .json).
--seed redraws with another seed (the configured seed + k after a failure). Exits
with an error if any row fails.

reasons [--output PATH]: reasons.parquet (one row per customer, snapshot and
horizon) from the stored contributions, the features and the saved Platt
parameters; no labels are read (a guard makes any labels read fail the run).

compare-reasons A B: whether two reasons files are identical after sorting by key.

refit, contributions, hand-check and reasons refuse to run when a tracked file differs from HEAD, so
the reports and the saved models tie to a commit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retention.explain import model as explain_model
from retention.explain.config import load_explain_config
from retention.explain.contributions import run_contributions, same_outputs
from retention.explain.describe import (
    facts_check,
    facts_reference_by_snapshot,
    illustrative_rows,
    table_b_groups,
    table_b_month,
)
from retention.explain.hand_check import run_hand_check
from retention.explain.model import load_settings, run_check, save_models
from retention.models.__main__ import interim_dir, processed_dir
from retention.models.data import column_kind
from retention.models.protocol import load_protocol
from retention.models.resources import peak_rss_gb
from retention.models.stages import assert_no_label_building_code
from retention.models.tracking import head_commit, tree_dirty


def explain_dir(fraction: float | None) -> Path:
    return interim_dir(fraction) / "explain"


def parse_expected(values: list[str]) -> dict[int, str]:
    out = {}
    for v in values:
        h, _, sha = v.partition("=")
        if not sha:
            raise SystemExit(f"--expect-sha takes HORIZON=SHA256, got {v!r}")
        out[int(h)] = sha.strip()
    return out


def refit(args) -> int:
    repo = Path.cwd()
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the refit ties to code")
    t0 = time.time()
    protocol = load_protocol()
    processed = processed_dir(args.fraction)
    cache = interim_dir(args.fraction) / "labels_asof"
    report, refits = run_check(protocol, processed, cache, load_settings())
    assert_no_label_building_code()
    report |= {
        "git_head": head_commit(repo),
        "tree_clean": True,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
        "save": bool(args.save),
    }
    out_dir = explain_dir(args.fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "refit_report_save.json" if args.save else "refit_report.json"
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("REFIT CHECK FAILED; nothing saved:", *report["failures"], sep="\n  ")
        return 1
    if args.save:
        written = save_models(out_dir / "model", refits, parse_expected(args.expect_sha))
        print("saved:", *[str(p) for p in written], sep="\n  ")
    return 0


def audit_label_reads(log: list[dict]) -> None:
    """Record every labels.parquet read of the explain code; an outcome column raises.

    Wraps the reader in every explain module that reads labels (the model rows and
    the hand check).
    """
    from retention.explain import hand_check as hand_check_module

    for module in (explain_model, hand_check_module):
        real = module.load_labels

        def audited(path, snapshots, columns, protocol, unlock=None, _real=real):
            if unlock is not None:
                raise PermissionError("the explain code never passes an unlock")
            outcome = [c for c in columns if column_kind(c) != "pre_t"]
            if outcome:
                raise PermissionError(f"outcome columns requested: {outcome}")
            log.append({"snapshots": [str(t) for t in snapshots], "columns": list(columns)})
            return _real(path, snapshots, columns, protocol)

        module.load_labels = audited


def require_label_reads(log: list[dict]) -> None:
    """A run that read rows must have recorded its labels reads; none means the audit missed."""
    if not log:
        raise RuntimeError("the labels-read audit recorded nothing: a read bypassed it")


def contributions(args) -> int:
    repo = Path.cwd()
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the run ties to code")
    t0 = time.time()
    protocol = load_protocol()
    settings = load_settings()
    cfg = load_explain_config(settings.columns)
    processed = processed_dir(args.fraction)
    model_dir = explain_dir(args.fraction) / "model"
    refit_report = json.loads((explain_dir(args.fraction) / "refit_report_save.json").read_text())
    rehearsal = json.loads((processed / "rehearsal.json").read_text())
    out_dir = Path(args.output_dir) if args.output_dir else processed / "explain" / "contributions"
    reads: list[dict] = []
    audit_label_reads(reads)
    report, results = run_contributions(
        protocol, processed, model_dir, refit_report, out_dir, cfg, settings, rehearsal=rehearsal
    )
    require_label_reads(reads)
    assert_no_label_building_code()
    report |= {
        "label_reads": reads,
        "git_head": head_commit(repo),
        "tree_clean": True,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
        "library_versions": explain_model.library_versions(),
    }
    outputs = {
        "contributions_report.json": report,
        "table_b_months.json": [table_b_month(cfg, r) for r in results],
        "table_b_groups.json": table_b_groups(cfg, results),
        "facts_check.json": facts_check(cfg, results),
        "facts_reference_by_snapshot.json": facts_reference_by_snapshot(cfg, results),
        "illustrative_rows.json": illustrative_rows(cfg, results),
    }
    for name, data in outputs.items():
        (out_dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "seconds", "peak_rss_gb", "git_head")}))
    return 0


def hand_check(args) -> int:
    repo = Path.cwd()
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the run ties to code")
    t0 = time.time()
    from retention.explain.reasons import Renderer

    protocol = load_protocol()
    settings = load_settings()
    cfg = load_explain_config(settings.columns)
    processed = processed_dir(args.fraction)
    reads: list[dict] = []
    audit_label_reads(reads)
    from retention.explain.hand_check import load_hand_check_config

    seed = args.seed if args.seed is not None else int(load_hand_check_config()["seed"])
    out = processed / "explain" / f"hand_check_{seed}.md"
    report = run_hand_check(
        protocol,
        processed,
        interim_dir(args.fraction),
        processed / "explain" / "contributions",
        explain_dir(args.fraction) / "model",
        cfg,
        Renderer.from_files(cfg),
        out,
        seed,
    )
    require_label_reads(reads)
    assert_no_label_building_code()
    print(out.read_text())
    summary = {
        "seed": seed,
        "rows": report["rows"],
        "passed": report["passed"],
        "label_reads": sorted({c for r in reads for c in r["columns"]}),
        "label_read_count": len(reads),
        "git_head": head_commit(repo),
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    print(json.dumps(summary))
    return 0 if report["passed"] == report["rows"] else 1


def forbid_label_reads() -> None:
    """Make every labels reader the explain code can reach fail (for a no-labels run)."""
    from retention.explain import hand_check as hand_check_module
    from retention.models import data as data_module
    from retention.models import frames as frames_module

    def refuse(*args, **kwargs):
        raise PermissionError("this run reads no labels")

    for module in (explain_model, hand_check_module, data_module, frames_module):
        module.load_labels = refuse


def reasons(args) -> int:
    from retention.explain.build_reasons import committed_sha256, run_reasons, sha256_file
    from retention.explain.contributions import EXPECTED_SHA256
    from retention.explain.reasons import PHRASES, Renderer

    repo = Path.cwd()
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the run ties to code")
    t0 = time.time()
    protocol = load_protocol()
    settings = load_settings()
    cfg = load_explain_config(settings.columns)
    renderer = Renderer.from_files(cfg)
    if committed_sha256(repo, PHRASES) != sha256_file(PHRASES):
        raise SystemExit("the phrases file differs from the committed one")
    processed = processed_dir(args.fraction)
    out = Path(args.output) if args.output else processed / "explain" / "reasons.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    forbid_label_reads()
    meta = {
        "git_head": head_commit(repo),
        "explain_yaml_sha256": sha256_file(Path("configs/explain.yaml")),
        "booster_sha256": json.dumps({str(h): v for h, v in EXPECTED_SHA256.items()}),
    }
    report = run_reasons(
        protocol,
        processed,
        processed / "explain" / "contributions",
        explain_dir(args.fraction) / "model",
        cfg,
        renderer,
        out,
        metadata=meta,
    )
    assert_no_label_building_code()
    report |= meta | {
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
        "output": str(out),
    }
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "seconds", "peak_rss_gb", "git_head")}))
    return 0


def compare_reasons(args) -> int:
    from retention.explain.build_reasons import same_reasons

    result = same_reasons(Path(args.a), Path(args.b))
    print(json.dumps(result))
    return 0 if result["identical"] else 1


def compare(args) -> int:
    result = same_outputs(Path(args.a), Path(args.b))
    print(json.dumps(result))
    return 0 if result["identical"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.explain")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("refit", help="refit the explained classifiers and check them")
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--expect-sha", action="append", default=[], metavar="HORIZON=SHA256")
    c = sub.add_parser("contributions", help="SHAP contributions of the saved boosters")
    c.add_argument("--fraction", type=float, default=None)
    c.add_argument("--output-dir", default=None)
    hc = sub.add_parser("hand-check", help="the 20-row hand check of the reasons")
    hc.add_argument("--fraction", type=float, default=None)
    hc.add_argument("--seed", type=int, default=None, help="redraw seed (configured seed + k)")
    rs = sub.add_parser("reasons", help="reasons.parquet for every explained row")
    rs.add_argument("--fraction", type=float, default=None)
    rs.add_argument("--output", default=None)
    cr = sub.add_parser("compare-reasons", help="compare two reasons files")
    cr.add_argument("a")
    cr.add_argument("b")
    k = sub.add_parser("compare-contributions", help="compare two contribution runs")
    k.add_argument("a")
    k.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "contributions":
        return contributions(args)
    if args.command == "compare-contributions":
        return compare(args)
    if args.command == "hand-check":
        return hand_check(args)
    if args.command == "reasons":
        return reasons(args)
    if args.command == "compare-reasons":
        return compare_reasons(args)
    if args.save and not args.expect_sha:
        parser.error("--save needs --expect-sha for every horizon")
    return refit(args)


if __name__ == "__main__":
    sys.exit(main())
