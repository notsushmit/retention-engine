"""Command line entry: `python -m retention.explain <command>`.

refit: refit the 7-day and 30-day classifiers twice each, compare their raw-score
summaries with the 2016 rehearsal's logged ones, and write refit_report.json.
Nothing else is written. Exits with an error if any pass mark fails.

refit --save --expect-sha 7=<hex> --expect-sha 30=<hex>: the same refit and
checks in a new process; the boosters and Platt parameters are saved only if
every check passes and each booster's SHA-256 equals the expected one.

Both refuse to run when a tracked file differs from HEAD, so the report and the
saved models tie to a commit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retention.explain.model import load_settings, run_check, save_models
from retention.models.__main__ import interim_dir, processed_dir
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.explain")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("refit", help="refit the explained classifiers and check them")
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--expect-sha", action="append", default=[], metavar="HORIZON=SHA256")
    args = parser.parse_args(argv)
    if args.save and not args.expect_sha:
        parser.error("--save needs --expect-sha for every horizon")
    return refit(args)


if __name__ == "__main__":
    sys.exit(main())
