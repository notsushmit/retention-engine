"""Phase 7 steps: python -m retention.optimize {recount,value}.

Each step runs once, at a clean commit, under caffeinate -i, and logs its run.
The registered order: recount, value, then the headline budget level is
registered in its own commit, then plan, observed, report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from retention.data.sample import resolve_fraction, sample_dir
from retention.explain.model import library_versions, p2_main_rows
from retention.models.protocol import load_protocol
from retention.models.resources import peak_rss_gb
from retention.models.stages import assert_no_label_building_code
from retention.optimize.config import load_actions_config
from retention.optimize.effects import build_options, missing_deadline_count
from retention.optimize.price import price_per_day, price_table, read_transactions
from retention.optimize.solve import breaches, greedy
from retention.optimize.value import (
    EXTRA_COLUMNS,
    INTERVAL_ENDS_ALL,
    customer_table,
    disagreement,
    dt01_settings,
    headline_level,
    read_p30,
    refit_dt01,
    reproduce,
    write_json,
)


def _kkbox() -> dict:
    return yaml.safe_load(Path("configs/kkbox.yaml").read_text())


def processed_dir(fraction: float | None) -> Path:
    raw = _kkbox()
    return sample_dir(Path(raw["labels"]["processed_dir"]), resolve_fraction(fraction, raw["data"]))


def interim_dir(fraction: float | None) -> Path:
    raw = _kkbox()
    return sample_dir(Path(raw["data"]["interim_dir"]), resolve_fraction(fraction, raw["data"]))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def head_commit(repo: Path) -> str:
    return git("-C", str(repo), "rev-parse", "--short=7", "HEAD")


def tree_dirty(repo: Path) -> bool:
    return bool(git("-C", str(repo), "status", "--porcelain", "--untracked-files=no"))


def out_dir(fraction: float | None) -> Path:
    return processed_dir(fraction) / "optimize"


def require_clean(repo: Path) -> None:
    if tree_dirty(repo):
        raise SystemExit("a tracked file differs from HEAD; commit first so the run ties to code")


def run_once(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: this step runs once and is never rerun")


def population(processed: Path, snapshots: tuple[date, ...], columns: list[str], protocol):
    return {t: p2_main_rows(processed, t, columns, protocol) for t in snapshots}


def recount(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / "recount.json")
    t0 = time.time()
    cfg = load_actions_config()
    protocol = load_protocol()
    processed = processed_dir(args.fraction)
    rows = population(processed, cfg.snapshots, list(EXTRA_COLUMNS), protocol)
    tx = read_transactions(list(cfg.snapshots))
    per_snapshot = {}
    for t, r in rows.items():
        pr = price_per_day(tx, r[["msno", "snapshot_date"]])
        per_snapshot[t.isoformat()] = {
            "customers": len(r),
            "no_usable_price": int((~pr.has_price).sum()),
            "missing_deadline": int(r.days_until_paid_expiry.isna().sum()),
            "auto_renew_off": int((r.last_is_auto_renew == 0).sum()),
            "auto_renew_on": int((r.last_is_auto_renew == 1).sum()),
        }
    assert_no_label_building_code()
    report = {
        "step": "recount",
        "kind": "counts only; no outcome column read",
        "price_table": price_table(),
        "per_snapshot": per_snapshot,
        "git_head": head_commit(repo),
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(out / "recount.json", report)
    print(json.dumps(report, indent=2))
    return 0


def value(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / "dt01_refit.json")
    t0 = time.time()
    cfg = load_actions_config()
    protocol = load_protocol()
    processed = processed_dir(args.fraction)
    cache = interim_dir(args.fraction) / "labels_asof"
    s = dt01_settings()
    model, n_fit = refit_dt01(protocol, processed, cache, s, cfg)
    columns = list(dict.fromkeys([*s.columns, *EXTRA_COLUMNS]))
    rows = population(processed, cfg.snapshots, columns, protocol)
    rehearsal = json.loads((processed / "rehearsal.json").read_text())
    table, failures = reproduce(model, rehearsal, rows, cfg)
    assert_no_label_building_code()
    peak = peak_rss_gb()
    if peak >= cfg.memory_ceiling_gb:
        failures.append(f"peak RSS {peak} GB at or above the {cfg.memory_ceiling_gb} GB ceiling")
    booster = model.model.booster.model_to_string()
    report = {
        "step": "value: dt01 refit",
        "fit_rows": n_fit,
        "expanded_rows": model.expanded_rows,
        "reproduction": table,
        "failures": failures,
        "passed": not failures,
        "booster_sha256": hashlib.sha256(booster.encode()).hexdigest(),
        "library_versions": library_versions(),
        "threads": cfg.threads,
        "git_head": head_commit(repo),
        "seconds_refit": round(time.time() - t0, 1),
        "peak_rss_gb": peak,
    }
    write_json(out / "dt01_refit.json", report)
    print(json.dumps(report, indent=2))
    if failures:
        print("REPRODUCTION FAILED; nothing saved:", *failures, sep="\n  ")
        return 1

    model_dir = interim_dir(args.fraction) / "optimize" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dt01.txt").write_text(booster)
    p30 = read_p30(processed / "explain" / "reasons.parquet", list(cfg.snapshots))
    tx = read_transactions(list(cfg.snapshots))
    tables, per_snapshot = [], {}
    for t, r in rows.items():
        surv = 1 - model.lapse_probability(r, list(INTERVAL_ENDS_ALL))[0]
        prices = price_per_day(tx, r[["msno", "snapshot_date"]])
        tab = customer_table(r, surv, p30[p30.snapshot_date == t], prices)
        tables.append(tab)
        per_snapshot[t.isoformat()] = {
            "customers": len(tab),
            "no_usable_price": int((~tab.has_price).sum()),
            "missing_deadline": missing_deadline_count(tab),
            "model_disagreement": disagreement(tab),
        }
    customers = pd.concat(tables, ignore_index=True)
    table_out = pa.Table.from_pandas(customers, preserve_index=False)
    meta = {"label": "value inputs: dt01 curve shape, c01 probability, KKBox prices"}
    table_out = table_out.replace_schema_metadata(
        {**(table_out.schema.metadata or {}), b"optimize": json.dumps(meta).encode()}
    )
    pq.write_table(table_out, out / "value.parquet", compression="zstd")

    base = cfg.base
    snap = customers[customers.snapshot_date == base.snapshot].reset_index(drop=True)
    opts = build_options(snap, cfg, base)
    plan = greedy(opts, None)
    bad = breaches(opts, plan, None, None)
    spend = {
        "snapshot": base.snapshot.isoformat(),
        "scenario": base.fields(),
        "effect_basis": "assumed",
        "no_limit_spend": plan.cost,
        "headline_level_by_rule": headline_level(plan.cost, cfg.budgets),
        "guard_breaches": bad,
        "note": "only the spend is logged; no variant, no comparison, no outcome",
    }
    write_json(out / "no_limit_spend.json", spend)
    assert_no_label_building_code()
    report_value = {
        "step": "value: customer inputs",
        "per_snapshot": per_snapshot,
        "seconds_total": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(out / "value_report.json", report_value)
    print(json.dumps(report_value | {"no_limit_spend": spend}, indent=2, default=str))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.optimize")
    parser.add_argument("--fraction", type=float, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("recount", help="counts only: prices and no-price customers")
    sub.add_parser("value", help="dt01 refit with reproduction, value inputs, no-limit spend")
    args = parser.parse_args(argv)
    np.seterr(all="ignore")
    return {"recount": recount, "value": value}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
