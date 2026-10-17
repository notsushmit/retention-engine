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
from retention.optimize.observed import read_once, shares
from retention.optimize.price import price_per_day, price_table, read_transactions
from retention.optimize.report import REPORT, load_logs, render, wording_violations
from retention.optimize.solve import (
    NONE,
    ExactCheckError,
    Plan,
    breaches,
    checked_plan,
    greedy,
    highs,
    steps,
)
from retention.optimize.stability import (
    Comparison,
    action_ranks,
    budget_curve,
    compare,
    discount_sentence,
    overall_verdict,
    ranked_by_loss,
    robust_core,
    robust_core_sentence,
    scored_net,
)
from retention.optimize.strategies import (
    baseline_choices,
    best_action,
    fill,
    projected_hours,
    random_draws,
    summarize,
    time_random,
)
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


# --- plan, observed, report ---------------------------------------------------------------


def _label(b: float | None) -> str:
    return "no limit" if b is None else f"{b:,.0f}"


def synthetic_options(n: int, cfg) -> object:
    """Synthetic customers at the real row count, for the random-baseline timing only."""
    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 10, n)
    days = rng.uniform(0, 60, n)
    c = pd.DataFrame(
        {
            "msno": [f"s{i:07d}" for i in range(n)],
            "p30": p,
            "auto_renew": np.where(rng.random(n) < 0.15, "off", "on"),
            "price_per_day": rng.choice([99 / 30, 149 / 30, 180 / 30], n),
            "days_30_90": days,
            "days_30_365": days * 4,
            "days_until_paid_expiry": pd.array(rng.integers(0, 30, n), dtype="Int32"),
        }
    )
    return build_options(c, cfg, cfg.base)


def cell_rows(opts, snapshot, sc, budget, choices, random_stats) -> list[dict]:
    rows = []
    for name, choice in choices.items():
        rows.append({**summarize(opts, choice), "strategy": name})
    mean = {m: v["mean"] for m, v in random_stats.items()}
    r = {**mean, "strategy": "random"}
    for m, v in random_stats.items():
        r[f"{m}_p2_5"] = v["p2_5"]
        r[f"{m}_p97_5"] = v["p97_5"]
    rows.append(r)
    fields = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in sc.fields().items()}
    for row in rows:
        row.update(fields)
        row.update(
            {
                "snapshot": snapshot.isoformat(),
                "budget": np.nan if budget is None else float(budget),
                "budget_label": _label(budget),
                "effect_basis": "assumed",
            }
        )
    return rows


def plan(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / "plan_report.json")
    cfg = load_actions_config()
    if cfg.headline_budget_level == "not yet registered":
        raise SystemExit("register the headline budget level first (its own commit)")
    refit = json.loads((out / "dt01_refit.json").read_text())
    if not refit["passed"]:
        raise SystemExit("the dt01 refit did not pass; no plan is run")
    t0 = time.time()
    customers = pd.read_parquet(out / "value.parquet")
    customers["snapshot_date"] = pd.to_datetime(customers.snapshot_date).dt.date
    by_snap = {
        t: customers[customers.snapshot_date == t].reset_index(drop=True) for t in cfg.snapshots
    }
    head = cfg.headline_budget_level
    head_budget = None if head is None else float(head)

    # timing on synthetic data before any real cell (registered stop)
    syn = synthetic_options(len(by_snap[cfg.base.snapshot]), cfg)
    syn_plan = greedy(syn, 50000.0)
    syn_seconds = time_random(syn, 50000.0, syn_plan, cfg.random_draws)
    cells = len(cfg.scenarios()) * len(cfg.budgets) + len(cfg.budgets)
    projected = projected_hours(syn_seconds, cells)
    timing = {
        "synthetic_seconds_per_cell": syn_seconds,
        "cells": cells,
        "projected_hours": projected,
    }
    if projected > cfg.timing_stop_hours:
        write_json(out / "plan_stopped.json", timing | {"reason": "projected time above the stop"})
        raise SystemExit(f"projected {projected:.2f} h is above {cfg.timing_stop_hours} h")

    index = {a: k for k, a in enumerate(cfg.action_order)}
    summary_rows, checks, stab_rows, plans_rows = [], [], [], []
    breach_log, repairs = [], 0
    optimizer_choice: dict[tuple, np.ndarray] = {}
    options: dict[tuple, object] = {}
    work = [(cfg.base.snapshot, name, sc) for name, sc in cfg.scenarios().items()]
    work += [(cfg.snapshots[0], "base", replace_snapshot(cfg.base, cfg.snapshots[0]))]
    first_cell = True
    for snapshot, name, sc in work:
        opts = build_options(by_snap[snapshot], cfg, sc)
        ordered = steps(opts)
        for budget in cfg.budgets:
            c0 = time.time()
            limit = sc.contact_limit
            chk = {"snapshot": snapshot.isoformat(), "scenario": name, "budget": _label(budget)}
            try:
                plan_, exact = checked_plan(opts, budget, limit, cfg.exact_time_limit, ordered)
            except ExactCheckError as e:
                write_json(out / "plan_stopped.json", chk | {"reason": str(e)})
                raise SystemExit(f"exact check failed: {e}") from e
            repairs += plan_.repairs
            chk |= exact
            checks.append(chk)
            choices = baseline_choices(opts, budget, limit, plan_, index)
            for strat, choice in choices.items():
                bad = breaches(opts, Plan(choice, 0, 0, 0, strat), budget, limit)
                if bad:
                    breach_log.append({**chk, "strategy": strat, "breaches": bad})
            rnd = random_draws(opts, budget, limit, plan_, cfg.random_draws, cfg.random_seed)
            summary_rows += cell_rows(opts, snapshot, sc, budget, choices, rnd)
            chk["seconds"] = round(time.time() - c0, 2)
            optimizer_choice[(snapshot, name, budget)] = plan_.choice
            options[(snapshot, name)] = opts
            if name == "base":
                for strat, choice in choices.items():
                    idx = np.flatnonzero(choice != NONE)
                    plans_rows += [
                        (snapshot.isoformat(), name, _label(budget), strat, opts.msno[i],
                         cfg.action_order[choice[i]])
                        for i in idx
                    ]  # fmt: skip
            if first_cell:
                first_cell = False
                recheck = projected_hours(time.time() - c0, cells)
                timing["recheck_projected_hours"] = recheck
                if recheck > cfg.timing_stop_hours:
                    write_json(out / "plan_stopped.json", timing | {"reason": "recheck above stop"})
                    raise SystemExit(f"recheck projected {recheck:.2f} h above the stop")
        print(f"{snapshot} {name}: done ({time.time() - t0:.0f} s)", flush=True)

    # determinism of HiGHS on the base point at the headline budget
    nov = cfg.base.snapshot
    base_opts = options[(nov, "base")]
    lim = cfg.exact_time_limit
    h1, h2 = highs(base_opts, head_budget, None, lim), highs(base_opts, head_budget, None, lim)
    if h1.extra["proven"] and h2.extra["proven"]:
        determinism = "identical" if (h1.choice == h2.choice).all() else "different"
    else:
        determinism = "not checked: not proven"
    if determinism == "different":
        write_json(out / "plan_stopped.json", {"reason": "two proven HiGHS solves differ"})
        raise SystemExit("two proven HiGHS solves differ")

    th, strict = (cfg.jaccard_min, cfg.regret_max), cfg.stricter
    for name, sc in cfg.variants.items():
        for budget in cfg.budgets:
            c = compare(
                name,
                budget,
                options[(nov, name)],
                optimizer_choice[(nov, "base", budget)],
                optimizer_choice[(nov, name, budget)],
                th,
                strict,
                sc.contact_limit,
            )
            stab_rows.append({**c.__dict__, "budget_label": _label(budget)})
    at_head = [
        Comparison(**{k: v for k, v in r.items() if k != "budget_label"})
        for r in stab_rows
        if r["budget_label"] == _label(head_budget)
    ]
    overall, failed = overall_verdict(at_head)
    core = robust_core(
        optimizer_choice[(nov, "base", head_budget)],
        {n: optimizer_choice[(nov, n, head_budget)] for n in cfg.variants},
    )
    r_full = action_ranks(base_opts, optimizer_choice[(nov, "base", head_budget)])["discount"]
    r_scaled = action_ranks(
        options[(nov, "discount_scaled_by_margin")],
        optimizer_choice[(nov, "discount_scaled_by_margin", head_budget)],
    )["discount"]
    low = next(c for c in at_head if c.variant == "low_effect")
    curve = budget_curve(
        [
            (b, scored_net(base_opts, optimizer_choice[(nov, "base", b)]))
            for b in cfg.budgets
            if b is not None
        ]
    )
    curve.append((None, scored_net(base_opts, optimizer_choice[(nov, "base", None)]), None))

    summary = pd.DataFrame(summary_rows)
    fields_ok = bool(
        summary[["effect_basis", "form", "level", "margin", "cost_level",
                 "discount_cost_scaled_by_margin", "value_horizon_days", "budget_label"]]
        .notna().all().all()
    )  # fmt: skip
    _write_parquet(summary, out / "summary.parquet")
    _write_parquet(pd.DataFrame(stab_rows), out / "stability.parquet")
    _write_parquet(
        pd.DataFrame(plans_rows, columns=["snapshot", "scenario", "budget_label", "strategy",
                                          "msno", "action"]),
        out / "plans.parquet",
    )  # fmt: skip
    diffs = [c["highs"] - c["greedy"] for c in checks if c["highs_proven"]]
    unproven = sum(1 for c in checks if not c["highs_proven"])
    report = {
        "step": "plan",
        "headline_budget": "no_limit" if head_budget is None else head_budget,
        "timing": timing,
        "solver_checks": checks,
        "solver_summary": {
            "cells": len(checks),
            "checked": len(checks),
            "proven": len(checks) - unproven,
            "unproven": unproven,
            "unproven_share": unproven / len(checks),
            "time_limit_seconds": cfg.exact_time_limit,
            "max_highs_minus_greedy": max(diffs) if diffs else 0.0,
            "breaches": len(breach_log),
            "repairs": repairs,
        },
        "highs_deterministic": {
            "scenario": "base",
            "budget": _label(head_budget),
            "result": determinism,
        },
        "breaches": breach_log,
        "budget_curve": curve,
        "verdict": {
            "overall": overall,
            "failed": failed,
            "ranked_by_loss": ranked_by_loss(at_head),
            "low_effect_loss": low.loss,
            "robust_core": core,
            "robust_core_sentence": robust_core_sentence(core),
            "discount_ranks": {"full_cost": r_full, "margin_scaled": r_scaled},
            "discount_sentence": discount_sentence(r_full, r_scaled),
        },
        "expectations": EXPECTATIONS,
        "criterion_4": {
            "primary_solves": sum(1 for c in checks if c["snapshot"] == nov.isoformat()),
            "secondary_solves": sum(1 for c in checks if c["snapshot"] != nov.isoformat()),
            "rows_carry_scenario_fields": fields_ok,
        },
        "git_head": head_commit(repo),
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(out / "plan_report.json", report)
    print(json.dumps(report["solver_summary"] | report["verdict"], indent=2, default=str))
    return 1 if breach_log or report["peak_rss_gb"] >= cfg.memory_ceiling_gb else 0


EXPECTATIONS = [
    "the low_effect variant gives an empty plan",
    "the absolute form is likely the least stable (a flat effect favours high-value customers "
    "whatever their risk)",
    "the margin and cost variants mostly change spend more than who is targeted",
]


def replace_snapshot(sc, snapshot: date):
    from dataclasses import replace

    return replace(sc, snapshot=snapshot)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"effect_basis": "assumed", "label": load_actions_config().label}
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"optimize": json.dumps(meta).encode()}
    )
    pq.write_table(table, path, compression="zstd")


def observed_step(args) -> int:
    repo = Path.cwd()
    require_clean(repo)
    out = out_dir(args.fraction)
    run_once(out / "observed_churn.json")
    cfg = load_actions_config()
    protocol = load_protocol()
    processed = processed_dir(args.fraction)
    customers = pd.read_parquet(out / "value.parquet")
    customers["snapshot_date"] = pd.to_datetime(customers.snapshot_date).dt.date
    plans = pd.read_parquet(out / "plans.parquet")
    head = cfg.headline_budget_level
    head_budget = None if head is None else float(head)
    index = {a: k for k, a in enumerate(cfg.action_order)}
    result: dict = {}
    for snapshot in (cfg.base.snapshot, cfg.snapshots[0]):
        c = customers[customers.snapshot_date == snapshot].reset_index(drop=True)
        sc = replace_snapshot(cfg.base, snapshot)
        opts = build_options(c, cfg, sc)
        p = plans[
            (plans.snapshot == snapshot.isoformat()) & (plans.budget_label == _label(head_budget))
        ]
        choices = {}
        for strat, g in p.groupby("strategy"):
            ch = np.full(opts.n, NONE, dtype=np.int64)
            pos = pd.Series(np.arange(opts.n), index=opts.msno)
            ch[pos[g.msno].to_numpy()] = g.action.map(index).to_numpy()
            choices[strat] = ch
        y = read_once(processed / "labels.parquet", snapshot, protocol, out, head_commit(repo))
        by = shares(opts.msno, choices, y)
        opt = greedy(opts, head_budget)
        draws = []
        for d in range(cfg.random_draws):
            order = np.random.default_rng(cfg.random_seed + d).permutation(opts.n)
            count = opt.contacts if head_budget is None else None
            ch = fill(order, best_action(opts), opts.cost, head_budget, count)
            draws.append(shares(opts.msno, {"random": ch}, y)["random"])
        vals = np.array([d["share_lapsed"] for d in draws if d["share_lapsed"] is not None])
        by["random"] = {
            "targeted": int(round(np.mean([d["targeted"] for d in draws]))),
            "share_lapsed": float(vals.mean()) if len(vals) else None,
            "p2_5": float(np.percentile(vals, 2.5)) if len(vals) else None,
            "p97_5": float(np.percentile(vals, 97.5)) if len(vals) else None,
            "label": cfg.observed_label,
        }
        result[snapshot.isoformat()] = by
    assert_no_label_building_code()
    report = {
        "step": "observed",
        "label": cfg.observed_label,
        "scenario": "base",
        "budget": _label(head_budget),
        "shares": result,
        "git_head": head_commit(repo),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(out / "observed_churn.json", report)
    print(json.dumps(report, indent=2, default=str))
    return 0


def report_step(args) -> int:
    cfg = load_actions_config()
    text = render(cfg, load_logs(out_dir(args.fraction)))
    bad = wording_violations(text, cfg)
    if bad:
        print("WORDING RULES BROKEN; report not written:", *bad, sep="\n  ")
        return 1
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"wrote {REPORT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retention.optimize")
    parser.add_argument("--fraction", type=float, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("recount", help="counts only: prices and no-price customers")
    sub.add_parser("value", help="dt01 refit with reproduction, value inputs, no-limit spend")
    sub.add_parser("plan", help="every scenario and budget, strategies, stability")
    sub.add_parser("observed", help="the one-time observed churn read")
    sub.add_parser("report", help="render reports/strategy_backtest.md from the logs")
    args = parser.parse_args(argv)
    np.seterr(all="ignore")
    steps_ = {
        "recount": recount,
        "value": value,
        "plan": plan,
        "observed": observed_step,
        "report": report_step,
    }
    return steps_[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
