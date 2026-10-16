"""Command line entry: `python -m retention.uplift <command>`.

check-public: checks both randomized datasets against configs/uplift.yaml (hash, rows,
columns, arms, treatment share), stops on the first mismatch, and writes Parquet to
data/interim/public/ only after every check of both files has passed. Refuses to run with
uncommitted changes to tracked files. Writes check_public.json beside the Parquet files.

recount: rebuilds the KKBox renewal units (pre-T label columns and transactions dated on
or before 2016-12-31 only; no outcome column) and compares every registered count, and the
2016 figures quoted in docs/causal_assumptions.md, with the recount. Stops (exit code 1)
if any differs. Writes recount.json under data/processed/kkbox/sample_020/uplift/.

propensity: the KKBox propensity step only (no outcome is read): the recount check, the
registered covariates and column rules, the penalty chosen on treatment with
customer-grouped folds, out-of-fold propensities, the overlap check, the three trimming
rules with stabilised weights and the balance table (registered fill), and the
estimability mark under primary trimming. Writes propensity.json, propensity_scores.parquet
and balance_<rule>.parquet under data/processed/kkbox/sample_020/uplift/.

kept-composition: from the saved propensity scores and pre-T covariates (no refit, no
outcome), the kept units per trimming rule by month, arm and past-year discount, and the
mirror check of month_2016_09 against days_until_expiry in each balance table. Writes
kept_composition.json.

validate: the single public-data validation run (configs/uplift.yaml, validation). Runs
once (a marker before the first fit; refuses once validation.json exists; a repeat of an
unfinished attempt needs --rerun-reason). Writes validation.json, qini_curves.parquet and
uplift_scores.parquet under data/processed/public/.

qini-figures: draws one SVG per primary cell in docs/img/ from qini_curves.parquet only.

groups: the registered customer groups on the public data (configs/uplift.yaml, groups);
writes groups.json and group_members.parquet under data/processed/public/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from retention.models.protocol import load_protocol
from retention.models.resources import peak_rss_gb
from retention.uplift import (
    balance,
    config,
    figures,
    groups,
    kkbox_bootstrap,
    kkbox_units,
    learners,
    outcome,
    placebo,
    propensity,
    public_data,
    public_validation,
    report,
)

OUT = Path("data/interim/public")
KKBOX = Path("data/processed/kkbox/sample_020")
KKBOX_OUT = KKBOX / "uplift"
# Figures quoted in docs/causal_assumptions.md (from the counts runs). The recount stops if
# any differs. First paragraph: full-price or discounted first renewals, 2016-01 to 2016-11.
DOC_FIGURES_2016 = {"units": 1582100, "method_23": 3457, "customers": 233945, "m23_customers": 399}
DOC_FACTS = {
    # every first renewal below list on a priced plan, 2016-01 to 2016-11: method -> count
    "below_list_by_method_2016": {"23": 234},
    # distinct (list, paid) pairs among those rows
    "below_list_pairs_2016": ["149->129", "149->99"],
    # analysed discounted renewals by depth (1 - paid / list), rounded to 3 places
    "analysed_treated_depths": {"0.134": 69, "0.336": 1},
    # analysed units whose last transaction before T was discounted
    "last_discount_ratio_above_0": 0,
    "analysed_units": 620,
    # units with a discounted row in the 365 days before T, by month and arm
    "past_year_discount": {
        "2016-08": {"treated": [0, 2], "untreated": [89, 308]},
        "2016-09": {"treated": [65, 68], "untreated": [25, 242]},
    },
    # every such earlier row, as list -> paid on a method
    "past_year_discount_rows": ["23:149->129"],
    # month range of the earlier discounted row, September units, by arm
    "past_year_discount_months_2016_09": {
        "treated": ["2015-11", "2016-08"],
        "untreated": ["2015-09", "2015-10"],
    },
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _require_clean() -> None:
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("tracked files have uncommitted changes; commit first")


def check_public() -> dict:
    _require_clean()
    start = time.perf_counter()
    v = config.load()["validation"]["datasets"]
    h, c = v["hillstrom"], v["criteo"]
    log: dict = {
        "commit": _git("rev-parse", "--short", "HEAD").strip(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        log["hillstrom"] = public_data.check_hillstrom(
            Path(h["local_file"]),
            h["sha256"],
            h["expected_rows"],
            h["expected_columns"],
            [
                h["treatment_arms"]["mens_vs_none"],
                h["treatment_arms"]["womens_vs_none"],
                h["control_arm"],
            ],
            h["arm_share_tolerance"],
        )
        log["criteo"] = public_data.check_criteo(
            Path(c["local_file"]),
            c["expected_sha256"],
            c["expected_rows"],
            c["expected_columns"],
            c["treatment_share"],
            c["treatment_share_tolerance"],
        )
        log["status"] = "all checks passed"
        log["hillstrom"]["parquet"] = public_data.to_parquet(
            Path(h["local_file"]), OUT / "hillstrom.parquet"
        )
        log["criteo"]["parquet"] = public_data.to_parquet(
            Path(c["local_file"]), OUT / "criteo_uplift_v2.1.parquet"
        )
    except public_data.CheckFailed as err:
        log["status"] = "stopped"
        log["failure"] = str(err)
    log["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    log["seconds"] = round(time.perf_counter() - start, 1)
    log["peak_rss_gb"] = peak_rss_gb()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_public.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def _months(first: date, last: date) -> list[date]:
    return [d.date() for d in pd.date_range(first, last, freq="MS")]


def _units(snapshots: list[date], tx: pd.DataFrame) -> pd.DataFrame:
    """Renewal units, reading at most four snapshots of labels at a time."""
    protocol = load_protocol()
    parts = []
    for i in range(0, len(snapshots), 4):
        members = kkbox_units.read_members(KKBOX / "labels.parquet", snapshots[i : i + 4], protocol)
        parts.append(kkbox_units.renewal_units(members, tx))
    return pd.concat(parts, ignore_index=True)


def _doc_facts(year: pd.DataFrame, analysed: pd.DataFrame, tx: pd.DataFrame) -> dict:
    """Pre-T facts quoted in the doc; reads no outcome."""
    below = year[(year.plan_list_price > 0) & (year.actual_amount_paid < year.plan_list_price)]
    below = below[below.actual_amount_paid > 0]
    pair_iter = zip(below.plan_list_price, below.actual_amount_paid, strict=True)
    pairs = sorted({f"{lp}->{ap}" for lp, ap in pair_iter})
    treated = analysed[analysed.treated == 1]
    depths = treated.depth.round(3).map("{:.3f}".format).value_counts().sort_index()
    cov = kkbox_units.covariates(analysed, KKBOX / "features.parquet", tx)
    month = pd.to_datetime(analysed.snapshot_date).dt.strftime("%Y-%m").to_numpy()
    arm = np.where(analysed.treated.to_numpy() == 1, "treated", "untreated")
    hist = cov.any_discount_365.to_numpy() == 1
    past: dict = {}
    for m in sorted(set(month)):
        past[m] = {}
        for a in ("treated", "untreated"):
            sel = (month == m) & (arm == a)
            past[m][a] = [int(hist[sel].sum()), int(sel.sum())]
    rows = kkbox_units.earlier_discounts(analysed, tx)
    kinds = sorted(
        {
            f"{pm}:{lp}->{ap}"
            for pm, lp, ap in zip(
                rows.payment_method_id, rows.plan_list_price, rows.actual_amount_paid, strict=True
            )
        }
    )
    sep = rows[pd.to_datetime(rows.snapshot_date) == pd.Timestamp("2016-09-01")]
    months = {}
    for a, flag in (("treated", 1), ("untreated", 0)):
        d = pd.to_datetime(sep.transaction_date[sep.treated == flag])
        months[a] = [d.min().strftime("%Y-%m"), d.max().strftime("%Y-%m")]
    return {
        "below_list_by_method_2016": {
            str(k): int(v) for k, v in below.payment_method_id.value_counts().items()
        },
        "below_list_pairs_2016": pairs,
        "analysed_treated_depths": {k: int(v) for k, v in depths.items()},
        "last_discount_ratio_above_0": int((cov.last_discount_ratio.fillna(0) > 0).sum()),
        "analysed_units": len(analysed),
        "past_year_discount": past,
        "past_year_discount_rows": kinds,
        "past_year_discount_months_2016_09": months,
    }


def recount() -> dict:
    _require_clean()
    start = time.perf_counter()
    registered = config.load()["kkbox"]["counts_before_outcome"]
    tx = kkbox_units.read_transactions(KKBOX / "transactions.parquet")
    units = _units(kkbox_units.SNAPSHOTS, tx)
    reg = kkbox_units.registered_set(units)
    analysed = kkbox_units.analysed_set(reg)
    counts = kkbox_units.recount(reg, analysed)
    year = _units(_months(date(2016, 1, 1), date(2016, 11, 1)), tx)
    fd = year[year.kind.isin(["full", "discounted"])]
    m23 = fd.payment_method_id.eq(kkbox_units.PAYMENT_METHOD)
    doc = {
        "units": len(fd),
        "method_23": int(m23.sum()),
        "customers": int(fd.msno.nunique()),
        "m23_customers": int(fd.msno[m23].nunique()),
    }
    facts = _doc_facts(year, analysed, tx)
    log: dict = {
        "commit": _git("rev-parse", "--short", "HEAD").strip(),
        "registered": {k: registered[k] for k in asdict(counts)},
        "recounted": asdict(counts),
        "share_with_zero_chance": {
            "registered": registered["share_with_zero_chance"],
            "recounted": round(1 - counts.method_23 / counts.registered_set_all_methods, 3),
        },
        "doc_figures_2016": {"quoted": DOC_FIGURES_2016, "recounted": doc},
        "kinds_2016": year.kind.value_counts().to_dict(),
        "doc_facts": {"quoted": DOC_FACTS, "recounted": facts},
    }
    try:
        kkbox_units.check_counts(counts, registered)
        if doc != DOC_FIGURES_2016:
            raise kkbox_units.CountMismatch(f"2016 doc figures differ: {doc}")
        if facts != DOC_FACTS:
            diff = sorted(k for k in DOC_FACTS if DOC_FACTS[k] != facts.get(k))
            raise kkbox_units.CountMismatch(f"doc facts differ: {diff}")
        zero = log["share_with_zero_chance"]
        if zero["registered"] != zero["recounted"]:
            raise kkbox_units.CountMismatch("share_with_zero_chance differs")
        log["status"] = "every count matches"
    except kkbox_units.CountMismatch as err:
        log["status"] = "stopped"
        log["failure"] = str(err)
    log["seconds"] = round(time.perf_counter() - start, 1)
    log["peak_rss_gb"] = peak_rss_gb()
    KKBOX_OUT.mkdir(parents=True, exist_ok=True)
    (KKBOX_OUT / "recount.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


FOLDS, FOLD_SEED, THREADS, MIN_COUNT = 5, 0, 4, 5


def _quantiles(v: np.ndarray) -> dict:
    q = np.quantile(v, [0, 0.25, 0.5, 0.75, 1])
    return dict(
        zip(["min", "q25", "median", "q75", "max"], (round(float(x), 4) for x in q), strict=True)
    )


def _arm_counts(t: np.ndarray) -> dict:
    return {"treated": int((t == 1).sum()), "untreated": int((t == 0).sum())}


MEMBERS_COLUMNS = ("tenure_days", "registered_via")


def _inputs(drop_members: bool = False):
    """Analysed units (counts checked), pre-T covariates, the coded design, transactions."""
    reg_counts = config.load()["kkbox"]["counts_before_outcome"]
    tx = kkbox_units.read_transactions(KKBOX / "transactions.parquet")
    reg = kkbox_units.registered_set(_units(kkbox_units.SNAPSHOTS, tx))
    analysed = kkbox_units.analysed_set(reg)
    kkbox_units.check_counts(kkbox_units.recount(reg, analysed), reg_counts)
    cov = kkbox_units.covariates(analysed, KKBOX / "features.parquet", tx)
    return analysed, cov, _design(cov, drop_members), tx


def _design(cov: pd.DataFrame, drop_members: bool = False) -> propensity.Design:
    fs1 = [c for c in kkbox_units.fs1_columns() if not (drop_members and c in MEMBERS_COLUMNS)]
    categorical = [c for c in fs1 if c in kkbox_units.CATEGORICAL]
    numeric = [c for c in fs1 if c not in categorical] + ["month_2016_09", "any_discount_365"]
    return propensity.build_design(cov, numeric, categorical, MIN_COUNT)


def propensity_step() -> dict:
    _require_clean()
    start = time.perf_counter()
    k = config.load()["kkbox"]
    p = k["propensity"]
    analysed, cov, design, tx = _inputs()
    t = analysed.treated.to_numpy()
    groups = analysed.msno.to_numpy()
    with threadpool_limits(limits=THREADS):
        fold_id = propensity.folds(t, groups, FOLDS, FOLD_SEED)
        c, losses = propensity.choose_c(design.x, t, fold_id, p["penalty"]["C_grid"])
        e = propensity.oof_propensity(design.x, t, fold_id, c)
    primary = p["trimming"]["primary"]
    log: dict = {
        "commit": _git("rev-parse", "--short", "HEAD").strip(),
        "units": _arm_counts(t),
        "inputs": list(design.x.columns),
        "n_inputs": design.x.shape[1],
        "dropped": design.dropped,
        "C_losses": {str(k_): round(v, 5) for k_, v in losses.items()},
        "C_chosen": c,
        "overlap": {
            "treated": _quantiles(e[t == 1]),
            "untreated": _quantiles(e[t == 0]),
            "outside_primary_band": _arm_counts(
                t[~propensity.trim_mask(e, primary["low"], primary["high"])]
            ),
        },
    }
    xb = balance.fill_for_balance(design.x)
    days = cov.days_since_last_discount.to_numpy()
    rules = {}
    KKBOX_OUT.mkdir(parents=True, exist_ok=True)
    for name in ("primary", "stricter", "none"):
        band = p["trimming"][name]
        keep = propensity.trim_mask(e, band["low"], band["high"])
        tk = t[keep]
        r: dict = {"band": band, "kept": _arm_counts(tk)}
        if min(r["kept"].values()) >= 2:
            w = propensity.stabilised_weights(tk, e[keep])
            r["kish_ess"] = {
                "treated": round(propensity.kish_ess(w[tk == 1]), 1),
                "untreated": round(propensity.kish_ess(w[tk == 0]), 1),
            }
            table = balance.balance_table(xb[keep].reset_index(drop=True), tk, w)
            table.to_parquet(KKBOX_OUT / f"balance_{name}.parquet", index=False)
            worst = balance.tied_max(table.smd_after.abs(), table.column)
            r["balance"] = {
                "columns": len(table),
                "failing": int((~table.passes).sum()),
                "max_abs_smd_after": round(float(table.smd_after.abs().max()), 3),
                "max_abs_smd_before": round(float(table.smd_before.abs().max()), 3),
                "worst_columns": worst,
                "passes": bool(table.passes.all()),
            }
            hist = keep & ~np.isnan(days)
            th = t[hist]
            if min((th == 1).sum(), (th == 0).sum()) >= 2:
                wh = propensity.stabilised_weights(th, e[hist])
                r["diagnostic_days_since_last_discount"] = {
                    "units_with_history": _arm_counts(th),
                    "smd_before": round(balance.smd(days[hist], th), 3),
                    "smd_after": round(balance.smd(days[hist], th, wh), 3),
                }
            else:
                r["diagnostic_days_since_last_discount"] = {
                    "units_with_history": _arm_counts(th),
                    "note": "fewer than 2 units with history in an arm; not computable",
                }
        else:
            r["note"] = "an arm has fewer than 2 units; weights and balance not computable"
        rules[name] = r
    log["trimming"] = rules
    kept = rules["primary"]["kept"]
    minimum = (
        k["expectations"]["min_treated_after_trimming"],
        k["expectations"]["min_untreated_after_trimming"],
    )
    log["estimability"] = {
        "rule": "at least 30 treated and 30 untreated after primary trimming",
        "kept_primary": kept,
        "verdict": "estimable"
        if kept["treated"] >= minimum[0] and kept["untreated"] >= minimum[1]
        else "not estimable",
    }
    scores = analysed[["msno", "snapshot_date", "treated"]].assign(propensity=e, fold=fold_id)
    scores.to_parquet(KKBOX_OUT / "propensity_scores.parquet", index=False)
    log["seconds"] = round(time.perf_counter() - start, 1)
    log["peak_rss_gb"] = peak_rss_gb()
    (KKBOX_OUT / "propensity.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def kept_composition() -> dict:
    _require_clean()
    p = config.load()["kkbox"]["propensity"]
    scores = pd.read_parquet(KKBOX_OUT / "propensity_scores.parquet")
    tx = kkbox_units.read_transactions(KKBOX / "transactions.parquet")
    analysed = kkbox_units.analysed_set(
        kkbox_units.registered_set(_units(kkbox_units.SNAPSHOTS, tx))
    )
    cov = kkbox_units.covariates(analysed, KKBOX / "features.parquet", tx)
    scores["snapshot_date"] = pd.to_datetime(scores.snapshot_date)
    d = scores.merge(
        cov[["msno", "snapshot_date", "any_discount_365"]],
        on=["msno", "snapshot_date"],
        validate="one_to_one",
    )
    d["month"] = d.snapshot_date.dt.strftime("%Y-%m")
    d["arm"] = np.where(d.treated == 1, "treated", "untreated")
    d["past_year_discount"] = np.where(d.any_discount_365 == 1, "yes", "no")
    log: dict = {"commit": _git("rev-parse", "--short", "HEAD").strip(), "kept": {}, "mirror": {}}
    for name in ("primary", "stricter", "none"):
        band = p["trimming"][name]
        kept = d[propensity.trim_mask(d.propensity.to_numpy(), band["low"], band["high"])]
        groups = kept.groupby(["month", "arm", "past_year_discount"]).size()
        log["kept"][name] = {"/".join(k): int(v) for k, v in groups.items()}
        t = pd.read_parquet(KKBOX_OUT / f"balance_{name}.parquet").set_index("column")
        m, e = t.loc["month_2016_09"], t.loc["days_until_expiry"]
        log["mirror"][name] = {
            "month_2016_09": [round(m.smd_before, 3), round(m.smd_after, 3)],
            "days_until_expiry": [round(e.smd_before, 3), round(e.smd_after, 3)],
            "mirror": bool(
                abs(m.smd_before + e.smd_before) < 1e-9 and abs(m.smd_after + e.smd_after) < 1e-9
            ),
        }
    (KKBOX_OUT / "kept_composition.json").write_text(json.dumps(log, indent=2) + "\n")
    return log


LOGS = Path("data/interim/kkbox/sample_020/user_logs.parquet")
RULES = ("primary", "stricter", "none")
B, BOOT_SEED = 1000, 0


def _fit_propensity(design: propensity.Design, t: np.ndarray, groups: np.ndarray, grid: list):
    with threadpool_limits(limits=THREADS):
        fold_id = propensity.folds(t, groups, FOLDS, FOLD_SEED)
        c, _ = propensity.choose_c(design.x, t, fold_id, grid)
        e = propensity.oof_propensity(design.x, t, fold_id, c)
    return fold_id, c, e


def _check_saved_scores(analysed: pd.DataFrame, e: np.ndarray, fold_id: np.ndarray) -> float:
    """The refit must reproduce the logged propensity step; stop otherwise."""
    saved = pd.read_parquet(KKBOX_OUT / "propensity_scores.parquet")
    now = analysed[["msno", "snapshot_date"]].assign(
        snapshot_date=pd.to_datetime(analysed.snapshot_date), propensity=e, fold=fold_id
    )
    saved["snapshot_date"] = pd.to_datetime(saved.snapshot_date)
    m = now.merge(saved, on=["msno", "snapshot_date"], suffixes=("", "_saved"), validate="1:1")
    diff = float((m.propensity - m.propensity_saved).abs().max())
    if len(m) != len(now) or diff > 1e-9 or not (m.fold == m.fold_saved).all():
        raise SystemExit(f"the refit propensity differs from the logged step (max {diff:.2e})")
    return diff


def _estimate(name, x, y, t, groups, kept, fold_id, e, c, kind, budget):
    """Point estimates on the kept units and their customer bootstrap, with the fit logs
    of the main estimate and of the bootstrap (summed over resamples)."""
    k = kept
    fit_log, boot_log = learners.FitLog(), learners.FitLog()
    try:
        est = learners.estimate(
            x[k].reset_index(drop=True),
            y[k],
            t[k],
            fold_id[k],
            e[k],
            kind,
            learners.KKBOX_FIXED,
            fit_log,
        )
        estimates = {"aipw": est.aipw, "t_learner": est.t_learner, "x_learner": est.x_learner}
    except learners.NotComputable as err:
        nan = float("nan")
        estimates = {"aipw": nan, "t_learner": nan, "x_learner": nan}
        result = kkbox_bootstrap.Result(estimates, "not computable", f"point estimate: {err}")
        return result, fit_log, boot_log
    stat = kkbox_bootstrap.effect_statistic(
        x, t, y, groups, kept, c, kind, learners.KKBOX_FIXED, boot_log
    )
    with threadpool_limits(limits=THREADS):
        result = kkbox_bootstrap.bootstrap(name, stat, estimates, groups, B, BOOT_SEED, budget)
    return result, fit_log, boot_log


def _fit_record(main: learners.FitLog, boot: learners.FitLog) -> dict:
    return {
        "main": main.summary(),
        "main_degenerate_fold_arm": main.degenerate,
        "bootstrap": boot.summary(),
    }


def _naive(name, y, t, groups, rows, budget):
    est = {"naive": learners.naive_difference(y[rows], t[rows])}
    stat = kkbox_bootstrap.naive_statistic(t, y, rows)
    return kkbox_bootstrap.bootstrap(name, stat, est, groups, B, BOOT_SEED, budget)


def outcome_step(rerun_reason: str | None = None) -> dict:
    """Placebo, then the one KKBox outcome read, then every registered estimate."""
    _require_clean()
    step_start = time.perf_counter()
    run_started = outcome.now()
    commit = _git("rev-parse", "--short", "HEAD").strip()
    outcome.guard_not_read(KKBOX_OUT, rerun_reason)
    k = config.load()["kkbox"]
    p = k["propensity"]
    analysed, cov, design, _ = _inputs()
    t = analysed.treated.to_numpy()
    groups = analysed.msno.to_numpy()
    fold_id, c, e = _fit_propensity(design, t, groups, p["penalty"]["C_grid"])
    max_diff = _check_saved_scores(analysed, e, fold_id)
    kept = {
        r: propensity.trim_mask(e, p["trimming"][r]["low"], p["trimming"][r]["high"]) for r in RULES
    }
    x = design.x
    sizes = {r: int(kept[r].sum()) for r in RULES}
    fits: dict = {}
    budget = kkbox_bootstrap.Budget(step_start)

    def later(*rules: str, base: str) -> float:
        return float(sum(B * sizes[r] / sizes[base] for r in rules))

    def compute_placebo() -> dict:
        y_p = placebo.placebo_days(LOGS, analysed)
        cats = placebo.leak_categories(cov)
        leak = placebo.leak_count(cats, t, kept["primary"])
        placebo.check_leak(leak)
        budget.later_work = later("primary", "stricter", "none", "primary", base="primary")
        res, main_log, boot_log = _estimate(
            "placebo", x, y_p, t, groups, kept["primary"], fold_id, e, c, "regressor", budget
        )
        fits["placebo"] = _fit_record(main_log, boot_log)
        sd = float(np.std(y_p[kept["primary"]], ddof=1))
        return {
            "label": report.LABEL,
            "interval_scope": report.INTERVAL_SCOPE,
            "outcome": "distinct days with a log row in days T-56 to T-29, inclusive",
            "leak_count": leak,
            "placebo_sd_kept": round(sd, 3),
            "power_limit_sd": {"by_counts": 0.53, "by_effective_sample_size": 0.74},
            "outcome_models": fits["placebo"],
            "rows": report.rows(
                res, "placebo, primary trimming", report.degenerate_note(main_log, False)
            ),
        }

    def write_log(result: dict) -> Path:
        return outcome.write_placebo_log(KKBOX_OUT / "placebo.json", result, commit, analysed)

    def read(log_path: Path) -> np.ndarray:
        return outcome.read_outcome(
            analysed,
            out_dir=KKBOX_OUT,
            placebo_log=log_path,
            commit=commit,
            run_started=run_started,
            protocol=load_protocol(),
            rerun_reason=rerun_reason,
        )

    placebo_result, _, y = outcome.placebo_then_outcome(compute_placebo, write_log, read)
    all_rows = list(placebo_result["rows"])
    order = list(RULES)
    for i, r in enumerate(order):
        # what remains after this bootstrap: the later rules and the sensitivity run
        budget.later_work = later(*order[i + 1 :], "primary", base=r)
        res, main_log, boot_log = _estimate(
            r, x, y, t, groups, kept[r], fold_id, e, c, "classifier", budget
        )
        fits[r] = _fit_record(main_log, boot_log)
        note = report.degenerate_note(main_log, primary=r == "primary")
        all_rows += report.rows(res, f"{r} trimming", note)
    every = np.ones(len(t), dtype=bool)
    all_rows += report.rows(_naive("naive_analysed", y, t, groups, every, None), "analysed set")
    all_rows += report.rows(
        _naive("naive_trimmed", y, t, groups, kept["primary"], None), "primary trimming"
    )
    # registered sensitivity: the same pipeline without tenure_days and registered_via
    design_s = _design(cov, drop_members=True)
    fold_s, c_s, e_s = _fit_propensity(design_s, t, groups, p["penalty"]["C_grid"])
    kept_s = propensity.trim_mask(
        e_s, p["trimming"]["primary"]["low"], p["trimming"]["primary"]["high"]
    )
    arms_s = {"treated": int(kept_s[t == 1].sum()), "untreated": int(kept_s[t == 0].sum())}
    mins = (
        k["expectations"]["min_treated_after_trimming"],
        k["expectations"]["min_untreated_after_trimming"],
    )
    sens: dict = {"label": report.LABEL, "C": c_s, "kept_primary": arms_s}
    if arms_s["treated"] >= mins[0] and arms_s["untreated"] >= mins[1]:
        budget.later_work = 0.0
        tk = t[kept_s]
        w = propensity.stabilised_weights(tk, e_s[kept_s])
        xb = balance.fill_for_balance(design_s.x)
        table_s = balance.balance_table(xb[kept_s].reset_index(drop=True), tk, w)
        sens["verdict"] = "estimable"
        sens["balance_failing"] = int((~table_s.passes).sum())
        sens["balance_columns"] = len(table_s)
        res, main_log, boot_log = _estimate(
            "without_members",
            design_s.x,
            y,
            t,
            groups,
            kept_s,
            fold_s,
            e_s,
            c_s,
            "classifier",
            budget,
        )
        fits["without_members"] = _fit_record(main_log, boot_log)
        all_rows += report.rows(
            res,
            "primary trimming, without tenure and registration channel",
            report.degenerate_note(main_log, primary=False),
        )
    else:
        sens["verdict"] = "not estimable"
    table = report.table(all_rows)
    table.to_parquet(KKBOX_OUT / "outcome_table.parquet", index=False)
    log = {
        "label": report.LABEL,
        "reading": report.READING,
        "interval_scope": report.INTERVAL_SCOPE,
        "commit": commit,
        "run_started": run_started.isoformat(),
        "rerun_reason": rerun_reason,
        "propensity_refit_max_abs_diff": max_diff,
        "kept": {
            r: {"treated": int(kept[r][t == 1].sum()), "untreated": int(kept[r][t == 0].sum())}
            for r in RULES
        },
        "placebo": {key: v for key, v in placebo_result.items() if key != "rows"},
        "table": table.to_dict(orient="records"),
        "sentences": [report.sentence(r) for r in all_rows],
        "sensitivity_without_members": sens,
        "outcome_models": fits,
        "budget": budget.projections,
        "seconds": round(time.perf_counter() - step_start, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    text = json.dumps(log, indent=2, default=str) + "\n"
    (KKBOX_OUT / "outcome_first_run.json").write_text(text)
    (KKBOX_OUT / outcome.LOG).write_text(text)
    return log


PUBLIC_OUT = Path("data/processed/public")


def validate(rerun_reason: str | None = None) -> dict:
    _require_clean()
    public_validation.guard(PUBLIC_OUT, rerun_reason)
    hill = pd.read_parquet(OUT / "hillstrom.parquet")
    crit = pd.read_parquet(OUT / "criteo_uplift_v2.1.parquet")
    commit = _git("rev-parse", "--short", "HEAD").strip()
    with threadpool_limits(limits=THREADS):
        log = public_validation.run_validation(hill, crit, PUBLIC_OUT, commit, rerun_reason)
    log["peak_rss_gb"] = peak_rss_gb()
    public_validation.write_log(PUBLIC_OUT, log)
    return {
        "commit": log["commit"],
        "seconds": log["seconds"],
        "peak_rss_gb": log["peak_rss_gb"],
        "log": str(PUBLIC_OUT / public_validation.LOG),
    }


def qini_figures() -> dict:
    log = json.loads((PUBLIC_OUT / public_validation.LOG).read_text())
    note = (
        f"drawn from {PUBLIC_OUT / figures.CURVES_NAME}, "
        f"validation run at commit {log['commit']}; no dataset row is read"
    )
    paths = figures.write_figures(PUBLIC_OUT / figures.CURVES_NAME, Path("docs/img"), note)
    return {"figures": [str(p) for p in paths]}


def groups_step() -> dict:
    _require_clean()
    hill = pd.read_parquet(OUT / "hillstrom.parquet")
    crit = pd.read_parquet(OUT / "criteo_uplift_v2.1.parquet")
    log = json.loads((PUBLIC_OUT / public_validation.LOG).read_text())
    scores = pd.read_parquet(PUBLIC_OUT / "uplift_scores.parquet")
    start = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        out = groups.run_groups(hill, crit, scores, log, PUBLIC_OUT)
    out["commit"] = _git("rev-parse", "--short", "HEAD").strip()
    out["seconds"] = round(time.perf_counter() - start, 1)
    out["peak_rss_gb"] = peak_rss_gb()
    (PUBLIC_OUT / "groups.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


COMMANDS = {
    "check-public": check_public,
    "recount": recount,
    "propensity": propensity_step,
    "kept-composition": kept_composition,
    "outcome": outcome_step,
    "validate": validate,
    "qini-figures": qini_figures,
    "groups": groups_step,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.uplift")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument(
        "--rerun-reason", default=None, help="outcome and validate: why an attempt repeats"
    )
    args = parser.parse_args()
    run = COMMANDS[args.command]
    log = run(args.rerun_reason) if args.command in ("outcome", "validate") else run()
    print(json.dumps(log, indent=2, default=str))
    if log.get("status") == "stopped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
