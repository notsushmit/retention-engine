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
from retention.uplift import balance, config, kkbox_units, propensity, public_data

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


def propensity_step() -> dict:
    _require_clean()
    start = time.perf_counter()
    k = config.load()["kkbox"]
    reg_counts, p = k["counts_before_outcome"], k["propensity"]
    tx = kkbox_units.read_transactions(KKBOX / "transactions.parquet")
    reg = kkbox_units.registered_set(_units(kkbox_units.SNAPSHOTS, tx))
    analysed = kkbox_units.analysed_set(reg)
    kkbox_units.check_counts(kkbox_units.recount(reg, analysed), reg_counts)
    cov = kkbox_units.covariates(analysed, KKBOX / "features.parquet", tx)
    fs1 = kkbox_units.fs1_columns()
    categorical = [c for c in fs1 if c in kkbox_units.CATEGORICAL]
    numeric = [c for c in fs1 if c not in categorical] + ["month_2016_09", "any_discount_365"]
    design = propensity.build_design(cov, numeric, categorical, MIN_COUNT)
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
            worst = table.loc[table.smd_after.abs().idxmax()]
            r["balance"] = {
                "columns": len(table),
                "failing": int((~table.passes).sum()),
                "max_abs_smd_after": round(float(abs(worst.smd_after)), 3),
                "max_abs_smd_before": round(float(table.smd_before.abs().max()), 3),
                "worst_column": worst.column,
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


COMMANDS = {"check-public": check_public, "recount": recount, "propensity": propensity_step}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.uplift")
    parser.add_argument("command", choices=list(COMMANDS))
    args = parser.parse_args()
    run = COMMANDS[args.command]
    log = run()
    print(json.dumps(log, indent=2, default=str))
    if log.get("status") == "stopped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
