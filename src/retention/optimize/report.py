"""reports/strategy_backtest.md, rendered only from the logged runs, with fixed templates.

Every number in the report comes from a logged file under
data/processed/kkbox/<sample>/optimize; nothing is typed by hand. The wording
rules of configs/actions.yaml are checked on the rendered text before it is
written (wording_violations): banned words outside the fixed disclaimers, the
"Under the assumed <form> <level> effect" prefix on every sentence that uses an
effect, a retained value or a net value, "uplift" only as "assumed uplift",
"retained" only inside the registered quantity names, the caption on every
comparison table, and no figure or name from the public uplift datasets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from retention.optimize.config import ActionsConfig

REPORT = Path("reports/strategy_backtest.md")
STRATEGY_NAMES = {
    "optimizer": "optimizer",
    "highest_risk": "highest risk (best action)",
    "highest_assumed_uplift_x_value": "highest assumed uplift x value (best action)",
    "random": "random (best action)",
    "fixed_action_reminder": "highest risk (reminder only)",
    "no_action": "no action",
}
QUANTITY_WORDS = ("expected retained", "net value", "assumed uplift")
COMPARISON_MARK = "<!-- comparison table -->"


def money(x: float) -> str:
    return f"{x:,.0f}"


def num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}"


def pct(x: float | None) -> str:
    return "not defined" if x is None else f"{100 * x:.1f}%"


def budget_label(b: float | None) -> str:
    return "no limit" if b is None or (isinstance(b, float) and pd.isna(b)) else money(b)


def prefix(form: str, level: str) -> str:
    return f"Under the assumed {form} {level} effect"


def caption_lines(cfg: ActionsConfig, fields: dict[str, Any], budget: str) -> list[str]:
    second = (
        f"effect_basis: assumed; form {fields['form']}; level {fields['level']}; "
        f"margin {fields['margin']}; cost level {fields['cost_level']}; discount cost "
        f"{'scaled by margin' if fields['discount_cost_scaled_by_margin'] else 'in full'}; "
        f"value horizon {fields['value_horizon_days']} days; budget {budget}"
    )
    return [COMPARISON_MARK, f"*{cfg.caption}.*", f"*{second}.*", ""]


def comparison_table(cfg: ActionsConfig, rows: pd.DataFrame, fields: dict[str, Any]) -> list[str]:
    """One comparison table (one snapshot, scenario and budget), caption first."""
    budget = budget_label(rows.budget.iloc[0])
    out = caption_lines(cfg, fields, budget)
    out += [
        "| Strategy | Contacted | Expected retained customers | Expected retained value "
        "| Expected cost | Net value |",
        "|---|---|---|---|---|---|",
    ]
    for key, name in STRATEGY_NAMES.items():
        r = rows[rows.strategy == key]
        if r.empty:
            continue
        r = r.iloc[0]

        def cell(m: str, fmt=money, r=r, key=key) -> str:
            if key == "random":
                return f"{fmt(r[m])} ({fmt(r[m + '_p2_5'])} to {fmt(r[m + '_p97_5'])})"
            return fmt(r[m])

        out.append(
            f"| {name} | {cell('contacted', lambda v: f'{v:,.0f}')} "
            f"| {cell('expected_retained_customers', num)} | {cell('expected_retained_value')} "
            f"| {cell('expected_cost')} | {cell('net_value')} |"
        )
    out.append("")
    return out


def render(cfg: ActionsConfig, logs: dict[str, Any]) -> str:
    """The whole report from the logged runs (see load_logs)."""
    plan = logs["plan_report"]
    summary: pd.DataFrame = logs["summary"]
    stab: pd.DataFrame = logs["stability"]
    recount, value, refit = logs["recount"], logs["value_report"], logs["dt01_refit"]
    base = cfg.base
    p = prefix(base.form, base.level)
    head = plan["headline_budget"]
    head_label = budget_label(None if head == "no_limit" else float(head))
    L: list[str] = ["# Strategy backtest", ""]
    L += [cfg.disclaimers[0] + " " + cfg.disclaimers[1], ""]
    L += [
        "Every effect, cost and margin below is a labelled assumption "
        f"({cfg.label}). Money is in price units, the unit of the KKBox price columns.",
        "",
    ]

    L += ["## Inputs", ""]
    nov = recount["per_snapshot"][base.snapshot.isoformat()]
    L += [
        f"Customers on {base.snapshot.isoformat()}: {nov['customers']:,} "
        f"({nov['auto_renew_off']:,} with auto-renew off, {nov['auto_renew_on']:,} on). "
        f"No usable price: {nov['no_usable_price']:,}. "
        f"Missing deadline: {nov['missing_deadline']:,}.",
        "",
        "Paid rows dated 2016-01-01 to 2016-11-30, most common plans:",
        "",
        "| Plan days | List price | Paid | Rows | Share |",
        "|---|---|---|---|---|",
    ]
    for r in recount["price_table"]["top"][:5]:
        L.append(
            f"| {r['plan_days']} | {r['list_price']} | {r['paid']} | {r['n']:,} "
            f"| {pct(r['share'])} |"
        )
    L.append("")
    L += [
        f"The dt01 refit reproduced the rehearsal's logged score summaries exactly "
        f"({len(refit['reproduction'])} cells). Risk comes from c01 and value from dt01, two "
        "different models: of the customers in c01's top decile of risk, "
        f"{value['per_snapshot'][base.snapshot.isoformat()]['model_disagreement']['disagree']:,} "
        "have a dt01 30-day lapse probability below dt01's median.",
        "",
    ]

    L += [f"## Comparison at the headline budget ({head_label}), {base.snapshot.isoformat()}", ""]
    rows = summary[
        (summary.snapshot == base.snapshot.isoformat())
        & (summary.scenario == "base")
        & (summary.budget_label == head_label)
    ]
    L += comparison_table(cfg, rows, base.fields())
    opt = rows[rows.strategy == "optimizer"].iloc[0]
    L += [
        f"{p}, the optimizer's expected retained value is {money(opt.expected_retained_value)} "
        f"price units at an expected cost of {money(opt.expected_cost)}, a net value of "
        f"{money(opt.net_value)}.",
        "",
    ]

    L += ["## Budget curve (base point)", "", "| Budget | Net value | Extra net per extra unit |"]
    L += ["|---|---|---|"]
    for b, net, extra in plan["budget_curve"]:
        L.append(f"| {budget_label(b)} | {money(net)} | {'' if extra is None else num(extra, 3)} |")
    L.append("")

    L += ["## Sensitivity", ""]
    v = plan["verdict"]
    L += [
        "Stability thresholds (the user's choice, not based on data): Jaccard at least "
        f"{cfg.jaccard_min} and relative regret at most {pct(cfg.regret_max)}; the stricter "
        f"reading ({cfg.stricter[0]}, {pct(cfg.stricter[1])}) is shown, not judged.",
        "",
        "| Variant | Loss | Relative regret | Jaccard | Action agreement | Stable | Stricter |",
        "|---|---|---|---|---|---|---|",
    ]
    at = stab[stab.budget_label == head_label]
    for r in at.itertuples():
        L.append(
            f"| {r.variant} | {money(r.loss)} | {pct(_none(r.relative_regret))} "
            f"| {num(r.jaccard, 2)} "
            f"| {pct(_none(r.action_agreement))} | {_yes(r.stable)} | {_yes(r.stable_stricter)} |"
        )
    L.append("")
    L += [f"Overall verdict at the headline budget: {v['overall']}.", ""]
    if v["failed"]:
        L += [
            "The plan is driven by assumptions, not data. Variants that failed: "
            f"{', '.join(v['failed'])}.",
            "",
            "Assumptions ranked by loss (largest first): "
            + "; ".join(f"{n} {money(x)}" for n, x in v["ranked_by_loss"])
            + ".",
            "",
        ]
    L += [
        f"{prefix(base.form, 'low')} (zero for every action), the base plan spends "
        f"{money(v['low_effect_loss'])} price units for nothing.",
        "",
        v["robust_core_sentence"],
        "",
        f"Discount cost rule: {v['discount_sentence']}.",
        "",
    ]
    if v["failed"]:
        L += [
            "Recommendation: a randomized holdout test of the action behind the largest loss "
            "before any use of this plan.",
            "",
        ]
    L += ["Expectations registered before the run:", ""]
    L += [f"- {e}" for e in plan["expectations"]] + [""]

    L += ["## Secondary snapshot", ""]
    sec = cfg.snapshots[0].isoformat()
    L += [
        f"{sec} holds the half of customers used to fit the Platt calibration, so its "
        "probabilities are partly in-sample.",
        "",
    ]
    rows = summary[
        (summary.snapshot == sec)
        & (summary.scenario == "base")
        & (summary.budget_label == head_label)
    ]
    if not rows.empty:
        L += comparison_table(cfg, rows, base.fields() | {"snapshot": sec})

    obs = logs.get("observed")
    if obs:
        L += ["## Observed churn among targeted customers", ""]
        L += [f"*{cfg.observed_label}.*", ""]
        L += [
            "| Snapshot | Strategy | Targeted | Share lapsed within 30 days |",
            "|---|---|---|---|",
        ]
        for snap, by in obs["shares"].items():
            for key, name in STRATEGY_NAMES.items():
                if key in by and by[key]["targeted"]:
                    share = pct(by[key]["share_lapsed"])
                    L.append(f"| {snap} | {name} | {by[key]['targeted']:,} | {share} |")
        L.append("")

    L += ["## Solver checks", ""]
    L += solver_lines(cfg, plan["solver_summary"])
    return "\n".join(L).rstrip() + "\n"


def solver_lines(cfg: ActionsConfig, chk: dict[str, Any]) -> list[str]:
    """The exact check's summary, with the unproven count always stated."""
    limit = f"{chk['time_limit_seconds']:.0f} s"
    out = [
        f"Cells checked by HiGHS: {chk['cells']}; proven optimal within {limit}: {chk['proven']}; "
        f"not proven: {chk['unproven']}, of which with no feasible HiGHS solution: "
        f"{chk.get('no_feasible_solution', 0)}. Largest HiGHS minus greedy on proven cells: "
        f"{num(chk['max_highs_minus_greedy'], 6)}. Budget or limit breaches: {chk['breaches']}; "
        f"repairs: {chk['repairs']}.",
        "",
    ]
    if chk["unproven_share"] > cfg.unproven_statement_share:
        out += [
            f"The exact check is limited: in {chk['unproven']} of {chk['cells']} cells "
            f"({pct(chk['unproven_share'])}) HiGHS did not prove the optimum within {limit}, so "
            "the greedy plan there is checked only against its LP bound.",
            "",
        ]
    return out


def _none(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else x


def _yes(x: Any) -> str:
    x = _none(x)
    return "not judged" if x is None else ("yes" if x else "no")


# --- wording rules ----------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    """Sentences of the prose lines (tables, headings and marks excluded), line by line."""
    out = []
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith(("|", "#", "<!--")):
            continue
        line = ln.strip().strip("*").removeprefix("- ")
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return out


def wording_violations(text: str, cfg: ActionsConfig, external_scope: bool = True) -> list[str]:
    out = []
    lower = text.lower()
    for d in sorted(cfg.disclaimers, key=len, reverse=True):
        lower = lower.replace(d.lower(), " ")
    for w in cfg.banned_words:
        if re.search(rf"\b{re.escape(w)}\b", lower):
            out.append(f"banned word {w!r}")
    stripped = lower
    for ok in cfg.retained_allowed:
        stripped = stripped.replace(ok, " ")
    if re.search(r"\bretained\b", stripped):
        out.append("'retained' outside the registered quantity names")
    if re.search(r"\buplift\b", lower.replace(cfg.uplift_allowed, " ")):
        out.append("'uplift' without 'assumed'")
    for s in _sentences(text):
        if s in cfg.disclaimers or any(s.startswith(d) for d in cfg.disclaimers):
            continue
        uses = any(q in s.lower() for q in QUANTITY_WORDS) and re.search(r"\d", s)
        if uses and not s.startswith("Under the assumed "):
            out.append(f"effect sentence without the prefix: {s[:80]!r}")
    if external_scope:
        for name in cfg.external_names:
            if name.lower() in text.lower():
                out.append(f"external dataset name {name!r}")
    lines = text.splitlines()
    for k, ln in enumerate(lines):
        if ln == COMPARISON_MARK and (k + 1 >= len(lines) or lines[k + 1] != f"*{cfg.caption}.*"):
            out.append("comparison table without the registered caption")
    n_tables = sum(1 for ln in lines if ln.startswith("| Strategy |"))
    if n_tables != lines.count(COMPARISON_MARK):
        out.append("a comparison table is not marked and captioned")
    return out


def load_logs(out_dir: Path) -> dict[str, Any]:
    def j(name: str) -> dict[str, Any]:
        return json.loads((out_dir / name).read_text())

    logs = {
        "recount": j("recount.json"),
        "dt01_refit": j("dt01_refit.json"),
        "value_report": j("value_report.json"),
        "plan_report": j("plan_report.json"),
        "summary": pd.read_parquet(out_dir / "summary.parquet"),
        "stability": pd.read_parquet(out_dir / "stability.parquet"),
    }
    if (out_dir / "observed_churn.json").exists():
        logs["observed"] = j("observed_churn.json")
    return logs
