"""reports/model_comparison.md, built only from the logged test run.

Every number comes from the test run's test_results.json and tags, or is a
registered setting read from configs/evaluation.yaml (labelled as such). The
wording of every verdict is the registered wording. After every interval the
registered sentence on what the interval covers is printed.
"""

from __future__ import annotations

from typing import Any

from retention.models.report import DESCRIPTIVE_BREAKDOWNS, DESCRIPTIVE_MODELS, REQUIRED_NOTES

COMPARATORS = {
    "lightgbm_classifier": ("classifier", "the LightGBM classifier"),
    "strongest_simple_baseline": ("banded_logistic", "the strongest simple baseline"),
}
MODEL_NAMES = {
    "survival_dt01": "Survival, discrete-time dt01 (registered pick)",
    "classifier": "LightGBM classifier c01",
    "banded_logistic": "Banded logistic (strongest simple baseline)",
    "survival_dt02": "Survival, discrete-time dt02 (descriptive, added after the fold results)",
}
CONCORDANCE_NOTE = (
    "Concordance is cut at each month's stop day, so it is not compared across months."
)
PR_AUC_NOTE = "PR AUC depends on the event rate, so it is compared only within a month."
# Stated in configs/evaluation.yaml before scoring (calibration section).
PREREGISTERED_INTERCEPT = (
    "Stated before scoring: the logged 7-day P2 rates are 1.32% in 2016-10 (the calibration "
    "month) and 1.78% in 2017-01. If the features do not explain the January rise, a model "
    "calibrated on October has an intercept of about logit(0.0178) - logit(0.0132) = -4.0106 - "
    "(-4.3143) = +0.30 at 7 days in 2017-01, outside the limit. The limits above were not set "
    "around this."
)
# Limits found after the test run by read-only checks (counts from 2016 data and
# daily payment-row counts, not from the test run); labelled as such in the report.
EXPLORATORY_LIMITS = [
    "The month-level data-quality screen can miss a single missing day: its batch indicator "
    "looks only at the month's last day, and the cohort indicators can stay under their "
    "limits. A daily batch-size indicator is needed for drift monitoring.",
    "Exploratory, post hoc (read-only, 2016 data): on 2016-12-05 the sample has 1,991 "
    "auto-renew payment rows against 4,929 to 5,254 on the 5th of September to November 2016. "
    "Of the 5,430 7-day lapses of the 2016-12-01 snapshot (P2, main segments), 3,339 paid "
    "again in January 2017; 3,129 of those had coverage ending on 2016-12-05 and 3,092 paid on "
    "2017-01-05, a 31-day gap, one day past the 30-day renewal window. 2016-12-01 is not a test "
    "month and was never a usable 7-day snapshot; none of these customers is a row of the "
    "2017-01-01 snapshot. It was used as a stand-in month in a rehearsal of this stage, and "
    "those rehearsal numbers are not reported.",
    "Exploratory, post hoc (daily counts of auto-renew payment rows, rule written before the "
    "counts were read; a day is a gap below 60% of the median of the same day over the three "
    "preceding months): gap days in 2016-09 to 2017-03 are 2016-12-05 (ratio 0.395), "
    "2016-12-30 (0.013, an artifact of the rule: the month-end batch fell on the 31st, with "
    "38,508 rows) and 2017-02-13 (2,080 rows against a baseline of 4,822; ratio 0.431). The "
    "renewals deciding a test month's 7-day outcomes are dated from T to T + 37 days: "
    "2017-01-01 to 2017-02-07 for 2017-01 (no gap day) and 2017-02-01 to 2017-03-10 for "
    "2017-02 (contains 2017-02-13). The 2017-02 7-day cell could be affected by missing "
    "renewals: missing renewals make renewing customers look lapsed, so the direction is "
    "upward for its observed rate; the size is not estimated. Reasoning, not shown: the same "
    "day also lies in the renewal windows of the 2017-01 30-day cell and the 59-day "
    "demonstration. No verdict or number in this report changes.",
]
PREREGISTERED_EXPECTATIONS = [
    (
        "Cox and XGBoost AFT at 7 days",
        "likely loss to the 7-day classifier",
        "not scored on test: only the frozen family (discrete-time) runs on the test months",
    ),
    ("Discrete-time at 7 days", "likely tie with the 7-day classifier", None),
    ("Any survival model at 30 days", "no structural edge", None),
]


def month(m: str) -> str:
    return m[:7]


def cell_label(cid: str) -> str:
    """'7 days, 2017-01-01' -> '7 days, 2017-01'."""
    if ", " not in cid:
        return month(cid)
    h, m = cid.split(", ", 1)
    return f"{h}, {month(m)}" if m[:4].isdigit() else cid


def failed_text(entry: str) -> str:
    """A goal summary's failed cell in plain words."""
    where, verdict = entry.rsplit(": ", 1)
    h, m, against = where.split(", ")
    if against == "survival":
        return f"{h} in {month(m)}, calibration ({verdict})"
    who = COMPARATORS[against][1]
    return f"{h} in {month(m)}, against {who} ({verdict_text(verdict, who, int(h.split()[0]), m)})"


def check_report(text: str, interval_note: str) -> list[str]:
    """Problems with the fixed rules: the interval sentence after every block with an
    interval, no decision IDs, no absolute paths."""
    import re

    problems = []
    blocks = re.split(r"\n\s*\n", text)
    for i, block in enumerate(blocks):
        if re.search(r"\[[+-]?\d+\.\d+, [+-]?\d+\.\d+\]", block) and interval_note not in block:
            following = blocks[i + 1] if i + 1 < len(blocks) else ""
            if following.strip() != interval_note:
                problems.append(f"no interval sentence after: {block[:60]!r}")
    if re.search(r"\bD-\d{3}\b", text):
        problems.append("a decision ID")
    if "/Users/" in text or re.search(r"(^|\s)/(home|private|tmp|var)/", text):
        problems.append("an absolute path")
    return problems


def fmt(x: float, digits: int = 4, signed: bool = False) -> str:
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def interval(v: dict[str, Any], digits: int = 4, signed: bool = False) -> str:
    if v.get("status") != "ok":
        return f"not computable ({v.get('reason', '')})"
    return (
        f"{fmt(v['point'], digits, signed)} "
        f"[{fmt(v['lo'], digits, signed)}, {fmt(v['hi'], digits, signed)}]"
    )


def value(v: dict[str, Any], digits: int = 4) -> str:
    return fmt(v["value"], digits) if v.get("status") == "ok" else "not computable"


def verdict_text(verdict: str, label: str, h: int, m: str) -> str:
    """The registered wording of a discrimination verdict."""
    return {
        "win": f"survival beats {label} at {h} days in {month(m)}",
        "below_threshold": "better by less than the 0.01 minimum; not a win",
        "no_difference": "no difference shown",
        "loss": f"survival is worse than {label}",
        "mixed": "mixed: not shown at 7 days",
        "not_computable": "not computable",
        "missing": "missing (counts as not met)",
    }[verdict]


def size_note(comp: dict[str, Any], verdict: str, minimum: float) -> str:
    """For a loss smaller than the minimum difference: its size, beside the unchanged verdict."""
    if verdict != "loss" or abs(comp["point"]) >= minimum:
        return ""
    return (
        f" (size {abs(comp['point']):.4f}, below the {minimum} minimum difference; the "
        "registered loss rule has no size threshold, so the verdict stays a loss)"
    )


def calibration_text(cv: dict[str, Any], limits: dict[str, Any]) -> str:
    if cv.get("status") != "ok":
        return "not computable (counts as not met)"
    s, i = limits["slope"], limits["intercept"]
    word = "passes" if cv["pass"] else "fails"
    text = f"{word} the limits (slope {s[0]} to {s[1]}, intercept {i[0]:+} to {i[1]:+})"
    return text + (", borderline: an interval crosses a limit" if cv["borderline"] else "")


class Doc:
    def __init__(self, interval_note: str) -> None:
        self.lines: list[str] = []
        self.note = interval_note

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def table(self, header: list[str], rows: list[list[str]], intervals: bool = True) -> None:
        self.add("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
        self.add(*["| " + " | ".join(r) + " |" for r in rows])
        self.add("")
        if intervals:
            self.add(self.note, "")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def render_report(results: dict[str, Any], meta: dict[str, Any]) -> str:
    """The report text. `meta`: run id, tags, registered settings and the commit history."""
    raw = meta["config"]
    limits = raw["calibration"]
    minimum = raw["discrimination"]["min_difference"]
    doc = Doc(meta["interval_note"])
    primary = results["primary"]
    goal_rows = results["goal_cells"]
    summary = results["goal_summary"]
    cells7 = [k for k in primary if k.startswith("7 days")]
    cell30 = [k for k in primary if k.startswith("30 days")]

    # --- summary paragraph -----------------------------------------------------------
    parts = []
    for cid in cells7 + cell30:
        h, m = int(cid.split()[0]), cid.split(", ")[1]
        against = []
        for key, (model, label) in COMPARATORS.items():
            v = results["goal_results"].get(f"discrimination|{h}|{m}|{key}", "missing")
            comp = primary[cid]["comparisons"][f"survival_dt01 minus {model}"]
            note = size_note(comp, v, minimum)
            against.append(f"against {label}: {verdict_text(v, label, h, m)}{note}")
        parts.append(f"at {h} days in {month(m)}, " + "; ".join(against))
    cal = []
    for cid in cells7 + cell30:
        h, m = int(cid.split()[0]), cid.split(", ")[1]
        cv = primary[cid].get("calibration_verdict_survival", {"status": "missing"})
        cal.append(f"at {h} days in {month(m)} it {calibration_text(cv, limits)}")
    d, c = summary["discrimination"], summary["calibration"]
    para = (
        "This report scores the frozen models once on the test months (MLflow test run "
        f"{meta['run_id']}). Discrimination of the survival model (discrete-time dt01), "
        "verdicts at the registered minimum difference of 0.01 in AUC: "
        + "; ".join(parts)
        + ". Calibration of the survival model's final probabilities: "
        + "; ".join(cal)
        + ". The goal that survival beats the baseline on the test snapshot is "
        + ("met" if d["met"] else "not met")
        + f" ({d['cells_met']} of {d['cells']} cells met"
        + ("" if d["met"] else "; failed: " + "; ".join(failed_text(x) for x in d["not_met"]))
        + "). The goal that calibration sits near the diagonal is "
        + ("met" if c["met"] else "not met")
        + f" ({c['cells_met']} of {c['cells']} cells met"
        + ("" if c["met"] else "; failed: " + "; ".join(failed_text(x) for x in c["not_met"]))
        + "). Not meeting a goal is a legitimate outcome of a pre-registered test."
    )
    doc.add(
        "# Model comparison: the test months",
        "",
        "Generated by `python -m retention.models report` from the logged MLflow test run; "
        "do not edit by hand.",
        "",
        "## Summary",
        "",
        para,
        "",
    )

    rows = []
    for r in goal_rows:
        h, m = r["horizon"], r["month"]
        cid = f"{h} days, {m}"
        if r["kind"] == "discrimination":
            model, label = COMPARATORS[r["against"]]
            comp = primary[cid]["comparisons"][f"survival_dt01 minus {model}"]
            rows.append(
                [
                    f"{h} days, {month(m)}",
                    f"Discrimination against {label}",
                    "AUC difference " + interval(comp, signed=True),
                    verdict_text(r["status"], label, h, m) + size_note(comp, r["status"], minimum),
                    "met" if r["met"] else "not met",
                ]
            )
        else:
            sm = primary[cid]["models"]["survival_dt01"]
            cv = primary[cid]["calibration_verdict_survival"]
            rows.append(
                [
                    f"{h} days, {month(m)}",
                    "Calibration of the survival model",
                    f"slope {interval(sm['slope'], 3)}; "
                    f"intercept (slope fixed at 1) {interval(sm['intercept'], 3, signed=True)}",
                    calibration_text(cv, limits),
                    "met" if r["met"] else "not met",
                ]
            )
    doc.add("### Summary table", "")
    doc.table(["Cell", "Criterion", "Result", "Registered verdict", "Goal cell"], rows)
    comb = results["combined_7_day"]
    doc.add(
        "Both 7-day months together (a cell wins at 7 days only if both months win): "
        + "; ".join(
            f"against {label}: {verdict_text(comb[key]['combined'], label, 7, 'both')}"
            for key, (_, label) in COMPARATORS.items()
        )
        + ".",
        "",
    )

    # --- set-up ----------------------------------------------------------------------
    tags = meta["tags"]
    doc.add(
        "## Set-up",
        "",
        f"- Test run: {meta['run_id']}; code at commit {tags['git_commit']} "
        f"(working tree clean: {'yes' if tags['git_dirty'] == 'false' else 'no'}); "
        "configuration SHA-256 "
        f"{tags['config_sha256']}; frozen choices SHA-256 {tags['choices_sha256']}; "
        f"binding pre-registration commit {tags['binding_commit']}.",
        "- Models (frozen before the test): feature set "
        f"{meta['choices']['feature_set']}; LightGBM classifier "
        f"{meta['choices']['classifier_hyperparameters'][7]} at 7, 30 and 59 days; survival "
        f"family {meta['choices']['survival_family']['family']} "
        f"({meta['choices']['survival_family']['candidate']}); Platt calibration at every "
        "horizon; strongest simple baseline banded logistic. Fit on 2016-08 and 2016-09, "
        "calibrated on the calibration half of 2016-10, labels as of 2017-01-01; test outcomes "
        "as of 2017-03-31.",
        f"- Bootstrap: {meta['replicates']:,} customer-clustered replicates (registered), "
        "95% percentile intervals.",
        f"- Runtime {float(tags['timing.seconds']):.0f} s; peak memory of the main process "
        f"{float(tags['timing.peak_rss_gb']):.2f} GB.",
        "",
    )
    rows = []
    for cid, cellv in list(primary.items()) + list(results["demonstration_59"].items()):
        rows.append(
            [
                cell_label(cid),
                f"{cellv['rows']:,}",
                f"{cellv['lapses']:,}",
                f"{cellv['lapses'] / cellv['rows']:.2%}",
                str(cellv["stop_day"]),
            ]
        )
    doc.add("Scored cells (P2, main segments: paid and 100%-discount-only):", "")
    doc.table(["Cell", "Rows", "Lapses", "Lapse rate", "Stop day"], rows, intervals=False)

    # --- discrimination ---------------------------------------------------------------
    doc.add("## Discrimination", "", PR_AUC_NOTE, "")
    for cid in cells7 + cell30:
        cellv = primary[cid]
        h, m = int(cid.split()[0]), cid.split(", ")[1]
        doc.add(f"### {h} days, {month(m)}", "")
        rows = [
            [
                MODEL_NAMES[name],
                interval(mv["auc_raw"]),
                value(mv["pr_auc"]),
                fmt(mv["log_loss"]) if "log_loss" in mv else "",
                fmt(mv["brier"], 5) if "brier" in mv else "",
            ]
            for name, mv in cellv["models"].items()
        ]
        doc.table(["Model", "AUC (raw scores)", "PR AUC", "Log loss", "Brier"], rows)
        doc.add(PR_AUC_NOTE, "")
        rows = [
            [
                comp,
                interval(v, signed=True),
                v["verdicts"]["0.01"],
                v["verdicts"]["0.005"],
                v["verdicts"]["0.02"],
            ]
            for comp, v in cellv["comparisons"].items()
        ]
        doc.table(
            ["Paired AUC difference", "d [95% interval]", "Verdict at 0.01", "at 0.005", "at 0.02"],
            rows,
        )
    rows = []
    for what, expected, fixed in PREREGISTERED_EXPECTATIONS:
        if fixed is not None:
            observed = fixed
        elif what.startswith("Discrete-time"):
            observed = "; ".join(
                f"{month(cid.split(', ')[1])}: "
                + verdict_text(
                    primary[cid]["comparisons"]["survival_dt01 minus classifier"]["verdicts"][
                        "0.01"
                    ],
                    "the LightGBM classifier",
                    7,
                    cid.split(", ")[1],
                )
                for cid in cells7
            )
        else:
            v = primary[cell30[0]]["comparisons"]["survival_dt01 minus classifier"]
            observed = (
                "against the LightGBM classifier: "
                + verdict_text(v["verdicts"]["0.01"], "the LightGBM classifier", 30, "")
                + f" ({interval(v, signed=True)})"
            )
        rows.append([what, expected, observed])
    doc.add("### Pre-registered expectations beside the results", "")
    doc.table(["Expectation", "Stated before scoring", "Result"], rows)

    # --- calibration ------------------------------------------------------------------
    doc.add(
        "## Calibration",
        "",
        "On the final probabilities (Platt, clip, ordering fix, clip). Pass or fail uses the "
        f"point estimates against limits fixed in advance (slope {limits['slope'][0]} to "
        f"{limits['slope'][1]}, intercept {limits['intercept'][0]:+} to "
        f"{limits['intercept'][1]:+}); an interval crossing a limit marks the result "
        "borderline. The intercept reported everywhere is the registered pass/fail one: "
        "calibration in the large, from the fit with the slope fixed at 1. The intercept of the "
        "two-parameter fit was not logged. Only the survival model's calibration is a goal "
        "cell; the other models are shown for comparison. ECE is descriptive.",
        "",
    )
    for cid in cells7 + cell30:
        cellv = primary[cid]
        h, m = int(cid.split()[0]), cid.split(", ")[1]
        doc.add(f"### {h} days, {month(m)}", "")
        rows = []
        for name, mv in cellv["models"].items():
            e = mv["ece"]
            rows.append(
                [
                    MODEL_NAMES[name],
                    interval(mv["slope"], 3),
                    interval(mv["intercept"], 3, signed=True),
                    interval(e),
                    f"{e['share_of_rate']:.0%}" if e["status"] == "ok" else "",
                    "yes" if e["interpretable"] else "no",
                ]
            )
        doc.table(
            [
                "Model",
                "Slope",
                "Intercept (slope fixed at 1)",
                "ECE",
                "ECE / event rate",
                "ECE interpretable",
            ],
            rows,
        )
        cv = cellv["calibration_verdict_survival"]
        doc.add(f"Survival model: {calibration_text(cv, limits)}.", "")
        fix = cellv["order_fix"]
        doc.add(
            f"Ordering fix for the {month(m)} month (one fix over the 7, 30 and 59-day "
            "probabilities together, so it is the same in every cell of the month; "
            "descriptive): "
            + "; ".join(
                f"{n}: {f['rows_changed']:,} rows changed ({f['share_changed']:.2%}), "
                f"largest change {f['max_change']:.3f}, {f['clipped']:,} probabilities clipped"
                for n, f in fix.items()
            )
            + ".",
            "",
        )
        if h == 7 and month(m) == "2017-01":
            sm = cellv["models"]["survival_dt01"]
            doc.add(
                f"{PREREGISTERED_INTERCEPT} Result: the survival "
                f"model's 7-day intercept in 2017-01 is "
                f"{interval(sm['intercept'], 3, signed=True)}; the observed 7-day lapse rate "
                f"in 2017-01 is {cellv['lapses'] / cellv['rows']:.2%}. This failure "
                f"({fmt(sm['intercept']['point'], 3, signed=True)} against the "
                f"{limits['intercept'][1]:+} limit) is the scenario written down before the run: "
                "the features do not explain the January rise, so a model calibrated on October "
                "predicts too little risk in January; the intercept is smaller than the +0.30 "
                "stated for features that explain none of the rise. The cell has no gap day in "
                "its renewal window (see the limits). Nothing was recalibrated.",
                "",
                doc.note,
                "",
            )

    # --- survival-only metrics ----------------------------------------------------------
    doc.add("## Survival metrics per month", "", CONCORDANCE_NOTE, "")
    for m, sv in results["survival_metrics"].items():
        rows = [
            [
                name,
                value(mv["concordance"]),
                value(mv["ibs_primary_grid"]),
                value(mv["ibs_every_day"]),
                fmt(mv.get("skill_vs_km_overall", float("nan")), 3),
                fmt(mv.get("skill_vs_km_by_band", float("nan")), 3),
            ]
            for name, mv in sv.items()
            if isinstance(mv, dict)
        ]
        doc.add(f"### {month(m)} (stop day {sv['stop_day']})", "")
        doc.table(
            [
                "Model",
                "Concordance",
                "IBS (interval ends)",
                "IBS (every day)",
                "Skill vs Kaplan-Meier",
                "Skill vs Kaplan-Meier by expiry band",
            ],
            rows,
            intervals=False,
        )
        doc.add(CONCORDANCE_NOTE, "")
    doc.add(
        "dt01_plus_2016_11 is the registered sensitivity fit that adds the partly observed "
        "check month 2016-11; dt02 is descriptive (see below).",
        "",
    )

    # --- sensitivities ----------------------------------------------------------------
    doc.add(
        "## Sensitivities",
        "",
        "### Each 7-day month on its own, and both together",
        "",
        "The two 7-day months are the finding (each is a table above); with 2017-01 left out "
        "the 7-day result is the 2017-02 table, and with 2017-02 left out it is the 2017-01 "
        "table. The pooled interval below is a summary, never the headline.",
        "",
    )
    pooled = results["pooled_7_day_summary"]["7 days, both months"]["comparisons"]
    doc.table(
        ["Pooled over both 7-day months", "d [95% interval]"],
        [[k, interval(v, signed=True)] for k, v in pooled.items()],
    )
    doc.add("### Without 100%-discount-only customers", "")
    rows = []
    for cid, cellv in results["without_discount100_only"].items():
        for comp, v in cellv["comparisons"].items():
            rows.append(
                [
                    cell_label(cid),
                    f"{cellv['rows']:,}",
                    comp,
                    interval(v, signed=True),
                    v["verdicts"]["0.01"],
                ]
            )
        rows.append(
            [
                cell_label(cid),
                f"{cellv['rows']:,}",
                "survival calibration slope; intercept (slope fixed at 1)",
                f"{interval(cellv['survival_slope'], 3)}; "
                f"{interval(cellv['survival_intercept'], 3, signed=True)}",
                "",
            ]
        )
    doc.table(["Cell", "Rows", "Quantity", "Value [95% interval]", "Verdict at 0.01"], rows)
    doc.add(
        "### By segment",
        "",
        "Trial-only and promo-only customers are outside the training population and are never "
        "pooled into a headline number. AUC without intervals (descriptive).",
        "",
    )
    rows = []
    for cid, cellv in results["by_segment"].items():
        for seg, sv in cellv["segments"].items():
            rows.append(
                [
                    cell_label(cid),
                    seg,
                    "yes" if sv["in_training_population"] else "no",
                    f"{sv['rows']:,}",
                    f"{sv['lapses']:,}",
                    f"{sv['rate']:.2%}",
                    value(sv["auc_survival_dt01"]),
                    value(sv["auc_classifier"]),
                ]
            )
    doc.table(
        [
            "Cell",
            "Segment",
            "In training population",
            "Rows",
            "Lapses",
            "Rate",
            "AUC survival dt01",
            "AUC classifier",
        ],
        rows,
        intervals=False,
    )
    floor = raw["never_seen"]
    doc.add(
        "### Customers never seen in the fit or calibration months",
        "",
        "Comparisons between models only; a number is shown only with at least "
        f"{floor['min_events']} events and {floor['min_non_events']} non-events (a floor for "
        "showing a number, not a claim that the subset is large enough).",
        "",
    )
    rows = []
    for sec in ("primary_never_seen", "demonstration_59_never_seen"):
        for cid, cellv in results[sec].items():
            if cellv["status"] != "ok":
                rows.append(
                    [
                        cell_label(cid),
                        f"{cellv['rows']:,}",
                        f"{cellv['lapses']:,}",
                        "",
                        cellv["reason"],
                        "",
                    ]
                )
                continue
            for comp, v in cellv["comparisons"].items():
                rows.append(
                    [
                        cell_label(cid),
                        f"{cellv['rows']:,}",
                        f"{cellv['lapses']:,}",
                        comp,
                        interval(v, signed=True),
                        v["verdicts"]["0.01"],
                    ]
                )
    doc.table(["Cell", "Rows", "Lapses", "Comparison", "d [95% interval]", "Verdict at 0.01"], rows)
    doc.add(
        "### Feature-group ablations and the survival model fitted with 2016-11",
        "",
        "plus_counts_365 adds the 365-day counts (feature set fs3); minus_segment_state drops "
        "segment and paid state; plus_profile adds the profile group (never selectable: its "
        "source is a snapshot taken after the data ends).",
        "",
    )
    rows = []
    for cid, cellv in results["sensitivities"].items():
        for comp, v in cellv["comparisons"].items():
            rows.append([cell_label(cid), comp, interval(v, signed=True), v["verdicts"]["0.01"]])
    doc.table(["Cell", "Comparison", "d [95% interval]", "Verdict at 0.01"], rows)
    doc.add("### All customers with coverage (P1, secondary)", "", PR_AUC_NOTE, "")
    rows = []
    for cid, cellv in results["p1_secondary"].items():
        for name, mv in cellv["models"].items():
            rows.append(
                [
                    cell_label(cid),
                    f"{cellv['rows']:,}",
                    MODEL_NAMES[name],
                    interval(mv["auc_raw"]),
                    value(mv["pr_auc"]),
                    interval(mv["slope"], 3),
                    interval(mv["intercept"], 3, signed=True),
                ]
            )
        for comp, v in cellv["comparisons"].items():
            rows.append(
                [cell_label(cid), "", comp, interval(v, signed=True), "", v["verdicts"]["0.01"], ""]
            )
    doc.table(
        [
            "Cell",
            "Rows",
            "Model or comparison",
            "AUC or d",
            "PR AUC",
            "Slope or verdict",
            "Intercept (slope fixed at 1)",
        ],
        rows,
    )

    # --- 59-day demonstration ------------------------------------------------------------
    doc.add(
        "## 59-day demonstration",
        "",
        "One test month (2017-01), lapse by day 59 in every family (day 59 is the last day "
        "observable in 2017-01). Reported here, outside the headline table; it is not a goal "
        "cell.",
        "",
        PR_AUC_NOTE,
        "",
    )
    for cellv in results["demonstration_59"].values():
        rows = [
            [
                MODEL_NAMES[name],
                interval(mv["auc_raw"]),
                value(mv["pr_auc"]),
                interval(mv["slope"], 3),
                interval(mv["intercept"], 3, signed=True),
                interval(mv["ece"]),
            ]
            for name, mv in cellv["models"].items()
        ]
        doc.table(["Model", "AUC", "PR AUC", "Slope", "Intercept (slope fixed at 1)", "ECE"], rows)
        rows = [
            [comp, interval(v, signed=True), v["verdicts"]["0.01"]]
            for comp, v in cellv["comparisons"].items()
        ]
        doc.table(["Paired AUC difference", "d [95% interval]", "Verdict at 0.01"], rows)

    # --- descriptive additions -------------------------------------------------------------
    b = DESCRIPTIVE_BREAKDOWNS["last_row"]
    dt02 = DESCRIPTIVE_MODELS["dt02"]
    doc.add(
        "## Added after the fold results were seen (descriptive; decide nothing)",
        "",
        "### dt02",
        "",
        "dt02 is the discrete-time candidate with the expiry offset as an input. It was added "
        f"{dt02['added']}: on the folds it tied the classifier and beat the registered pick "
        "dt01, but it was lower on one fold, so the registered selection rule kept dt01. This "
        "input makes the model less pure survival modelling. Its rows appear in the tables "
        f"above, labelled; it decides {dt02['decides']}.",
        "",
        "### Last row: paid against zero-paid",
        "",
        f"Zero-paid groups the last-row kinds {', '.join(b['groups']['zero-paid'])}; it was "
        "defined before any 2017 outcome was read and decides nothing.",
        "",
    )
    rows = []
    for cid, cellv in results["descriptive_last_row"].items():
        for g, gv in cellv["groups"].items():
            rows.append(
                [
                    cell_label(cid),
                    g,
                    f"{gv['rows']:,}",
                    f"{gv['share_of_rows']:.2%}",
                    f"{gv['lapses']:,}",
                    f"{gv['lapse_rate']:.2%}",
                    interval(gv["auc_survival_dt01"]),
                    interval(gv["auc_classifier"]),
                    interval(gv["slope_survival_dt01"], 3),
                    interval(gv["intercept_survival_dt01"], 3, signed=True),
                    interval(gv["slope_classifier"], 3),
                    interval(gv["intercept_classifier"], 3, signed=True),
                ]
            )
    doc.table(
        [
            "Cell",
            "Group",
            "Rows",
            "Share",
            "Lapses",
            "Rate",
            "AUC survival",
            "AUC classifier",
            "Slope survival",
            "Intercept survival (slope fixed at 1)",
            "Slope classifier",
            "Intercept classifier (slope fixed at 1)",
        ],
        rows,
    )

    # --- shape checks and unseen levels ---------------------------------------------------
    doc.add("## Shape checks (descriptive)", "")
    rows = [
        [
            month(m),
            f"{sv['predicted_lapses_days_8_30']:,.0f}",
            f"{sv['observed_lapses_days_8_30']:,}",
            f"{sv['classifier_crossing_share']:.2%}",
        ]
        for m, sv in results["shape_checks"].items()
    ]
    doc.table(
        [
            "Month",
            "Lapses in days 8-30 predicted by S(7) - S(30)",
            "Observed",
            "Customers whose three classifiers cross",
        ],
        rows,
        intervals=False,
    )
    doc.add(
        "The survival curve between 7 and 30 days is uncalibrated; the crossing share describes "
        "three separate classifiers, not a property of one model.",
        "",
    )
    outside = []
    for cid, cellv in results["by_segment"].items():
        if not cid.startswith("7 days"):
            continue
        segs = {
            k: v["rows"] for k, v in cellv["segments"].items() if not v["in_training_population"]
        }
        outside.append(
            f"{month(cid.split(', ')[1])}: "
            + " + ".join(f"{n:,} {k}" for k, n in sorted(segs.items()))
            + f" = {sum(segs.values()):,}"
        )
    unseen = []
    for m, models in results["unseen_levels"].items():
        for name in ("classifier_primary_h7", "dt01"):
            counts = {k: v for k, v in models[name].items() if v}
            unseen.append(
                f"{month(m)}, {name}: "
                + (", ".join(f"{k} {v:,}" for k, v in counts.items()) if counts else "none")
            )
    doc.add(
        "## Unseen category levels",
        "",
        "Zero was expected. The counts cover every scored P2 row of the month, before the "
        "main-segment filter: they are the trial-only and promo-only rows ("
        + "; ".join(outside)
        + "), whose segment level never occurs in the training rows "
        "(the discrete-time model counts each such row once per interval end). No trial-only or "
        "promo-only row is inside any primary, calibration or 59-day cell: those cells hold "
        "only the paid and 100%-discount-only segments, the registered population. The "
        "by-segment table reports them separately. Counts: " + "; ".join(unseen) + ".",
        "",
    )

    # --- methods and limits ---------------------------------------------------------------
    doc.add(
        "## Methods and limits",
        "",
        f"- The numbers come from one run on macOS arm64 (Python {tags['version.python']}, "
        f"numpy {tags['version.numpy']}, LightGBM {tags['version.lightgbm']}, scikit-learn "
        f"{tags['version.scikit-learn']}).",
        "- The 30-day test rests on one test month (2017-01); the 7-day test has two "
        "(2017-01 and 2017-02); the 59-day demonstration has one.",
        f"- {meta['interval_note']} The bootstrap draws customers with replacement; a drawn "
        "customer brings all their rows.",
        "- The calibration limits (slope and intercept) and the 0.01 minimum difference were "
        "fixed in advance; no probability is recalibrated on test data.",
        *[f"- {n}" for n in REQUIRED_NOTES],
        "- The calibration half of 2016-10 and the selection half share the month's "
        "conditions; the split separates customers, not months (a mild dependence).",
        "- Classifier hyperparameters at 30 and 59 days were chosen on one fold (f30), a weaker "
        "guard than the two 7-day folds.",
        "- No calibration plot was logged: the run holds slope, intercept and ECE, not the "
        "per-bin values a reliability plot needs, so none is drawn here.",
        "- Each calibration limit failure is read from the point estimates, as registered; the "
        "intervals only mark it borderline.",
        "- Exploratory, post hoc checks (anything not registered, such as the 2016 rehearsal "
        "months or score means) are kept out of this report and labelled exploratory wherever "
        "they appear; none changes a registered value or verdict.",
        *[f"- {n}" for n in EXPLORATORY_LIMITS],
        "",
        "### History of commits",
        "",
        *[f"- {line}" for line in meta["history"]],
        "",
    )
    return doc.text()
