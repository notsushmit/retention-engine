"""Every number's tooltip, built from registered text in configs/ (C2: no new numeric claim).

Each entry names the API field it explains and quotes the registered calculation; the
connecting words add no number. tests/test_dashboard_tooltips.py checks that every
number in a tooltip appears in a config or a run report and that every tooltip passes
the wording rules.
"""

from __future__ import annotations

from functools import cache

from dashboard.settings import Settings, load


def flat(text: str) -> str:
    return " ".join(str(text).split())


@cache
def catalog(s: Settings | None = None) -> dict[str, str]:
    s = s or load()
    sv, ac = s.serving, s.actions
    e = sv["dashboard_endpoints"]
    b, r, c = e["summary"]["buckets"], e["summary"]["reasons"], e["summary"]["counts"]
    a2 = next(a for a in sv["contract"]["amendments"] if a["id"] == "A2")
    a4 = sv["contract"]["action_reading"]
    risk = sv["contract"]["scored"]["risk"]
    calib = sv["ordering"]["descriptive"]["calibration_before_after_fix"]
    final = f"Final probability, {flat(risk['value'])}."
    prefix = e["plans"]["labels"]["effect_prefix"]
    inactive = flat(sv["step5"]["known_customers"]["not_active"])
    order = flat(sv["api"]["at_risk"]["order"])
    before_fix = flat(sv["contract"]["scored"]["reasons"]["with"])
    horizons = ", ".join(map(str, calib["horizons"]))
    knowable = sv["api"]["alerts"]["trend"]["knowable_rule"]
    return {
        "days_1_7": f"/summary buckets.days_1_7: {b['days_1_7']}; {b['label']}.",
        "days_8_30": f"/summary buckets.days_8_30: {b['days_8_30']}; {b['label']}.",
        "days_31_59": (
            f"/summary buckets.days_31_59: {b['days_31_59']}; {b['label']}; "
            "every 59-day probability is a demonstration."
        ),
        "bucket_share": s.raw["bucket_shares"]["tooltip"],
        "outside": f"/summary counts: {flat(c['outside_share_of_active'])}. {c['outside_label']}",
        "scored": "/summary counts.scored: customers with a score at the store's date.",
        "not_active": f"/summary counts.not_active: {inactive}.",
        "reason_share": (
            f"/summary reasons, segment all: {flat(r['counted_as'])}; share = customers / "
            f"{flat(r['population'])}. {flat(s.c1['suppression']['applies_to'])}."
        ),
        "replay": (
            "/health replay_progress: events consumed of the registered "
            f"{sv['api']['health']['replay_expected_events']:,}."
        ),
        "probability": f"/at-risk rows.probability: {final} Order: {order}.",
        "matching": "/at-risk matching: scored customers that pass the filters.",
        "p7": f"/score risk.days_7: {final}",
        "p30": f"/score risk.days_30: {final}",
        "p59": f"/score risk.days_59: {final} Grade: {risk['grade_59']}.",
        "reason_probability": (f"/score reasons.probability_before_fix: {before_fix}."),
        "paid_days": f"/score paid_days: {flat(a2['name'])}; {flat(a2['definition'])}",
        "action_values": (
            f"{prefix}, /score action: {flat(a4['definition'])}; net value = "
            f"{ac['budget']['net_value']}; pays: {flat(a4['pays'])}. Unit: {s.money_unit}."
        ),
        "plan_values": (
            f"{prefix}, /plans rows: {flat(e['plans']['source'])}; "
            f"net value = {ac['budget']['net_value']}. "
            f"Unit: {s.money_unit}."
        ),
        "plan_range": f"/plans rows.range: {ac['strategies']['random']['range_label']}.",
        "calibration": (
            f"/model calibration: {', '.join(calib['report'])} at {horizons} "
            f"days, {flat(calib['slice'])}; {e['model']['calibration']['label']}."
        ),
        "model_files": f"/model files: {flat(sv['registry']['load'])}",
        "version": f"/model version: the version under the alias {sv['registry']['alias']}.",
        "alert_rates": (f"/alerts trend: stored Phase 5 values; shown when {knowable}."),
    }


def tip(key: str) -> str:
    return catalog()[key]
