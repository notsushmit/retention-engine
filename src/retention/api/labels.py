"""Every fixed string the service returns, read once from configs/serving.yaml.

Nothing here is written in code: each value comes from a registered entry (the
contract, its amendments A2, A4, A5 and A6, step 4's decisions D1 and D2, the
registry).
Two December 2016 strings are registered as descriptions, not as exact text; they
are taken out of those descriptions by a fixed pattern, and loading stops if the
registered wording no longer fits it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from retention.api.config import load_serving

WITHHELD = re.compile(r'^withheld, code "(?P<code>[^"]+)"$')
WARNING = re.compile(r"^returned with a (?P<warning>data warning about .+)$")
KNOWABLE = re.compile(r"^week start \+ (?P<days>\d+) days on or before the store's date$")


@dataclass(frozen=True)
class Labels:
    statuses: tuple[str, str, str]  # scored, outside_population, not_active
    outside_codes: tuple[str, ...]  # the fixed codes
    unseen_prefix: str  # "unseen_level"
    grade_59: str
    reasons_fixed_line: str
    reasons_withheld_code: str
    december_warning: str
    paid_days_field: str
    paid_days_name: str
    paid_days_labels: tuple[str, ...]
    archetype_status: str
    effect_basis: str
    scenario_keys: tuple[str, ...]
    instability_note: str
    no_estimate_line: str
    probability_basis: str
    probability_note: str
    action_field: str
    reads_as: str
    no_price_flag: str
    no_price_reads_as: str
    scenario_text: dict[str, str]
    never_in_unpaid: tuple[str, ...]
    errors: dict[int, str]
    never_in_any_body: tuple[str, ...]
    bundle: dict[str, str]
    unseen_generic_code: str  # A5: an unseen level in a feature naming a never-used attribute
    msno_exempt_field: str  # A6: the one field whose own-ID value the response scan skips
    plan_label: str  # on every plan field of /at-risk
    at_risk_header_note: str
    no_plan_status: str
    trend_labels: tuple[str, ...]  # on every trend alert
    event_label: str
    hidden_dimensions: tuple[str, ...]  # removed from served alerts (api.alerts, A7)
    knowable_days: int  # a trend alert is knowable when week start + this many days <= T


def load_labels(cfg: dict[str, Any] | None = None) -> Labels:
    c = cfg if cfg is not None else load_serving()
    k = c["contract"]
    scored = k["scored"]
    req = scored["action"]["required"]
    a2 = next(a for a in k["amendments"] if a["id"] == "A2")
    a4 = k["action_reading"]
    d = c["step4"]["decisions"]
    withheld = WITHHELD.match(k["december_2016"]["reasons"])
    warning = WARNING.match(k["december_2016"]["scores"])
    if withheld is None or warning is None:
        raise ValueError("the registered December 2016 wording no longer fits its pattern")
    codes = list(k["outside_population"]["codes"])
    unseen = [x for x in codes if x.endswith(":<feature>")]
    if len(unseen) != 1:
        raise ValueError("the contract must register exactly one unseen-level code")
    if d["no_usable_price"]["reads_as"] != a4["reads_as_when_not_paying"]:
        raise ValueError("D2's reading and A4's reading differ")
    a5, a6 = c["unseen_level_codes"], c["msno_scan_exemption"]
    risk, alerts, step5 = c["api"]["at_risk"], c["api"]["alerts"], c["step5"]
    if (
        risk["plan_label"] != step5["plan_fields"]["every_plan_field"]
        or risk["header_note"] != step5["plan_fields"]["header_note"]
        or alerts["event"]["label"] != step5["alerts"]["event_label"]
    ):
        raise ValueError("the api and step5 registrations of the /at-risk or /alerts labels differ")
    knowable = KNOWABLE.match(alerts["trend"]["knowable_rule"])
    if knowable is None:
        raise ValueError("the registered knowable rule no longer fits its pattern")
    generic = a5["generic_code"]
    if not generic.startswith(unseen[0].removesuffix("<feature>")) or generic.endswith(":"):
        raise ValueError("A5's generic code must be an unseen-level code")
    return Labels(
        statuses=(
            "scored",
            k["outside_population"]["status"],
            k["not_active"]["status"],
        ),
        outside_codes=tuple(x for x in codes if x not in unseen),
        unseen_prefix=unseen[0].removesuffix(":<feature>"),
        grade_59=scored["risk"]["grade_59"],
        reasons_fixed_line=scored["reasons"]["fixed_line"],
        reasons_withheld_code=withheld["code"],
        december_warning=warning["warning"],
        paid_days_field=a2["field"],
        paid_days_name=a2["name"],
        paid_days_labels=tuple(a2["labels"]),
        archetype_status=scored["archetype"]["status"],
        effect_basis=req["effect_basis"],
        scenario_keys=tuple(req["scenario"]),
        instability_note=req["instability_note"],
        no_estimate_line=req["no_estimate_line"],
        probability_basis=d["action_probability"]["label"],
        probability_note=" ".join(d["action_probability"]["note"].split()),
        action_field=a4["field"],
        reads_as=a4["reads_as_when_not_paying"],
        no_price_flag=d["no_usable_price"]["flag"],
        no_price_reads_as=d["no_usable_price"]["reads_as"],
        scenario_text=dict(a4["scenario_text"]),
        never_in_unpaid=tuple(a4["never_in_a_block_that_does_not_pay"]),
        errors={int(code): text for code, text in k["errors"].items()},
        never_in_any_body=tuple(k["never_in_any_body"]),
        bundle=dict(c["registry"]["bundle"]),
        unseen_generic_code=generic,
        msno_exempt_field=a6["field"],
        plan_label=risk["plan_label"],
        at_risk_header_note=risk["header_note"],
        no_plan_status=risk["no_plan_status"],
        trend_labels=tuple(alerts["trend"]["labels"]),
        event_label=alerts["event"]["label"],
        hidden_dimensions=tuple(alerts["trend"]["hidden_dimensions"]),
        knowable_days=int(knowable["days"]),
    )


LABELS = load_labels()


# --- GET /summary, /plans and /model (configs/serving.yaml, dashboard_endpoints) --------------


@dataclass(frozen=True)
class DashboardLabels:
    outside_label: str  # the fixed disclaimer, verbatim
    bucket_label: str
    top_n: int
    reasons_label: str  # with "<horizon>"
    min_cell_size: int
    no_group: str
    plan_label: str
    no_plan_status: str
    caption: str
    effect_prefix: str
    effect_basis: str
    instability_note: str
    no_estimate_line: str
    dates: str
    money_unit: str
    range_label: str
    calibration_label: str
    no_59: str
    drift: str
    retraining: str


NO_GROUP = re.compile(r'"(?P<text>no group listed)" when none is listed$')


def load_dashboard_labels(cfg: dict[str, Any] | None = None, labels: Labels = LABELS):
    c = cfg if cfg is not None else load_serving()
    e = c["dashboard_endpoints"]
    s, p, m = e["summary"], e["plans"], e["model"]
    actions = yaml.safe_load(Path("configs/actions.yaml").read_text())
    strategies = actions["strategies"]
    pl = p["labels"]
    disclaimers = c["wording"]["fixed_disclaimers"]
    reasons = s["reasons"]
    no_group = NO_GROUP.search(reasons["counted_as"])
    if no_group is None:
        raise ValueError("the registered no-group wording no longer fits its pattern")
    if s["counts"]["outside_label"] not in disclaimers or pl["dates"] not in disclaimers:
        raise ValueError("a /summary or /plans disclaimer differs from the fixed disclaimers")
    if pl["caption"] != "configs/actions.yaml strategies.table_caption":
        raise ValueError("the /plans caption must name the registered table caption")
    if (
        pl["instability_note"] != labels.instability_note
        or pl["no_estimate_line"] != labels.no_estimate_line
        or pl["effect_basis"] != labels.effect_basis
        or reasons["explains"] != labels.reasons_fixed_line
        or m["grade_59"] != labels.grade_59
        or s["buckets"]["grade_31_59"] != labels.grade_59
    ):
        raise ValueError("a dashboard endpoint label differs from the contract's")
    prefix = actions["wording"]["sentence_prefix"]
    if prefix.replace("<scenario>", "relative").replace("<level>", "base") != pl["effect_prefix"]:
        raise ValueError("the /plans effect prefix differs from the registered sentence prefix")
    top_n = int(reasons["top_n"])
    if f"{top_n:,}" not in reasons["label"]:
        raise ValueError("the reasons label must name the registered top_n")
    return DashboardLabels(
        outside_label=s["counts"]["outside_label"],
        bucket_label=s["buckets"]["label"],
        top_n=top_n,
        reasons_label=reasons["label"],
        min_cell_size=int(reasons["min_cell_size"]),
        no_group=no_group["text"],
        plan_label=pl["plan"],
        no_plan_status=c["api"]["at_risk"]["no_plan_status"],
        caption=strategies["table_caption"],
        effect_prefix=pl["effect_prefix"],
        effect_basis=pl["effect_basis"],
        instability_note=pl["instability_note"],
        no_estimate_line=pl["no_estimate_line"],
        dates=pl["dates"],
        money_unit=pl["money_unit"],
        range_label=strategies["random"]["range_label"],
        calibration_label=m["calibration"]["label"],
        no_59=m["calibration"]["no_59"],
        drift=m["drift"],
        retraining=m["retraining"],
    )


DASH = load_dashboard_labels()


@dataclass(frozen=True)
class OpsLabels:
    """The registered fixed strings of GET /drift and the /model additions (ops_endpoints)."""

    drift_label: str
    drift_absent: str  # the sentence an absent drift file answers with
    drift_see: str  # /model.drift when a valid drift file is present
    synthetic_sentence: str
    retrain_label: str
    decisions: tuple[str, ...]


QUOTED = re.compile(r'"([^"]+)"')


def load_ops_labels(cfg: dict[str, Any] | None = None) -> OpsLabels:
    c = cfg if cfg is not None else load_serving()
    e = c["ops_endpoints"]
    ops = yaml.safe_load(Path("configs/ops.yaml").read_text())
    d, m = e["drift"], e["model_additions"]
    if d["label"] != ops["drift"]["label"] or m["label"] != ops["retrain"]["label"]:
        raise ValueError("an ops_endpoints label differs from configs/ops.yaml")
    absent = QUOTED.findall(d["absent"])
    see = QUOTED.findall(" ".join(str(m["drift_field"]).split()))
    synthetic = QUOTED.findall(d["synthetic"])
    if len(absent) < 2 or not see or not synthetic:
        raise ValueError("the registered ops_endpoints wording no longer fits its patterns")
    return OpsLabels(
        drift_label=d["label"],
        drift_absent=absent[1],
        drift_see=see[0],
        synthetic_sentence=synthetic[0],
        retrain_label=m["label"],
        decisions=tuple(m["decision_values"]),
    )


OPS = load_ops_labels()
if OPS.drift_absent != DASH.drift:
    raise ValueError("the absent-drift sentence must be the registered /model drift line")
