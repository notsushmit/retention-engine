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
from typing import Any

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
