"""Every fixed string the service returns, read once from configs/serving.yaml.

Nothing here is written in code: each value comes from a registered entry (the
contract, its amendments A2 and A4, step 4's decisions D1 and D2, the registry).
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
    )


LABELS = load_labels()
