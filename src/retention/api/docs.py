"""docs/scoring_api.md, generated from configs/serving.yaml and the response schemas.

`python -m retention.api write-docs` writes it; tests/test_scoring_docs.py checks
that the committed file equals a fresh build and passes the wording rules. Every
registered value is read from the config or the schemas; the few notes below come
from the decision log (D-174, D-199, D-203, D-206) and hold no number that is not
also in the config.

wording_violations applies configs/serving.yaml `wording` on top of Phase 7's
rules (configs/actions.yaml wording_rules, unchanged): "real-time", "production"
and "identical" are banned; "live" only inside the exact first disclaimer; "60
days" not at all; a never-used attribute only inside the exact registered
sentences listed in EXACT_EXCEPTIONS (docs plan, D-207), each at most once.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from retention.api import schemas as sch
from retention.api import service as svc
from retention.api.config import load_serving
from retention.api.labels import LABELS
from retention.optimize.config import load_actions_config
from retention.optimize.report import wording_violations as phase7_violations

DOC = Path("docs/scoring_api.md")
CFG = load_serving()
WORDING = CFG["wording"]
MUST_STATE = tuple(CFG["step5"]["docs"]["must_state"])
DISCLAIMERS = tuple(WORDING["fixed_disclaimers"])
LIVE_EXCEPTION = WORDING["live_exception"]
TENURE_SENTENCE = MUST_STATE[4]  # the registered sentence saying tenure is never a reason
SELF_CHECK_SENTENCE = MUST_STATE[5]  # the registered self-check sentence ("missing-tenure")
EXACT_EXCEPTIONS = (TENURE_SENTENCE, SELF_CHECK_SENTENCE)  # each allowed once, verbatim

# --- notes from the decision log (no number that is not in the config) --------------------------

A3_NOTE = (
    "The group that the registered statement above says is never shown as a reason is listed "
    "as a reason in {listed:,} rows of Phase 4's reasons.parquet (2016-10-01 and 2016-11-01); "
    "that file, Phase 4's docs and its phrases file are left as they are (amendment A3)."
)
DERIVED_NOTE = (
    'The three derived strings are the December 2016 withheld code "{code}", the warning '
    '"{warning}" and the status "{status}"; they are taken by fixed patterns from the '
    "contract's descriptions in src/retention/api/labels.py, not registered verbatim."
)
WORKAROUND_NOTE = (
    "POST /score scores each customer together with a placeholder copy of the row and returns "
    "only the customer's own answer. This is a workaround that relies on NumPy's summation "
    "behaviour, chosen after the exact-equality test failed, not a fix of its source: Phase 7's "
    "expected-days sum adds a one-row array in another order than an array of several rows "
    "(D-199)."
)
CI_SKIP_NOTE = (
    "The test that recomputes the knowable count from the pinned alerts file skips in CI, "
    "because the pinned file is not in git; it runs where the file exists and checks "
    "amendment A8's count."
)
UNREGISTERED = (
    "/health carries a reason field beside the registered fields.",
    "msno must be a non-empty string; as_of must be written YYYY-MM-DD.",
    "A 422 body never returns the validation details, so no field name a caller sent comes back.",
    'Unknown paths and methods keep the framework\'s "Not Found" and "Method Not Allowed" bodies.',
    "Every /health, /score, /at-risk and /alerts call hashes the downloaded model files and "
    "both pinned files again.",
    "Startup refuses loaded model files that differ from the bundle the response schemas register.",
    "The /at-risk auto_renew filter takes on or off; in_plan takes true or false.",
    "An in_plan filter on a date without a plan is a 422.",
    "Each /at-risk row shows the customer's auto_renew.",
    "The /at-risk header counts list only the stored outside-population codes.",
    "/alerts takes no parameters.",
    "Cancel events are read as objects with msno and date; the replay must write that form.",
)


# --- amendments A9 and A10, and the registered notes of steps 7 and 8 ---------------------------


def amendments_a9_a10() -> list[str]:
    """A9 and A10 as registered, each with its registered wording (not met as written)."""
    a9, a10 = CFG["stream_parity_columns"], CFG["http_parity_a10"]
    tolerance = a10["judged"]["float_relative_tolerance"]
    return [
        "## Amendments A9 and A10",
        "",
        f"- {a9['id']} ({a9['date']}, amends {a9['amends']}): feature parity is judged on "
        f"{a9['columns_judged']} columns; {len(a9['excluded'])} metadata columns are excluded "
        "because they cannot be rebuilt from rows dated before the store's date. The registered "
        f"wording: {a9['registered_wording']}.",
        f"- A10 (amends {a10['amends']}): {a10['label'].removeprefix('amendment A10, ')}. "
        "HTTP parity is judged with every "
        f"non-float field exact and every float within relative {tolerance:g}. The registered "
        f"wording: {a10['registered_wording']}. {a10['limit']}",
        "",
    ]


def stack_notes() -> list[str]:
    """The registered docs notes of steps 7 and 8 and the API memory limit."""
    limit = CFG["api_memory_limit"]
    notes = [
        *CFG["step7"]["docs_notes"],
        *CFG["step7_images"]["docs_notes"],
        *CFG["step8"]["docs_notes"],
        f"The API container's memory limit is {limit['api']}, the user's choice; Valkey has "
        "no limit.",
    ]
    return ["## Notes on the stack and the runs", "", *[f"- {n}" for n in notes], ""]


# --- schema field lists --------------------------------------------------------------------------


def nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = nested_model(arg)
        if found is not None:
            return found
    return None


def field_lines(model: type[BaseModel], depth: int = 0) -> list[str]:
    out = []
    for name, field in model.model_fields.items():
        out.append(f"{'  ' * depth}- `{name}`")
        inner = nested_model(field.annotation)
        if inner is not None and depth < 2:
            out += field_lines(inner, depth + 1)
    return out


# --- the document --------------------------------------------------------------------------------


def build() -> str:
    k, api = CFG["contract"], CFG["api"]
    a3 = next(a for a in CFG["step4"]["amendments"] if a["id"] == "A3")
    a5, a6 = CFG["unseen_level_codes"], CFG["msno_scan_exemption"]
    a7, a8 = CFG["alerts_hidden_segments"], CFG["alerts_knowable_count"]
    risk, alerts = api["at_risk"], api["alerts"]
    lines = [
        "# Scoring API",
        "",
        "Generated by `python -m retention.api write-docs` from configs/serving.yaml and the "
        "response schemas. Do not edit by hand; tests/test_scoring_docs.py checks that this file "
        "equals a fresh build.",
        "",
        *[f"> {d}" for d in DISCLAIMERS],
        "",
        "## Registered statements",
        "",
        *[f"- {s}" for s in MUST_STATE],
        "",
        "## Endpoints",
        "",
        "### GET /health",
        "",
        f"Fields: {', '.join(f'`{f}`' for f in api['health']['fields'])}, and `reason` "
        "(not registered; see below). It answers 503 with the reason when the service is not "
        "ready.",
        "",
        "### POST /score",
        "",
        "Request: `msno` (required) and `as_of` (optional). Features come from the service's own "
        "store; callers never post feature values.",
        "",
        f"Answers: `{LABELS.statuses[0]}`, `{LABELS.statuses[1]}` (codes "
        + ", ".join(f"`{c}`" for c in k["outside_population"]["codes"])
        + f", and `{LABELS.unseen_generic_code}` under amendment A5) or `{LABELS.statuses[2]}`.",
        "",
        "Errors:",
        "",
        *[f"- {code}: {text}" for code, text in k["errors"].items()],
        "",
        "### GET /at-risk",
        "",
        f"Parameters: `horizon` (one of {', '.join(map(str, risk['horizon']['allowed']))}; "
        f"default {risk['horizon']['default']}), `limit` (default {risk['limit']['default']}, "
        f"at most {risk['limit']['max']}), `offset`, and the filters "
        + ", ".join(f"`{f}`" for f in risk["filters"])
        + ". Anything else is a 422.",
        "",
        f"Order: {risk['order']}. Plan cell: strategy {risk['plan_cell']['strategy']}, scenario "
        f"{risk['plan_cell']['scenario']}, budget {risk['plan_cell']['budget']:,}.",
        "",
        f'Every plan field reads "{LABELS.plan_label}". '
        f'Header note: "{LABELS.at_risk_header_note}" '
        f'A date without a plan reads "{LABELS.no_plan_status}".',
        "",
        "### GET /alerts",
        "",
        f"Kinds: {', '.join(alerts['kinds'])}. A trend alert is shown when the registered "
        f"rule holds: {alerts['trend']['knowable_rule']}. Every trend alert carries the labels "
        + ", ".join(f'"{x}"' for x in LABELS.trend_labels)
        + ", and its stored data-quality flags. The dimensions registered in "
        "api.alerts.trend.hidden_dimensions are removed from top_segments, too_small_segments "
        "(amendment A7) and dimension_totals, and the number removed is shown. Each cancel "
        f'event reads "{LABELS.event_label}".',
        "",
        "## Response fields",
        "",
    ]
    for title, model in (
        ("POST /score, scored", sch.Scored),
        ("POST /score, outside_population", sch.Outside),
        ("POST /score, not_active", sch.NotActive),
        ("GET /at-risk", sch.AtRiskPage),
        ("GET /alerts", sch.Alerts),
        ("GET /health", svc.Health),
        ("Error bodies", sch.ErrorBody),
    ):
        lines += [f"### {title}", "", *field_lines(model), ""]
    lines += [
        "## Amendments A5 to A8",
        "",
        f"- {a5['id']} ({a5['date']}, amends {a5['amends']}): an unseen level in a feature "
        "whose name contains a never-used attribute as a whole word (the response scan's own "
        f"test) reads `{a5['generic_code']}`; for any other feature the code names the feature, "
        "`unseen_level:<feature>`.",
        f"- {a6['id']} ({a6['date']}, amends {', '.join(a6['amends'])}): {a6['exempt']}; "
        f"{a6['scanned_in_full']} are scanned in full. An identifier exemption, not a relaxation "
        "of the never-used rule. "
        f"{a6['counts']['known_at_2016_11_01']['with_a_whole_word_hit']} of "
        f"{a6['counts']['known_at_2016_11_01']['distinct_ids']:,} known IDs contain one "
        "never-used word as a whole word; "
        f"{a6['counts']['union_of_the_seven_tables']['with_a_whole_word_hit']} of "
        f"{a6['counts']['union_of_the_seven_tables']['distinct_ids']:,} distinct IDs across the "
        "seven tables do (distinct IDs, not table rows).",
        f"- {a7['id']} ({a7['date']}, amends {a7['amends']}): {a7['rule']}. Whole-word hits "
        f"before hiding, summed over the trend alerts: top_segments "
        f"{a7['counts']['top_segments']}, dimension_totals {a7['counts']['dimension_totals']}, "
        f"too_small_segments {a7['counts']['too_small_segments']}, every other field "
        f"{a7['counts']['every_other_field']}. The pinned file is {a7['pinned_file']}.",
        f"- {a8['id']} ({a8['date']}, amends {a8['amends']}): the registered value "
        f"{a8['registered_value']} was wrong; under the unchanged rule ({a8['rule']}) the pinned "
        f"file gives {a8['knowable_at_2016_11_01']} at 2016-11-01; "
        f"{a8['first_not_knowable']['alert_id']} (week {a8['first_not_knowable']['week']}) is "
        f"knowable from {a8['first_not_knowable']['knowable_from']}.",
        "",
        "## Notes",
        "",
        "- " + A3_NOTE.format(listed=a3["counts"]["total_listed"]),
        "- "
        + DERIVED_NOTE.format(
            code=LABELS.reasons_withheld_code,
            warning=LABELS.december_warning,
            status=LABELS.statuses[0],
        ),
        "- " + WORKAROUND_NOTE,
        "- " + CI_SKIP_NOTE,
        "",
        *amendments_a9_a10(),
        *stack_notes(),
        "## Choices not registered",
        "",
        *[f"- {c}" for c in UNREGISTERED],
        "",
    ]
    return "\n".join(lines)


def write(path: Path | None = None) -> Path:
    path = DOC if path is None else path
    path.write_text(build())
    return path


# --- the wording rules ---------------------------------------------------------------------------

BANNED = tuple(w for w in WORDING["banned_words"] if w != "live")


def wording_violations(text: str) -> list[str]:
    """Phase 7's rules plus configs/serving.yaml `wording`, never loosened."""
    out = phase7_violations(text, load_actions_config())
    if text.count(LIVE_EXCEPTION) > 1:
        out.append("the live disclaimer is used twice")
    rest = text.replace(LIVE_EXCEPTION, " ", 1)
    for w in (*BANNED, "live"):
        if re.search(rf"\b{re.escape(w)}\b", rest, re.I):
            out.append(f"banned word {w!r}")
    if re.search(r"\b60 days\b", text, re.I):
        out.append("'60 days'")
    scanned = text
    for sentence in EXACT_EXCEPTIONS:
        if scanned.count(sentence) > 1:
            out.append(f"an exact-string exception is used twice: {sentence[:40]!r}")
        scanned = scanned.replace(sentence, " ", 1)
    out += [f"never-used attribute {w!r}" for w in sch.never_used_hits(scanned)]
    for s in re.split(r"(?<=[.;])\s+", text):
        if "59-day probabilit" in s and "demonstration" not in s:
            out.append(f"a 59-day probability without 'demonstration': {s[:60]!r}")
    return out
