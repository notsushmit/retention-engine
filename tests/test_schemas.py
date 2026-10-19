"""Response schemas: every label required and fixed; A4's reading; the never-used scan."""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from retention.api import labels as lab
from retention.api import schemas as s
from retention.api import scoring as sc
from retention.api.config import load_serving
from retention.api.labels import LABELS
from retention.api.saved import verify_files
from retention.optimize.effects import build_options, check_customers
from retention.optimize.value import customer_table
from test_scoring_action import ACT, table

NOV, DEC = date(2016, 11, 1), date(2016, 12, 1)


def blocks_by_case() -> dict[str, dict]:
    acts = sc.actions(table(400, seed=3, no_price_every=7), ACT)
    out = {}
    for b in sc.action_blocks(acts, ACT.base):
        case = (
            "none_allowed"
            if b["highest_net_value_action"] == "none"
            else ("pays" if b["pays"] else "does_not_pay")
        )
        if "price" in b:
            case = "no_price"
        out.setdefault(case, b)
    return out


BLOCKS = blocks_by_case()


def scored(as_of: date = NOV, **change) -> dict:
    reasons = None
    if not (as_of.year == 2016 and as_of.month == 12):
        reasons = [
            {
                "horizon": h,
                "probability_before_fix": 0.02,
                "text": "Risk is close to typical.",
                "explains": LABELS.reasons_fixed_line,
            }
            for h in (7, 30)
        ]
    out = {
        "status": "scored",
        "msno": "abc",
        "features_as_of": as_of,
        "risk": {
            "days_7": {"probability": 0.01, "text": "1.0%"},
            "days_30": {"probability": 0.04, "text": "4.0%"},
            "days_59": {"probability": 0.07, "text": "7.0%", "grade": LABELS.grade_59},
            "moved_by_ordering_fix": False,
        },
        "reasons": reasons,
        "paid_days": sc.paid_days_block(58.3),
        "archetype": {"value": None, "status": LABELS.archetype_status},
        "action": BLOCKS["pays"],
        "model": {
            "registered_model": "retention-scoring",
            "version": 1,
            "files": LABELS.bundle,
            "config_sha256": "0" * 64,
        },
    }
    if reasons is None:
        out |= {
            "reasons_withheld": LABELS.reasons_withheld_code,
            "data_warning": LABELS.december_warning,
        }
    return out | change


# --- valid responses ----------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["pays", "does_not_pay", "none_allowed", "no_price"])
def test_every_a4_case_builds(case):
    r = s.Scored(**scored(action=BLOCKS[case]))
    assert r.action.pays == (case == "pays")
    assert (r.action.reads_as is None) == (case == "pays")
    assert (r.action.price is not None) == (case == "no_price")


def test_december_withholds_reasons_and_warns():
    r = s.Scored(**scored(DEC))
    assert r.reasons is None and r.reasons_withheld == "2016-12 not explained"
    with pytest.raises(ValidationError, match="withheld in December"):
        s.Scored(**scored(DEC, reasons=scored()["reasons"]))
    with pytest.raises(ValidationError, match="both reasons are shown"):
        s.Scored(**scored(NOV, reasons=None))


def test_other_answers_build():
    s.Outside(
        status="outside_population",
        msno="a",
        features_as_of=NOV,
        code="segment_trial_only",
        next_expiry_in_days=4,
    )
    s.Outside(
        status="outside_population",
        msno="a",
        features_as_of=NOV,
        code="unseen_level:last_payment_method_id",
        next_expiry_in_days=None,
    )
    s.NotActive(status="not_active", msno="a", features_as_of=NOV)
    for code in (404, 409, 422, 503):
        assert s.error_body(code).detail == LABELS.errors[code]


# --- a missing or changed label cannot be built ---------------------------------------------------

LABEL_FIELDS = [
    ("status", None),
    ("archetype", None),
    ("paid_days", None),
    ("action", None),
    ("model", None),
    ("risk", ("days_59", "grade")),
    ("paid_days", ("field",)),
    ("paid_days", ("name",)),
    ("paid_days", ("labels",)),
    ("archetype", ("status",)),
    ("action", ("effect_basis",)),
    ("action", ("scenario",)),
    ("action", ("instability_note",)),
    ("action", ("no_estimate_line",)),
    ("action", ("probability_basis",)),
    ("action", ("scenario", "discount_cost_rule")),
    ("action", ("scenario", "value_horizon")),
    ("action", ("scenario", "budget")),
    ("model", ("files",)),
]


def without(d: dict, top: str, path: tuple | None) -> dict:
    d = copy.deepcopy(d)
    if path is None:
        del d[top]
        return d
    node = d[top]
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    return d


def changed(d: dict, top: str, path: tuple) -> dict:
    d = copy.deepcopy(d)
    node = d[top]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = ("x",) if isinstance(node[path[-1]], tuple | list) else "something else"
    return d


@pytest.mark.parametrize(("top", "path"), LABEL_FIELDS)
def test_a_response_missing_a_label_cannot_be_built(top, path):
    with pytest.raises(ValidationError):
        s.Scored(**without(scored(), top, path))


@pytest.mark.parametrize(("top", "path"), [x for x in LABEL_FIELDS if x[1] is not None])
def test_a_response_with_another_label_value_cannot_be_built(top, path):
    with pytest.raises(ValidationError):
        s.Scored(**changed(scored(), top, path))


def test_reason_labels_are_required_and_fixed():
    r = scored()["reasons"][0]
    with pytest.raises(ValidationError):
        s.Reason(**{k: v for k, v in r.items() if k != "explains"})
    with pytest.raises(ValidationError):
        s.Reason(**(r | {"explains": "something else"}))


def test_extra_fields_are_refused():
    with pytest.raises(ValidationError):
        s.Scored(**scored(), tenure_days=12)


# --- reading rules --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"pays": True},  # on a block whose net value is not above 0
        {"reads_as": None},
        {"highest_net_value_action": "none", "cost": 1.0},
    ],
)
def test_a_block_breaking_a4_cannot_be_built(change):
    with pytest.raises(ValidationError):
        s.ActionBlock(**(BLOCKS["does_not_pay"] | change))


def test_a_price_flag_on_a_paying_block_cannot_be_built():
    with pytest.raises(ValidationError):
        s.ActionBlock(**(BLOCKS["pays"] | {"price": "no usable price"}))


def test_risk_must_not_fall_and_reasons_may_not_name_membership():
    bad = scored()["risk"] | {"days_30": {"probability": 0.001, "text": "0.1%"}}
    with pytest.raises(ValidationError, match="must not fall"):
        s.Risk(**bad)
    r = scored()["reasons"][0] | {"text": "Member for 6 months."}
    with pytest.raises(ValidationError, match="member"):
        s.Reason(**r)


def test_outside_codes_and_error_bodies_are_the_registered_ones():
    base = {
        "status": "outside_population",
        "msno": "a",
        "features_as_of": NOV,
        "next_expiry_in_days": 3,
    }
    for bad in ("unknown_code", "unseen_level:"):
        with pytest.raises(ValidationError):
            s.Outside(**base, code=bad)
    with pytest.raises(ValidationError):
        s.ErrorBody(status_code=404, detail="not found")
    with pytest.raises(ValidationError):
        s.ErrorBody(status_code=404, detail=LABELS.errors[404], reason="why")


# --- no "recommend" in a block that does not pay --------------------------------------------------


def test_no_block_that_does_not_pay_contains_recommend():
    acts = sc.actions(table(500, seed=9, no_price_every=5), ACT)
    unpaid = [b for b in sc.action_blocks(acts, ACT.base) if not b["pays"]]
    assert len(unpaid) > 0
    for b in unpaid:
        body = s.Scored(**scored(action=b)).model_dump_json().lower()
        assert "recommend" not in body
    sneaky = BLOCKS["does_not_pay"] | {
        "scenario": BLOCKS["does_not_pay"]["scenario"] | {"form": "Recommended"}
    }
    with pytest.raises(ValidationError, match="recommend"):
        s.ActionBlock(**sneaky)


# --- the never-used scan over every message the scoring code can raise ----------------------------


def all_messages() -> list[str]:
    msgs: list[str] = []

    def catch(fn, *a, **k):
        try:
            fn(*a, **k)
        except Exception as e:  # noqa: BLE001 (collecting every message on purpose)
            msgs.append(str(e))
            return
        raise AssertionError(f"{fn.__name__} did not raise")

    # scoring
    st = pd.DataFrame(
        {"msno": ["a"], "in_exp30": [True], "segment": ["odd"], "days_until_expiry": [1]}
    )
    catch(sc.population_status, st, ("paid", "discount100_only"))
    catch(sc.probabilities, {}, st, 1)
    blk = BLOCKS["pays"]
    for k in sc.REQUIRED_ACTION_FIELDS:
        catch(sc.check_action_block, {x: v for x, v in blk.items() if x != k})
    catch(sc.check_action_block, blk | {"effect_basis": "estimated"})
    catch(sc.check_action_block, blk | {"net_value": -1.0})
    catch(sc.check_action_block, blk | {"reads_as": "no action pays"})
    catch(
        sc.check_action_block,
        BLOCKS["does_not_pay"]
        | {"price": "no usable price", "pays": True, "net_value": 1.0, "reads_as": None},
    )
    catch(sc.scenario_of, dataclasses.replace(ACT.base, value_horizon_days=365))
    catch(sc.action_labels, ACT.base, dataclasses.replace(LABELS, scenario_keys=("form",)))
    # Phase 7's customer table and options
    tab = table(5)
    catch(check_customers, tab.drop(columns="p30"))
    catch(check_customers, tab.assign(auto_renew="maybe"))
    catch(check_customers, tab.assign(p30=2.0))
    catch(check_customers, pd.concat([tab, tab.iloc[:1]]))
    rows = pd.DataFrame(
        {
            "msno": ["a"],
            "snapshot_date": [pd.Timestamp(NOV)],
            "last_is_auto_renew": [1],
            "days_until_paid_expiry": [3],
        }
    )
    surv = np.ones((1, 12))
    no_p30 = pd.DataFrame({"msno": ["b"], "snapshot_date": [NOV], "p30": [0.1]})
    prices = pd.DataFrame({"msno": ["a"], "price_per_day": [1.0], "has_price": [True]})
    catch(customer_table, rows, surv, no_p30, prices)
    p30 = no_p30.assign(msno="a")
    catch(customer_table, rows, surv, p30, prices.assign(msno="b").assign(price_per_day=np.nan))
    catch(build_options, tab.drop(columns="auto_renew"), ACT, ACT.base)
    # labels
    cfg = load_serving()
    bad = copy.deepcopy(cfg)
    bad["contract"]["december_2016"]["reasons"] = "withheld"
    catch(lab.load_labels, bad)
    bad = copy.deepcopy(cfg)
    bad["contract"]["outside_population"]["codes"].append("other:<feature>")
    catch(lab.load_labels, bad)
    bad = copy.deepcopy(cfg)
    bad["step4"]["decisions"]["no_usable_price"]["reads_as"] = "nothing"
    catch(lab.load_labels, bad)
    # saved files
    catch(verify_files, __import__("pathlib").Path("/nonexistent"), {"booster_7d.txt": "0" * 64})
    # schema validation errors
    for top, path in LABEL_FIELDS:
        catch(s.Scored, **without(scored(), top, path))
    catch(s.Scored, **scored(DEC, reasons=scored()["reasons"]))
    catch(s.ActionBlock, **(BLOCKS["does_not_pay"] | {"pays": True}))
    catch(
        s.Outside,
        status="outside_population",
        msno="a",
        features_as_of=NOV,
        code="x",
        next_expiry_in_days=1,
    )
    catch(s.ErrorBody, status_code=404, detail="x")
    # error bodies
    msgs += [s.error_body(c).model_dump_json() for c in (404, 409, 422)]
    msgs.append(
        s.error_body(503, reason="a model file differs from the registered one").model_dump_json()
    )
    return msgs


def test_the_scan_finds_a_planted_name():
    assert s.never_used_hits("Tenure and CITY, not percentage or usage") == ["Tenure", "CITY"]


def test_no_message_the_scoring_code_can_raise_names_a_never_used_attribute():
    msgs = all_messages()
    assert len(msgs) > 40
    hits = {m[:80]: s.never_used_hits(m) for m in msgs if s.never_used_hits(m)}
    assert hits == {}


def test_no_valid_response_body_names_a_never_used_attribute():
    for case in BLOCKS:
        body = s.Scored(**scored(action=BLOCKS[case])).model_dump_json()
        assert s.never_used_hits(body) == []
        assert "member for" not in json.loads(body)["reasons"][0]["text"].lower()
