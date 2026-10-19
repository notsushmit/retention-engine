"""Scoring, part 3: expected paid days (A2) and the action block (D1, D2), on Phase 7's code."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.api import scoring as sc
from retention.api.config import load_serving
from retention.api.labels import LABELS
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import load_protocol
from retention.optimize.config import load_actions_config
from retention.optimize.effects import build_options
from retention.optimize.solve import NONE
from retention.optimize.strategies import best_action
from retention.optimize.value import INTERVAL_ENDS_ALL, expected_days
from test_save_dt01 import COLS, RAW, rows

P = load_protocol()
ACT = load_actions_config()
SERVING = load_serving()
T = date(2016, 11, 1)


def table(n: int = 400, seed: int = 0, no_price_every: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = rng.choice([4.97, 4.3, 3.3], n)
    has = np.ones(n, bool)
    if no_price_every:
        has[::no_price_every] = False
        price = np.where(has, price, 0.0)
    return pd.DataFrame(
        {
            "msno": [f"c{i:04d}" for i in range(n)],
            "p30": rng.uniform(0.001, 0.6, n),
            "auto_renew": rng.choice(["off", "on"], n),
            "price_per_day": price,
            "days_30_90": rng.uniform(40, 59.7, n),
            "days_30_365": rng.uniform(100, 300, n),
            "days_until_paid_expiry": pd.array(rng.integers(0, 30, n), "Int32"),
            "has_price": has,
        }
    )


# --- expected paid days through Phase 7's customer table ---------------------------------------


@pytest.fixture(scope="module")
def dt01() -> DiscreteTimeModel:
    cand = {"num_leaves": 7, "n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 20}
    return DiscreteTimeModel(
        False,
        list(INTERVAL_ENDS_ALL),
        cand,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(rows(), COLS)


def test_paid_days_are_phase_7_s_expected_days(dt01):
    r = rows(50, seed=4)
    r = r.assign(
        msno=[f"m{i:02d}" for i in range(50)],
        snapshot_date=pd.Timestamp(T),
        days_until_paid_expiry=r.days_until_expiry,
    )
    surv = sc.survival(dt01, r)
    assert np.array_equal(surv, 1 - dt01.lapse_probability(r, list(INTERVAL_ENDS_ALL))[0])
    reasons_30 = pd.DataFrame({"msno": r.msno, "snapshot_date": T, "probability": 0.05})
    prices = pd.DataFrame({"msno": r.msno, "price_per_day": 4.97, "has_price": True})
    tab = sc.customers(r, surv, reasons_30, prices)
    assert np.array_equal(tab.days_30_90.to_numpy(), expected_days(surv).days_30_90.to_numpy())
    assert np.array_equal(tab.p30.to_numpy(), np.full(50, 0.05))
    block = sc.paid_days_block(float(tab.days_30_90.iloc[0]))
    assert block["field"] == "expected_paid_days_31_90"
    assert block["name"] == "expected paid days in days 31 to 90, given no lapse by day 30"
    assert "near the ceiling for almost everyone; not a ranking signal" in block["labels"]


# --- the action ---------------------------------------------------------------------------------


def test_actions_are_phase_7_s_best_action_under_the_base_scenario():
    tab = table()
    acts = sc.actions(tab, ACT)
    opts = build_options(tab, ACT, ACT.base)
    best = best_action(opts)
    assert np.array_equal(acts.best, best)
    f = acts.frame()
    chosen = best != NONE
    i = np.flatnonzero(chosen)
    assert np.array_equal(f.net_value.to_numpy()[i], opts.net[i, best[i]])
    assert np.array_equal(f.cost.to_numpy()[i], opts.cost[i, best[i]])
    assert np.array_equal(
        f.expected_retained_value.to_numpy()[i], opts.value[i] * opts.effect[i, best[i]]
    )
    assert (f.highest_net_value_action.to_numpy()[~chosen] == "none").all()
    assert np.array_equal(f.pays.to_numpy(), chosen & (f.net_value.to_numpy() > 0))
    assert (f.loc[~chosen, ["net_value", "cost", "expected_retained_value"]] == 0).all(axis=None)


def test_no_usable_price_offers_no_positive_net_action_and_says_so():
    tab = table(no_price_every=4)
    acts = sc.actions(tab, ACT)
    no_price = ~tab.has_price.to_numpy()
    allowed_net = np.where(acts.opts.allowed, acts.opts.net, -np.inf)
    assert (allowed_net[no_price] <= 0).all()  # valued at 0: nothing can pay
    blocks = sc.action_blocks(acts, ACT.base)
    for k in np.flatnonzero(no_price):
        assert blocks[k]["price"] == "no usable price"
        assert blocks[k]["reads_as"] == "no action pays"
        assert blocks[k]["net_value"] <= 0
        assert blocks[k]["pays"] is False
    for k in np.flatnonzero(~no_price):
        assert "price" not in blocks[k]
        assert ("reads_as" in blocks[k]) == (not blocks[k]["pays"])


# --- labels on every block ------------------------------------------------------------------------


def test_every_block_carries_the_four_fields_and_the_probability_basis():
    blocks = sc.action_blocks(sc.actions(table(60), ACT), ACT.base)
    req = SERVING["contract"]["scored"]["action"]["required"]
    for b in blocks:
        assert b["effect_basis"] == "assumed"
        assert list(b["scenario"]) == req["scenario"]
        assert b["instability_note"] == "The plan is unstable under 5 of 11 variants."
        assert b["no_estimate_line"] == (
            "No KKBox effect estimate exists; every effect is an assumption."
        )
        assert b["probability_basis"] == (
            "computed from the 30-day probability before the ordering fix, as in Phase 7"
        )


@pytest.mark.parametrize("field", sc.REQUIRED_ACTION_FIELDS)
def test_a_block_missing_any_registered_field_is_refused(field):
    block = sc.action_blocks(sc.actions(table(5), ACT), ACT.base)[0]
    with pytest.raises(ValueError, match="lacks"):
        sc.check_action_block({k: v for k, v in block.items() if k != field})


def test_an_effect_basis_other_than_assumed_is_refused():
    block = sc.action_blocks(sc.actions(table(5), ACT), ACT.base)[0]
    with pytest.raises(ValueError, match="assumed"):
        sc.check_action_block(block | {"effect_basis": "estimated"})


def test_the_scenario_comes_from_phase_7_s_base_scenario_and_a4_text():
    s = sc.scenario_of(ACT.base)
    assert (s["form"], s["level"], s["margin"], s["cost_level"]) == (
        "relative",
        "base",
        0.5,
        "base",
    )
    a4 = SERVING["contract"]["action_reading"]["scenario_text"]
    assert {k: s[k] for k in a4} == a4
    with pytest.raises(ValueError, match="wording is registered"):
        sc.scenario_of(dataclasses.replace(ACT.base, value_horizon_days=365))
    labels = dataclasses.replace(LABELS, scenario_keys=("form",))
    with pytest.raises(ValueError, match="scenario fields"):
        sc.action_labels(ACT.base, labels)


def test_action_labels_are_read_from_the_registered_config():
    raw = yaml.safe_load(open("configs/serving.yaml"))
    labels = sc.action_labels(ACT.base)
    req = raw["contract"]["scored"]["action"]["required"]
    assert labels["instability_note"] == req["instability_note"]
    assert labels["no_estimate_line"] == req["no_estimate_line"]


def test_the_three_a4_cases():
    tab = table(400, seed=3, no_price_every=7)
    acts = sc.actions(tab, ACT)
    f = acts.frame()
    blocks = sc.action_blocks(acts, ACT.base)
    cases = {"pays": 0, "does_not_pay": 0, "none_allowed": 0}
    for b, r in zip(blocks, f.to_dict("records"), strict=True):
        if r["highest_net_value_action"] == "none":
            cases["none_allowed"] += 1
            assert not b["pays"] and b["reads_as"] == "no action pays"
            assert (b["net_value"], b["cost"], b["expected_retained_value"]) == (0.0, 0.0, 0.0)
        elif b["pays"]:
            cases["pays"] += 1
            assert b["net_value"] > 0 and "reads_as" not in b
        else:
            cases["does_not_pay"] += 1
            assert b["net_value"] <= 0 and b["reads_as"] == "no action pays"
    assert all(v > 0 for v in cases.values()), cases


@pytest.mark.parametrize(
    "change",
    [
        {"pays": True, "net_value": -1.0},
        {"pays": False, "net_value": 2.0},
        {"pays": True, "net_value": 2.0, "reads_as": "no action pays"},
    ],
)
def test_a_block_breaking_a4_is_refused(change):
    block = sc.action_blocks(sc.actions(table(5), ACT), ACT.base)[0]
    block = {k: v for k, v in block.items() if k != "reads_as"} | change
    with pytest.raises(ValueError):
        sc.check_action_block(block)
