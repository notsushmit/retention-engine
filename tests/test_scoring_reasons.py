"""Scoring, part 2: reasons with tenure never shown (amendment A3), by two separate routes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from retention.api import scoring as sc
from retention.explain import build_reasons, selection
from retention.explain.build_reasons import reason_rows
from retention.explain.contributions import SavedModel
from retention.explain.model import booster_sha256, fit_models
from retention.models.protocol import load_protocol
from test_explain_build_reasons import CFG, NOV, OCT, PLATT, SHA, R, make_facts, make_part
from test_registry import SETTINGS, synthetic

P = load_protocol()
NEVER = ("age", "gender", "registered_via", "tenure", "city", "member for")


def with_tenure_pushing(part: pd.DataFrame, every: int, push: float = 2.0) -> pd.DataFrame:
    """Make tenure the strongest raising group on every `every`-th row, keeping raw and d
    consistent with the group sums (as Phase 4's contributions are)."""
    out = part.copy()
    rows = out.index[::every]
    out.loc[rows, "g_tenure"] += push
    out.loc[rows, "raw_score"] += push
    out["d"] = out.platt_b * (out.raw_score - out.base_value)
    return out


@pytest.fixture(scope="module")
def case():
    part = with_tenure_pushing(make_part(120, 7, NOV, horizon=30), every=3)
    facts = make_facts(part.msno, 7)
    args = (30, NOV, PLATT, P, SHA)
    return part, facts, args


def service(part, facts, args):
    return sc.service_reasons(CFG, R, part, facts, *args)


def self_check(part, facts, args):
    return sc.self_check_reasons(CFG, R, part, facts, *args)


# --- the mask ------------------------------------------------------------------------------------


def test_forced_mask_hides_hidden_and_shown_only_when_known_groups():
    mask = sc.forced_hidden_mask(CFG, 4)
    hidden = {g for g in CFG.group_names if g in CFG.hidden or g in CFG.shown_only_when_known}
    assert hidden == {"hidden", "tenure"}
    for k, g in enumerate(CFG.group_names):
        assert mask[:, k].tolist() == [g not in hidden] * 4


def test_service_route_never_reads_tenure_days(case):
    part, facts, args = case
    a = service(part, facts, args)
    b = service(part, facts.assign(tenure_days=pd.NA), args)
    c = service(part, facts.assign(tenure_days=pd.array([9999] * len(facts), "Int32")), args)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(a, c, check_exact=True)


# --- the two routes ------------------------------------------------------------------------------


def test_routes_do_not_share_the_masking_call(case, monkeypatch):
    part, facts, args = case

    def refuse(*a, **k):
        raise RuntimeError("shown_mask called")

    monkeypatch.setattr(selection, "shown_mask", refuse)
    monkeypatch.setattr(build_reasons, "shown_mask", refuse)
    assert not hasattr(sc, "shown_mask")  # the service module never imports it
    service(part, facts, args)  # the service route masks without Phase 4's shown_mask
    with pytest.raises(RuntimeError, match="shown_mask called"):
        self_check(part, facts, args)  # the self-check masks through it


def test_self_check_equals_the_service_route(case):
    part, facts, args = case
    pd.testing.assert_frame_equal(
        service(part, facts, args), self_check(part, facts, args), check_exact=True
    )


def test_rows_without_tenure_equal_phase_4_and_rows_with_it_do_not(case):
    part, facts, args = case
    phase4 = reason_rows(CFG, R, part, facts, *args)
    served = service(part, facts, args)
    slots = ["raising_1", "raising_2", "raising_3", "lowering_1", "lowering_2"]
    listed = (phase4[slots] == "tenure").any(axis=1).to_numpy()
    assert 0 < listed.sum() < len(listed)
    pd.testing.assert_frame_equal(
        served[~listed].reset_index(drop=True),
        phase4[~listed].reset_index(drop=True),
        check_exact=True,
    )
    assert (served[listed].text.to_numpy() != phase4[listed].text.to_numpy()).all()


# --- no tenure anywhere --------------------------------------------------------------------------


@pytest.mark.parametrize("horizon_and_month", [(30, NOV), (7, OCT)])
def test_no_tenure_slot_and_no_member_for_text(horizon_and_month):
    h, t = horizon_and_month
    part = with_tenure_pushing(make_part(90, 11, t, horizon=h), every=2)
    facts = make_facts(part.msno, 11)
    phase4 = reason_rows(CFG, R, part, facts, h, t, PLATT, P, SHA)
    served = sc.service_reasons(CFG, R, part, facts, h, t, PLATT, P, SHA)
    slots = ["raising_1", "raising_2", "raising_3", "lowering_1", "lowering_2"]
    assert (phase4[slots] == "tenure").any(axis=1).sum() > 0  # the test can see tenure
    assert not (served[slots] == "tenure").any(axis=None)
    assert not served.text.str.lower().str.contains("member for").any()


def error_messages(case) -> list[str]:
    """Every error the reasons routes raise: duplicate facts, a non-positive Platt slope."""
    part, facts, args = case
    msgs = []
    with pytest.raises(pd.errors.MergeError) as e:
        service(part, pd.concat([facts, facts.iloc[:1]]), args)
    msgs.append(str(e.value))
    with pytest.raises(ValueError) as e:
        service(part.assign(platt_b=0.0), facts, args)
    msgs.append(str(e.value))
    return msgs


def test_error_bodies_name_no_never_used_attribute(case):
    msgs = error_messages(case)
    assert len(msgs) == 2
    for m in msgs:
        assert not any(w in m.lower() for w in NEVER), m


# --- the contributions part -----------------------------------------------------------------------


def test_contribution_part_follows_phase_4(monkeypatch):
    cfg = dataclasses.replace(
        CFG,
        groups={"renewal_timing": ["a"], "tenure": ["b"], "hidden": ["cat"]},
        hidden=("hidden",),
        shown_only_when_known={"tenure": "b"},
    )
    model, platt = fit_models(synthetic(1500, 3), synthetic(600, 4), SETTINGS, 7, P)
    saved = SavedModel(7, model, platt.a, platt.b, booster_sha256(model))
    rows = synthetic(40, 5).assign(msno=[f"x{i:02d}" for i in range(40)])
    rows["snapshot_date"] = pd.Timestamp(NOV)

    def refuse(*a, **k):
        raise RuntimeError("shown_mask called")

    monkeypatch.setattr(selection, "shown_mask", refuse)
    part = sc.contribution_part(saved, rows, cfg, ["a", "b", "cat"], 7)
    assert part.msno.tolist() == rows.msno.tolist()
    phi = part[["phi_a", "phi_b", "phi_cat"]].to_numpy()
    assert np.allclose(phi.sum(axis=1) + part.base_value, part.raw_score, atol=1e-9, rtol=0)
    assert np.array_equal(part.g_tenure.to_numpy(), part.phi_b.to_numpy())
    assert np.array_equal(
        part.d.to_numpy(), platt.b * (part.raw_score - part.base_value).to_numpy()
    )
