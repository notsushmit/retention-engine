"""Archetypes: seeds, samples, the vote, one-to-one matching, selection, criteria, diagnostic,
and the expiry-branch decision table."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from retention.explain import archetypes as A
from retention.explain.archetypes import (
    ELEVEN,
    NOISE,
    RULE,
    TEN,
    Fit,
    Rows,
    Setting,
    ami_with_expiry,
    apply_failure_rule,
    assign,
    choose,
    decide_version,
    draw,
    group_matrix,
    jaccard_matrix,
    match,
    profile_ok,
    qualification,
    restates_expiry,
    same_selection,
    selection_frame,
    share_ok,
    unassigned_shares,
    version_holds_up,
    vote,
)

ACFG = A.load_archetypes_config()
OCT, NOV, JAN, FEB = A.OCT16, A.NOV16, A.JAN17, A.FEB17


def test_seeds_are_the_registered_ones():
    s = ACFG["seeds"]
    assert A.BASE_SEED == 20161001
    assert A.SEEDS["resamples"] == tuple(s["resamples"])
    for k, v in s.items():
        if k != "resamples":
            assert A.SEEDS[k] == v, k


# --- matching -----------------------------------------------------------------------------------


def test_jaccard_values():
    a = np.array([0, 0, 1, 1, NOISE])
    b = np.array([5, 5, 5, 7, 7])
    m, ca, cb = jaccard_matrix(a, b)
    assert ca == [0, 1] and cb == [5, 7]
    assert m.tolist() == [[2 / 3, 0.0], [1 / 4, 1 / 3]]


def test_matching_is_one_to_one():
    # Clusters 0 and 1 of `a` both overlap cluster 9 of `b` most; only one may take it.
    a = np.array([0] * 10 + [1] * 10 + [2] * 10)
    b = np.array([9] * 10 + [9] * 6 + [8] * 4 + [8] * 10)
    m = match(a, b)
    partners = [p for p, _ in m.values() if p is not None]
    assert len(partners) == len(set(partners))
    assert m[0] == (9, pytest.approx(10 / 16))
    assert m[1][0] in (8, None)


def test_unmatched_archetype_scores_zero_and_noise_is_never_matched():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([4, 4, 5, 5, NOISE, NOISE])
    m = match(a, b)
    assert m[2] == (None, 0.0)
    assert all(p != NOISE for p, _ in m.values())


# --- the vote and assignment ----------------------------------------------------------------------


def make_fit(points, labels, k=3):
    x = np.asarray(points, float)
    return Fit(np.arange(len(x)), x, np.asarray(labels), NearestNeighbors(n_neighbors=k).fit(x))


def test_vote_majority_noise_and_ties():
    f = make_fit([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], [0, 0, NOISE, 1, NOISE, 1])
    assert vote(f, np.array([[0.4]])).tolist() == [0]  # 0, 0, noise
    assert vote(f, np.array([[11.2]])).tolist() == [1]  # noise, 1, 1
    g = make_fit([[0.0], [1.0], [3.0], [4.0]], [0, 1, 0, 1], k=4)
    # every label has 2 votes: the nearest row decides
    assert vote(g, np.array([[0.9]])).tolist() == [1]
    assert vote(g, np.array([[0.1]])).tolist() == [0]
    h = make_fit([[0.0], [1.0], [2.0]], [NOISE, NOISE, 0])
    assert vote(h, np.array([[0.5]])).tolist() == [NOISE]


def test_fit_sample_rows_keep_their_hdbscan_label():
    f = make_fit([[0.0], [1.0], [2.0], [3.0]], [NOISE, 0, 0, 0])
    f.sample = np.array([10, 11, 12, 13])
    x = np.zeros((20, 1))
    x[10:14, 0] = [0.0, 1.0, 2.0, 3.0]
    out = assign(f, x, np.array([10, 13, 5]))
    assert out[0] == NOISE and out[1] == 0  # own labels, not the vote
    assert out[2] in (0, NOISE)


# --- representation and samples -------------------------------------------------------------------


def test_group_matrix_scales_by_b_and_zeroes_missing_tenure():
    part = pd.DataFrame({"platt_b": [2.0, 2.0], "g_a": [1.0, -1.0], "g_tenure": [0.5, 0.5]})
    x = group_matrix(part, ["a", "tenure"], np.array([True, False]))
    assert x.tolist() == [[2.0, 1.0], [-2.0, 0.0]]


def test_input_groups_leave_out_the_hidden_group():
    assert "hidden" not in ACFG["input"]["groups"] and len(ACFG["input"]["groups"]) == 11


def synthetic_rows(n_per=400, seed=0, groups=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    snaps = [OCT, NOV, JAN, FEB]
    centers = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
    x, snap, rule = [], [], []
    for t in snaps:
        k = rng.integers(0, 3, n_per)
        x.append(centers[k] + 0.15 * rng.normal(size=(n_per, 3)))
        snap += [t] * n_per
        rule.append(rng.random(n_per) < 0.05)
    x = np.vstack(x)
    return Rows(
        msno=np.array([f"m{i:05d}" for i in range(len(x))]),
        snapshot=np.array(snap, dtype=object),
        rule=np.concatenate(rule),
        x=x,
        groups=list(groups),
        days_until_expiry=rng.integers(0, 31, len(x)).astype(float),
        x7=x[:, ::-1].copy(),
    )


def small_config():
    acfg = copy.deepcopy(ACFG)
    acfg["clustering"]["fit_sample"] = {"rows": 200, "per_snapshot": {OCT: 100, NOV: 100}}
    acfg["evaluation_sets"]["E"]["per_snapshot"] = {OCT: 80, NOV: 80}
    acfg["evaluation_sets"]["E_2017"]["per_snapshot"] = {JAN: 80, FEB: 80}
    acfg["expiry_diagnostic"]["rows"] = 150
    acfg["clustering"]["min_cluster_size_grid"] = [10, 20, 40]
    acfg["clustering"]["min_samples"] = 5
    acfg["seeds"] = dict(acfg["seeds"])
    return acfg


def test_samples_are_disjoint_where_required_deterministic_and_non_rule():
    rows = synthetic_rows()
    acfg = small_config()
    s = A.draw_samples(rows, acfg)
    t = A.draw_samples(rows, acfg)
    assert np.array_equal(s.fit, t.fit) and np.array_equal(s.evaluation, t.evaluation)
    assert not set(s.fit) & set(s.evaluation)
    assert not set(s.fit_2017) & set(s.evaluation_2017)
    for idx in (s.fit, s.evaluation, s.fit_2017, s.diagnostic, *s.resamples):
        assert not rows.rule[idx].any()
    assert (rows.snapshot[s.diagnostic] == OCT).all()
    assert set(rows.snapshot[s.month_fit[NOV]]) == {NOV}
    assert len(s.resamples) == 20
    assert not np.array_equal(s.resamples[0], s.resamples[1])
    one = draw(rows, 0, {OCT: 50})
    assert np.array_equal(one, draw(rows, 0, {OCT: 50}))


# --- selection ------------------------------------------------------------------------------------


def setting(size, aris, ok=True):
    f = make_fit([[0.0], [1.0], [2.0]], [0, 1, 2])
    return Setting(size, f, np.array([0, 1, 2]), [], list(aris), ok)


def test_choose_takes_the_highest_median_ari_and_breaks_ties_to_the_larger_size():
    a, b, c = setting(200, [0.8, 0.8]), setting(400, [0.85, 0.85]), setting(800, [0.9, 0.95], False)
    assert choose([a, b, c]).min_cluster_size == 400  # 800 does not qualify
    d = setting(800, [0.85, 0.85])
    assert choose([a, b, d]).min_cluster_size == 800
    assert choose([c]) is None


def test_selection_and_judging_on_three_clear_clusters():
    rows = synthetic_rows()
    acfg = small_config()
    s = A.draw_samples(rows, acfg)
    chosen, settings = A.select_setting(rows.x, rows.rule, s, acfg, threads=1)
    again, _ = A.select_setting(rows.x, rows.rule, s, acfg, threads=1)
    assert chosen is not None and chosen.reference.n_clusters == 3
    assert np.array_equal(chosen.reference.labels, again.reference.labels)
    j = A.judge(rows, s, acfg, chosen, threads=1)
    assert all(j["holds_up"].values()) and j["7_count"]["passed"]
    for v in j["6_unassigned"].values():
        assert v["all_rows_with_rule"] <= v["non_rule_rows"] + 1e-12
    diag = A.expiry_diagnostic(rows, s, acfg, chosen, threads=1)
    assert diag["AMI_30_reference"] < 0.1  # clusters unrelated to the random expiry days
    assert {"restates_expiry", "assumption_unsupported"} <= set(diag)


# --- criteria pieces ------------------------------------------------------------------------------

CRIT = ACFG["criteria"]


def test_profile_rule_cosine_largest_group_and_near_zero():
    c3 = CRIT["3_shap_profile"]
    ok, info = profile_ok(np.array([1.0, 0.2, 0.0]), np.array([0.9, 0.25, 0.05]), c3)
    assert ok and info["rule"] == "cosine"
    ok, _ = profile_ok(np.array([1.0, 0.2, 0.0]), np.array([0.2, 1.0, 0.0]), c3)
    assert not ok
    ok, info = profile_ok(np.array([0.05, 0.05, 0.0]), np.array([-0.05, 0.05, 0.0]), c3)
    assert ok and info["rule"] == "distance"  # near zero: distance, not cosine
    ok, _ = profile_ok(np.array([0.05, 0.05, 0.0]), np.array([0.3, 0.05, 0.0]), c3)
    assert not ok


def test_share_rule_needs_both_limits():
    c4 = CRIT["4_shares"]
    assert share_ok(0.30, 0.33, c4)[0]
    ok, info = share_ok(0.30, 0.36, c4)  # 6 points: absolute fails, ratio fine
    assert not ok and not info["absolute_ok"] and info["ratio_ok"]
    ok, info = share_ok(0.02, 0.05, c4)  # 3 points, ratio 2.5: ratio fails alone
    assert not ok and info["absolute_ok"] and not info["ratio_ok"]
    ok, info = share_ok(0.008, 0.004, c4)  # under 0.5% in the month
    assert not ok and not info["minimum_ok"]


def test_ami_with_expiry_branch_values():
    days = np.repeat(np.arange(31), 20).astype(float)
    assert ami_with_expiry((days // 8).astype(int), days) > 0.5
    rng = np.random.default_rng(0)
    assert abs(ami_with_expiry(rng.integers(0, 4, len(days)), days)) < 0.05


def test_the_share_series_has_a_labelled_gap_month():
    assert ACFG["input"]["not_explained"] == [date(2016, 12, 1)]


# --- unassigned shares: three ways, reported only -------------------------------------------------


def test_unassigned_shares_and_their_denominators():
    f = make_fit([[0.0], [1.0], [2.0], [3.0]], [NOISE, NOISE, 0, 0])
    f.sample = np.array([0, 1, 2, 3])
    # rows 0-3 fit sample (2 noise); 4-7 voted (1 noise); 8-9 rule-defined (assigned)
    labels = np.array([NOISE, NOISE, 0, 0, NOISE, 0, 0, 0, RULE, RULE])
    e = np.array([0] * 9 + [NOISE])
    out = unassigned_shares(f, labels, e)
    assert (out["fit_sample"]["unassigned"], out["fit_sample"]["rows"]) == (2, 4)
    assert (out["voted"]["unassigned"], out["voted"]["rows"]) == (1, 4)
    assert (out["overall"]["unassigned"], out["overall"]["rows"]) == (3, 10)
    assert out["on_E"]["share"] == pytest.approx(0.1)
    for v in out.values():
        assert v["denominator"]
    assert "decides qualification" in out["on_E"]["denominator"]
    for k in ("fit_sample", "voted", "overall"):
        assert "reported only" in out[k]["denominator"]
    assert "count as assigned" in out["overall"]["denominator"]
    # half the fit sample is noise and 30% overall, but qualification reads E alone
    assert qualification(3, e, ACFG["clustering"]["selection"]) == []
    assert qualification(3, np.array([NOISE] * 3 + [0] * 7), ACFG["clustering"]["selection"])


def test_two_grid_runs_are_identical_after_sorting_with_threads():
    rows = synthetic_rows()
    acfg = small_config()
    s = A.draw_samples(rows, acfg)
    runs = []
    for _ in range(2):
        chosen, settings = A.select_setting(rows.x, rows.rule, s, acfg, threads=4)
        report = {
            "settings": [x.summary() for x in settings],
            "chosen": chosen.min_cluster_size,
            "seconds": 1.0,
        }
        runs.append((report, selection_frame(rows.msno, rows.snapshot, settings), settings))
    (ra, fa, sa), (rb, fb, _) = runs
    rb["seconds"] = 2.0  # time is left out of the comparison
    fb = fb.sample(frac=1.0, random_state=0)  # row order is not part of the output
    result = same_selection(ra, fa, rb, fb)
    assert result["identical"] and result["rows"] == 3 * len(rows.msno)
    # the E labels used for qualification are the labels written for those rows
    for x in sa:
        assert np.array_equal(x.labels[s.evaluation], x.e_labels)
        assert (x.labels[rows.rule] == RULE).all()
    changed = fb.copy()
    changed.iloc[0, changed.columns.get_loc("label")] += 1
    assert not same_selection(ra, fa, rb, changed)["labels_identical"]


# --- the expiry-branch decision table -------------------------------------------------------------


def test_branch_thresholds_are_the_registered_ones():
    b = ACFG["expiry_diagnostic"]["branches"]
    assert b["restates_expiry"]["when"] == f"AMI_30_reference >= {A.RESTATES_EXPIRY_MIN_AMI}"
    assert b["assumption_unsupported"]["when"] == f"AMI_7 < AMI_30 + {A.UNSUPPORTED_MARGIN:.2f}"


BELOW = float(np.nextafter(0.5, 0.0))


def test_restates_expiry_at_exactly_one_half_and_one_step_below():
    assert restates_expiry(0.5)
    assert not restates_expiry(BELOW)
    assert BELOW < 0.5 and np.nextafter(BELOW, 1.0) == 0.5


@pytest.mark.parametrize(
    "holds_11, holds_10, chosen, beside, failed",
    [
        (True, True, TEN, ELEVEN, []),
        (True, False, ELEVEN, None, [TEN]),
        (False, True, TEN, None, [ELEVEN]),
        (False, False, None, None, [ELEVEN, TEN]),
    ],
)
def test_decision_table_at_one_half(holds_11, holds_10, chosen, beside, failed):
    d = decide_version(0.5, holds_11, holds_10)
    assert d["archetype_set"] == chosen and d["reported_beside"] == beside
    assert d["failed_versions"] == failed
    assert d["done_when_met"] is (chosen is not None)


def test_decision_table_below_one_half_judges_the_11_group_version_only():
    assert decide_version(BELOW, True, None)["archetype_set"] == ELEVEN
    d = decide_version(BELOW, False, None)
    assert d["archetype_set"] is None and d["failed_versions"] == [ELEVEN]
    assert not d["done_when_met"]
    with pytest.raises(ValueError):
        decide_version(BELOW, True, True)  # the 10-group version is never judged below 0.5
    with pytest.raises(ValueError):
        decide_version(0.5, True, None)  # at 0.5 both must be judged


def judgement(holds, c6=True, c7=True):
    return {
        "holds_up": holds,
        "6_unassigned": {"2016-10-01": {"passed": c6}, "2016-11-01": {"passed": True}},
        "7_count": {"archetypes": len(holds), "passed": c7},
    }


def test_version_holds_up_is_strict():
    assert version_holds_up(judgement({0: True, 1: True, 2: True}))
    # one failing archetype with criteria 6 and 7 passing is not a holding version
    assert not version_holds_up(judgement({0: True, 1: True, 2: True, 3: False}))
    assert not version_holds_up(judgement({0: True, 1: True, 2: True}, c6=False))
    assert not version_holds_up(judgement({0: True, 1: True, 2: True}, c7=False))
    assert not version_holds_up(judgement({}))
    assert not version_holds_up(None)  # no grid setting qualified


def test_reduced_set_is_reported_but_never_selected_on():
    # four archetypes; archetype 3 (5% of rows) fails criteria 1 to 5
    n = 100
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 5 + [NOISE] * 5)
    rows = Rows(
        msno=np.array([f"m{i}" for i in range(n)]),
        snapshot=np.array([OCT] * n, dtype=object),
        rule=np.zeros(n, bool),
        x=np.zeros((n, 2)),
        groups=["a", "b"],
        days_until_expiry=np.zeros(n),
    )
    j = judgement({0: True, 1: True, 2: True, 3: False}) | {"_labels": labels}
    reduced = apply_failure_rule(j, rows, ACFG["criteria"])
    assert reduced["kept_archetypes"] == [0, 1, 2] and reduced["failed_archetypes"] == [3]
    assert reduced["unassigned_after"]["2016-10-01"] == pytest.approx(0.10)
    assert reduced["6_after_passed"] and reduced["7_after_passed"]
    holds = version_holds_up(j)
    assert not holds
    assert decide_version(BELOW, holds, None)["archetype_set"] is None
    assert decide_version(0.5, holds, False)["archetype_set"] is None


def test_below_one_half_nothing_is_rerun(monkeypatch):
    rows = synthetic_rows(groups=("renewal_timing", "b", "c"))
    acfg = small_config()
    s = A.draw_samples(rows, acfg)
    chosen, _ = A.select_setting(rows.x, rows.rule, s, acfg, threads=1)

    def refuse(*args, **kwargs):
        raise AssertionError("nothing is rerun below the AMI branch")

    monkeypatch.setattr(A, "select_setting", refuse)
    monkeypatch.setattr(A, "without_group", refuse)
    out = A.judge_versions(rows, s, acfg, chosen, BELOW, threads=1)
    assert TEN not in out
    assert out["decision"]["branch"].startswith("11 groups only")
    assert out["decision"]["archetype_set"] == (ELEVEN if out[ELEVEN]["holds_up"] else None)
    assert "never used to choose" in out[ELEVEN]["reduced_set_reported_only"]["note"]


def test_at_one_half_the_10_group_version_is_reselected_and_judged():
    rows = synthetic_rows(groups=("renewal_timing", "b", "c"))
    acfg = small_config()
    s = A.draw_samples(rows, acfg)
    chosen, _ = A.select_setting(rows.x, rows.rule, s, acfg, threads=1)
    out = A.judge_versions(rows, s, acfg, chosen, 0.5, threads=1)
    assert out[ELEVEN]["groups"] == ["renewal_timing", "b", "c"]
    assert out[TEN]["groups"] == ["b", "c"]
    assert len(out[TEN]["settings"]) == 3 and out[TEN]["chosen"] is not None
    assert out["decision"] == decide_version(0.5, out[ELEVEN]["holds_up"], out[TEN]["holds_up"])
