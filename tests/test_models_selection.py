import numpy as np
import pytest
import yaml

from retention.models.bootstrap import FoldScores
from retention.models.selection import (
    Candidate,
    Comparison,
    auc_comparison,
    ladder,
    simplest_first,
)


def cand(id, family="lightgbm", groups=6, columns=46, tuned=True, calibration=None):
    return Candidate(id, family, groups, columns, tuned, calibration)


def test_order_is_the_registered_one_axis_by_axis():
    cfg = yaml.safe_load(open("configs/evaluation.yaml"))["selection"]
    assert cfg["simplicity_order"] == [
        "family",
        "feature_groups",
        "columns",
        "default_over_tuned",
        "platt_over_isotonic",
        "candidate_id",
    ]
    # Family decides first: a simpler family with more groups beats a complex one with fewer.
    a = cand("x", "cox", groups=7)
    b = cand("y", "discrete_time", groups=4)
    assert simplest_first([b, a]) == [a, b]
    # Then feature groups, then columns.
    assert [
        c.id for c in simplest_first([cand("p", groups=6), cand("q", groups=5, columns=99)])
    ] == [
        "q",
        "p",
    ]
    assert [c.id for c in simplest_first([cand("p", columns=44), cand("q", columns=40)])] == [
        "q",
        "p",
    ]
    # Then default over tuned, then Platt over isotonic.
    assert [c.id for c in simplest_first([cand("a"), cand("b", tuned=False)])] == ["b", "a"]
    iso, platt = cand("a", calibration="isotonic"), cand("b", calibration="platt")
    assert simplest_first([iso, platt]) == [platt, iso]


def test_candidate_id_breaks_ties_last():
    assert [c.id for c in simplest_first([cand("c03"), cand("c01"), cand("c02")])] == [
        "c01",
        "c02",
        "c03",
    ]


def test_families_without_a_registered_order_cannot_be_mixed():
    with pytest.raises(ValueError, match="no registered order"):
        simplest_first([cand("a", "lightgbm"), cand("b", "cox")])


def table(results):
    """compare() from a hand-made table: (candidate, current) -> Comparison."""

    def compare(c, cur):
        return results[(c.id, cur.id)]

    return compare


def comp(d, lo, hi, cand_folds, cur_folds):
    return Comparison(d, lo, hi, tuple(cand_folds), tuple(cur_folds))


def test_a_mean_winner_that_loses_on_one_fold_does_not_replace():
    c00, c01 = cand("c00", tuned=False), cand("c01")
    results = {("c01", "c00"): comp(0.010, 0.002, 0.018, [0.80, 0.70], [0.77, 0.71])}
    chosen, steps = ladder([c01, c00], table(results))
    assert chosen.id == "c00"
    assert steps[0].interval_above_zero and not steps[0].better_on_every_fold
    assert not steps[0].replaced


def test_a_candidate_better_by_less_than_its_interval_does_not_replace():
    c00, c01 = cand("c00", tuned=False), cand("c01")
    results = {("c01", "c00"): comp(0.004, -0.001, 0.009, [0.80, 0.72], [0.79, 0.71])}
    chosen, steps = ladder([c00, c01], table(results))
    assert chosen.id == "c00" and steps[0].better_on_every_fold and not steps[0].replaced


def test_a_tie_keeps_the_simpler_candidate():
    simple, complex_ = cand("fs1", groups=5), cand("fs3", groups=6)
    results = {("fs3", "fs1"): comp(0.000, -0.003, 0.003, [0.75, 0.70], [0.75, 0.70])}
    assert ladder([complex_, simple], table(results))[0].id == "fs1"


def test_the_ladder_compares_with_the_current_pick_not_the_first():
    a, b, c = cand("c00", tuned=False), cand("c01"), cand("c02")
    results = {
        ("c01", "c00"): comp(0.02, 0.01, 0.03, [0.80, 0.80], [0.78, 0.78]),  # b replaces a
        ("c02", "c01"): comp(0.005, -0.001, 0.011, [0.81, 0.80], [0.80, 0.79]),  # c ties b
    }
    chosen, steps = ladder([c, b, a], table(results))
    assert chosen.id == "c01" and [s.current for s in steps] == ["c00", "c01"]


def test_auc_comparison_on_synthetic_folds():
    rng = np.random.default_rng(0)
    folds = {}
    for name, noise in (("c00", 2.0), ("c01", 0.5)):
        fs = []
        for k in range(2):
            r = np.random.default_rng(k)
            y = r.integers(0, 2, 2000)
            score = y + rng.normal(0, noise, 2000)
            fs.append(FoldScores(np.arange(2000) + 10_000 * k, y, score, score))
        folds[name] = fs
    chosen, steps = ladder(
        [cand("c00", tuned=False), cand("c01")], auc_comparison(folds, replicates=200, seed=1)
    )
    assert chosen.id == "c01" and steps[0].comparison.lo > 0
