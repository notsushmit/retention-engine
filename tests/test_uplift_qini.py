"""Hand-made Qini cases with known answers, the tie rule, and the Poisson bootstrap."""

import numpy as np
import pytest

from retention.uplift import qini


def paired_case(blocks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """100 pairs ranked treated, control, treated, control, ...; four blocks of 25 pairs.

    blocks[j] = (treated outcome, control outcome) for every pair in block j.
    Scores fall with position, so cut k_i = 2i holds exactly i treated and i control.
    """
    y, t = [], []
    for yt, yc in blocks:
        for _ in range(25):
            y += [yt, yc]
            t += [1, 0]
    n = len(y)
    return np.array(y, float), np.array(t, float), np.arange(n, 0, -1, dtype=float)


# Hand case B. Blocks (treated, control): (1, 0), (1, 1), (0, 0), (0, 0).
# N = 200; at cut i (i = 1..100) the top 200 * i / 100 = 2i customers are i treated and
# i control, so N_T / N_C = 1 and Q(phi_i) = (YT_i - YC_i) / 200 = D_i / 200.
#   pairs  1-25: D_i = i            sum = 1 + ... + 25 = 325
#   pairs 26-50: D_i = 25           sum = 25 * 25      = 625
#   pairs 51-75: D_i = 25           sum = 625
#   pairs 76-100: D_i = 25          sum = 625
#   sum of D_i = 2200, mean Q = 2200 / 100 / 200 = 0.11
#   Q(1) = D_100 / 200 = 25 / 200 = 0.125
#   mean of phi_i * Q(1) = 0.125 * (1 + ... + 100) / 100 / 100 = 0.125 * 0.505 = 0.063125
#   A = 0.11 - 0.063125 = 0.046875
HAND_B_BLOCKS = [(1, 0), (1, 1), (0, 0), (0, 0)]
HAND_B_AREA = 0.046875

# Hand case A. Blocks (1, 0), (1, 1), (0, 0), (0, 1): the last block gives control outcomes.
#   pairs 76-100: D_i = 25 - (i - 75) = 100 - i   sum = 24 + 23 + ... + 0 = 300
#   sum of D_i = 325 + 625 + 625 + 300 = 1875, mean Q = 1875 / 20000 = 0.09375
#   Q(1) = 0 / 200 = 0, so A = 0.09375
HAND_A_BLOCKS = [(1, 0), (1, 1), (0, 0), (0, 1)]
HAND_A_AREA = 0.09375


def area_by_definition(y: np.ndarray, t: np.ndarray, order: np.ndarray) -> float:
    """The registered definition written out with a plain loop (no cumulative sums)."""
    n = len(y)
    q = []
    for i in range(1, 101):
        top = order[: (i * n) // 100]
        n_t, n_c = t[top].sum(), (1 - t[top]).sum()
        y_t, y_c = (y[top] * t[top]).sum(), (y[top] * (1 - t[top])).sum()
        q.append((y_t - (y_c * n_t / n_c if n_c else 0.0)) / n)
    q = np.array(q)
    phi = np.arange(1, 101) / 100
    return float(np.mean(q - phi * q[-1]))


@pytest.mark.parametrize(
    ("blocks", "area"), [(HAND_A_BLOCKS, HAND_A_AREA), (HAND_B_BLOCKS, HAND_B_AREA)]
)
def test_hand_case_area_by_hand_and_in_code(blocks, area):
    y, t, s = paired_case(blocks)
    assert qini.qini_area(y, t, s) == pytest.approx(area, abs=1e-12)
    assert area_by_definition(y, t, np.arange(len(y))) == pytest.approx(area, abs=1e-12)


def test_hand_case_curve_points():
    y, t, s = paired_case(HAND_B_BLOCKS)
    phi, q = qini.qini_curve(y, t, s)
    assert phi[0] == 0.01 and phi[-1] == 1.0
    assert q[0] == pytest.approx(1 / 200)  # one treated responder, one control non-responder
    assert q[24] == pytest.approx(25 / 200)
    assert q[-1] == pytest.approx(25 / 200)


def test_unequal_arms_use_the_ratio():
    # top 100 of 200: 75 treated (30 responders), 25 control (5 responders);
    # Q(0.5) = (30 - 5 * 75 / 25) / 200 = 15 / 200
    y = np.zeros(200)
    t = np.zeros(200)
    t[:75] = 1
    y[:30] = 1
    y[75:80] = 1
    t[100:125] = 1
    s = np.arange(200, 0, -1, dtype=float)
    _, q = qini.qini_curve(y, t, s)
    assert q[49] == pytest.approx(15 / 200)
    assert qini.qini_area(y, t, s) == pytest.approx(area_by_definition(y, t, np.arange(200)))


def test_reversed_ranking_mirrors_the_perfect_one_on_equal_counts():
    y, t, s = paired_case(HAND_B_BLOCKS)
    assert qini.qini_area(y, t, -s) == pytest.approx(-HAND_B_AREA, abs=1e-12)


def synthetic(n: int = 2000, seed: int = 1):
    """Known potential outcomes.

    20% persuadable, 30% sure thing, 40% lost cause, 10% do-not-disturb.
    """
    rng = np.random.default_rng(seed)
    kind = rng.choice(4, size=n, p=[0.2, 0.3, 0.4, 0.1])
    y1 = np.isin(kind, [0, 1]).astype(float)
    y0 = np.isin(kind, [1, 3]).astype(float)
    t = np.zeros(n)
    t[rng.permutation(n)[: n // 2]] = 1
    return np.where(t == 1, y1, y0), t, y1 - y0


def random_areas(y, t, k: int = 1000) -> np.ndarray:
    return np.array(
        [qini.qini_area(y, t, np.random.default_rng(1000 + i).random(len(y))) for i in range(k)]
    )


def test_perfect_ranking_beats_every_random_ranking():
    y, t, tau = synthetic()
    perfect = qini.qini_area(y, t, tau)
    assert perfect > random_areas(y, t).max()


def test_random_rankings_average_near_zero():
    y, t, _ = synthetic()
    areas = random_areas(y, t)
    assert abs(areas.mean()) < 3 * areas.std(ddof=1) / np.sqrt(len(areas))


def test_tie_rule_same_seed_same_area_and_constant_score_near_zero():
    y, t, _ = synthetic()
    flat = np.zeros(len(y))
    assert qini.qini_area(y, t, flat, seed=7) == qini.qini_area(y, t, flat, seed=7)
    assert qini.qini_area(y, t, flat, seed=7) != qini.qini_area(y, t, flat, seed=8)
    areas = np.array([qini.qini_area(y, t, flat, seed=i) for i in range(1000)])
    assert abs(areas.mean()) < 3 * areas.std(ddof=1) / np.sqrt(len(areas))


def test_ties_are_broken_only_within_equal_scores():
    score = np.array([1.0, 2.0, 2.0, 2.0, 0.5])
    for seed in range(20):
        order = qini.rank_order(score, seed)
        assert set(order[:3]) == {1, 2, 3}
        assert list(order[3:]) == [0, 4]


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="at least 100"):
        qini.qini_area(np.zeros(50), np.zeros(50), np.zeros(50))
    with pytest.raises(ValueError, match="0 or 1"):
        qini.qini_area(np.zeros(200), np.full(200, 2.0), np.zeros(200))


def test_bootstrap_keeps_the_ranking_fixed(monkeypatch):
    y, t, tau = synthetic(n=500)
    calls = []
    real = qini.rank_order

    def counting(score, seed=0):
        calls.append(seed)
        return real(score, seed)

    monkeypatch.setattr(qini, "rank_order", counting)
    out = qini.bootstrap_area(y, t, tau, b=50, seed=3)
    assert len(calls) == 1  # one ranking for all 50 replicates
    order = real(tau, 0)
    cuts = qini.cut_positions(len(y))
    for w, rep in zip(qini.replicate_weights(len(y), 50, 3), out.replicates, strict=True):
        q = qini._curve_sorted(y[order], t[order], w[order], cuts)
        assert rep == pytest.approx(qini._area_from_curve(q), abs=1e-12)
    assert out.estimate == pytest.approx(qini.qini_area(y, t, tau))


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    y, t, tau = synthetic(n=500)
    a = qini.bootstrap_area(y, t, tau, b=200, seed=0)
    b = qini.bootstrap_area(y, t, tau, b=200, seed=0)
    assert np.array_equal(a.replicates, b.replicates)
    assert a.low < a.estimate < a.high


def test_paired_difference_of_a_ranking_with_itself_is_zero():
    y, t, tau = synthetic(n=500)
    d = qini.bootstrap_area_difference(y, t, tau, tau, b=50, seed=0)
    assert d.estimate == 0 and np.all(d.replicates == 0)


def test_naive_difference_hand_case():
    # treated 3 of 4 respond, control 1 of 4: 0.75 - 0.25 = 0.5
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0], float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0], float)
    out = qini.bootstrap_naive_difference(np.tile(y, 25), np.tile(t, 25), b=100, seed=0)
    assert out.estimate == pytest.approx(0.5)
    assert out.low < 0.5 < out.high


def test_cross_check_against_sklift_curve_points():
    sklift_metrics = pytest.importorskip("sklift.metrics")
    y, t, _ = synthetic(n=1000)
    score = np.random.default_rng(5).random(len(y))  # distinct scores, no ties
    _, q = qini.qini_curve(y, t, score)
    x_ref, y_ref = sklift_metrics.qini_curve(y, score, t)
    ref = dict(zip(x_ref, y_ref, strict=True))
    for k, value in zip(qini.cut_positions(len(y)), q, strict=True):
        assert value * len(y) == pytest.approx(ref[k])
