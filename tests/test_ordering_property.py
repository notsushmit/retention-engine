"""The ordering fix as the service applies it: p7 <= p30 <= p59 on every row."""

import numpy as np
import pytest

from retention.api.ordering import final_probabilities, order_violations, platt_from
from retention.models.metrics import CLIP

SEED = 20161001


def random_calibrators(rng: np.random.Generator) -> list:
    return [platt_from(rng.normal(0, 1), rng.uniform(0.3, 2.0), SEED) for _ in range(3)]


@pytest.mark.parametrize("k", range(50))
def test_final_probabilities_are_ordered_on_seeded_random_inputs(k):
    rng = np.random.default_rng(SEED + k)
    raw = rng.uniform(0, 1, size=(2_000, 3))
    raw[rng.random(raw.shape) < 0.02] = 0.0  # clipped at the low end
    raw[rng.random(raw.shape) < 0.02] = 1.0  # and at the high end
    final = final_probabilities(raw, random_calibrators(rng))["final"]
    assert order_violations(final) == 0
    assert np.all(np.diff(final, axis=1) >= 0)
    assert final.min() >= CLIP and final.max() <= 1 - CLIP


def test_a_violating_pair_becomes_its_average():
    identity = [platt_from(0.0, 1.0, SEED) for _ in range(3)]
    raw = np.array([[0.20, 0.10, 0.40], [0.10, 0.20, 0.30], [0.50, 0.30, 0.10]])
    res = final_probabilities(raw, identity)
    np.testing.assert_allclose(res["final"][0], [0.15, 0.15, 0.40], atol=1e-12)
    np.testing.assert_allclose(res["final"][1], [0.10, 0.20, 0.30], atol=1e-12)
    np.testing.assert_allclose(res["final"][2], [0.30, 0.30, 0.30], atol=1e-12)
    assert res["rows_changed"] == 2
    assert order_violations(res["final"]) == 0


def test_ordered_rows_are_left_alone():
    identity = [platt_from(0.0, 1.0, SEED) for _ in range(3)]
    rng = np.random.default_rng(SEED)
    raw = np.sort(rng.uniform(0.01, 0.99, size=(500, 3)), axis=1)
    res = final_probabilities(raw, identity)
    assert res["rows_changed"] == 0
    np.testing.assert_allclose(res["final"], raw, rtol=0, atol=1e-12)


def test_order_violations_counts_rows_not_pairs():
    final = np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3], [0.1, 0.3, 0.2]])
    assert order_violations(final) == 2


def test_wrong_shape_or_calibrator_count_is_refused():
    cals = [platt_from(0.0, 1.0, SEED) for _ in range(3)]
    with pytest.raises(ValueError):
        final_probabilities(np.zeros((4, 2)), cals)
    with pytest.raises(ValueError):
        final_probabilities(np.zeros((4, 3)), cals[:2])
