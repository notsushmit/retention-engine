import numpy as np
import pytest

from retention.models.calibration import Isotonic, Platt, calibrate, clip, order_fix
from retention.models.metrics import CLIP


@pytest.mark.parametrize(
    "row, expected",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),  # already ordered: unchanged
        ([0.3, 0.2, 0.5], [0.25, 0.25, 0.5]),  # one violating pair becomes its average
        ([0.1, 0.4, 0.2], [0.1, 0.3, 0.3]),
        ([0.5, 0.4, 0.3], [0.4, 0.4, 0.4]),  # all three pooled
        ([0.4, 0.1, 0.4], [0.25, 0.25, 0.4]),
    ],
)
def test_order_fix_hand_cases(row, expected):
    assert order_fix(np.array([row])).tolist()[0] == pytest.approx(expected)


def test_order_fix_is_least_squares_and_non_decreasing_on_random_rows():
    rng = np.random.default_rng(0)
    p = rng.random((500, 3))
    out = order_fix(p)
    assert (np.diff(out, axis=1) >= -1e-12).all()
    assert np.allclose(out.mean(axis=1), p.mean(axis=1))  # pooling keeps each row's mean


class Fixed:
    def __init__(self, values):
        self.values = values

    def predict(self, p):
        return np.full(len(p), self.values)


def test_calibrate_runs_calibrate_clip_fix_clip_in_order():
    raw = np.zeros((2, 3))
    # Calibrated values: row 0 violates (0.2 > 0.1) and the first column is above 1 - 1e-6.
    cals = [Fixed(np.array([0.2, 1.0])), Fixed(np.array([0.1, 0.5])), Fixed(np.array([0.4, 0.6]))]
    out = calibrate(raw, cals)
    assert out["before_fix"][0].tolist() == pytest.approx([0.2, 0.1, 0.4])
    assert out["final"][0].tolist() == pytest.approx([0.15, 0.15, 0.4])  # the fix moved two values
    assert out["before_fix"][1, 0] == 1 - CLIP  # clipped before the fix
    # (1, 0.5) pools to 0.75, still above 0.6, so all three pool to their mean, 0.7.
    assert out["final"][1].tolist() == pytest.approx([(1 - CLIP + 0.5 + 0.6) / 3] * 3)
    assert out["rows_changed"] == 2 and out["clipped"] == 1
    assert (np.diff(out["final"], axis=1) >= 0).all()


def test_clip_counts():
    assert clip(np.array([0.0, 0.5, 1.0]))[1] == 2


def test_platt_recovers_a_known_slope_and_intercept():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 200_000)
    z = np.log(p / (1 - p))
    y = (rng.random(len(p)) < 1 / (1 + np.exp(-(0.4 + 1.5 * z)))).astype(int)
    m = Platt(seed=0).fit(p, y)
    assert m.a == pytest.approx(0.4, abs=0.03) and m.b == pytest.approx(1.5, abs=0.03)


def test_isotonic_refuses_a_slice_below_the_event_floor():
    p, y = np.linspace(0, 1, 5000), np.zeros(5000, int)
    y[:976] = 1
    with pytest.raises(ValueError, match="at least 1000 events"):
        Isotonic(min_events=1000).fit(p, y)
    y[:1000] = 1
    out = Isotonic(min_events=1000).fit(p, y).predict(p)
    assert (np.diff(out) >= 0).all()
