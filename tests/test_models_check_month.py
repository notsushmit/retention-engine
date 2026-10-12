import math

import pandas as pd
import pytest

from retention.models.check_month import tenure_band, wilson


def test_wilson_interval_hand_case():
    p, lo, hi = wilson(10, 100)
    assert p == 0.1
    assert lo == pytest.approx(0.0552, abs=1e-4) and hi == pytest.approx(0.1744, abs=1e-4)
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_tenure_bands_and_missing():
    days = pd.Series([0, 89, 90, 364, 365, 729, 730, 1825, 1826, None], dtype="Float64")
    assert tenure_band(days).tolist() == [
        "< 90 days",
        "< 90 days",
        "90-364 days",
        "90-364 days",
        "1-2 years",
        "1-2 years",
        "2-5 years",
        "2-5 years",
        "5+ years",
        "missing",
    ]
