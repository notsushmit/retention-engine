import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from retention.models.baselines import NAMES, Baselines
from retention.models.metrics import auc
from retention.models.protocol import load_protocol

P = load_protocol()


def rows(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 30, n)
    auto = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(-3 + 2.5 * (days <= 7) - 1.5 * auto)))
    return pd.DataFrame(
        {"days_until_expiry": days, "last_is_auto_renew": auto, "y": rng.random(n) < p}
    ).astype({"y": int})


def test_every_baseline_scores_every_row_and_the_base_rate_is_constant():
    fit, score = rows(seed=0), rows(seed=1)
    b = Baselines(fit, [7, 14, 21, 29], P.threads, P.seed)
    for name in NAMES:
        s = b.score(name, score)
        assert s.shape == (len(score),) and np.isfinite(s).all()
    assert np.unique(b.score("base_rate", score)).tolist() == [pytest.approx(fit.y.mean())]
    assert auc(score.y, b.score("base_rate", score)) == 0.5


def test_the_empty_30_plus_band_is_dropped_and_the_banded_model_learns_the_step():
    fit, score = rows(seed=0), rows(seed=1)
    b = Baselines(fit, [7, 14, 21, 29], P.threads, P.seed)
    assert b.dropped_bands == 1  # no row has 30 or more days to expiry
    assert auc(score.y, b.score("banded_logistic", score)) > auc(
        score.y, b.score("two_feature_logistic", score)
    )


def test_no_model_process_loads_the_label_building_code():
    code = (
        "import sys; import retention.models.stages, retention.models.frames, "
        "retention.models.baselines, retention.models.__main__; "
        "print('retention.labels.snapshots' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_label_building_command_path_loads_no_model_library():
    code = (
        "import sys; import retention.models.__main__, retention.models.data; "
        "import retention.labels.snapshots; "
        "print([m for m in ('lightgbm', 'xgboost', 'sklearn', 'lifelines') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
