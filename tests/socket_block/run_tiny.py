"""A tiny Evidently report and a tiny Prefect flow, run with telemetry off (A13).

Usage: python run_tiny.py <folder for the drift defaults>. Prints one JSON line.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from retention.ops.telemetry import telemetry_off

telemetry_off()  # before the first import of either package

from retention.ops import drift as dr  # noqa: E402

rng = np.random.default_rng(1)
cols = dr.compared_columns()


def frame(shift: float) -> pd.DataFrame:
    data = {c: rng.normal(shift, 1, 60) for c in cols}
    for c in dr.CATEGORICAL:
        data[c] = rng.integers(0, 3, 60)
    return pd.DataFrame(data)


folder = Path(sys.argv[1])
dr.log_defaults(folder)
report = dr.run_report(frame(0), frame(2), folder)

from prefect import flow, task  # noqa: E402


@task
def add(a: int, b: int) -> int:
    return a + b


@flow
def tiny() -> int:
    return add(1, 2)


print(json.dumps({"evidently_drifted": report["columns_drifted"], "prefect": tiny()}))
