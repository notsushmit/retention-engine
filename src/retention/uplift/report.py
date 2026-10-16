"""KKBox report tables and sentences (configs/uplift.yaml, kkbox.outcome_read and outcome_step).

Every row and every sentence carries the label "balance not met"; every table gives the
95% interval before the point estimate; only averages are reported (no row can carry a
customer or unit field), so no per-customer effect, customer group or counterfactual is
built from KKBox. The reading is fixed in advance: "balance not met, assumptions
unsupported".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from retention.uplift.kkbox_bootstrap import Result

LABEL = "balance not met"
READING = "balance not met, assumptions unsupported"
INTERVAL_SCOPE = (
    "The interval covers sampling variation given the trimmed set, "
    "not the uncertainty of the trimming itself."
)
COLUMNS = [
    "label",
    "set",
    "statistic",
    "ci_low",
    "ci_high",
    "estimate",
    "interval_status",
    "resamples",
]


def rows(result: Result, set_name: str) -> list[dict[str, Any]]:
    out = []
    for stat, est in result.estimates.items():
        lo, hi = result.intervals[stat] if result.intervals else (None, None)
        status = "computed" if result.status == "computed" else f"not computable: {result.reason}"
        out.append(
            {
                "label": LABEL,
                "set": set_name,
                "statistic": stat,
                "ci_low": lo,
                "ci_high": hi,
                "estimate": est,
                "interval_status": status,
                "resamples": result.resamples_done,
            }
        )
    return out


def table(all_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows in the registered column order; any other field (such as a unit id) is refused."""
    for r in all_rows:
        extra = set(r) - set(COLUMNS)
        if extra:
            raise ValueError(f"report rows carry only aggregates; refused fields {sorted(extra)}")
        if r["label"] != LABEL:
            raise ValueError("every row carries the label")
    return pd.DataFrame(all_rows, columns=COLUMNS)


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def sentence(row: dict[str, Any]) -> str:
    head = f"{READING[0].upper()}{READING[1:]}: {row['statistic']} on {row['set']}"
    if row["interval_status"] == "computed":
        body = (
            f", 95% interval {_num(row['ci_low'])} to {_num(row['ci_high'])}, "
            f"point estimate {_num(row['estimate'])}"
        )
    else:
        body = f", interval {row['interval_status']}, point estimate {_num(row['estimate'])}"
    return f"{head}{body}. This is not a finding. {INTERVAL_SCOPE}"
