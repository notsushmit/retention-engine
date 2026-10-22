"""Model health: GET /health and GET /model: the model, its files, the calibration table
before and after the ordering fix; drift is not monitored in this phase."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.settings import load
from dashboard.tooltips import tip

S = load()


def render(client: Client, health: dict[str, Any], capture: bool) -> None:
    m = client.model()
    if m["model_checksums_match"]:
        st.header(
            S.page("model_health")["takeaway"].replace("<version>", str(m["version"])),
            help=tip("model_files"),
        )
    else:
        st.header("The model files differ from their registered hashes", help=tip("model_files"))
    cal = m["calibration"]
    thirty = next(h for h in cal["horizons"] if h["horizon"] == 30)
    cp.big(
        f"{thirty['slope_after_fix']:.4f}",
        "30-day calibration slope after the fix",
        tip("calibration"),
    )
    table = pd.DataFrame(cal["horizons"])
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(help=tip("calibration"), format="%.4f")
            for c in table.columns
            if c.startswith(("slope", "intercept"))
        },
    )
    st.caption(
        f"{cal['label']}. {cp.count(cal['rows'])} rows, {cp.count(cal['rows_changed'])} moved by "
        f"the ordering fix. {cal['no_59']}.",
        help=tip("calibration"),
    )
    cp.number("Version", str(m["version"]), tip("version"))
    files = pd.DataFrame([{"file": k, "SHA-256": v} for k, v in m["files"].items()])
    st.dataframe(files, hide_index=True)
    st.caption(
        f"{m['registered_model']}; config SHA-256 {m['config_sha256']}.", help=tip("model_files")
    )
    st.caption(
        f"{m['drift']}. Retraining: {m['retraining']}. Every 59-day probability: {m['grade_59']}."
    )
