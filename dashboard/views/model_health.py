"""Model health: GET /health and GET /model: the model, its files, the calibration table
before and after the ordering fix, and (C8) the drift panel from GET /drift and the last
retraining comparison from GET /model."""

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
    decimals = {"horizon": 0, "events": 0}
    decimals |= {c: 4 for c in table.columns if c.startswith(("slope", "intercept"))}
    st.dataframe(
        cp.as_text(table, decimals),
        hide_index=True,
        column_config={
            c: st.column_config.TextColumn(help=tip("calibration"))
            for c in table.columns
            if c.startswith(("slope", "intercept"))
        },
    )
    st.caption(
        f"Calibration: {cal['label']}. {cp.count(cal['rows'])} rows, "
        f"{cp.count(cal['rows_changed'])} moved by "
        f"the ordering fix. Note: {cal['no_59']}.",
        help=tip("calibration"),
    )
    cp.number("Version", str(m["version"]), tip("version"))
    files = pd.DataFrame([{"file": k, "SHA-256": v} for k, v in m["files"].items()])
    st.dataframe(files, hide_index=True)
    st.caption(
        f"{m['registered_model']}; config SHA-256 {m['config_sha256']}.", help=tip("model_files")
    )
    ops_panels(client, m)
    st.caption(
        f"{m['drift']}. Retraining: {m['retraining']}. Every 59-day probability is a demonstration."
    )


def ops_panels(client: Client, m: dict[str, Any]) -> None:
    """The drift panel and the last retraining comparison (configs/dashboard.yaml, C8)."""
    if m["drift"] != S.serving["dashboard_endpoints"]["model"]["drift"]:
        drift_panel(client.drift())
    if m.get("last_retrain"):
        retrain_panel(m["last_retrain"])


def drift_panel(d: dict[str, Any]) -> None:
    if d["status"] != "monitored":
        st.caption(d["label"])
        return
    cp.line("Drift", help=tip("drift_columns"))
    n = d["columns_compared"]["count"]
    cp.number(
        "Drifted columns",
        f"{len(d['columns_drifted'])} of {cp.count(n)} ({cp.share(d['drift_share'])})",
        tip("drift_share"),
    )
    cp.number("Dataset alert", "yes" if d["dataset_alert"] else "no", tip("drift_share"))
    mg = d["margin"]
    cp.number(
        "Share over the unshifted store",
        f"{mg['difference']:.2f} (required {mg['required']:.2f})",
        tip("drift_margin"),
    )
    cp.number("Margin and alert rules met", "yes" if d["margin_met"] else "no", tip("drift_margin"))
    if d["columns_drifted"]:
        table = pd.DataFrame(d["columns_drifted"])
        st.dataframe(
            cp.as_text(table, {"threshold": 2, "statistic": 3}),
            hide_index=True,
            column_config={
                c: st.column_config.TextColumn(help=tip("drift_columns"))
                for c in ("statistic", "threshold")
            },
        )
    out = d["batch_size"]["days_outside_band"]
    cp.number("Days outside the event-count band", cp.count(len(out)), tip("drift_batch"))
    st.caption(d["label"])


def retrain_panel(r: dict[str, Any]) -> None:
    cp.line("Last retraining comparison", help=tip("retrain_auc"))
    cp.number("Decision", r["decision"], tip("retrain_auc"))
    cp.number("30-day AUC, candidate", f"{r['candidate_auc_30']:.4f}", tip("retrain_auc"))
    cp.number("30-day AUC, current model", f"{r['current_auc_30']:.4f}", tip("retrain_auc"))
    cp.number(
        "Difference and its interval",
        f"{r['difference']:.4f} ({r['interval_low']:.4f} to {r['interval_high']:.4f})",
        tip("retrain_auc"),
    )
    st.caption(r["label"])
