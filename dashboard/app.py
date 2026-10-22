"""The dashboard: five pages, grouped in the sidebar, each reading only the scoring API.

Run with `make dashboard` (127.0.0.1 only). configs/dashboard.yaml, C1 and C2.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit puts only this folder on the path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.entry import run  # noqa: E402
from dashboard.settings import load  # noqa: E402

st.set_page_config(page_title="Retention dashboard", layout="wide")


def page_for(key: str):
    def show() -> None:
        run(key)

    show.__name__ = f"page_{key}"
    return show


groups: dict[str, list] = {}
for p in load().pages:
    groups.setdefault(p.group, []).append(
        st.Page(page_for(p.key), title=p.title, url_path=p.key.replace("_", "-"))
    )
st.navigation(groups).run()
