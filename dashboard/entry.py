"""Run one page inside the shared frame: the client from session state, capture mode from
the URL (C2: ?capture=1)."""

from __future__ import annotations

import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.views import VIEWS

CLIENT = "client"


def client() -> Client:
    if CLIENT not in st.session_state:
        st.session_state[CLIENT] = Client()
    return st.session_state[CLIENT]


def capture() -> bool:
    return st.query_params.get("capture") == "1"


def run(key: str) -> None:
    cp.frame(client(), key, VIEWS[key].render, capture())
