"""Run one page inside the shared frame: one API client for the whole app (cached with
st.cache_resource, so browser sessions do not each leave connections open), capture mode
from the URL (C2: ?capture=1). A client placed in session state (the tests' fake API)
takes precedence."""

from __future__ import annotations

import streamlit as st

from dashboard import components as cp
from dashboard.client import Client
from dashboard.views import VIEWS

CLIENT = "client"


@st.cache_resource
def shared_client() -> Client:
    """The one client every session uses (found after a run, D-258: one client per session
    left connections to the API in CLOSE_WAIT)."""
    return Client()


def client() -> Client:
    if CLIENT in st.session_state:
        return st.session_state[CLIENT]
    return shared_client()


def capture() -> bool:
    return st.query_params.get("capture") == "1"


def run(key: str) -> None:
    cp.frame(client(), key, VIEWS[key].render, capture())
