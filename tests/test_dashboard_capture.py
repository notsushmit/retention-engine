"""The capture script: the Overview only with capture=1, never the pages with customer IDs."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("capture_pages", "scripts/capture_pages.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


@pytest.mark.parametrize("key", ["at_risk_list", "customer_profile", "nowhere"])
def test_pages_with_customer_ids_are_refused(key):
    with pytest.raises(capture.CaptureRefused):
        capture.url(key)


def test_the_overview_is_allowed_only_in_capture_mode():
    assert capture.url("overview") == "http://127.0.0.1:8501/?capture=1"


@pytest.mark.parametrize(
    "key,path", [("action_planner", "/action-planner"), ("model_health", "/model-health")]
)
def test_the_two_aggregate_pages_are_allowed(key, path):
    assert capture.url(key) == f"http://127.0.0.1:8501{path}"


def test_main_exits_2_when_any_requested_page_is_refused(capsys):
    assert capture.main(["overview", "customer_profile"]) == 2
    out = capsys.readouterr()
    assert "capture=1" in out.out and "refused" in out.err


def test_the_default_run_lists_only_allowed_pages(capsys):
    assert capture.main([]) == 0
    assert len(capsys.readouterr().out.split()) == 3


def test_make_dashboard_serves_on_loopback_only():
    text = Path("Makefile").read_text()
    rule = re.search(r"^dashboard:\n\t(.+)$", text, re.M).group(1)
    assert "--server.address 127.0.0.1" in rule and "dashboard/app.py" in rule
