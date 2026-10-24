"""The ops group is installed only where it is wanted, and a test that needs it never hides."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

from ops_group import MUST_RUN_ENV, SKIP_REASON, needs_ops_group, skip_marker

MAIN_CI = Path(".github/workflows/ci.yml")


def test_the_skip_reason_is_the_registered_loud_message():
    assert SKIP_REASON == "not checked: ops group absent"


@pytest.mark.parametrize(
    ("present", "must_run", "skipped"),
    [(False, False, True), (True, False, False), (False, True, False), (True, True, False)],
)
def test_skip_marker(present, must_run, skipped):
    mark = skip_marker(present, must_run)
    assert (mark is not None) is skipped
    if mark is not None:
        assert mark.mark.kwargs["reason"] == SKIP_REASON


def test_a_decorated_test_is_skipped_loudly_without_the_group(monkeypatch):
    monkeypatch.delenv(MUST_RUN_ENV, raising=False)
    monkeypatch.setattr("ops_group.find_spec", lambda name: None)

    @needs_ops_group
    def marked():
        raise AssertionError("must not run")

    marks = getattr(marked, "pytestmark", [])
    assert [m.kwargs["reason"] for m in marks] == [SKIP_REASON]


def test_a_decorated_test_is_not_skipped_when_it_must_run(monkeypatch):
    monkeypatch.setenv(MUST_RUN_ENV, "1")
    monkeypatch.setattr("ops_group.find_spec", lambda name: None)

    @needs_ops_group
    def marked():
        return None

    assert getattr(marked, "pytestmark", []) == []  # it runs, and fails at its import


def test_the_main_ci_job_does_not_install_the_ops_group():
    text = MAIN_CI.read_text()
    assert "--group ops" not in text and "ops" not in re.findall(r"--group (\w+)", text)
    assert "uv run pytest -rs" in text


def test_make_setup_does_not_install_the_ops_group():
    makefile = Path("Makefile").read_text()
    assert "--group ops" not in makefile


@needs_ops_group
def test_the_installed_ops_packages_are_the_locked_versions():
    """Reads package metadata only: importing either package is left to the tests that need it."""
    lock = Path("uv.lock").read_text()
    for name in ("evidently", "prefect"):
        locked = re.search(rf'name = "{name}"\nversion = "([^"]+)"', lock).group(1)
        assert metadata.version(name) == locked, name
