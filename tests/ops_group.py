"""Tests that need the ops group (evidently, prefect): skip loudly where it is not installed.

The main CI job and `make test` do not install the ops group, so a test marked with
`needs_ops_group` is reported as skipped with "not checked: ops group absent" (shown by -rs).
The ops workflow sets OPS_TESTS_MUST_RUN=1: there the tests are never skipped, so a missing
group fails them instead of hiding them (configs/ops.yaml, amendment A13, ci_install).
"""

from __future__ import annotations

import os
from importlib.util import find_spec

import pytest

SKIP_REASON = "not checked: ops group absent"
MUST_RUN_ENV = "OPS_TESTS_MUST_RUN"


def ops_group_present() -> bool:
    """True when evidently and prefect can be imported (find_spec imports neither)."""
    return find_spec("evidently") is not None and find_spec("prefect") is not None


def skip_marker(present: bool, must_run: bool):
    """The skip mark for a test that needs the ops group, or None when it must run."""
    if present or must_run:
        return None
    return pytest.mark.skip(reason=SKIP_REASON)


def must_run() -> bool:
    return os.environ.get(MUST_RUN_ENV) == "1"


def needs_ops_group(test):
    """Decorator: skip loudly without the ops group; never skip when OPS_TESTS_MUST_RUN=1."""
    mark = skip_marker(ops_group_present(), must_run())
    return mark(test) if mark is not None else test
