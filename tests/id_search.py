"""One way to look for customer IDs in a body or a report (found after a CI run, D-265).

A SHA-256 is 64 hex characters, and a short test ID such as "c022" can sit inside one by
chance. Runs of 32 or more hex characters are blanked out before the search; every other
character is still searched, so an ID anywhere else in the text is found. The one blind
spot: an ID that is itself part of a run of 32 or more hex characters is blanked with it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

HEX_RUN = re.compile(r"[0-9a-f]{32,}")  # a hash, not a name


def blank_hashes(text: str) -> str:
    return HEX_RUN.sub(" ", text)


def ids_in(text: str, ids: Iterable[str]) -> list[str]:
    """The IDs that occur in the text outside any hash, in the order given."""
    searched = blank_hashes(text)
    return [m for m in ids if m in searched]


def assert_no_ids(text: str, ids: Iterable[str], where: str = "") -> None:
    found = ids_in(text, ids)
    assert not found, f"{where}: customer IDs found: {found[:5]}"
