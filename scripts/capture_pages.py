"""Print the URLs of the dashboard pages that may be captured for the public repo or video.

configs/dashboard.yaml step3_conditions.capture_mode (C2): the Overview only with
capture=1 (no top-20 table, no cancel events), the Action planner and Model health
always; the At-risk list and the Customer profile are refused, because they show
customer IDs. Screenshots are taken by hand; this script takes none.

    uv run python scripts/capture_pages.py overview action_planner model_health
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.settings import load  # noqa: E402

BASE = "http://127.0.0.1:8501"


class CaptureRefused(ValueError):
    """A page that shows customer IDs."""


def rules() -> tuple[dict[str, str], list[str]]:
    script = load().c2["capture_mode"]["script"]
    return dict(script["allowed"]), list(script["refused"])


def url(key: str, base: str = BASE) -> str:
    allowed, refused = rules()
    if key in refused or key not in allowed:
        raise CaptureRefused(f"{key} shows customer IDs and is never captured")
    first = load().pages[0].key
    path = "/" if key == first else "/" + key.replace("_", "-")
    if allowed[key] == "capture=1 only":
        return f"{base}{path}?capture=1"
    return f"{base}{path}"


def main(argv: list[str]) -> int:
    keys = argv or list(rules()[0])
    status = 0
    for key in keys:
        try:
            print(url(key))
        except CaptureRefused as e:
            print(f"refused: {e}", file=sys.stderr)
            status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
