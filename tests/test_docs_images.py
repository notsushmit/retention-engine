"""docs/img holds only the registered images, so a stray screenshot cannot be committed:
the four earlier charts at the top level and, under dashboard/, only the three captures."""

from __future__ import annotations

import subprocess
from pathlib import Path

IMG = Path("docs/img")
TOP = {
    "label_timeline.svg",
    "qini_criteo_visit.svg",
    "qini_hillstrom_mens_vs_none_visit.svg",
    "qini_hillstrom_womens_vs_none_visit.svg",
}
CAPTURES = {"overview.png", "planner.png", "model_health.png"}


def files() -> set[str]:
    """Every file under docs/img that git would track (ignored files such as .DS_Store left out)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", str(IMG)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {str(Path(p).relative_to(IMG)) for p in out if Path(p).exists()}


def test_docs_img_holds_only_the_registered_images():
    allowed = TOP | {f"dashboard/{c}" for c in CAPTURES}
    assert files() - allowed == set()


def test_the_dashboard_folder_holds_only_the_three_captures():
    found = {f.removeprefix("dashboard/") for f in files() if f.startswith("dashboard/")}
    assert found <= CAPTURES
