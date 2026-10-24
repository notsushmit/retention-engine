"""docs/img holds only the registered images, so a stray screenshot cannot be committed:
the four earlier charts at the top level and, under dashboard/, only the three captures."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

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
    extra = {f for f in files() if not f.startswith("dashboard/")} - TOP
    assert extra == set()


PAGES = ("overview", "planner", "model_health")
NAME = re.compile(r"^(" + "|".join(PAGES) + r")(_\d+)?\.png$")
MAGIC = {".png": b"\x89PNG\r\n\x1a\n", ".svg": b"<svg"}


def test_the_dashboard_folder_holds_only_page_pngs():
    found = {f.removeprefix("dashboard/") for f in files() if f.startswith("dashboard/")}
    assert {f for f in found if not NAME.match(f)} == set()
    assert CAPTURES <= found


@pytest.mark.parametrize("name", sorted(CAPTURES))
def test_each_capture_is_really_a_png(name):
    path = IMG / "dashboard" / name
    assert path.read_bytes()[:8] == MAGIC[".png"]


def test_every_image_file_matches_its_extension():
    for f in files():
        path = IMG / f
        magic = MAGIC.get(path.suffix)
        assert magic is not None, f
        head = path.read_bytes()[:256]
        assert (magic in head) if path.suffix == ".svg" else head.startswith(magic), f
