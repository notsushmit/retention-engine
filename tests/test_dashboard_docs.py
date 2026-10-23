"""The README "Dashboard" section and docs/dashboard.md (step 5 docs).

Both pass every wording rule (the dashboard's two exact-string exceptions, each at most
once). Every number comes from configs/ or from LISTED, where each figure names its
source. The README's image links exist, and the statements the user asked for are there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from test_dashboard_wording import check

README = Path("README.md").read_text()
SECTION = README[README.index("## Dashboard") :]
DOC = Path("docs/dashboard.md").read_text()
TEXTS = {"README Dashboard": SECTION, "docs/dashboard.md": DOC}

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
HEXISH = re.compile(r"#?\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{4,}\b")
LISTED = {
    "60613": "8,890 + 31 + 51,692, the three registered outside-population counts "
    "(configs/serving.yaml step4.more_pass_marks.population_counts)",
    "26": "60,613 / 230,286 = 26.3%, from the same registered counts, rounded",
    "1.1": "the SIL Open Font License version, read from each OFL.txt in dashboard/static/fonts",
    "1.65": "the streamlit version pinned in uv.lock (1.65.0)",
}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(HEXISH.sub(" ", text))}


CONFIG_NUMBERS = set().union(*(numbers(p.read_text()) for p in Path("configs").glob("*.yaml")))


@pytest.mark.parametrize("name", list(TEXTS))
def test_each_text_passes_every_wording_rule(name):
    assert check(TEXTS[name]) == []


@pytest.mark.parametrize("name", list(TEXTS))
def test_every_number_comes_from_a_config_or_the_listed_set(name):
    assert numbers(TEXTS[name]) - CONFIG_NUMBERS - set(LISTED) == set()


def test_every_listed_figure_is_still_used_and_right():
    used = numbers(SECTION) | numbers(DOC)
    assert set(LISTED) <= used
    assert 8_890 + 31 + 51_692 == 60_613 and round(100 * 60_613 / 230_286) == 26
    lock = Path("uv.lock").read_text()
    assert re.search(r'name = "streamlit"\nversion = "1\.65\.0"', lock)


def test_a_number_with_no_source_is_caught():
    assert numbers("about 41,999 customers") - CONFIG_NUMBERS - set(LISTED) == {"41999"}


def test_the_readme_image_links_exist_and_are_the_three_captures():
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", SECTION)
    assert links == [
        "docs/img/dashboard/overview.png",
        "docs/img/dashboard/planner.png",
        "docs/img/dashboard/model_health.png",
    ]
    assert all(Path(p).is_file() for p in links)


@pytest.mark.parametrize(
    "statement",
    [
        "reads only the scoring API",
        "Replay of 2016 data, not real-time.",
        "make dashboard",
        '"Expiring more than 30 days ahead" shows 22%: 51,692 of the 230,286 customers',
        "60,613 of the same 230,286, about 26%",
        "SIL Open Font License 1.1",
        "not access control",
        "must never be hosted with customer rows",
    ],
)
def test_the_readme_states(statement):
    assert statement in SECTION


@pytest.mark.parametrize(
    "statement",
    [
        "a display rule, not a disclosure control",
        "C1, found after a result",
        "Found after a run",
        "8fb8960",
        "measured before the 1.5 GiB limit and before the dashboard endpoints",
        "Capture mode is a convenience for screenshots, not access control",
        "Reasons are model attributions, not causes.",
    ],
)
def test_the_doc_states(statement):
    assert statement in DOC


def test_the_readme_latency_sentence_names_both_limits():
    assert "measured before the 1.5 GiB limit and before the dashboard endpoints" in README


def test_every_token_named_in_the_readme_is_registered():
    from dashboard.settings import load

    tokens = {v.upper() for v in load().tokens.values()}
    named = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", SECTION)}
    assert named == tokens
