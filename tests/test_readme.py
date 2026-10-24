"""The README: its order, the screenshots, the diagram, the stack, the numbers and the wording.

The README is public text inside the wording scope. Every number outside the Scoring
service section (checked by tests/test_serving_docs.py) comes from configs/, reports/,
docs/ or LISTED. Every node of the architecture diagram names a component that exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from test_dashboard_wording import check

README = Path("README.md").read_text()
LINES = README.splitlines()
SCORING = README[README.index("## Scoring service") : README.index("## Limitations")]
REST = README.replace(SCORING, "")

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
HEXISH = re.compile(r"#?\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{4,}\b")
LISTED = {
    "60613": "8,890 + 31 + 51,692, the three registered outside-population counts",
    "26": "60,613 / 230,286 = 26.3%, rounded",
    "1.1": "the SIL Open Font License version, read from each OFL.txt in dashboard/static/fonts",
    "2018": "the competition's name, WSDM Cup 2018 (configs/kkbox.yaml names the competition)",
}
# Every node of the diagram, and the file or folder it stands for.
COMPONENTS = {
    "KKBox files (Kaggle)": "configs/kkbox.yaml",
    "Data: download, sample, clean": "src/retention/data",
    "Churn labels": "src/retention/labels",
    "Feature table": "src/retention/features",
    "Models: LightGBM classifier, discrete-time survival, Platt calibration": (
        "src/retention/models"
    ),
    "Trend alerts": "src/retention/trends",
    "Treatment-effect validation": "src/retention/uplift",
    "Reasons": "src/retention/explain",
    "Budget optimizer": "src/retention/optimize",
    "MLflow registry": "src/retention/api/registry.py",
    "Replay stream": "src/retention/stream",
    "Valkey store": "src/retention/api/store.py",
    "Scoring API (FastAPI)": "src/retention/api/app.py",
    "Dashboard (Streamlit)": "dashboard/app.py",
    "Docker compose stack": "docker-compose.yml",
}
# Every tool in the tech stack, and the names it must appear under in uv.lock or the
# container files.
TOOLS = {
    "duckdb": "duckdb",
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "kaggle": "kaggle",
    "py7zr": "py7zr",
    "lightgbm": "lightgbm",
    "scikit-learn": "scikit-learn",
    "lifelines": "lifelines",
    "xgboost": "xgboost",
    "scipy": "scipy",
    "ruptures": "ruptures",
    "mlflow": "mlflow",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
    "redis": "redis",
    "streamlit": "streamlit",
    "altair": "altair",
    "httpx": "httpx",
    "pytest": "pytest",
    "ruff": "ruff",
    "pre-commit": "pre-commit",
}


def section(title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", README, flags=re.S | re.M)
    assert m, title
    return m.group(1)


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(HEXISH.sub(" ", text))}


SOURCES = set().union(
    *(numbers(p.read_text()) for p in Path("configs").glob("*.yaml")),
    *(numbers(p.read_text()) for p in Path("reports").glob("*.md")),
    *(numbers(p.read_text()) for p in Path("docs").glob("*.md")),
)


# --- order and the screenshots ---------------------------------------------------------------


def brief() -> tuple[int, str]:
    """The brief: the original opening restored at the user's request, with two phrases
    changed for the wording rules, then one sentence of honest framing."""
    i = next(k for k, line in enumerate(LINES) if line.startswith("Predicts which subscribers"))
    j = next(k for k, line in enumerate(LINES) if line.startswith("Everything runs on recorded"))
    return j, "\n".join(LINES[i : j + 1])


def test_the_title_tagline_and_badge_come_first():
    assert LINES[0] == "# Retention decision engine"
    assert "actions/workflows/ci.yml/badge.svg" in "\n".join(LINES[:4])
    tagline = next(line for line in LINES[1:8] if line.startswith("A churn"))
    assert "replay of 2016 KKBox" in tagline and "not a hosted service" in tagline


def test_the_brief_keeps_the_original_opening_and_its_honest_framing():
    _, text = brief()
    assert "Predicts which subscribers will churn and when" in text
    assert "The primary data is the KKBox churn prediction challenge (WSDM Cup 2018)." in text
    assert "Free trials and free promotional periods alone do not count." in text
    for needed in ("replayed", "assumed effects", "discount"):
        assert needed in text
    assert "explains why" not in text and "works" not in text


def test_the_first_image_comes_after_the_brief_and_within_25_lines():
    i, _ = brief()
    first = next(k for k, line in enumerate(LINES) if "<img " in line)
    assert i < first < 25


def test_the_three_screenshots_exist_with_alt_text_and_captions_that_carry_their_limits():
    srcs = re.findall(r'<img src="([^"]+)" alt="([^"]+)"', README)
    assert [s for s, _ in srcs] == [
        "docs/img/dashboard/overview.png",
        "docs/img/dashboard/planner.png",
        "docs/img/dashboard/model_health.png",
    ]
    assert all(Path(s).is_file() and alt for s, alt in srcs)
    head = "\n".join(LINES[:40])
    assert "Replayed 2016 data, one store date (2016-11-01)" in head
    assert "every value rests on assumed effects" in head
    assert "one month of 2016" in head
    assert "(docs/dashboard.md)" in head


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")


def test_the_contents_link_to_every_later_section():
    links = re.findall(r"^- \[([^\]]+)\]\(#([^)]+)\)$", section("Contents"), flags=re.M)
    headings = re.findall(r"^## (.+)$", README, flags=re.M)
    assert [t for t, _ in links] == [h for h in headings if h != "Contents"]
    assert all(anchor == slug(t) for t, anchor in links)


def test_the_quick_start_has_at_most_five_commands():
    block = re.search(r"```bash\n(.*?)```", section("Quick start"), flags=re.S).group(1)
    commands = [line for line in block.splitlines() if line.strip()]
    assert len(commands) <= 5
    assert commands[0].startswith("uv sync") and commands[-1] == "make test"
    assert "--reload-reason" in section("Quick start")


# --- the diagram and the stack ----------------------------------------------------------------


def mermaid() -> str:
    return re.search(r"```mermaid\n(.*?)```", section("Architecture"), flags=re.S).group(1)


def test_the_diagram_is_a_simple_left_to_right_flowchart():
    src = mermaid()
    assert src.splitlines()[0] == "flowchart LR"
    assert 'subgraph compose["Docker compose stack"]' in src


def test_every_node_names_a_real_component():
    labels = re.findall(r'\w+\["([^"]+)"\]', mermaid())
    assert labels and set(labels) <= set(COMPONENTS), set(labels) - set(COMPONENTS)
    assert all(Path(COMPONENTS[label]).exists() for label in labels)


def test_the_diagram_marks_the_compose_stack_with_the_api_and_valkey():
    src = mermaid()
    inside = re.search(r"subgraph .*?\n(.*?)\n\s*end\n", src, flags=re.S).group(1)
    assert "Valkey store" in inside and "Scoring API (FastAPI)" in inside
    compose = Path("docker-compose.yml").read_text()
    assert "valkey:" in compose and "api:" in compose


def test_every_tool_in_the_stack_is_in_the_lock_or_the_container_files():
    lock = Path("uv.lock").read_text()
    for name in TOOLS.values():
        assert f'name = "{name}"' in lock, name
    stack = section("Tech stack").lower()
    assert all(tool in stack for tool in TOOLS)
    assert "valkey/valkey" in Path("docker-compose.yml").read_text()
    assert "python:3.11" in Path("Dockerfile").read_text()


def test_the_build_table_has_one_line_per_phase():
    rows = re.findall(r"^\| (\d+) \|", section("How it was built"), flags=re.M)
    assert rows == [str(i) for i in range(10)]
    for link in re.findall(r"\]\(((?:docs|reports)/[^)]+)\)", section("How it was built")):
        assert Path(link).is_file(), link


# --- numbers and wording ------------------------------------------------------------------------


def test_every_number_outside_the_scoring_section_has_a_source():
    assert numbers(REST) - SOURCES - set(LISTED) == set()


def test_every_listed_figure_is_used_and_right():
    assert set(LISTED) <= numbers(REST)
    assert 8_890 + 31 + 51_692 == 60_613 and round(100 * 60_613 / 230_286) == 26


def test_a_number_with_no_source_is_caught():
    assert numbers("about 41,999 customers") - SOURCES - set(LISTED) == {"41999"}


def test_the_readme_passes_every_wording_rule():
    assert check(README) == []


def test_the_readme_claims_no_deployment():
    assert not re.search(r"\b(deployed|hosted at|is hosted|live demo)\b", README, re.I)


@pytest.mark.parametrize(
    "line",
    [
        "Plans exist for two dates only",
        "Every 59-day probability is a demonstration",
        "Drift is not monitored",
        "There is no hosted demo",
    ],
)
def test_the_limitations_are_stated(line):
    assert line in section("Limitations")


def test_the_readme_stays_short():
    assert len(LINES) < 400


def test_the_setup_doc_passes_every_wording_rule():
    assert check(Path("docs/setup.md").read_text()) == []
