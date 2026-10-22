"""Every public dashboard string passes the wording rules (configs/actions.yaml wording_rules,
configs/serving.yaml wording, configs/dashboard.yaml wording), and every tooltip number is a
registered one (C2)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import dashboard_fake as f
from dashboard.settings import load
from dashboard.tooltips import catalog
from retention.api.docs import wording_violations
from test_dashboard_pages import PAGES, md, render

S = load()
BANNER = S.raw["wording"]["real_time_exception"]
ATTRIBUTION = S.raw["heatmap_rules"]["attribution_line"]
EXCEPTIONS = (BANNER, ATTRIBUTION)


def check(text: str) -> list[str]:
    """The dashboard's two exact-string exceptions, each at most once, then the rules."""
    out = []
    for e in EXCEPTIONS:
        if text.count(e) > 1:
            out.append(f"exception used twice: {e!r}")
        text = text.replace(e, " ", 1)
    return out + wording_violations(text)


def test_the_two_exceptions_are_the_registered_strings():
    assert BANNER == S.banner == "Replay of 2016 data, not real-time."
    assert ATTRIBUTION == S.c1["overview_bars"]["attribution_line"]


def test_the_rules_catch_a_planted_breach():
    assert check("This plan saves customers.") != []
    assert check(BANNER + " " + BANNER) != []
    assert check("The 59-day probability is 0.4.") != []
    assert check("a real-time view") != []


def page_text(at) -> str:
    parts = [m.value for m in md(at)] + [c.value for c in at.caption]
    parts += [h.value for h in at.header] + [w.value for w in at.warning]
    parts += [f"{m.label}" for m in at.metric] + [t.label for t in at.tabs]
    parts += [m.proto.help for m in at.metric]
    return "\n".join(re.sub(r"<[^>]+>", "\n", p) for p in parts)


@pytest.mark.parametrize("key", PAGES)
def test_every_rendered_page_passes_the_wording_rules(key):
    at = render(key)
    if key == "customer_profile":
        at.text_input[0].input(f.SCORED_ID).run()
    assert check(page_text(at)) == []


@pytest.mark.parametrize("msno", [f.OUT_ID, f.INACTIVE_ID, "nobody-here"])
def test_the_other_profile_answers_pass_the_wording_rules(msno):
    at = render("customer_profile")
    at.text_input[0].input(msno).run()
    assert check(page_text(at)) == []


def test_capture_mode_passes_the_wording_rules():
    assert check(page_text(render("overview", capture=True))) == []


def source_strings() -> list[str]:
    out = []
    for path in sorted(Path("dashboard").rglob("*.py")):
        tree = ast.parse(path.read_text())
        docs = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef)
            and n.body
            and isinstance(n.body[0], ast.Expr)
        }
        out += [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
        ]
    return out


def test_every_fixed_string_in_the_dashboard_code_passes_the_wording_rules():
    bad = {s: check(s) for s in source_strings() if check(s)}
    assert bad == {}


def test_no_60_days_anywhere_in_the_dashboard():
    for path in Path("dashboard").rglob("*.py"):
        assert not re.search(r"\b60 days\b", path.read_text(), re.I), path


# --- tooltips (C2) --------------------------------------------------------------------------------

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


REGISTERED = set().union(*(numbers(p.read_text()) for p in Path("configs").glob("*.yaml")))


@pytest.mark.parametrize("key", sorted(catalog()))
def test_every_tooltip_number_is_registered_and_the_tooltip_passes_the_rules(key):
    text = catalog()[key]
    assert numbers(text) - REGISTERED == set()
    assert check(text) == []


def test_every_tooltip_names_its_api_field():
    for key, text in catalog().items():
        assert re.search(r"/(summary|health|at-risk|score|plans|model|alerts)\b", text), key


def test_the_planner_caption_starts_with_the_registered_prefix():
    from dashboard.views.action_planner import HEADLINE_CAPTION

    prefix = S.serving["dashboard_endpoints"]["plans"]["labels"]["effect_prefix"]
    assert HEADLINE_CAPTION.startswith(prefix + ",")
