"""The look: only registered tokens, the accent never as text, direct labels on every bar,
self-hosted fonts equal to their recorded hashes (configs/dashboard.yaml design_tokens, C2)."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from dashboard import components as cp
from dashboard.settings import load
from dashboard_fake import summary

S = load()
TOKENS = {v.upper() for v in S.tokens.values()}
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b")
CSS = Path("dashboard/static/style.css").read_text()
THEME = tomllib.loads(Path(".streamlit/config.toml").read_text())
FONTS = Path("dashboard/static/fonts")


def test_eight_tokens_are_registered():
    assert len(TOKENS) == 8 and S.tokens["accent"] == "#FE5A00"


def test_the_css_uses_only_registered_tokens_and_no_named_or_functional_colour():
    assert {h.upper() for h in HEX.findall(CSS)} <= TOKENS
    assert not re.search(r"\b(rgb|rgba|hsl|hsla)\(", CSS)
    assert not re.search(r":\s*(white|black|red|green|blue|orange|gray|grey)\b", CSS, re.I)
    assert "http" not in CSS and "@import" not in CSS


def test_the_accent_is_never_a_text_colour_in_the_css():
    for decl in re.findall(r"(?<![-\w])color\s*:\s*([^;]+);", CSS):
        assert S.tokens["accent"].upper() not in decl.upper()


def theme_colours(node) -> list[str]:
    out = []
    if isinstance(node, dict):
        for v in node.values():
            out += theme_colours(v)
    elif isinstance(node, list):
        for v in node:
            out += theme_colours(v)
    elif isinstance(node, str):
        out += HEX.findall(node)
    return out


def test_the_theme_uses_only_registered_tokens_and_no_accent():
    found = {h.upper() for h in theme_colours(THEME["theme"])}
    assert found <= TOKENS
    assert S.tokens["accent"].upper() not in found  # the theme colours text and widgets


def test_the_server_is_loopback_only_and_sends_no_usage_statistics():
    assert THEME["server"]["address"] == "127.0.0.1" and THEME["server"]["port"] == 8501
    assert THEME["server"]["enableStaticServing"] is True
    assert THEME["browser"]["gatherUsageStats"] is False


def test_every_font_face_is_a_self_hosted_file():
    for face in THEME["theme"]["fontFaces"]:
        url = face["url"]
        assert url.startswith("app/static/fonts/") and "://" not in url
        assert (Path("dashboard/static") / url.removeprefix("app/static/")).is_file()
    assert {f["family"] for f in THEME["theme"]["fontFaces"]} == {
        "Space Grotesk",
        "Instrument Serif",
        "JetBrains Mono",
    }


def test_the_font_files_equal_their_recorded_hashes_and_carry_the_licence():
    rows = re.findall(
        r"^\| (\S+\.woff2) \| .* \| ([0-9a-f]{64}) \|$", (FONTS / "SOURCES.md").read_text(), re.M
    )
    assert len(rows) == 4
    for name, sha in rows:
        path = FONTS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        licence = (path.parent / "OFL.txt").read_text()
        assert "SIL OPEN FONT LICENSE Version 1.1" in licence


# --- the reason chart -----------------------------------------------------------------------------


def spec() -> dict:
    return cp.reason_chart(summary()).to_dict()


def walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, (*path, i))
    else:
        yield path, node


def test_the_chart_uses_only_registered_tokens():
    text = json.dumps(spec())
    assert {h.upper() for h in HEX.findall(text)} <= TOKENS


def test_one_chart_with_two_panels_and_no_pie_or_arc():
    s = spec()
    assert len(s["hconcat"]) == 2
    marks = [v for p, v in walk(s) if p and p[-1] == "type" and "mark" in p]
    assert set(marks) <= {"bar", "text"}


def test_every_panel_has_bars_and_direct_labels_in_ink():
    for panel in spec()["hconcat"]:
        kinds = {layer["mark"]["type"]: layer for layer in panel["layer"]}
        assert set(kinds) == {"bar", "text"}
        assert kinds["text"]["encoding"]["text"]["field"] == "label"
        assert kinds["text"]["mark"]["color"].upper() == S.tokens["ink"].upper()


def test_the_accent_fills_only_the_leading_bar_and_no_text():
    s = summary()
    for reasons in s["reasons"]:
        df = cp.bars_frame(reasons)
        lead = cp.leading(cp.all_cells(reasons))
        assert df.loc[df.fill == S.tokens["accent"], "group"].tolist() == [lead["group"]]
        assert set(df.fill) <= {S.tokens["accent"], S.tokens["bar_grey"]}
    for p, v in walk(spec()):
        if isinstance(v, str) and v.upper() == S.tokens["accent"].upper():
            assert "text" not in p and "title" not in p and "axis" not in str(p)


def test_every_bar_has_a_label_and_a_suppressed_bar_reads_fewer_than_20():
    df = cp.bars_frame(summary()["reasons"][1])
    assert df.label.notna().all()
    assert df.set_index("group").at["plan_length", "label"] == "fewer than 20"
    assert df.set_index("group").at["renewal_timing", "label"] == "82%"


@pytest.mark.parametrize(
    "horizon,expected",
    [
        (7, "At 7 days all 1,000 share renewal_timing as their first group"),
        (30, "At 30 days renewal_timing leads with 823 of the top 1,000 at risk"),
    ],
)
def test_the_sentence_is_generated_from_the_body(horizon, expected):
    reasons = next(r for r in summary()["reasons"] if r["horizon"] == horizon)
    assert cp.sentence(reasons) == expected


# --- found after a run (D-258): labels, formats, toolbar, layout, the application name ----------


@pytest.mark.parametrize(
    "values",
    [[6171.5, 6171.5, 6100.0, 0.0, -7366.7], [5.0, 4.9, 4.8, 4.7], [1.0], [3.0, -2.0, 1.0]],
)
def test_spread_keeps_the_order_and_a_minimum_gap(values):
    gap = 0.5
    pos = cp.spread(values, gap)
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ys = [pos[i] for i in order]
    assert all(a - b >= gap - 1e-9 for a, b in zip(ys, ys[1:], strict=False))
    assert all(p <= v for p, v in zip(pos, values, strict=True))


def test_the_planner_end_labels_never_overlap():
    from dashboard.views import action_planner as ap
    from dashboard_fake import plans

    rows = plans()["rows"]
    for r in rows:  # the real run's crowding: the top strategies end at nearly the same value
        if r["budget_label"] == "no limit":
            r["net_value"] = {"optimizer": 6171.5, "highest_risk": 6171.5, "random": -7366.7}[
                r["strategy"]
            ]
    df = pd.DataFrame(
        [
            {"strategy": r["strategy"], "budget": r["budget_label"], "net": r["net_value"]}
            for r in rows
        ]
    )
    order = ap.budget_order(rows)
    ends = ap.label_positions(df, order).sort_values("label_y", ascending=False)
    span = df.net.max() - df.net.min()
    gaps = -ends.label_y.diff().dropna()
    assert (gaps >= ap.LABEL_GAP * span - 1e-9).all()
    spec = ap.chart(rows).to_dict()
    text = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "text")
    assert (
        text["encoding"]["y"]["field"] == "label_y"
        and text["encoding"]["text"]["field"] == "strategy"
    )


def test_fixed_formats_use_separators_and_fixed_decimals():
    assert cp.fixed(3017.587, 2) == "3,017.59"
    assert cp.fixed(22.0208, 1) == "22.0"
    assert cp.fixed(1000, 0) == "1,000"
    assert cp.fixed(-0.0163, 4) == "-0.0163"
    assert cp.fixed(None, 2) == "-"


def test_the_full_menu_stays_and_the_deploy_button_is_hidden():
    assert THEME["client"]["toolbarMode"] == "developer"
    assert re.search(r'\[data-testid="stAppDeployButton"\]\s*\{\s*display:\s*none', CSS)


def test_the_59_day_box_does_not_wrap_its_label():
    assert re.search(r"\.re-demo59 \.re-label[^{]*\{[^}]*white-space:\s*nowrap", CSS)


def test_the_application_name_uses_the_bold_grotesk_in_ink():
    rule = re.search(r"\.re-app \{([^}]*)\}", CSS).group(1)
    assert '"Space Grotesk"' in rule and "font-weight: 700" in rule
    assert S.tokens["ink"] in rule
    assert S.raw["app_name"]["text"] == "Retention Decision Engine"


def test_the_application_name_sits_in_a_header_bar_and_is_larger(  # C4
):
    rule = re.search(r"\.re-app \{([^}]*)\}", CSS).group(1)
    for decl in ("position: fixed", "top: 0", "left: 0", "height: 60px"):
        assert decl in rule, decl
    right = float(re.search(r"right: ([\d.]+)rem", rule).group(1))
    assert right >= 3  # Streamlit's main menu at the right stays visible
    assert "pointer-events: none" in rule  # the controls underneath keep working
    size = float(re.search(r"font-size: ([\d.]+)rem", rule).group(1))
    assert size > 1.6  # larger than C3's line
    assert S.raw["app_name_header"]["amends"] == "app_name.where"


def test_the_big_number_leaves_room_above_its_caption():
    big = re.search(r"\.re-big \{([^}]*)\}", CSS).group(1)
    assert float(re.search(r"line-height: ([\d.]+)", big).group(1)) >= 1.2
    label = re.search(r"\.re-big-label \{([^}]*)\}", CSS).group(1)
    assert float(re.search(r"margin-top: ([\d.]+)rem", label).group(1)) >= 0.75
