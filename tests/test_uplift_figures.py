"""Qini figures: drawn from the logged curves only, with the expected coordinates."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from retention.uplift import figures

NS = {"s": "http://www.w3.org/2000/svg"}


def hand_curves(cell: str = "criteo_visit") -> pd.DataFrame:
    phi = np.arange(1, 101) / 100
    rows = []
    # X-learner: straight up to 0.02 by phi = 0.5, flat after; outcome ranking: the diagonal
    for ranking, q in (
        ("x_learner", np.minimum(phi, 0.5) * 0.04),
        ("outcome_probability", phi * 0.02),
        ("t_learner", phi * 0.02),
        ("random", phi * 0.02),
    ):
        rows.append(pd.DataFrame({"cell": cell, "ranking": ranking, "phi": phi, "q": q}))
    return pd.concat(rows, ignore_index=True)


def write(tmp, df, name=figures.CURVES_NAME):
    path = tmp / name
    df.to_parquet(path, index=False)
    return path


def test_refuses_any_other_input(tmp_path):
    with pytest.raises(ValueError, match="only"):
        figures.load_curves(write(tmp_path, hand_curves(), "uplift_scores.parquet"))
    with pytest.raises(ValueError, match="exactly"):
        figures.load_curves(write(tmp_path, hand_curves().assign(msno="a")))
    bad = hand_curves()
    bad.loc[0, "ranking"] = "kkbox_discount"
    with pytest.raises(ValueError, match="unknown rankings"):
        figures.load_curves(write(tmp_path, bad))


def test_hand_made_curve_gives_the_expected_coordinates():
    w = figures.WIDTH - figures.LEFT - figures.RIGHT
    h = figures.HEIGHT - figures.TOP - figures.BOTTOM
    x, y = figures.to_xy(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.01, 0.02]), 0.0, 0.02)
    assert x.tolist() == [figures.LEFT, figures.LEFT + w / 2, figures.LEFT + w]
    assert y.tolist() == [figures.TOP + h, figures.TOP + h / 2, figures.TOP]


def polylines(svg: str) -> dict[str, list[tuple[float, float]]]:
    root = ET.fromstring(svg)
    out = {}
    for p in root.iter("{http://www.w3.org/2000/svg}polyline"):
        pts = [tuple(map(float, xy.split(","))) for xy in p.get("points").split()]
        out[p.get("class")] = pts
    return out


def test_svg_draws_the_logged_points_and_the_random_diagonal():
    svg = figures.svg_for(hand_curves(), "criteo_visit", "test")
    lines = polylines(svg)
    assert set(lines) == {"ref", "s1", "s2"}
    assert len(lines["s1"]) == 101 and len(lines["s2"]) == 101  # the origin and 100 cut points
    assert len(lines["ref"]) == 2  # the random diagonal from (0, 0) to (1, Q(1))
    # every line starts at the origin and ends at (1, Q(1))
    assert lines["s1"][0] == lines["s2"][0] == lines["ref"][0]
    assert lines["s1"][-1] == lines["s2"][-1] == lines["ref"][-1]
    # the X-learner's point at phi = 0.5 lies above the diagonal's (smaller pixel y)
    assert lines["s1"][50][1] < lines["s2"][50][1]


def test_no_raw_data_and_the_caption_says_there_is_no_band():
    svg = figures.svg_for(hand_curves(), "criteo_visit", "test")
    for word in ("msno", "recency", "history", "zip_code", "f0", "exposure", "treatment,"):
        assert word not in svg
    assert figures.CAPTION in svg
    # numbers only in coordinates, ticks and labels: no long numeric runs from a data row
    assert not re.search(r"\d{7,}", svg)


def test_write_figures_needs_every_primary_cell(tmp_path):
    path = write(tmp_path, hand_curves("criteo_visit"))
    with pytest.raises(ValueError, match="no logged curve"):
        figures.write_figures(path, tmp_path / "img", "test")
    all_cells = pd.concat([hand_curves(c) for c in figures.TITLES], ignore_index=True)
    out = figures.write_figures(write(tmp_path, all_cells), tmp_path / "img", "test")
    assert [p.name for p in out] == [f"qini_{c}.svg" for c in figures.TITLES]
    for p in out:
        ET.fromstring(p.read_text())  # well-formed


def test_zero_tick_is_never_negative_zero():
    svg = figures.svg_for(hand_curves(), "criteo_visit", "test")
    assert ">-0<" not in svg and ">0<" in svg
