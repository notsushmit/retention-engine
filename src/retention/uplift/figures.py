"""Qini figures drawn from the logged validation curves only (hand-coded SVG, no plotting library).

Input: data/processed/public/qini_curves.parquet, exactly the columns cell, ranking, phi
and q, as the validation run wrote it. Anything else is refused. Output: one SVG per
primary cell in docs/img/, with the X-learner, the outcome-probability ranking and the
random diagonal (the straight line from 0 to Q(1)). The log holds intervals for the
area only, not for each curve point, so no band is drawn and the caption says so.
Only the 100 logged curve points per ranking (plus the origin) are written; no row of
either dataset is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CURVES_NAME = "qini_curves.parquet"
COLUMNS = ["cell", "ranking", "phi", "q"]
RANKINGS = {"x_learner", "t_learner", "outcome_probability", "random"}
DRAWN = ("x_learner", "outcome_probability")
LABELS = {"x_learner": "X-learner", "outcome_probability": "Outcome-probability ranking"}
TITLES = {
    "hillstrom_mens_vs_none_visit": ("Hillstrom: men's email against no email", "visits"),
    "hillstrom_womens_vs_none_visit": ("Hillstrom: women's email against no email", "visits"),
    "criteo_visit": ("Criteo: treated against control", "visits"),
}
# plot box (pixels)
WIDTH, HEIGHT = 680, 420
LEFT, RIGHT, TOP, BOTTOM = 84, 196, 64, 92
CAPTION = (
    "No interval band: the log holds a 95% interval for the area only, not for each curve point."
)

STYLE = """<style>
  .surface { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .label { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .tick { fill: #52514e; font: 11px system-ui, -apple-system, sans-serif; }
  .grid { stroke: #e3e2dd; stroke-width: 1; }
  .axis { stroke: #8a8984; stroke-width: 1; }
  .ref { stroke: #8a8984; stroke-width: 1.5; stroke-dasharray: 5 4; fill: none; }
  .s1 { stroke: #2a78d6; stroke-width: 2; fill: none; }
  .s2 { stroke: #eb6834; stroke-width: 2; fill: none; }
  .k1 { fill: #2a78d6; } .k2 { fill: #eb6834; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title { fill: #ffffff; }
    .label, .tick { fill: #c3c2b7; }
    .grid { stroke: #33332f; }
    .s1 { stroke: #3987e5; } .s2 { stroke: #d95926; }
    .k1 { fill: #3987e5; } .k2 { fill: #d95926; }
  }
</style>"""


def load_curves(path: Path) -> pd.DataFrame:
    """The logged curves, or a refusal for any other input."""
    path = Path(path)
    if path.name != CURVES_NAME:
        raise ValueError(f"figures are drawn from {CURVES_NAME} only, not {path.name}")
    df = pd.read_parquet(path)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"{CURVES_NAME} must hold exactly {COLUMNS}, got {list(df.columns)}")
    if not set(df.ranking) <= RANKINGS:
        raise ValueError(f"unknown rankings {sorted(set(df.ranking) - RANKINGS)}")
    return df


def nice_ticks(lo: float, hi: float, n: int = 5) -> np.ndarray:
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    return np.arange(np.ceil(lo / step) * step, hi + step / 2, step)


def to_xy(
    phi: np.ndarray, q: np.ndarray, ymin: float, ymax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates: phi 0..1 across the plot box, q from ymin (bottom) to ymax (top)."""
    w = WIDTH - LEFT - RIGHT
    h = HEIGHT - TOP - BOTTOM
    x = LEFT + np.asarray(phi, dtype=float) * w
    y = TOP + (ymax - np.asarray(q, dtype=float)) / (ymax - ymin) * h
    return x, y


def _points(x: np.ndarray, y: np.ndarray) -> str:
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y, strict=True))


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) >= 0.001 or v == 0 else f"{v:.4f}"


def svg_for(curves: pd.DataFrame, cell: str, source_note: str) -> str:
    """One panel: two rankings and the random diagonal, from the logged points."""
    title, unit = TITLES[cell]
    c = curves[curves.cell == cell]
    series = {}
    for r in DRAWN:
        s = c[c.ranking == r].sort_values("phi")
        series[r] = (np.r_[0.0, s.phi.to_numpy()], np.r_[0.0, s.q.to_numpy()])
    q1 = float(series["x_learner"][1][-1])
    allq = np.concatenate([q for _, q in series.values()] + [np.array([0.0, q1])])
    pad = 0.06 * (allq.max() - allq.min())
    ymin, ymax = min(0.0, allq.min() - pad), allq.max() + pad
    ticks = nice_ticks(ymin, ymax)
    ymin, ymax = min(ymin, ticks[0]), max(ymax, ticks[-1])
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t d">',
        f"<!-- {source_note} -->",
        STYLE,
        f'<title id="t">Qini curve. {title}, {unit}.</title>',
        f'<desc id="d">Extra {unit} per customer in the evaluation half against the share of '
        f"customers targeted, for the X-learner, the outcome-probability ranking and a random "
        f"ranking. {CAPTION}</desc>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="title" x="{LEFT}" y="28">{title}</text>',
        f'<text class="label" x="{LEFT}" y="46">Qini curve, {unit}</text>',
    ]
    x0, x1 = LEFT, WIDTH - RIGHT
    for tv in ticks:
        _, ty = to_xy(np.array([0.0]), np.array([tv]), ymin, ymax)
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{ty[0]:.1f}" y2="{ty[0]:.1f}"/>')
        out.append(
            f'<text class="tick" x="{x0 - 8}" y="{ty[0] + 4:.1f}" text-anchor="end">'
            f"{_fmt(tv)}</text>"
        )
    for xv in (0, 0.25, 0.5, 0.75, 1.0):
        tx, _ = to_xy(np.array([xv]), np.array([0.0]), ymin, ymax)
        out.append(
            f'<text class="tick" x="{tx[0]:.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="middle">'
            f"{int(xv * 100)}%</text>"
        )
    _, y0 = to_xy(np.array([0.0]), np.array([0.0]), ymin, ymax)
    out.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0[0]:.1f}" y2="{y0[0]:.1f}"/>')
    rx, ry = to_xy(np.array([0.0, 1.0]), np.array([0.0, q1]), ymin, ymax)
    out.append(f'<polyline class="ref" points="{_points(rx, ry)}"/>')
    ends = []
    for i, r in enumerate(DRAWN, start=1):
        px, py = to_xy(*series[r], ymin, ymax)
        out.append(f'<polyline class="s{i}" points="{_points(px, py)}"/>')
        ends.append((py[-1], r, i))
    # every line ends at the same point (1, Q(1)), so the labels are stacked beside it,
    # each with a key in its line's style
    end_y = ends[0][0]
    keys = [(f"k{i}", LABELS[r]) for _, r, i in ends] + [("ref", "Random ranking")]
    for k, (cls, text) in enumerate(keys):
        ly = end_y - 16 + 18 * k
        if cls == "ref":
            out.append(
                f'<line class="ref" x1="{x1 + 10}" x2="{x1 + 22}" y1="{ly:.1f}" y2="{ly:.1f}"/>'
            )
        else:
            out.append(
                f'<rect class="{cls}" x="{x1 + 10}" y="{ly - 1.5:.1f}" '
                'width="12" height="3" rx="1"/>'
            )
        out.append(f'<text class="label" x="{x1 + 28}" y="{ly + 4:.1f}">{text}</text>')
    out.append(
        f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{HEIGHT - BOTTOM + 42}" '
        f'text-anchor="middle">Share of customers targeted, highest score first</text>'
    )
    out.append(
        f'<text class="label" transform="translate(20,{(TOP + HEIGHT - BOTTOM) / 2:.1f}) '
        'rotate(-90)" '
        f'text-anchor="middle">Extra {unit} per customer</text>'
    )
    out.append(f'<text class="tick" x="{LEFT}" y="{HEIGHT - 14}">{CAPTION}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_figures(curves_path: Path, out_dir: Path, source_note: str) -> list[Path]:
    curves = load_curves(curves_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell in TITLES:
        if cell not in set(curves.cell):
            raise ValueError(f"no logged curve for {cell}")
        path = out_dir / f"qini_{cell}.svg"
        path.write_text(svg_for(curves, cell, source_note))
        paths.append(path)
    return paths
