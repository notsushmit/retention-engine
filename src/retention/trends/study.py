"""Synthetic study of the PELT change-point detector (no real rate is used).

Series: 40 weekly points. Noise: rate = base + multiplier * binomial standard
error at the week's real event count * standard normal. The detector's penalty
is c * sigma2 * log(T), sigma2 the binomial variance at the series' own mean
rate and counts (not inflated by the multiplier). A change point detects a
shift at week s if it lies within the tolerance of s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_WEEKS = 40


def penalty(series: np.ndarray, counts: np.ndarray, c: float) -> float:
    p = float(np.clip(series.mean(), 1e-6, 1 - 1e-6))
    sigma2 = float(np.mean(p * (1 - p) / counts))
    return c * sigma2 * float(np.log(len(series)))


def detect(series: np.ndarray, counts: np.ndarray, c: float, min_size: int = 3) -> list[int]:
    """Change points as indexes of the first week of a new segment."""
    import ruptures as rpt

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(series.reshape(-1, 1))
    bkps = algo.predict(pen=penalty(series, counts, c))
    return [int(b) for b in bkps[:-1]]


def make_series(
    rng: np.random.Generator,
    counts: np.ndarray,
    base: float,
    mult: float,
    shifts: list[tuple[int, float]] | None = None,
    spike: tuple[int, int, float] | None = None,
) -> np.ndarray:
    """`shifts`: (first week, size in rate) steps; `spike`: (start, weeks, factor of base)."""
    se = np.sqrt(base * (1 - base) / counts)
    level = np.full(len(counts), base)
    for s, size in shifts or []:
        level[s:] += size
    if spike is not None:
        start, weeks, factor = spike
        level[start : start + weeks] = base * factor
    return level + mult * se * rng.standard_normal(len(counts))


def hit(cps: list[int], week: int, tol: int) -> bool:
    return any(abs(k - week) <= tol for k in cps)


def run_study(real_counts: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Every cell of the registered study. `cfg` is the `detector` section."""
    st = cfg["study"]
    tol = int(cfg["tolerance_weeks"])
    reps = int(st["replicates_per_cell"])
    c_grid = [float(c) for c in cfg["c_grid"]]
    sizes = [float(x) / 100 for x in st["shift_sizes_points"]]
    mults = [1.0, float(cfg["pass_marks_noise_multiplier"]), 2.0]
    out: dict[str, Any] = {"cells": []}
    for ib, base in enumerate(st["base_rates"]):
        for im, mult in enumerate(mults):
            rng = np.random.default_rng(np.random.SeedSequence([int(st["seed"]), ib, im]))
            n_start = len(real_counts) - N_WEEKS + 1
            starts = rng.integers(0, min(6, n_start), size=reps)
            counts = [real_counts[s : s + N_WEEKS] for s in starts]
            shift_weeks = rng.integers(8, N_WEEKS - 8, size=(len(sizes), reps))
            two_a = rng.integers(8, 17, size=reps)
            two_b = rng.integers(24, 33, size=reps)
            spike_start = rng.integers(8, N_WEEKS - 12, size=reps)
            # series are drawn once per cell and shared by every c
            sets: dict[str, list[tuple[np.ndarray, np.ndarray, Any]]] = {"stable": []}
            for r in range(reps):
                sets["stable"].append((make_series(rng, counts[r], base, mult), counts[r], None))
            for i, size in enumerate(sizes):
                sets[f"shift_{round(size * 100)}"] = [
                    (
                        make_series(rng, counts[r], base, mult, [(int(shift_weeks[i, r]), size)]),
                        counts[r],
                        int(shift_weeks[i, r]),
                    )
                    for r in range(reps)
                ]
            sets["two_shift"] = [
                (
                    make_series(
                        rng, counts[r], base, mult, [(int(two_a[r]), 0.03), (int(two_b[r]), 0.03)]
                    ),
                    counts[r],
                    (int(two_a[r]), int(two_b[r])),
                )
                for r in range(reps)
            ]
            for factor in (3.0, 4.0):
                for weeks in (1, 4):
                    sets[f"spike_{int(factor)}x_{weeks}w"] = [
                        (
                            make_series(
                                rng,
                                counts[r],
                                base,
                                mult,
                                spike=(int(spike_start[r]), weeks, factor),
                            ),
                            counts[r],
                            (int(spike_start[r]), weeks),
                        )
                        for r in range(reps)
                    ]
            for c in c_grid:
                cell: dict[str, Any] = {"base_rate": base, "noise_multiplier": mult, "c": c}
                fa = np.array([len(detect(s, n, c)) for s, n, _ in sets["stable"]])
                cell["false_alarms_mean"] = float(fa.mean())
                cell["false_alarms_share_any"] = float((fa > 0).mean())
                cell["false_alarms_distribution"] = {
                    "0": float((fa == 0).mean()),
                    "1": float((fa == 1).mean()),
                    "2_or_more": float((fa >= 2).mean()),
                }
                for size in sizes:
                    key = f"shift_{round(size * 100)}"
                    d = [hit(detect(s, n, c), w, tol) for s, n, w in sets[key]]
                    cell[f"detect_{round(size * 100)}pt"] = float(np.mean(d))
                both = []
                for s, n, (wa, wb) in sets["two_shift"]:
                    cps = detect(s, n, c)
                    both.append(hit(cps, wa, tol) and hit(cps, wb, tol))
                cell["two_shift_both_detected"] = float(np.mean(both))
                for key in [k for k in sets if k.startswith("spike")]:
                    res = {"no_cp": 0, "both_edges": 0, "one_edge": 0, "other": 0}
                    ncp = []
                    for s, n, (start, weeks) in sets[key]:
                        cps = detect(s, n, c)
                        ncp.append(len(cps))
                        if not cps:
                            res["no_cp"] += 1
                            continue
                        a, b = hit(cps, start, tol), hit(cps, start + weeks, tol)
                        res["both_edges" if a and b else "one_edge" if a or b else "other"] += 1
                    cell[key] = {k: v / reps for k, v in res.items()} | {
                        "mean_change_points": float(np.mean(ncp))
                    }
                out["cells"].append(cell)
    marks = cfg["pass_marks"]
    mult_marks = float(cfg["pass_marks_noise_multiplier"])
    verdict: dict[str, Any] = {}
    chosen = None
    for c in c_grid:
        rows = [x for x in out["cells"] if x["c"] == c and x["noise_multiplier"] == mult_marks]
        ok = all(
            r["detect_3pt"] >= marks["detect_3_points_within_tolerance"]
            and r["detect_5pt"] >= marks["detect_5_points_within_tolerance"]
            and r["false_alarms_mean"] <= marks["stable_mean_false_alarms_per_stretch_at_most"]
            and r["false_alarms_share_any"]
            <= marks["stable_share_of_stretches_with_any_false_alarm_at_most"]
            for r in rows
        )
        verdict[str(c)] = ok
        if ok and chosen is None:
            chosen = c
    out["pass_by_c"] = verdict
    out["chosen_c"] = chosen if chosen is not None else 5.0
    out["marks_met"] = chosen is not None
    return out
