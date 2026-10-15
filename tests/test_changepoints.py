"""Change-point detector on synthetic series (ruptures needed)."""

import numpy as np
import pytest

pytest.importorskip("ruptures")

from retention.trends import study  # noqa: E402

COUNTS = np.full(40, 36000.0)


def test_big_shift_found_within_tolerance_and_stable_series_quiet() -> None:
    rng = np.random.default_rng(1)
    s = study.make_series(rng, COUNTS, 0.05, 1.0, shifts=[(20, 0.05)])
    cps = study.detect(s, COUNTS, c=3.0)
    assert study.hit(cps, 20, 1)
    quiet = [
        len(
            study.detect(
                study.make_series(np.random.default_rng(i), COUNTS, 0.05, 1.0), COUNTS, 3.0
            )
        )
        for i in range(30)
    ]
    assert sum(quiet) <= 3


def test_min_segment_blocks_one_week_spike_edges() -> None:
    rng = np.random.default_rng(2)
    s = study.make_series(rng, COUNTS, 0.05, 1.0, spike=(20, 1, 4.0))
    cps = study.detect(s, COUNTS, c=3.0, min_size=3)
    # a 1-week spike cannot form a segment of 3 weeks, so no change point sits at both edges
    assert not (study.hit(cps, 20, 0) and study.hit(cps, 21, 0))


def test_penalty_scales_with_c_and_is_deterministic() -> None:
    s = study.make_series(np.random.default_rng(3), COUNTS, 0.1, 1.0)
    assert study.penalty(s, COUNTS, 2.0) == pytest.approx(2 * study.penalty(s, COUNTS, 1.0))
    a = study.detect(s, COUNTS, 3.0)
    assert a == study.detect(s, COUNTS, 3.0)


def test_run_study_small_is_reproducible() -> None:
    cfg = {
        "tolerance_weeks": 1,
        "c_grid": [3],
        "pass_marks_noise_multiplier": 1.5,
        "study": {
            "seed": 5,
            "replicates_per_cell": 5,
            "base_rates": [0.05],
            "shift_sizes_points": [3, 5],
        },
        "pass_marks": {
            "detect_3_points_within_tolerance": 0.8,
            "detect_5_points_within_tolerance": 0.9,
            "stable_mean_false_alarms_per_stretch_at_most": 0.5,
            "stable_share_of_stretches_with_any_false_alarm_at_most": 0.1,
        },
    }
    counts = np.full(46, 36000.0)
    assert study.run_study(counts, cfg) == study.run_study(counts, cfg)
