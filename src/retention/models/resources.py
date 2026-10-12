"""Runtime and peak memory of the current process, for the per-stage reports."""

from __future__ import annotations

import resource
import sys


def peak_rss_gb() -> float:
    """Peak resident set size of this process so far, in GB (macOS reports bytes, Linux KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)


def peak_rss_children_gb() -> float:
    """Largest peak resident set size of any finished child process (bootstrap workers), in GB."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)
