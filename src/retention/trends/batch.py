"""Daily batch-size indicator from auto-renew payment rows (no outcome is read).

Units: calendar days 2..27 that are not in the month-end block are judged one by
one; the last `block_days` days of each month are judged as one block (sum).
A unit's baseline is the median of the same unit over the preceding three
months (at least two present). A unit below `gap_below_ratio` of its baseline is
a gap. Because the block is judged as a sum, a batch that moves from the 30th to
the 31st leaves the block ratio unchanged. Limit: a shortfall inside a block
that is offset by an overflow in the same block is not flagged.

The weekday rule: counts are divided by a weekday multiplier before the baseline
is formed, but only if the clean months show a weekday effect larger than the
day-to-day noise (`weekday_multipliers`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

CONFIG = Path("configs/changepoints.yaml")


@dataclass(frozen=True)
class IndicatorConfig:
    block_days: int
    first_day: int
    last_day: int
    gap_below_ratio: float
    baseline_months: int
    min_baseline_values: int
    clean_months: tuple[str, ...]
    max_false_flags: int
    known_gap_days: tuple[str, ...]
    known_non_gap_days: tuple[str, ...]
    noise_mad_scale: float


def load_indicator_config(path: Path = CONFIG) -> IndicatorConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())["daily_indicator"]
    first, last = raw["day_classes"]["calendar_days"]
    return IndicatorConfig(
        block_days=int(raw["day_classes"]["month_end_block_days"]),
        first_day=int(first),
        last_day=int(last),
        gap_below_ratio=float(raw["gap_below_ratio"]),
        baseline_months=3,  # baseline: median_of_same_class_over_preceding_3_months
        min_baseline_values=2,
        clean_months=tuple(str(m) for m in raw["clean_months"]),
        max_false_flags=int(raw["max_false_flags_in_clean_months"]),
        known_gap_days=tuple(str(d) for d in raw["known_gap_days"]),
        known_non_gap_days=tuple(str(d) for d in raw["known_non_gap_days"]),
        noise_mad_scale=1.4826,
    )


def daily_counts(timeline: Path, first: str, last: str) -> pd.Series:
    """Non-cancel auto-renew rows per transaction date, zero-filled over [first, last]."""
    import duckdb

    df = duckdb.sql(
        f"select transaction_date::date d, count(*) n from '{timeline}' "
        "where is_auto_renew = 1 and is_cancel = 0 "
        f"and transaction_date between '{first}' and '{last}' group by 1"
    ).df()
    s = pd.Series(df.n.values, index=pd.to_datetime(df.d)).sort_index()
    return s.reindex(pd.date_range(first, last, freq="D"), fill_value=0).astype(float)


def units(counts: pd.Series, cfg: IndicatorConfig) -> pd.DataFrame:
    """One row per judged unit: start, end, key, kind, count (days or month-end blocks)."""
    rows: list[dict[str, Any]] = []
    for month, grp in counts.groupby(counts.index.to_period("M")):
        n = len(grp)
        # a month is judged only if every day is present
        if n != month.days_in_month:
            continue
        block = grp.iloc[-cfg.block_days :]
        rows.append(
            {
                "start": block.index[0],
                "end": block.index[-1],
                "month": month,
                "key": "block",
                "kind": "block",
                "count": float(block.sum()),
            }
        )
        for ts, v in grp.iloc[: -cfg.block_days].items():
            if cfg.first_day <= ts.day <= cfg.last_day:
                rows.append(
                    {
                        "start": ts,
                        "end": ts,
                        "month": month,
                        "key": f"d{ts.day}",
                        "kind": "day",
                        "count": float(v),
                    }
                )
    return pd.DataFrame(rows)


def weekday_multipliers(counts: pd.Series, cfg: IndicatorConfig) -> dict[str, Any]:
    """Weekday effect on days 2..27 of the clean months (not in a block), by the registered rule."""
    u = units(counts, cfg)
    u = u[(u.kind == "day") & u.month.astype(str).isin(cfg.clean_months)].copy()
    ref = u.groupby("key")["count"].transform("median")
    u["r"] = u["count"] / ref
    u["wd"] = u.start.dt.dayofweek
    med = u.groupby("wd")["r"].median()
    mad = float((u["r"] - u["r"].median()).abs().median())
    noise = cfg.noise_mad_scale * mad
    effect = float((med - 1).abs().max())
    return {
        "per_weekday_median_ratio": {int(k): round(float(v), 4) for k, v in med.items()},
        "noise": round(noise, 4),
        "max_abs_effect": round(effect, 4),
        "use_weekday": bool(effect > noise),
        "multipliers": {int(k): float(v) for k, v in med.items()},
    }


def indicator(
    counts: pd.Series, cfg: IndicatorConfig, multipliers: dict[int, float] | None = None
) -> pd.DataFrame:
    """Units with baseline, ratio and flag. NaN baseline means 'not judged' (no flag)."""
    u = units(counts, cfg)
    if multipliers:
        wd = u.start.dt.dayofweek.map(multipliers)
        # a block spans several weekdays; the multiplier applies to days only
        u["adj"] = np.where(u.kind == "day", u["count"] / wd, u["count"])
    else:
        u["adj"] = u["count"]
    u = u.sort_values(["key", "start"]).reset_index(drop=True)
    base = np.full(len(u), np.nan)
    for idx in u.groupby("key").groups.values():
        sub = u.loc[idx]
        by_month = dict(zip(sub.month, sub.adj, strict=True))
        for i, m in zip(idx, sub.month, strict=True):
            prev = [
                by_month[m - k] for k in range(1, cfg.baseline_months + 1) if (m - k) in by_month
            ]
            if len(prev) >= cfg.min_baseline_values:
                base[i] = float(np.median(prev))
    u["baseline"] = base
    u["ratio"] = u["adj"] / u["baseline"]
    u["flag"] = u["ratio"] < cfg.gap_below_ratio
    return u.sort_values("start").reset_index(drop=True)


def gap_dates(ind: pd.DataFrame) -> list[pd.Timestamp]:
    """Every calendar date covered by a flagged unit."""
    out: list[pd.Timestamp] = []
    for r in ind[ind.flag].itertuples():
        out.extend(pd.date_range(r.start, r.end, freq="D"))
    return out


def week_flags(
    week_starts: pd.DatetimeIndex, gaps: list[pd.Timestamp], window_days: int = 30
) -> pd.Series:
    """A week is flagged if a gap date lies from its start to its end plus 30 days."""
    g = pd.DatetimeIndex(sorted(gaps))
    return pd.Series(
        [
            bool(((g >= w) & (g <= w + pd.Timedelta(days=6 + window_days))).any())
            for w in week_starts
        ],
        index=week_starts,
    )
