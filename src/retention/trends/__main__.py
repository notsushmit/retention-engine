"""Command line entry: `python -m retention.trends <command>`.

clean-months: the daily batch indicator over March to November 2016 (the registered
clean months): weekday test, false flags against the registered limit. Writes
batch_clean.json. Reads payment rows only; no outcome is read, and no day of
2016-12 or 2017-02 is loaded (the count window ends 2016-11-30).

known-days: scores the registered known days with the weekday decision and baselines
of the clean-month run. Refuses to run unless batch_clean.json exists and a tracked
file matches HEAD.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from retention.trends import batch

OUT = Path("data/processed/kkbox/sample_020/trends")
TIMELINE = Path("data/processed/kkbox/sample_020/timeline.parquet")


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def clean_months() -> dict:
    cfg = batch.load_indicator_config()
    # three months of history before March 2016 so the first clean month has a baseline
    counts = batch.daily_counts(TIMELINE, "2015-12-01", "2016-11-30")
    wk = batch.weekday_multipliers(counts, cfg)
    mult = wk["multipliers"] if wk["use_weekday"] else None
    ind = batch.indicator(counts, cfg, mult)
    judged = ind[ind.month.astype(str).isin(cfg.clean_months) & ind.baseline.notna()]
    flagged = judged[judged.flag]
    return {
        "commit": _head(),
        "weekday": {k: v for k, v in wk.items() if k != "multipliers"},
        "weekday_multipliers_used": wk["multipliers"] if wk["use_weekday"] else None,
        "units_judged": int(len(judged)),
        "false_flags": [
            {
                "start": str(r.start.date()),
                "end": str(r.end.date()),
                "kind": r.kind,
                "count": r.count,
                "baseline": r.baseline,
                "ratio": round(float(r.ratio), 3),
            }
            for r in flagged.itertuples()
        ],
        "limit": cfg.max_false_flags,
        "within_limit": int(len(flagged)) <= cfg.max_false_flags,
        "min_ratio_clean": round(float(judged.ratio.min()), 3),
        "ratio_quantiles_clean": {
            q: round(float(judged.ratio.quantile(q)), 3) for q in (0.01, 0.05, 0.5)
        },
    }


def known_days() -> dict:
    cfg = batch.load_indicator_config()
    clean = json.loads((OUT / "batch_clean.json").read_text())
    counts = batch.daily_counts(TIMELINE, "2016-09-01", "2017-03-31")
    # baselines need three months; load from 2016-09 so 2016-12 and 2017-02 have them
    mult = clean["weekday_multipliers_used"]
    mult = {int(k): v for k, v in mult.items()} if mult else None
    ind = batch.indicator(counts, cfg, mult)

    def find(day: str):
        d = pd.Timestamp(day)
        hit = ind[(ind.start <= d) & (ind.end >= d)]
        if hit.empty:
            return {"date": day, "unit": None, "note": "not judged (day 1)"}
        r = hit.iloc[0]
        return {
            "date": day,
            "unit": f"{r.start.date()}..{r.end.date()}",
            "kind": r.kind,
            "count": r["count"],
            "baseline": r.baseline,
            "ratio": round(float(r.ratio), 3),
            "flag": bool(r.flag),
        }

    around: dict[str, list[dict]] = {}
    for d in (*cfg.known_gap_days, *cfg.known_non_gap_days):
        t = pd.Timestamp(d)
        around[d] = [find(str((t + pd.Timedelta(days=k)).date())) for k in (-2, -1, 1, 2)]
    blocks = [
        {
            "unit": f"{r.start.date()}..{r.end.date()}",
            "count": r["count"],
            "baseline": r.baseline,
            "ratio": round(float(r.ratio), 3),
            "flag": bool(r.flag),
        }
        for _, r in ind[
            (ind.kind == "block") & ind.month.astype(str).isin(["2016-12", "2017-02"])
        ].iterrows()
    ]
    gaps = [find(d) for d in cfg.known_gap_days]
    non = [find(d) for d in cfg.known_non_gap_days]
    return {
        "commit": _head(),
        "known_gap_days": gaps,
        "known_non_gap_days": non,
        "two_days_either_side": around,
        "month_end_blocks": blocks,
        "flags_known_gap_days": all(g["flag"] for g in gaps),
        "does_not_flag_moved_batch_day": not any(n["flag"] for n in non),
        "false_flags_within_limit": clean["within_limit"],
    }


def main() -> None:
    p = argparse.ArgumentParser(prog="retention.trends")
    p.add_argument("command", choices=["clean-months", "known-days"])
    a = p.parse_args()
    if _dirty():
        raise SystemExit("a tracked file differs from HEAD; commit first")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "clean-months":
        res, name = clean_months(), "batch_clean.json"
    else:
        if not (OUT / "batch_clean.json").exists():
            raise SystemExit("run clean-months first and look at it")
        res, name = known_days(), "batch_known_days.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
